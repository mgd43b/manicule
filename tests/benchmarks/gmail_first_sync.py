"""Measure a Gmail first sync through durable acquisition as the journal grows.

Issue #492: a first sync of about 200,000 messages ran at 62% of the quota ceiling because every
retained blob was admitted by a scan of all the blobs retained before it, so the CPU spent per
message grew with the run. This drives the real client, pacer, connector, pipeline, journal and
blob store against :class:`~tests.connectors.fake_gmail.FakeGmail` holding synthetic messages
(``example.org`` addresses, nothing real), with retained source bytes on. Pacing runs on a
simulated clock, so waiting for quota costs nothing and what remains is the CPU a message costs.

Each window of fetched messages reports the CPU seconds, wall seconds and **SQLite virtual-machine
steps** it took, per message. Steps are what a regression test can assert: they do not move with
the machine, and statements per message did not move either while the cost grew sixfold.

Run the published matrix (about two minutes per 10,000 messages once the cost is flat)::

    .venv/bin/python -m tests.benchmarks.gmail_first_sync --messages 200000 --window 10000

Or acquisition only (``manicule connector sync --acquire-only``)::

    .venv/bin/python -m tests.benchmarks.gmail_first_sync --messages 20000 --acquire-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import event

from manicule.storage.docstore import SqliteDocStore
from manicule.storage.engine import create_engine, prepare_data_dir
from manicule.storage.migrator import upgrade
from manicule.storage.vectors import LanceVectorStore
from tests.connectors.fake_gmail import FakeGmail, SimulatedTime
from tests.connectors.gmail_support import connector_over, mail_pipeline
from tests.fakes import HashEmbedder

if TYPE_CHECKING:
    import httpx
    from sqlalchemy.ext.asyncio import AsyncEngine

PAGE_SIZE = 250
"""Ids per listing page. Gmail's own is 500, which the connector admits as two journal slices;
250 keeps the benchmark's journal batches whole."""
STEP_PERIOD = 100
"""SQLite calls the progress handler once per this many virtual-machine instructions."""


@dataclass(frozen=True)
class Window:
    """What one run of fetched messages cost, per message."""

    fetched: int
    cpu_ms: float
    wall_ms: float
    steps: float


@dataclass
class Measurement:
    messages: int
    acquire_only: bool
    seconds: float = 0.0
    error: str = ""
    windows: list[Window] = field(default_factory=list[Window])

    @property
    def messages_per_minute(self) -> float:
        return self.messages / self.seconds * 60 if self.seconds else 0.0


class _Steps:
    """SQLite virtual-machine steps on every connection an engine opens, to the nearest period."""

    def __init__(self, engine: AsyncEngine) -> None:
        self.count = 0
        event.listen(engine.sync_engine, "connect", self._on_connect)

    def _tick(self) -> int:
        self.count += 1
        return 0

    def _on_connect(self, dbapi_connection: Any, _record: object) -> None:
        dbapi_connection.await_(
            dbapi_connection.driver_connection.set_progress_handler(self._tick, STEP_PERIOD)
        )


class _Windows:
    """Closes a :class:`Window` each time ``window`` more raw messages have been fetched."""

    def __init__(self, steps: _Steps, measured: Measurement, window: int, *, echo: bool) -> None:
        self._steps = steps
        self._measured = measured
        self._window = window
        self._echo = echo
        self._fetched = 0
        self._mark = self._now()

    def _now(self) -> tuple[float, float, int]:
        return time.process_time(), time.perf_counter(), self._steps.count

    def on_request(self, request: httpx.Request) -> None:
        if request.url.params.get("format") != "raw":
            return
        self._fetched += 1
        if self._fetched % self._window:
            return
        now = self._now()
        row = Window(
            fetched=self._fetched,
            cpu_ms=(now[0] - self._mark[0]) * 1000 / self._window,
            wall_ms=(now[1] - self._mark[1]) * 1000 / self._window,
            steps=(now[2] - self._mark[2]) / self._window,
        )
        self._measured.windows.append(row)
        self._mark = now
        if self._echo:
            print(json.dumps(asdict(row)), flush=True)  # noqa: T201 - progress


async def measure(
    *, messages: int, window: int, acquire_only: bool = False, echo: bool = False
) -> Measurement:
    """Sync ``messages`` synthetic messages and report the cost per message of each ``window``."""
    measured = Measurement(messages=messages, acquire_only=acquire_only)
    with tempfile.TemporaryDirectory(prefix="manicule-gmail-first-sync-") as directory:
        data_dir = Path(directory) / "data"
        prepare_data_dir(data_dir)
        engine = create_engine(data_dir)
        steps = _Steps(engine)
        vectors = LanceVectorStore(data_dir / "vectors")
        try:
            await upgrade(engine)
            store = SqliteDocStore(engine, max_journal_records=messages * 3 + 1_000)
            await store.ensure_workspace()
            await vectors.ensure_ready(HashEmbedder().fingerprint)
            pipeline = mail_pipeline(store, engine, data_dir, vectors)
            clock = SimulatedTime()
            server = FakeGmail(clock=clock, page_size=PAGE_SIZE)
            server.add_many(messages)
            connector = connector_over(server, clock)
            await connector.setup()
            server.before_request = _Windows(steps, measured, window, echo=echo).on_request
            started = time.perf_counter()
            try:
                report = await pipeline.run(
                    connector, retain_source_bytes=True, acquire_only=acquire_only
                )
            finally:
                measured.seconds = time.perf_counter() - started
                await connector.teardown()
                await pipeline.aclose()
            measured.error = report.error
        finally:
            await vectors.teardown()
            await engine.dispose()
    return measured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--messages", type=int, default=20_000)
    parser.add_argument("--window", type=int, default=2_500)
    parser.add_argument("--acquire-only", action="store_true")
    arguments = parser.parse_args()
    result = asyncio.run(
        measure(
            messages=arguments.messages,
            window=arguments.window,
            acquire_only=arguments.acquire_only,
            echo=True,
        )
    )
    print(  # noqa: T201 - the report
        json.dumps(
            {
                "messages": result.messages,
                "acquire_only": result.acquire_only,
                "seconds": round(result.seconds, 1),
                "messages_per_second": round(result.messages / result.seconds, 1),
                "error": result.error,
            }
        )
    )


if __name__ == "__main__":
    main()
