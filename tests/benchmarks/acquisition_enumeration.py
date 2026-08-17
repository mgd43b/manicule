"""Measure scalar versus page-atomic durable enumeration admission.

The scalar mode is the compatibility path: one source record per capacity-guarded writer
transaction.  The page mode is the Confluence production path added for issue #228.  Both use
synthetic ``example.test`` identities and the public SQLite journal API.

Run the published matrix::

    .venv/bin/python -m tests.benchmarks.acquisition_enumeration
"""

from __future__ import annotations

import argparse
import asyncio
import json
import resource
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from sqlalchemy import event

from manicule.core.acquisition import AcquisitionSource
from manicule.core.sources import DiscoveredDoc, DocRef
from manicule.storage.docstore import SqliteDocStore
from manicule.storage.engine import create_engine
from manicule.storage.migrator import upgrade

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection

NOW = datetime(2026, 8, 17, tzinfo=UTC)
PAGE_SIZE = 250


@dataclass(frozen=True)
class Measurement:
    records: int
    mode: str
    source_pages: int
    writer_transactions: int
    statements: int
    seconds: float
    peak_rss_bytes: int


def _source(index: int) -> AcquisitionSource:
    source_id = f"synthetic-{index:05d}"
    return AcquisitionSource.from_discovered(
        DiscoveredDoc(
            ref=DocRef(
                source_id=source_id,
                uri=f"https://wiki.example.test/content/{index}",
            ),
            version_token="synthetic-version",  # noqa: S106 - source revision
            media_type="text/plain",
            size_bytes=4,
        )
    )


def _peak_rss_bytes() -> int:
    observed = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return observed if sys.platform == "darwin" else observed * 1024


async def measure(*, records: int, mode: Literal["scalar", "page"]) -> Measurement:
    if records < 1:
        raise ValueError("records must be positive")
    with tempfile.TemporaryDirectory(prefix="manicule-enumeration-") as directory:
        engine = create_engine(Path(directory))
        store = SqliteDocStore(engine, max_journal_records=records + 1)
        try:
            await upgrade(engine)
            await store.ensure_workspace()
            created = await store.create_acquisition_run("enumeration-benchmark", "synthetic-wiki")
            claimed = await store.claim_acquisition_run(
                created.id,
                "benchmark-worker",
                now=NOW,
                expires_at=NOW + timedelta(hours=1),
            )
            if claimed is None:  # pragma: no cover - one owner in a new database
                raise AssertionError("synthetic benchmark run was not claimed")
            statements = 0
            writer_transactions = 0

            def counted(
                _connection: Connection,
                _cursor: object,
                statement: str,
                _parameters: object,
                _context: object,
                _executemany: bool,
            ) -> None:
                nonlocal statements, writer_transactions
                statements += 1
                if statement.lstrip().upper().startswith("BEGIN IMMEDIATE"):
                    writer_transactions += 1

            event.listen(engine.sync_engine, "before_cursor_execute", counted)
            started = time.perf_counter()
            if mode == "scalar":
                for index in range(records):
                    await store.append_acquisition_record(
                        claimed.id,
                        index,
                        _source(index),
                        lease_owner="benchmark-worker",
                        lease_generation=claimed.lease_generation,
                        now=NOW,
                    )
            else:
                for start in range(0, records, PAGE_SIZE):
                    await store.append_acquisition_records(
                        claimed.id,
                        start,
                        tuple(
                            _source(index)
                            for index in range(start, min(records, start + PAGE_SIZE))
                        ),
                        lease_owner="benchmark-worker",
                        lease_generation=claimed.lease_generation,
                        now=NOW,
                    )
            seconds = time.perf_counter() - started
            event.remove(engine.sync_engine, "before_cursor_execute", counted)
            return Measurement(
                records=records,
                mode=mode,
                source_pages=(records + PAGE_SIZE - 1) // PAGE_SIZE,
                writer_transactions=writer_transactions,
                statements=statements,
                seconds=seconds,
                peak_rss_bytes=_peak_rss_bytes(),
            )
        finally:
            await engine.dispose()


async def _main(records: list[int], modes: list[Literal["scalar", "page"]]) -> None:
    for record_count in records:
        for mode in modes:
            result = await measure(records=record_count, mode=mode)
            print(json.dumps(asdict(result), sort_keys=True))  # noqa: T201 - benchmark output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, nargs="+", default=[250, 2_500, 10_000])
    parser.add_argument(
        "--modes", choices=("scalar", "page"), nargs="+", default=["scalar", "page"]
    )
    arguments = parser.parse_args()
    asyncio.run(_main(arguments.records, arguments.modes))


if __name__ == "__main__":
    main()
