"""Measure the durable exact-reuse transition at 1,000 and 8,000 records.

This is the reproducible evidence program for the SQLite recovery writer boundary. It creates
only synthetic ``example.test`` records in temporary migrated storage, prepares them through the
public journal API, then measures the concurrent ``ACQUIRING -> ACQUIRED(REUSED)`` edge. SQL
counting starts after preparation, so the result prices the edge whose old aggregate scans made a
large recovery quadratic rather than database creation or fixture setup.

Run the published matrix::

    .venv/bin/python -m tests.benchmarks.acquisition_reuse

Or take one faster measurement::

    .venv/bin/python -m tests.benchmarks.acquisition_reuse --records 1000 --workers 8
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
from typing import TYPE_CHECKING

from sqlalchemy import event

from manicule.core.acquisition import (
    AcquiredSource,
    AcquisitionRecordState,
    AcquisitionSource,
    SnapshotItemOutcome,
)
from manicule.core.content import RawDocument
from manicule.core.sources import DiscoveredDoc, DocRef
from manicule.storage.blobs import BlobStore, StoredBlob
from manicule.storage.docstore import SqliteDocStore
from manicule.storage.engine import create_engine
from manicule.storage.migrator import upgrade

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection
    from sqlalchemy.ext.asyncio import AsyncEngine

NOW = datetime(2026, 8, 17, tzinfo=UTC)


@dataclass(frozen=True)
class Measurement:
    records: int
    workers: int
    statements: int
    seconds: float
    peak_rss_bytes: int
    acquired: int
    reused: int
    backlog_bytes: int


def _source(index: int) -> AcquisitionSource:
    source_id = f"synthetic-{index:05d}"
    return AcquisitionSource.from_discovered(
        DiscoveredDoc(
            ref=DocRef(
                source_id=source_id,
                uri=f"https://wiki.example.test/content/{index}",
            ),
            version_token="same-version",  # noqa: S106 - synthetic source revision
            media_type="text/plain",
            size_bytes=4,
        )
    )


def _acquired(index: int) -> AcquiredSource:
    return AcquiredSource.from_raw(
        RawDocument(
            source_id=f"synthetic-{index:05d}",
            uri=f"https://wiki.example.test/content/{index}",
            media_type="text/plain",
            content=b"same",
        )
    )


def _peak_rss_bytes() -> int:
    observed = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return observed if sys.platform == "darwin" else observed * 1024


async def _prepare(
    engine: AsyncEngine, store: SqliteDocStore, *, records: int
) -> tuple[str, int, StoredBlob]:
    await upgrade(engine)
    await store.ensure_workspace()
    blob = await BlobStore(engine, Path(engine.url.database or ".").parent).put(
        b"same", "text/plain"
    )
    assert isinstance(blob, StoredBlob)  # new temp storage cannot collide
    created = await store.create_acquisition_run("reuse-benchmark", "synthetic-recovery")
    claimed = await store.claim_acquisition_run(
        created.id,
        "benchmark-worker",
        now=NOW,
        expires_at=NOW + timedelta(hours=1),
    )
    if claimed is None:  # pragma: no cover - one owner in a new database
        raise AssertionError("synthetic benchmark run was not claimed")
    for index in range(records):
        await store.append_acquisition_record(
            claimed.id,
            index,
            _source(index),
            lease_owner="benchmark-worker",
            lease_generation=claimed.lease_generation,
            now=NOW,
        )
        await store.transition_acquisition_record(
            claimed.id,
            f"synthetic-{index:05d}",
            AcquisitionRecordState.DISCOVERED,
            AcquisitionRecordState.ACQUIRING,
            lease_owner="benchmark-worker",
            lease_generation=claimed.lease_generation,
            now=NOW,
        )
    return claimed.id, claimed.lease_generation, blob


async def measure(*, records: int, workers: int) -> Measurement:
    if records < 1 or workers < 1:
        raise ValueError("records and workers must be positive")
    with tempfile.TemporaryDirectory(prefix="manicule-acquisition-reuse-") as directory:
        data_dir = Path(directory)
        engine = create_engine(data_dir)
        store = SqliteDocStore(engine, min_disk_headroom_bytes=1)
        try:
            run_id, generation, blob = await _prepare(engine, store, records=records)
            statements = 0

            def counted(
                _connection: Connection,
                _cursor: object,
                _statement: str,
                _parameters: object,
                _context: object,
                _executemany: bool,
            ) -> None:
                nonlocal statements
                statements += 1

            event.listen(engine.sync_engine, "before_cursor_execute", counted)
            semaphore = asyncio.Semaphore(workers)

            async def reuse(index: int) -> None:
                async with semaphore:
                    await store.transition_acquisition_record(
                        run_id,
                        f"synthetic-{index:05d}",
                        AcquisitionRecordState.ACQUIRING,
                        AcquisitionRecordState.ACQUIRED,
                        lease_owner="benchmark-worker",
                        lease_generation=generation,
                        now=NOW,
                        blob_ref=blob.hash,
                        acquired_source=_acquired(index),
                        snapshot_outcome=SnapshotItemOutcome.REUSED,
                    )

            started = time.perf_counter()
            await asyncio.gather(*(reuse(index) for index in range(records)))
            seconds = time.perf_counter() - started
            event.remove(engine.sync_engine, "before_cursor_execute", counted)
            durable = await store.get_acquisition_run(run_id)
            if durable is None:  # pragma: no cover - the benchmark never deletes its run
                raise AssertionError("synthetic benchmark run disappeared")
            return Measurement(
                records=records,
                workers=workers,
                statements=statements,
                seconds=seconds,
                peak_rss_bytes=_peak_rss_bytes(),
                acquired=durable.acquired_count,
                reused=durable.reused_count,
                backlog_bytes=durable.acquired_blob_bytes,
            )
        finally:
            await engine.dispose()


async def _main(records: list[int], workers: list[int]) -> None:
    for record_count in records:
        for worker_count in workers:
            result = await measure(records=record_count, workers=worker_count)
            print(json.dumps(asdict(result), sort_keys=True))  # noqa: T201 - benchmark output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, nargs="+", default=[1_000, 8_000])
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 8])
    arguments = parser.parse_args()
    asyncio.run(_main(arguments.records, arguments.workers))


if __name__ == "__main__":
    main()
