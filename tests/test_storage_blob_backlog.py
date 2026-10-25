"""The unpublished-blob total that admission reads, against the scan it replaced.

``BlobStore`` used to recompute the bytes retained and not yet published by scanning ``blobs``
against ``documents`` and ``document_versions`` on every new blob, which made a first sync's work
per message grow with the messages already fetched (issue #492). The total is now a row kept by
triggers (``manicule.storage.blob_backlog``). These tests hold the triggers to the scan at every
step a blob or a reference can take, and the fast path to the scan at the limit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import event, text

from manicule.ingest.capacity import CapacityRefusedError
from manicule.storage import blob_backlog
from manicule.storage.blobs import GC_PENDING_PREFIX, BlobStore, StoredBlob, content_hash
from manicule.storage.docstore import SqliteDocStore
from tests.storage_helpers import make_document

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine

pytestmark = pytest.mark.asyncio

_EXACT = (
    "SELECT stored_bytes FROM blobs WHERE algo NOT LIKE 'gc_pending:%' "
    "AND typeof(stored_bytes) = 'integer' AND hash NOT IN ("
    "SELECT original_ref FROM documents WHERE original_ref IS NOT NULL "
    "UNION SELECT original_ref FROM document_versions WHERE original_ref IS NOT NULL)"
)


async def _totals(engine: AsyncEngine) -> tuple[int, int, int, int]:
    """(trigger total, exact scan, pending rows, exact pending rows)."""
    async with engine.connect() as connection:
        kept = (
            await connection.execute(
                text(
                    "SELECT unpublished_bytes, unsized_rows, gc_pending_rows "
                    "FROM blob_backlog_totals"
                )
            )
        ).one()
        exact = sum(row[0] for row in await connection.execute(text(_EXACT)))
        pending = (
            await connection.execute(
                text("SELECT count(*) FROM blobs WHERE algo LIKE 'gc_pending:%'")
            )
        ).scalar_one()
    assert kept[1] == 0
    return kept[0], exact, kept[2], pending


async def _agrees(engine: AsyncEngine) -> int:
    kept, exact, pending_kept, pending = await _totals(engine)
    assert kept == exact
    assert pending_kept == pending
    return kept


async def test_the_gc_pattern_is_the_blob_stores_prefix() -> None:
    assert f"'{GC_PENDING_PREFIX}%'" in blob_backlog.CREATE_TRIGGERS[0]


async def test_every_trigger_exists_at_head(engine: AsyncEngine) -> None:
    """A batch rebuild of ``documents`` would drop them, and nothing else would notice."""
    async with engine.connect() as connection:
        found = {
            row[0]
            for row in await connection.execute(
                text("SELECT name FROM sqlite_master WHERE type = 'trigger'")
            )
        }
    assert set(blob_backlog.TRIGGER_NAMES) <= found


async def test_the_total_follows_every_way_a_blob_or_reference_changes(
    engine: AsyncEngine, store: SqliteDocStore, data_dir: Path
) -> None:
    blobs = BlobStore(engine, data_dir, min_disk_headroom_bytes=1)
    stored: list[StoredBlob] = []
    for size in (11, 23, 37):
        outcome = await blobs.put(bytes([size]) * size * 100, "application/octet-stream")
        assert isinstance(outcome, StoredBlob)
        stored.append(outcome)
        await _agrees(engine)
    first, second, third = stored
    assert await _agrees(engine) == sum(item.stored_bytes for item in stored)

    # A first reference takes a blob out; a second changes nothing.
    one = make_document(source_id="one", body=b"1").model_copy(update={"original_ref": first.hash})
    two = make_document(source_id="two", body=b"2").model_copy(update={"original_ref": first.hash})
    await store.upsert_document(one)
    assert await _agrees(engine) == second.stored_bytes + third.stored_bytes
    await store.upsert_document(two)
    assert await _agrees(engine) == second.stored_bytes + third.stored_bytes

    # Moving a reference puts the old blob back only if it was the last.
    await store.upsert_document(one.model_copy(update={"original_ref": second.hash}))
    await _agrees(engine)
    await store.upsert_document(two.model_copy(update={"original_ref": third.hash}))
    await _agrees(engine)

    # Dropping a reference, a document and a blob.
    await store.delete_document(one.id)
    await _agrees(engine)
    async with engine.begin() as connection:
        await connection.execute(
            text("UPDATE blobs SET algo = :algo WHERE hash = :hash"),
            {"algo": f"{GC_PENDING_PREFIX}blake2b", "hash": first.hash},
        )
    assert (await _totals(engine))[2] == 1
    await _agrees(engine)
    async with engine.begin() as connection:
        await connection.execute(
            text("UPDATE blobs SET algo = 'blake2b' WHERE hash = :hash"), {"hash": first.hash}
        )
    await _agrees(engine)
    async with engine.begin() as connection:
        await connection.execute(text("DELETE FROM documents"))
        await connection.execute(text("DELETE FROM blobs WHERE hash = :hash"), {"hash": first.hash})
    await _agrees(engine)


async def test_a_size_that_is_not_an_integer_sends_admission_to_the_scan(
    engine: AsyncEngine, data_dir: Path
) -> None:
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "INSERT INTO blobs (hash, algo, media_type, size_bytes, stored_bytes, compression, "
                "created_at) VALUES ('ab', 'blake2b', NULL, 1, 'many', 'none', CURRENT_TIMESTAMP)"
            )
        )
        unsized = (
            await connection.execute(text("SELECT unsized_rows FROM blob_backlog_totals"))
        ).scalar_one()
    assert unsized == 1
    blobs = BlobStore(engine, data_dir, min_disk_headroom_bytes=1)
    with pytest.raises(CapacityRefusedError):
        await blobs.put(b"x", "application/octet-stream")


async def test_a_total_that_overflowed_is_not_trusted(engine: AsyncEngine, data_dir: Path) -> None:
    huge = 9_000_000_000_000_000_000
    async with engine.begin() as connection:
        for index in range(2):
            await connection.execute(
                text(
                    "INSERT INTO blobs (hash, algo, media_type, size_bytes, stored_bytes, "
                    "compression, created_at) VALUES (:hash, 'blake2b', NULL, :size, :size, "
                    "'none', CURRENT_TIMESTAMP)"
                ),
                {"hash": f"{index + 10:032x}", "size": huge},
            )
        kind = (
            await connection.execute(
                text("SELECT typeof(unpublished_bytes) FROM blob_backlog_totals")
            )
        ).scalar_one()
    assert kind == "real"
    blobs = BlobStore(engine, data_dir, min_disk_headroom_bytes=1)
    with pytest.raises(CapacityRefusedError) as caught:
        await blobs.put(b"x", "application/octet-stream")
    # The scan stops counting one past the limit, as it always did.
    assert caught.value.diagnostic.used == caught.value.diagnostic.limit + 1


@pytest.mark.parametrize("room", [0, 1])
async def test_the_fast_path_admits_to_the_byte_and_refuses_one_past(
    engine: AsyncEngine, data_dir: Path, room: int
) -> None:
    """The boundary, through the total and then through the scan, must be the same byte."""
    probe = BlobStore(engine, data_dir, min_disk_headroom_bytes=1)
    held = await probe.put(b"a" * 40, "application/octet-stream")
    assert isinstance(held, StoredBlob)
    payload = b"b"
    needed = len(payload)
    limit = held.stored_bytes + needed - 1 + room
    limited = BlobStore(
        engine, data_dir, min_disk_headroom_bytes=1, max_acquired_blob_backlog_bytes=limit
    )
    if room:
        assert isinstance(await limited.put(payload, "application/octet-stream"), StoredBlob)
        assert limited.path_for(content_hash(payload)).exists()
    else:
        with pytest.raises(CapacityRefusedError) as caught:
            await limited.put(payload, "application/octet-stream")
        assert caught.value.diagnostic.used == held.stored_bytes
        assert caught.value.diagnostic.requested == needed
        assert not limited.path_for(content_hash(payload)).exists()


async def test_admitting_one_blob_reads_the_same_rows_whatever_the_backlog(
    engine: AsyncEngine, data_dir: Path
) -> None:
    """Counts SQLite virtual-machine steps, not statements: the statement count never moved.

    A first sync's cost per message grew with the messages already retained while the number of
    statements per message stayed flat (issue #492), so a test of statements would have passed
    before the fix. Steps are deterministic, unlike wall time.
    """

    async def steps_for_one_put(preseeded: int, label: str) -> int:
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < :c) "
                    "INSERT INTO blobs (hash, algo, media_type, size_bytes, stored_bytes, "
                    "compression, created_at) SELECT printf('%s%032x', :label, i), 'blake2b', "
                    "NULL, 10, 10, 'none', CURRENT_TIMESTAMP FROM n"
                ),
                {"c": preseeded, "label": label},
            )
        blobs = BlobStore(engine, data_dir, min_disk_headroom_bytes=1)
        counted = [0]

        def count(dbapi_connection: Any, _record: object) -> None:
            def tick() -> int:
                counted[0] += 1
                return 0

            dbapi_connection.await_(
                dbapi_connection.driver_connection.set_progress_handler(tick, 10)
            )

        await engine.dispose()
        event.listen(engine.sync_engine, "connect", count)
        try:
            outcome = await blobs.put(label.encode() * 3, "application/octet-stream")
        finally:
            event.remove(engine.sync_engine, "connect", count)
        assert isinstance(outcome, StoredBlob)
        return counted[0]

    small = await steps_for_one_put(1_000, "a")
    large = await steps_for_one_put(20_000, "b")
    assert large < small * 2, f"one admission took {small} steps at 1k and {large} at 21k"
