"""Retained original bytes: what is kept, what is refused, and what the sweep reclaims."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import threading
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text

from manicule.core.content import RawDocument
from manicule.core.ids import content_hash
from manicule.storage.blobs import BlobStore, OmittedBlob, StoredBlob, should_compress
from manicule.storage.docstore import SqliteDocStore
from tests.storage_helpers import make_document

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine


async def test_bytes_survive_a_round_trip_so_reparsing_never_refetches(
    engine: AsyncEngine, data_dir: Path
) -> None:
    """Rung 3 of the ladder, and the only thing between a parser fix and a re-crawl."""
    blobs = BlobStore(engine, data_dir)
    payload = b"%PDF-1.7 not really a pdf but bytes are bytes"
    stored = await blobs.put(payload, "application/pdf")
    assert isinstance(stored, StoredBlob)
    assert await blobs.get(stored.hash) == payload


async def test_identical_bytes_are_stored_once(engine: AsyncEngine, data_dir: Path) -> None:
    """The same attachment reachable from forty pages should cost one copy."""
    blobs = BlobStore(engine, data_dir)
    first = await blobs.put(b"shared attachment", "application/pdf")
    second = await blobs.put(b"shared attachment", "application/pdf")
    assert isinstance(first, StoredBlob)
    assert isinstance(second, StoredBlob)
    assert first.hash == second.hash
    assert sum(1 for path in blobs.root.rglob("*") if path.is_file()) == 1


async def test_concurrent_media_types_reuse_one_coherent_blob_representation(
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same content address cannot let file bytes and SQLite compression disagree."""
    blobs = BlobStore(engine, data_dir)
    barrier = threading.Barrier(2)
    original = BlobStore._publish_durable  # pyright: ignore[reportPrivateUsage]

    def gated_publish(destination: Path, payload: bytes) -> bytes:
        barrier.wait(timeout=5)
        return original(destination, payload)

    monkeypatch.setattr(BlobStore, "_publish_durable", staticmethod(gated_publish))
    shared = b"the same source bytes " * 200
    text_raw = RawDocument(
        source_id="text-copy",
        uri="https://private.invalid/text-copy",
        media_type="text/plain",
        content=shared,
    )
    binary_raw = text_raw.model_copy(
        update={
            "source_id": "binary-copy",
            "uri": "https://private.invalid/binary-copy",
            "media_type": "application/pdf",
        }
    )

    text_result, binary_result = await asyncio.gather(
        blobs.retain_acquisition("run\0text-copy", text_raw),
        blobs.retain_acquisition("run\0binary-copy", binary_raw),
    )
    monkeypatch.setattr(BlobStore, "_publish_durable", staticmethod(original))

    assert text_result[0].ref == binary_result[0].ref == content_hash(shared)
    assert await blobs.get(content_hash(shared)) == shared
    assert await blobs.verify(content_hash(shared))
    stage_compressions = {
        json.loads(path.read_text())["compression"]
        for path in (blobs.root / "acquisition-staging").iterdir()
    }
    assert len(stage_compressions) == 1
    assert (await blobs.resume_acquisition("run\0text-copy")) == text_result
    assert (await blobs.resume_acquisition("run\0binary-copy")) == binary_result


async def test_blob_file_and_destination_directory_are_fsynced_before_database_reference(
    engine: AsyncEngine, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rename is not durable until its directory entry is fsynced."""
    calls: list[str] = []
    real_fsync = os.fsync

    def observed_fsync(descriptor: int) -> None:
        calls.append("directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", observed_fsync)
    stored = await BlobStore(engine, data_dir).put(b"durable bytes", "application/pdf")

    assert isinstance(stored, StoredBlob)
    assert "file" in calls
    assert calls[-1] == "directory", "the destination parent was not synced after rename"


async def test_directory_fsync_failure_creates_no_database_reference(
    engine: AsyncEngine, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An orphan file costs space; a blob row pointing at a non-durable name costs correctness."""
    blobs = BlobStore(engine, data_dir)
    original = BlobStore._fsync_directory  # pyright: ignore[reportPrivateUsage]
    destination = blobs.path_for(content_hash(b"fsync refusal"))

    def refusing_final_sync(path: Path) -> None:
        if path == destination.parent and destination.exists():
            msg = "synthetic directory fsync refusal"
            raise OSError(msg)
        original(path)

    monkeypatch.setattr(BlobStore, "_fsync_directory", staticmethod(refusing_final_sync))
    with pytest.raises(OSError, match="synthetic directory fsync refusal"):
        await blobs.put(b"fsync refusal", "application/pdf")

    async with engine.connect() as connection:
        count = (await connection.execute(text("SELECT COUNT(*) FROM blobs"))).scalar_one()
    assert count == 0


async def test_cancelled_staging_write_is_joined_before_cancellation_returns(
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker thread cannot create a sensitive marker after its task has returned."""
    blobs = BlobStore(engine, data_dir)
    entered = threading.Event()
    release = threading.Event()
    original = BlobStore._write_durable  # pyright: ignore[reportPrivateUsage]

    def gated_write(destination: Path, payload: bytes) -> None:
        if destination.parent.name == "acquisition-staging":
            entered.set()
            assert release.wait(timeout=5)
        original(destination, payload)

    monkeypatch.setattr(BlobStore, "_write_durable", staticmethod(gated_write))
    raw = RawDocument(
        source_id="private-id",
        uri="https://private.invalid/cancelled",
        media_type="text/plain",
        content="sensitive body",
    )
    task = asyncio.create_task(blobs.retain_acquisition("run\0private-id", raw))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done(), "cancellation returned while the durable write thread was live"
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done(), "repeated cancellation detached the durable write thread"
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    staging = blobs.root / "acquisition-staging"
    assert not list(staging.glob("*.partial"))
    assert len([path for path in staging.iterdir() if path.is_file()]) == 1


async def test_stale_staging_partial_cleanup_is_bounded_and_aggregate_only(
    engine: AsyncEngine, data_dir: Path
) -> None:
    """Abandoned metadata is removed without inventory output disclosing its identity."""
    blobs = BlobStore(engine, data_dir)
    staging = blobs.root / "acquisition-staging"
    staging.mkdir(parents=True)
    for index in range(3):
        path = staging / f"opaque-{index}.partial"
        path.write_text(f"secret-uri-{index}")
        os.utime(path, (1, 1))

    report = await blobs.cleanup_staging_partials(stale_after_seconds=60, limit=2)

    assert report.scanned == 2
    assert report.removed == 2
    assert report.truncated
    assert "secret" not in repr(report)
    fresh = staging / "active.partial"
    fresh.write_text("still being written")
    await blobs.cleanup_staging_partials(stale_after_seconds=60, limit=10)
    assert fresh.exists()


async def test_text_is_compressed_and_binary_is_not(engine: AsyncEngine, data_dir: Path) -> None:
    """Gzipping an already-compressed PDF spends CPU to make the file slightly larger."""
    blobs = BlobStore(engine, data_dir)
    prose = await blobs.put(b"prose " * 500, "text/markdown")
    binary = await blobs.put(bytes(range(256)) * 4, "application/pdf")
    assert isinstance(prose, StoredBlob)
    assert isinstance(binary, StoredBlob)
    assert prose.compression == "gzip"
    assert prose.stored_bytes < prose.size_bytes
    assert binary.compression == "none"
    assert binary.stored_bytes == binary.size_bytes


def test_media_types_are_classified_by_whether_compression_helps() -> None:
    """A ``+json`` suffix is still JSON, and an absent media type is not a guess."""
    assert should_compress("text/plain")
    assert should_compress("application/vnd.api+json")
    assert not should_compress("image/png")
    assert not should_compress(None)


async def test_oversized_bytes_are_refused_with_a_stated_reason(
    engine: AsyncEngine, data_dir: Path
) -> None:
    """Absent with a reason, visible in diagnostics, never a silent partial success.

    A four-gigabyte video attachment must not silently double the data directory.
    """
    blobs = BlobStore(engine, data_dir, max_bytes=16)
    result = await blobs.put(b"x" * 64, "application/octet-stream")
    assert isinstance(result, OmittedBlob)
    assert "retention cap" in result.reason
    assert not any(path.is_file() for path in blobs.root.rglob("*"))


async def test_a_corrupted_blob_is_detectable_without_a_reference_copy(
    engine: AsyncEngine, data_dir: Path
) -> None:
    """Content addressing is what makes verification free."""
    blobs = BlobStore(engine, data_dir)
    stored = await blobs.put(b"original bytes", "application/pdf")
    assert isinstance(stored, StoredBlob)
    assert await blobs.verify(stored.hash)

    blobs.path_for(stored.hash).write_bytes(b"tampered")
    assert not await blobs.verify(stored.hash)


async def test_the_sweep_reclaims_only_unreferenced_blobs(
    engine: AsyncEngine, data_dir: Path
) -> None:
    """Mark and sweep, never refcounts: a refcount is wrong silently in both directions."""
    blobs = BlobStore(engine, data_dir)
    store = SqliteDocStore(engine)
    await store.ensure_workspace()

    kept = await blobs.put(b"referenced bytes", "text/plain")
    dropped = await blobs.put(b"orphan bytes", "text/plain")
    assert isinstance(kept, StoredBlob)
    assert isinstance(dropped, StoredBlob)

    document = make_document(body=b"referenced bytes")
    await store.upsert_document(document.model_copy(update={"original_ref": kept.hash}))

    collected = await blobs.collect_garbage()
    assert collected == [dropped.hash]
    assert await blobs.get(kept.hash) == b"referenced bytes"
    assert await blobs.get(dropped.hash) is None


async def test_a_leaked_file_is_found_by_the_directory_scan(
    engine: AsyncEngine, data_dir: Path
) -> None:
    """Crashing between deleting the row and unlinking the file leaks disk, not correctness.

    The reverse ordering would leave a reference pointing at nothing.
    """
    blobs = BlobStore(engine, data_dir)
    leaked = blobs.path_for(content_hash(b"leaked"))
    leaked.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    leaked.write_bytes(b"leaked")

    orphans = await blobs.orphaned_files()
    assert [path.name for path in orphans] == [leaked.name]


async def test_reading_a_blob_that_was_never_stored_is_not_an_error(
    engine: AsyncEngine, data_dir: Path
) -> None:
    """A document whose bytes were refused by the cap still has to be readable as a document."""
    blobs = BlobStore(engine, data_dir)
    assert await blobs.get(content_hash(b"never stored")) is None
