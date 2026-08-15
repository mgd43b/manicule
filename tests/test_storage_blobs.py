"""Retained original bytes: what is kept, what is refused, and what the sweep reclaims."""

from __future__ import annotations

import asyncio
import errno
import json
import os
import stat
import threading
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, override

import pytest
from sqlalchemy import text

from manicule.core.acquisition import AcquiredSource, AcquisitionRecordState, AcquisitionSource
from manicule.core.content import RawDocument
from manicule.core.ids import content_hash
from manicule.core.sources import DiscoveredDoc, DocRef, Watermark
from manicule.ingest.capacity import CapacityRefusedError, CapacityResource
from manicule.storage.blobs import (
    BlobStore,
    OmittedBlob,
    StagingCleanup,
    StoredBlob,
    should_compress,
)
from manicule.storage.docstore import SqliteDocStore
from tests.storage_helpers import make_document

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


def _payload_files(root: Path) -> list[Path]:
    """Persistent lock shards are coordination metadata, not retained customer payloads."""
    return [
        path for path in root.rglob("*") if path.is_file() and "acquisition-locks" not in path.parts
    ]


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
    assert sum(1 for path in (blobs.root / "blake2b").rglob("*") if path.is_file()) == 1


async def test_concurrent_media_types_reuse_one_coherent_blob_representation(
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    """Same content address cannot let file bytes and SQLite compression disagree."""
    blobs = BlobStore(engine, data_dir)
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
    """A failed post-link durability check leaves neither a row nor an untracked blob."""
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
    assert not destination.exists()


def test_raced_directory_creation_must_still_certify_its_parent(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A peer may create the shard and die before syncing its directory entry."""
    target = data_dir / "race-parent" / "shard"
    raced = target.parent
    path_type = type(data_dir)
    original_mkdir = path_type.mkdir
    original_sync = BlobStore._fsync_directory  # pyright: ignore[reportPrivateUsage]

    def raced_mkdir(path: Path, mode: int = 0o777, parents: bool = False) -> None:
        if path == raced:
            original_mkdir(path, mode=mode, parents=parents)
            raise FileExistsError
        original_mkdir(path, mode=mode, parents=parents)

    def refusing_raced_parent_sync(path: Path) -> None:
        if path == raced.parent:
            msg = "synthetic raced-parent fsync refusal"
            raise OSError(msg)
        original_sync(path)

    monkeypatch.setattr(path_type, "mkdir", raced_mkdir)
    monkeypatch.setattr(BlobStore, "_fsync_directory", staticmethod(refusing_raced_parent_sync))

    with pytest.raises(OSError, match="raced-parent fsync refusal"):
        BlobStore._mkdir_durable(target)  # pyright: ignore[reportPrivateUsage]


async def test_staging_partial_is_private_while_live_and_after_publication_failure(
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sensitive envelopes are 0600 at creation, not only after their write completes."""
    blobs = BlobStore(engine, data_dir)
    staging = blobs.root / "acquisition-staging"
    partials = blobs._stage_partial_root()  # pyright: ignore[reportPrivateUsage]
    destination = staging / "opaque-marker"
    entered = threading.Event()
    release = threading.Event()
    original_fsync = os.fsync
    write_durable = BlobStore._write_durable  # pyright: ignore[reportPrivateUsage]

    def gated_file_sync(descriptor: int) -> None:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            entered.set()
            assert release.wait(timeout=5)
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", gated_file_sync)
    write = asyncio.create_task(
        asyncio.to_thread(
            write_durable,
            destination,
            b"private source metadata",
            temporary_dir=partials,
        )
    )
    assert await asyncio.to_thread(entered.wait, 5)
    partial = next(partials.glob("*.partial"))
    assert stat.S_IMODE(partial.stat().st_mode) == 0o600
    release.set()
    await write

    original_directory_sync = BlobStore._fsync_directory  # pyright: ignore[reportPrivateUsage]

    def refusing_publication_sync(path: Path) -> None:
        if path == staging:
            msg = "synthetic staging directory fsync refusal"
            raise OSError(msg)
        original_directory_sync(path)

    monkeypatch.setattr(os, "fsync", original_fsync)
    monkeypatch.setattr(BlobStore, "_fsync_directory", staticmethod(refusing_publication_sync))
    failed_destination = staging / "failed-marker"
    with pytest.raises(OSError, match="staging directory fsync refusal"):
        await asyncio.to_thread(write_durable, failed_destination, b"private failed metadata")
    assert stat.S_IMODE(failed_destination.stat().st_mode) == 0o600


async def test_canceled_staging_write_is_joined_before_cancellation_returns(
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker thread cannot create a sensitive marker after its task has returned."""
    blobs = BlobStore(engine, data_dir)
    entered = threading.Event()
    release = threading.Event()
    original = BlobStore._write_durable  # pyright: ignore[reportPrivateUsage]

    def gated_write(
        destination: Path, payload: bytes, *, temporary_dir: Path | None = None
    ) -> None:
        if destination.parent.name == "acquisition-staging":
            entered.set()
            assert release.wait(timeout=5)
        original(destination, payload, temporary_dir=temporary_dir)

    monkeypatch.setattr(BlobStore, "_write_durable", staticmethod(gated_write))
    raw = RawDocument(
        source_id="private-id",
        uri="https://private.invalid/canceled",
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
    assert not list(blobs._stage_partial_root().glob("*.partial"))  # pyright: ignore[reportPrivateUsage]
    assert len([path for path in staging.iterdir() if path.is_file()]) == 1


async def test_durable_thread_failure_takes_precedence_over_cancellation() -> None:
    """A storage refusal is not hidden by cancellation received while joining its thread."""
    entered = threading.Event()
    release = threading.Event()

    def refusing_write() -> None:
        entered.set()
        assert release.wait(timeout=5)
        msg = "synthetic durable write refusal"
        raise OSError(msg)

    task = asyncio.create_task(
        BlobStore._durable_thread_call(refusing_write)  # pyright: ignore[reportPrivateUsage]
    )
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()

    with pytest.raises(OSError, match="durable write refusal"):
        await task


async def test_canceled_marker_completion_waits_for_parent_fsync(
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A committed association cannot return while its marker deletion is non-durable."""
    blobs = BlobStore(engine, data_dir)
    raw = RawDocument(
        source_id="completed-id",
        uri="https://private.invalid/completed",
        media_type="text/plain",
        content="completed body",
    )
    await blobs.retain_acquisition("run\0completed-id", raw)
    staging = blobs.root / "acquisition-staging"
    marker = next(path for path in staging.iterdir() if not path.name.endswith(".partial"))
    entered = threading.Event()
    release = threading.Event()
    original = BlobStore._fsync_directory  # pyright: ignore[reportPrivateUsage]

    def gated_sync(path: Path) -> None:
        if path == staging:
            assert not marker.exists()
            entered.set()
            assert release.wait(timeout=5)
        original(path)

    monkeypatch.setattr(BlobStore, "_fsync_directory", staticmethod(gated_sync))
    task = asyncio.create_task(blobs.complete_acquisition("run\0completed-id"))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done(), "repeated cancellation returned before the directory sync"
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not marker.exists()


async def test_stale_staging_partial_cleanup_is_bounded_and_aggregate_only(
    engine: AsyncEngine, data_dir: Path
) -> None:
    """Abandoned metadata is removed without inventory output disclosing its identity."""
    blobs = BlobStore(engine, data_dir)
    staging = blobs.root / "acquisition-staging"
    staging.mkdir(parents=True)
    for index in range(20):
        (staging / f"ordinary-marker-{index}").write_text("live recovery envelope")
    partials = blobs._stage_partial_root()  # pyright: ignore[reportPrivateUsage]
    partials.mkdir(parents=True)
    for index in range(3):
        path = partials / f"opaque-{index}.partial"
        path.write_text(f"secret-uri-{index}")
        os.utime(path, (1, 1))

    report = await blobs.cleanup_staging_partials(stale_after_seconds=60, limit=2)

    assert report.scanned == 2
    assert report.removed == 2
    assert report.truncated
    assert "secret" not in repr(report)
    fresh = partials / "active.partial"
    fresh.write_text("still being written")
    await blobs.cleanup_staging_partials(stale_after_seconds=60, limit=10)
    assert fresh.exists()


async def test_staging_cleanup_candidate_traversal_is_bounded(
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A small cleanup budget cannot stat the entire abandoned-partial population."""
    blobs = BlobStore(engine, data_dir)
    partials = blobs._stage_partial_root()  # pyright: ignore[reportPrivateUsage]
    partials.mkdir(parents=True)
    for index in range(25):
        path = partials / f"opaque-{index}.partial"
        path.write_text("private metadata")
        os.utime(path, (1, 1))
    path_type = type(data_dir)
    original_stat = path_type.stat
    candidate_stats = 0

    def counted_stat(path: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal candidate_stats
        if path.parent == partials:
            candidate_stats += 1
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(path_type, "stat", counted_stat)
    report = await blobs.cleanup_staging_partials(stale_after_seconds=60, limit=2)

    assert report == StagingCleanup(scanned=2, removed=2, truncated=True)
    assert candidate_stats <= 4, "cleanup traversed candidates beyond its advertised budget"


async def test_concurrent_physical_backlog_admission_is_bounded_and_deduplicated(
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    limited = BlobStore(
        engine,
        data_dir,
        min_disk_headroom_bytes=1,
        max_acquired_blob_backlog_bytes=10,
    )
    distinct = await asyncio.gather(
        limited.put(b"a" * 10, "application/octet-stream"),
        limited.put(b"b" * 10, "application/octet-stream"),
        return_exceptions=True,
    )

    assert sum(isinstance(outcome, StoredBlob) for outcome in distinct) == 1
    assert sum(isinstance(outcome, CapacityRefusedError) for outcome in distinct) == 1
    assert len(_payload_files(limited.root)) == 1
    winner_index = next(
        index for index, outcome in enumerate(distinct) if isinstance(outcome, StoredBlob)
    )
    winner = distinct[winner_index]
    assert isinstance(winner, StoredBlob)
    loser_payload = (b"a" * 10, b"b" * 10)[1 - winner_index]
    assert await limited.get(content_hash(loser_payload)) is None
    duplicate = await limited.put(await limited.get(winner.hash) or b"", "application/octet-stream")
    assert isinstance(duplicate, StoredBlob)
    assert duplicate.hash == winner.hash
    assert len(_payload_files(limited.root)) == 1


async def test_orphan_recovery_charges_the_published_representation(
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    payload = b"a" * 10_000
    blobs = BlobStore(
        engine,
        data_dir,
        min_disk_headroom_bytes=1,
        max_acquired_blob_backlog_bytes=100,
    )
    destination = blobs.path_for(content_hash(payload))
    BlobStore._write_durable(destination, payload)  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(CapacityRefusedError) as caught:
        await blobs.put(payload, "text/plain")

    assert caught.value.diagnostic.resource is CapacityResource.ACQUIRED_BLOB_BACKLOG_BYTES
    assert caught.value.diagnostic.requested == len(payload)
    assert await blobs.get(content_hash(payload)) is None


async def test_missing_file_reconstruction_reserves_descriptor_growth_and_rolls_back(
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    payload = b"a" * 10_000
    initial = BlobStore(
        engine,
        data_dir,
        min_disk_headroom_bytes=1,
        max_acquired_blob_backlog_bytes=20_000,
    )
    compressed = await initial.put(payload, "text/plain")
    assert isinstance(compressed, StoredBlob)
    assert compressed.stored_bytes < 100
    destination = initial.path_for(compressed.hash)
    destination.unlink()

    first = BlobStore(
        engine,
        data_dir,
        min_disk_headroom_bytes=1,
        max_acquired_blob_backlog_bytes=100,
    )
    second = BlobStore(
        engine,
        data_dir,
        min_disk_headroom_bytes=1,
        max_acquired_blob_backlog_bytes=100,
    )
    refused = await asyncio.gather(
        first.put(payload, "application/octet-stream"),
        second.put(payload, "application/octet-stream"),
        return_exceptions=True,
    )

    assert all(isinstance(outcome, CapacityRefusedError) for outcome in refused)
    assert not destination.exists()
    async with engine.connect() as connection:
        descriptor = (
            await connection.execute(
                text("SELECT stored_bytes, compression FROM blobs WHERE hash = :digest"),
                {"digest": compressed.hash},
            )
        ).one()
    assert descriptor == (compressed.stored_bytes, "gzip")

    reconstructed = await asyncio.gather(
        first.put(payload, "text/plain"), second.put(payload, "text/plain")
    )
    assert all(isinstance(outcome, StoredBlob) for outcome in reconstructed)
    assert await first.get(compressed.hash) == payload


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


async def test_rebuild_inventory_streams_compressed_verification_without_decompress_allocation(
    engine: AsyncEngine, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blobs = BlobStore(engine, data_dir)
    stored = await blobs.put(b"bounded prose " * 10_000, "text/plain")
    assert isinstance(stored, StoredBlob)
    assert stored.compression == "gzip"

    def forbidden(_: bytes) -> bytes:
        raise AssertionError("inventory must not call gzip.decompress")

    monkeypatch.setattr("manicule.storage.blobs.gzip.decompress", forbidden)
    assert await blobs.contains(stored.hash)


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
    assert not _payload_files(blobs.root)


async def test_disk_headroom_refuses_before_a_file_or_row_is_acknowledged(
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blobs = BlobStore(engine, data_dir, min_disk_headroom_bytes=8)

    def nearly_full(path: object) -> SimpleNamespace:
        del path
        return SimpleNamespace(total=100, used=90, free=10)

    monkeypatch.setattr(
        "manicule.storage.blobs.shutil.disk_usage",
        nearly_full,
    )
    payload = b"three"

    with pytest.raises(CapacityRefusedError) as caught:
        await blobs.put(payload, "application/octet-stream")

    assert caught.value.diagnostic.resource is CapacityResource.DISK_HEADROOM_BYTES
    assert await blobs.get(content_hash(payload)) is None
    assert not _payload_files(blobs.root)


async def test_enospc_leaves_no_partial_file_or_dangling_database_reference(
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blobs = BlobStore(engine, data_dir, min_disk_headroom_bytes=1)
    store = SqliteDocStore(engine)
    await store.ensure_workspace()
    published = make_document(source_id="published-before-enospc")
    await store.upsert_document(published)
    before = await store.find_document("fs", "published-before-enospc")
    payload = b"private body cinder"
    private_path = "/private/customer?token=fake-secret-cinder"
    original_open = os.open

    def no_space(path: os.PathLike[str] | str, *args: Any, **kwargs: Any) -> int:
        if os.fspath(path).endswith(".partial"):
            raise OSError(errno.ENOSPC, "private body cinder", private_path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", no_space)

    with pytest.raises(CapacityRefusedError) as caught:
        await blobs.put(payload, "application/octet-stream")

    rendered = f"{caught.value!s}\n{caught.value!r}\n{caught.value.diagnostic.as_metadata()!r}"
    assert payload.decode() not in rendered
    assert private_path not in rendered
    assert "fake-secret-cinder" not in rendered
    assert await blobs.get(content_hash(payload)) is None
    assert not _payload_files(blobs.root)
    assert await store.find_document("fs", "published-before-enospc") == before


async def test_database_full_after_rename_removes_only_the_new_file(
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sqlalchemy.ext.asyncio import AsyncSession  # noqa: PLC0415

    blobs = BlobStore(engine, data_dir, min_disk_headroom_bytes=1)
    preexisting = await blobs.put(b"deduplicated", "application/octet-stream")
    assert isinstance(preexisting, StoredBlob)
    preexisting_path = blobs.path_for(preexisting.hash)

    async def database_full(session: AsyncSession) -> None:
        del session
        raise OSError(
            errno.ENOSPC,
            "database full at private title",
            "/private/customer?token=fake-secret-cinder",
        )

    monkeypatch.setattr(AsyncSession, "commit", database_full)
    new_payload = b"newly renamed"
    with pytest.raises(CapacityRefusedError):
        await blobs.put(new_payload, "application/octet-stream")
    with pytest.raises(CapacityRefusedError):
        await blobs.put(b"deduplicated", "application/octet-stream")

    assert not blobs.path_for(content_hash(new_payload)).exists()
    assert preexisting_path.exists(), "rollback must not delete a deduplicated preexisting file"
    assert await blobs.get(content_hash(new_payload)) is None
    assert await blobs.get(preexisting.hash) == b"deduplicated"
    assert not any(path.name.endswith(".partial") for path in blobs.root.rglob("*"))


async def test_marker_failure_rolls_back_each_new_blob_before_capacity_can_be_bypassed(
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blobs = BlobStore(
        engine,
        data_dir,
        min_disk_headroom_bytes=1,
        max_acquired_blob_backlog_bytes=100,
    )
    original_write = BlobStore._write_durable  # pyright: ignore[reportPrivateUsage]

    def refuse_marker(
        destination: Path, payload: bytes, *, temporary_dir: Path | None = None
    ) -> None:
        if destination.parent.name == "acquisition-staging":
            raise OSError(errno.ENOSPC, "private marker write refused")
        original_write(destination, payload, temporary_dir=temporary_dir)

    monkeypatch.setattr(BlobStore, "_write_durable", staticmethod(refuse_marker))
    for number in range(3):
        raw = RawDocument(
            source_id=f"private-{number}",
            uri=f"https://private.invalid/{number}?token=fake-secret",
            media_type="application/octet-stream",
            content=bytes([number]) * 60,
        )
        with pytest.raises(CapacityRefusedError):
            await blobs.retain_acquisition(f"run\0private-{number}", raw)

    async with engine.connect() as connection:
        count = (await connection.execute(text("SELECT COUNT(*) FROM blobs"))).scalar_one()
    assert count == 0
    assert not any(path.is_file() for path in (blobs.root / "blake2b").rglob("*"))
    assert not (blobs.root / "acquisition-staging").exists()

    monkeypatch.setattr(BlobStore, "_write_durable", staticmethod(original_write))
    accepted = RawDocument(
        source_id="accepted",
        uri="memory://accepted",
        media_type="application/octet-stream",
        content=b"a" * 60,
    )
    await blobs.retain_acquisition("run\0accepted", accepted)
    refused = accepted.model_copy(
        update={"source_id": "refused", "uri": "memory://refused", "content": b"b" * 60}
    )
    with pytest.raises(CapacityRefusedError):
        await blobs.retain_acquisition("run\0refused", refused)


async def test_transient_owned_destination_unlink_failure_is_retried_and_fsynced(
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blobs = BlobStore(
        engine,
        data_dir,
        min_disk_headroom_bytes=1,
        max_acquired_blob_backlog_bytes=100,
    )
    original_write = BlobStore._write_durable  # pyright: ignore[reportPrivateUsage]
    original_remove = BlobStore._remove_published  # pyright: ignore[reportPrivateUsage]
    original_sync = BlobStore._fsync_directory  # pyright: ignore[reportPrivateUsage]
    failed_once: set[Path] = set()
    synced: list[Path] = []

    def refuse_marker(
        destination: Path, payload: bytes, *, temporary_dir: Path | None = None
    ) -> None:
        if destination.parent.name == "acquisition-staging":
            raise OSError(errno.ENOSPC, "private marker refused")
        original_write(destination, payload, temporary_dir=temporary_dir)

    def transient_remove(destination: Path) -> None:
        if destination not in failed_once:
            failed_once.add(destination)
            raise OSError(errno.EIO, "transient owned unlink refused")
        original_remove(destination)

    def observed_sync(path: Path) -> None:
        synced.append(path)
        original_sync(path)

    monkeypatch.setattr(BlobStore, "_write_durable", staticmethod(refuse_marker))
    monkeypatch.setattr(BlobStore, "_remove_published", staticmethod(transient_remove))
    monkeypatch.setattr(BlobStore, "_fsync_directory", staticmethod(observed_sync))
    for number in range(3):
        raw = RawDocument(
            source_id=f"private-{number}",
            uri=f"https://private.invalid/{number}?token=fake-secret",
            media_type="application/octet-stream",
            content=bytes([number]) * 60,
        )
        with pytest.raises(CapacityRefusedError) as caught:
            await blobs.retain_acquisition(f"run\0private-{number}", raw)
        assert "fake-secret" not in str(caught.value)

    async with engine.connect() as connection:
        count = (await connection.execute(text("SELECT COUNT(*) FROM blobs"))).scalar_one()
    assert count == 0
    assert not _payload_files(blobs.root)
    assert {destination.parent for destination in failed_once} <= set(synced)


async def test_permanent_owned_unlink_refusal_is_capacity_accounted_without_looping(
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blobs = BlobStore(
        engine,
        data_dir,
        min_disk_headroom_bytes=1,
        max_acquired_blob_backlog_bytes=100,
    )
    original_write = BlobStore._write_durable  # pyright: ignore[reportPrivateUsage]
    original_remove = BlobStore._remove_published  # pyright: ignore[reportPrivateUsage]
    original_sync = BlobStore._fsync_directory  # pyright: ignore[reportPrivateUsage]
    remove_attempts = 0
    synced: list[Path] = []

    def refuse_marker(
        destination: Path, payload: bytes, *, temporary_dir: Path | None = None
    ) -> None:
        if destination.parent.name == "acquisition-staging":
            raise OSError(errno.ENOSPC, "marker refused")
        original_write(destination, payload, temporary_dir=temporary_dir)

    def permanent_remove(destination: Path) -> None:
        nonlocal remove_attempts
        del destination
        remove_attempts += 1
        raise PermissionError(errno.EACCES, "permanent owned unlink refusal")

    monkeypatch.setattr(BlobStore, "_write_durable", staticmethod(refuse_marker))
    monkeypatch.setattr(BlobStore, "_remove_published", staticmethod(permanent_remove))
    first = RawDocument(
        source_id="first",
        uri="memory://first",
        media_type="application/octet-stream",
        content=b"a" * 60,
    )
    with pytest.raises(CapacityRefusedError):
        await blobs.retain_acquisition("run\0first", first)
    assert remove_attempts == 2

    digest = content_hash(first.as_bytes())
    assert await blobs.collect_garbage() == []
    assert remove_attempts == 4
    assert not blobs.path_for(digest).exists()
    assert len(list(blobs._gc_root().glob("*.blob"))) == 1  # pyright: ignore[reportPrivateUsage]
    assert len(list(blobs._gc_root().glob("*.json"))) == 1  # pyright: ignore[reportPrivateUsage]
    async with engine.connect() as connection:
        descriptor_count = (
            await connection.execute(text("SELECT COUNT(*) FROM blobs"))
        ).scalar_one()
    assert descriptor_count == 0

    second = first.model_copy(
        update={"source_id": "second", "uri": "memory://second", "content": b"b" * 60}
    )
    with pytest.raises(CapacityRefusedError):
        await blobs.retain_acquisition("run\0second", second)
    assert remove_attempts == 4, "the capacity refusal must happen before another publication"

    assert len(list(blobs._gc_root().glob("*.blob"))) == 1  # pyright: ignore[reportPrivateUsage]

    def observed_sync(path: Path) -> None:
        synced.append(path)
        original_sync(path)

    monkeypatch.setattr(BlobStore, "_remove_published", staticmethod(original_remove))
    monkeypatch.setattr(BlobStore, "_fsync_directory", staticmethod(observed_sync))
    assert await blobs.collect_garbage() == [digest]
    assert not _payload_files(blobs.root)
    assert blobs._gc_root() in synced  # pyright: ignore[reportPrivateUsage]
    async with engine.connect() as connection:
        assert (await connection.execute(text("SELECT COUNT(*) FROM blobs"))).scalar_one() == 0


async def test_same_digest_waiter_committed_before_cleanup_keeps_the_shared_file(
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blobs = BlobStore(engine, data_dir, min_disk_headroom_bytes=1)
    original_write = BlobStore._write_durable  # pyright: ignore[reportPrivateUsage]
    original_cleanup = BlobStore._cleanup_owned_blob  # pyright: ignore[reportPrivateUsage]
    cleanup_arrived = asyncio.Event()
    release_cleanup = asyncio.Event()

    def refuse_marker(
        destination: Path, payload: bytes, *, temporary_dir: Path | None = None
    ) -> None:
        if destination.parent.name == "acquisition-staging":
            raise OSError(errno.ENOSPC, "marker refused")
        original_write(destination, payload, temporary_dir=temporary_dir)

    async def gated_cleanup(
        store: BlobStore,
        digest: str,
        destination: Path,
        stored: StoredBlob,
        media_type: str | None,
    ) -> None:
        cleanup_arrived.set()
        await release_cleanup.wait()
        await original_cleanup(store, digest, destination, stored, media_type)

    monkeypatch.setattr(BlobStore, "_write_durable", staticmethod(refuse_marker))
    monkeypatch.setattr(BlobStore, "_cleanup_owned_blob", gated_cleanup)
    payload = b"same digest waiter wins the cleanup reservation"
    raw = RawDocument(
        source_id="owner",
        uri="memory://owner",
        media_type="application/octet-stream",
        content=payload,
    )
    owner = asyncio.create_task(blobs.retain_acquisition("run\0owner", raw))
    await cleanup_arrived.wait()

    waiter = await blobs.put(payload, "application/octet-stream")
    assert isinstance(waiter, StoredBlob)
    release_cleanup.set()
    with pytest.raises(CapacityRefusedError):
        await owner

    assert blobs.path_for(waiter.hash).exists()
    assert await blobs.get(waiter.hash) == payload


async def test_owned_unlink_holds_reservation_until_same_digest_waiter_can_republish(
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blobs = BlobStore(engine, data_dir, min_disk_headroom_bytes=1)
    original_write = BlobStore._write_durable  # pyright: ignore[reportPrivateUsage]
    original_remove = BlobStore._remove_published  # pyright: ignore[reportPrivateUsage]
    unlink_arrived = threading.Event()
    release_unlink = threading.Event()

    def refuse_marker(
        destination: Path, payload: bytes, *, temporary_dir: Path | None = None
    ) -> None:
        if destination.parent.name == "acquisition-staging":
            raise OSError(errno.ENOSPC, "marker refused")
        original_write(destination, payload, temporary_dir=temporary_dir)

    def gated_remove(destination: Path) -> None:
        unlink_arrived.set()
        assert release_unlink.wait(timeout=5)
        original_remove(destination)

    monkeypatch.setattr(BlobStore, "_write_durable", staticmethod(refuse_marker))
    monkeypatch.setattr(BlobStore, "_remove_published", staticmethod(gated_remove))
    payload = b"waiter must not adopt across owner rollback cleanup"
    raw = RawDocument(
        source_id="owner",
        uri="memory://owner",
        media_type="application/octet-stream",
        content=payload,
    )
    owner = asyncio.create_task(blobs.retain_acquisition("run\0owner", raw))
    assert await asyncio.to_thread(unlink_arrived.wait, 5)
    waiter = asyncio.create_task(blobs.put(payload, "application/octet-stream"))

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(asyncio.shield(waiter), timeout=0.1)
    release_unlink.set()
    with pytest.raises(CapacityRefusedError):
        await owner
    stored = await waiter

    assert isinstance(stored, StoredBlob)
    assert blobs.path_for(stored.hash).exists()
    assert await blobs.get(stored.hash) == payload


async def test_conflicting_marker_retry_preserves_prior_durable_evidence(
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blobs = BlobStore(engine, data_dir, min_disk_headroom_bytes=1)
    key = "run\0replaced"
    old = RawDocument(
        source_id="old",
        uri="memory://old",
        media_type="application/octet-stream",
        content=b"old marker body",
    )
    await blobs.retain_acquisition(key, old)
    fresh = old.model_copy(
        update={"source_id": "fresh", "uri": "memory://fresh", "content": b"fresh marker body"}
    )
    fresh_digest = content_hash(fresh.as_bytes())
    staging = blobs.root / "acquisition-staging"
    original_sync = BlobStore._fsync_directory  # pyright: ignore[reportPrivateUsage]

    def fail_replaced_marker_sync(path: Path) -> None:
        if path == staging:
            raise OSError(errno.ENOSPC, "replaced marker directory sync refused")
        original_sync(path)

    monkeypatch.setattr(BlobStore, "_fsync_directory", staticmethod(fail_replaced_marker_sync))
    with pytest.raises(RuntimeError, match="conflicts with durable recovery evidence"):
        await blobs.retain_acquisition(key, fresh)

    marker = blobs._stage_path(key)  # pyright: ignore[reportPrivateUsage]
    old_digest = content_hash(old.as_bytes())
    assert json.loads(marker.read_text())["blob_ref"] == old_digest
    assert not blobs.path_for(fresh_digest).exists()
    monkeypatch.setattr(BlobStore, "_fsync_directory", staticmethod(original_sync))
    resumed = await blobs.resume_acquisition(key)
    assert resumed is not None
    assert resumed[0].ref == old_digest
    assert resumed[1].source_id == "old"


async def test_repeated_post_link_failures_do_not_escape_backlog_accounting(
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blobs = BlobStore(
        engine,
        data_dir,
        min_disk_headroom_bytes=1,
        max_acquired_blob_backlog_bytes=100,
    )
    original_sync = BlobStore._fsync_directory  # pyright: ignore[reportPrivateUsage]

    def refuse_published_sync(path: Path) -> None:
        if path.parent.parent.name == "blake2b" and any(
            child.is_file() and not child.name.endswith(".partial") for child in path.iterdir()
        ):
            raise OSError(errno.ENOSPC, "post-link directory sync refused")
        original_sync(path)

    monkeypatch.setattr(BlobStore, "_fsync_directory", staticmethod(refuse_published_sync))
    for number in range(3):
        with pytest.raises(CapacityRefusedError):
            await blobs.put(bytes([number]) * 60, "application/octet-stream")

    async with engine.connect() as connection:
        count = (await connection.execute(text("SELECT COUNT(*) FROM blobs"))).scalar_one()
    assert count == 0
    assert not any(path.is_file() for path in (blobs.root / "blake2b").rglob("*"))

    monkeypatch.setattr(BlobStore, "_fsync_directory", staticmethod(original_sync))
    await blobs.put(b"a" * 60, "application/octet-stream")
    with pytest.raises(CapacityRefusedError):
        await blobs.put(b"b" * 60, "application/octet-stream")


async def test_failed_publication_temp_unlink_is_retried_and_never_escapes_capacity(
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blobs = BlobStore(
        engine,
        data_dir,
        min_disk_headroom_bytes=1,
        max_acquired_blob_backlog_bytes=100,
    )
    path_type = type(data_dir)
    original_unlink = path_type.unlink
    failed_once: set[Path] = set()

    def fail_first_partial_unlink(path: Path, *, missing_ok: bool = False) -> None:
        if path.name.endswith(".partial") and path not in failed_once:
            failed_once.add(path)
            raise OSError(errno.ENOSPC, "publication temp unlink refused")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(path_type, "unlink", fail_first_partial_unlink)
    for number in range(3):
        with pytest.raises(CapacityRefusedError):
            await blobs.put(bytes([number]) * 60, "application/octet-stream")

    async with engine.connect() as connection:
        count = (await connection.execute(text("SELECT COUNT(*) FROM blobs"))).scalar_one()
    assert count == 0
    assert not _payload_files(blobs.root)

    monkeypatch.setattr(path_type, "unlink", original_unlink)
    await blobs.put(b"a" * 60, "application/octet-stream")
    with pytest.raises(CapacityRefusedError):
        await blobs.put(b"b" * 60, "application/octet-stream")


async def test_double_cancellation_waits_for_rollback_and_owned_cleanup(
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sqlalchemy.ext.asyncio import AsyncSession  # noqa: PLC0415

    blobs = BlobStore(engine, data_dir, min_disk_headroom_bytes=1)
    original_write = BlobStore._write_durable  # pyright: ignore[reportPrivateUsage]
    original_rollback = AsyncSession.rollback
    rollback_arrived = asyncio.Event()
    release_rollback = asyncio.Event()
    gated = False

    def refuse_marker(
        destination: Path, payload: bytes, *, temporary_dir: Path | None = None
    ) -> None:
        if destination.parent.name == "acquisition-staging":
            raise OSError(errno.ENOSPC, "marker refused")
        original_write(destination, payload, temporary_dir=temporary_dir)

    async def gated_rollback(session: AsyncSession) -> None:
        nonlocal gated
        if not gated:
            gated = True
            rollback_arrived.set()
            await release_rollback.wait()
        await original_rollback(session)

    monkeypatch.setattr(BlobStore, "_write_durable", staticmethod(refuse_marker))
    monkeypatch.setattr(AsyncSession, "rollback", gated_rollback)
    raw = RawDocument(
        source_id="private-cancel",
        uri="https://private.invalid/cancel?token=fake-secret",
        media_type="application/octet-stream",
        content=b"private cancellation body",
    )
    task = asyncio.create_task(blobs.retain_acquisition("run\0cancel", raw))
    await rollback_arrived.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release_rollback.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    digest = content_hash(raw.as_bytes())
    assert not blobs.path_for(digest).exists()
    assert not _payload_files(blobs.root)
    async with engine.connect() as connection:
        count = (await connection.execute(text("SELECT COUNT(*) FROM blobs"))).scalar_one()
    assert count == 0


@pytest.mark.parametrize("stale_marker", [[], None, "private scalar", 7])
async def test_nonmapping_stale_marker_cannot_interrupt_owned_blob_cleanup(
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    stale_marker: object,
) -> None:
    blobs = BlobStore(engine, data_dir, min_disk_headroom_bytes=1)
    key = "run\0stale-scalar"
    marker = blobs._stage_path(key)  # pyright: ignore[reportPrivateUsage]
    BlobStore._write_durable(  # pyright: ignore[reportPrivateUsage]
        marker,
        json.dumps(stale_marker).encode(),
        temporary_dir=blobs._stage_partial_root(),  # pyright: ignore[reportPrivateUsage]
    )
    original_write = BlobStore._write_durable  # pyright: ignore[reportPrivateUsage]

    def refuse_replacement(
        destination: Path, payload: bytes, *, temporary_dir: Path | None = None
    ) -> None:
        if destination == marker:
            raise OSError(errno.ENOSPC, "marker replacement refused")
        original_write(destination, payload, temporary_dir=temporary_dir)

    monkeypatch.setattr(BlobStore, "_write_durable", staticmethod(refuse_replacement))
    raw = RawDocument(
        source_id="private-stale",
        uri="https://private.invalid/stale?token=fake-secret",
        media_type="application/octet-stream",
        content=b"private stale marker body",
    )
    with pytest.raises(CapacityRefusedError) as caught:
        await blobs.retain_acquisition(key, raw)

    digest = content_hash(raw.as_bytes())
    assert not blobs.path_for(digest).exists()
    assert json.loads(marker.read_text()) == stale_marker
    assert "private scalar" not in str(caught.value)


async def test_durable_marker_preserves_a_blob_when_descriptor_commit_fails(
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sqlalchemy.ext.asyncio import AsyncSession  # noqa: PLC0415

    blobs = BlobStore(engine, data_dir, min_disk_headroom_bytes=1)
    raw = RawDocument(
        source_id="private-recoverable",
        uri="https://private.invalid/recoverable?token=fake-secret",
        media_type="application/octet-stream",
        content=b"recoverable private bytes",
    )

    async def database_full(session: AsyncSession) -> None:
        del session
        raise OSError(errno.ENOSPC, "database descriptor refused")

    monkeypatch.setattr(AsyncSession, "commit", database_full)
    with pytest.raises(CapacityRefusedError):
        await blobs.retain_acquisition("run\0recoverable", raw)

    digest = content_hash(raw.as_bytes())
    assert blobs.path_for(digest).exists()
    async with engine.connect() as connection:
        count = (await connection.execute(text("SELECT COUNT(*) FROM blobs"))).scalar_one()
    assert count == 0

    monkeypatch.undo()
    resumed = await blobs.resume_acquisition("run\0recoverable")
    assert resumed is not None
    assert resumed[0].ref == digest
    assert resumed[1].source_id == raw.source_id


async def test_postcommit_error_keeps_the_file_for_the_durable_row(
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sqlalchemy.ext.asyncio import AsyncSession  # noqa: PLC0415

    blobs = BlobStore(engine, data_dir, min_disk_headroom_bytes=1)
    original_commit = AsyncSession.commit
    private_detail = "committed then failed at /private/customer?token=fake-secret-cinder"

    async def commit_then_fail(session: AsyncSession) -> None:
        await original_commit(session)
        raise OSError(errno.ENOSPC, private_detail)

    monkeypatch.setattr(AsyncSession, "commit", commit_then_fail)
    payload = b"durable despite ambiguous error"

    with pytest.raises(CapacityRefusedError) as caught:
        await blobs.put(payload, "application/octet-stream")

    digest = content_hash(payload)
    rendered = f"{caught.value!s}\n{caught.value!r}"
    assert private_detail not in rendered
    assert caught.value.__context__ is None
    assert blobs.path_for(digest).exists()
    assert await blobs.get(digest) == payload
    assert await blobs.verify(digest)


async def test_ambiguous_descriptor_probe_preserves_the_only_recoverable_file(
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sqlalchemy.ext.asyncio import AsyncSession  # noqa: PLC0415

    blobs = BlobStore(engine, data_dir, min_disk_headroom_bytes=1)

    async def commit_failed(session: AsyncSession) -> None:
        del session
        raise OSError(errno.ENOSPC, "commit outcome unknown")

    async def probe_ambiguous(session: object, digest: str) -> bool:
        del session, digest
        return True

    monkeypatch.setattr(AsyncSession, "commit", commit_failed)
    monkeypatch.setattr(
        BlobStore, "_descriptor_is_durable_or_ambiguous", staticmethod(probe_ambiguous)
    )
    payload = b"preserve until descriptor ambiguity is resolved"

    with pytest.raises(CapacityRefusedError):
        await blobs.put(payload, "application/octet-stream")

    digest = content_hash(payload)
    assert blobs.path_for(digest).exists()
    async with engine.connect() as connection:
        count = (await connection.execute(text("SELECT COUNT(*) FROM blobs"))).scalar_one()
    assert count == 0


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


async def test_gc_double_cancellation_waits_for_file_and_descriptor_deletion(
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blobs = BlobStore(engine, data_dir, min_disk_headroom_bytes=1)
    stored = await blobs.put(b"cancel-safe garbage", "application/octet-stream")
    assert isinstance(stored, StoredBlob)
    original_remove = BlobStore._remove_published  # pyright: ignore[reportPrivateUsage]
    unlink_arrived = threading.Event()
    release_unlink = threading.Event()

    def gated_remove(destination: Path) -> None:
        unlink_arrived.set()
        assert release_unlink.wait(timeout=5)
        original_remove(destination)

    monkeypatch.setattr(BlobStore, "_remove_published", staticmethod(gated_remove))
    task = asyncio.create_task(blobs.collect_garbage())
    assert await asyncio.to_thread(unlink_arrived.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release_unlink.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert not blobs.path_for(stored.hash).exists()
    async with engine.connect() as connection:
        assert (await connection.execute(text("SELECT COUNT(*) FROM blobs"))).scalar_one() == 0


async def test_slow_gc_filesystem_phase_does_not_hold_the_sqlite_writer(
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blobs = BlobStore(engine, data_dir, min_disk_headroom_bytes=1)
    garbage = await blobs.put(b"slow garbage", "application/octet-stream")
    assert isinstance(garbage, StoredBlob)
    original_quarantine = BlobStore._quarantine_durable  # pyright: ignore[reportPrivateUsage]
    filesystem_arrived = threading.Event()
    release_filesystem = threading.Event()

    def gated_quarantine(destination: Path, quarantine: Path) -> None:
        filesystem_arrived.set()
        assert release_filesystem.wait(timeout=5)
        original_quarantine(destination, quarantine)

    monkeypatch.setattr(BlobStore, "_quarantine_durable", staticmethod(gated_quarantine))
    collection = asyncio.create_task(blobs.collect_garbage())
    assert await asyncio.to_thread(filesystem_arrived.wait, 5)

    unrelated = await asyncio.wait_for(
        blobs.put(b"unrelated writer", "application/octet-stream"), timeout=1
    )
    assert isinstance(unrelated, StoredBlob)
    release_filesystem.set()
    assert await collection == [garbage.hash]
    assert await blobs.get(unrelated.hash) == b"unrelated writer"


async def test_gc_pending_refuses_same_digest_adoption_until_retry(
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blobs = BlobStore(engine, data_dir, min_disk_headroom_bytes=1)
    payload = b"same digest during collection"
    stored = await blobs.put(payload, "application/octet-stream")
    assert isinstance(stored, StoredBlob)
    original_quarantine = BlobStore._quarantine_durable  # pyright: ignore[reportPrivateUsage]
    filesystem_arrived = threading.Event()
    release_filesystem = threading.Event()

    def gated_quarantine(destination: Path, quarantine: Path) -> None:
        filesystem_arrived.set()
        assert release_filesystem.wait(timeout=5)
        original_quarantine(destination, quarantine)

    monkeypatch.setattr(BlobStore, "_quarantine_durable", staticmethod(gated_quarantine))
    collection = asyncio.create_task(blobs.collect_garbage())
    assert await asyncio.to_thread(filesystem_arrived.wait, 5)

    with pytest.raises(CapacityRefusedError):
        await blobs.put(payload, "application/octet-stream")
    release_filesystem.set()
    assert await collection == [stored.hash]

    retried = await blobs.put(payload, "application/octet-stream")
    assert isinstance(retried, StoredBlob)
    assert await blobs.get(retried.hash) == payload


async def test_reference_created_during_gc_restores_quarantine_and_cancels_intent(
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blobs = BlobStore(engine, data_dir, min_disk_headroom_bytes=1)
    store = SqliteDocStore(engine)
    await store.ensure_workspace()
    payload = b"referenced during collection"
    stored = await blobs.put(payload, "application/octet-stream")
    assert isinstance(stored, StoredBlob)
    original_quarantine = BlobStore._quarantine_durable  # pyright: ignore[reportPrivateUsage]
    filesystem_arrived = threading.Event()
    release_filesystem = threading.Event()

    def gated_quarantine(destination: Path, quarantine: Path) -> None:
        filesystem_arrived.set()
        assert release_filesystem.wait(timeout=5)
        original_quarantine(destination, quarantine)

    monkeypatch.setattr(BlobStore, "_quarantine_durable", staticmethod(gated_quarantine))
    collection = asyncio.create_task(blobs.collect_garbage())
    assert await asyncio.to_thread(filesystem_arrived.wait, 5)
    document = make_document(body=payload).model_copy(update={"original_ref": stored.hash})
    await asyncio.wait_for(store.upsert_document(document), timeout=1)
    release_filesystem.set()

    assert await collection == []
    assert await blobs.get(stored.hash) == payload
    assert not any(blobs._gc_root().glob("*"))  # pyright: ignore[reportPrivateUsage]


async def test_gc_postcommit_error_leaves_no_file_or_ambiguous_descriptor(
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sqlalchemy.ext.asyncio import AsyncSession  # noqa: PLC0415

    blobs = BlobStore(engine, data_dir, min_disk_headroom_bytes=1)
    stored = await blobs.put(b"postcommit garbage", "application/octet-stream")
    assert isinstance(stored, StoredBlob)
    original_commit = AsyncSession.commit
    commits = 0

    async def commit_then_fail(session: AsyncSession) -> None:
        nonlocal commits
        commits += 1
        await original_commit(session)
        if commits == 2:
            raise OSError(errno.EIO, "garbage collection commit outcome ambiguous")

    monkeypatch.setattr(AsyncSession, "commit", commit_then_fail)
    with pytest.raises(OSError, match="commit outcome ambiguous"):
        await blobs.collect_garbage()

    assert not blobs.path_for(stored.hash).exists()
    async with engine.connect() as connection:
        assert (await connection.execute(text("SELECT COUNT(*) FROM blobs"))).scalar_one() == 0
    monkeypatch.setattr(AsyncSession, "commit", original_commit)
    restarted = BlobStore(engine, data_dir, min_disk_headroom_bytes=1)
    assert await restarted.collect_garbage() == [stored.hash]
    assert not _payload_files(restarted.root)


async def test_gc_precommit_failure_keeps_counted_restart_state_not_a_readable_row(
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sqlalchemy.ext.asyncio import AsyncSession  # noqa: PLC0415

    blobs = BlobStore(engine, data_dir, min_disk_headroom_bytes=1)
    payload = b"precommit garbage"
    stored = await blobs.put(payload, "application/octet-stream")
    assert isinstance(stored, StoredBlob)
    original_commit = AsyncSession.commit
    commits = 0

    async def fail_final_commit(session: AsyncSession) -> None:
        nonlocal commits
        commits += 1
        if commits == 2:
            raise OSError(errno.EIO, "garbage collection precommit refusal")
        await original_commit(session)

    monkeypatch.setattr(AsyncSession, "commit", fail_final_commit)
    with pytest.raises(OSError, match="precommit refusal"):
        await blobs.collect_garbage()

    assert not blobs.path_for(stored.hash).exists()
    assert await blobs.get(stored.hash) is None
    async with engine.connect() as connection:
        algo = (await connection.execute(text("SELECT algo FROM blobs"))).scalar_one()
    assert str(algo).startswith("gc_pending:")
    with pytest.raises(CapacityRefusedError):
        await blobs.put(payload, "application/octet-stream")

    monkeypatch.setattr(AsyncSession, "commit", original_commit)
    restarted = BlobStore(engine, data_dir, min_disk_headroom_bytes=1)
    assert await restarted.collect_garbage() == [stored.hash]
    assert not _payload_files(restarted.root)


async def test_gc_parent_fsync_failure_after_unlink_keeps_counted_restart_state(
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blobs = BlobStore(engine, data_dir, min_disk_headroom_bytes=1)
    payload = b"directory fsync garbage"
    stored = await blobs.put(payload, "application/octet-stream")
    assert isinstance(stored, StoredBlob)
    destination = blobs.path_for(stored.hash)
    original_sync = BlobStore._fsync_directory  # pyright: ignore[reportPrivateUsage]
    failed = False

    def fail_after_destination_unlink(path: Path) -> None:
        nonlocal failed
        if path == destination.parent and not destination.exists() and not failed:
            failed = True
            raise OSError(errno.EIO, "destination parent fsync refusal")
        original_sync(path)

    monkeypatch.setattr(BlobStore, "_fsync_directory", staticmethod(fail_after_destination_unlink))
    assert await blobs.collect_garbage() == []
    assert failed
    assert not destination.exists()
    assert await blobs.get(stored.hash) is None
    async with engine.connect() as connection:
        algo = (await connection.execute(text("SELECT algo FROM blobs"))).scalar_one()
    assert str(algo).startswith("gc_pending:")

    monkeypatch.setattr(BlobStore, "_fsync_directory", staticmethod(original_sync))
    restarted = BlobStore(engine, data_dir, min_disk_headroom_bytes=1)
    assert await restarted.collect_garbage() == [stored.hash]
    assert not _payload_files(restarted.root)


async def test_restart_recovers_pending_row_without_intent_after_new_reference(
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    blobs = BlobStore(engine, data_dir, min_disk_headroom_bytes=1)
    store = SqliteDocStore(engine)
    await store.ensure_workspace()
    payload = b"phase-one crash recovery"
    stored = await blobs.put(payload, "application/octet-stream")
    assert isinstance(stored, StoredBlob)
    token = "a" * 32
    async with engine.begin() as connection:
        await connection.execute(
            text("UPDATE blobs SET algo = :algo WHERE hash = :digest"),
            {"algo": f"gc_pending:{token}", "digest": stored.hash},
        )
    document = make_document(body=payload).model_copy(update={"original_ref": stored.hash})
    await store.upsert_document(document)

    restarted = BlobStore(engine, data_dir, min_disk_headroom_bytes=1)
    assert await restarted.collect_garbage() == []
    assert await restarted.get(stored.hash) == payload
    async with engine.connect() as connection:
        algo = (
            await connection.execute(
                text("SELECT algo FROM blobs WHERE hash = :digest"), {"digest": stored.hash}
            )
        ).scalar_one()
    assert algo == "blake2b"
    assert not restarted._gc_root().exists()  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    "corruption",
    [
        "representation",
        "intent_token",
        "intent_bool",
        "intent_extra",
        "intent_scalar",
        "intent_malformed",
        "intent_destination",
    ],
)
async def test_malformed_gc_artifacts_never_become_readable_or_promoted(
    engine: AsyncEngine,
    data_dir: Path,
    corruption: str,
) -> None:
    blobs = BlobStore(
        engine,
        data_dir,
        min_disk_headroom_bytes=1,
        max_acquired_blob_backlog_bytes=100,
    )
    store = SqliteDocStore(engine)
    await store.ensure_workspace()
    payload = b"valid retained body"
    stored = await blobs.put(payload, "application/octet-stream")
    assert isinstance(stored, StoredBlob)
    token = "b" * 32
    async with engine.begin() as connection:
        await connection.execute(
            text("UPDATE blobs SET algo = :algo WHERE hash = :digest"),
            {"algo": f"gc_pending:{token}", "digest": stored.hash},
        )
    quarantine, intent = blobs._gc_paths(  # pyright: ignore[reportPrivateUsage]
        stored.hash, token
    )
    intent_payload: object = {
        "digest": stored.hash,
        "stored_bytes": stored.stored_bytes,
        "token": token,
    }
    if corruption == "intent_token":
        assert isinstance(intent_payload, dict)
        intent_payload["token"] = "c" * 32
    elif corruption == "intent_bool":
        assert isinstance(intent_payload, dict)
        intent_payload["stored_bytes"] = True
    elif corruption in {"intent_extra", "intent_destination"}:
        assert isinstance(intent_payload, dict)
        intent_payload["private"] = "private-secret-cinder"
    elif corruption == "intent_scalar":
        intent_payload = "private-secret-cinder"
    if corruption == "intent_malformed":
        encoded_intent = b'{"private-secret-cinder":'
    else:
        encoded_intent = json.dumps(intent_payload).encode()
    BlobStore._write_durable(  # pyright: ignore[reportPrivateUsage]
        intent, encoded_intent
    )
    if corruption != "intent_destination":
        BlobStore._quarantine_durable(  # pyright: ignore[reportPrivateUsage]
            blobs.path_for(stored.hash), quarantine
        )
    if corruption == "representation":
        BlobStore._write_durable(  # pyright: ignore[reportPrivateUsage]
            quarantine, b"private malformed quarantine body"
        )
    document = make_document(body=payload).model_copy(update={"original_ref": stored.hash})
    await store.upsert_document(document)

    assert await blobs.collect_garbage() == []
    assert await blobs.get(stored.hash) is None
    with pytest.raises(CapacityRefusedError) as caught:
        await blobs.put(payload, "application/octet-stream")
    rendered = f"{caught.value!s}\n{caught.value!r}"
    assert "private malformed" not in rendered
    assert "private-secret-cinder" not in rendered
    async with engine.connect() as connection:
        algo = (
            await connection.execute(
                text("SELECT algo FROM blobs WHERE hash = :digest"), {"digest": stored.hash}
            )
        ).scalar_one()
    assert str(algo).startswith("gc_pending:")
    assert quarantine.exists() is (corruption != "intent_destination")
    assert blobs.path_for(stored.hash).exists() is (corruption == "intent_destination")


@pytest.mark.parametrize(
    ("compression", "size_bytes", "stored_bytes"),
    [
        ("brotli", 1, 1),
        ("none", True, 1),
        ("none", 1, True),
    ],
)
def test_gc_representation_requires_exact_descriptor_domains(
    data_dir: Path,
    compression: str,
    size_bytes: int,
    stored_bytes: int,
) -> None:
    payload = b"x"
    path = data_dir / "representation"
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)

    assert not BlobStore._gc_representation_matches(  # pyright: ignore[reportPrivateUsage]
        path,
        digest=content_hash(payload),
        compression=compression,
        size_bytes=size_bytes,
        stored_bytes=stored_bytes,
    )


async def test_invalid_pending_compression_never_normalizes_or_becomes_readable(
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    blobs = BlobStore(
        engine,
        data_dir,
        min_disk_headroom_bytes=1,
        max_acquired_blob_backlog_bytes=100,
    )
    store = SqliteDocStore(engine)
    await store.ensure_workspace()
    payload = b"invalid descriptor body"
    stored = await blobs.put(payload, "application/octet-stream")
    assert isinstance(stored, StoredBlob)
    token = "d" * 32
    async with engine.begin() as connection:
        await connection.execute(
            text("UPDATE blobs SET algo = :algo, compression = 'brotli' WHERE hash = :digest"),
            {"algo": f"gc_pending:{token}", "digest": stored.hash},
        )
    document = make_document(body=payload).model_copy(update={"original_ref": stored.hash})
    await store.upsert_document(document)

    assert await blobs.collect_garbage() == []
    assert await blobs.get(stored.hash) is None
    with pytest.raises(CapacityRefusedError):
        await blobs.put(payload, "application/octet-stream")
    async with engine.connect() as connection:
        algo, compression = (
            await connection.execute(
                text("SELECT algo, compression FROM blobs WHERE hash = :digest"),
                {"digest": stored.hash},
            )
        ).one()
    assert algo == f"gc_pending:{token}"
    assert compression == "brotli"


@pytest.mark.parametrize("token", ["e" * 32, "../private-secret-cinder"])
async def test_referenced_pending_descriptor_remains_capacity_accounted(
    engine: AsyncEngine,
    data_dir: Path,
    token: str,
) -> None:
    blobs = BlobStore(
        engine,
        data_dir,
        min_disk_headroom_bytes=1,
        max_acquired_blob_backlog_bytes=100,
    )
    store = SqliteDocStore(engine)
    await store.ensure_workspace()
    stored = await blobs.put(b"a" * 60, "application/octet-stream")
    assert isinstance(stored, StoredBlob)
    async with engine.begin() as connection:
        await connection.execute(
            text("UPDATE blobs SET algo = :algo WHERE hash = :digest"),
            {"algo": f"gc_pending:{token}", "digest": stored.hash},
        )
    document = make_document(body=b"a" * 60).model_copy(update={"original_ref": stored.hash})
    await store.upsert_document(document)

    with pytest.raises(CapacityRefusedError) as caught:
        await blobs.put(b"b" * 60, "application/octet-stream")
    rendered = f"{caught.value!s}\n{caught.value!r}"
    assert "private-secret-cinder" not in rendered
    assert await blobs.get(stored.hash) is None
    assert not blobs.path_for(content_hash(b"b" * 60)).exists()
    if token.startswith("../"):
        assert await blobs.collect_garbage() == []
        assert not blobs._gc_root().exists()  # pyright: ignore[reportPrivateUsage]


async def test_pending_capacity_uses_actual_malformed_quarantine_size(
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    blobs = BlobStore(
        engine,
        data_dir,
        min_disk_headroom_bytes=1,
        max_acquired_blob_backlog_bytes=100,
    )
    stored = await blobs.put(b"a" * 10, "application/octet-stream")
    assert isinstance(stored, StoredBlob)
    token = "f" * 32
    async with engine.begin() as connection:
        await connection.execute(
            text("UPDATE blobs SET algo = :algo WHERE hash = :digest"),
            {"algo": f"gc_pending:{token}", "digest": stored.hash},
        )
    quarantine, intent = blobs._gc_paths(  # pyright: ignore[reportPrivateUsage]
        stored.hash, token
    )
    BlobStore._write_durable(  # pyright: ignore[reportPrivateUsage]
        intent,
        json.dumps({"digest": stored.hash, "stored_bytes": 10, "token": token}).encode(),
    )
    BlobStore._quarantine_durable(  # pyright: ignore[reportPrivateUsage]
        blobs.path_for(stored.hash), quarantine
    )
    BlobStore._write_durable(  # pyright: ignore[reportPrivateUsage]
        quarantine, b"private malformed quarantine" + b"x" * 77
    )

    with pytest.raises(CapacityRefusedError) as caught:
        await blobs.put(b"b", "application/octet-stream")

    assert caught.value.diagnostic.used == 105
    assert "private malformed" not in f"{caught.value!s}\n{caught.value!r}"
    assert not blobs.path_for(content_hash(b"b")).exists()


async def test_hard_linked_pending_representations_are_charged_once(
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    blobs = BlobStore(
        engine,
        data_dir,
        min_disk_headroom_bytes=1,
        max_acquired_blob_backlog_bytes=100,
    )
    stored = await blobs.put(b"a" * 60, "application/octet-stream")
    assert isinstance(stored, StoredBlob)
    token = "1" * 32
    async with engine.begin() as connection:
        await connection.execute(
            text("UPDATE blobs SET algo = :algo WHERE hash = :digest"),
            {"algo": f"gc_pending:{token}", "digest": stored.hash},
        )
    quarantine, intent = blobs._gc_paths(  # pyright: ignore[reportPrivateUsage]
        stored.hash, token
    )
    BlobStore._write_durable(  # pyright: ignore[reportPrivateUsage]
        intent,
        json.dumps({"digest": stored.hash, "stored_bytes": 60, "token": token}).encode(),
    )
    os.link(blobs.path_for(stored.hash), quarantine)

    admitted = await blobs.put(b"b" * 40, "application/octet-stream")

    assert isinstance(admitted, StoredBlob)
    assert admitted.stored_bytes == 40


async def test_cross_identity_hard_link_claims_are_charged_once(
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    blobs = BlobStore(
        engine,
        data_dir,
        min_disk_headroom_bytes=1,
        max_acquired_blob_backlog_bytes=100,
    )
    first_quarantine, first_intent = blobs._gc_paths(  # pyright: ignore[reportPrivateUsage]
        "7" * 32, "8" * 32
    )
    second_quarantine, second_intent = blobs._gc_paths(  # pyright: ignore[reportPrivateUsage]
        "9" * 32, "a" * 32
    )
    for intent, digest, token in (
        (first_intent, "7" * 32, "8" * 32),
        (second_intent, "9" * 32, "a" * 32),
    ):
        BlobStore._write_durable(  # pyright: ignore[reportPrivateUsage]
            intent,
            json.dumps({"digest": digest, "stored_bytes": 60, "token": token}).encode(),
        )
    BlobStore._write_durable(  # pyright: ignore[reportPrivateUsage]
        first_quarantine, b"q" * 60
    )
    os.link(first_quarantine, second_quarantine)

    admitted = await blobs.put(b"b" * 40, "application/octet-stream")

    assert isinstance(admitted, StoredBlob)
    assert admitted.stored_bytes == 40


@pytest.mark.parametrize(("claimed", "actual"), [(80, 105), (105, 10)])
async def test_orphan_intent_counts_larger_of_claimed_and_actual_bytes(
    engine: AsyncEngine,
    data_dir: Path,
    claimed: int,
    actual: int,
) -> None:
    blobs = BlobStore(
        engine,
        data_dir,
        min_disk_headroom_bytes=1,
        max_acquired_blob_backlog_bytes=100,
    )
    digest = "2" * 32
    token = "3" * 32
    quarantine, intent = blobs._gc_paths(digest, token)  # pyright: ignore[reportPrivateUsage]
    BlobStore._write_durable(  # pyright: ignore[reportPrivateUsage]
        intent,
        json.dumps(
            {
                "digest": "4" * 32,
                "stored_bytes": claimed,
                "token": "5" * 32,
            }
        ).encode(),
    )
    BlobStore._write_durable(  # pyright: ignore[reportPrivateUsage]
        quarantine, b"q" * actual
    )

    with pytest.raises(CapacityRefusedError) as caught:
        await blobs.put(b"b", "application/octet-stream")

    assert caught.value.diagnostic.used == max(claimed, actual)


async def test_scalar_pending_size_refuses_capacity_without_rendering_value(
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    blobs = BlobStore(
        engine,
        data_dir,
        min_disk_headroom_bytes=1,
        max_acquired_blob_backlog_bytes=100,
    )
    stored = await blobs.put(b"a" * 10, "application/octet-stream")
    assert isinstance(stored, StoredBlob)
    async with engine.begin() as connection:
        await connection.execute(
            text("UPDATE blobs SET algo = :algo, stored_bytes = :value WHERE hash = :digest"),
            {
                "algo": f"gc_pending:{'6' * 32}",
                "digest": stored.hash,
                "value": "private-secret-cinder",
            },
        )

    with pytest.raises(CapacityRefusedError) as caught:
        await blobs.put(b"b", "application/octet-stream")

    assert caught.value.diagnostic.resource is CapacityResource.ACQUIRED_BLOB_BACKLOG_BYTES
    rendered = f"{caught.value!s}\n{caught.value!r}"
    assert "private-secret-cinder" not in rendered
    assert not blobs.path_for(content_hash(b"b")).exists()


async def test_huge_valid_descriptors_saturate_without_sqlite_sum_overflow(
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    five_exabytes = 5_000_000_000_000_000_000
    exact_total = five_exabytes * 2
    async with engine.begin() as connection:
        for index in range(2):
            await connection.execute(
                text(
                    "INSERT INTO blobs "
                    "(hash, algo, media_type, size_bytes, stored_bytes, compression, created_at) "
                    "VALUES (:digest, 'blake2b', NULL, :size, :size, 'none', CURRENT_TIMESTAMP)"
                ),
                {"digest": f"{index + 10:032x}", "size": five_exabytes},
            )
    limited = BlobStore(
        engine,
        data_dir,
        min_disk_headroom_bytes=1,
        max_acquired_blob_backlog_bytes=exact_total,
    )

    with pytest.raises(CapacityRefusedError) as caught:
        await limited.put(b"x", "application/octet-stream")

    assert caught.value.diagnostic.used == exact_total
    assert caught.value.diagnostic.requested == 1
    assert caught.value.diagnostic.limit == exact_total
    assert not limited.path_for(content_hash(b"x")).exists()

    boundary = BlobStore(
        engine,
        data_dir,
        min_disk_headroom_bytes=1,
        max_acquired_blob_backlog_bytes=exact_total + 1,
    )
    admitted = await boundary.put(b"x", "application/octet-stream")
    assert isinstance(admitted, StoredBlob)
    assert admitted.stored_bytes == 1


async def test_gc_capacity_artifact_scan_is_bounded(
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blobs = BlobStore(
        engine,
        data_dir,
        min_disk_headroom_bytes=1,
        max_acquired_blob_backlog_bytes=100,
    )
    root = blobs._gc_root()  # pyright: ignore[reportPrivateUsage]
    root.mkdir(parents=True)
    for index in range(20):
        (root / f"opaque-{index}.blob").write_bytes(b"private")
    monkeypatch.setattr("manicule.storage.blobs.GC_CAPACITY_SCAN_LIMIT", 2)
    path_type = type(root)
    original_iterdir = path_type.iterdir
    yielded = 0

    def counted_iterdir(path: Path) -> Any:
        nonlocal yielded
        for candidate in original_iterdir(path):
            if path == root:
                yielded += 1
            yield candidate

    monkeypatch.setattr(path_type, "iterdir", counted_iterdir)

    with pytest.raises(CapacityRefusedError):
        await blobs.put(b"b", "application/octet-stream")

    assert yielded == 3
    assert not blobs.path_for(content_hash(b"b")).exists()


async def test_nondirectory_gc_root_refuses_admission_and_skips_collection(
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    blobs = BlobStore(
        engine,
        data_dir,
        min_disk_headroom_bytes=1,
        max_acquired_blob_backlog_bytes=100,
    )
    root = blobs._gc_root()  # pyright: ignore[reportPrivateUsage]
    root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.write_bytes(b"private malformed garbage-collection root")

    with pytest.raises(CapacityRefusedError) as caught:
        await blobs.put(b"new", "application/octet-stream")

    assert caught.value.diagnostic.resource is CapacityResource.ACQUIRED_BLOB_BACKLOG_BYTES
    assert await blobs.collect_garbage() == []
    assert not blobs.path_for(content_hash(b"new")).exists()
    assert "private malformed" not in str(caught.value)


async def test_pending_descriptor_inventory_overflow_refuses_and_releases_writer(
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blobs = BlobStore(
        engine,
        data_dir,
        min_disk_headroom_bytes=1,
        max_acquired_blob_backlog_bytes=100,
    )
    seed = await blobs.put(b"seed", "application/octet-stream")
    assert isinstance(seed, StoredBlob)
    monkeypatch.setattr("manicule.storage.blobs.GC_CAPACITY_SCAN_LIMIT", 2)
    async with engine.begin() as connection:
        await connection.execute(
            text("UPDATE blobs SET algo = :algo WHERE hash = :digest"),
            {"algo": f"gc_pending:{'b' * 32}", "digest": seed.hash},
        )
        for index in range(2):
            await connection.execute(
                text(
                    "INSERT INTO blobs "
                    "(hash, algo, media_type, size_bytes, stored_bytes, compression, created_at) "
                    "SELECT :digest, :algo, media_type, size_bytes, stored_bytes, compression, "
                    "created_at FROM blobs WHERE hash = :seed"
                ),
                {
                    "algo": f"gc_pending:{index:032x}",
                    "digest": f"{index + 1:032x}",
                    "seed": seed.hash,
                },
            )

    def unexpected_inventory(_claims: object) -> int:
        msg = "overflow must refuse before filesystem inventory"
        raise AssertionError(msg)

    monkeypatch.setattr(blobs, "_gc_artifact_bytes", unexpected_inventory)

    with pytest.raises(CapacityRefusedError):
        await blobs.put(b"new", "application/octet-stream")

    async def unrelated_writer() -> None:
        async with engine.begin() as connection:
            await connection.execute(
                text("UPDATE blobs SET media_type = media_type WHERE hash = :digest"),
                {"digest": seed.hash},
            )

    await asyncio.wait_for(unrelated_writer(), timeout=1)
    assert not blobs.path_for(content_hash(b"new")).exists()


async def test_sweep_atomically_rechecks_marker_created_after_candidate_selection(
    engine: AsyncEngine, data_dir: Path
) -> None:
    """A collector cannot delete a blob after acquisition establishes its durable root."""
    candidate_selected = asyncio.Event()
    release_delete = asyncio.Event()

    class GatedBlobStore(BlobStore):
        @override
        async def _run_gc_intent(self, digest: str, token: str) -> bool:
            candidate_selected.set()
            await release_delete.wait()
            return await super()._run_gc_intent(digest, token)

    blobs = GatedBlobStore(engine, data_dir)
    raw = RawDocument(
        source_id="raced-source",
        uri="memory:raced-source",
        media_type="text/plain",
        content=b"raced bytes",
    )
    stored = await blobs.put(raw.as_bytes(), raw.media_type)
    assert isinstance(stored, StoredBlob)

    collection = asyncio.create_task(blobs.collect_garbage())
    await asyncio.wait_for(candidate_selected.wait(), timeout=5)
    async with engine.connect() as connection:
        assert (
            await connection.execute(
                text("SELECT count(*) FROM blobs WHERE hash = :digest"),
                {"digest": stored.hash},
            )
        ).scalar_one() == 1
    acquired = AcquiredSource.from_raw(raw)
    name = blobs._stage_name("raced-run\0raced-source")  # pyright: ignore[reportPrivateUsage]
    async with blobs._marker_locks([name]):  # pyright: ignore[reportPrivateUsage]
        await blobs._record_marker(  # pyright: ignore[reportPrivateUsage]
            name,
            {
                "run_id": "raced-run",
                "source_id": "raced-source",
                "blob_ref": stored.hash,
                "compression": stored.compression,
                "acquired_source": acquired.model_dump(mode="json"),
            },
            legacy=False,
        )
    release_delete.set()

    assert await collection == []
    retained, _ = await blobs.retain_acquisition("raced-run\0raced-source", raw)
    assert retained.ref == stored.hash
    assert await blobs.get(stored.hash) == raw.as_bytes()


async def test_sweep_final_lock_rechecks_blob_row_recreated_by_ordinary_put(
    engine: AsyncEngine, data_dir: Path
) -> None:
    """A normal put recreating the row in the delete/unlink gap keeps the physical bytes."""
    row_deleted = asyncio.Event()
    release_unlink = asyncio.Event()

    class GatedBlobStore(BlobStore):
        @override
        async def _run_gc_intent(self, digest: str, token: str) -> bool:
            row_deleted.set()
            await release_unlink.wait()
            return await super()._run_gc_intent(digest, token)

    blobs = GatedBlobStore(engine, data_dir)
    payload = b"ordinary put race"
    stored = await blobs.put(payload, "text/plain")
    assert isinstance(stored, StoredBlob)

    collection = asyncio.create_task(blobs.collect_garbage())
    await asyncio.wait_for(row_deleted.wait(), timeout=5)
    with pytest.raises(CapacityRefusedError):
        await blobs.put(payload, "text/plain")
    release_unlink.set()

    assert await collection == [stored.hash]
    recreated = await blobs.put(payload, "text/plain")
    assert isinstance(recreated, StoredBlob)
    assert await blobs.get(stored.hash) == payload


async def test_cross_process_lock_files_are_bounded_by_fixed_shard_pool(
    engine: AsyncEngine, data_dir: Path
) -> None:
    blobs = BlobStore(engine, data_dir)
    keys = [f"marker-{index}" for index in range(5_000)]

    async with blobs._marker_locks(keys):  # pyright: ignore[reportPrivateUsage]
        pass

    lock_root = blobs.root / "acquisition-locks"
    assert len(list(lock_root.iterdir())) <= 256


async def test_conflicting_marker_retry_preserves_durable_evidence_and_repairs_stale_inventory(
    engine: AsyncEngine, data_dir: Path
) -> None:
    blobs = BlobStore(engine, data_dir)
    key = "retry-run\0retry-source"
    old = RawDocument(
        source_id="retry-source",
        uri="memory:retry-source",
        media_type="text/plain",
        content=b"old bytes",
    )
    new = old.model_copy(update={"content": b"new bytes"})
    old_result = await blobs.retain_acquisition(key, old)

    with pytest.raises(RuntimeError, match="conflicts with durable recovery evidence"):
        await blobs.retain_acquisition(key, new)
    assert await blobs.resume_acquisition(key) == old_result

    new_blob = await blobs.put(new.as_bytes(), new.media_type)
    assert isinstance(new_blob, StoredBlob)
    new_acquired = AcquiredSource.from_raw(new)
    path = blobs._stage_path(key)  # pyright: ignore[reportPrivateUsage]
    path.write_text(
        json.dumps(
            {
                "run_id": "retry-run",
                "source_id": "retry-source",
                "blob_ref": new_blob.hash,
                "compression": new_blob.compression,
                "acquired_source": new_acquired.model_dump(mode="json"),
            }
        )
    )

    repaired = await blobs.retain_acquisition(key, new)
    assert repaired[0].ref == new_blob.hash
    async with engine.connect() as connection:
        inventory_ref = (
            await connection.execute(
                text("SELECT blob_ref FROM acquisition_markers WHERE run_id = 'retry-run'")
            )
        ).scalar_one()
    assert inventory_ref == new_blob.hash


async def test_inventory_only_crash_allows_newer_serialized_retry(
    engine: AsyncEngine, data_dir: Path
) -> None:
    blobs = BlobStore(engine, data_dir)
    key = "inventory-only-run\0inventory-only-source"
    old = RawDocument(
        source_id="inventory-only-source",
        uri="memory:inventory-only-source",
        media_type="text/plain",
        content=b"inventory old",
    )
    new = old.model_copy(update={"content": b"inventory new"})
    await blobs.retain_acquisition(key, old)
    marker = blobs._stage_path(key)  # pyright: ignore[reportPrivateUsage]
    marker.unlink()

    retained = await blobs.retain_acquisition(key, new)

    assert retained[0].ref == content_hash(b"inventory new")
    assert await blobs.resume_acquisition(key) == retained


async def test_old_missing_marker_for_authoritative_run_survives_reconcile_retry_race(
    engine: AsyncEngine, data_dir: Path
) -> None:
    blobs = BlobStore(engine, data_dir)
    store = SqliteDocStore(engine)
    await store.ensure_workspace()
    await store.create_acquisition_run("authoritative-run", "authoritative-connector")
    key = "authoritative-run\0authoritative-source"
    old = RawDocument(
        source_id="authoritative-source",
        uri="memory:authoritative-source",
        media_type="text/plain",
        content=b"authoritative old",
    )
    new = old.model_copy(update={"content": b"authoritative new"})
    await blobs.retain_acquisition(key, old)
    blobs._stage_path(key).unlink()  # pyright: ignore[reportPrivateUsage]
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "UPDATE acquisition_markers SET created_at = :old "
                "WHERE run_id = 'authoritative-run'"
            ),
            {"old": "2020-01-01 00:00:00.000000"},
        )

    retried, _reconciled = await asyncio.gather(
        blobs.retain_acquisition(key, new),
        blobs.reconcile_acquisition_markers(),
    )

    assert retried[0].ref == content_hash(b"authoritative new")
    assert await blobs.resume_acquisition(key) == retried


async def test_preassociation_marker_supersession_releases_only_the_fenced_blob(
    engine: AsyncEngine, data_dir: Path
) -> None:
    blobs = BlobStore(engine, data_dir)
    store = SqliteDocStore(engine)
    await store.ensure_workspace()
    now = datetime(2026, 8, 15, 12, tzinfo=UTC)

    async def staged_run(run_id: str, connector: str, body: bytes) -> tuple[str, str]:
        await store.create_acquisition_run(run_id, connector)
        run = await store.claim_acquisition_run(
            run_id, "owner", now=now, expires_at=now + timedelta(minutes=1)
        )
        assert run is not None
        source_id = f"{run_id}-source"
        source = DiscoveredDoc(ref=DocRef(source_id=source_id, uri=f"memory:{source_id}"))
        await store.append_acquisition_record(
            run_id,
            0,
            AcquisitionSource.from_discovered(source),
            lease_owner="owner",
            lease_generation=run.lease_generation,
            now=now,
        )
        await store.transition_acquisition_record(
            run_id,
            source_id,
            AcquisitionRecordState.DISCOVERED,
            AcquisitionRecordState.ACQUIRING,
            lease_owner="owner",
            lease_generation=run.lease_generation,
            now=now,
        )
        raw = RawDocument(
            source_id=source_id, uri=f"memory:{source_id}", media_type="text/plain", content=body
        )
        retained, _ = await blobs.retain_acquisition(f"{run_id}\0{source_id}", raw)
        assert retained.ref is not None
        return source_id, retained.ref

    stale_source, stale_ref = await staged_run("stale-run", "stale", b"stale bytes")
    live_source, live_ref = await staged_run("live-run", "live", b"live bytes")
    live_path = blobs._stage_path(  # pyright: ignore[reportPrivateUsage]
        f"live-run\0{live_source}"
    )
    legacy_payload = json.loads(live_path.read_text())
    legacy_payload.pop("run_id")
    legacy_payload.pop("source_id")
    live_path.write_text(json.dumps(legacy_payload))
    async with engine.begin() as connection:
        await connection.execute(
            text("DELETE FROM acquisition_markers WHERE name = :name"),
            {"name": live_path.name},
        )
    await store.set_watermark("stale", Watermark(value="new", observed_at=now))
    replacement = await store.claim_or_create_acquisition_run(
        "stale",
        "replacement",
        "replacement-owner",
        now=now + timedelta(minutes=2),
        expires_at=now + timedelta(minutes=3),
    )
    assert replacement is not None

    restarted = BlobStore(engine, data_dir)
    assert not await restarted.reconcile_acquisition_markers()
    assert await restarted.reconcile_acquisition_markers()
    assert await restarted.resume_acquisition(f"stale-run\0{stale_source}") is None
    assert await restarted.resume_acquisition(f"live-run\0{live_source}") is not None
    assert await store.cleanup_acquisition_history(datetime(2100, 1, 1, tzinfo=UTC), limit=10) == 1
    assert await restarted.collect_garbage() == [stale_ref]
    assert await restarted.get(live_ref) == b"live bytes"


async def test_legacy_marker_is_inferred_or_expires_without_blocking_forever(
    engine: AsyncEngine, data_dir: Path
) -> None:
    blobs = BlobStore(engine, data_dir)
    stored = await blobs.put(b"legacy orphan", "text/plain")
    assert isinstance(stored, StoredBlob)
    key = "missing-run\0legacy-source"
    path = blobs._stage_path(key)  # pyright: ignore[reportPrivateUsage]
    path.parent.mkdir(parents=True, exist_ok=True)
    legacy_raw = RawDocument(
        source_id="legacy-source",
        uri="memory:legacy-source",
        media_type="text/plain",
        content=b"legacy orphan",
    )
    path.write_text(
        json.dumps(
            {
                "blob_ref": stored.hash,
                "compression": "none",
                "acquired_source": AcquiredSource.from_raw(legacy_raw).model_dump(mode="json"),
            }
        )
    )
    old = (datetime.now(UTC) - timedelta(days=31)).timestamp()
    os.utime(path, (old, old))

    restarted = BlobStore(engine, data_dir)
    assert not await restarted.reconcile_acquisition_markers()
    assert not path.exists()
    assert await restarted.reconcile_acquisition_markers()
    assert await restarted.collect_garbage() == [stored.hash]


async def test_explicit_marker_is_removed_when_its_cascaded_run_owner_disappears(
    engine: AsyncEngine, data_dir: Path
) -> None:
    blobs = BlobStore(engine, data_dir)
    store = SqliteDocStore(engine)
    await store.ensure_workspace()
    await store.create_acquisition_run("vanished-run", "vanished-connector")
    raw = RawDocument(
        source_id="vanished-source",
        uri="memory:vanished-source",
        media_type="text/plain",
        content=b"vanished bytes",
    )
    retained, _ = await blobs.retain_acquisition("vanished-run\0vanished-source", raw)
    assert retained.ref is not None
    marker = blobs._stage_path(  # pyright: ignore[reportPrivateUsage]
        "vanished-run\0vanished-source"
    )
    async with engine.begin() as connection:
        await connection.execute(text("DELETE FROM acquisition_runs WHERE id = 'vanished-run'"))

    await blobs.reconcile_acquisition_markers()

    assert not marker.exists()
    assert await blobs.collect_garbage() == [retained.ref]


async def test_large_forged_marker_directory_reconciles_in_bounded_pages(
    engine: AsyncEngine, data_dir: Path
) -> None:
    blobs = BlobStore(engine, data_dir)
    staging = blobs.root / "acquisition-staging"
    staging.mkdir(parents=True)
    for index in range(250):
        (staging / f"forged-{index:04d}").write_text("{}")

    assert not await blobs.reconcile_acquisition_markers()
    async with engine.connect() as connection:
        first = (
            await connection.execute(text("SELECT count(*) FROM acquisition_markers"))
        ).scalar_one()
    assert first == 100
    assert not await blobs.reconcile_acquisition_markers()
    assert not await blobs.reconcile_acquisition_markers()
    assert await blobs.reconcile_acquisition_markers()
    async with engine.connect() as connection:
        final = (
            await connection.execute(text("SELECT count(*) FROM acquisition_markers"))
        ).scalar_one()
    assert final == 250

    async with engine.connect() as connection:
        plan = (
            await connection.execute(
                text(
                    "EXPLAIN QUERY PLAN SELECT run_id, source_id, marker_name "
                    "FROM acquisition_records WHERE marker_name IN ('forged-0000')"
                )
            )
        ).all()
    assert any("ix_acquisition_records_marker_name" in str(row) for row in plan)


async def test_legacy_admission_cannot_overwrite_concurrent_new_marker_registration(
    engine: AsyncEngine, data_dir: Path
) -> None:
    admission_ready = asyncio.Event()
    release_admission = asyncio.Event()

    class GatedLegacyStore(BlobStore):
        @override
        async def _record_markers(
            self,
            markers: Sequence[tuple[str, dict[str, object], bool, datetime]],
        ) -> None:
            admission_ready.set()
            await release_admission.wait()
            await super()._record_markers(markers)

    legacy_store = GatedLegacyStore(engine, data_dir)
    newer_store = BlobStore(engine, data_dir)
    await newer_store._cleanup_staging_once()  # pyright: ignore[reportPrivateUsage]
    store = SqliteDocStore(engine)
    await store.ensure_workspace()
    await store.create_acquisition_run("concurrent-run", "concurrent-connector")
    key = "concurrent-run\0concurrent-source"
    old = RawDocument(
        source_id="concurrent-source",
        uri="memory:concurrent-source",
        media_type="text/plain",
        content=b"legacy bytes",
    )
    new = old.model_copy(update={"content": b"new bytes"})
    old_blob = await legacy_store.put(old.as_bytes(), old.media_type)
    assert isinstance(old_blob, StoredBlob)
    marker = legacy_store._stage_path(key)  # pyright: ignore[reportPrivateUsage]
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "blob_ref": old_blob.hash,
                "compression": old_blob.compression,
                "acquired_source": AcquiredSource.from_raw(old).model_dump(mode="json"),
            }
        )
    )

    admission = asyncio.create_task(legacy_store.reconcile_acquisition_markers())
    await asyncio.wait_for(admission_ready.wait(), timeout=5)
    retained, _ = await newer_store.retain_acquisition(key, new)
    release_admission.set()
    await admission

    async with engine.connect() as connection:
        inventory_ref = (
            await connection.execute(
                text("SELECT blob_ref FROM acquisition_markers WHERE name = :name"),
                {"name": marker.name},
            )
        ).scalar_one()
    assert inventory_ref == retained.ref == content_hash(b"new bytes")


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
