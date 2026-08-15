"""Retained original bytes: what is kept, what is refused, and what the sweep reclaims."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import threading
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, override

import pytest
from sqlalchemy import text

from manicule.core.acquisition import AcquiredSource, AcquisitionRecordState, AcquisitionSource
from manicule.core.content import RawDocument
from manicule.core.ids import content_hash
from manicule.core.sources import DiscoveredDoc, DocRef, Watermark
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
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600


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


async def test_sweep_atomically_rechecks_marker_created_after_candidate_selection(
    engine: AsyncEngine, data_dir: Path
) -> None:
    """A collector cannot delete a blob after acquisition establishes its durable root."""
    candidate_selected = asyncio.Event()
    release_delete = asyncio.Event()

    class GatedBlobStore(BlobStore):
        @override
        async def _unlink_blob_if_still_unreferenced(self, digest: str) -> bool:
            candidate_selected.set()
            await release_delete.wait()
            return await super()._unlink_blob_if_still_unreferenced(digest)

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
        ).scalar_one() == 0
    retained, _ = await blobs.retain_acquisition("raced-run\0raced-source", raw)
    assert retained.ref == stored.hash
    release_delete.set()

    assert await collection == []
    assert await blobs.get(stored.hash) == raw.as_bytes()


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
    await store.set_watermark(
        "stale", Watermark(value="new", observed_at=now)
    )
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
    assert await store.cleanup_acquisition_history(
        datetime(2100, 1, 1, tzinfo=UTC), limit=10
    ) == 1
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
        await connection.execute(
            text("DELETE FROM acquisition_runs WHERE id = 'vanished-run'")
        )

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
