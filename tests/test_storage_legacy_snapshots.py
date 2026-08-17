"""Legacy published originals become honest, source-free snapshot manifests."""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import func, select, update

from manicule.core.acquisition import (
    AcquiredSource,
    AcquisitionFailureCode,
    AcquisitionRecordState,
    AcquisitionRun,
    AcquisitionRunState,
    AcquisitionSource,
    SnapshotCompleteness,
    SnapshotItemOutcome,
    SnapshotPromotionPolicy,
)
from manicule.core.content import DocumentStatus
from manicule.core.sources import DocRef
from manicule.storage import models
from manicule.storage.blobs import BlobStore, StoredBlob
from manicule.storage.docstore import SqliteDocStore
from manicule.storage.engine import create_engine
from manicule.storage.legacy_snapshots import (
    LEASE_DURATION,
    LEGACY_SCOPE,
    LEGACY_SCOPE_PREFIX,
    LegacySnapshotMigration,
    legacy_scope_fingerprint,
    legacy_snapshot_run_id,
    migrate_legacy_snapshots,
)
from manicule.storage.migrator import upgrade
from manicule.storage.types import utcnow
from tests.storage_helpers import make_document

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


async def _legacy_document(
    store: SqliteDocStore,
    blobs: BlobStore,
    source_id: str,
    body: bytes,
    *,
    source: str = "wiki",
    version_token: str | None = None,
) -> tuple[models.Document, StoredBlob]:
    retained = await blobs.put(body, "text/markdown")
    assert isinstance(retained, StoredBlob)
    document = make_document(
        source=source, source_id=source_id, body=body, workspace_id=store.workspace_id
    ).model_copy(
        update={
            "original_ref": retained.hash,
            "version_token": version_token or f"version-{source_id}",
            "metadata": {
                "citation": "synthetic",
                "_source": {
                    "provider": "wiki",
                    "provider_url": "https://wiki.example.test",
                    "source_id": source_id,
                },
            },
        }
    )
    await store.upsert_document(document)
    async with store.sessions() as session:
        row = (
            await session.execute(select(models.Document).where(models.Document.id == document.id))
        ).scalar_one()
    return row, retained


async def _legacy_run(store: SqliteDocStore, connector: str) -> AcquisitionRun:
    async with store.sessions() as session:
        run_id = (
            await session.execute(
                select(models.AcquisitionRun.id).where(
                    models.AcquisitionRun.workspace_id == store.workspace_id,
                    models.AcquisitionRun.connector_name == connector,
                    models.AcquisitionRun.scope_fingerprint.like(f"{LEGACY_SCOPE_PREFIX}%"),
                )
            )
        ).scalar_one()
    run = await store.get_acquisition_run(run_id)
    assert run is not None
    return run


async def _promote_current_document(
    store: SqliteDocStore, row: models.Document, retained: StoredBlob, *, run_id: str
) -> None:
    created = await store.create_acquisition_run(run_id, row.source, scope_fingerprint="all")
    now = utcnow()
    claimed = await store.claim_acquisition_run(
        created.id, "worker", now=now, expires_at=now + LEASE_DURATION
    )
    assert claimed is not None
    await store.append_acquisition_record(
        claimed.id,
        0,
        AcquisitionSource(
            ref=DocRef(source_id=row.source_id, uri=row.uri),
            version_token=row.version_token,
            media_type=row.media_type,
        ),
        lease_owner="worker",
        lease_generation=claimed.lease_generation,
        now=utcnow(),
    )
    await store.complete_acquisition_enumeration(
        claimed.id,
        None,
        lease_owner="worker",
        lease_generation=claimed.lease_generation,
        now=utcnow(),
    )
    await store.transition_acquisition_record(
        claimed.id,
        row.source_id,
        AcquisitionRecordState.DISCOVERED,
        AcquisitionRecordState.ACQUIRING,
        lease_owner="worker",
        lease_generation=claimed.lease_generation,
        now=utcnow(),
    )
    await store.transition_acquisition_record(
        claimed.id,
        row.source_id,
        AcquisitionRecordState.ACQUIRING,
        AcquisitionRecordState.ACQUIRED,
        lease_owner="worker",
        lease_generation=claimed.lease_generation,
        now=utcnow(),
        blob_ref=retained.hash,
        acquired_source=AcquiredSource(
            source_id=row.source_id,
            uri=row.uri,
            media_type=row.media_type,
            encoding="utf-8",
            text_content=False,
            content_hash=retained.hash,
            byte_length=retained.size_bytes,
        ),
        fetched_version_token=row.version_token,
    )
    await store.complete_snapshot_acquisition(
        claimed.id,
        lease_owner="worker",
        lease_generation=claimed.lease_generation,
        now=utcnow(),
    )
    await store.promote_snapshot_and_commit_watermark(
        claimed.id,
        expected_scope_fingerprint="all",
        lease_owner="worker",
        lease_generation=claimed.lease_generation,
        now=utcnow(),
    )


@pytest.mark.contract
async def test_retained_missing_and_corrupt_originals_migrate_without_changing_publication(
    engine: AsyncEngine, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SqliteDocStore(engine)
    await store.ensure_workspace()
    blobs = BlobStore(engine, data_dir)
    retained_row, retained = await _legacy_document(store, blobs, "a-retained", b"alpha")
    missing_row, missing = await _legacy_document(store, blobs, "b-missing", b"beta")
    corrupt_row, corrupt = await _legacy_document(store, blobs, "c-corrupt", b"gamma")
    blobs.path_for(corrupt.hash).write_bytes(b"not gamma")
    get_blob = blobs.get

    async def race_safe_get(ref: str) -> bytes | None:
        if ref == missing.hash:
            raise FileNotFoundError
        return await get_blob(ref)

    monkeypatch.setattr(blobs, "get", race_safe_get)

    before = {
        row.id: (row.publication_id, row.status, row.content_hash)
        for row in (retained_row, missing_row, corrupt_row)
    }
    result = await migrate_legacy_snapshots(store, blobs, page_size=1)

    assert (result.retained, result.missing, result.corrupt, result.promoted) == (1, 1, 1, 1)
    run = await _legacy_run(store, "wiki")
    assert run.scope_inventory_complete is False
    assert run.completeness is SnapshotCompleteness.PARTIAL
    assert run.promoted_at is not None
    assert run.watermark_committed_at is None
    assert run.candidate_watermark is None
    assert run.lease_owner is None
    assert run.lease_expires_at is None
    assert run.omission_count == 2
    assert run.omission_reasons == {"missing_body": 1, "corrupt_body": 1}

    records = await store.list_acquisition_records(run.id, limit=10)
    assert [record.snapshot_outcome for record in records] == [
        SnapshotItemOutcome.RETAINED,
        SnapshotItemOutcome.OMITTED,
        SnapshotItemOutcome.OMITTED,
    ]
    assert records[0].source.version_token == "version-a-retained"  # noqa: S105 - source token
    assert records[0].fetched_version_token == "version-a-retained"  # noqa: S105 - source token
    assert records[0].acquired_source is not None
    assert records[0].acquired_source.byte_length == len(b"alpha")
    assert records[0].acquired_source.metadata["citation"] == "synthetic"
    assert records[1].snapshot_diagnostic is not None
    assert records[1].snapshot_diagnostic.code is AcquisitionFailureCode.MISSING_BODY
    assert records[2].snapshot_diagnostic is not None
    assert records[2].snapshot_diagnostic.code is AcquisitionFailureCode.CORRUPT_BODY
    assert await store.verify_snapshot_manifest(run.id)

    async with store.sessions() as session:
        rows = (
            (await session.execute(select(models.Document).where(models.Document.id.in_(before))))
            .scalars()
            .all()
        )
    assert {row.id: (row.publication_id, row.status, row.content_hash) for row in rows} == before
    assert all(row.status is DocumentStatus.INDEXED for row in rows)

    # Snapshot ownership, not the derived publication, now keeps the healthy original alive.
    await store.delete_document(retained_row.id)
    assert retained.hash not in await blobs.collect_garbage()


@pytest.mark.contract
async def test_interruption_resumes_and_rerun_deduplicates_manifest_rows(
    engine: AsyncEngine, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SqliteDocStore(engine)
    await store.ensure_workspace()
    blobs = BlobStore(engine, data_dir)
    await _legacy_document(store, blobs, "a", b"alpha")
    await _legacy_document(store, blobs, "b", b"beta")
    get = blobs.get
    calls = 0

    async def interrupted(digest: str) -> bytes | None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise asyncio.CancelledError
        return await get(digest)

    monkeypatch.setattr(blobs, "get", interrupted)
    with pytest.raises(asyncio.CancelledError):
        await migrate_legacy_snapshots(store, blobs, page_size=1)
    monkeypatch.setattr(blobs, "get", get)

    resumed = await migrate_legacy_snapshots(store, blobs, page_size=1)
    rerun = await migrate_legacy_snapshots(store, blobs, page_size=1)
    run = await _legacy_run(store, "wiki")
    async with store.sessions() as session:
        counts = (
            await session.execute(
                select(
                    func.count(func.distinct(models.AcquisitionRun.id)),
                    func.count(models.AcquisitionRecord.id),
                )
                .select_from(models.AcquisitionRun)
                .join(
                    models.AcquisitionRecord,
                    models.AcquisitionRecord.run_id == models.AcquisitionRun.id,
                )
                .where(models.AcquisitionRun.id == run.id)
            )
        ).one()
    assert resumed.resumed == 1
    assert rerun.promoted == 0
    assert counts == (1, 2)
    assert await store.verify_snapshot_manifest(run.id)


@pytest.mark.contract
async def test_interruption_after_promotion_resumes_settlement(
    engine: AsyncEngine, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SqliteDocStore(engine)
    await store.ensure_workspace()
    blobs = BlobStore(engine, data_dir)
    await _legacy_document(store, blobs, "published-before-interrupt", b"durable")
    promote = store.promote_snapshot_and_commit_watermark

    async def interrupt_after_promotion(
        run_id: str,
        *,
        expected_scope_fingerprint: str,
        lease_owner: str,
        lease_generation: int,
        now: datetime,
    ) -> AcquisitionRun:
        await promote(
            run_id,
            expected_scope_fingerprint=expected_scope_fingerprint,
            lease_owner=lease_owner,
            lease_generation=lease_generation,
            now=now,
        )
        raise asyncio.CancelledError

    monkeypatch.setattr(store, "promote_snapshot_and_commit_watermark", interrupt_after_promotion)
    with pytest.raises(asyncio.CancelledError):
        await migrate_legacy_snapshots(store, blobs)
    promoted = await _legacy_run(store, "wiki")
    assert promoted.promoted_at is not None
    assert promoted.state is AcquisitionRunState.ACQUIRING

    monkeypatch.setattr(store, "promote_snapshot_and_commit_watermark", promote)
    resumed = await migrate_legacy_snapshots(store, blobs)
    settled = await _legacy_run(store, "wiki")
    assert resumed.resumed == 1
    assert settled.state is AcquisitionRunState.SETTLED


@pytest.mark.contract
async def test_migration_is_workspace_and_connector_isolated(
    engine: AsyncEngine, data_dir: Path
) -> None:
    first = SqliteDocStore(engine, workspace_id="first")
    second = SqliteDocStore(engine, workspace_id="second")
    await first.ensure_workspace()
    await second.ensure_workspace()
    blobs = BlobStore(engine, data_dir)
    await _legacy_document(first, blobs, "shared-id", b"first", source="wiki")
    await _legacy_document(first, blobs, "drive-id", b"drive", source="drive")
    await _legacy_document(second, blobs, "shared-id", b"second", source="wiki")

    migrated = await migrate_legacy_snapshots(first, blobs, page_size=1)
    assert (migrated.connectors, migrated.promoted) == (2, 2)
    async with first.sessions() as session:
        first_runs = (
            (
                await session.execute(
                    select(models.AcquisitionRun).where(
                        models.AcquisitionRun.workspace_id == "first"
                    )
                )
            )
            .scalars()
            .all()
        )
        second_runs = (
            (
                await session.execute(
                    select(models.AcquisitionRun).where(
                        models.AcquisitionRun.workspace_id == "second"
                    )
                )
            )
            .scalars()
            .all()
        )
    assert {run.connector_name for run in first_runs} == {"wiki", "drive"}
    assert second_runs == []

    await migrate_legacy_snapshots(second, blobs, page_size=1)
    second_run = await _legacy_run(second, "wiki")
    records = await second.list_acquisition_records(second_run.id)
    first_records = await first.list_acquisition_records((await _legacy_run(first, "wiki")).id)
    assert len(records) == 1
    assert len(first_records) == 1
    assert records[0].acquired_source is not None
    assert first_records[0].acquired_source is not None
    assert records[0].acquired_source.content_hash != first_records[0].acquired_source.content_hash


@pytest.mark.contract
async def test_a_real_promoted_inventory_is_not_shadowed_by_a_legacy_manifest(
    engine: AsyncEngine, data_dir: Path
) -> None:
    store = SqliteDocStore(engine)
    await store.ensure_workspace()
    blobs = BlobStore(engine, data_dir)
    row, retained = await _legacy_document(store, blobs, "already-covered", b"covered")
    created = await store.create_acquisition_run("real-run", "wiki", scope_fingerprint="all")
    now = utcnow()
    claimed = await store.claim_acquisition_run(
        created.id, "worker", now=now, expires_at=now + LEASE_DURATION
    )
    assert claimed is not None
    await store.append_acquisition_record(
        claimed.id,
        0,
        AcquisitionSource(
            ref=DocRef(source_id=row.source_id, uri=row.uri),
            version_token=row.version_token,
            media_type=row.media_type,
        ),
        lease_owner="worker",
        lease_generation=claimed.lease_generation,
        now=utcnow(),
    )
    await store.complete_acquisition_enumeration(
        claimed.id,
        None,
        lease_owner="worker",
        lease_generation=claimed.lease_generation,
        now=utcnow(),
    )
    await store.transition_acquisition_record(
        claimed.id,
        row.source_id,
        AcquisitionRecordState.DISCOVERED,
        AcquisitionRecordState.ACQUIRING,
        lease_owner="worker",
        lease_generation=claimed.lease_generation,
        now=utcnow(),
    )
    await store.transition_acquisition_record(
        claimed.id,
        row.source_id,
        AcquisitionRecordState.ACQUIRING,
        AcquisitionRecordState.ACQUIRED,
        lease_owner="worker",
        lease_generation=claimed.lease_generation,
        now=utcnow(),
        blob_ref=retained.hash,
        acquired_source=AcquiredSource(
            source_id=row.source_id,
            uri=row.uri,
            media_type=row.media_type,
            encoding="utf-8",
            text_content=False,
            content_hash=retained.hash,
            byte_length=len(b"covered"),
        ),
        fetched_version_token=row.version_token,
    )
    await store.complete_snapshot_acquisition(
        claimed.id,
        lease_owner="worker",
        lease_generation=claimed.lease_generation,
        now=utcnow(),
    )
    await store.promote_snapshot_and_commit_watermark(
        claimed.id,
        expected_scope_fingerprint="all",
        lease_owner="worker",
        lease_generation=claimed.lease_generation,
        now=utcnow(),
    )

    migrated = await migrate_legacy_snapshots(store, blobs)
    assert migrated.connectors == 0
    async with store.sessions() as session:
        legacy = (
            await session.execute(
                select(func.count())
                .select_from(models.AcquisitionRun)
                .where(models.AcquisitionRun.scope_inventory_complete.is_(False))
            )
        ).scalar_one()
    assert legacy == 0


@pytest.mark.contract
async def test_manifest_coverage_scans_matching_history_in_bounded_keyset_pages(
    engine: AsyncEngine, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from manicule.storage import legacy_snapshots  # noqa: PLC0415 - page-size test hook

    store = SqliteDocStore(engine)
    await store.ensure_workspace()
    blobs = BlobStore(engine, data_dir)
    row, retained = await _legacy_document(store, blobs, "long-history", b"covered")
    run_ids = ["history-a", "history-b", "history-c"]
    for run_id in run_ids:
        await _promote_current_document(store, row, retained, run_id=run_id)
    async with store.sessions.begin() as session:
        await session.execute(
            update(models.AcquisitionRun)
            .where(models.AcquisitionRun.id.in_(run_ids[:2]))
            .values(membership_hash="invalid")
        )
    assert await store.verify_snapshot_manifest(run_ids[-1])

    page_lengths: list[int] = []
    cache_lengths: list[int] = []
    original_page = getattr(  # noqa: B009 - avoids a private-use type-check diagnostic
        legacy_snapshots, "_matching_manifest_run_id_page"
    )
    original_remember = getattr(  # noqa: B009 - bounded-cache test hook
        legacy_snapshots, "_remember_manifest_verification"
    )

    async def tracked_page(
        store_arg: SqliteDocStore,
        row_arg: models.Document,
        *,
        after_run_id: str | None,
        page_size: int,
    ) -> list[str]:
        page = await original_page(
            store_arg,
            row_arg,
            after_run_id=after_run_id,
            page_size=page_size,
        )
        page_lengths.append(len(page))
        return page

    def tracked_remember(cache: OrderedDict[str, bool], run_id: str, verified: bool) -> None:
        original_remember(cache, run_id, verified)
        cache_lengths.append(len(cache))

    monkeypatch.setattr(legacy_snapshots, "MANIFEST_CANDIDATE_PAGE_SIZE", 2)
    monkeypatch.setattr(legacy_snapshots, "MANIFEST_VERIFICATION_CACHE_SIZE", 2)
    monkeypatch.setattr(legacy_snapshots, "_matching_manifest_run_id_page", tracked_page)
    monkeypatch.setattr(legacy_snapshots, "_remember_manifest_verification", tracked_remember)
    migrated = await migrate_legacy_snapshots(store, blobs)

    assert migrated.connectors == 0
    assert page_lengths == [2, 1]
    assert cache_lengths == [1, 2, 2]


@pytest.mark.contract
async def test_a_superseded_promoted_manifest_does_not_cover_legacy_ownership(
    engine: AsyncEngine, data_dir: Path
) -> None:
    store = SqliteDocStore(engine)
    await store.ensure_workspace()
    blobs = BlobStore(engine, data_dir)
    row, retained = await _legacy_document(store, blobs, "superseded", b"still needs ownership")
    run_id = "superseded-real-run"
    await _promote_current_document(store, row, retained, run_id=run_id)
    assert await store.verify_snapshot_manifest(run_id)

    async with store.sessions.begin() as session:
        await session.execute(
            update(models.AcquisitionRun)
            .where(models.AcquisitionRun.id == run_id)
            .values(superseded_at=utcnow())
        )

    # A superseded run can still have a valid evidence digest, but cleanup may delete it.
    assert await store.verify_snapshot_manifest(run_id)
    migrated = await migrate_legacy_snapshots(store, blobs)

    assert (migrated.connectors, migrated.retained, migrated.promoted) == (1, 1, 1)
    legacy = await _legacy_run(store, "wiki")
    assert legacy.id != run_id
    assert await store.verify_snapshot_manifest(legacy.id)


@pytest.mark.contract
async def test_an_old_promoted_version_does_not_cover_the_current_retained_original(
    engine: AsyncEngine, data_dir: Path
) -> None:
    store = SqliteDocStore(engine)
    await store.ensure_workspace()
    blobs = BlobStore(engine, data_dir)
    old_row, old_blob = await _legacy_document(store, blobs, "changed", b"old bytes")
    await _promote_current_document(store, old_row, old_blob, run_id="old-real-run")

    current_row, current_blob = await _legacy_document(
        store,
        blobs,
        "changed",
        b"new bytes",
        version_token="version-changed-new",  # noqa: S106 - source revision, not a credential
    )
    assert current_blob.hash != old_blob.hash
    migrated = await migrate_legacy_snapshots(store, blobs)

    assert (migrated.retained, migrated.promoted) == (1, 1)
    records = await store.list_acquisition_records((await _legacy_run(store, "wiki")).id)
    assert len(records) == 1
    assert records[0].blob_ref == current_blob.hash
    assert records[0].fetched_version_token == current_row.version_token


@pytest.mark.contract
async def test_retained_documents_attach_ownership_to_a_connector_tombstone(
    engine: AsyncEngine, data_dir: Path
) -> None:
    store = SqliteDocStore(engine)
    await store.ensure_workspace()
    blobs = BlobStore(engine, data_dir)
    await _legacy_document(store, blobs, "survivor", b"survives connector deletion")
    seed = await store.create_acquisition_run("pre-delete-run", "wiki")
    async with store.sessions.begin() as session:
        await session.execute(
            update(models.Connector)
            .where(models.Connector.id == seed.connector_id)
            .values(deleted_at=utcnow())
        )

    migrated = await migrate_legacy_snapshots(store, blobs)

    assert (migrated.retained, migrated.promoted) == (1, 1)
    run = await _legacy_run(store, "wiki")
    assert run.connector_id == seed.connector_id
    async with store.sessions() as session:
        tombstone = await session.get(models.Connector, seed.connector_id)
    assert tombstone is not None
    assert tombstone.deleted_at is not None


@pytest.mark.contract
async def test_a_leased_run_remains_discoverable_after_its_document_disappears(
    engine: AsyncEngine, data_dir: Path
) -> None:
    store = SqliteDocStore(engine)
    await store.ensure_workspace()
    blobs = BlobStore(engine, data_dir)
    row, _ = await _legacy_document(store, blobs, "leased", b"owned only after resume")
    run_id = legacy_snapshot_run_id(store.workspace_id, "wiki")
    run = await store.create_acquisition_run(
        run_id,
        "wiki",
        source_scope=LEGACY_SCOPE,
        scope_fingerprint=legacy_scope_fingerprint(store.workspace_id, "wiki"),
        scope_inventory_complete=False,
        promotion_policy=SnapshotPromotionPolicy.ALLOW_OMISSIONS,
    )
    now = utcnow()
    claimed = await store.claim_acquisition_run(
        run.id, "dead-process", now=now, expires_at=now + timedelta(minutes=5)
    )
    assert claimed is not None
    deferred = await migrate_legacy_snapshots(store, blobs)
    assert deferred.deferred == 1

    await store.delete_document(row.id)
    await store.release_acquisition_lease(
        run.id, "dead-process", claimed.lease_generation, now=utcnow()
    )
    resumed = await migrate_legacy_snapshots(store, blobs)

    assert (resumed.connectors, resumed.resumed, resumed.missing, resumed.promoted) == (1, 1, 0, 1)
    settled = await store.get_acquisition_run(run.id)
    assert settled is not None
    assert settled.state is AcquisitionRunState.SETTLED


async def test_writer_runtime_refuses_to_serve_while_legacy_ownership_is_deferred(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from manicule.app.runtime import Runtime  # noqa: PLC0415 - exercises startup boundary
    from manicule.storage import legacy_snapshots  # noqa: PLC0415

    async def deferred(*args: object, **kwargs: object) -> LegacySnapshotMigration:
        del args, kwargs
        return LegacySnapshotMigration(deferred=1)

    monkeypatch.setattr(legacy_snapshots, "migrate_legacy_snapshots", deferred)
    async with Runtime.open(data_dir=data_dir) as runtime:
        with pytest.raises(RuntimeError, match="writer startup refused"):
            await runtime.documents()


@pytest.mark.contract
async def test_slow_validation_is_kept_alive_by_an_independent_lease_heartbeat(
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from manicule.storage import legacy_snapshots  # noqa: PLC0415

    store = SqliteDocStore(engine)
    await store.ensure_workspace()
    blobs = BlobStore(engine, data_dir)
    await _legacy_document(store, blobs, "slow", b"slow but still owned")

    # **Slow the read where the read actually happens.** This used to patch `Path.read_bytes`,
    # which was right when it was written and stopped being right in #219 (c08122d8), where
    # `BlobStore` moved to `os.open`/`os.fdopen` descriptors. Nothing failed: the patch simply
    # stopped matching, no validation was ever slowed, the heartbeat was never exercised, and
    # the test passed for nothing across every release since. Sabotaging the heartbeat to sleep
    # ten times the lease instead of renewing still left it green, in 0.4s.
    #
    # `_read_blob` is the seam because it *is* the blocking filesystem read — `get` hands it to
    # `asyncio.to_thread` precisely to keep that latency off the event loop, which is the same
    # property this test depends on: the delay must occupy a worker thread and leave the loop
    # free for the heartbeat. A `time.sleep` on the loop would prove nothing except that a
    # blocked loop cannot renew a lease.
    #
    # It patches every call rather than matching one path. Path matching is what rotted the
    # first time, and it was fragile even before #219: `get` resolves through
    # `_authoritative_path`, which returns an evidence pin when one exists and the sharded path
    # otherwise, so `path_for` was only ever one of two right answers. Exactly one read happens
    # here, verified below, so "every call" and "the validation read" are the same set.
    # The ratios carry the meaning; the absolute scale only has to survive a loaded CI runner.
    # Validation outlasts one whole lease, and the heartbeat renews at LEASE_DURATION/3, so a
    # correct implementation renews twice during the read and a broken one loses the lease.
    #
    # The scale was 60 ms with a 120 ms read. That is not survivable on CI's two-core runners:
    # a heartbeat *due* in 20 ms is not necessarily *scheduled* within the 60 ms before the
    # lease expires, and a tenth of a second of scheduler jitter is ordinary there. Ten times
    # slower keeps every ratio, buys 400 ms of slack per renewal, and costs about a second.
    lease = timedelta(milliseconds=600)
    read_seconds = 2 * lease.total_seconds()

    reads: list[Path] = []
    # Reaching a private static method is the point: it is the seam where the blocking read
    # happens, and a public wrapper would not be one.
    original_read_blob = BlobStore._read_blob  # pyright: ignore[reportPrivateUsage]

    def slow_read_blob(path: Path, compression: str) -> bytes:
        reads.append(path)
        time.sleep(read_seconds)
        return original_read_blob(path, compression)

    monkeypatch.setattr(legacy_snapshots, "LEASE_DURATION", lease)
    monkeypatch.setattr(BlobStore, "_read_blob", staticmethod(slow_read_blob))
    migrated = await migrate_legacy_snapshots(store, blobs)

    # **The guard against this test going quiet again.** Everything below passes whether or not
    # validation was slow, so without this the suite cannot tell "the heartbeat held the lease"
    # from "no read was ever delayed". That distinction is the entire test.
    assert reads, (
        "no blob read was intercepted, so validation was never slowed and the lease heartbeat "
        "was not exercised — this test is asserting nothing. BlobStore's read path has moved "
        "again; re-point the patch at it rather than deleting this assertion."
    )
    assert (migrated.retained, migrated.promoted, migrated.deferred) == (1, 1, 0)
    assert await store.verify_snapshot_manifest((await _legacy_run(store, "wiki")).id)


@pytest.mark.contract
async def test_pre_journal_schema_upgrades_then_gains_legacy_snapshot_ownership(
    data_dir: Path,
) -> None:
    engine = create_engine(data_dir)
    try:
        await upgrade(engine, revision="6e31b7d592ac")
        old_store = SqliteDocStore(engine)
        await old_store.ensure_workspace()
        blobs = BlobStore(engine, data_dir)
        await _legacy_document(old_store, blobs, "pre-journal", b"survives upgrade")

        await upgrade(engine)
        upgraded = SqliteDocStore(engine)
        migrated = await migrate_legacy_snapshots(upgraded, blobs, page_size=1)
        assert (migrated.retained, migrated.promoted) == (1, 1)
        run = await _legacy_run(upgraded, "wiki")
        assert run.scope_inventory_complete is False
        assert await upgraded.verify_snapshot_manifest(run.id)
    finally:
        await engine.dispose()
