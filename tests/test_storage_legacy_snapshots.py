"""Legacy published originals become honest, source-free snapshot manifests."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import func, select

from manicule.core.acquisition import (
    AcquiredSource,
    AcquisitionFailureCode,
    AcquisitionRecordState,
    AcquisitionRun,
    AcquisitionRunState,
    AcquisitionSource,
    SnapshotCompleteness,
    SnapshotItemOutcome,
)
from manicule.core.content import DocumentStatus
from manicule.core.sources import DocRef
from manicule.storage import models
from manicule.storage.blobs import BlobStore, StoredBlob
from manicule.storage.docstore import SqliteDocStore
from manicule.storage.engine import create_engine
from manicule.storage.legacy_snapshots import (
    LEASE_DURATION,
    LEGACY_SCOPE_PREFIX,
    migrate_legacy_snapshots,
)
from manicule.storage.migrator import upgrade
from manicule.storage.types import utcnow
from tests.storage_helpers import make_document

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine


async def _legacy_document(
    store: SqliteDocStore,
    blobs: BlobStore,
    source_id: str,
    body: bytes,
    *,
    source: str = "wiki",
) -> tuple[models.Document, StoredBlob]:
    retained = await blobs.put(body, "text/markdown")
    assert isinstance(retained, StoredBlob)
    document = make_document(
        source=source, source_id=source_id, body=body, workspace_id=store.workspace_id
    ).model_copy(
        update={
            "original_ref": retained.hash,
            "version_token": f"version-{source_id}",
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
            await session.execute(
                select(models.Document).where(models.Document.id == document.id)
            )
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


@pytest.mark.contract
async def test_retained_missing_and_corrupt_originals_migrate_without_changing_publication(
    engine: AsyncEngine, data_dir: Path
) -> None:
    store = SqliteDocStore(engine)
    await store.ensure_workspace()
    blobs = BlobStore(engine, data_dir)
    retained_row, retained = await _legacy_document(store, blobs, "a-retained", b"alpha")
    missing_row, missing = await _legacy_document(store, blobs, "b-missing", b"beta")
    corrupt_row, corrupt = await _legacy_document(store, blobs, "c-corrupt", b"gamma")
    blobs.path_for(missing.hash).unlink()
    blobs.path_for(corrupt.hash).write_bytes(b"not gamma")

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
            (
                await session.execute(
                    select(models.Document).where(models.Document.id.in_(before))
                )
            )
            .scalars()
            .all()
        )
    assert {
        row.id: (row.publication_id, row.status, row.content_hash) for row in rows
    } == before
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
            await session.execute(
                select(models.AcquisitionRun).where(
                    models.AcquisitionRun.workspace_id == "first"
                )
            )
        ).scalars().all()
        second_runs = (
            await session.execute(
                select(models.AcquisitionRun).where(
                    models.AcquisitionRun.workspace_id == "second"
                )
            )
        ).scalars().all()
    assert {run.connector_name for run in first_runs} == {"wiki", "drive"}
    assert second_runs == []

    await migrate_legacy_snapshots(second, blobs, page_size=1)
    second_run = await _legacy_run(second, "wiki")
    records = await second.list_acquisition_records(second_run.id)
    assert len(records) == 1
    assert records[0].acquired_source is not None
    assert records[0].acquired_source.content_hash != (
        await first.list_acquisition_records((await _legacy_run(first, "wiki")).id)
    )[0].acquired_source.content_hash  # type: ignore[union-attr]


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
