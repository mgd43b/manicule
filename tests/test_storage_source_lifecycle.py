"""Distinct retention boundaries for authoritative source and derived state."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast

import pytest
from sqlalchemy import select

from manicule.core.acquisition import (
    AcquisitionRecordState,
    AcquisitionRunState,
    SnapshotCompleteness,
    SnapshotItemOutcome,
)
from manicule.core.content import DocumentStatus, JsonValue
from manicule.core.rebuild import RebuildState
from manicule.core.source_lifecycle import LifecycleRefusalError
from manicule.storage import models
from manicule.storage.blobs import BlobStore, StoredBlob
from manicule.storage.docstore import SqliteDocStore
from manicule.storage.engine import session_factory
from tests.storage_helpers import make_chunk, make_document

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine

NOW = datetime(2026, 8, 15, 12, tzinfo=UTC)


async def _snapshot(
    engine: AsyncEngine,
    data_dir: Path,
    *,
    run_id: str,
    body: bytes,
    source_id: str,
    state: AcquisitionRunState = AcquisitionRunState.SETTLED,
    promoted_at: datetime = NOW,
    candidate_watermark: dict[str, str] | None = None,
    watermark_committed_at: datetime | None = None,
) -> str:
    blob = await BlobStore(engine, data_dir).put(body, "text/plain")
    assert isinstance(blob, StoredBlob)
    sessions = session_factory(engine)
    async with sessions.begin() as session:
        connector = await session.get(models.Connector, "wiki")
        if connector is None:
            session.add(
                models.Connector(
                    id="wiki",
                    workspace_id="default",
                    name="wiki",
                    type="synthetic",
                    config={},
                )
            )
            await session.flush()
        session.add(
            models.AcquisitionRun(
                id=run_id,
                workspace_id="default",
                connector_id="wiki",
                connector_name="wiki",
                source_scope="all",
                scope_fingerprint="scope-v1",
                state=state,
                enumeration_completed_at=NOW,
                acquisition_completed_at=NOW,
                promoted_at=promoted_at,
                candidate_watermark=candidate_watermark,
                watermark_committed_at=watermark_committed_at,
                membership_hash=f"membership-{run_id}",
                completeness=SnapshotCompleteness.COMPLETE,
                discovered_count=1,
                acquired_count=1,
            )
        )
        await session.flush()
        session.add(
            models.AcquisitionRecord(
                id=f"record-{run_id}",
                run_id=run_id,
                workspace_id="default",
                connector_id="wiki",
                sequence=0,
                source_id=source_id,
                source_record={"source_id": source_id},
                state=(
                    AcquisitionRecordState.SETTLED
                    if state is AcquisitionRunState.SETTLED
                    else AcquisitionRecordState.RETRY
                ),
                snapshot_outcome=SnapshotItemOutcome.RETAINED,
                blob_ref=blob.hash,
                acquired_source={"source_id": source_id},
            )
        )
    return blob.hash


@pytest.mark.asyncio
async def test_derived_reset_preserves_snapshot_versions_and_blob_gc_roots(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    blob_ref = await _snapshot(
        engine, data_dir, run_id="snapshot-reset", body=b"authoritative", source_id="page-1"
    )
    document = make_document(source="wiki", source_id="page-1").model_copy(
        update={"original_ref": blob_ref}
    )
    await store.upsert_document(document)
    await store.replace_chunks(document.id, [make_chunk(document, 0, "derived text")])
    sessions = session_factory(engine)
    async with sessions.begin() as session:
        session.add(
            models.DocumentVersion(
                id="version-1",
                document_id=document.id,
                version=1,
                content_hash=blob_ref,
                original_ref=blob_ref,
                created_at=NOW - timedelta(days=60),
            )
        )
        session.add(
            models.Collection(
                id="reset-collection",
                workspace_id="default",
                name="Reset collection",
                description=None,
                auto_rules={},
            )
        )
        session.add(
            models.CollectionDocument(
                collection_id="reset-collection",
                document_id=document.id,
            )
        )
        session.add(
            models.DerivedGeneration(
                id="published-before-reset",
                workspace_id="default",
                snapshot_run_id="snapshot-reset",
                snapshot_membership_hash="membership-snapshot-reset",
                expected_item_count=1,
                target_digest="same-target",
                publication_identity_digest="published-identity",
                target={"parser": "same-target"},
                state=RebuildState.PUBLISHED,
                vector_publication_id="published-before-reset",
                published_at=NOW,
            )
        )
        session.add(
            models.AcquisitionRun(
                id="unfinished-before-reset",
                workspace_id="default",
                connector_id="wiki",
                connector_name="wiki",
                source_scope="all",
                scope_fingerprint="scope-unfinished",
                state=AcquisitionRunState.INDEXING,
                lease_owner="stale-worker",
                lease_generation=3,
                lease_expires_at=NOW + timedelta(minutes=5),
            )
        )
        session.add(
            models.DerivedGeneration(
                id="retry-after-reset",
                workspace_id="default",
                snapshot_run_id="snapshot-reset",
                snapshot_membership_hash="membership-snapshot-reset",
                expected_item_count=1,
                target_digest="retry-target",
                publication_identity_digest="retry-identity",
                target={"parser": "retry-target"},
                state=RebuildState.BUILDING,
            )
        )

    plan = await store.plan_reset_derived()
    outcome = await store.reset_derived()

    assert plan.snapshot_items == 1
    assert outcome.removed_items == 1
    assert await store.get_document(document.id) is None, "reset documents are no longer live"
    assert await store.count_chunks() == 0
    async with sessions() as session:
        retained = await session.get(models.Document, document.id)
        assert retained is not None
        assert retained.original_ref == blob_ref
        assert retained.status is DocumentStatus.DELETED
        assert retained.deleted_at is not None
        assert await session.get(models.DocumentVersion, "version-1") is not None
        assert await session.get(models.AcquisitionRun, "snapshot-reset") is not None
        unfinished = await session.get(models.AcquisitionRun, "unfinished-before-reset")
        assert unfinished is not None
        assert unfinished.state is AcquisitionRunState.SETTLED
        assert unfinished.superseded_at is not None
        assert unfinished.lease_owner is None
        assert unfinished.lease_generation == 4
        retired = await session.get(models.DerivedGeneration, "published-before-reset")
        assert retired is not None
        assert retired.state is RebuildState.CANCELED
        interrupted = await session.get(models.DerivedGeneration, "retry-after-reset")
        assert interrupted is not None
        assert interrupted.state is RebuildState.CANCELED
        assert interrupted.lease_generation == 1
        assert (
            await session.get(
                models.CollectionDocument,
                {"collection_id": "reset-collection", "document_id": document.id},
            )
            is None
        )
        assert await session.get(models.Collection, "reset-collection") is not None
    assert await BlobStore(engine, data_dir).collect_garbage() == []


@pytest.mark.asyncio
async def test_generation_cleanup_preserves_live_latest_and_resumable_generations(
    store: SqliteDocStore,
    engine: AsyncEngine,
) -> None:
    sessions = session_factory(engine)
    async with sessions.begin() as session:
        # The FK is the safety property under test; a settled promoted snapshot is enough.
        session.add(
            models.Connector(
                id="wiki", workspace_id="default", name="wiki", type="synthetic", config={}
            )
        )
        await session.flush()
        session.add(
            models.AcquisitionRun(
                id="snapshot-generations",
                workspace_id="default",
                connector_id="wiki",
                connector_name="wiki",
                source_scope="all",
                scope_fingerprint="scope-v1",
                state=AcquisitionRunState.SETTLED,
                promoted_at=NOW,
                acquisition_completed_at=NOW,
                membership_hash="members",
            )
        )
        await session.flush()
        for generation_id, state, publication, published_at in (
            ("failed", RebuildState.FAILED, "failed-vectors", None),
            ("old", RebuildState.PUBLISHED, "old-vectors", NOW - timedelta(days=1)),
            ("current", RebuildState.PUBLISHED, "current-vectors", NOW),
            ("retry", RebuildState.BUILDING, "retry-vectors", None),
            ("certified", RebuildState.VALIDATING, "certified-vectors", None),
        ):
            session.add(
                models.DerivedGeneration(
                    id=generation_id,
                    workspace_id="default",
                    snapshot_run_id="snapshot-generations",
                    snapshot_membership_hash="members",
                    expected_item_count=0,
                    target_digest=generation_id,
                    publication_identity_digest=generation_id,
                    target={},
                    state=state,
                    vector_publication_id=publication,
                    published_at=published_at,
                )
            )
            session.add(
                models.DerivedGenerationItem(
                    generation_id=generation_id,
                    sequence=0,
                    payload_digest=generation_id,
                    document_id=generation_id,
                    payload={},
                    temporary_bytes=10,
                )
            )
    plan = await store.plan_derived_generation_cleanup()
    removed = 0
    released = 0
    async for page in store.obsolete_generation_publications():
        for generation in page:
            count, bytes_ = await store.cleanup_obsolete_generation(generation.generation_id)
            removed += count
            released += bytes_

    assert plan.eligible_items == 2
    assert plan.eligible_bytes == 20
    assert removed == 2
    assert released == 20
    async with sessions() as session:
        remaining = set((await session.execute(select(models.DerivedGeneration.id))).scalars())
    assert remaining == {"current", "retry", "certified"}
    assert [page async for page in store.obsolete_generation_publications()] == []


@pytest.mark.asyncio
async def test_generation_cleanup_is_keyset_bounded_and_crash_retryable(
    store: SqliteDocStore,
    engine: AsyncEngine,
) -> None:
    sessions = session_factory(engine)
    async with sessions.begin() as session:
        session.add(
            models.Connector(
                id="wiki", workspace_id="default", name="wiki", type="synthetic", config={}
            )
        )
        await session.flush()
        session.add(
            models.AcquisitionRun(
                id="snapshot-many-generations",
                workspace_id="default",
                connector_id="wiki",
                connector_name="wiki",
                source_scope="all",
                scope_fingerprint="scope-v1",
                state=AcquisitionRunState.SETTLED,
                promoted_at=NOW,
                acquisition_completed_at=NOW,
                membership_hash="members",
            )
        )
        await session.flush()
        session.add_all(
            [
                models.DerivedGeneration(
                    id=f"obsolete-{index:04d}",
                    workspace_id="default",
                    snapshot_run_id="snapshot-many-generations",
                    snapshot_membership_hash="members",
                    expected_item_count=0,
                    target_digest=f"target-{index}",
                    publication_identity_digest=f"identity-{index}",
                    target={},
                    state=RebuildState.CANCELED,
                    vector_publication_id=f"vectors-{index}",
                    expected_vector_table=f"reembed-table-{index}",
                )
                for index in range(257)
            ]
        )

    pages = [page async for page in store.obsolete_generation_publications(page_size=17)]
    assert sum(len(page) for page in pages) == 257
    assert max(map(len, pages)) == 17
    assert [item.generation_id for page in pages for item in page] == sorted(
        item.generation_id for page in pages for item in page
    )
    assert pages[0][0].expected_vector_table == "reembed-table-0"
    with pytest.raises(ValueError, match=r"1\.\.128"):
        await anext(store.obsolete_generation_publications(page_size=129))

    # A crash after the idempotent physical delete but before the relational CAS leaves the
    # durable cleanup unit selectable on retry.
    first = pages[0][0]
    retried = [page async for page in store.obsolete_generation_publications(page_size=1)]
    assert retried[0][0] == first
    assert await store.cleanup_obsolete_generation(first.generation_id) == (1, 0)
    after = [page async for page in store.obsolete_generation_publications(page_size=1)]
    assert after[0][0].generation_id != first.generation_id


@pytest.mark.asyncio
async def test_history_release_is_policy_bounded_and_distinct_blob_safe(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    old_blob = await BlobStore(engine, data_dir).put(b"old-only", "text/plain")
    shared_blob = await BlobStore(engine, data_dir).put(b"shared", "text/plain")
    assert isinstance(old_blob, StoredBlob)
    assert isinstance(shared_blob, StoredBlob)
    document = make_document(source="wiki", source_id="history").model_copy(
        update={"original_ref": shared_blob.hash}
    )
    await store.upsert_document(document)
    sessions = session_factory(engine)
    async with sessions.begin() as session:
        session.add_all(
            [
                models.DocumentVersion(
                    id="old-only",
                    document_id=document.id,
                    version=1,
                    content_hash=old_blob.hash,
                    original_ref=old_blob.hash,
                    created_at=NOW - timedelta(days=90),
                ),
                models.DocumentVersion(
                    id="old-shared",
                    document_id=document.id,
                    version=2,
                    content_hash=shared_blob.hash,
                    original_ref=shared_blob.hash,
                    created_at=NOW - timedelta(days=60),
                ),
                models.DocumentVersion(
                    id="recent",
                    document_id=document.id,
                    version=3,
                    content_hash=old_blob.hash,
                    original_ref=old_blob.hash,
                    created_at=NOW - timedelta(days=1),
                ),
            ]
        )
    plan = await store.plan_source_history_release(NOW - timedelta(days=30))
    outcome = await store.release_source_history(NOW - timedelta(days=30))

    assert plan.eligible_items == 2
    # The recent version still pins old_blob and the document pins shared_blob.
    assert plan.eligible_bytes == 0
    assert outcome.removed_items == 2
    async with sessions() as session:
        old_only = await session.get(models.DocumentVersion, "old-only")
        old_shared = await session.get(models.DocumentVersion, "old-shared")
        assert old_only is not None
        assert old_shared is not None
        assert old_only.original_ref is None
        assert old_shared.original_ref is None
        assert old_only.bytes_released_at is not None
        assert old_shared.bytes_released_at is not None
        assert await session.get(models.DocumentVersion, "recent") is not None
    repeated = await store.plan_source_history_release(NOW - timedelta(days=30))
    assert repeated.eligible_items == 0
    assert repeated.eligible_bytes == 0


@pytest.mark.asyncio
async def test_snapshot_delete_requires_current_dry_run_and_preserves_retry_roots(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    unique_ref = await _snapshot(
        engine,
        data_dir,
        run_id="delete-me",
        body=b"delete-only",
        source_id="delete-page",
        candidate_watermark={"value": "snapshot-current"},
        watermark_committed_at=NOW,
    )
    published = make_document(source="wiki", source_id="delete-page").model_copy(
        update={"original_ref": unique_ref}
    )
    await store.upsert_document(published)
    sessions = session_factory(engine)
    async with sessions.begin() as session:
        connector = await session.get(models.Connector, "wiki")
        assert connector is not None
        connector.watermark = {"value": "snapshot-current"}
        connector.watermark_scope_fingerprint = "scope-v1"
    await _snapshot(
        engine,
        data_dir,
        run_id="retry-root",
        body=b"retry-body",
        source_id="retry-page",
        state=AcquisitionRunState.INDEXING,
        promoted_at=NOW - timedelta(days=1),
    )

    plan = await store.plan_snapshot_deletion("delete-me")
    assert plan.snapshot_items == 1
    assert plan.unrecoverable_items == 1
    assert plan.unrecoverable_bytes == len(b"delete-only")
    assert plan.confirmation is not None

    async with sessions.begin() as session:
        session.add(
            models.DocumentVersion(
                id="late-history-root",
                document_id=published.id,
                version=1,
                content_hash=unique_ref,
                original_ref=unique_ref,
                created_at=NOW,
            )
        )
    with pytest.raises(LifecycleRefusalError, match="confirmation token"):
        await store.delete_snapshot("delete-me", confirmation=plan.confirmation)
    async with sessions.begin() as session:
        late = await session.get(models.DocumentVersion, "late-history-root")
        assert late is not None
        await session.delete(late)
    plan = await store.plan_snapshot_deletion("delete-me")
    assert plan.confirmation is not None

    with pytest.raises(LifecycleRefusalError, match="confirmation token"):
        await store.delete_snapshot("delete-me", confirmation="stale-token")
    with pytest.raises(LifecycleRefusalError, match="resumable work"):
        await store.plan_snapshot_deletion("retry-root")

    outcome = await store.delete_snapshot("delete-me", confirmation=plan.confirmation)
    assert outcome.snapshot_items == 1
    assert outcome.released_bytes == len(b"delete-only")
    assert unique_ref in await BlobStore(engine, data_dir).collect_garbage()
    retained = await store.get_document(published.id)
    assert retained is not None
    assert retained.original_ref is None
    sessions = session_factory(engine)
    async with sessions() as session:
        retained_row = await session.get(models.Document, published.id)
        assert retained_row is not None
        assert retained_row.original_omitted_reason == "source snapshot deleted by operator"
    sessions = session_factory(engine)
    async with sessions() as session:
        assert await session.get(models.AcquisitionRun, "retry-root") is not None
        connector = await session.get(models.Connector, "wiki")
        assert connector is not None
        assert connector.watermark is None
        assert connector.watermark_scope_fingerprint is None


@pytest.mark.asyncio
async def test_snapshot_delete_preserves_a_connector_watermark_that_advanced_after_dry_run(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    await _snapshot(
        engine,
        data_dir,
        run_id="delete-stale-watermark",
        body=b"old snapshot",
        source_id="old-page",
        candidate_watermark={"value": "old"},
        watermark_committed_at=NOW,
    )
    sessions = session_factory(engine)
    async with sessions.begin() as session:
        connector = await session.get(models.Connector, "wiki")
        assert connector is not None
        connector.watermark = {"value": "old"}
        connector.watermark_scope_fingerprint = "scope-v1"
        connector.last_synced_at = NOW
    plan = await store.plan_snapshot_deletion("delete-stale-watermark")
    assert plan.confirmation is not None

    async with sessions.begin() as session:
        connector = await session.get(models.Connector, "wiki")
        assert connector is not None
        connector.watermark = {"value": "new"}
        connector.watermark_scope_fingerprint = "scope-v2"
        connector.last_synced_at = NOW + timedelta(minutes=1)

    await store.delete_snapshot("delete-stale-watermark", confirmation=plan.confirmation)

    async with sessions() as session:
        connector = await session.get(models.Connector, "wiki")
        assert connector is not None
        assert connector.watermark == {"value": "new"}
        assert connector.watermark_scope_fingerprint == "scope-v2"
        assert connector.last_synced_at == NOW + timedelta(minutes=1)


@pytest.mark.asyncio
async def test_snapshot_delete_clears_the_matching_legacy_unscoped_watermark(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    watermark = {"value": "legacy-position"}
    await _snapshot(
        engine,
        data_dir,
        run_id="delete-legacy-watermark",
        body=b"synthetic legacy snapshot",
        source_id="legacy-page",
        candidate_watermark=watermark,
        watermark_committed_at=NOW,
    )
    sessions = session_factory(engine)
    async with sessions.begin() as session:
        connector = await session.get(models.Connector, "wiki")
        assert connector is not None
        connector.watermark = cast("JsonValue", watermark)
        connector.watermark_scope_fingerprint = None
        connector.last_synced_at = NOW
    plan = await store.plan_snapshot_deletion("delete-legacy-watermark")
    assert plan.confirmation is not None

    await store.delete_snapshot("delete-legacy-watermark", confirmation=plan.confirmation)

    async with sessions() as session:
        connector = await session.get(models.Connector, "wiki")
        assert connector is not None
        assert connector.watermark is None
        assert connector.watermark_scope_fingerprint is None
        assert connector.last_synced_at is None


@pytest.mark.asyncio
async def test_snapshot_dry_run_counts_shared_blob_once_and_reveals_no_metadata(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    await _snapshot(engine, data_dir, run_id="shared-a", body=b"same bytes", source_id="private-a")
    await _snapshot(engine, data_dir, run_id="shared-b", body=b"same bytes", source_id="private-b")

    with pytest.raises(LifecycleRefusalError, match="current promoted snapshot"):
        await store.plan_snapshot_deletion("shared-a")
    plan = await store.plan_snapshot_deletion("shared-b")

    assert plan.unrecoverable_items == 0
    assert plan.unrecoverable_bytes == 0
    dumped = plan.model_dump_json()
    assert "private-b" not in dumped
    assert "example.test" not in dumped


@pytest.mark.asyncio
async def test_snapshot_impact_counts_items_but_shared_bytes_only_once(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    blob_ref = await _snapshot(
        engine, data_dir, run_id="same-run", body=b"one retained body", source_id="page-a"
    )
    sessions = session_factory(engine)
    async with sessions.begin() as session:
        session.add(
            models.AcquisitionRecord(
                id="record-same-run-b",
                run_id="same-run",
                workspace_id="default",
                connector_id="wiki",
                sequence=1,
                source_id="page-b",
                source_record={"source_id": "page-b"},
                state=AcquisitionRecordState.SETTLED,
                snapshot_outcome=SnapshotItemOutcome.REUSED,
                blob_ref=blob_ref,
                acquired_source={"source_id": "page-b"},
            )
        )
        run = await session.get(models.AcquisitionRun, "same-run")
        assert run is not None
        run.discovered_count = 2

    plan = await store.plan_snapshot_deletion("same-run")

    assert plan.snapshot_items == 2
    assert plan.unrecoverable_items == 2
    assert plan.unrecoverable_bytes == len(b"one retained body")
