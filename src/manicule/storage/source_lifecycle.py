"""Reference-safe source and derived lifecycle operations over SQLite.

Every plan is an aggregate.  No source id, URI, blob digest, or document title crosses this
boundary.  Mutations are single database transactions, which makes cancellation before commit
a no-op and a retry after commit idempotent.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, cast

from sqlalchemy import delete, exists, func, null, or_, select, union, update

from manicule.core.acquisition import AcquisitionRunState
from manicule.core.content import DocumentStatus
from manicule.core.rebuild import RebuildState, vector_publication_id
from manicule.core.source_lifecycle import (
    LifecycleOperation,
    LifecycleOutcome,
    LifecyclePlan,
    LifecycleRefusalError,
)
from manicule.storage import models
from manicule.storage.scoped import WorkspaceScoped
from manicule.storage.types import utcnow

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy import CursorResult
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql.elements import ColumnElement
    from sqlalchemy.sql.selectable import Select, Subquery

GENERATION_CLEANUP_PAGE = 128
RESET_PUBLICATION = "reset-derived"


@dataclass(frozen=True, slots=True)
class GenerationPublication:
    """One retryable physical cleanup unit with its immutable #187 binding."""

    generation_id: str
    vector_publication_id: str | None
    replay_target_publication_id: str | None
    expected_vector_table: str | None


@dataclass(frozen=True, slots=True)
class ResetPreparation:
    """Aggregate durable cleanup binding committed by a confirmed workspace reset."""

    documents: int
    chunks: int
    memberships: int
    generations_terminalized: int
    snapshots: int
    vector_namespace: str
    vector_table: str | None


@dataclass(frozen=True, slots=True)
class BoundVectorTombstone:
    """One exact retryable physical row deletion, scoped to its immutable layout."""

    vector_id: str
    vector_namespace: str
    vector_table: str | None


class SourceLifecycleMixin(WorkspaceScoped):
    """Workspace-scoped retention boundaries mixed into :class:`SqliteDocStore`."""

    async def plan_reset_derived(self) -> LifecyclePlan:
        async with self._sessions() as session:
            documents = await self._workspace_document_count(session)
            chunks = await self._workspace_chunk_count(session)
        return LifecyclePlan(
            operation=LifecycleOperation.RESET_DERIVED,
            eligible_items=documents,
            protected_items=documents,
            snapshot_items=await self._snapshot_item_count(),
            # `eligible_bytes` is intentionally zero: source bytes are outside this operation.
        ).model_copy(update={"eligible_items": chunks})

    async def reset_binding(self) -> tuple[str, str | None]:
        """The durable physical layout this workspace must finish cleaning."""
        async with self._sessions() as session:
            state = (
                await session.execute(
                    select(
                        models.IndexState.vector_namespace,
                        models.IndexState.vector_table,
                    ).where(models.IndexState.workspace_id == self._workspace_id)
                )
            ).one_or_none()
        if state is None:
            return "workspace", None
        return str(state.vector_namespace), state.vector_table

    async def reset_vector_tombstones(
        self, *, limit: int = GENERATION_CLEANUP_PAGE
    ) -> tuple[BoundVectorTombstone, ...]:
        """Read one bounded workspace-owned cleanup page without exposing document identity."""
        if limit < 1 or limit > GENERATION_CLEANUP_PAGE:
            raise ValueError(f"reset tombstone page size must be 1..{GENERATION_CLEANUP_PAGE}")
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    select(
                        models.VectorTombstone.chunk_id,
                        models.VectorTombstone.vector_namespace,
                        models.VectorTombstone.vector_table,
                    )
                    .where(models.VectorTombstone.workspace_id == self._workspace_id)
                    .order_by(
                        models.VectorTombstone.deleted_at,
                        models.VectorTombstone.chunk_id,
                    )
                    .limit(limit)
                )
            ).all()
        return tuple(
            BoundVectorTombstone(str(row.chunk_id), str(row.vector_namespace), row.vector_table)
            for row in rows
        )

    async def finish_reset_identity(self) -> bool:
        """Clear only this workspace's identity after every physical cleanup has succeeded."""
        async with self._sessions.begin() as session:
            result = cast(
                "CursorResult[object]",
                await session.execute(
                    delete(models.IndexState).where(
                        models.IndexState.workspace_id == self._workspace_id
                    )
                ),
            )
        return bool(result.rowcount)

    async def retire_reset_pointer(self) -> None:
        """Make terminal shadow generations non-live while retaining retry identity metadata."""
        async with self._sessions.begin() as session:
            await session.execute(
                update(models.IndexState)
                .where(models.IndexState.workspace_id == self._workspace_id)
                .values(vector_table=None, vector_inventory_digest=None, updated_at=utcnow())
            )

    async def other_legacy_vector_consumers(self) -> int:
        """How many other workspaces still depend on the upgraded shared vector root."""
        async with self._sessions() as session:
            return int(
                await session.scalar(
                    select(func.count(models.IndexState.workspace_id)).where(
                        models.IndexState.workspace_id != self._workspace_id,
                        models.IndexState.vector_namespace == "legacy",
                    )
                )
                or 0
            )

    async def clear_unowned_legacy_tombstones(self) -> int:
        """Forget upgrade-era tombstones only after the last shared root is removed.

        Rows that could not be attributed during migration deliberately remain unclaimable by
        any workspace.  Deleting the shared legacy store is the first point at which their
        cleanup is certain, and therefore the only safe point at which to retire the ledger.
        """
        async with self._sessions.begin() as session:
            result = cast(
                "CursorResult[object]",
                await session.execute(
                    delete(models.VectorTombstone).where(
                        models.VectorTombstone.workspace_id.is_(None)
                    )
                ),
            )
        return int(result.rowcount or 0)

    async def prepare_reset_derived(self) -> ResetPreparation:
        """Retire all workspace derived visibility and fence every resumable generation."""
        from manicule.storage.acquisition import (  # noqa: PLC0415
            rebuild_acquisition_blob_backlog,
        )

        async with self._sessions.begin() as session:
            epoch_result = cast(
                "CursorResult[object]",
                await session.execute(
                    update(models.Workspace)
                    .where(models.Workspace.id == self._workspace_id)
                    .values(derived_reset_epoch=models.Workspace.derived_reset_epoch + 1)
                ),
            )
            if not epoch_result.rowcount:  # pragma: no cover - stores ensure their workspace
                raise RuntimeError("reset workspace does not exist")
            document_ids = select(models.Document.id).where(
                models.Document.workspace_id == self._workspace_id
            )
            documents = await self._workspace_document_count(session)
            chunks = await self._workspace_chunk_count(session)
            memberships = int(
                await session.scalar(
                    select(func.count())
                    .select_from(models.CollectionDocument)
                    .where(models.CollectionDocument.document_id.in_(document_ids))
                )
                or 0
            )
            retired_at = utcnow()
            acquisition_result = cast(
                "CursorResult[object]",
                await session.execute(
                    update(models.AcquisitionRun)
                    .where(
                        models.AcquisitionRun.workspace_id == self._workspace_id,
                        models.AcquisitionRun.state != AcquisitionRunState.SETTLED,
                    )
                    .values(
                        state=AcquisitionRunState.SETTLED,
                        acquired_blob_bytes=0,
                        superseded_at=retired_at,
                        lease_owner=None,
                        lease_generation=models.AcquisitionRun.lease_generation + 1,
                        lease_expires_at=None,
                        updated_at=retired_at,
                    )
                ),
            )
            await rebuild_acquisition_blob_backlog(session)
            await session.execute(delete(models.Chunk).where(self._workspace_chunk()))
            await session.execute(
                delete(models.CollectionDocument).where(
                    models.CollectionDocument.document_id.in_(document_ids)
                )
            )
            await session.execute(
                delete(models.DocumentTag).where(models.DocumentTag.document_id.in_(document_ids))
            )
            generations = (
                (
                    await session.execute(
                        select(models.DerivedGeneration).where(
                            models.DerivedGeneration.workspace_id == self._workspace_id,
                            models.DerivedGeneration.state.notin_(
                                {RebuildState.FAILED, RebuildState.CANCELED}
                            ),
                        )
                    )
                )
                .scalars()
                .all()
            )
            for generation in generations:
                if (
                    generation.vector_publication_id is None
                    and generation.lease_owner is not None
                    and generation.lease_generation > 0
                ):
                    generation.vector_publication_id = vector_publication_id(
                        generation.id,
                        generation.lease_owner,
                        generation.lease_generation,
                    )
                generation.state = RebuildState.CANCELED
                generation.published_at = None
                generation.diagnostic_code = None
                generation.lease_owner = None
                generation.lease_expires_at = None
                generation.lease_generation += 1
                generation.updated_at = utcnow()
            from pydantic import TypeAdapter  # noqa: PLC0415

            from manicule.ingest.reembed import ReembedRun, ReembedState  # noqa: PLC0415

            adapter = TypeAdapter(ReembedRun)
            reembed_runs = (
                (
                    await session.execute(
                        select(models.ReembedRunRecord).where(
                            models.ReembedRunRecord.workspace_id == self._workspace_id,
                            models.ReembedRunRecord.state.in_(
                                {
                                    ReembedState.PLANNED.value,
                                    ReembedState.BUILDING.value,
                                    ReembedState.VALIDATING.value,
                                    ReembedState.READY.value,
                                    ReembedState.PUBLISHED.value,
                                }
                            ),
                        )
                    )
                )
                .scalars()
                .all()
            )
            for row in reembed_runs:
                run = adapter.validate_json(row.checkpoint_json)
                failed = replace(
                    run,
                    state=ReembedState.FAILED,
                    revision=run.revision + 1,
                    failure="workspace derived index reset",
                )
                row.state = ReembedState.FAILED.value
                row.checkpoint_json = adapter.dump_json(failed).decode("utf-8")
                row.revision = failed.revision
                row.lease_owner = None
                row.lease_generation += 1
                row.lease_expires_at = None
                row.updated_at = utcnow()
            if reembed_runs:
                await session.execute(
                    delete(models.ReembedPublicationReceipt).where(
                        models.ReembedPublicationReceipt.workspace_id == self._workspace_id,
                        models.ReembedPublicationReceipt.run_id.in_(
                            [row.id for row in reembed_runs]
                        ),
                    )
                )
            await session.execute(
                update(models.ReembedShadowGeneration)
                .where(
                    models.ReembedShadowGeneration.workspace_id == self._workspace_id,
                    models.ReembedShadowGeneration.state.in_({"building", "sealed", "published"}),
                )
                .values(state="superseded")
            )
            await session.execute(
                update(models.Document)
                .where(
                    models.Document.workspace_id == self._workspace_id,
                    models.Document.deleted_at.is_(None),
                )
                .values(
                    status=DocumentStatus.DELETED,
                    status_detail="derived state reset; rebuild from retained snapshot",
                    failed_stage=None,
                    parse_fp=None,
                    chunk_fp=None,
                    embed_fp=None,
                    glossary_fp=None,
                    indexed_at=None,
                    publication_id=RESET_PUBLICATION,
                    deleted_at=utcnow(),
                    updated_at=utcnow(),
                )
            )
            state = await session.get(models.IndexState, self._workspace_id)
            namespace = "workspace" if state is None else state.vector_namespace
            vector_table = None if state is None else state.vector_table
        return ResetPreparation(
            documents=documents,
            chunks=chunks,
            memberships=memberships,
            generations_terminalized=int(acquisition_result.rowcount or 0)
            + len(generations)
            + len(reembed_runs),
            snapshots=await self._snapshot_item_count(),
            vector_namespace=namespace,
            vector_table=vector_table,
        )

    async def reset_reembed_run_ids(self) -> tuple[str, ...]:
        """Terminal re-embedding runs whose workspace-owned shadow directories can be removed."""
        async with self._sessions() as session:
            values = (
                await session.execute(
                    select(models.ReembedRunRecord.id)
                    .join(
                        models.ReembedShadowGeneration,
                        (
                            models.ReembedShadowGeneration.workspace_id
                            == models.ReembedRunRecord.workspace_id
                        )
                        & (models.ReembedShadowGeneration.run_id == models.ReembedRunRecord.id),
                    )
                    .where(
                        models.ReembedRunRecord.workspace_id == self._workspace_id,
                        models.ReembedRunRecord.state.in_({"failed", "superseded"}),
                    )
                    .order_by(models.ReembedRunRecord.id)
                )
            ).scalars()
        return tuple(str(value) for value in values)

    async def reset_derived(self) -> LifecycleOutcome:
        """Retire derived visibility while retaining retryable physical cleanup metadata."""
        prepared = await self.prepare_reset_derived()
        return LifecycleOutcome(
            operation=LifecycleOperation.RESET_DERIVED,
            removed_items=prepared.chunks,
            snapshot_items=prepared.snapshots,
        )

    async def plan_derived_generation_cleanup(self) -> LifecyclePlan:
        async with self._sessions() as session:
            eligible = self._obsolete_generation_predicate()
            count = int(
                await session.scalar(
                    select(func.count(models.DerivedGeneration.id)).where(eligible)
                )
                or 0
            )
            temporary = int(
                await session.scalar(
                    select(func.coalesce(func.sum(models.DerivedGenerationItem.temporary_bytes), 0))
                    .join(
                        models.DerivedGeneration,
                        models.DerivedGeneration.id == models.DerivedGenerationItem.generation_id,
                    )
                    .where(eligible)
                )
                or 0
            )
            total = int(
                await session.scalar(
                    select(func.count(models.DerivedGeneration.id)).where(
                        models.DerivedGeneration.workspace_id == self._workspace_id
                    )
                )
                or 0
            )
        return LifecyclePlan(
            operation=LifecycleOperation.CLEANUP_DERIVED_GENERATIONS,
            eligible_items=count,
            eligible_bytes=temporary,
            protected_items=total - count,
        )

    async def cleanup_obsolete_generation(self, generation_id: str) -> tuple[int, int]:
        """CAS-delete one still-obsolete generation after its physical namespace is gone."""
        async with self._sessions.begin() as session:
            eligible = self._obsolete_generation_predicate(generation_id=generation_id)
            temporary = int(
                await session.scalar(
                    select(func.coalesce(func.sum(models.DerivedGenerationItem.temporary_bytes), 0))
                    .join(
                        models.DerivedGeneration,
                        models.DerivedGeneration.id == models.DerivedGenerationItem.generation_id,
                    )
                    .where(eligible)
                )
                or 0
            )
            result = cast(
                "CursorResult[object]",
                await session.execute(delete(models.DerivedGeneration).where(eligible)),
            )
            return result.rowcount, temporary if result.rowcount else 0

    async def obsolete_generation_publications(
        self, *, page_size: int = GENERATION_CLEANUP_PAGE
    ) -> AsyncIterator[tuple[GenerationPublication, ...]]:
        """Yield stable keyset pages; never materialize the workspace generation inventory."""
        if page_size < 1 or page_size > GENERATION_CLEANUP_PAGE:
            raise ValueError(f"generation cleanup page size must be 1..{GENERATION_CLEANUP_PAGE}")
        after = ""
        while True:
            async with self._sessions() as session:
                rows = (
                    await session.execute(
                        select(
                            models.DerivedGeneration.id,
                            models.DerivedGeneration.vector_publication_id,
                            models.DerivedGeneration.replay_target_publication_id,
                            models.DerivedGeneration.expected_vector_table,
                        )
                        .where(
                            self._obsolete_generation_predicate(),
                            models.DerivedGeneration.id > after,
                        )
                        .order_by(models.DerivedGeneration.id)
                        .limit(page_size)
                    )
                ).all()
            if not rows:
                return
            page = tuple(GenerationPublication(*row) for row in rows)
            yield page
            after = page[-1].generation_id

    async def plan_source_history_release(self, cutoff: datetime) -> LifecyclePlan:
        async with self._sessions() as session:
            eligible = self._eligible_versions(cutoff)
            count = int(await session.scalar(select(func.count()).select_from(eligible)) or 0)
            bytes_ = await self._newly_unreferenced_version_bytes(session, eligible)
        return LifecyclePlan(
            operation=LifecycleOperation.RELEASE_SOURCE_HISTORY,
            eligible_items=count,
            eligible_bytes=bytes_,
        )

    async def release_source_history(self, cutoff: datetime) -> LifecycleOutcome:
        """Release prior-version bytes without deleting the historical version record."""
        async with self._sessions.begin() as session:
            eligible = self._eligible_versions(cutoff)
            count = int(await session.scalar(select(func.count()).select_from(eligible)) or 0)
            bytes_ = await self._newly_unreferenced_version_bytes(session, eligible)
            if count:
                await session.execute(
                    update(models.DocumentVersion)
                    .where(models.DocumentVersion.id.in_(select(eligible.c.id)))
                    .values(original_ref=None, bytes_released_at=utcnow())
                )
        return LifecycleOutcome(
            operation=LifecycleOperation.RELEASE_SOURCE_HISTORY,
            removed_items=count,
            released_bytes=bytes_,
        )

    async def plan_snapshot_deletion(self, run_id: str) -> LifecyclePlan:
        async with self._sessions() as session:
            run = await self._deletable_snapshot(session, run_id)
            item_count, unique_items, unique_bytes = await self._snapshot_impact(session, run)
            token = self._confirmation(
                run.id,
                run.membership_hash or "",
                item_count,
                unique_items,
                unique_bytes,
            )
        return LifecyclePlan(
            operation=LifecycleOperation.DELETE_SNAPSHOT,
            eligible_items=item_count,
            eligible_bytes=unique_bytes,
            snapshot_items=item_count,
            unrecoverable_items=unique_items,
            unrecoverable_bytes=unique_bytes,
            confirmation=token,
        )

    async def delete_snapshot(self, run_id: str, *, confirmation: str) -> LifecycleOutcome:
        """Delete one settled snapshot only when its exact dry-run token is confirmed."""
        async with self._sessions.begin() as session:
            run = await self._deletable_snapshot(session, run_id)
            item_count, unique_items, unique_bytes = await self._snapshot_impact(session, run)
            expected = self._confirmation(
                run.id,
                run.membership_hash or "",
                item_count,
                unique_items,
                unique_bytes,
            )
            if not confirmation or not hmac.compare_digest(confirmation, expected):
                raise LifecycleRefusalError(
                    "snapshot deletion requires the confirmation token from its current dry run"
                )
            removable_documents = self._removable_document_ids(run)
            await session.execute(
                update(models.Document)
                .where(models.Document.id.in_(removable_documents))
                .values(
                    original_ref=None,
                    original_omitted_reason="source snapshot deleted by operator",
                    updated_at=utcnow(),
                )
            )
            watermark_scope_matches: ColumnElement[bool] = (
                models.Connector.watermark_scope_fingerprint == run.scope_fingerprint
            )
            if run.candidate_watermark is not None:
                watermark_scope_matches = or_(
                    watermark_scope_matches,
                    models.Connector.watermark_scope_fingerprint.is_(None),
                )
            await session.execute(
                update(models.Connector)
                .where(
                    models.Connector.id == run.connector_id,
                    models.Connector.workspace_id == self._workspace_id,
                    watermark_scope_matches,
                    models.Connector.watermark == run.candidate_watermark,
                )
                .values(
                    # SQLAlchemy's JSON type otherwise binds Python None as JSON `null`.  The
                    # next snapshot promotion uses `IS NULL` for its base-watermark CAS, so the
                    # lifecycle boundary must clear the column to SQL NULL explicitly.
                    watermark=null(),
                    watermark_scope_fingerprint=None,
                    last_synced_at=None,
                )
            )
            result = cast(
                "CursorResult[object]",
                await session.execute(
                    delete(models.AcquisitionRun).where(
                        models.AcquisitionRun.id == run.id,
                        models.AcquisitionRun.workspace_id == self._workspace_id,
                    )
                ),
            )
            if result.rowcount != 1:
                raise LifecycleRefusalError("snapshot changed before deletion committed")
        return LifecycleOutcome(
            operation=LifecycleOperation.DELETE_SNAPSHOT,
            removed_items=1,
            released_bytes=unique_bytes,
            snapshot_items=item_count,
        )

    async def _deletable_snapshot(
        self, session: AsyncSession, run_id: str
    ) -> models.AcquisitionRun:
        run = await session.get(models.AcquisitionRun, run_id)
        if run is None or run.workspace_id != self._workspace_id or run.promoted_at is None:
            raise LifecycleRefusalError("no promoted snapshot with that identity in this workspace")
        if run.state is not AcquisitionRunState.SETTLED:
            raise LifecycleRefusalError("snapshot still has resumable work and cannot be deleted")
        dependent = await session.scalar(
            select(exists().where(models.DerivedGeneration.snapshot_run_id == run.id))
        )
        if not dependent:
            dependent = await session.scalar(
                select(exists().where(models.DerivedGenerationSnapshot.run_id == run.id))
            )
        if dependent:
            raise LifecycleRefusalError(
                "snapshot still anchors a derived generation; clean obsolete generations first"
            )
        marker = await session.scalar(
            select(exists().where(models.AcquisitionMarker.run_id == run.id))
        )
        if marker:
            raise LifecycleRefusalError("snapshot still has durable acquisition recovery markers")
        newer = await session.scalar(
            select(
                exists().where(
                    models.AcquisitionRun.workspace_id == self._workspace_id,
                    models.AcquisitionRun.connector_id == run.connector_id,
                    models.AcquisitionRun.scope_fingerprint == run.scope_fingerprint,
                    models.AcquisitionRun.promoted_at.is_not(None),
                    (
                        (models.AcquisitionRun.promoted_at > run.promoted_at)
                        | (
                            (models.AcquisitionRun.promoted_at == run.promoted_at)
                            & (models.AcquisitionRun.id > run.id)
                        )
                    ),
                )
            )
        )
        if newer:
            raise LifecycleRefusalError(
                "only the current promoted snapshot can use snapshot deletion; "
                "release older source history under retention policy"
            )
        return run

    async def _snapshot_impact(
        self, session: AsyncSession, run: models.AcquisitionRun
    ) -> tuple[int, int, int]:
        run_id = run.id
        item_count = int(
            await session.scalar(
                select(func.count(models.AcquisitionRecord.id)).where(
                    models.AcquisitionRecord.run_id == run_id
                )
            )
            or 0
        )
        target = select(models.AcquisitionRecord.blob_ref).where(
            models.AcquisitionRecord.run_id == run_id,
            models.AcquisitionRecord.blob_ref.is_not(None),
        )
        other_roots = union(
            select(models.AcquisitionRecord.blob_ref).where(
                models.AcquisitionRecord.run_id != run_id,
                models.AcquisitionRecord.blob_ref.is_not(None),
            ),
            select(models.Document.original_ref).where(
                models.Document.id.not_in(self._removable_document_ids(run)),
                models.Document.original_ref.is_not(None),
            ),
            select(models.DocumentVersion.original_ref).where(
                models.DocumentVersion.original_ref.is_not(None)
            ),
            select(models.AcquisitionMarker.blob_ref).where(
                models.AcquisitionMarker.blob_ref.is_not(None)
            ),
        )
        unique = (
            target.where(models.AcquisitionRecord.blob_ref.not_in(other_roots))
            .distinct()
            .subquery()
        )
        unique_row = (
            await session.execute(
                select(
                    func.count(unique.c.blob_ref),
                    func.coalesce(func.sum(models.Blob.size_bytes), 0),
                )
                .select_from(unique)
                .join(models.Blob, models.Blob.hash == unique.c.blob_ref)
            )
        ).one()
        unique_items = int(
            await session.scalar(
                select(func.count(models.AcquisitionRecord.id))
                .select_from(models.AcquisitionRecord)
                .join(unique, unique.c.blob_ref == models.AcquisitionRecord.blob_ref)
                .where(models.AcquisitionRecord.run_id == run_id)
            )
            or 0
        )
        return item_count, unique_items, int(unique_row[1])

    def _removable_document_ids(self, run: models.AcquisitionRun) -> Select[tuple[str]]:
        target = (
            select(
                models.AcquisitionRecord.source_id,
                models.AcquisitionRecord.blob_ref,
            )
            .where(
                models.AcquisitionRecord.run_id == run.id,
                models.AcquisitionRecord.blob_ref.is_not(None),
            )
            .subquery()
        )
        return (
            select(models.Document.id)
            .join(
                target,
                (target.c.source_id == models.Document.source_id)
                & (target.c.blob_ref == models.Document.original_ref),
            )
            .where(
                models.Document.workspace_id == self._workspace_id,
                models.Document.source == run.connector_name,
            )
        )

    def _obsolete_generation_predicate(
        self, *, generation_id: str | None = None
    ) -> ColumnElement[bool]:
        generation = models.DerivedGeneration
        newest_published = (
            select(generation.id)
            .where(
                generation.workspace_id == self._workspace_id,
                generation.state == RebuildState.PUBLISHED,
            )
            .order_by(generation.published_at.desc(), generation.id.desc())
            .limit(1)
            .scalar_subquery()
        )
        live = exists().where(
            models.Document.workspace_id == self._workspace_id,
            models.Document.publication_id == generation.id,
        )
        predicate = (
            (generation.workspace_id == self._workspace_id)
            & generation.state.in_(
                {RebuildState.FAILED, RebuildState.CANCELED, RebuildState.PUBLISHED}
            )
            & or_(generation.state != RebuildState.PUBLISHED, generation.id != newest_published)
            & ~live
        )
        if generation_id is not None:
            predicate &= generation.id == generation_id
        return predicate

    def _eligible_versions(self, cutoff: datetime) -> Subquery:
        return (
            select(models.DocumentVersion.id, models.DocumentVersion.original_ref)
            .join(models.Document, models.Document.id == models.DocumentVersion.document_id)
            .where(
                models.Document.workspace_id == self._workspace_id,
                models.DocumentVersion.created_at < cutoff,
                models.DocumentVersion.original_ref.is_not(None),
                models.DocumentVersion.bytes_released_at.is_(None),
            )
            .subquery()
        )

    async def _newly_unreferenced_version_bytes(
        self,
        session: AsyncSession,
        eligible: Subquery,
    ) -> int:
        other_roots = union(
            select(models.Document.original_ref).where(models.Document.original_ref.is_not(None)),
            select(models.AcquisitionRecord.blob_ref).where(
                models.AcquisitionRecord.blob_ref.is_not(None)
            ),
            select(models.AcquisitionMarker.blob_ref).where(
                models.AcquisitionMarker.blob_ref.is_not(None)
            ),
            select(models.DocumentVersion.original_ref).where(
                models.DocumentVersion.id.not_in(select(eligible.c.id)),
                models.DocumentVersion.original_ref.is_not(None),
            ),
        )
        candidate_refs = (
            select(eligible.c.original_ref)
            .where(
                eligible.c.original_ref.is_not(None),
                eligible.c.original_ref.not_in(other_roots),
            )
            .distinct()
        )
        return int(
            await session.scalar(
                select(func.coalesce(func.sum(models.Blob.size_bytes), 0)).where(
                    models.Blob.hash.in_(candidate_refs)
                )
            )
            or 0
        )

    async def _snapshot_item_count(self) -> int:
        async with self._sessions() as session:
            return int(
                await session.scalar(
                    select(func.count(models.AcquisitionRecord.id)).where(
                        models.AcquisitionRecord.workspace_id == self._workspace_id
                    )
                )
                or 0
            )

    async def _workspace_document_count(self, session: AsyncSession) -> int:
        return int(
            await session.scalar(
                select(func.count(models.Document.id)).where(
                    models.Document.workspace_id == self._workspace_id,
                    models.Document.deleted_at.is_(None),
                )
            )
            or 0
        )

    async def _workspace_chunk_count(self, session: AsyncSession) -> int:
        return int(
            await session.scalar(select(func.count(models.Chunk.id)).where(self._workspace_chunk()))
            or 0
        )

    def _workspace_chunk(self) -> ColumnElement[bool]:
        return models.Chunk.document_id.in_(
            select(models.Document.id).where(models.Document.workspace_id == self._workspace_id)
        )

    @staticmethod
    def _confirmation(
        run_id: str,
        membership_hash: str,
        snapshot_items: int,
        unrecoverable_items: int,
        unrecoverable_bytes: int,
    ) -> str:
        evidence = (
            f"snapshot-delete\0{run_id}\0{membership_hash}\0{snapshot_items}\0"
            f"{unrecoverable_items}\0{unrecoverable_bytes}"
        )
        return hashlib.sha256(evidence.encode()).hexdigest()


__all__ = [
    "GENERATION_CLEANUP_PAGE",
    "GenerationPublication",
    "SourceLifecycleMixin",
]
