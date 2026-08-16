"""Reference-safe source and derived lifecycle operations over SQLite.

Every plan is an aggregate.  No source id, URI, blob digest, or document title crosses this
boundary.  Mutations are single database transactions, which makes cancellation before commit
a no-op and a retry after commit idempotent.
"""

from __future__ import annotations

import hashlib
import hmac
from datetime import datetime
from typing import TYPE_CHECKING, cast

from sqlalchemy import delete, exists, func, select, true, union, update

from manicule.core.acquisition import AcquisitionRunState
from manicule.core.content import DocumentStatus
from manicule.core.rebuild import RebuildState
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
    from sqlalchemy import CursorResult
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql.elements import ColumnElement
    from sqlalchemy.sql.selectable import Select, Subquery


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

    async def reset_derived(self) -> LifecycleOutcome:
        """Remove derived rows while retaining documents, versions, manifests and blobs."""
        async with self._sessions.begin() as session:
            chunk_count = await self._workspace_chunk_count(session)
            await session.execute(delete(models.Chunk).where(self._workspace_chunk()))
            # A published generation is a replay receipt for the derived corpus we just
            # removed.  Leaving it behind makes an identical-target rebuild return PUBLISHED
            # without rebuilding anything.  Only completed publications are invalidated;
            # planned/building/validating rows remain durable retry work.
            await session.execute(
                delete(models.DerivedGeneration).where(
                    models.DerivedGeneration.workspace_id == self._workspace_id,
                    models.DerivedGeneration.state == RebuildState.PUBLISHED,
                )
            )
            await session.execute(
                update(models.Document)
                .where(models.Document.workspace_id == self._workspace_id)
                .values(
                    status=DocumentStatus.PENDING,
                    status_detail="derived state reset; rebuild from retained snapshot",
                    failed_stage=None,
                    parse_fp=None,
                    chunk_fp=None,
                    embed_fp=None,
                    glossary_fp=None,
                    indexed_at=None,
                    updated_at=utcnow(),
                )
            )
        return LifecycleOutcome(
            operation=LifecycleOperation.RESET_DERIVED,
            removed_items=chunk_count,
            snapshot_items=await self._snapshot_item_count(),
        )

    async def plan_derived_generation_cleanup(self) -> LifecyclePlan:
        async with self._sessions() as session:
            eligible = await self._obsolete_generation_ids(session)
            temporary = await self._generation_bytes(session, eligible)
            protected = (
                await session.execute(
                    select(func.count(models.DerivedGeneration.id)).where(
                        models.DerivedGeneration.workspace_id == self._workspace_id,
                        models.DerivedGeneration.id.not_in(eligible) if eligible else true(),
                    )
                )
            ).scalar_one()
        return LifecyclePlan(
            operation=LifecycleOperation.CLEANUP_DERIVED_GENERATIONS,
            eligible_items=len(eligible),
            eligible_bytes=temporary,
            protected_items=int(protected),
        )

    async def cleanup_derived_generations(self) -> LifecycleOutcome:
        """Delete only terminal generations that cannot be the published generation."""
        async with self._sessions.begin() as session:
            eligible = await self._obsolete_generation_ids(session)
            temporary = await self._generation_bytes(session, eligible)
            if eligible:
                await session.execute(
                    delete(models.DerivedGeneration).where(
                        models.DerivedGeneration.workspace_id == self._workspace_id,
                        models.DerivedGeneration.id.in_(eligible),
                    )
                )
        return LifecycleOutcome(
            operation=LifecycleOperation.CLEANUP_DERIVED_GENERATIONS,
            removed_items=len(eligible),
            released_bytes=temporary,
        )

    async def obsolete_vector_publications(self) -> tuple[str, ...]:
        """Internal physical vector namespaces selected by the same cleanup predicate."""
        async with self._sessions() as session:
            eligible = await self._obsolete_generation_ids(session)
            if not eligible:
                return ()
            return tuple(
                publication
                for publication in (
                    await session.execute(
                        select(models.DerivedGeneration.vector_publication_id).where(
                            models.DerivedGeneration.id.in_(eligible)
                        )
                    )
                ).scalars()
                if publication is not None
            )

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
            await session.execute(
                update(models.Connector)
                .where(
                    models.Connector.id == run.connector_id,
                    models.Connector.workspace_id == self._workspace_id,
                    models.Connector.watermark_scope_fingerprint == run.scope_fingerprint,
                    models.Connector.watermark == run.candidate_watermark,
                )
                .values(
                    watermark=None,
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

    async def _obsolete_generation_ids(self, session: AsyncSession) -> list[str]:
        rows = list(
            (
                await session.execute(
                    select(
                        models.DerivedGeneration.id,
                        models.DerivedGeneration.state,
                        models.DerivedGeneration.vector_publication_id,
                        models.DerivedGeneration.published_at,
                    ).where(models.DerivedGeneration.workspace_id == self._workspace_id)
                )
            ).tuples()
        )
        live_publications = set(
            (
                await session.execute(
                    select(models.Document.publication_id).where(
                        models.Document.workspace_id == self._workspace_id
                    )
                )
            ).scalars()
        )
        published = [row for row in rows if row[1] is RebuildState.PUBLISHED]
        newest_published = (
            max(
                published,
                key=lambda row: (
                    row[3].isoformat() if row[3] is not None else "",
                    row[0],
                ),
            )[0]
            if published
            else None
        )
        return [
            row[0]
            for row in rows
            if row[1] in {RebuildState.FAILED, RebuildState.CANCELED, RebuildState.PUBLISHED}
            and row[0] != newest_published
            and row[2] not in live_publications
        ]

    @staticmethod
    async def _generation_bytes(session: AsyncSession, generation_ids: list[str]) -> int:
        if not generation_ids:
            return 0
        return int(
            await session.scalar(
                select(
                    func.coalesce(func.sum(models.DerivedGenerationItem.temporary_bytes), 0)
                ).where(models.DerivedGenerationItem.generation_id.in_(generation_ids))
            )
            or 0
        )

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
                    models.Document.workspace_id == self._workspace_id
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


__all__ = ["SourceLifecycleMixin"]
