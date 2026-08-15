"""SQLite-owned full inventories and deletion proposals.

No method here accepts a caller-provided collection and calls it complete. Enumeration rows
become deletion authority only through :meth:`complete_reconciliation_inventory`, and every
consumer revalidates the resulting handle inside its write transaction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import delete, exists, func, insert, literal, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from manicule.core.reconciliation import (
    CompletedInventory,
    ReconciliationAssessment,
    ReconciliationRunState,
)
from manicule.storage import models
from manicule.storage.acquisition import AcquisitionConflictError
from manicule.storage.scoped import WorkspaceScoped
from manicule.storage.types import utcnow

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from sqlalchemy import CursorResult
    from sqlalchemy.ext.asyncio import AsyncSession

INVENTORY_PAGE_LIMIT = 1_000


class ReconciliationInventoryError(AcquisitionConflictError):
    """A reconciliation run or completed-inventory guard did not match."""


def _handle(row: models.ReconciliationRun) -> CompletedInventory:
    if row.completed_at is None:
        msg = f"reconciliation run {row.id!r} is not complete"
        raise ReconciliationInventoryError(msg)
    return CompletedInventory(
        run_id=row.id,
        workspace_id=row.workspace_id,
        connector_id=row.connector_id,
        connector=row.connector_name,
        scope=row.scope,
        completed_at=row.completed_at,
        seen_count=row.seen_count,
    )


class ReconciliationJournalMixin(WorkspaceScoped):
    """Workspace-scoped durable inventory, diff, proposal, and confirmation operations."""

    async def begin_reconciliation_inventory(self, run_id: str, connector: str, scope: str) -> None:
        """Start a fresh full enumeration and retire any abandoned partial one."""
        if not run_id or not connector or not scope:
            msg = "run_id, connector, and scope must not be empty"
            raise ValueError(msg)
        async with self._sessions.begin() as session:
            connector_row = await self._reconciliation_connector(session, connector)
            now = utcnow()
            await session.execute(
                update(models.ReconciliationRun)
                .where(
                    models.ReconciliationRun.workspace_id == self._workspace_id,
                    models.ReconciliationRun.connector_id == connector_row.id,
                    models.ReconciliationRun.state.in_(
                        (
                            ReconciliationRunState.ENUMERATING,
                            ReconciliationRunState.COMPLETED,
                            ReconciliationRunState.PROPOSED,
                        )
                    ),
                )
                .values(state=ReconciliationRunState.CANCELED, updated_at=now)
            )
            # Connector metadata holds one operator-facing proposal. Starting a newer full
            # inventory makes every older question stale, even when configuration changed its
            # scope; leaving the JSON behind would let confirmation select a different run than
            # the one the operator reviewed.
            await self._merge_metadata(session, connector, {"proposed_deletion": None})
            await session.execute(
                sqlite_insert(models.ReconciliationRun)
                .values(
                    id=run_id,
                    workspace_id=self._workspace_id,
                    connector_id=connector_row.id,
                    connector_name=connector,
                    scope=scope,
                    state=ReconciliationRunState.ENUMERATING,
                    created_at=now,
                    updated_at=now,
                )
                .on_conflict_do_nothing(index_elements=[models.ReconciliationRun.id])
            )
            row = await session.get(models.ReconciliationRun, run_id)
            if (
                row is None
                or row.workspace_id != self._workspace_id
                or row.connector_id != connector_row.id
                or row.connector_name != connector
                or row.scope != scope
                or row.state is not ReconciliationRunState.ENUMERATING
            ):
                msg = f"reconciliation run {run_id!r} is not a new inventory for this scope"
                raise ReconciliationInventoryError(msg)

    async def append_reconciliation_inventory_page(
        self, run_id: str, connector: str, scope: str, source_ids: Sequence[str]
    ) -> int:
        """Persist one bounded page, deduplicating repeated source identities in SQLite."""
        if len(source_ids) > INVENTORY_PAGE_LIMIT:
            msg = f"inventory pages are limited to {INVENTORY_PAGE_LIMIT} identities"
            raise ValueError(msg)
        if any(not source_id for source_id in source_ids):
            msg = "source identities must not be empty"
            raise ValueError(msg)
        async with self._sessions.begin() as session:
            row = await self._required_enumerating_run(session, run_id, connector, scope)
            before = row.seen_count
            for source_id in source_ids:
                await session.execute(
                    sqlite_insert(models.ReconciliationInventoryItem)
                    .values(
                        run_id=row.id,
                        workspace_id=row.workspace_id,
                        connector_id=row.connector_id,
                        source_id=source_id,
                    )
                    .on_conflict_do_nothing(
                        index_elements=[
                            models.ReconciliationInventoryItem.run_id,
                            models.ReconciliationInventoryItem.source_id,
                        ]
                    )
                )
            row.seen_count = (
                await session.execute(
                    select(func.count())
                    .select_from(models.ReconciliationInventoryItem)
                    .where(models.ReconciliationInventoryItem.run_id == row.id)
                )
            ).scalar_one()
            row.updated_at = utcnow()
            return row.seen_count - before

    async def cancel_reconciliation_inventory(
        self, run_id: str, connector: str, scope: str
    ) -> bool:
        """Make a partial/error/canceled enumeration permanently ineligible for diffing."""
        async with self._sessions.begin() as session:
            result = cast(
                "CursorResult[Any]",
                await session.execute(
                    update(models.ReconciliationRun)
                    .where(
                        models.ReconciliationRun.id == run_id,
                        models.ReconciliationRun.workspace_id == self._workspace_id,
                        models.ReconciliationRun.connector_name == connector,
                        models.ReconciliationRun.scope == scope,
                        models.ReconciliationRun.state == ReconciliationRunState.ENUMERATING,
                    )
                    .values(state=ReconciliationRunState.CANCELED, updated_at=utcnow())
                ),
            )
            return result.rowcount == 1

    async def complete_reconciliation_inventory(
        self, run_id: str, connector: str, scope: str, *, now: datetime
    ) -> CompletedInventory:
        """Atomically close enumeration and mint the only accepted diff authority."""
        if now.tzinfo is None:
            msg = "completion time must be timezone-aware"
            raise ValueError(msg)
        async with self._sessions.begin() as session:
            row = await self._required_enumerating_run(session, run_id, connector, scope)
            actual = (
                await session.execute(
                    select(func.count())
                    .select_from(models.ReconciliationInventoryItem)
                    .where(
                        models.ReconciliationInventoryItem.run_id == row.id,
                        models.ReconciliationInventoryItem.workspace_id == self._workspace_id,
                        models.ReconciliationInventoryItem.connector_id == row.connector_id,
                    )
                )
            ).scalar_one()
            row.seen_count = actual
            row.completed_at = now
            row.state = ReconciliationRunState.COMPLETED
            row.updated_at = utcnow()
            await session.flush()
            return _handle(row)

    async def latest_completed_reconciliation_inventory(
        self, connector: str, scope: str
    ) -> CompletedInventory | None:
        """Find crash-recovery work without contacting the source again."""
        async with self._sessions() as session:
            row = (
                await session.execute(
                    select(models.ReconciliationRun)
                    .where(
                        models.ReconciliationRun.workspace_id == self._workspace_id,
                        models.ReconciliationRun.connector_name == connector,
                        models.ReconciliationRun.scope == scope,
                        models.ReconciliationRun.state == ReconciliationRunState.COMPLETED,
                    )
                    .order_by(
                        models.ReconciliationRun.completed_at.desc(),
                        models.ReconciliationRun.id.desc(),
                    )
                    .limit(1)
                )
            ).scalar_one_or_none()
            return None if row is None else _handle(row)

    async def assess_reconciliation_inventory(
        self,
        inventory: CompletedInventory,
        *,
        max_delete_fraction: float,
        dry_run: bool = False,
        now: datetime,
    ) -> ReconciliationAssessment:
        """Diff and apply with server-side anti-joins; no corpus identity list enters Python."""
        if not 0 <= max_delete_fraction <= 1:
            msg = "max_delete_fraction must be between zero and one"
            raise ValueError(msg)
        if now.tzinfo is None:
            msg = "assessment time must be timezone-aware"
            raise ValueError(msg)
        if dry_run:
            # Commit consumption before computing the preview. If the process dies during the
            # query, the inventory remains terminal instead of rolling back to reusable deletion
            # authority. Ordinary apply keeps its original one-transaction crash recovery.
            async with self._sessions.begin() as session:
                run = await self._required_completed_handle(session, inventory)
                run.state = ReconciliationRunState.DRY_RUN
                run.updated_at = utcnow()
            return await self._assess_consumed_dry_run(inventory)

        async with self._sessions.begin() as session:
            run = await self._required_completed_handle(session, inventory)
            live = self._live_documents(run)
            absent = ~exists(
                select(models.ReconciliationInventoryItem.source_id).where(
                    models.ReconciliationInventoryItem.run_id == run.id,
                    models.ReconciliationInventoryItem.workspace_id == run.workspace_id,
                    models.ReconciliationInventoryItem.connector_id == run.connector_id,
                    models.ReconciliationInventoryItem.source_id == models.Document.source_id,
                )
            )
            live_count = (
                await session.execute(
                    select(func.count()).select_from(models.Document).where(*live)
                )
            ).scalar_one()
            missing_count = (
                await session.execute(
                    select(func.count()).select_from(models.Document).where(*live, absent)
                )
            ).scalar_one()
            run.live_count = live_count
            run.missing_count = missing_count
            run.updated_at = utcnow()
            await session.execute(
                delete(models.ReconciliationCandidate).where(
                    models.ReconciliationCandidate.run_id == run.id
                )
            )
            if missing_count:
                candidates = select(
                    literal(run.id),
                    models.Document.id,
                    literal(run.workspace_id),
                    literal(run.connector_id),
                    models.Document.publication_id,
                    models.Document.content_hash,
                    models.Document.version_token,
                    models.Document.last_seen_at,
                ).where(*live, absent)
                await session.execute(
                    insert(models.ReconciliationCandidate).from_select(
                        [
                            "run_id",
                            "document_id",
                            "workspace_id",
                            "connector_id",
                            "publication_id",
                            "content_hash",
                            "version_token",
                            "last_seen_at",
                        ],
                        candidates,
                    )
                )
            fraction = missing_count / live_count if live_count else 0.0
            if fraction > max_delete_fraction:
                run.state = ReconciliationRunState.PROPOSED
                refusal = (
                    f"reconciling {run.connector_name!r} would soft-delete {missing_count} of "
                    f"{live_count} live document(s), above the {max_delete_fraction:.0%} ceiling. "
                    "The durable proposal is recorded for confirmation."
                )
                await self._record_proposal(session, run, now)
                return ReconciliationAssessment(
                    connector=run.connector_name,
                    scope=run.scope,
                    seen_count=run.seen_count,
                    live_count=live_count,
                    missing_count=missing_count,
                    refused=refusal,
                )

            applied = await self._apply_candidates(session, run, now)
            run.state = ReconciliationRunState.APPLIED
            await self._record_clean(session, run.connector_name, now)
            return ReconciliationAssessment(
                connector=run.connector_name,
                scope=run.scope,
                seen_count=run.seen_count,
                live_count=live_count,
                missing_count=missing_count,
                applied_count=applied,
            )

    async def _assess_consumed_dry_run(
        self, inventory: CompletedInventory
    ) -> ReconciliationAssessment:
        """Compute a preview only after its deletion authority is durably consumed."""
        async with self._sessions.begin() as session:
            run = await session.get(models.ReconciliationRun, inventory.run_id)
            if (
                run is None
                or run.workspace_id != self._workspace_id
                or run.workspace_id != inventory.workspace_id
                or run.connector_id != inventory.connector_id
                or run.connector_name != inventory.connector
                or run.scope != inventory.scope
                or run.completed_at != inventory.completed_at
                or run.seen_count != inventory.seen_count
                or run.state is not ReconciliationRunState.DRY_RUN
            ):
                msg = "consumed dry-run inventory no longer matches its durable identity"
                raise ReconciliationInventoryError(msg)
            live = self._live_documents(run)
            absent = ~exists(
                select(models.ReconciliationInventoryItem.source_id).where(
                    models.ReconciliationInventoryItem.run_id == run.id,
                    models.ReconciliationInventoryItem.workspace_id == run.workspace_id,
                    models.ReconciliationInventoryItem.connector_id == run.connector_id,
                    models.ReconciliationInventoryItem.source_id == models.Document.source_id,
                )
            )
            live_count = (
                await session.execute(
                    select(func.count()).select_from(models.Document).where(*live)
                )
            ).scalar_one()
            missing_count = (
                await session.execute(
                    select(func.count()).select_from(models.Document).where(*live, absent)
                )
            ).scalar_one()
            run.live_count = live_count
            run.missing_count = missing_count
            run.updated_at = utcnow()
            return ReconciliationAssessment(
                connector=run.connector_name,
                scope=run.scope,
                seen_count=run.seen_count,
                live_count=live_count,
                missing_count=missing_count,
                dry_run=True,
            )

    async def confirm_reconciliation_proposal(
        self, connector: str, *, scope: str, now: datetime
    ) -> ReconciliationAssessment | None:
        """Apply exactly the stored candidate revisions, skipping documents changed since."""
        if now.tzinfo is None:
            msg = "confirmation time must be timezone-aware"
            raise ValueError(msg)
        if not scope:
            msg = "confirmation requires the current non-empty reconciliation scope"
            raise ValueError(msg)
        async with self._sessions.begin() as session:
            connector_row = await self._reconciliation_connector(session, connector)
            metadata = dict(cast("Any", connector_row.run_metadata) or {})
            proposal = metadata.get("proposed_deletion")
            if not isinstance(proposal, dict):
                return None
            typed_proposal = cast("dict[str, Any]", proposal)
            run_id = typed_proposal.get("run_id")
            recorded_scope = typed_proposal.get("scope")
            if not isinstance(run_id, str) or not isinstance(recorded_scope, str):
                return None
            if scope != recorded_scope:
                return None
            clauses = [
                models.ReconciliationRun.id == run_id,
                models.ReconciliationRun.workspace_id == self._workspace_id,
                models.ReconciliationRun.connector_id == connector_row.id,
                models.ReconciliationRun.connector_name == connector,
                models.ReconciliationRun.scope == recorded_scope,
                models.ReconciliationRun.state == ReconciliationRunState.PROPOSED,
            ]
            claimed = cast(
                "CursorResult[Any]",
                await session.execute(
                    update(models.ReconciliationRun)
                    .where(*clauses)
                    .values(updated_at=models.ReconciliationRun.updated_at)
                ),
            )
            if claimed.rowcount != 1:
                return None
            row = (
                await session.execute(select(models.ReconciliationRun).where(*clauses))
            ).scalar_one_or_none()
            if row is None:
                return None
            applied = await self._apply_candidates(session, row, now)
            row.state = ReconciliationRunState.APPLIED
            row.updated_at = utcnow()
            await self._record_clean(session, row.connector_name, now)
            return ReconciliationAssessment(
                connector=row.connector_name,
                scope=row.scope,
                seen_count=row.seen_count,
                live_count=row.live_count or 0,
                missing_count=row.missing_count or 0,
                applied_count=applied,
            )

    async def _required_enumerating_run(
        self, session: AsyncSession, run_id: str, connector: str, scope: str
    ) -> models.ReconciliationRun:
        guarded = cast(
            "CursorResult[Any]",
            await session.execute(
                update(models.ReconciliationRun)
                .where(
                    models.ReconciliationRun.id == run_id,
                    models.ReconciliationRun.workspace_id == self._workspace_id,
                    models.ReconciliationRun.connector_name == connector,
                    models.ReconciliationRun.scope == scope,
                    models.ReconciliationRun.state == ReconciliationRunState.ENUMERATING,
                )
                .values(updated_at=models.ReconciliationRun.updated_at)
            ),
        )
        if guarded.rowcount != 1:
            msg = f"reconciliation run {run_id!r} is not an active inventory for this scope"
            raise ReconciliationInventoryError(msg)
        row = await session.get(models.ReconciliationRun, run_id)
        if (
            row is None
            or row.workspace_id != self._workspace_id
            or row.connector_name != connector
            or row.scope != scope
            or row.state is not ReconciliationRunState.ENUMERATING
        ):
            msg = f"reconciliation run {run_id!r} is not an active inventory for this scope"
            raise ReconciliationInventoryError(msg)
        return row

    async def _required_completed_handle(
        self, session: AsyncSession, inventory: CompletedInventory
    ) -> models.ReconciliationRun:
        guarded = cast(
            "CursorResult[Any]",
            await session.execute(
                update(models.ReconciliationRun)
                .where(
                    models.ReconciliationRun.id == inventory.run_id,
                    models.ReconciliationRun.workspace_id == self._workspace_id,
                    models.ReconciliationRun.workspace_id == inventory.workspace_id,
                    models.ReconciliationRun.connector_id == inventory.connector_id,
                    models.ReconciliationRun.connector_name == inventory.connector,
                    models.ReconciliationRun.scope == inventory.scope,
                    models.ReconciliationRun.completed_at == inventory.completed_at,
                    models.ReconciliationRun.seen_count == inventory.seen_count,
                    models.ReconciliationRun.state == ReconciliationRunState.COMPLETED,
                )
                .values(updated_at=models.ReconciliationRun.updated_at)
            ),
        )
        if guarded.rowcount != 1:
            msg = (
                "completed inventory handle does not match durable workspace/connector/scope state"
            )
            raise ReconciliationInventoryError(msg)
        row = await session.get(models.ReconciliationRun, inventory.run_id)
        if (
            row is None
            or row.workspace_id != self._workspace_id
            or row.workspace_id != inventory.workspace_id
            or row.connector_id != inventory.connector_id
            or row.connector_name != inventory.connector
            or row.scope != inventory.scope
            or row.completed_at != inventory.completed_at
            or row.seen_count != inventory.seen_count
            or row.state is not ReconciliationRunState.COMPLETED
        ):
            msg = (
                "completed inventory handle does not match durable workspace/connector/scope state"
            )
            raise ReconciliationInventoryError(msg)
        return row

    def _live_documents(self, run: models.ReconciliationRun) -> tuple[Any, ...]:
        return (
            models.Document.workspace_id == self._workspace_id,
            models.Document.source == run.connector_name,
            models.Document.deleted_at.is_(None),
        )

    async def _apply_candidates(
        self, session: AsyncSession, run: models.ReconciliationRun, now: datetime
    ) -> int:
        candidate_matches = exists(
            select(models.ReconciliationCandidate.document_id).where(
                models.ReconciliationCandidate.run_id == run.id,
                models.ReconciliationCandidate.workspace_id == self._workspace_id,
                models.ReconciliationCandidate.connector_id == run.connector_id,
                models.ReconciliationCandidate.document_id == models.Document.id,
                models.ReconciliationCandidate.publication_id == models.Document.publication_id,
                models.ReconciliationCandidate.content_hash == models.Document.content_hash,
                models.ReconciliationCandidate.version_token.is_not_distinct_from(
                    models.Document.version_token
                ),
                models.ReconciliationCandidate.last_seen_at.is_not_distinct_from(
                    models.Document.last_seen_at
                ),
            )
        )
        result = cast(
            "CursorResult[Any]",
            await session.execute(
                update(models.Document)
                .where(*self._live_documents(run), candidate_matches)
                .values(deleted_at=now, updated_at=utcnow())
            ),
        )
        return result.rowcount

    async def _record_proposal(
        self, session: AsyncSession, run: models.ReconciliationRun, now: datetime
    ) -> None:
        await self._merge_metadata(
            session,
            run.connector_name,
            {
                "proposed_deletion": {
                    "run_id": run.id,
                    "scope": run.scope,
                    "live": run.live_count,
                    "missing": run.missing_count,
                    "recorded_at": now.isoformat(),
                }
            },
        )

    async def _record_clean(self, session: AsyncSession, connector: str, now: datetime) -> None:
        await self._merge_metadata(
            session,
            connector,
            {"last_clean_reconcile_at": now.isoformat(), "proposed_deletion": None},
        )

    async def _merge_metadata(
        self, session: AsyncSession, connector: str, updates: dict[str, Any]
    ) -> None:
        row = await self._reconciliation_connector(session, connector)
        merged = dict(cast("Any", row.run_metadata) or {})
        for key, value in updates.items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = value
        row.run_metadata = cast("Any", merged)

    async def _reconciliation_connector(
        self, session: AsyncSession, connector: str
    ) -> models.Connector:
        connector_id = f"{self._workspace_id}:{connector}"
        await session.execute(
            sqlite_insert(models.Connector)
            .values(
                id=connector_id,
                workspace_id=self._workspace_id,
                name=connector,
                type=connector,
                config={},
            )
            .on_conflict_do_nothing(index_elements=[models.Connector.id])
        )
        row = (
            await session.execute(
                select(models.Connector).where(
                    models.Connector.id == connector_id,
                    models.Connector.workspace_id == self._workspace_id,
                    models.Connector.name == connector,
                    models.Connector.deleted_at.is_(None),
                )
            )
        ).scalar_one_or_none()
        if row is None:
            msg = f"connector {connector!r} is unavailable in this workspace"
            raise ReconciliationInventoryError(msg)
        return row


__all__ = [
    "INVENTORY_PAGE_LIMIT",
    "ReconciliationInventoryError",
    "ReconciliationJournalMixin",
]
