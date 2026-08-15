"""SQLite implementation of the durable acquisition journal."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, cast

from pydantic import TypeAdapter
from sqlalchemy import func, or_, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from manicule.core.acquisition import (
    UNSET,
    AcquiredSource,
    AcquisitionDiagnostic,
    AcquisitionRecord,
    AcquisitionRecordState,
    AcquisitionRun,
    AcquisitionRunState,
    AcquisitionSource,
    UnsetValue,
)
from manicule.core.errors import ManiculeError, UnknownEntityError
from manicule.core.sources import Watermark
from manicule.storage import models
from manicule.storage.scoped import WorkspaceScoped
from manicule.storage.types import utcnow

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from sqlalchemy import CursorResult
    from sqlalchemy.ext.asyncio import AsyncSession

_WATERMARK = TypeAdapter(Watermark)
_SOURCE = TypeAdapter(AcquisitionSource)
_ACQUIRED_SOURCE = TypeAdapter(AcquiredSource)
_DIAGNOSTIC = TypeAdapter(AcquisitionDiagnostic)

_RUN_TRANSITIONS: dict[AcquisitionRunState, set[AcquisitionRunState]] = {
    AcquisitionRunState.ENUMERATING: set(),
    AcquisitionRunState.ACQUIRING: {
        AcquisitionRunState.INDEXING,
        AcquisitionRunState.SETTLED,
    },
    AcquisitionRunState.INDEXING: {AcquisitionRunState.SETTLED},
    AcquisitionRunState.SETTLED: set(),
}

_RECORD_TRANSITIONS: dict[AcquisitionRecordState, set[AcquisitionRecordState]] = {
    AcquisitionRecordState.DISCOVERED: {
        AcquisitionRecordState.ACQUIRING,
        AcquisitionRecordState.UNCHANGED,
        AcquisitionRecordState.RETRY,
    },
    AcquisitionRecordState.ACQUIRING: {
        AcquisitionRecordState.ACQUIRED,
        AcquisitionRecordState.UNCHANGED,
        AcquisitionRecordState.RETRY,
        AcquisitionRecordState.SETTLED,
    },
    AcquisitionRecordState.ACQUIRED: {
        AcquisitionRecordState.INDEXING,
        AcquisitionRecordState.RETRY,
    },
    AcquisitionRecordState.UNCHANGED: set(),
    AcquisitionRecordState.INDEXING: {
        AcquisitionRecordState.SETTLED,
        AcquisitionRecordState.RETRY,
    },
    AcquisitionRecordState.RETRY: {
        AcquisitionRecordState.ACQUIRING,
        AcquisitionRecordState.INDEXING,
        AcquisitionRecordState.SETTLED,
    },
    AcquisitionRecordState.SETTLED: set(),
}


class AcquisitionConflictError(ManiculeError):
    """A compare-and-swap or idempotency guard did not match."""


class InvalidAcquisitionTransitionError(ManiculeError):
    """A caller requested a lifecycle edge the state machine does not contain."""


class AcquisitionCoverageError(ManiculeError):
    """A candidate watermark cannot yet represent durable source coverage."""


class AcquisitionWatermarkConflictError(AcquisitionConflictError):
    """The connector watermark moved after this run began."""


def _record_id(run_id: str, source_id: str) -> str:
    payload = f"{len(run_id)}:{run_id}{source_id}".encode()
    return hashlib.blake2b(payload, digest_size=20).hexdigest()


def _run(row: models.AcquisitionRun) -> AcquisitionRun:
    return AcquisitionRun(
        id=row.id,
        workspace_id=row.workspace_id,
        connector_id=row.connector_id,
        connector=row.connector_name,
        state=row.state,
        base_watermark=(
            None if row.base_watermark is None else _WATERMARK.validate_python(row.base_watermark)
        ),
        candidate_watermark=(
            None
            if row.candidate_watermark is None
            else _WATERMARK.validate_python(row.candidate_watermark)
        ),
        enumeration_completed_at=row.enumeration_completed_at,
        watermark_committed_at=row.watermark_committed_at,
        lease_owner=row.lease_owner,
        lease_generation=row.lease_generation,
        lease_expires_at=row.lease_expires_at,
        discovered_count=row.discovered_count,
        acquired_count=row.acquired_count,
        indexed_count=row.indexed_count,
        unchanged_count=row.unchanged_count,
        retry_count=row.retry_count,
        metadata_bytes=row.metadata_bytes,
        acquired_blob_bytes=row.acquired_blob_bytes,
        diagnostic=(
            None if row.diagnostic is None else _DIAGNOSTIC.validate_python(row.diagnostic)
        ),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _record(row: models.AcquisitionRecord) -> AcquisitionRecord:
    return AcquisitionRecord(
        run_id=row.run_id,
        sequence=row.sequence,
        source=_SOURCE.validate_python(row.source_record),
        state=row.state,
        blob_ref=row.blob_ref,
        acquired_source=(
            None
            if row.acquired_source is None
            else _ACQUIRED_SOURCE.validate_python(row.acquired_source)
        ),
        fetched_version_token=row.fetched_version_token,
        attempts=row.attempts,
        diagnostic=(
            None if row.diagnostic is None else _DIAGNOSTIC.validate_python(row.diagnostic)
        ),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class AcquisitionJournalMixin(WorkspaceScoped):
    """Workspace-scoped durable run and record operations."""

    async def create_acquisition_run(self, run_id: str, connector: str) -> AcquisitionRun:
        """Create an immutable run identity, idempotently for the same connector."""
        if not run_id or not connector:
            msg = "run_id and connector must not be empty"
            raise ValueError(msg)
        async with self._sessions.begin() as session:
            connector_row = await self._ensure_connector(session, connector)
            statement = (
                sqlite_insert(models.AcquisitionRun)
                .values(
                    id=run_id,
                    workspace_id=self._workspace_id,
                    connector_id=connector_row.id,
                    connector_name=connector,
                    state=AcquisitionRunState.ENUMERATING,
                    base_watermark=connector_row.watermark,
                    created_at=utcnow(),
                    updated_at=utcnow(),
                )
                .on_conflict_do_nothing(index_elements=[models.AcquisitionRun.id])
            )
            await session.execute(statement)
            row = await session.get(models.AcquisitionRun, run_id)
            if (
                row is None
                or row.workspace_id != self._workspace_id
                or row.connector_id != connector_row.id
            ):
                msg = (
                    f"acquisition run {run_id!r} already belongs to another connector or workspace"
                )
                raise AcquisitionConflictError(msg)
            return _run(row)

    async def get_acquisition_run(self, run_id: str) -> AcquisitionRun | None:
        async with self._sessions() as session:
            row = await self._run_row(session, run_id)
            return None if row is None else _run(row)

    async def latest_unsettled_acquisition_run(self, connector: str) -> AcquisitionRun | None:
        async with self._sessions() as session:
            row = (
                await session.execute(
                    select(models.AcquisitionRun)
                    .where(
                        models.AcquisitionRun.workspace_id == self._workspace_id,
                        models.AcquisitionRun.connector_name == connector,
                        models.AcquisitionRun.state != AcquisitionRunState.SETTLED,
                    )
                    .order_by(
                        models.AcquisitionRun.created_at.desc(), models.AcquisitionRun.id.desc()
                    )
                    .limit(1)
                )
            ).scalar_one_or_none()
            return None if row is None else _run(row)

    async def append_acquisition_record(
        self,
        run_id: str,
        sequence: int,
        source: AcquisitionSource,
        *,
        lease_owner: str,
        lease_generation: int,
        now: datetime,
    ) -> AcquisitionRecord:
        """Commit one source record before acknowledging it to discovery."""
        if sequence < 0:
            msg = "sequence must not be negative"
            raise ValueError(msg)
        source_json = source.model_dump(mode="json")
        encoded_bytes = len(json.dumps(source_json, sort_keys=True, separators=(",", ":")).encode())
        async with self._sessions.begin() as session:
            run = await self._required_run_row(session, run_id)
            self._require_live_lease(run, lease_owner, lease_generation, now)
            if (
                run.state is not AcquisitionRunState.ENUMERATING
                or run.enumeration_completed_at is not None
            ):
                msg = f"acquisition run {run_id!r} is no longer accepting discovery records"
                raise AcquisitionConflictError(msg)
            inserted = cast(
                "CursorResult[Any]",
                await session.execute(
                    sqlite_insert(models.AcquisitionRecord)
                    .values(
                        id=_record_id(run_id, source.source_id),
                        run_id=run.id,
                        workspace_id=run.workspace_id,
                        connector_id=run.connector_id,
                        sequence=sequence,
                        source_id=source.source_id,
                        source_record=source_json,
                        state=AcquisitionRecordState.DISCOVERED,
                        created_at=utcnow(),
                        updated_at=utcnow(),
                    )
                    .on_conflict_do_nothing(index_elements=["run_id", "source_id"])
                ),
            )
            row = (
                await session.execute(
                    select(models.AcquisitionRecord).where(
                        models.AcquisitionRecord.run_id == run_id,
                        models.AcquisitionRecord.source_id == source.source_id,
                    )
                )
            ).scalar_one()
            if cast("Any", row.source_record) != source_json:
                msg = f"source identity {source.source_id!r} was rediscovered with different data"
                raise AcquisitionConflictError(msg)
            if inserted.rowcount:
                run.discovered_count += 1
                run.metadata_bytes += encoded_bytes
                run.updated_at = utcnow()
            return _record(row)

    async def complete_acquisition_enumeration(
        self,
        run_id: str,
        candidate_watermark: Watermark | None,
        *,
        lease_owner: str,
        lease_generation: int,
        now: datetime,
    ) -> AcquisitionRun:
        """Persist the true-end marker and candidate watermark in one transaction."""
        completed_at = utcnow()
        candidate = (
            None if candidate_watermark is None else candidate_watermark.model_dump(mode="json")
        )
        async with self._sessions.begin() as session:
            result = cast(
                "CursorResult[Any]",
                await session.execute(
                    update(models.AcquisitionRun)
                    .where(
                        models.AcquisitionRun.id == run_id,
                        models.AcquisitionRun.workspace_id == self._workspace_id,
                        models.AcquisitionRun.state == AcquisitionRunState.ENUMERATING,
                        models.AcquisitionRun.enumeration_completed_at.is_(None),
                        models.AcquisitionRun.lease_owner == lease_owner,
                        models.AcquisitionRun.lease_generation == lease_generation,
                        models.AcquisitionRun.lease_expires_at > now,
                    )
                    .values(
                        state=AcquisitionRunState.ACQUIRING,
                        candidate_watermark=candidate,
                        enumeration_completed_at=completed_at,
                        updated_at=completed_at,
                    )
                ),
            )
            if result.rowcount != 1:
                row = await self._required_run_row(session, run_id)
                self._require_live_lease(row, lease_owner, lease_generation, now)
                if row.candidate_watermark != candidate or row.enumeration_completed_at is None:
                    msg = f"acquisition run {run_id!r} cannot complete enumeration from {row.state}"
                    raise AcquisitionConflictError(msg)
            return _run(await self._required_run_row(session, run_id))

    async def claim_acquisition_run(
        self, run_id: str, owner: str, *, now: datetime, expires_at: datetime
    ) -> AcquisitionRun | None:
        """Claim or take over an expired lease, incrementing its fencing generation."""
        if expires_at <= now:
            msg = "lease expiry must be after now"
            raise ValueError(msg)
        async with self._sessions.begin() as session:
            result = cast(
                "CursorResult[Any]",
                await session.execute(
                    update(models.AcquisitionRun)
                    .where(
                        models.AcquisitionRun.id == run_id,
                        models.AcquisitionRun.workspace_id == self._workspace_id,
                        models.AcquisitionRun.state != AcquisitionRunState.SETTLED,
                        or_(
                            models.AcquisitionRun.lease_owner.is_(None),
                            models.AcquisitionRun.lease_expires_at <= now,
                        ),
                    )
                    .values(
                        lease_owner=owner,
                        lease_generation=models.AcquisitionRun.lease_generation + 1,
                        lease_expires_at=expires_at,
                        updated_at=utcnow(),
                    )
                ),
            )
            if result.rowcount != 1:
                return None
            return _run(await self._required_run_row(session, run_id))

    async def renew_acquisition_lease(
        self,
        run_id: str,
        owner: str,
        generation: int,
        *,
        now: datetime,
        expires_at: datetime,
    ) -> bool:
        """Renew only the live lease named by its owner and fencing generation."""
        if expires_at <= now:
            msg = "lease expiry must be after now"
            raise ValueError(msg)
        async with self._sessions.begin() as session:
            result = cast(
                "CursorResult[Any]",
                await session.execute(
                    update(models.AcquisitionRun)
                    .where(
                        models.AcquisitionRun.id == run_id,
                        models.AcquisitionRun.workspace_id == self._workspace_id,
                        models.AcquisitionRun.lease_owner == owner,
                        models.AcquisitionRun.lease_generation == generation,
                        models.AcquisitionRun.lease_expires_at > now,
                        models.AcquisitionRun.state != AcquisitionRunState.SETTLED,
                    )
                    .values(lease_expires_at=expires_at, updated_at=utcnow())
                ),
            )
            return result.rowcount == 1

    async def release_acquisition_lease(
        self,
        run_id: str,
        owner: str,
        generation: int,
        *,
        now: datetime,
    ) -> bool:
        """Release an unfinished run only while its exact live generation is still owned."""
        async with self._sessions.begin() as session:
            result = cast(
                "CursorResult[Any]",
                await session.execute(
                    update(models.AcquisitionRun)
                    .where(
                        models.AcquisitionRun.id == run_id,
                        models.AcquisitionRun.workspace_id == self._workspace_id,
                        models.AcquisitionRun.lease_owner == owner,
                        models.AcquisitionRun.lease_generation == generation,
                        models.AcquisitionRun.lease_expires_at > now,
                        models.AcquisitionRun.state != AcquisitionRunState.SETTLED,
                    )
                    .values(lease_owner=None, lease_expires_at=None, updated_at=utcnow())
                ),
            )
            return result.rowcount == 1

    async def transition_acquisition_run(
        self,
        run_id: str,
        expected: AcquisitionRunState,
        target: AcquisitionRunState,
        *,
        lease_owner: str,
        lease_generation: int,
        now: datetime,
        diagnostic: AcquisitionDiagnostic | None = None,
    ) -> AcquisitionRun:
        """Apply one explicit forward lifecycle edge under a generation fence."""
        if target not in _RUN_TRANSITIONS[expected]:
            msg = f"invalid acquisition run transition: {expected} -> {target}"
            raise InvalidAcquisitionTransitionError(msg)
        async with self._sessions.begin() as session:
            if target is AcquisitionRunState.SETTLED:
                active = (
                    await session.execute(
                        select(func.count())
                        .select_from(models.AcquisitionRecord)
                        .where(
                            models.AcquisitionRecord.run_id == run_id,
                            models.AcquisitionRecord.state.in_(
                                (
                                    AcquisitionRecordState.DISCOVERED,
                                    AcquisitionRecordState.ACQUIRING,
                                    AcquisitionRecordState.ACQUIRED,
                                    AcquisitionRecordState.INDEXING,
                                    AcquisitionRecordState.RETRY,
                                )
                            ),
                        )
                    )
                ).scalar_one()
                if active:
                    msg = f"acquisition run {run_id!r} still has {active} active records"
                    raise AcquisitionConflictError(msg)
            result = cast(
                "CursorResult[Any]",
                await session.execute(
                    update(models.AcquisitionRun)
                    .where(
                        models.AcquisitionRun.id == run_id,
                        models.AcquisitionRun.workspace_id == self._workspace_id,
                        models.AcquisitionRun.state == expected,
                        models.AcquisitionRun.lease_owner == lease_owner,
                        models.AcquisitionRun.lease_generation == lease_generation,
                        models.AcquisitionRun.lease_expires_at > now,
                    )
                    .values(
                        state=target,
                        diagnostic=(
                            None if diagnostic is None else diagnostic.model_dump(mode="json")
                        ),
                        updated_at=utcnow(),
                    )
                ),
            )
            if result.rowcount != 1:
                msg = f"acquisition run {run_id!r} state or lease changed"
                raise AcquisitionConflictError(msg)
            return _run(await self._required_run_row(session, run_id))

    async def list_acquisition_records(
        self,
        run_id: str,
        *,
        states: Sequence[AcquisitionRecordState] | None = None,
        after_sequence: int | None = None,
        limit: int = 100,
    ) -> Sequence[AcquisitionRecord]:
        """Read a bounded sequence page without accumulating a run in memory."""
        if after_sequence is not None and after_sequence < 0:
            msg = "after_sequence must not be negative"
            raise ValueError(msg)
        async with self._sessions() as session:
            await self._required_run_row(session, run_id)
            statement = select(models.AcquisitionRecord).where(
                models.AcquisitionRecord.run_id == run_id,
                models.AcquisitionRecord.workspace_id == self._workspace_id,
            )
            if states:
                statement = statement.where(models.AcquisitionRecord.state.in_(states))
            if after_sequence is not None:
                statement = statement.where(models.AcquisitionRecord.sequence > after_sequence)
            rows = (
                await session.execute(
                    statement.order_by(models.AcquisitionRecord.sequence).limit(limit)
                )
            ).scalars()
            return [_record(row) for row in rows]

    async def transition_acquisition_record(  # noqa: PLR0912 - validates one atomic state edge
        self,
        run_id: str,
        source_id: str,
        expected: AcquisitionRecordState,
        target: AcquisitionRecordState,
        *,
        lease_owner: str,
        lease_generation: int,
        now: datetime,
        blob_ref: str | None = None,
        acquired_source: AcquiredSource | None = None,
        fetched_version_token: str | UnsetValue | None = UNSET,
        diagnostic: AcquisitionDiagnostic | None = None,
    ) -> AcquisitionRecord:
        """Move one record under both a state CAS and the owning run's lease fence."""
        if target not in _RECORD_TRANSITIONS[expected]:
            msg = f"invalid acquisition record transition: {expected} -> {target}"
            raise InvalidAcquisitionTransitionError(msg)
        if target is AcquisitionRecordState.ACQUIRED and blob_ref is None:
            msg = "an acquired record requires a retained blob reference"
            raise InvalidAcquisitionTransitionError(msg)
        if target is AcquisitionRecordState.ACQUIRED and acquired_source is None:
            msg = "an acquired record requires its complete retained source envelope"
            raise InvalidAcquisitionTransitionError(msg)
        if acquired_source is not None:
            if acquired_source.source_id != source_id:
                msg = "the acquired source identity does not match the journal record"
                raise InvalidAcquisitionTransitionError(msg)
            if blob_ref is not None and acquired_source.content_hash != blob_ref:
                msg = "the acquired source hash does not match its retained blob"
                raise InvalidAcquisitionTransitionError(msg)
        async with self._sessions.begin() as session:
            run = await self._required_run_row(session, run_id)
            self._require_live_lease(run, lease_owner, lease_generation, now)
            if run.state is AcquisitionRunState.SETTLED:
                msg = f"acquisition run {run_id!r} is settled"
                raise AcquisitionConflictError(msg)
            if target is AcquisitionRecordState.INDEXING and blob_ref is None:
                record = (
                    await session.execute(
                        select(models.AcquisitionRecord).where(
                            models.AcquisitionRecord.run_id == run_id,
                            models.AcquisitionRecord.workspace_id == self._workspace_id,
                            models.AcquisitionRecord.source_id == source_id,
                            models.AcquisitionRecord.state == expected,
                        )
                    )
                ).scalar_one_or_none()
                if record is not None and record.blob_ref is None:
                    msg = "an indexing record requires a retained blob reference"
                    raise InvalidAcquisitionTransitionError(msg)
            values: dict[str, object] = {
                "state": target,
                "diagnostic": None if diagnostic is None else diagnostic.model_dump(mode="json"),
                "updated_at": utcnow(),
            }
            if fetched_version_token is not UNSET:
                values["fetched_version_token"] = fetched_version_token
            if target is AcquisitionRecordState.ACQUIRING:
                values["attempts"] = models.AcquisitionRecord.attempts + 1
            if blob_ref is not None:
                values["blob_ref"] = blob_ref
            if acquired_source is not None:
                values["acquired_source"] = acquired_source.model_dump(mode="json")
            result = cast(
                "CursorResult[Any]",
                await session.execute(
                    update(models.AcquisitionRecord)
                    .where(
                        models.AcquisitionRecord.run_id == run_id,
                        models.AcquisitionRecord.workspace_id == self._workspace_id,
                        models.AcquisitionRecord.source_id == source_id,
                        models.AcquisitionRecord.state == expected,
                    )
                    .values(**values)
                ),
            )
            if result.rowcount != 1:
                msg = f"acquisition record {source_id!r} state changed"
                raise AcquisitionConflictError(msg)
            await self._refresh_counters(session, run)
            row = (
                await session.execute(
                    select(models.AcquisitionRecord).where(
                        models.AcquisitionRecord.run_id == run_id,
                        models.AcquisitionRecord.source_id == source_id,
                    )
                )
            ).scalar_one()
            return _record(row)

    async def commit_acquisition_watermark(
        self,
        run_id: str,
        *,
        lease_owner: str,
        lease_generation: int,
        now: datetime,
    ) -> bool:
        """Atomically publish the candidate only after every source record has coverage."""
        async with self._sessions.begin() as session:
            run = await self._required_run_row(session, run_id)
            self._require_live_lease(run, lease_owner, lease_generation, now)
            if run.watermark_committed_at is not None:
                return True
            if run.enumeration_completed_at is None:
                msg = "enumeration is incomplete"
                raise AcquisitionCoverageError(msg)
            if run.candidate_watermark is None:
                return False
            uncovered = (
                await session.execute(
                    select(func.count())
                    .select_from(models.AcquisitionRecord)
                    .where(
                        models.AcquisitionRecord.run_id == run_id,
                        ~or_(
                            models.AcquisitionRecord.state == AcquisitionRecordState.UNCHANGED,
                            models.AcquisitionRecord.blob_ref.is_not(None),
                        ),
                    )
                )
            ).scalar_one()
            if uncovered:
                msg = f"{uncovered} acquisition records do not have durable source coverage"
                raise AcquisitionCoverageError(msg)
            base_matches = (
                models.Connector.watermark.is_(None)
                if run.base_watermark is None
                else models.Connector.watermark == run.base_watermark
            )
            committed_at = utcnow()
            result = cast(
                "CursorResult[Any]",
                await session.execute(
                    update(models.Connector)
                    .where(
                        models.Connector.id == run.connector_id,
                        models.Connector.workspace_id == self._workspace_id,
                        base_matches,
                    )
                    .values(watermark=run.candidate_watermark, last_synced_at=committed_at)
                ),
            )
            if result.rowcount != 1:
                msg = "connector watermark changed after this acquisition run began"
                raise AcquisitionWatermarkConflictError(msg)
            marker = cast(
                "CursorResult[Any]",
                await session.execute(
                    update(models.AcquisitionRun)
                    .where(
                        models.AcquisitionRun.id == run_id,
                        models.AcquisitionRun.workspace_id == self._workspace_id,
                        models.AcquisitionRun.watermark_committed_at.is_(None),
                    )
                    .values(watermark_committed_at=committed_at, updated_at=committed_at)
                ),
            )
            if marker.rowcount != 1:  # pragma: no cover - connector update holds the write lock
                msg = "acquisition watermark commit raced another writer"
                raise AcquisitionConflictError(msg)
            return True

    async def _ensure_connector(self, session: AsyncSession, connector: str) -> models.Connector:
        row = (
            await session.execute(
                select(models.Connector).where(
                    models.Connector.workspace_id == self._workspace_id,
                    models.Connector.name == connector,
                    models.Connector.deleted_at.is_(None),
                )
            )
        ).scalar_one_or_none()
        if row is None:
            row = models.Connector(
                id=f"{self._workspace_id}:{connector}",
                workspace_id=self._workspace_id,
                name=connector,
                type=connector,
                config={},
            )
            session.add(row)
            await session.flush()
        return row

    async def _run_row(self, session: AsyncSession, run_id: str) -> models.AcquisitionRun | None:
        return (
            await session.execute(
                select(models.AcquisitionRun).where(
                    models.AcquisitionRun.id == run_id,
                    models.AcquisitionRun.workspace_id == self._workspace_id,
                )
            )
        ).scalar_one_or_none()

    async def _required_run_row(self, session: AsyncSession, run_id: str) -> models.AcquisitionRun:
        row = await self._run_row(session, run_id)
        if row is None:
            msg = f"acquisition run {run_id!r} does not exist"
            raise UnknownEntityError(msg)
        return row

    async def _refresh_counters(self, session: AsyncSession, run: models.AcquisitionRun) -> None:
        rows = (
            await session.execute(
                select(models.AcquisitionRecord.state, func.count())
                .where(models.AcquisitionRecord.run_id == run.id)
                .group_by(models.AcquisitionRecord.state)
            )
        ).all()
        counts: dict[AcquisitionRecordState, int] = {}
        for state, count in rows:
            counts[state] = count
        run.acquired_count = sum(
            counts.get(state, 0)
            for state in (
                AcquisitionRecordState.ACQUIRED,
                AcquisitionRecordState.INDEXING,
                AcquisitionRecordState.SETTLED,
            )
        )
        run.indexed_count = (
            await session.execute(
                select(func.count())
                .select_from(models.AcquisitionRecord)
                .where(
                    models.AcquisitionRecord.run_id == run.id,
                    models.AcquisitionRecord.state == AcquisitionRecordState.SETTLED,
                    models.AcquisitionRecord.blob_ref.is_not(None),
                )
            )
        ).scalar_one()
        run.unchanged_count = counts.get(AcquisitionRecordState.UNCHANGED, 0)
        run.retry_count = counts.get(AcquisitionRecordState.RETRY, 0)
        run.acquired_blob_bytes = (
            await session.execute(
                select(func.coalesce(func.sum(models.Blob.size_bytes), 0))
                .select_from(models.AcquisitionRecord)
                .join(models.Blob, models.Blob.hash == models.AcquisitionRecord.blob_ref)
                .where(models.AcquisitionRecord.run_id == run.id)
            )
        ).scalar_one()
        run.updated_at = utcnow()

    @staticmethod
    def _require_live_lease(
        run: models.AcquisitionRun,
        owner: str,
        generation: int,
        now: datetime,
    ) -> None:
        if (
            run.lease_owner != owner
            or run.lease_generation != generation
            or run.lease_expires_at is None
            or run.lease_expires_at <= now
        ):
            msg = f"acquisition run {run.id!r} lease changed or expired"
            raise AcquisitionConflictError(msg)


__all__ = [
    "AcquisitionConflictError",
    "AcquisitionCoverageError",
    "AcquisitionJournalMixin",
    "AcquisitionWatermarkConflictError",
    "InvalidAcquisitionTransitionError",
]
