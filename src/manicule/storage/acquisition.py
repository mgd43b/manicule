"""SQLite implementation of the durable acquisition journal."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import TYPE_CHECKING, Any, cast

from pydantic import TypeAdapter
from sqlalchemy import and_, case, delete, exists, func, or_, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from manicule.core.acquisition import (
    UNSET,
    AcquiredSource,
    AcquisitionDiagnostic,
    AcquisitionFailureCode,
    AcquisitionFence,
    AcquisitionRecord,
    AcquisitionRecordState,
    AcquisitionRun,
    AcquisitionRunState,
    AcquisitionSource,
    AcquisitionStage,
    SnapshotCompleteness,
    SnapshotItemOutcome,
    SnapshotPromotionPolicy,
    UnsetValue,
)
from manicule.core.content import JsonValue
from manicule.core.errors import AcquisitionLeaseLostError, ManiculeError, UnknownEntityError
from manicule.core.ids import acquisition_marker_id
from manicule.core.sources import Watermark
from manicule.storage import models
from manicule.storage.scoped import WorkspaceScoped
from manicule.storage.types import next_observation, utcnow

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime

    from sqlalchemy import CursorResult
    from sqlalchemy.ext.asyncio import AsyncSession

_WATERMARK = TypeAdapter(Watermark)
_SOURCE = TypeAdapter(AcquisitionSource)
_ACQUIRED_SOURCE = TypeAdapter(AcquiredSource)
_DIAGNOSTIC = TypeAdapter(AcquisitionDiagnostic)
_OMISSION_REASONS = TypeAdapter(dict[AcquisitionFailureCode, int])

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
        AcquisitionRecordState.OMITTED,
    },
    AcquisitionRecordState.SETTLED: set(),
    AcquisitionRecordState.OMITTED: set(),
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
        source_scope=row.source_scope,
        scope_fingerprint=row.scope_fingerprint,
        promotion_policy=row.promotion_policy,
        state=row.state,
        base_watermark=(
            None if row.base_watermark is None else _WATERMARK.validate_python(row.base_watermark)
        ),
        base_watermark_scope_fingerprint=row.base_watermark_scope_fingerprint,
        candidate_watermark=(
            None
            if row.candidate_watermark is None
            else _WATERMARK.validate_python(row.candidate_watermark)
        ),
        enumeration_completed_at=row.enumeration_completed_at,
        acquisition_completed_at=row.acquisition_completed_at,
        promoted_at=row.promoted_at,
        watermark_committed_at=row.watermark_committed_at,
        superseded_at=row.superseded_at,
        superseded_by=row.superseded_by,
        membership_hash=row.membership_hash,
        completeness=row.completeness,
        omission_count=row.omission_count,
        omission_reasons=_OMISSION_REASONS.validate_python(row.omission_reasons),
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
        diagnostic=_safe_diagnostic(row.diagnostic),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _record(row: models.AcquisitionRecord) -> AcquisitionRecord:
    return AcquisitionRecord(
        run_id=row.run_id,
        sequence=row.sequence,
        source=_SOURCE.validate_python(row.source_record),
        state=row.state,
        snapshot_outcome=(
            None if row.snapshot_outcome is None else SnapshotItemOutcome(row.snapshot_outcome)
        ),
        blob_ref=row.blob_ref,
        acquired_source=(
            None
            if row.acquired_source is None
            else _ACQUIRED_SOURCE.validate_python(row.acquired_source)
        ),
        fetched_version_token=row.fetched_version_token,
        attempts=row.attempts,
        snapshot_diagnostic=_safe_snapshot_diagnostic(row.snapshot_diagnostic),
        diagnostic=_safe_diagnostic(row.diagnostic),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _safe_snapshot_diagnostic(raw: object) -> AcquisitionDiagnostic | None:
    """Read frozen evidence without ever rendering corrupt legacy or substituted input."""
    if raw is None:
        return None
    diagnostic = _safe_diagnostic(raw)
    if diagnostic is not None and diagnostic.stage is AcquisitionStage.ACQUISITION:
        return diagnostic
    return AcquisitionDiagnostic(
        stage=AcquisitionStage.ACQUISITION,
        code=AcquisitionFailureCode.LEGACY_UNVERIFIED,
    )


def _safe_diagnostic(raw: object) -> AcquisitionDiagnostic | None:
    """Normalize corrupt persisted diagnostics to one bounded, non-sensitive envelope."""
    if raw is None:
        return None
    try:
        return _DIAGNOSTIC.validate_python(raw)
    except ValueError:
        return AcquisitionDiagnostic(
            stage=AcquisitionStage.ACQUISITION,
            code=AcquisitionFailureCode.LEGACY_UNVERIFIED,
        )


def _canonical_snapshot_diagnostic(raw: object, *, missing_evidence: bool) -> JsonValue | None:
    """Canonical frozen omission evidence, never a copy of untrusted persisted JSON."""
    if not missing_evidence:
        return None
    diagnostic = _safe_snapshot_diagnostic(raw) or AcquisitionDiagnostic(
        stage=AcquisitionStage.ACQUISITION,
        code=AcquisitionFailureCode.LEGACY_UNVERIFIED,
    )
    return cast("JsonValue", diagnostic.model_dump(mode="json"))


def _same_canonical_json(raw: object, canonical: object) -> bool:
    """Compare JSON shapes without Python's ``1 == True`` coercion."""
    try:
        return json.dumps(raw, sort_keys=True, separators=(",", ":")) == json.dumps(
            canonical, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError):
        return False


def _manifest_member(row: models.AcquisitionRecord) -> dict[str, object]:
    """Canonical immutable source evidence; derivation state is deliberately excluded."""
    missing_evidence = row.blob_ref is None or row.acquired_source is None
    canonical_diagnostic = _canonical_snapshot_diagnostic(
        row.snapshot_diagnostic, missing_evidence=missing_evidence
    )
    if not _same_canonical_json(row.snapshot_diagnostic, canonical_diagnostic):
        msg = "snapshot diagnostic evidence is not in canonical typed form"
        raise AcquisitionConflictError(msg)
    return {
        "sequence": row.sequence,
        "source_id": row.source_id,
        "source": row.source_record,
        "fetched_version_token": row.fetched_version_token,
        "blob_ref": row.blob_ref,
        "acquired_source": row.acquired_source,
        "snapshot_outcome": (None if row.snapshot_outcome is None else str(row.snapshot_outcome)),
        "snapshot_diagnostic": canonical_diagnostic,
    }


async def _manifest_digest(session: AsyncSession, run_id: str) -> str:
    """Hash a manifest in bounded keyset pages after every acquisition outcome is final."""
    digest = hashlib.blake2b(digest_size=32)
    after = -1
    while True:
        rows = (
            (
                await session.execute(
                    select(models.AcquisitionRecord)
                    .where(
                        models.AcquisitionRecord.run_id == run_id,
                        models.AcquisitionRecord.sequence > after,
                    )
                    .order_by(models.AcquisitionRecord.sequence)
                    .limit(100)
                )
            )
            .scalars()
            .all()
        )
        if not rows:
            return digest.hexdigest()
        for record in rows:
            digest.update(
                json.dumps(_manifest_member(record), sort_keys=True, separators=(",", ":")).encode()
            )
            digest.update(b"\n")
        after = rows[-1].sequence


async def _canonicalize_snapshot_diagnostics(session: AsyncSession, run_id: str) -> None:
    """Overwrite every frozen diagnostic from its retained-evidence rule, in bounded pages."""
    after = -1
    while True:
        rows = (
            (
                await session.execute(
                    select(models.AcquisitionRecord)
                    .where(
                        models.AcquisitionRecord.run_id == run_id,
                        models.AcquisitionRecord.sequence > after,
                    )
                    .order_by(models.AcquisitionRecord.sequence)
                    .limit(100)
                )
            )
            .scalars()
            .all()
        )
        if not rows:
            await session.flush()
            return
        for record in rows:
            missing_evidence = record.blob_ref is None or record.acquired_source is None
            record.snapshot_diagnostic = cast(
                "Any",
                _canonical_snapshot_diagnostic(
                    record.snapshot_diagnostic, missing_evidence=missing_evidence
                ),
            )
        after = rows[-1].sequence


async def _manifest_matches(session: AsyncSession, run_id: str, expected: str) -> bool:
    """Verify without leaking malformed evidence through a read-only lookup."""
    try:
        actual = await _manifest_digest(session, run_id)
    except AcquisitionConflictError:
        return False
    return hmac.compare_digest(expected, actual)


class AcquisitionJournalMixin(WorkspaceScoped):
    """Workspace-scoped durable run and record operations."""

    async def create_acquisition_run(
        self,
        run_id: str,
        connector: str,
        *,
        source_scope: str = "",
        scope_fingerprint: str = "",
        promotion_policy: SnapshotPromotionPolicy = SnapshotPromotionPolicy.REQUIRE_COMPLETE,
    ) -> AcquisitionRun:
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
                    source_scope=source_scope,
                    scope_fingerprint=scope_fingerprint,
                    promotion_policy=promotion_policy,
                    state=AcquisitionRunState.ENUMERATING,
                    base_watermark=connector_row.watermark,
                    base_watermark_scope_fingerprint=connector_row.watermark_scope_fingerprint,
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
                or row.source_scope != source_scope
                or row.scope_fingerprint != scope_fingerprint
                or SnapshotPromotionPolicy(row.promotion_policy) is not promotion_policy
            ):
                msg = f"acquisition run {run_id!r} conflicts with the requested run identity"
                raise AcquisitionConflictError(msg)
            return _run(row)

    async def get_acquisition_watermark(
        self, connector: str, scope_fingerprint: str
    ) -> Watermark | None:
        """Return a cursor only when it was committed for this exact source scope."""
        async with self._sessions() as session:
            row = (
                await session.execute(
                    select(models.Connector).where(
                        models.Connector.workspace_id == self._workspace_id,
                        models.Connector.name == connector,
                        models.Connector.deleted_at.is_(None),
                    )
                )
            ).scalar_one_or_none()
            if (
                row is None
                or row.watermark is None
                or row.watermark_scope_fingerprint != scope_fingerprint
            ):
                return None
            return _WATERMARK.validate_python(row.watermark)

    async def latest_promoted_snapshot(
        self, connector: str, scope_fingerprint: str
    ) -> AcquisitionRun | None:
        """Return the newest authoritative manifest for this exact connector scope."""
        async with self._sessions() as session:
            row = (
                await session.execute(
                    select(models.AcquisitionRun)
                    .where(
                        models.AcquisitionRun.workspace_id == self._workspace_id,
                        models.AcquisitionRun.connector_name == connector,
                        models.AcquisitionRun.superseded_at.is_(None),
                        models.AcquisitionRun.scope_fingerprint == scope_fingerprint,
                        models.AcquisitionRun.promoted_at.is_not(None),
                    )
                    .order_by(
                        models.AcquisitionRun.promoted_at.desc(),
                        models.AcquisitionRun.id.desc(),
                    )
                    .limit(1)
                )
            ).scalar_one_or_none()
            if (
                row is None
                or not row.membership_hash
                or not await _manifest_matches(session, row.id, row.membership_hash)
            ):
                return None
            return _run(row)

    async def reusable_snapshot_record(
        self,
        connector: str,
        scope_fingerprint: str,
        source_id: str,
        version_token: str | None,
    ) -> AcquisitionRecord | None:
        """Find validated retained bytes for an unchanged revision in the same exact scope."""
        async with self._sessions() as session:
            run = (
                await session.execute(
                    select(models.AcquisitionRun)
                    .where(
                        models.AcquisitionRun.workspace_id == self._workspace_id,
                        models.AcquisitionRun.connector_name == connector,
                        models.AcquisitionRun.scope_fingerprint == scope_fingerprint,
                        models.AcquisitionRun.promoted_at.is_not(None),
                    )
                    .order_by(
                        models.AcquisitionRun.promoted_at.desc(),
                        models.AcquisitionRun.id.desc(),
                    )
                    .limit(1)
                )
            ).scalar_one_or_none()
            if (
                run is None
                or not run.membership_hash
                or not await _manifest_matches(session, run.id, run.membership_hash)
            ):
                return None

            row = (
                await session.execute(
                    select(models.AcquisitionRecord)
                    .where(
                        models.AcquisitionRecord.run_id == run.id,
                        models.AcquisitionRecord.source_id == source_id,
                        models.AcquisitionRecord.blob_ref.is_not(None),
                        models.AcquisitionRecord.acquired_source.is_not(None),
                        or_(
                            models.AcquisitionRecord.fetched_version_token == version_token,
                            models.AcquisitionRecord.source_record["version_token"].as_string()
                            == version_token,
                        ),
                    )
                    .limit(1)
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            return _record(row)

    async def reusable_record_from_verified_snapshot(
        self,
        run_id: str,
        source_id: str,
        version_token: str | None,
    ) -> AcquisitionRecord | None:
        """Look up retained evidence after this operation verified ``run_id`` once."""
        async with self._sessions() as session:
            run = await self._run_row(session, run_id)
            if (
                run is None
                or run.workspace_id != self._workspace_id
                or run.promoted_at is None
                or not run.membership_hash
            ):
                return None
            row = (
                await session.execute(
                    select(models.AcquisitionRecord)
                    .where(
                        models.AcquisitionRecord.run_id == run.id,
                        models.AcquisitionRecord.source_id == source_id,
                        models.AcquisitionRecord.blob_ref.is_not(None),
                        models.AcquisitionRecord.acquired_source.is_not(None),
                        or_(
                            models.AcquisitionRecord.fetched_version_token == version_token,
                            models.AcquisitionRecord.source_record["version_token"].as_string()
                            == version_token,
                        ),
                    )
                    .limit(1)
                )
            ).scalar_one_or_none()
            return None if row is None else _record(row)

    async def verify_snapshot_manifest(self, run_id: str) -> bool:
        """Verify the canonical evidence digest without loading an unbounded manifest."""
        async with self._sessions() as session:
            run = await self._required_run_row(session, run_id)
            if run.acquisition_completed_at is None or not run.membership_hash:
                return False
            return await _manifest_matches(session, run_id, run.membership_hash)

    async def get_acquisition_run(self, run_id: str) -> AcquisitionRun | None:
        async with self._sessions() as session:
            row = await self._run_row(session, run_id)
            return None if row is None else _run(row)

    async def latest_unsettled_acquisition_run(
        self, connector: str, *, scope_fingerprint: str | None = None
    ) -> AcquisitionRun | None:
        async with self._sessions() as session:
            statement = select(models.AcquisitionRun).where(
                models.AcquisitionRun.workspace_id == self._workspace_id,
                models.AcquisitionRun.connector_name == connector,
                models.AcquisitionRun.state != AcquisitionRunState.SETTLED,
                models.AcquisitionRun.superseded_at.is_(None),
            )
            if scope_fingerprint is not None:
                statement = statement.where(
                    models.AcquisitionRun.scope_fingerprint == scope_fingerprint
                )
            row = (
                await session.execute(
                    statement.order_by(
                        models.AcquisitionRun.created_at.desc(), models.AcquisitionRun.id.desc()
                    ).limit(1)
                )
            ).scalar_one_or_none()
            return None if row is None else _run(row)

    async def claim_or_create_acquisition_run(
        self,
        connector: str,
        run_id: str,
        owner: str,
        *,
        source_scope: str = "",
        scope_fingerprint: str = "",
        promotion_policy: SnapshotPromotionPolicy = SnapshotPromotionPolicy.REQUIRE_COMPLETE,
        now: datetime,
        expires_at: datetime,
    ) -> AcquisitionRun | None:
        """Select recovery work and acquire it in the same SQLite write transaction.

        The idempotent connector write is the serialization point. Without it two processes can
        both observe no unfinished run and create two enumerations from the same watermark.
        Keeping selection, creation and claim behind one write lock makes a repeated sync either
        the owner of the newest durable run or a clean loser.
        """
        if not connector or not run_id or not owner:
            msg = "connector, run_id and owner must not be empty"
            raise ValueError(msg)
        if expires_at <= now:
            msg = "lease expiry must be after now"
            raise ValueError(msg)
        async with self._sessions.begin() as session:
            connector_id = f"{self._workspace_id}:{connector}"
            # The insert is also the serialization point when this is the connector's first
            # ever sync. A read followed by `_ensure_connector` would let both callers observe
            # absence before either held SQLite's write lock.
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
            connector_row = (
                await session.execute(
                    select(models.Connector).where(
                        models.Connector.id == connector_id,
                        models.Connector.workspace_id == self._workspace_id,
                        models.Connector.name == connector,
                        models.Connector.deleted_at.is_(None),
                    )
                )
            ).scalar_one_or_none()
            if connector_row is None:
                msg = f"connector {connector!r} is unavailable"
                raise AcquisitionConflictError(msg)
            candidates = (
                (
                    await session.execute(
                        select(models.AcquisitionRun)
                        .where(
                            models.AcquisitionRun.workspace_id == self._workspace_id,
                            models.AcquisitionRun.connector_id == connector_row.id,
                            models.AcquisitionRun.state != AcquisitionRunState.SETTLED,
                            models.AcquisitionRun.superseded_at.is_(None),
                        )
                        .order_by(
                            models.AcquisitionRun.created_at.desc(), models.AcquisitionRun.id.desc()
                        )
                    )
                )
                .scalars()
                .all()
            )
            safe: list[models.AcquisitionRun] = []
            for candidate in candidates:
                same_scope = candidate.scope_fingerprint == scope_fingerprint
                base_is_current = (
                    same_scope
                    and candidate.base_watermark == connector_row.watermark
                    and candidate.base_watermark_scope_fingerprint
                    == connector_row.watermark_scope_fingerprint
                )
                committed_is_current = (
                    same_scope
                    and candidate.scope_fingerprint == connector_row.watermark_scope_fingerprint
                    and candidate.watermark_committed_at is not None
                    and candidate.candidate_watermark == connector_row.watermark
                )
                if base_is_current or committed_is_current:
                    safe.append(candidate)
            # Ordering above makes the first safe run authoritative. Every other overlap is
            # fenced in this same writer transaction, including an older live owner with the
            # same base watermark. Leaving it active would preserve the duplicate-run race this
            # API exists to reconcile on upgraded databases.
            row = safe[0] if safe else None
            superseded = [candidate for candidate in candidates if candidate is not row]
            for candidate in superseded:
                candidate.superseded_at = utcnow()
                candidate.lease_owner = None
                candidate.lease_expires_at = None
                candidate.lease_generation += 1
                candidate.updated_at = utcnow()
            if row is None:
                row = models.AcquisitionRun(
                    id=run_id,
                    workspace_id=self._workspace_id,
                    connector_id=connector_row.id,
                    connector_name=connector,
                    source_scope=source_scope,
                    scope_fingerprint=scope_fingerprint,
                    promotion_policy=promotion_policy,
                    state=AcquisitionRunState.ENUMERATING,
                    base_watermark=connector_row.watermark,
                    base_watermark_scope_fingerprint=connector_row.watermark_scope_fingerprint,
                    lease_owner=owner,
                    lease_generation=1,
                    lease_expires_at=expires_at,
                    created_at=utcnow(),
                    updated_at=utcnow(),
                )
                session.add(row)
                for obsolete in superseded:
                    obsolete.superseded_by = row.id
                await session.flush()
                return _run(row)
            for obsolete in superseded:
                obsolete.superseded_by = row.id
            if row.lease_owner is not None and (
                row.lease_expires_at is None or row.lease_expires_at > now
            ):
                return None
            row.lease_owner = owner
            row.lease_generation += 1
            row.lease_expires_at = expires_at
            row.updated_at = utcnow()
            # A source call interrupted before it committed has no durable outcome, so replay it
            # as retry work. INDEXING is already a precise recovery state and stays untouched:
            # container indexing uses it to force re-expansion of partially published members.
            interrupted = AcquisitionDiagnostic(
                stage=AcquisitionStage.ACQUISITION,
                code=AcquisitionFailureCode.INTERRUPTED,
            ).model_dump(mode="json")
            await session.execute(
                update(models.AcquisitionRecord)
                .where(
                    models.AcquisitionRecord.run_id == row.id,
                    models.AcquisitionRecord.workspace_id == self._workspace_id,
                    models.AcquisitionRecord.state == AcquisitionRecordState.ACQUIRING,
                )
                .values(
                    state=AcquisitionRecordState.RETRY,
                    diagnostic=interrupted,
                    updated_at=utcnow(),
                )
            )
            await self._refresh_counters(session, row)
            await session.flush()
            return _run(row)

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
                        marker_name=acquisition_marker_id(run_id, source.source_id),
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
                        models.AcquisitionRun.superseded_at.is_(None),
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
                        models.AcquisitionRun.superseded_at.is_(None),
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
                        models.AcquisitionRun.superseded_at.is_(None),
                    )
                    .values(lease_owner=None, lease_expires_at=None, updated_at=utcnow())
                ),
            )
            return result.rowcount == 1

    async def record_acquisition_run_metadata(
        self,
        run_id: str,
        owner: str,
        generation: int,
        *,
        now: datetime,
        updates: Mapping[str, Any],
        release: bool,
    ) -> bool:
        """Order connector diagnostics and optional release behind the run generation."""
        values: dict[str, object] = {
            "lease_generation": models.AcquisitionRun.lease_generation,
            "updated_at": utcnow(),
        }
        if release:
            values.update(lease_owner=None, lease_expires_at=None)
        async with self._sessions.begin() as session:
            matched = cast(
                "CursorResult[Any]",
                await session.execute(
                    update(models.AcquisitionRun)
                    .where(
                        models.AcquisitionRun.id == run_id,
                        models.AcquisitionRun.workspace_id == self._workspace_id,
                        models.AcquisitionRun.lease_owner == owner,
                        models.AcquisitionRun.lease_generation == generation,
                        models.AcquisitionRun.lease_expires_at > now,
                        models.AcquisitionRun.superseded_at.is_(None),
                    )
                    .values(**values)
                ),
            )
            if matched.rowcount != 1:
                return False
            run = await self._required_run_row(session, run_id)
            connector = await session.get(models.Connector, run.connector_id)
            if connector is None:  # pragma: no cover - acquisition FK guarantees it
                msg = f"connector for acquisition run {run_id!r} vanished"
                raise RuntimeError(msg)
            merged: dict[str, Any] = dict(cast("Any", connector.run_metadata) or {})
            for key, value in updates.items():
                if value is None:
                    merged.pop(key, None)
                else:
                    merged[key] = value
            connector.run_metadata = cast("Any", merged)
            return True

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
            values: dict[str, object] = {
                "state": target,
                "diagnostic": (None if diagnostic is None else diagnostic.model_dump(mode="json")),
                "updated_at": utcnow(),
            }
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
                    .values(**values)
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

    async def transition_acquisition_record(  # noqa: PLR0912, PLR0915 - one atomic state edge
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
            if run.acquisition_completed_at is not None and (
                blob_ref is not None
                or acquired_source is not None
                or fetched_version_token is not UNSET
            ):
                msg = "snapshot evidence is frozen after acquisition completion"
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
            if run.acquisition_completed_at is None:
                if target is AcquisitionRecordState.RETRY and expected in {
                    AcquisitionRecordState.DISCOVERED,
                    AcquisitionRecordState.ACQUIRING,
                }:
                    values["snapshot_diagnostic"] = (
                        None if diagnostic is None else diagnostic.model_dump(mode="json")
                    )
                elif target in {
                    AcquisitionRecordState.ACQUIRED,
                    AcquisitionRecordState.UNCHANGED,
                }:
                    values["snapshot_diagnostic"] = None
            if fetched_version_token is not UNSET:
                values["fetched_version_token"] = fetched_version_token
            if target is AcquisitionRecordState.ACQUIRING:
                values["attempts"] = models.AcquisitionRecord.attempts + 1
            if blob_ref is not None:
                values["blob_ref"] = blob_ref
            if acquired_source is not None:
                values["acquired_source"] = acquired_source.model_dump(mode="json")
            if target is AcquisitionRecordState.ACQUIRED:
                values["snapshot_outcome"] = SnapshotItemOutcome.RETAINED
            elif target is AcquisitionRecordState.UNCHANGED:
                values["snapshot_outcome"] = (
                    SnapshotItemOutcome.REUSED
                    if blob_ref is not None and acquired_source is not None
                    else SnapshotItemOutcome.OMITTED
                )
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

    async def settle_unchanged_acquisition_record(
        self,
        run_id: str,
        source_id: str,
        document_id: str,
        *,
        lease_owner: str,
        lease_generation: int,
        now: datetime,
        blob_ref: str | None = None,
        acquired_source: AcquiredSource | None = None,
        fetched_version_token: str | None = None,
    ) -> AcquisitionRecord:
        """Commit unchanged coverage and presence under one generation-fenced writer lock."""
        fence = AcquisitionFence(
            run_id=run_id,
            owner=lease_owner,
            generation=lease_generation,
            now=now,
        )
        async with self._sessions.begin() as session:
            await self._fence_acquisition_mutation(session, fence)
            values: dict[str, object] = {
                "state": AcquisitionRecordState.UNCHANGED,
                "fetched_version_token": fetched_version_token,
                "diagnostic": None,
                "updated_at": utcnow(),
            }
            if blob_ref is not None:
                values["blob_ref"] = blob_ref
            if acquired_source is not None:
                values["acquired_source"] = acquired_source.model_dump(mode="json")
            values["snapshot_outcome"] = (
                SnapshotItemOutcome.REUSED
                if blob_ref is not None and acquired_source is not None
                else SnapshotItemOutcome.OMITTED
            )
            transitioned = cast(
                "CursorResult[Any]",
                await session.execute(
                    update(models.AcquisitionRecord)
                    .where(
                        models.AcquisitionRecord.run_id == run_id,
                        models.AcquisitionRecord.workspace_id == self._workspace_id,
                        models.AcquisitionRecord.source_id == source_id,
                        models.AcquisitionRecord.state == AcquisitionRecordState.ACQUIRING,
                    )
                    .values(**values)
                ),
            )
            if transitioned.rowcount != 1:
                msg = f"acquisition record {source_id!r} state changed"
                raise AcquisitionConflictError(msg)
            document = (
                await session.execute(
                    select(models.Document).where(
                        models.Document.id == document_id,
                        models.Document.workspace_id == self._workspace_id,
                        models.Document.deleted_at.is_(None),
                    )
                )
            ).scalar_one_or_none()
            if document is None:
                msg = f"unchanged document {document_id!r} is no longer live"
                raise AcquisitionConflictError(msg)
            document.last_seen_at = next_observation(document.last_seen_at, utcnow())
            run = await self._required_run_row(session, run_id)
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

    async def complete_snapshot_acquisition(
        self,
        run_id: str,
        *,
        lease_owner: str,
        lease_generation: int,
        now: datetime,
    ) -> AcquisitionRun:
        """Freeze membership and the aggregate acquisition outcome under the run lease.

        This marker is deliberately separate from enumeration and promotion. A strict snapshot
        with one missing body remains resumable and unmarked; an omission-tolerant snapshot may
        freeze that same bounded, typed omission set before an atomic promotion decision.
        """
        async with self._sessions.begin() as session:
            run = await self._required_run_row(session, run_id)
            self._require_live_lease(run, lease_owner, lease_generation, now)
            if run.acquisition_completed_at is not None:
                return _run(run)
            if run.enumeration_completed_at is None:
                msg = "enumeration is incomplete"
                raise AcquisitionCoverageError(msg)
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
                            )
                        ),
                    )
                )
            ).scalar_one()
            if active:
                msg = f"{active} acquisition records are still in progress"
                raise AcquisitionCoverageError(msg)
            await _canonicalize_snapshot_diagnostics(session, run_id)
            omissions = (
                await session.execute(
                    select(func.count())
                    .select_from(models.AcquisitionRecord)
                    .where(
                        models.AcquisitionRecord.run_id == run_id,
                        or_(
                            models.AcquisitionRecord.blob_ref.is_(None),
                            models.AcquisitionRecord.acquired_source.is_(None),
                        ),
                    )
                )
            ).scalar_one()
            reason_rows = (
                await session.execute(
                    select(models.AcquisitionRecord.snapshot_diagnostic, func.count())
                    .where(
                        models.AcquisitionRecord.run_id == run_id,
                        or_(
                            models.AcquisitionRecord.blob_ref.is_(None),
                            models.AcquisitionRecord.acquired_source.is_(None),
                        ),
                    )
                    .group_by(models.AcquisitionRecord.snapshot_diagnostic)
                )
            ).all()
            reasons: dict[str, int] = {}
            for diagnostic, count in reason_rows:
                code = "unknown" if not diagnostic else str(diagnostic.get("code", "unknown"))
                reasons[code] = reasons.get(code, 0) + count

            run.omission_count = omissions
            run.omission_reasons = cast("Any", reasons)
            policy = SnapshotPromotionPolicy(run.promotion_policy)
            coverage_error = bool(omissions) and (
                policy is SnapshotPromotionPolicy.REQUIRE_COMPLETE
            )
            if not coverage_error:
                if policy is SnapshotPromotionPolicy.ALLOW_OMISSIONS:
                    missing_evidence = or_(
                        models.AcquisitionRecord.blob_ref.is_(None),
                        models.AcquisitionRecord.acquired_source.is_(None),
                    )
                    await session.execute(
                        update(models.AcquisitionRecord)
                        .where(models.AcquisitionRecord.run_id == run_id, missing_evidence)
                        .values(
                            snapshot_outcome=SnapshotItemOutcome.OMITTED,
                            state=case(
                                (
                                    models.AcquisitionRecord.state == AcquisitionRecordState.RETRY,
                                    AcquisitionRecordState.OMITTED,
                                ),
                                else_=models.AcquisitionRecord.state,
                            ),
                            updated_at=now,
                        )
                    )
                    await self._refresh_counters(session, run)
                run.acquisition_completed_at = now
                run.membership_hash = await _manifest_digest(session, run_id)
            run.updated_at = now
            result = _run(run)
        if coverage_error:
            msg = f"{omissions} required source records lack validated retained bytes"
            raise AcquisitionCoverageError(msg)
        return result

    async def promote_snapshot_and_commit_watermark(
        self,
        run_id: str,
        *,
        expected_scope_fingerprint: str,
        lease_owner: str,
        lease_generation: int,
        now: datetime,
    ) -> AcquisitionRun:
        """Atomically make the frozen manifest authoritative and publish its watermark."""
        async with self._sessions.begin() as session:
            run = await self._required_run_row(session, run_id)
            self._require_live_lease(run, lease_owner, lease_generation, now)
            if run.scope_fingerprint != expected_scope_fingerprint:
                msg = "snapshot scope fingerprint does not match the promotion request"
                raise AcquisitionConflictError(msg)
            if run.acquisition_completed_at is None:
                msg = "snapshot acquisition is incomplete"
                raise AcquisitionCoverageError(msg)
            if (
                SnapshotPromotionPolicy(run.promotion_policy)
                is SnapshotPromotionPolicy.REQUIRE_COMPLETE
                and run.omission_count
            ):
                msg = f"{run.omission_count} required source records were omitted"
                raise AcquisitionCoverageError(msg)

            if not run.membership_hash or not hmac.compare_digest(
                run.membership_hash, await _manifest_digest(session, run_id)
            ):
                msg = "snapshot manifest evidence no longer matches its acquisition digest"
                raise AcquisitionConflictError(msg)
            if run.promoted_at is not None:
                return _run(run)

            promoted_at = now

            committed_at: datetime | None = None
            if run.candidate_watermark is not None:
                base_matches = (
                    models.Connector.watermark.is_(None)
                    if run.base_watermark is None
                    else models.Connector.watermark == run.base_watermark
                )
                result = cast(
                    "CursorResult[Any]",
                    await session.execute(
                        update(models.Connector)
                        .where(
                            models.Connector.id == run.connector_id,
                            models.Connector.workspace_id == self._workspace_id,
                            models.Connector.watermark_scope_fingerprint
                            == run.base_watermark_scope_fingerprint,
                            base_matches,
                        )
                        .values(
                            watermark=run.candidate_watermark,
                            watermark_scope_fingerprint=run.scope_fingerprint,
                            last_synced_at=promoted_at,
                        )
                    ),
                )
                if result.rowcount != 1:
                    msg = "connector watermark changed after this snapshot enumeration began"
                    raise AcquisitionWatermarkConflictError(msg)
                committed_at = promoted_at

            run.promoted_at = promoted_at
            run.watermark_committed_at = committed_at
            run.completeness = (
                SnapshotCompleteness.PARTIAL
                if run.omission_count
                else SnapshotCompleteness.COMPLETE
            )
            run.updated_at = promoted_at
            return _run(run)

    async def commit_acquisition_watermark(
        self,
        run_id: str,
        *,
        lease_owner: str,
        lease_generation: int,
        now: datetime,
    ) -> bool:
        """Preserve the pre-promotion API for callers not constructing manifests."""
        return await self._legacy_commit_acquisition_watermark(
            run_id,
            lease_owner=lease_owner,
            lease_generation=lease_generation,
            now=now,
        )

    async def _legacy_commit_acquisition_watermark(
        self,
        run_id: str,
        *,
        lease_owner: str,
        lease_generation: int,
        now: datetime,
    ) -> bool:
        """Former implementation retained temporarily for migration archaeology."""
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
            legacy_omissions = (
                await session.execute(
                    select(func.count())
                    .select_from(models.AcquisitionRecord)
                    .where(
                        models.AcquisitionRecord.run_id == run_id,
                        or_(
                            models.AcquisitionRecord.blob_ref.is_(None),
                            models.AcquisitionRecord.acquired_source.is_(None),
                        ),
                    )
                )
            ).scalar_one()
            await _canonicalize_snapshot_diagnostics(session, run_id)
            await session.execute(
                update(models.AcquisitionRecord)
                .where(models.AcquisitionRecord.run_id == run_id)
                .values(
                    snapshot_outcome=case(
                        (
                            models.AcquisitionRecord.blob_ref.is_not(None)
                            & models.AcquisitionRecord.acquired_source.is_not(None)
                            & (models.AcquisitionRecord.state == AcquisitionRecordState.UNCHANGED),
                            SnapshotItemOutcome.REUSED,
                        ),
                        (
                            models.AcquisitionRecord.blob_ref.is_not(None)
                            & models.AcquisitionRecord.acquired_source.is_not(None),
                            SnapshotItemOutcome.RETAINED,
                        ),
                        else_=SnapshotItemOutcome.OMITTED,
                    )
                )
            )
            membership_hash = await _manifest_digest(session, run_id)
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
                        models.Connector.watermark_scope_fingerprint
                        == run.base_watermark_scope_fingerprint,
                        base_matches,
                    )
                    .values(
                        watermark=run.candidate_watermark,
                        watermark_scope_fingerprint=run.scope_fingerprint or None,
                        last_synced_at=committed_at,
                    )
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
                    .values(
                        acquisition_completed_at=committed_at,
                        promoted_at=committed_at,
                        watermark_committed_at=committed_at,
                        membership_hash=membership_hash,
                        completeness=(
                            SnapshotCompleteness.PARTIAL
                            if legacy_omissions
                            else SnapshotCompleteness.COMPLETE
                        ),
                        omission_count=legacy_omissions,
                        omission_reasons=(
                            {"legacy_unverified": legacy_omissions} if legacy_omissions else {}
                        ),
                        updated_at=committed_at,
                    )
                ),
            )
            if marker.rowcount != 1:  # pragma: no cover - connector update holds the write lock
                msg = "acquisition watermark commit raced another writer"
                raise AcquisitionConflictError(msg)
            return True

    async def cleanup_acquisition_history(self, cutoff: datetime, *, limit: int = 100) -> int:
        """Remove old settled/superseded journal history in bounded batches.

        Deleting a run cascades to its records. A retained blob becomes eligible for the
        existing mark-and-sweep collector only when publications and version history also no
        longer reference it. Active records exclude an authoritative settled run from cleanup;
        they do not preserve a superseded run forever because its fenced work cannot progress.
        """
        if limit < 1:
            msg = "cleanup limit must be positive"
            raise ValueError(msg)
        async with self._sessions.begin() as session:
            run_ids = (
                (
                    await session.execute(
                        select(models.AcquisitionRun.id)
                        .where(
                            models.AcquisitionRun.workspace_id == self._workspace_id,
                            models.AcquisitionRun.updated_at < cutoff,
                            ~exists(
                                select(models.AcquisitionMarker.name).where(
                                    models.AcquisitionMarker.run_id == models.AcquisitionRun.id
                                )
                            ),
                            or_(
                                models.AcquisitionRun.superseded_at.is_not(None),
                                and_(
                                    models.AcquisitionRun.state == AcquisitionRunState.SETTLED,
                                    ~exists(
                                        select(models.AcquisitionRecord.id).where(
                                            models.AcquisitionRecord.run_id
                                            == models.AcquisitionRun.id,
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
                                    ),
                                ),
                            ),
                        )
                        .order_by(models.AcquisitionRun.updated_at, models.AcquisitionRun.id)
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )
            if not run_ids:
                return 0
            result = cast(
                "CursorResult[Any]",
                await session.execute(
                    delete(models.AcquisitionRun).where(
                        models.AcquisitionRun.workspace_id == self._workspace_id,
                        models.AcquisitionRun.id.in_(run_ids),
                        or_(
                            models.AcquisitionRun.state == AcquisitionRunState.SETTLED,
                            models.AcquisitionRun.superseded_at.is_not(None),
                        ),
                    )
                ),
            )
            return result.rowcount

    async def _fence_acquisition_mutation(
        self, session: AsyncSession, fence: AcquisitionFence
    ) -> None:
        """Take SQLite's writer lock while validating the exact durable generation.

        Fenced document helpers call this as their transaction's first statement. Therefore a
        takeover either commits before this check (and the mutation is refused) or waits until
        the guarded mutation commits; there is no check-then-await window between the two.
        """
        result = cast(
            "CursorResult[Any]",
            await session.execute(
                update(models.AcquisitionRun)
                .where(
                    models.AcquisitionRun.id == fence.run_id,
                    models.AcquisitionRun.workspace_id == self._workspace_id,
                    models.AcquisitionRun.lease_owner == fence.owner,
                    models.AcquisitionRun.lease_generation == fence.generation,
                    models.AcquisitionRun.lease_expires_at > fence.now,
                    models.AcquisitionRun.state != AcquisitionRunState.SETTLED,
                    models.AcquisitionRun.superseded_at.is_(None),
                )
                .values(lease_generation=models.AcquisitionRun.lease_generation)
            ),
        )
        if result.rowcount != 1:
            msg = f"acquisition run {fence.run_id!r} lease changed or expired"
            raise AcquisitionLeaseLostError(msg)

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
            run.superseded_at is not None
            or run.lease_owner != owner
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
