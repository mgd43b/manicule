"""One-time, source-free ownership of retained originals from pre-snapshot corpora.

This is deliberately a client of the durable acquisition API.  A legacy corpus becomes an
ordinary immutable manifest, not a parallel list of blob references with weaker verification
rules.  The one distinction the journal preserves is decisive: these rows came from the local
published-document inventory, not a complete enumeration of the remote source scope.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import and_, exists, or_, select
from sqlalchemy.sql.elements import ColumnElement

from manicule.core.acquisition import (
    AcquiredSource,
    AcquisitionDiagnostic,
    AcquisitionFailureCode,
    AcquisitionRecord,
    AcquisitionRecordState,
    AcquisitionRunState,
    AcquisitionSource,
    AcquisitionStage,
    SnapshotPromotionPolicy,
)
from manicule.core.content import Metadata
from manicule.core.ids import content_hash
from manicule.core.provenance import PROVENANCE_KEY
from manicule.core.sources import DocRef
from manicule.storage import models
from manicule.storage.types import utcnow

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manicule.storage.blobs import BlobStore
    from manicule.storage.docstore import SqliteDocStore

LEGACY_SCOPE = "legacy-published-documents; remote scope unknown"
LEGACY_SCOPE_PREFIX = "legacy-unverified:"
DEFAULT_PAGE_SIZE = 100
LEASE_DURATION = timedelta(minutes=5)


@dataclass(frozen=True, slots=True)
class LegacySnapshotMigration:
    """Privacy-safe aggregate result of one bounded-memory migration sweep."""

    connectors: int = 0
    retained: int = 0
    missing: int = 0
    corrupt: int = 0
    promoted: int = 0
    resumed: int = 0
    deferred: int = 0

    def plus(self, other: LegacySnapshotMigration) -> LegacySnapshotMigration:
        return LegacySnapshotMigration(
            **{
                field: getattr(self, field) + getattr(other, field)
                for field in self.__dataclass_fields__
            }
        )


def _identity(prefix: str, workspace: str, connector: str) -> str:
    payload = f"{len(workspace)}:{workspace}{connector}".encode()
    return f"{prefix}{hashlib.blake2b(payload, digest_size=20).hexdigest()}"


def _scope_fingerprint(workspace: str, connector: str) -> str:
    return _identity(LEGACY_SCOPE_PREFIX, workspace, connector)


def _run_id(workspace: str, connector: str) -> str:
    return _identity("legacy-snapshot-v1-", workspace, connector)


def _diagnostic(code: AcquisitionFailureCode) -> AcquisitionDiagnostic:
    return AcquisitionDiagnostic(stage=AcquisitionStage.ACQUISITION, code=code, retryable=True)


async def _document_page(
    store: SqliteDocStore,
    connector: str,
    *,
    after_source_id: str | None,
    page_size: int,
) -> Sequence[models.Document]:
    async with store.sessions() as session:
        statement = select(models.Document).where(
            models.Document.workspace_id == store.workspace_id,
            models.Document.source == connector,
            models.Document.deleted_at.is_(None),
            models.Document.original_ref.is_not(None),
            ~_covered_by_promoted_manifest(),
        )
        if after_source_id is not None:
            statement = statement.where(models.Document.source_id > after_source_id)
        return (
            (
                await session.execute(
                    statement.order_by(models.Document.source_id).limit(page_size)
                )
            )
            .scalars()
            .all()
        )


async def _document(
    store: SqliteDocStore, connector: str, source_id: str
) -> models.Document | None:
    async with store.sessions() as session:
        return (
            await session.execute(
                select(models.Document).where(
                    models.Document.workspace_id == store.workspace_id,
                    models.Document.source == connector,
                    models.Document.source_id == source_id,
                    models.Document.deleted_at.is_(None),
                )
            )
        ).scalar_one_or_none()


async def _checkpoint(store: SqliteDocStore, run_id: str) -> tuple[int, str | None]:
    """Read only the durable keyset checkpoint, never the manifest into memory."""
    async with store.sessions() as session:
        row = (
            await session.execute(
                select(models.AcquisitionRecord.sequence, models.AcquisitionRecord.source_id)
                .where(models.AcquisitionRecord.run_id == run_id)
                .order_by(models.AcquisitionRecord.sequence.desc())
                .limit(1)
            )
        ).one_or_none()
        return (0, None) if row is None else (row.sequence + 1, row.source_id)


def _metadata(row: models.Document) -> Metadata:
    raw = row.doc_metadata
    return dict(raw) if isinstance(raw, Mapping) else {}


async def _validate(
    blobs: BlobStore, row: models.Document
) -> tuple[AcquiredSource | None, AcquisitionFailureCode | None]:
    ref = row.original_ref
    if ref is None:  # guarded by the page query; retained for race-safe honesty
        return None, AcquisitionFailureCode.MISSING_BODY
    try:
        data = await blobs.get(ref)
    except (OSError, EOFError):
        return None, AcquisitionFailureCode.CORRUPT_BODY
    if data is None:
        return None, AcquisitionFailureCode.MISSING_BODY
    if content_hash(data) != ref or row.content_hash != ref:
        return None, AcquisitionFailureCode.CORRUPT_BODY
    return (
        AcquiredSource(
            source_id=row.source_id,
            uri=row.uri,
            media_type=row.media_type,
            # The legacy document schema did not retain whether connector content was text or
            # bytes, nor its encoding.  Bytes are the lossless and therefore honest envelope.
            encoding="utf-8",
            metadata=_metadata(row),
            text_content=False,
            content_hash=ref,
            byte_length=len(data),
        ),
        None,
    )


def _source(row: models.Document, acquired: AcquiredSource | None) -> AcquisitionSource:
    metadata = _metadata(row)
    raw_provenance = metadata.get(PROVENANCE_KEY)
    provenance = dict(raw_provenance) if isinstance(raw_provenance, Mapping) else {}
    return AcquisitionSource(
        ref=DocRef(source_id=row.source_id, uri=row.uri),
        version_token=row.version_token,
        title=row.title,
        media_type=row.media_type,
        size_bytes=None if acquired is None else acquired.byte_length,
        metadata=metadata,
        provenance=provenance,
    )


async def _process_record(
    store: SqliteDocStore,
    blobs: BlobStore,
    run_id: str,
    record: AcquisitionRecord,
    row: models.Document,
    *,
    owner: str,
    generation: int,
) -> LegacySnapshotMigration:
    acquired, failure = await _validate(blobs, row)
    now = utcnow()
    if record.state is AcquisitionRecordState.DISCOVERED:
        record = await store.transition_acquisition_record(
            run_id,
            row.source_id,
            AcquisitionRecordState.DISCOVERED,
            AcquisitionRecordState.ACQUIRING,
            lease_owner=owner,
            lease_generation=generation,
            now=now,
        )
    if failure is not None:
        if record.state is AcquisitionRecordState.ACQUIRING:
            await store.transition_acquisition_record(
                run_id,
                row.source_id,
                AcquisitionRecordState.ACQUIRING,
                AcquisitionRecordState.RETRY,
                lease_owner=owner,
                lease_generation=generation,
                now=utcnow(),
                diagnostic=_diagnostic(failure),
            )
        return LegacySnapshotMigration(
            missing=int(failure is AcquisitionFailureCode.MISSING_BODY),
            corrupt=int(failure is AcquisitionFailureCode.CORRUPT_BODY),
        )
    if record.state is AcquisitionRecordState.ACQUIRING and acquired is not None:
        await store.transition_acquisition_record(
            run_id,
            row.source_id,
            AcquisitionRecordState.ACQUIRING,
            AcquisitionRecordState.ACQUIRED,
            lease_owner=owner,
            lease_generation=generation,
            now=utcnow(),
            blob_ref=row.original_ref,
            acquired_source=acquired,
            fetched_version_token=row.version_token,
        )
        return LegacySnapshotMigration(retained=1)
    return LegacySnapshotMigration()


async def _omit_disappeared_record(
    store: SqliteDocStore,
    run_id: str,
    record: AcquisitionRecord,
    *,
    owner: str,
    generation: int,
) -> LegacySnapshotMigration:
    """Finish a journaled item whose publication vanished before local validation."""
    if record.state is AcquisitionRecordState.DISCOVERED:
        record = await store.transition_acquisition_record(
            run_id,
            record.source.source_id,
            AcquisitionRecordState.DISCOVERED,
            AcquisitionRecordState.ACQUIRING,
            lease_owner=owner,
            lease_generation=generation,
            now=utcnow(),
        )
    if record.state is AcquisitionRecordState.ACQUIRING:
        await store.transition_acquisition_record(
            run_id,
            record.source.source_id,
            AcquisitionRecordState.ACQUIRING,
            AcquisitionRecordState.RETRY,
            lease_owner=owner,
            lease_generation=generation,
            now=utcnow(),
            diagnostic=_diagnostic(AcquisitionFailureCode.MISSING_BODY),
        )
        return LegacySnapshotMigration(missing=1)
    return LegacySnapshotMigration()


async def _settle_records(
    store: SqliteDocStore, run_id: str, *, owner: str, generation: int, page_size: int
) -> None:
    after: int | None = None
    while True:
        records = await store.list_acquisition_records(
            run_id, after_sequence=after, limit=page_size
        )
        if not records:
            return
        for record in records:
            if record.state is AcquisitionRecordState.ACQUIRED:
                await store.transition_acquisition_record(
                    run_id,
                    record.source.source_id,
                    AcquisitionRecordState.ACQUIRED,
                    AcquisitionRecordState.INDEXING,
                    lease_owner=owner,
                    lease_generation=generation,
                    now=utcnow(),
                )
                await store.transition_acquisition_record(
                    run_id,
                    record.source.source_id,
                    AcquisitionRecordState.INDEXING,
                    AcquisitionRecordState.SETTLED,
                    lease_owner=owner,
                    lease_generation=generation,
                    now=utcnow(),
                )
            after = record.sequence


async def _migrate_connector(  # noqa: PLR0912 - resumable lifecycle dispatch
    store: SqliteDocStore, blobs: BlobStore, connector: str, *, page_size: int
) -> LegacySnapshotMigration:
    workspace = store.workspace_id
    run_id = _run_id(workspace, connector)
    scope_fingerprint = _scope_fingerprint(workspace, connector)
    prior = await store.get_acquisition_run(run_id)
    run = await store.create_acquisition_run(
        run_id,
        connector,
        source_scope=LEGACY_SCOPE,
        scope_fingerprint=scope_fingerprint,
        scope_inventory_complete=False,
        promotion_policy=SnapshotPromotionPolicy.ALLOW_OMISSIONS,
    )
    if run.state is AcquisitionRunState.SETTLED:
        return LegacySnapshotMigration(connectors=1, resumed=int(prior is not None))
    owner = f"legacy-snapshot-migration-{uuid4().hex}"
    now = utcnow()
    claimed = await store.claim_acquisition_run(
        run_id, owner, now=now, expires_at=now + LEASE_DURATION
    )
    if claimed is None:
        return LegacySnapshotMigration(connectors=1, deferred=1)
    generation = claimed.lease_generation
    result = LegacySnapshotMigration(connectors=1, resumed=int(prior is not None))
    try:
        if claimed.state is AcquisitionRunState.ENUMERATING:
            # First finish any record whose journal append committed before cancellation.
            while True:
                pending = await store.list_acquisition_records(
                    run_id,
                    states=(
                        AcquisitionRecordState.DISCOVERED,
                        AcquisitionRecordState.ACQUIRING,
                    ),
                    limit=page_size,
                )
                if not pending:
                    break
                for record in pending:
                    row = await _document(store, connector, record.source.source_id)
                    outcome = (
                        await _omit_disappeared_record(
                            store,
                            run_id,
                            record,
                            owner=owner,
                            generation=generation,
                        )
                        if row is None
                        else await _process_record(
                            store,
                            blobs,
                            run_id,
                            record,
                            row,
                            owner=owner,
                            generation=generation,
                        )
                    )
                    result = result.plus(outcome)

            sequence, after = await _checkpoint(store, run_id)
            while True:
                rows = await _document_page(
                    store,
                    connector,
                    after_source_id=after,
                    page_size=page_size,
                )
                if not rows:
                    break
                for row in rows:
                    source = _source(row, None)
                    record = await store.append_acquisition_record(
                        run_id,
                        sequence,
                        source,
                        lease_owner=owner,
                        lease_generation=generation,
                        now=utcnow(),
                    )
                    result = result.plus(
                        await _process_record(
                            store,
                            blobs,
                            run_id,
                            record,
                            row,
                            owner=owner,
                            generation=generation,
                        )
                    )
                    sequence += 1
                    after = row.source_id
                await store.renew_acquisition_lease(
                    run_id,
                    owner,
                    generation,
                    now=utcnow(),
                    expires_at=utcnow() + LEASE_DURATION,
                )
            claimed = await store.complete_acquisition_enumeration(
                run_id,
                None,
                lease_owner=owner,
                lease_generation=generation,
                now=utcnow(),
            )

        if claimed.state is AcquisitionRunState.ACQUIRING:
            if claimed.acquisition_completed_at is None:
                claimed = await store.complete_snapshot_acquisition(
                    run_id,
                    lease_owner=owner,
                    lease_generation=generation,
                    now=utcnow(),
                )
            if claimed.promoted_at is None:
                claimed = await store.promote_snapshot_and_commit_watermark(
                    run_id,
                    expected_scope_fingerprint=scope_fingerprint,
                    lease_owner=owner,
                    lease_generation=generation,
                    now=utcnow(),
                )
                result = result.plus(LegacySnapshotMigration(promoted=1))
            claimed = await store.transition_acquisition_run(
                run_id,
                AcquisitionRunState.ACQUIRING,
                AcquisitionRunState.INDEXING,
                lease_owner=owner,
                lease_generation=generation,
                now=utcnow(),
                diagnostic=_diagnostic(AcquisitionFailureCode.LEGACY_UNVERIFIED),
            )

        if claimed.state is AcquisitionRunState.INDEXING:
            await _settle_records(
                store,
                run_id,
                owner=owner,
                generation=generation,
                page_size=page_size,
            )
            await store.transition_acquisition_run(
                run_id,
                AcquisitionRunState.INDEXING,
                AcquisitionRunState.SETTLED,
                lease_owner=owner,
                lease_generation=generation,
                now=utcnow(),
                diagnostic=_diagnostic(AcquisitionFailureCode.LEGACY_UNVERIFIED),
            )
        return result
    finally:
        await store.release_acquisition_lease(run_id, owner, generation)


async def migrate_legacy_snapshots(
    store: SqliteDocStore,
    blobs: BlobStore,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> LegacySnapshotMigration:
    """Migrate this workspace in bounded keyset pages without opening any connector."""
    if page_size <= 0:
        msg = "legacy snapshot migration page_size must be positive"
        raise ValueError(msg)
    result = LegacySnapshotMigration()
    after: str | None = None
    while True:
        async with store.sessions() as session:
            legacy = _legacy_manifest_exists(settled=None)
            resumable = _legacy_manifest_exists(settled=False)
            statement = (
                select(models.Document.source)
                .where(
                    models.Document.workspace_id == store.workspace_id,
                    models.Document.deleted_at.is_(None),
                    models.Document.original_ref.is_not(None),
                    or_(
                        and_(~_covered_by_promoted_manifest(), ~legacy),
                        resumable,
                    ),
                )
                .distinct()
                .order_by(models.Document.source)
                .limit(page_size)
            )
            if after is not None:
                statement = statement.where(models.Document.source > after)
            sources = (await session.execute(statement)).scalars().all()
        if not sources:
            return result
        for connector in sources:
            result = result.plus(
                await _migrate_connector(store, blobs, connector, page_size=page_size)
            )
            after = connector


def _covered_by_promoted_manifest() -> ColumnElement[bool]:
    """Whether this correlated document already has immutable promoted byte ownership."""
    return exists(
        select(models.AcquisitionRecord.id)
        .join(
            models.AcquisitionRun,
            models.AcquisitionRun.id == models.AcquisitionRecord.run_id,
        )
        .where(
            models.AcquisitionRun.workspace_id == models.Document.workspace_id,
            models.AcquisitionRun.connector_name == models.Document.source,
            models.AcquisitionRun.promoted_at.is_not(None),
            models.AcquisitionRecord.source_id == models.Document.source_id,
            models.AcquisitionRecord.blob_ref.is_not(None),
            models.AcquisitionRecord.acquired_source.is_not(None),
        )
    )


def _legacy_manifest_exists(*, settled: bool | None) -> ColumnElement[bool]:
    """Whether this correlated connector already owns the deterministic migration run."""
    statement = select(models.AcquisitionRun.id).where(
        models.AcquisitionRun.workspace_id == models.Document.workspace_id,
        models.AcquisitionRun.connector_name == models.Document.source,
        models.AcquisitionRun.source_scope == LEGACY_SCOPE,
        models.AcquisitionRun.scope_inventory_complete.is_(False),
    )
    if settled is not None:
        state_predicate = models.AcquisitionRun.state == AcquisitionRunState.SETTLED
        statement = statement.where(state_predicate if settled else ~state_predicate)
    return exists(statement)


__all__ = ["LegacySnapshotMigration", "migrate_legacy_snapshots"]
