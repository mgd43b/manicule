"""Concrete durable storage for shadow-generation re-embedding.

SQLite is the authority for checkpoints, leases, immutable generation identity, and the live
publication decision. Lance writes run while a SQLite ``BEGIN IMMEDIATE`` transaction holds
the current fence, so another process cannot take the lease between validation and the
physical vector mutation.
"""

from __future__ import annotations

import hashlib
import math
import shutil
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from pydantic import TypeAdapter
from sqlalchemy import delete, func, insert, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from manicule.core.content import Chunk
from manicule.core.embedding import EmbedFingerprint, Vector, embedding_input_identity
from manicule.ingest.reembed import (
    ChunkKey,
    CorpusSnapshot,
    LivePublication,
    PublicationReceipt,
    PublishOutcome,
    ReembedCommitment,
    ReembedError,
    ReembedLease,
    ReembedRun,
    ReembedState,
    ShadowGeneration,
    ShadowInspection,
    SnapshotChunk,
    SnapshotChunkDigester,
    SnapshotDocument,
    SnapshotInventoryDigester,
)
from manicule.storage import models
from manicule.storage.rows import to_chunk, to_document
from manicule.storage.types import utcnow
from manicule.storage.vectors import LanceVectorStore, generation_pin

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

_RUN = TypeAdapter(ReembedRun)
_COMMITMENT = TypeAdapter(ReembedCommitment)
_RECEIPT = TypeAdapter(PublicationReceipt)
_INSPECTION = TypeAdapter(ShadowInspection)
_SNAPSHOT_DOCUMENT = TypeAdapter(SnapshotDocument)
_SNAPSHOT_CHUNK = TypeAdapter(SnapshotChunk)
_LIVE = TypeAdapter(LivePublication)
_INDEX_STATE_ID: Final = 1
_CORPUS_REVISION_ID: Final = 1
GENERATIONS_DIRNAME: Final = "generations"
SNAPSHOT_PAGE: Final = 256


def _json(adapter: TypeAdapter[Any], value: object) -> str:
    return adapter.dump_json(value).decode("utf-8")


def _generation_id(run_id: str) -> str:
    return f"reembed-{hashlib.sha256(run_id.encode('utf-8')).hexdigest()}"


class SqliteReembedCorpus:
    """Complete, durable, connector-free corpus snapshots over authoritative SQLite rows."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def discard_snapshot(self, snapshot_id: str) -> None:
        """Remove an unreferenced planning snapshot and its cascaded rows."""
        async with self._engine.begin() as connection:
            referenced = (
                await connection.execute(
                    select(models.ReembedRunRecord.id)
                    .where(
                        func.json_extract(models.ReembedRunRecord.commitment_json, "$.snapshot.id")
                        == snapshot_id
                    )
                    .limit(1)
                )
            ).scalar_one_or_none()
            if referenced is not None:
                raise ReembedError("a snapshot bound to a durable run cannot be discarded")
            await connection.execute(
                delete(models.ReembedCorpusSnapshot).where(
                    models.ReembedCorpusSnapshot.id == snapshot_id
                )
            )

    async def begin_snapshot(self) -> CorpusSnapshot:
        snapshot_id = f"snapshot-{uuid.uuid4().hex}"
        connection = await self._engine.connect()
        try:
            await connection.exec_driver_sql("BEGIN IMMEDIATE")
            revision = str(
                (
                    await connection.execute(
                        select(models.CorpusRevision.revision).where(
                            models.CorpusRevision.id == _CORPUS_REVISION_ID
                        )
                    )
                ).scalar_one()
            )
            state = (
                (
                    await connection.execute(
                        select(models.IndexState.__table__).where(
                            models.IndexState.id == _INDEX_STATE_ID
                        )
                    )
                )
                .mappings()
                .one_or_none()
            )
            if state is None or state.vector_table is None or state.embed_fingerprint is None:
                raise ReembedError(  # noqa: TRY301 - transaction rollback belongs to this scope
                    "the index has no complete live vector publication"
                )
            fingerprint = EmbedFingerprint.model_validate_json(state.embed_fingerprint).canonical()
            await connection.execute(
                insert(models.ReembedCorpusSnapshot).values(
                    id=snapshot_id,
                    revision=revision,
                    live_json="",
                    complete=False,
                    document_count=0,
                    chunk_count=0,
                    inventory_digest="",
                    chunk_inventory_digest="",
                    created_at=utcnow(),
                )
            )
            session = AsyncSession(bind=connection, expire_on_commit=False)
            try:
                documents_count, chunks_count, inventory, chunk_inventory = await self._materialize(
                    connection, session, snapshot_id
                )
            finally:
                await session.close()
            live = LivePublication(state.vector_table, fingerprint, chunk_inventory)
            await connection.execute(
                update(models.IndexState)
                .where(models.IndexState.id == _INDEX_STATE_ID)
                .values(vector_inventory_digest=chunk_inventory, updated_at=utcnow())
            )
            await connection.execute(
                update(models.ReembedCorpusSnapshot)
                .where(models.ReembedCorpusSnapshot.id == snapshot_id)
                .values(
                    live_json=_json(_LIVE, live),
                    complete=True,
                    document_count=documents_count,
                    chunk_count=chunks_count,
                    inventory_digest=inventory,
                    chunk_inventory_digest=chunk_inventory,
                )
            )
            await connection.commit()
            return CorpusSnapshot(snapshot_id, revision, live)
        except BaseException:
            await connection.rollback()
            raise
        finally:
            await connection.close()

    async def documents(
        self, snapshot: CorpusSnapshot, *, after: str | None, limit: int
    ) -> Sequence[SnapshotDocument]:
        await self._require_snapshot(snapshot)
        statement = select(models.ReembedSnapshotDocument.payload_json).where(
            models.ReembedSnapshotDocument.snapshot_id == snapshot.id
        )
        if after is not None:
            statement = statement.where(models.ReembedSnapshotDocument.document_id > after)
        async with self._engine.connect() as connection:
            values = (
                (
                    await connection.execute(
                        statement.order_by(models.ReembedSnapshotDocument.document_id).limit(limit)
                    )
                )
                .scalars()
                .all()
            )
        return [_SNAPSHOT_DOCUMENT.validate_json(value) for value in values]

    async def document(self, snapshot: CorpusSnapshot, document_id: str) -> SnapshotDocument | None:
        await self._require_snapshot(snapshot)
        async with self._engine.connect() as connection:
            value = (
                await connection.execute(
                    select(models.ReembedSnapshotDocument.payload_json).where(
                        models.ReembedSnapshotDocument.snapshot_id == snapshot.id,
                        models.ReembedSnapshotDocument.document_id == document_id,
                    )
                )
            ).scalar_one_or_none()
        return None if value is None else _SNAPSHOT_DOCUMENT.validate_json(value)

    async def chunks(
        self,
        snapshot: CorpusSnapshot,
        document_id: str,
        *,
        after: ChunkKey | None,
        limit: int,
    ) -> Sequence[SnapshotChunk]:
        await self._require_snapshot(snapshot)
        statement = select(models.ReembedSnapshotChunk.payload_json).where(
            models.ReembedSnapshotChunk.snapshot_id == snapshot.id,
            models.ReembedSnapshotChunk.document_id == document_id,
        )
        if after is not None:
            statement = statement.where(
                (models.ReembedSnapshotChunk.position > after.position)
                | (
                    (models.ReembedSnapshotChunk.position == after.position)
                    & (models.ReembedSnapshotChunk.chunk_id > after.id)
                )
            )
        async with self._engine.connect() as connection:
            values = (
                (
                    await connection.execute(
                        statement.order_by(
                            models.ReembedSnapshotChunk.position,
                            models.ReembedSnapshotChunk.chunk_id,
                        ).limit(limit)
                    )
                )
                .scalars()
                .all()
            )
        return [_SNAPSHOT_CHUNK.validate_json(value) for value in values]

    async def _materialize(
        self, connection: AsyncConnection, session: AsyncSession, snapshot_id: str
    ) -> tuple[int, int, str, str]:
        document_after: str | None = None
        documents_count = chunks_count = 0
        inventory = SnapshotInventoryDigester(
            str(
                (
                    await connection.execute(
                        select(models.ReembedCorpusSnapshot.revision).where(
                            models.ReembedCorpusSnapshot.id == snapshot_id
                        )
                    )
                ).scalar_one()
            )
        )
        chunk_digest = SnapshotChunkDigester()
        while True:
            statement = select(models.Document).where(models.Document.deleted_at.is_(None))
            if document_after is not None:
                statement = statement.where(models.Document.id > document_after)
            page = (
                (await session.execute(statement.order_by(models.Document.id).limit(SNAPSHOT_PAGE)))
                .scalars()
                .all()
            )
            if not page:
                break
            for row in page:
                stored_document = SnapshotDocument(
                    workspace_id=row.workspace_id,
                    document=to_document(row),
                    original_omitted_reason=row.original_omitted_reason,
                    chunk_fingerprint=row.chunk_fp,
                    embed_fingerprint=row.embed_fp,
                    glossary_fingerprint=row.glossary_fp,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                    last_seen_at=row.last_seen_at,
                    deleted_at=row.deleted_at,
                )
                document_json = _json(_SNAPSHOT_DOCUMENT, stored_document)
                inventory.add_document(_SNAPSHOT_DOCUMENT.validate_json(document_json))
                await connection.execute(
                    insert(models.ReembedSnapshotDocument).values(
                        snapshot_id=snapshot_id,
                        document_id=row.id,
                        payload_json=document_json,
                    )
                )
                chunk_after: tuple[int, str] | None = None
                while True:
                    chunks = select(models.Chunk).where(models.Chunk.document_id == row.id)
                    if chunk_after is not None:
                        position, chunk_id = chunk_after
                        chunks = chunks.where(
                            (models.Chunk.position > position)
                            | ((models.Chunk.position == position) & (models.Chunk.id > chunk_id))
                        )
                    chunk_page = (
                        (
                            await session.execute(
                                chunks.order_by(models.Chunk.position, models.Chunk.id).limit(
                                    SNAPSHOT_PAGE
                                )
                            )
                        )
                        .scalars()
                        .all()
                    )
                    if not chunk_page:
                        break
                    for chunk_row in chunk_page:
                        stored_chunk = SnapshotChunk(
                            chunk=to_chunk(chunk_row),
                            vector_id=chunk_row.vector_id,
                            publication_id=row.publication_id,
                            sequence=chunk_row.seq,
                            created_at=chunk_row.created_at,
                        )
                        chunk_json = _json(_SNAPSHOT_CHUNK, stored_chunk)
                        persisted_chunk = _SNAPSHOT_CHUNK.validate_json(chunk_json)
                        inventory.add_chunk(persisted_chunk)
                        chunk_digest.add(persisted_chunk)
                        await connection.execute(
                            insert(models.ReembedSnapshotChunk).values(
                                snapshot_id=snapshot_id,
                                document_id=row.id,
                                position=chunk_row.position,
                                chunk_id=chunk_row.id,
                                payload_json=chunk_json,
                            )
                        )
                        chunks_count += 1
                    last = chunk_page[-1]
                    chunk_after = (last.position, last.id)
                documents_count += 1
            document_after = page[-1].id
        return (
            documents_count,
            chunks_count,
            inventory.hexdigest(),
            chunk_digest.hexdigest(),
        )

    async def _require_snapshot(self, snapshot: CorpusSnapshot) -> None:
        async with self._engine.connect() as connection:
            row = (
                (
                    await connection.execute(
                        select(models.ReembedCorpusSnapshot.__table__).where(
                            models.ReembedCorpusSnapshot.id == snapshot.id
                        )
                    )
                )
                .mappings()
                .one_or_none()
            )
        if (
            row is None
            or not row.complete
            or row.revision != snapshot.revision
            or _LIVE.validate_json(row.live_json) != snapshot.live
        ):
            raise ReembedError("the durable complete-corpus snapshot is missing or mismatched")


class SqliteReembedStore:
    """SQLite implementation of the journal, lease authority, and publisher protocols."""

    def __init__(self, engine: AsyncEngine, *, clock: Callable[[], float] = time.time) -> None:
        self._engine = engine
        self._clock = clock

    @asynccontextmanager
    async def fenced(self, run_id: str, lease: ReembedLease) -> AsyncGenerator[AsyncConnection]:
        """Hold SQLite's writer lock and the current fence across an external mutation."""
        connection = await self._engine.connect()
        try:
            await connection.exec_driver_sql("BEGIN IMMEDIATE")
            await self._require_lease(connection, run_id, lease)
            yield connection
            await self._require_lease(connection, run_id, lease)
            await connection.commit()
        except BaseException:
            await connection.rollback()
            raise
        finally:
            await connection.close()

    async def create(
        self,
        run_id: str,
        commitment: ReembedCommitment,
        *,
        owner_token: str,
        ttl_seconds: float,
    ) -> tuple[ReembedRun, ReembedLease]:
        commitment_json = _json(_COMMITMENT, commitment)
        async with self._immediate() as connection:
            row = (
                (
                    await connection.execute(
                        select(models.ReembedRunRecord.__table__).where(
                            models.ReembedRunRecord.id == run_id
                        )
                    )
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                run = ReembedRun(id=run_id, commitment=commitment)
                await connection.execute(
                    insert(models.ReembedRunRecord).values(
                        id=run_id,
                        commitment_json=commitment_json,
                        state=run.state.value,
                        checkpoint_json=_json(_RUN, run),
                        revision=0,
                        lease_generation=0,
                        created_at=utcnow(),
                        updated_at=utcnow(),
                    )
                )
            else:
                if row.commitment_json != commitment_json:
                    raise ReembedError("run id already belongs to another immutable plan")
                run = _RUN.validate_json(row.checkpoint_json)
            lease = await self._acquire(connection, run_id, owner_token, ttl_seconds)
            return run, lease

    async def get(self, run_id: str) -> ReembedRun | None:
        async with self._engine.connect() as connection:
            value = (
                await connection.execute(
                    select(models.ReembedRunRecord.checkpoint_json).where(
                        models.ReembedRunRecord.id == run_id
                    )
                )
            ).scalar_one_or_none()
        return None if value is None else _RUN.validate_json(value)

    async def live_generation_id(self) -> str | None:
        """The currently published vector pointer, without requiring a complete identity."""
        async with self._engine.connect() as connection:
            return (
                await connection.execute(
                    select(models.IndexState.vector_table).where(
                        models.IndexState.id == _INDEX_STATE_ID
                    )
                )
            ).scalar_one_or_none()

    @asynccontextmanager
    async def cleanup_guard(self, run_id: str) -> AsyncGenerator[tuple[AsyncConnection, str]]:
        """Keep a terminal generation non-live until its physical deletion completes."""
        async with self._immediate() as connection:
            checkpoint = (
                await connection.execute(
                    select(models.ReembedRunRecord.checkpoint_json).where(
                        models.ReembedRunRecord.id == run_id
                    )
                )
            ).scalar_one_or_none()
            if checkpoint is None:
                raise ReembedError(f"no re-embedding run {run_id!r} exists")
            run = _RUN.validate_json(checkpoint)
            if run.state not in {ReembedState.FAILED, ReembedState.SUPERSEDED}:
                raise ReembedError("only failed or superseded shadow generations may be cleaned")
            generation_id = run.shadow_generation_id or _generation_id(run_id)
            live = (
                await connection.execute(
                    select(models.IndexState.vector_table).where(
                        models.IndexState.id == _INDEX_STATE_ID
                    )
                )
            ).scalar_one_or_none()
            if live == generation_id:
                raise ReembedError("the live shadow generation cannot be cleaned")
            yield connection, generation_id

    async def acquire(self, run_id: str, owner_token: str, *, ttl_seconds: float) -> ReembedLease:
        async with self._immediate() as connection:
            return await self._acquire(connection, run_id, owner_token, ttl_seconds)

    async def renew(self, run_id: str, lease: ReembedLease, *, ttl_seconds: float) -> ReembedLease:
        async with self._immediate() as connection:
            await self._require_lease(connection, run_id, lease)
            renewed = ReembedLease(lease.owner_token, lease.generation, self._clock() + ttl_seconds)
            await connection.execute(
                update(models.ReembedRunRecord)
                .where(models.ReembedRunRecord.id == run_id)
                .values(lease_expires_at=renewed.expires_at, updated_at=utcnow())
            )
            await self._require_lease(connection, run_id, renewed)
            return renewed

    async def release(self, run_id: str, lease: ReembedLease) -> None:
        """Expire the exact current fenced lease in one SQLite write transaction."""
        async with self._immediate() as connection:
            await self._require_lease(connection, run_id, lease)
            now = self._clock()
            changed = await connection.execute(
                update(models.ReembedRunRecord)
                .where(
                    models.ReembedRunRecord.id == run_id,
                    models.ReembedRunRecord.lease_owner == lease.owner_token,
                    models.ReembedRunRecord.lease_generation == lease.generation,
                    models.ReembedRunRecord.lease_expires_at == lease.expires_at,
                )
                .values(lease_expires_at=now, updated_at=utcnow())
            )
            if changed.rowcount != 1:
                raise ReembedError("stale or expired re-embedding lease")

    async def assert_current(self, run_id: str, lease: ReembedLease) -> None:
        """Check a fence without holding it across subsequent non-SQLite work."""
        async with self._immediate() as connection:
            await self._require_lease(connection, run_id, lease)

    async def assert_locked(
        self, connection: AsyncConnection, run_id: str, lease: ReembedLease
    ) -> None:
        """Recheck a fence inside a caller's already-held SQLite write transaction."""
        await self._require_lease(connection, run_id, lease)

    async def save(
        self, run: ReembedRun, *, expected_revision: int, lease: ReembedLease
    ) -> ReembedRun:
        async with self._immediate() as connection:
            await self._require_lease(connection, run.id, lease)
            saved = ReembedRun(
                id=run.id,
                commitment=run.commitment,
                state=run.state,
                document_after=run.document_after,
                active_document_id=run.active_document_id,
                chunk_after=run.chunk_after,
                documents_completed=run.documents_completed,
                chunks_completed=run.chunks_completed,
                shadow_generation_id=run.shadow_generation_id,
                receipt=run.receipt,
                revision=expected_revision + 1,
                failure=run.failure,
            )
            result = await connection.execute(
                update(models.ReembedRunRecord)
                .where(
                    models.ReembedRunRecord.id == run.id,
                    models.ReembedRunRecord.revision == expected_revision,
                )
                .values(
                    state=saved.state.value,
                    checkpoint_json=_json(_RUN, saved),
                    revision=saved.revision,
                    updated_at=utcnow(),
                )
            )
            if result.rowcount != 1:
                raise ReembedError("stale journal revision")
            await self._require_lease(connection, run.id, lease)
            return saved

    async def abandon(self, run_id: str, *, lease: ReembedLease) -> ReembedRun:
        """Durably make an unfinished run eligible for terminal shadow cleanup."""
        run = await self.get(run_id)
        if run is None:
            raise ReembedError(f"no re-embedding run {run_id!r} exists")
        if run.state in {ReembedState.PUBLISHED, ReembedState.SUPERSEDED}:
            raise ReembedError("a terminal publication decision cannot be abandoned")
        return await self.save(
            replace(run, state=ReembedState.FAILED, failure="abandoned by operator"),
            expected_revision=run.revision,
            lease=lease,
        )

    async def publish(
        self,
        run: ReembedRun,
        generation: ShadowGeneration,
        *,
        expected: LivePublication,
        expected_corpus_revision: str,
        lease: ReembedLease,
    ) -> PublicationReceipt:
        async with self.fenced(run.id, lease) as connection:
            if (
                expected != run.commitment.snapshot.live
                or expected_corpus_revision != run.commitment.snapshot.revision
            ):
                raise ReembedError("publication expectations do not match the immutable run")
            prior = (
                await connection.execute(
                    select(models.ReembedPublicationReceipt.receipt_json).where(
                        models.ReembedPublicationReceipt.run_id == run.id
                    )
                )
            ).scalar_one_or_none()
            if prior is not None:
                return _RECEIPT.validate_json(prior)

            await self._require_bound_snapshot(connection, run)

            observed = await self._live(connection)
            corpus_revision = str(
                (
                    await connection.execute(
                        select(models.CorpusRevision.revision).where(
                            models.CorpusRevision.id == _CORPUS_REVISION_ID
                        )
                    )
                ).scalar_one()
            )
            if observed != expected or corpus_revision != expected_corpus_revision:
                outcome = PublishOutcome.SUPERSEDED
                published_generation_id = None
                winner = observed
            else:
                shadow = (
                    (
                        await connection.execute(
                            select(models.ReembedShadowGeneration.__table__).where(
                                models.ReembedShadowGeneration.id == generation.id,
                                models.ReembedShadowGeneration.run_id == run.id,
                            )
                        )
                    )
                    .mappings()
                    .one_or_none()
                )
                if shadow is None:
                    raise ReembedError("the durable shadow generation disappeared before publish")
                if (
                    shadow.fingerprint != generation.fingerprint
                    or shadow.inventory_digest != generation.inventory_digest
                    or generation.fingerprint != run.commitment.target_fingerprint
                    or generation.inventory_digest != run.commitment.chunk_inventory_digest
                ):
                    raise ReembedError("publication generation does not match the immutable run")
                if shadow.state != "sealed" or shadow.seal_json is None:
                    raise ReembedError("publication requires the exact sealed shadow generation")
                seal = _INSPECTION.validate_json(shadow.seal_json)
                if (
                    seal.inventory_digest != generation.inventory_digest
                    or seal.fingerprint != generation.fingerprint
                    or not seal.retrieval_ready
                    or not seal.lineage_valid
                ):
                    raise ReembedError("the durable shadow seal is not publishable")
                fingerprint_json = shadow.fingerprint_json
                result = await connection.execute(
                    update(models.IndexState)
                    .where(
                        models.IndexState.id == _INDEX_STATE_ID,
                        models.IndexState.vector_table == expected.generation_id,
                        models.IndexState.vector_inventory_digest == expected.inventory_digest,
                    )
                    .values(
                        vector_table=generation.id,
                        embed_fingerprint=fingerprint_json,
                        vector_inventory_digest=generation.inventory_digest,
                        updated_at=utcnow(),
                    )
                )
                if result.rowcount != 1:
                    raise ReembedError("live publication changed inside its fenced transaction")
                await self._supersede_prior_winner(connection, run.id)
                await connection.execute(
                    update(models.ReembedShadowGeneration)
                    .where(models.ReembedShadowGeneration.id == generation.id)
                    .values(state="published")
                )
                outcome = PublishOutcome.PUBLISHED
                published_generation_id = generation.id
                winner = LivePublication(
                    generation.id, generation.fingerprint, generation.inventory_digest
                )
            receipt = PublicationReceipt(
                id=f"receipt:{run.id}",
                run_id=run.id,
                outcome=outcome,
                expected=expected,
                observed_winner=winner,
                published_generation_id=published_generation_id,
            )
            await connection.execute(
                insert(models.ReembedPublicationReceipt).values(
                    run_id=run.id, receipt_json=_json(_RECEIPT, receipt), created_at=utcnow()
                )
            )
            return receipt

    async def _require_bound_snapshot(self, connection: AsyncConnection, run: ReembedRun) -> None:
        row = (
            (
                await connection.execute(
                    select(models.ReembedCorpusSnapshot.__table__).where(
                        models.ReembedCorpusSnapshot.id == run.commitment.snapshot.id
                    )
                )
            )
            .mappings()
            .one_or_none()
        )
        snapshot = run.commitment.snapshot
        if (
            row is None
            or not row.complete
            or row.revision != snapshot.revision
            or _LIVE.validate_json(row.live_json) != snapshot.live
            or row.document_count != run.commitment.plan.documents
            or row.chunk_count != run.commitment.plan.chunks
            or row.inventory_digest != run.commitment.inventory_digest
            or row.chunk_inventory_digest != run.commitment.chunk_inventory_digest
        ):
            raise ReembedError("publication is not bound to a durable complete-corpus snapshot")

    async def _supersede_prior_winner(
        self, connection: AsyncConnection, current_run_id: str
    ) -> None:
        rows = (
            await connection.execute(
                select(
                    models.ReembedRunRecord.id,
                    models.ReembedRunRecord.checkpoint_json,
                    models.ReembedRunRecord.revision,
                )
                .join(
                    models.ReembedShadowGeneration,
                    models.ReembedShadowGeneration.run_id == models.ReembedRunRecord.id,
                )
                .where(
                    or_(
                        models.ReembedRunRecord.state == ReembedState.PUBLISHED.value,
                        models.ReembedShadowGeneration.state == "published",
                    ),
                    models.ReembedRunRecord.id != current_run_id,
                )
            )
        ).all()
        for row in rows:
            prior = _RUN.validate_json(row.checkpoint_json)
            superseded = replace(
                prior, state=ReembedState.SUPERSEDED, revision=int(row.revision) + 1
            )
            await connection.execute(
                update(models.ReembedRunRecord)
                .where(
                    models.ReembedRunRecord.id == row.id,
                    models.ReembedRunRecord.revision == row.revision,
                )
                .values(
                    state=ReembedState.SUPERSEDED.value,
                    checkpoint_json=_json(_RUN, superseded),
                    revision=superseded.revision,
                    updated_at=utcnow(),
                )
            )
            await connection.execute(
                update(models.ReembedShadowGeneration)
                .where(models.ReembedShadowGeneration.run_id == row.id)
                .values(state="superseded")
            )

    async def _live(self, connection: AsyncConnection) -> LivePublication:
        row = (
            (
                await connection.execute(
                    select(models.IndexState.__table__).where(
                        models.IndexState.id == _INDEX_STATE_ID
                    )
                )
            )
            .mappings()
            .one_or_none()
        )
        if row is None or row.vector_table is None or row.embed_fingerprint is None:
            raise ReembedError("the index has no complete live vector publication")
        fingerprint = EmbedFingerprint.model_validate_json(row.embed_fingerprint).canonical()
        return LivePublication(row.vector_table, fingerprint, row.vector_inventory_digest)

    async def _acquire(
        self, connection: AsyncConnection, run_id: str, owner_token: str, ttl_seconds: float
    ) -> ReembedLease:
        row = (
            (
                await connection.execute(
                    select(models.ReembedRunRecord.__table__).where(
                        models.ReembedRunRecord.id == run_id
                    )
                )
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            raise ReembedError(f"no re-embedding run {run_id!r} exists")
        now = self._clock()
        if row.lease_expires_at is not None and row.lease_expires_at > now:
            if row.lease_owner != owner_token:
                raise ReembedError("another owner holds the re-embedding lease")
            generation = row.lease_generation
        else:
            generation = row.lease_generation + 1
        lease = ReembedLease(owner_token, generation, now + ttl_seconds)
        await connection.execute(
            update(models.ReembedRunRecord)
            .where(models.ReembedRunRecord.id == run_id)
            .values(
                lease_owner=lease.owner_token,
                lease_generation=lease.generation,
                lease_expires_at=lease.expires_at,
                updated_at=utcnow(),
            )
        )
        return lease

    async def _require_lease(
        self, connection: AsyncConnection, run_id: str, lease: ReembedLease
    ) -> None:
        row = (
            await connection.execute(
                select(
                    models.ReembedRunRecord.lease_owner,
                    models.ReembedRunRecord.lease_generation,
                    models.ReembedRunRecord.lease_expires_at,
                ).where(models.ReembedRunRecord.id == run_id)
            )
        ).one_or_none()
        if (
            row is None
            or row.lease_owner != lease.owner_token
            or row.lease_generation != lease.generation
            or row.lease_expires_at != lease.expires_at
            or lease.expires_at <= self._clock()
        ):
            raise ReembedError("stale or expired re-embedding lease")

    @asynccontextmanager
    async def _immediate(self) -> AsyncGenerator[AsyncConnection]:
        connection = await self._engine.connect()
        try:
            await connection.exec_driver_sql("BEGIN IMMEDIATE")
            yield connection
            await connection.commit()
        except BaseException:
            await connection.rollback()
            raise
        finally:
            await connection.close()


class LanceShadowGenerations:
    """Named Lance directories whose rows remain invisible until SQLite publishes one."""

    def __init__(
        self,
        directory: Path,
        authority: SqliteReembedStore,
        *,
        mutation_hook: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._root = directory / GENERATIONS_DIRNAME
        self._authority = authority
        self._stores: dict[str, LanceVectorStore] = {}
        self._mutation_hook = mutation_hook

    def directory(self, generation_id: str) -> Path:
        return self._root / generation_id

    async def open_or_create(
        self,
        run_id: str,
        *,
        fingerprint: EmbedFingerprint,
        inventory_digest: str,
        lease: ReembedLease,
    ) -> ShadowGeneration:
        offered = ShadowGeneration(
            _generation_id(run_id), run_id, fingerprint.canonical(), inventory_digest
        )
        async with self._authority.fenced(run_id, lease) as connection:
            existing = (
                (
                    await connection.execute(
                        select(models.ReembedShadowGeneration.__table__).where(
                            models.ReembedShadowGeneration.run_id == run_id
                        )
                    )
                )
                .mappings()
                .one_or_none()
            )
            if existing is None:
                await connection.execute(
                    insert(models.ReembedShadowGeneration).values(
                        id=offered.id,
                        run_id=run_id,
                        fingerprint_json=fingerprint.model_dump_json(),
                        fingerprint=offered.fingerprint,
                        inventory_digest=inventory_digest,
                        state="building",
                        created_at=utcnow(),
                    )
                )
                generation = offered
            else:
                generation = ShadowGeneration(
                    existing.id, existing.run_id, existing.fingerprint, existing.inventory_digest
                )
                if generation != offered:
                    raise ReembedError("shadow identity is immutable")
            store = self._store(generation.id)
            if existing is not None and existing.state != "building":
                await store.open_existing(fingerprint)
            else:
                await store.ensure_ready(fingerprint)
            return generation

    async def upsert(
        self,
        generation: ShadowGeneration,
        chunks: Sequence[SnapshotChunk],
        vectors: Sequence[Vector],
        *,
        lease: ReembedLease,
    ) -> None:
        async with self._authority.fenced(generation.run_id, lease) as connection:
            store = self._store(generation.id)
            await store.ensure_ready(
                await self._fingerprint(connection, generation, require_building=True)
            )
            if self._mutation_hook is not None:
                await self._mutation_hook()
            await self._authority.assert_locked(connection, generation.run_id, lease)
            await store.upsert_snapshot(chunks, vectors, publication_id=generation.id)
            try:
                await self._authority.assert_locked(connection, generation.run_id, lease)
            except ReembedError:
                await store.delete_chunks([stored.vector_id for stored in chunks])
                raise

    async def inspect(
        self, generation: ShadowGeneration, *, lease: ReembedLease
    ) -> ShadowInspection:
        async with self._authority.fenced(generation.run_id, lease) as connection:
            store = self._store(generation.id)
            await store.ensure_ready(await self._fingerprint(connection, generation))
        revision = await store.storage_revision()
        fingerprint = await store.fingerprint()
        rows_count = valid_rows = 0
        dimensions: set[int] = set()
        finite = True
        lineage_valid = True
        digest = SnapshotChunkDigester()
        async for page in store.inspection_pages():
            for row in page:
                rows_count += 1
                try:
                    chunk = Chunk.model_validate_json(str(row["chunk_json"]))
                    source_vector_id = row["source_vector_id"]
                    source_publication_id = row["source_publication_id"]
                    source_sequence = row["source_sequence"]
                    if (
                        source_vector_id is None
                        or source_publication_id is None
                        or source_sequence is None
                    ):
                        lineage_valid = False
                        finite = False
                        continue
                    stored = SnapshotChunk(
                        chunk=chunk,
                        vector_id=str(source_vector_id),
                        publication_id=str(source_publication_id),
                        sequence=int(source_sequence),
                        created_at=(
                            None
                            if row.get("source_created_at") is None
                            else datetime.fromisoformat(str(row["source_created_at"]))
                        ),
                    )
                    vector = [float(value) for value in row["vector"]]
                    digest.add(stored)
                except (KeyError, TypeError, ValueError):
                    lineage_valid = False
                    finite = False
                    continue
                valid_rows += 1
                dimensions.add(len(vector))
                finite = finite and all(math.isfinite(value) for value in vector)
                lineage_valid = lineage_valid and self._row_lineage(
                    row, generation, stored, fingerprint
                )
        physical_rows = await store.count()
        if physical_rows != rows_count:
            # The keyset columns are validated data, not a trusted primary key. Corruption can
            # collapse more than one physical row onto the same cursor; the independent table
            # count makes such a skipped suffix a validation failure rather than a false seal.
            lineage_valid = False
            finite = False
            rows_count = physical_rows
        if await store.storage_revision() != revision:
            raise ReembedError("shadow changed during inspection")
        await self._authority.assert_current(generation.run_id, lease)
        return ShadowInspection(
            rows=rows_count,
            unique_chunks=valid_rows if lineage_valid else 0,
            dimension=dimensions.pop() if len(dimensions) == 1 else 0,
            finite=finite,
            fingerprint="" if fingerprint is None else fingerprint.canonical(),
            inventory_digest=digest.hexdigest(),
            lineage_valid=lineage_valid,
            retrieval_ready=fingerprint is not None,
            storage_revision=revision,
        )

    async def seal(
        self,
        generation: ShadowGeneration,
        inspection: ShadowInspection,
        *,
        lease: ReembedLease,
    ) -> None:
        async with self._authority.fenced(generation.run_id, lease) as connection:
            fingerprint = await self._fingerprint(connection, generation)
            commitment_json = (
                await connection.execute(
                    select(models.ReembedRunRecord.commitment_json).where(
                        models.ReembedRunRecord.id == generation.run_id
                    )
                )
            ).scalar_one()
            commitment = _COMMITMENT.validate_json(commitment_json)
            if (
                inspection.rows != commitment.plan.chunks
                or inspection.unique_chunks != commitment.plan.chunks
                or inspection.dimension != fingerprint.dimension
                or not inspection.finite
                or inspection.fingerprint != generation.fingerprint
                or inspection.inventory_digest != generation.inventory_digest
                or not inspection.lineage_valid
                or not inspection.retrieval_ready
                or not inspection.storage_revision
            ):
                raise ReembedError("only an exact validated shadow generation can be sealed")
            store = self._store(generation.id)
            await store.open_existing()
            if await store.storage_revision() != inspection.storage_revision:
                raise ReembedError("shadow changed between inspection and seal")
            current = (
                (
                    await connection.execute(
                        select(models.ReembedShadowGeneration.__table__).where(
                            models.ReembedShadowGeneration.id == generation.id
                        )
                    )
                )
                .mappings()
                .one()
            )
            seal_json = _json(_INSPECTION, inspection)
            if current.state == "sealed" and current.seal_json == seal_json:
                return
            if current.state != "building":
                raise ReembedError("shadow seal is immutable")
            result = await connection.execute(
                update(models.ReembedShadowGeneration)
                .where(
                    models.ReembedShadowGeneration.id == generation.id,
                    models.ReembedShadowGeneration.state == "building",
                )
                .values(state="sealed", seal_json=seal_json)
            )
            if result.rowcount != 1:
                raise ReembedError("shadow generation could not be sealed atomically")

    async def cleanup_terminal(self, run_id: str) -> bool:
        """Delete a failed or superseded generation; published/live generations are refused."""
        async with self._authority.cleanup_guard(run_id) as (connection, generation_id):
            path = self.directory(generation_id)
            async with generation_pin(path, exclusive=True):
                store = self._stores.pop(generation_id, None)
                if store is not None:
                    await store.teardown()
                removed = path.exists()
                if removed:
                    shutil.rmtree(path)
                await connection.execute(
                    delete(models.ReembedShadowGeneration).where(
                        models.ReembedShadowGeneration.id == generation_id,
                        models.ReembedShadowGeneration.run_id == run_id,
                    )
                )
                return removed

    def _store(self, generation_id: str) -> LanceVectorStore:
        return self._stores.setdefault(
            generation_id, LanceVectorStore(self.directory(generation_id))
        )

    async def _fingerprint(
        self,
        connection: AsyncConnection,
        generation: ShadowGeneration,
        *,
        require_building: bool = False,
    ) -> EmbedFingerprint:
        row = (
            (
                await connection.execute(
                    select(models.ReembedShadowGeneration.__table__).where(
                        models.ReembedShadowGeneration.id == generation.id,
                        models.ReembedShadowGeneration.run_id == generation.run_id,
                    )
                )
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            raise ReembedError("the durable shadow generation does not exist")
        if (
            row.fingerprint != generation.fingerprint
            or row.inventory_digest != generation.inventory_digest
        ):
            raise ReembedError("shadow identity is immutable")
        if require_building and row.state != "building":
            raise ReembedError("sealed shadow generation is immutable")
        return EmbedFingerprint.model_validate_json(row.fingerprint_json)

    def _row_lineage(
        self,
        row: Mapping[str, object],
        generation: ShadowGeneration,
        stored: SnapshotChunk,
        fingerprint: EmbedFingerprint | None,
    ) -> bool:
        chunk = stored.chunk
        return bool(
            str(row["id"]) == stored.vector_id
            and str(row["chunk_id"]) == chunk.id
            and str(row["publication_id"]) == stored.publication_id
            and str(row["document_id"]) == chunk.document_id
            and str(row["kind"]) == chunk.kind.value
            and (None if row["lang"] is None else str(row["lang"])) == chunk.lang
            and int(str(row["position"])) == chunk.position
            and fingerprint is not None
            and str(row["embed_identity"])
            == embedding_input_identity(
                chunk.embed_text, document_id=chunk.document_id, embed=fingerprint
            )
        )


__all__ = [
    "GENERATIONS_DIRNAME",
    "SNAPSHOT_PAGE",
    "LanceShadowGenerations",
    "SqliteReembedCorpus",
    "SqliteReembedStore",
]
