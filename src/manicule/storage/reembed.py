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
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from pydantic import TypeAdapter
from sqlalchemy import delete, insert, select, update

from manicule.core.content import Chunk
from manicule.core.embedding import EmbedFingerprint, Vector
from manicule.core.ids import vector_id
from manicule.ingest.reembed import (
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
)
from manicule.storage import models
from manicule.storage.types import utcnow
from manicule.storage.vectors import LanceVectorStore

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

_RUN = TypeAdapter(ReembedRun)
_COMMITMENT = TypeAdapter(ReembedCommitment)
_RECEIPT = TypeAdapter(PublicationReceipt)
_INDEX_STATE_ID: Final = 1
_CORPUS_REVISION_ID: Final = 1
GENERATIONS_DIRNAME: Final = "generations"


def _json(adapter: TypeAdapter[Any], value: object) -> str:
    return adapter.dump_json(value).decode("utf-8")


def _generation_id(run_id: str) -> str:
    return f"reembed-{hashlib.sha256(run_id.encode('utf-8')).hexdigest()}"


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
            return renewed

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
                await connection.execute(
                    update(models.Document)
                    .where(models.Document.deleted_at.is_(None))
                    .values(publication_id=generation.id, embed_fp=generation.fingerprint)
                )
                chunk_ids = (await connection.execute(select(models.Chunk.id))).scalars().all()
                for chunk_id in chunk_ids:
                    await connection.execute(
                        update(models.Chunk)
                        .where(models.Chunk.id == chunk_id)
                        .values(vector_id=vector_id(generation.id, str(chunk_id)))
                    )
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
        if row.vector_inventory_digest is None:
            raise ReembedError("the live vector publication has no inventory digest")
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
            await self._store(generation.id).ensure_ready(fingerprint)
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
            await store.ensure_ready(await self._fingerprint(connection, generation))
            if self._mutation_hook is not None:
                await self._mutation_hook()
            await store.upsert_snapshot(chunks, vectors, publication_id=generation.id)

    async def inspect(
        self, generation: ShadowGeneration, *, lease: ReembedLease
    ) -> ShadowInspection:
        async with self._authority.fenced(generation.run_id, lease) as connection:
            store = self._store(generation.id)
            await store.ensure_ready(await self._fingerprint(connection, generation))
            fingerprint = await store.fingerprint()
            rows = await store.inspection_rows()
            stored_rows: list[SnapshotChunk] = []
            dimensions: set[int] = set()
            finite = True
            lineage_valid = True
            for row in rows:
                try:
                    chunk = Chunk.model_validate_json(str(row["chunk_json"]))
                    source_vector_id = row["source_vector_id"]
                    source_sequence = row["source_sequence"]
                    if source_vector_id is None or source_sequence is None:
                        lineage_valid = False
                        finite = False
                        continue
                    stored = SnapshotChunk(
                        chunk=chunk,
                        vector_id=str(source_vector_id),
                        sequence=int(source_sequence),
                        created_at=(
                            None
                            if row.get("source_created_at") is None
                            else datetime.fromisoformat(str(row["source_created_at"]))
                        ),
                    )
                    vector = [float(value) for value in row["vector"]]
                except (KeyError, TypeError, ValueError):
                    lineage_valid = False
                    finite = False
                    continue
                stored_rows.append(stored)
                dimensions.add(len(vector))
                finite = finite and all(math.isfinite(value) for value in vector)
                lineage_valid = lineage_valid and (
                    str(row["id"]) == vector_id(generation.id, chunk.id)
                    and str(row["publication_id"]) == generation.id
                )
            stored_rows.sort(
                key=lambda item: (
                    item.chunk.document_id,
                    item.chunk.position,
                    item.chunk.id,
                    item.vector_id,
                    item.sequence,
                )
            )
            digest = SnapshotChunkDigester()
            for stored in stored_rows:
                digest.add(stored)
            return ShadowInspection(
                rows=len(rows),
                unique_chunks=len({stored.chunk.id for stored in stored_rows}),
                dimension=dimensions.pop() if len(dimensions) == 1 else 0,
                finite=finite,
                fingerprint="" if fingerprint is None else fingerprint.canonical(),
                inventory_digest=digest.hexdigest(),
                lineage_valid=lineage_valid,
                retrieval_ready=fingerprint is not None,
            )

    async def cleanup_terminal(self, run_id: str) -> bool:
        """Delete a failed or superseded generation; published/live generations are refused."""
        async with self._authority.cleanup_guard(run_id) as (connection, generation_id):
            store = self._stores.pop(generation_id, None)
            if store is not None:
                await store.teardown()
            path = self.directory(generation_id)
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
        self, connection: AsyncConnection, generation: ShadowGeneration
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
        return EmbedFingerprint.model_validate_json(row.fingerprint_json)


__all__ = ["GENERATIONS_DIRNAME", "LanceShadowGenerations", "SqliteReembedStore"]
