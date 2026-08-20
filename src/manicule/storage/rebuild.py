"""SQLite shadow-generation journal and atomic relational publisher."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

from pydantic import TypeAdapter
from sqlalchemy import delete, func, literal, select, text, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from manicule.core.acquisition import (
    AcquiredSource,
    AcquisitionRun,
    AcquisitionSource,
    SnapshotCompleteness,
    SnapshotItemOutcome,
    SnapshotPromotionPolicy,
)
from manicule.core.errors import ManiculeError
from manicule.core.fingerprints import ChunkFingerprint
from manicule.core.ids import document_id, glossary_entry_id
from manicule.core.rebuild import (
    DerivedReplacement,
    MissingSnapshotInput,
    RebuildCheckpoint,
    RebuildEstimate,
    RebuildLeaseConflictError,
    RebuildPublicationConflictError,
    RebuildPublicationValidationError,
    RebuildRefusalCode,
    RebuildState,
    RebuildTarget,
    RebuildTerminalGenerationError,
    SnapshotRebuildInput,
    vector_publication_id,
)
from manicule.ingest.reembed import SnapshotChunk, SnapshotChunkDigester
from manicule.storage import models
from manicule.storage.acquisition import (
    AcquisitionConflictError,
    AcquisitionJournalMixin,
    settle_published_snapshot,
    snapshot_manifest_matches,
)
from manicule.storage.fts import (
    CREATE_TRIGGERS,
    DROP_TRIGGERS,
    FTS_TABLE,
    INTEGRITY_CHECK_FTS,
    REBUILD_FTS,
    create_fts,
)
from manicule.storage.rows import apply_document, from_chunk, to_chunk
from manicule.storage.scoped import WorkspaceScoped
from manicule.storage.types import utcnow

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Iterator, Sequence
    from contextlib import AbstractAsyncContextManager
    from datetime import datetime

    from sqlalchemy import CursorResult
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

    from manicule.core.content import Chunk

_REPLACEMENT = TypeAdapter(DerivedReplacement)
_EVIDENCE_PAGE = 1
_INVENTORY_PAGE = 100


class BlobInventory(Protocol):
    """Cheap local existence check used by planning; no bytes cross this boundary."""

    async def contains(self, digest: str) -> bool: ...

    def evidence_fence(self) -> AbstractAsyncContextManager[EvidencePinInventory]: ...

    async def evidence_identity(
        self,
        digest: str,
        *,
        size_bytes: int,
        stored_bytes: int,
        compression: str,
        verify_content: bool,
    ) -> str | None: ...


class EvidencePinInventory(Protocol):
    """Filesystem inode pins retained for the duration of atomic publication."""

    async def pin(
        self,
        digest: str,
        *,
        size_bytes: int,
        stored_bytes: int,
        compression: str,
    ) -> str | None: ...

    async def verify(
        self,
        digest: str,
        *,
        size_bytes: int,
        stored_bytes: int,
        compression: str,
    ) -> str | None: ...

    async def validate(
        self,
        digest: str,
        *,
        size_bytes: int,
        stored_bytes: int,
        compression: str,
    ) -> str | None: ...

    def publication_fence(self) -> AbstractAsyncContextManager[None]: ...


class _DigestHasher(Protocol):
    def update(self, value: bytes, /) -> None: ...


@dataclass(frozen=True, slots=True)
class _EvidenceDescriptor:
    sequence: int
    source_id: str
    blob_ref: str
    acquired_content_hash: str
    algo: str
    size_bytes: int
    stored_bytes: int
    compression: str


@dataclass(frozen=True, slots=True)
class _EvidenceRepresentation:
    blob_ref: str
    algo: str
    size_bytes: int
    stored_bytes: int
    compression: str


@dataclass(frozen=True, slots=True)
class _EvidenceFence:
    inventory_digest: str
    verification_digest: str
    lease_generation: int
    pins: EvidencePinInventory


class GenerationVectorInventory(Protocol):
    """Validation-only view of vectors staged under an unpublished publication id."""

    async def publication_is_complete(
        self,
        publication_id: str,
        chunks: Sequence[Chunk],
        *,
        embedding_fingerprint: str,
    ) -> bool: ...

    async def publication_row_count(self, publication_id: str) -> int: ...

    async def publication_page_is_complete(
        self,
        publication_id: str,
        chunks: Sequence[Chunk],
        *,
        embedding_fingerprint: str,
    ) -> bool: ...

    async def copy_publication(
        self, source_publication_id: str, target_publication_id: str, chunks: Sequence[Chunk]
    ) -> None: ...


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _generation_id(
    workspace_id: str,
    snapshot_run_id: str,
    target_digest: str,
    publication_identity_digest: str = "",
) -> str:
    digest = hashlib.blake2b(digest_size=20)
    for value in (
        workspace_id,
        snapshot_run_id,
        target_digest,
        publication_identity_digest,
    ):
        encoded = value.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _payload(replacement: DerivedReplacement) -> dict[str, Any]:
    dumped = replacement.model_dump(mode="json")
    # Document.publication_id is excluded from ordinary API serialization because it is an
    # internal hydration pointer. A shadow generation is the one persistence boundary where
    # omitting it would silently turn a replacement back into the legacy publication.
    document = cast("dict[str, Any]", dumped["document"])
    document["publication_id"] = replacement.document.publication_id
    return dumped


def _vector_pages(chunks: Sequence[Chunk], *, max_bytes: int) -> Iterator[tuple[Chunk, ...]]:
    """Build bounded vector work pages by encoded bytes, never merely row count."""
    budget = max(1, min(max_bytes // 4, 1024 * 1024))
    page: list[Chunk] = []
    held = 0
    for chunk in chunks:
        cost = len(chunk.model_dump_json().encode())
        if cost > budget:
            raise RebuildPublicationValidationError(RebuildRefusalCode.MEMORY_BOUND)
        if page and held + cost > budget:
            yield tuple(page)
            page = []
            held = 0
        page.append(chunk)
        held += cost
    if page:
        yield tuple(page)


def _checkpoint(
    row: models.DerivedGeneration, *, predecessor_vector_publication_id: str | None = None
) -> RebuildCheckpoint:
    return RebuildCheckpoint(
        generation_id=row.id,
        state=row.state,
        expected_items=row.expected_item_count,
        next_sequence=row.next_sequence,
        documents_built=row.documents_built,
        chunks_built=row.chunks_built,
        vectors_reused=row.vectors_reused,
        vectors_embedded=row.vectors_embedded,
        lease_owner=row.lease_owner,
        lease_generation=row.lease_generation,
        lease_expires_at=row.lease_expires_at,
        fence_generation=row.fence_generation,
        diagnostic_code=(
            RebuildRefusalCode(row.diagnostic_code) if row.diagnostic_code is not None else None
        ),
        diagnostic_count=row.diagnostic_count,
        predecessor_vector_publication_id=predecessor_vector_publication_id,
    )


class SqliteRebuildStore(WorkspaceScoped):
    """Workspace-scoped durable rebuild store.

    All shadow rows are invisible to existing readers. Publication applies every relational
    replacement and moves ``index_state`` plus the generation state in the same SQLite write
    transaction; WAL readers that started earlier continue reading the old snapshot.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        workspace_id: str,
        blobs: BlobInventory,
        vectors: GenerationVectorInventory | None = None,
        sessions: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        super().__init__(engine, workspace_id=workspace_id, sessions=sessions)
        self._blobs = blobs
        self._vectors = vectors
        self._acquisition = AcquisitionJournalMixin(
            engine, workspace_id=workspace_id, sessions=sessions
        )

    def _required_vectors(self) -> GenerationVectorInventory:
        if self._vectors is None:
            raise RuntimeError("this metadata-only rebuild store has no vector capability")
        return self._vectors

    async def _publication_row_count(self, publication_id: str) -> int:
        try:
            return await self._required_vectors().publication_row_count(publication_id)
        except (ManiculeError, ValueError) as exc:
            raise RebuildPublicationValidationError from exc

    async def _publication_page_is_complete(
        self,
        publication_id: str,
        chunks: Sequence[Chunk],
        *,
        embedding_fingerprint: str,
    ) -> bool:
        try:
            return await self._required_vectors().publication_page_is_complete(
                publication_id,
                chunks,
                embedding_fingerprint=embedding_fingerprint,
            )
        except (ManiculeError, ValueError) as exc:
            raise RebuildPublicationValidationError from exc

    async def plan_rebuild(  # noqa: PLR0911, PLR0912, PLR0915 - ordered validation boundary
        self,
        snapshot_run_id: str,
        target: RebuildTarget,
        *,
        missing_limit: int,
        persist: bool = True,
    ) -> RebuildEstimate:
        anchor = await self._acquisition.get_acquisition_run(snapshot_run_id)
        if anchor is None or anchor.promoted_at is None or not anchor.membership_hash:
            return self._refused_estimate(
                snapshot_run_id, target, RebuildRefusalCode.SNAPSHOT_NOT_PROMOTED
            )
        async with self._sessions() as session:
            latest_rows = await self._latest_promoted_runs(session)
        if snapshot_run_id not in {row.id for row in latest_rows}:
            return self._refused_estimate(
                snapshot_run_id, target, RebuildRefusalCode.SNAPSHOT_CHANGED
            )
        runs: list[AcquisitionRun] = []
        for row in latest_rows:
            candidate = await self._acquisition.get_acquisition_run(row.id)
            if (
                candidate is None
                or candidate.promoted_at is None
                or not candidate.membership_hash
                or not await self._acquisition.verify_snapshot_manifest(row.id)
            ):
                return self._refused_estimate(
                    snapshot_run_id, target, RebuildRefusalCode.SNAPSHOT_CHANGED
                )
            runs.append(candidate)
        missing: list[MissingSnapshotInput] = []
        missing_count = 0
        documents = 0
        known_bytes = 0
        largest_input = 0
        snapshot_bindings: list[tuple[AcquisitionRun, int]] = []
        global_sequence = 0
        for run in runs:
            manifest_items = 0
            retained_items = 0
            async for record in self._acquisition.iter_acquisition_records(run.id):
                if record.sequence != manifest_items:
                    return self._refused_estimate(
                        snapshot_run_id, target, RebuildRefusalCode.SNAPSHOT_CHANGED
                    )
                manifest_items += 1
                if (
                    record.blob_ref is None
                    or record.acquired_source is None
                    or not await self._blobs.contains(record.blob_ref)
                ):
                    if (
                        run.promotion_policy is SnapshotPromotionPolicy.ALLOW_OMISSIONS
                        and record.snapshot_outcome is SnapshotItemOutcome.OMITTED
                        and record.blob_ref is None
                        and record.acquired_source is None
                    ):
                        continue
                    missing_count += 1
                    if len(missing) < missing_limit:
                        missing.append(MissingSnapshotInput(sequence=global_sequence))
                else:
                    retained_items += 1
                    documents += 1
                    known_bytes += record.acquired_source.byte_length
                    largest_input = max(largest_input, record.acquired_source.byte_length)
                global_sequence += 1
            if manifest_items != run.discovered_count:
                return self._refused_estimate(
                    snapshot_run_id, target, RebuildRefusalCode.SNAPSHOT_CHANGED
                )
            snapshot_bindings.append((run, retained_items))

        combined_membership = hashlib.sha256(
            _canonical(
                [[run.id, run.membership_hash, retained] for run, retained in snapshot_bindings]
            )
        ).hexdigest()
        run_ids = tuple(run.id for run, _ in snapshot_bindings)
        canonical_run_id = run_ids[0]

        target_json = target.model_dump(mode="json")
        target_digest = hashlib.sha256(_canonical(target_json)).hexdigest()
        try:
            target_max_tokens = ChunkFingerprint.model_validate_json(
                target.chunk_fingerprint
            ).max_tokens
        except ValueError:
            # Third-party and legacy chunkers may use opaque canonical identities. Keep their
            # estimate conservative and retain the historical default without claiming that
            # the opaque value encodes a structural policy.
            target_max_tokens = 512
        current_chunk_fingerprint: str | None = None
        over_budget_chunks = 0
        max_stored_chunk_tokens = 0

        def estimate(generation_id: str) -> RebuildEstimate:
            # Conservative estimates are deterministic functions of retained byte counts. They
            # do not sample bodies, so a dry run cannot leak content or exceed its own bound.
            estimated_chunks = (known_bytes + target_max_tokens - 1) // target_max_tokens
            peak = largest_input * 6 + (4096 if documents else 0)
            temporary = known_bytes * 6 + estimated_chunks * 4096
            return RebuildEstimate(
                generation_id=generation_id,
                snapshot_run_id=snapshot_run_id,
                documents=documents,
                expected_items=documents,
                known_source_bytes=known_bytes,
                estimated_chunks=estimated_chunks,
                estimated_seconds=documents * 0.02 + known_bytes / 10_000_000,
                estimated_peak_memory_bytes=peak,
                estimated_temporary_bytes=temporary,
                missing_count=missing_count,
                missing=tuple(missing),
                missing_truncated=missing_count > len(missing),
                refusal=(RebuildRefusalCode.MISSING_LOCAL_INPUT if missing_count else None),
                current_chunk_fingerprint=current_chunk_fingerprint,
                target_chunk_fingerprint=target.chunk_fingerprint,
                over_budget_chunks=over_budget_chunks,
                max_stored_chunk_tokens=max_stored_chunk_tokens,
                estimated_embedding_chunks=estimated_chunks,
                network_required=False,
            )

        if not persist:
            async with self._sessions() as session:
                if not await self._snapshot_set_is_current(session, run_ids):
                    return self._refused_estimate(
                        snapshot_run_id, target, RebuildRefusalCode.WORKSPACE_SCOPE_CHANGED
                    )
                index_state = await session.get(models.IndexState, self._workspace_id)
                current_chunk_fingerprint = (
                    None if index_state is None else index_state.chunk_fingerprint
                )
                over_budget_chunks, max_stored_chunk_tokens = await self._chunk_budget_diagnostic(
                    session, current_chunk_fingerprint
                )
                expected_vector_table = None if index_state is None else index_state.vector_table
                expected_vector_inventory_digest = (
                    None if index_state is None else index_state.vector_inventory_digest
                )
                publication_identity_digest = hashlib.sha256(
                    _canonical(
                        {
                            "vector_table": expected_vector_table,
                            "vector_inventory_digest": expected_vector_inventory_digest,
                        }
                    )
                ).hexdigest()
                generation_id = _generation_id(
                    self._workspace_id,
                    f"{canonical_run_id}:{combined_membership}",
                    target_digest,
                    publication_identity_digest,
                )
                published = await self._published_replay(
                    session,
                    snapshot_run_id=canonical_run_id,
                    snapshot_membership_hash=combined_membership,
                    target_digest=target_digest,
                    vector_table=expected_vector_table,
                    vector_inventory_digest=expected_vector_inventory_digest,
                )
                if published is not None:
                    generation_id = published.id
            return estimate(generation_id)

        now = utcnow()
        async with self._sessions.begin() as session:
            # Take SQLite's writer slot before observing any global publication identity. This
            # makes the scope gate, binding read, published replay and insert one serializable
            # decision with respect to #187's pointer swap transaction.
            locked = cast(
                "CursorResult[Any]",
                await session.execute(
                    update(models.Workspace)
                    .where(models.Workspace.id == self._workspace_id)
                    .values(name=models.Workspace.name)
                ),
            )
            if locked.rowcount != 1:
                raise RuntimeError("rebuild workspace disappeared during planning")
            if not await self._snapshot_set_is_current(session, run_ids):
                return self._refused_estimate(
                    snapshot_run_id, target, RebuildRefusalCode.WORKSPACE_SCOPE_CHANGED
                )
            index_state = await session.get(models.IndexState, self._workspace_id)
            current_chunk_fingerprint = (
                None if index_state is None else index_state.chunk_fingerprint
            )
            over_budget_chunks, max_stored_chunk_tokens = await self._chunk_budget_diagnostic(
                session, current_chunk_fingerprint
            )
            expected_vector_table = None if index_state is None else index_state.vector_table
            expected_vector_inventory_digest = (
                None if index_state is None else index_state.vector_inventory_digest
            )
            publication_identity_digest = hashlib.sha256(
                _canonical(
                    {
                        "vector_table": expected_vector_table,
                        "vector_inventory_digest": expected_vector_inventory_digest,
                    }
                )
            ).hexdigest()
            generation_id = _generation_id(
                self._workspace_id,
                f"{canonical_run_id}:{combined_membership}",
                target_digest,
                publication_identity_digest,
            )
            # A successful publication changes the live inventory digest by design. Recognize
            # that exact resulting corpus before constructing a new identity, so repeating the
            # same dry-run or run is idempotent instead of conflicting with its own result.
            published = await self._published_replay(
                session,
                snapshot_run_id=canonical_run_id,
                snapshot_membership_hash=combined_membership,
                target_digest=target_digest,
                vector_table=expected_vector_table,
                vector_inventory_digest=expected_vector_inventory_digest,
            )
            if published is not None:
                generation_id = published.id
            else:
                await session.execute(
                    sqlite_insert(models.DerivedGeneration)
                    .values(
                        id=generation_id,
                        workspace_id=self._workspace_id,
                        snapshot_run_id=canonical_run_id,
                        target_digest=target_digest,
                        publication_identity_digest=publication_identity_digest,
                        target=target_json,
                        snapshot_membership_hash=combined_membership,
                        expected_item_count=documents,
                        state=RebuildState.PLANNED,
                        next_sequence=0,
                        documents_built=0,
                        chunks_built=0,
                        vectors_reused=0,
                        vectors_embedded=0,
                        vector_publication_id=None,
                        expected_vector_table=expected_vector_table,
                        expected_vector_inventory_digest=expected_vector_inventory_digest,
                        lease_generation=0,
                        diagnostic_count=0,
                        created_at=now,
                        updated_at=now,
                    )
                    .on_conflict_do_nothing(
                        index_elements=[
                            models.DerivedGeneration.workspace_id,
                            models.DerivedGeneration.snapshot_run_id,
                            models.DerivedGeneration.snapshot_membership_hash,
                            models.DerivedGeneration.target_digest,
                            models.DerivedGeneration.publication_identity_digest,
                        ]
                    )
                )
                for ordinal, (bound_run, retained) in enumerate(snapshot_bindings):
                    await session.execute(
                        sqlite_insert(models.DerivedGenerationSnapshot)
                        .values(
                            generation_id=generation_id,
                            ordinal=ordinal,
                            run_id=bound_run.id,
                            connector_name=bound_run.connector,
                            scope_fingerprint=bound_run.scope_fingerprint,
                            membership_hash=bound_run.membership_hash,
                            expected_item_count=retained,
                        )
                        .on_conflict_do_nothing(
                            index_elements=[
                                models.DerivedGenerationSnapshot.generation_id,
                                models.DerivedGenerationSnapshot.ordinal,
                            ]
                        )
                    )
            generation = await self._required_generation(session, generation_id)
            persisted_bindings = await self._generation_snapshot_rows(session, generation_id)
            expected_bindings = [
                (
                    ordinal,
                    bound_run.id,
                    bound_run.connector,
                    bound_run.scope_fingerprint,
                    bound_run.membership_hash,
                    retained,
                )
                for ordinal, (bound_run, retained) in enumerate(snapshot_bindings)
            ]
            if (
                generation.snapshot_membership_hash != combined_membership
                or generation.expected_item_count != documents
                or [
                    (
                        binding.ordinal,
                        binding.run_id,
                        binding.connector_name,
                        binding.scope_fingerprint,
                        binding.membership_hash,
                        binding.expected_item_count,
                    )
                    for binding in persisted_bindings
                ]
                != expected_bindings
                or (
                    generation.state is not RebuildState.PUBLISHED
                    and (
                        generation.expected_vector_table != expected_vector_table
                        or generation.expected_vector_inventory_digest
                        != expected_vector_inventory_digest
                        or generation.publication_identity_digest != publication_identity_digest
                    )
                )
            ):
                raise RuntimeError("an existing rebuild plan names different snapshot evidence")

        return estimate(generation_id)

    async def _chunk_budget_diagnostic(
        self, session: AsyncSession, canonical: str | None
    ) -> tuple[int, int]:
        """Aggregate-safe live stored-count audit; no chunk or source identity is selected."""
        budget: int | None = None
        if canonical is not None:
            try:
                budget = ChunkFingerprint.model_validate_json(canonical).max_tokens
            except ValueError:
                budget = None
        over = func.count().filter(models.Chunk.token_count > budget) if budget else literal(0)
        statement = (
            select(over, func.coalesce(func.max(models.Chunk.token_count), 0))
            .select_from(models.Chunk)
            .join(models.Document, models.Document.id == models.Chunk.document_id)
            .where(
                models.Document.workspace_id == self._workspace_id,
                models.Document.deleted_at.is_(None),
            )
        )
        found, maximum = (await session.execute(statement)).one()
        return int(found), int(maximum)

    async def _published_replay(
        self,
        session: AsyncSession,
        *,
        snapshot_run_id: str,
        snapshot_membership_hash: str,
        target_digest: str,
        vector_table: str | None,
        vector_inventory_digest: str | None,
    ) -> models.DerivedGeneration | None:
        return (
            await session.execute(
                select(models.DerivedGeneration)
                .where(
                    models.DerivedGeneration.workspace_id == self._workspace_id,
                    models.DerivedGeneration.snapshot_run_id == snapshot_run_id,
                    models.DerivedGeneration.snapshot_membership_hash == snapshot_membership_hash,
                    models.DerivedGeneration.target_digest == target_digest,
                    models.DerivedGeneration.state == RebuildState.PUBLISHED,
                    models.DerivedGeneration.expected_vector_table == vector_table,
                    models.DerivedGeneration.published_vector_inventory_digest
                    == vector_inventory_digest,
                )
                .order_by(models.DerivedGeneration.published_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

    def _refused_estimate(
        self,
        snapshot_run_id: str,
        target: RebuildTarget,
        code: RebuildRefusalCode,
    ) -> RebuildEstimate:
        target_digest = hashlib.sha256(_canonical(target.model_dump(mode="json"))).hexdigest()
        return RebuildEstimate(
            generation_id=_generation_id(self._workspace_id, snapshot_run_id, target_digest),
            snapshot_run_id=snapshot_run_id,
            documents=0,
            expected_items=0,
            known_source_bytes=0,
            estimated_chunks=0,
            estimated_seconds=0,
            estimated_peak_memory_bytes=0,
            estimated_temporary_bytes=0,
            missing_count=0,
            refusal=code,
            target_chunk_fingerprint=target.chunk_fingerprint,
            network_required=False,
        )

    async def checkpoint(self, generation_id: str) -> RebuildCheckpoint:
        async with self._sessions() as session:
            return _checkpoint(await self._required_generation(session, generation_id))

    async def claim_generation(
        self,
        generation_id: str,
        owner: str,
        *,
        now: datetime,
        expires_at: datetime,
    ) -> RebuildCheckpoint:
        if not owner or expires_at <= now:
            raise ValueError("a generation lease needs an owner and a future expiry")
        async with self._sessions() as session:
            observed = await self._required_generation(session, generation_id)
            published = observed.state is RebuildState.PUBLISHED
        if published:
            return await self._repair_published_generation(generation_id, now=now)
        async with self._sessions.begin() as session:
            # Take SQLite's writer slot before reading max(fence_generation). The unique index
            # is the second guard; this ordering avoids turning ordinary concurrent claims into
            # a retryable uniqueness failure.
            await session.execute(
                update(models.DerivedGeneration)
                .where(
                    models.DerivedGeneration.id == generation_id,
                    models.DerivedGeneration.workspace_id == self._workspace_id,
                )
                .values(updated_at=models.DerivedGeneration.updated_at)
            )
            generation = await self._required_generation(session, generation_id)
            if generation.state is RebuildState.PUBLISHED:
                # A publisher won after the read-only probe. Release SQLite's writer slot
                # before the evidence preflight, then re-enter through the fenced replay path.
                await session.rollback()
                return await self._repair_published_generation(generation_id, now=now)
            if generation.state in {RebuildState.FAILED, RebuildState.CANCELED}:
                raise RebuildTerminalGenerationError("generation is terminal")
            if generation.lease_expires_at is not None and generation.lease_expires_at > now:
                if generation.lease_owner != owner:
                    raise RebuildLeaseConflictError("generation has another live owner")
                return _checkpoint(generation)
            predecessor = generation.vector_publication_id
            if generation.fence_generation is None:
                highest = cast(
                    "int",
                    (
                        await session.execute(
                            select(
                                func.coalesce(
                                    func.max(models.DerivedGeneration.fence_generation), 0
                                )
                            ).where(models.DerivedGeneration.workspace_id == self._workspace_id)
                        )
                    ).scalar_one(),
                )
                generation.fence_generation = highest + 1
            generation.lease_owner = owner
            generation.lease_generation += 1
            generation.lease_expires_at = expires_at
            generation.state = RebuildState.BUILDING
            generation.updated_at = now
            await session.flush()
            return _checkpoint(generation, predecessor_vector_publication_id=predecessor)

    async def _repair_published_generation(
        self, generation_id: str, *, now: datetime
    ) -> RebuildCheckpoint:
        async with (
            self._publication_evidence_fence(generation_id) as evidence_fence,
            self._sessions.begin() as session,
        ):
            generation = await self._required_generation(session, generation_id)
            if generation.state is not RebuildState.PUBLISHED:
                raise RebuildPublicationConflictError(RebuildRefusalCode.PUBLICATION_CONFLICT)
            await self._require_evidence_fence(session, generation, evidence_fence)
            await self._settle_published_generation(session, generation, now=now)
            await session.flush()
            await self._require_final_evidence_fence(session, generation, evidence_fence)
            return _checkpoint(generation)

    async def renew_generation(
        self,
        generation_id: str,
        owner: str,
        lease_generation: int,
        *,
        now: datetime,
        expires_at: datetime,
    ) -> RebuildCheckpoint:
        if expires_at <= now:
            raise ValueError("renewal expiry must be in the future")
        async with self._sessions.begin() as session:
            generation = await self._required_generation(session, generation_id)
            self._require_lease(generation, owner, lease_generation, now)
            generation.lease_expires_at = expires_at
            generation.updated_at = now
            return _checkpoint(generation)

    async def assert_generation_lease(
        self,
        generation_id: str,
        owner: str,
        lease_generation: int,
        *,
        now: datetime,
    ) -> None:
        """Fence an external mutation immediately at its await boundary."""
        async with self._sessions() as session:
            generation = await self._required_generation(session, generation_id)
            self._require_lease(generation, owner, lease_generation, now)
            if generation.state not in {RebuildState.BUILDING, RebuildState.VALIDATING}:
                raise RebuildLeaseConflictError("generation is not accepting fenced work")
            snapshots = await self._generation_snapshot_rows(session, generation.id)
            if not snapshots or not await self._snapshot_set_is_current(
                session, tuple(row.run_id for row in snapshots)
            ):
                raise RebuildPublicationConflictError(RebuildRefusalCode.WORKSPACE_SCOPE_CHANGED)
            await self._require_live_vector_binding(session, generation)

    async def copy_checkpointed_vectors(  # noqa: PLR0912 - explicit replay validation stages
        self,
        generation_id: str,
        source_publication_id: str,
        *,
        owner: str,
        lease_generation: int,
        now: datetime,
        cancel: asyncio.Event | None = None,
    ) -> None:
        """Replay only durable item vectors into a takeover's fresh physical namespace."""
        target_publication = vector_publication_id(generation_id, owner, lease_generation)
        after = -1
        expected_vectors = 0
        checkpoint_sequence: int | None = None
        while True:
            if cancel is not None and cancel.is_set():
                raise asyncio.CancelledError
            async with self._sessions() as session:
                generation = await self._required_generation(session, generation_id)
                self._require_lease(generation, owner, lease_generation, now)
                if checkpoint_sequence is None:
                    checkpoint_sequence = generation.next_sequence
                elif generation.next_sequence != checkpoint_sequence:
                    raise RebuildLeaseConflictError("checkpoint advanced during vector replay")
                target = RebuildTarget.model_validate(generation.target)
                rows = list(
                    (
                        await session.execute(
                            select(models.DerivedGenerationItem)
                            .where(
                                models.DerivedGenerationItem.generation_id == generation_id,
                                models.DerivedGenerationItem.sequence > after,
                                models.DerivedGenerationItem.sequence < generation.next_sequence,
                            )
                            .order_by(models.DerivedGenerationItem.sequence)
                            .limit(_EVIDENCE_PAGE)
                        )
                    ).scalars()
                )
            if not rows:
                break
            for row in rows:
                try:
                    replacement = _REPLACEMENT.validate_python(row.payload)
                except (RuntimeError, ValueError) as exc:
                    raise RebuildPublicationValidationError from exc
                chunks = replacement.flattened_chunks()
                expected_vectors += len(chunks)
                for page in _vector_pages(chunks, max_bytes=target.max_memory_bytes):
                    checked_at = utcnow()
                    await self.assert_generation_lease(
                        generation_id,
                        owner,
                        lease_generation,
                        now=checked_at,
                    )
                    try:
                        await self._required_vectors().copy_publication(
                            source_publication_id,
                            target_publication,
                            page,
                        )
                        page_complete = await self._publication_page_is_complete(
                            target_publication,
                            page,
                            embedding_fingerprint=target.embedding_fingerprint,
                        )
                    except (ManiculeError, ValueError) as exc:
                        raise RebuildPublicationValidationError from exc
                    if not page_complete:
                        raise RebuildPublicationValidationError
                    if cancel is not None and cancel.is_set():
                        raise asyncio.CancelledError
                    await self.assert_generation_lease(
                        generation_id,
                        owner,
                        lease_generation,
                        now=utcnow(),
                    )
            after = rows[-1].sequence
        if await self._publication_row_count(target_publication) != expected_vectors:
            raise RebuildPublicationValidationError
        async with self._sessions.begin() as session:
            generation = await self._required_generation(session, generation_id)
            self._require_lease(generation, owner, lease_generation, utcnow())
            if generation.next_sequence != checkpoint_sequence:
                raise RebuildLeaseConflictError("checkpoint advanced during vector replay")
            generation.vector_publication_id = target_publication
            generation.updated_at = utcnow()

    async def snapshot_inputs(
        self, generation_id: str, *, after_sequence: int, limit: int
    ) -> Sequence[SnapshotRebuildInput]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        async with self._sessions() as session:
            await self._required_generation(session, generation_id)
            snapshots = await self._generation_snapshot_rows(session, generation_id)
            if not snapshots or not await self._snapshot_set_is_current(
                session, tuple(row.run_id for row in snapshots)
            ):
                raise RebuildPublicationConflictError(RebuildRefusalCode.SNAPSHOT_CHANGED)
            skip = after_sequence + 1
            result: list[SnapshotRebuildInput] = []
            for snapshot in snapshots:
                if skip >= snapshot.expected_item_count:
                    skip -= snapshot.expected_item_count
                    continue
                ranked = (
                    select(
                        models.AcquisitionRecord.id.label("record_id"),
                        func.row_number()
                        .over(order_by=models.AcquisitionRecord.sequence)
                        .label("retained_ordinal"),
                    )
                    .where(
                        models.AcquisitionRecord.run_id == snapshot.run_id,
                        models.AcquisitionRecord.workspace_id == self._workspace_id,
                        models.AcquisitionRecord.blob_ref.is_not(None),
                        models.AcquisitionRecord.acquired_source.is_not(None),
                    )
                    .subquery()
                )
                records = list(
                    (
                        await session.execute(
                            select(models.AcquisitionRecord)
                            .join(ranked, ranked.c.record_id == models.AcquisitionRecord.id)
                            .where(
                                ranked.c.retained_ordinal > skip,
                            )
                            .order_by(ranked.c.retained_ordinal)
                            .limit(limit - len(result))
                        )
                    ).scalars()
                )
                skip = 0
                for record in records:
                    if record.blob_ref is None or record.acquired_source is None:
                        raise RebuildPublicationConflictError(RebuildRefusalCode.SNAPSHOT_CHANGED)
                    acquired = AcquiredSource.model_validate(record.acquired_source)
                    source = AcquisitionSource.model_validate(record.source_record)
                    result.append(
                        SnapshotRebuildInput(
                            sequence=after_sequence + 1 + len(result),
                            connector=snapshot.connector_name,
                            blob_ref=record.blob_ref,
                            source=acquired,
                            title=source.title,
                            version_token=(record.fetched_version_token or source.version_token),
                        )
                    )
                if len(result) == limit:
                    break
            return result

    async def check_capacity(
        self, generation_id: str, replacements: Sequence[DerivedReplacement]
    ) -> None:
        async with self._sessions() as session:
            generation = await self._required_generation(session, generation_id)
            target = RebuildTarget.model_validate(generation.target)
            held = (
                await session.execute(
                    select(
                        func.coalesce(func.sum(models.DerivedGenerationItem.temporary_bytes), 0)
                    ).where(models.DerivedGenerationItem.generation_id == generation_id)
                )
            ).scalar_one()
            additional = sum(
                self._temporary_cost(item, target, len(_canonical(_payload(item))))
                for item in replacements
            )
            if held + additional > target.max_temporary_bytes:
                raise RuntimeError(RebuildRefusalCode.TEMP_DISK_BOUND.value)

    async def stage_replacements(
        self,
        generation_id: str,
        replacements: Sequence[tuple[int, DerivedReplacement]],
        *,
        expected_next_sequence: int,
        owner: str,
        lease_generation: int,
        now: datetime,
    ) -> RebuildCheckpoint:
        if not replacements:
            raise RebuildPublicationValidationError
        if [sequence for sequence, _ in replacements] != list(
            range(expected_next_sequence, expected_next_sequence + len(replacements))
        ):
            raise RebuildPublicationValidationError
        async with self._sessions.begin() as session:
            generation = await self._required_generation(session, generation_id)
            self._require_lease(generation, owner, lease_generation, now)
            if generation.state is not RebuildState.BUILDING:
                raise RebuildPublicationConflictError(RebuildRefusalCode.PUBLICATION_CONFLICT)
            if generation.next_sequence != expected_next_sequence:
                raise RebuildPublicationConflictError(RebuildRefusalCode.PUBLICATION_CONFLICT)
            current_vector_publication = vector_publication_id(
                generation_id, owner, lease_generation
            )
            if generation.next_sequence > 0 and (
                generation.vector_publication_id != current_vector_publication
            ):
                raise RebuildLeaseConflictError("checkpoint vectors were not replayed")
            target = RebuildTarget.model_validate(generation.target)
            for sequence, replacement in replacements:
                try:
                    replacement.validate_identity()
                except ValueError as exc:
                    raise RebuildPublicationValidationError from exc
                if replacement.document.publication_id != generation_id:
                    raise RebuildPublicationValidationError
                payload = _payload(replacement)
                encoded = _canonical(payload)
                temporary_bytes = self._temporary_cost(replacement, target, len(encoded))
                digest = hashlib.sha256(encoded).hexdigest()
                existing = await session.get(
                    models.DerivedGenerationItem, (generation_id, sequence)
                )
                if existing is not None:
                    if existing.payload_digest != digest:
                        raise RebuildPublicationValidationError
                    continue
                session.add(
                    models.DerivedGenerationItem(
                        generation_id=generation_id,
                        sequence=sequence,
                        payload_digest=digest,
                        document_id=replacement.document.id,
                        payload=payload,
                        temporary_bytes=temporary_bytes,
                        created_at=utcnow(),
                    )
                )
            await session.flush()
            total_bytes = (
                await session.execute(
                    select(
                        func.coalesce(func.sum(models.DerivedGenerationItem.temporary_bytes), 0)
                    ).where(models.DerivedGenerationItem.generation_id == generation_id)
                )
            ).scalar_one()
            if total_bytes > target.max_temporary_bytes:
                raise RuntimeError(RebuildRefusalCode.TEMP_DISK_BOUND.value)
            generation.next_sequence += len(replacements)
            generation.documents_built += len(replacements)
            generation.chunks_built += sum(len(item.flattened_chunks()) for _, item in replacements)
            generation.vectors_reused += sum(
                nested.vector_reused for _, item in replacements for nested in item.flattened()
            )
            generation.vectors_embedded += sum(
                nested.vector_embedded for _, item in replacements for nested in item.flattened()
            )
            generation.vector_publication_id = current_vector_publication
            generation.updated_at = now
            await session.flush()
            return _checkpoint(generation)

    async def begin_validation(
        self,
        generation_id: str,
        *,
        owner: str,
        lease_generation: int,
        now: datetime,
    ) -> RebuildCheckpoint:
        async with self._sessions.begin() as session:
            generation = await self._required_generation(session, generation_id)
            self._require_lease(generation, owner, lease_generation, now)
            if generation.state is RebuildState.PUBLISHED:
                return _checkpoint(generation)
            if generation.state not in {RebuildState.BUILDING, RebuildState.VALIDATING}:
                raise RebuildPublicationConflictError(RebuildRefusalCode.PUBLICATION_CONFLICT)
            generation.state = RebuildState.VALIDATING
            generation.updated_at = now
            return _checkpoint(generation)

    async def validate_generation(self, generation_id: str) -> None:
        async with self._sessions() as session:
            generation = await self._required_generation(session, generation_id)
            if generation.state not in {RebuildState.VALIDATING, RebuildState.PUBLISHED}:
                raise RebuildPublicationValidationError
            await self._verify_complete_header(session, generation)
            target = RebuildTarget.model_validate(generation.target)
            physical_publication = vector_publication_id(
                generation_id, generation.lease_owner or "", generation.lease_generation
            )
            expected_vectors = 0
            after = -1
            while after + 1 < generation.expected_item_count:
                pairs = await self._evidence_page(session, generation, after=after)
                for row, snapshot in pairs:
                    replacement = self._validated_replacement(row, snapshot)
                    chunks = replacement.flattened_chunks()
                    expected_vectors += len(chunks)
                    for page in _vector_pages(chunks, max_bytes=target.max_memory_bytes):
                        if not await self._publication_page_is_complete(
                            physical_publication,
                            page,
                            embedding_fingerprint=target.embedding_fingerprint,
                        ):
                            raise RebuildPublicationValidationError
                after = pairs[-1][0].sequence
            if await self._publication_row_count(physical_publication) != expected_vectors:
                raise RebuildPublicationValidationError
        await self._record_evidence_verification(generation_id)

    async def publish_generation(
        self,
        generation_id: str,
        *,
        owner: str,
        lease_generation: int,
        now: datetime,
    ) -> RebuildCheckpoint:
        await self._assert_prepublication_scope(generation_id)
        async with self._publication_evidence_fence(generation_id) as evidence_fence:
            return await self._publish_generation_atomic(
                generation_id,
                owner=owner,
                lease_generation=lease_generation,
                now=now,
                evidence_fence=evidence_fence,
            )

    async def _assert_prepublication_scope(self, generation_id: str) -> None:
        """Preserve typed scope/snapshot refusal precedence before evidence-fence checks."""
        async with self._sessions() as session:
            generation = await self._required_generation(session, generation_id)
            if generation.state is RebuildState.PUBLISHED:
                return
            snapshots = await self._generation_snapshot_rows(session, generation.id)
            if not snapshots or not await self._snapshot_set_is_current(
                session, tuple(row.run_id for row in snapshots)
            ):
                raise RebuildPublicationConflictError(RebuildRefusalCode.WORKSPACE_SCOPE_CHANGED)

    async def _publish_generation_atomic(  # noqa: PLR0912, PLR0915
        self,
        generation_id: str,
        *,
        owner: str,
        lease_generation: int,
        now: datetime,
        evidence_fence: _EvidenceFence,
    ) -> RebuildCheckpoint:
        async with self._sessions.begin() as session:
            existing = await self._required_generation(session, generation_id)
            await self._require_evidence_fence(session, existing, evidence_fence)
            if existing.state is RebuildState.PUBLISHED:
                await self._settle_published_generation(session, existing, now=now)
                await session.flush()
                await self._require_final_evidence_fence(session, existing, evidence_fence)
                return _checkpoint(existing)
            claimed = cast(
                "CursorResult[Any]",
                await session.execute(
                    update(models.DerivedGeneration)
                    .where(
                        models.DerivedGeneration.id == generation_id,
                        models.DerivedGeneration.workspace_id == self._workspace_id,
                        models.DerivedGeneration.state == RebuildState.VALIDATING,
                        models.DerivedGeneration.lease_owner == owner,
                        models.DerivedGeneration.lease_generation == lease_generation,
                        models.DerivedGeneration.lease_expires_at > now,
                    )
                    .values(updated_at=models.DerivedGeneration.updated_at)
                ),
            )
            if claimed.rowcount != 1:
                raise RebuildLeaseConflictError("generation lease changed or expired")
            generation = await self._required_generation(session, generation_id)
            self._require_lease(generation, owner, lease_generation, now)
            if generation.state is not RebuildState.VALIDATING:
                raise RebuildPublicationValidationError
            snapshots = await self._generation_snapshot_rows(session, generation.id)
            if not snapshots or not await self._snapshot_set_is_current(
                session, tuple(row.run_id for row in snapshots)
            ):
                raise RebuildPublicationConflictError(RebuildRefusalCode.WORKSPACE_SCOPE_CHANGED)
            snapshot_runs = {
                run.id: run
                for run in (
                    await session.execute(
                        select(models.AcquisitionRun).where(
                            models.AcquisitionRun.id.in_(
                                tuple(snapshot.run_id for snapshot in snapshots)
                            )
                        )
                    )
                ).scalars()
            }
            if len(snapshot_runs) != len(snapshots):
                raise RebuildPublicationConflictError(RebuildRefusalCode.SNAPSHOT_CHANGED)
            await self._require_live_vector_binding(session, generation)

            highest_fence = (
                await session.execute(
                    select(func.max(models.DerivedGeneration.fence_generation)).where(
                        models.DerivedGeneration.workspace_id == self._workspace_id,
                        models.DerivedGeneration.state.in_(
                            (
                                RebuildState.BUILDING,
                                RebuildState.VALIDATING,
                                RebuildState.PUBLISHED,
                            )
                        ),
                    )
                )
            ).scalar_one_or_none()
            if highest_fence != generation.fence_generation:
                raise RebuildPublicationConflictError(RebuildRefusalCode.PUBLICATION_CONFLICT)

            await self._verify_complete_header(session, generation)
            target = RebuildTarget.model_validate(generation.target)
            physical_publication = vector_publication_id(generation_id, owner, lease_generation)
            # Ask SQLite to parse the tokenizer mini-language before deleting a single live row.
            probe = "__manicule_rebuild_fts_probe"
            probe_sql = create_fts(target.fts_tokenizer).replace(FTS_TABLE, probe)
            await session.execute(text(f"DROP TABLE IF EXISTS {probe}"))
            await session.execute(text(probe_sql))
            await session.execute(text(f"DROP TABLE {probe}"))

            expected_vectors = 0
            after = -1
            while after + 1 < generation.expected_item_count:
                pairs = await self._evidence_page(session, generation, after=after)
                for row, snapshot in pairs:
                    replacement = self._validated_replacement(row, snapshot)
                    chunks = replacement.flattened_chunks()
                    expected_vectors += len(chunks)
                    for page in _vector_pages(chunks, max_bytes=target.max_memory_bytes):
                        if not await self._publication_page_is_complete(
                            physical_publication,
                            page,
                            embedding_fingerprint=target.embedding_fingerprint,
                        ):
                            raise RebuildPublicationValidationError
                    await self._publish_item(
                        session,
                        replacement=replacement,
                        generation_id=generation_id,
                        vector_publication=physical_publication,
                        connector_name=replacement.document.source,
                        target=target,
                        snapshot=snapshot,
                    )
                after = pairs[-1][0].sequence
            if await self._publication_row_count(physical_publication) != expected_vectors:
                raise RebuildPublicationValidationError

            connectors = {snapshot.connector_name for snapshot in snapshots}
            complete_connectors = {
                connector
                for connector in connectors
                if all(
                    snapshot.connector_name != connector
                    or (
                        snapshot_runs[snapshot.run_id].completeness is not None
                        and SnapshotCompleteness(snapshot_runs[snapshot.run_id].completeness)
                        is SnapshotCompleteness.COMPLETE
                    )
                    for snapshot in snapshots
                )
            }
            for connector in complete_connectors:
                await session.execute(
                    update(models.Document)
                    .where(
                        models.Document.workspace_id == self._workspace_id,
                        models.Document.source == connector,
                        models.Document.deleted_at.is_(None),
                        models.Document.publication_id != physical_publication,
                    )
                    .values(deleted_at=utcnow())
                )

            state = await session.get(models.IndexState, self._workspace_id)
            tokenizer_rows = (
                (await session.execute(select(models.IndexState.fts_tokenizer).distinct()))
                .scalars()
                .all()
            )
            installed_tokenizers = {
                models.FTS_TOKENIZER if value is None else str(value) for value in tokenizer_rows
            }
            if len(installed_tokenizers) > 1:
                raise RebuildPublicationConflictError(RebuildRefusalCode.PUBLICATION_CONFLICT)
            current_tokenizer = (
                next(iter(installed_tokenizers)) if installed_tokenizers else models.FTS_TOKENIZER
            )
            if state is None:
                state = models.IndexState(
                    workspace_id=self._workspace_id, vector_namespace="workspace"
                )
                session.add(state)
            if current_tokenizer != target.fts_tokenizer:
                foreign_identity = int(
                    await session.scalar(
                        select(func.count(models.IndexState.workspace_id)).where(
                            models.IndexState.workspace_id != self._workspace_id
                        )
                    )
                    or 0
                )
                if foreign_identity:
                    # FTS5 is intentionally installation-global.  A workspace-private vector
                    # identity does not authorize rebuilding that shared table underneath a
                    # different workspace's published lexical index.
                    raise RebuildPublicationConflictError(
                        RebuildRefusalCode.WORKSPACE_SCOPE_CHANGED
                    )
                for statement in DROP_TRIGGERS:
                    await session.execute(text(statement))
                await session.execute(text(f"DROP TABLE IF EXISTS {FTS_TABLE}"))
                await session.execute(text(create_fts(target.fts_tokenizer)))
                for statement in CREATE_TRIGGERS:
                    await session.execute(text(statement))
                await session.execute(text(REBUILD_FTS))
            await session.execute(text(INTEGRITY_CHECK_FTS))
            state.chunk_fingerprint = target.chunk_fingerprint
            state.embed_fingerprint = target.embedding_config or target.embedding_fingerprint
            state.fts_tokenizer = target.fts_tokenizer
            state.vector_inventory_digest = await self._live_chunk_inventory_digest(session)
            generation.published_vector_inventory_digest = state.vector_inventory_digest
            generation.state = RebuildState.PUBLISHED
            generation.published_at = now
            generation.updated_at = now
            await self._settle_published_generation(session, generation, now=now)
            await session.flush()
            await self._require_final_evidence_fence(session, generation, evidence_fence)
            return _checkpoint(generation)

    async def _require_evidence_fence(
        self,
        session: AsyncSession,
        generation: models.DerivedGeneration,
        evidence_fence: _EvidenceFence,
    ) -> None:
        if (
            generation.evidence_inventory_digest != evidence_fence.inventory_digest
            or generation.evidence_verification_digest != evidence_fence.verification_digest
            or generation.evidence_verification_lease_generation != evidence_fence.lease_generation
            or generation.lease_generation != evidence_fence.lease_generation
        ):
            raise RebuildPublicationConflictError(RebuildRefusalCode.SNAPSHOT_CHANGED)
        if (
            await self._evidence_inventory_digest(session, generation)
            != evidence_fence.inventory_digest
        ):
            raise RebuildPublicationConflictError(RebuildRefusalCode.SNAPSHOT_CHANGED)

    async def _require_final_evidence_fence(
        self,
        session: AsyncSession,
        generation: models.DerivedGeneration,
        evidence_fence: _EvidenceFence,
    ) -> None:
        """Perform the last fd+name inode probe after flush and immediately before commit."""
        observed = await self._evidence_verification_digest(
            session,
            generation,
            evidence_fence.inventory_digest,
            verify_content=False,
            pins=evidence_fence.pins,
            final_probe=True,
        )
        if observed != evidence_fence.verification_digest:
            raise RebuildPublicationConflictError(RebuildRefusalCode.SNAPSHOT_CHANGED)

    async def _settle_published_generation(
        self,
        session: AsyncSession,
        generation: models.DerivedGeneration,
        *,
        now: datetime,
    ) -> None:
        """Join the live-generation flip to its exact source-work settlement."""
        await self._verify_complete_header(session, generation)
        snapshots = await self._generation_snapshot_rows(session, generation.id)
        for snapshot in snapshots:
            try:
                await settle_published_snapshot(
                    session,
                    workspace_id=self._workspace_id,
                    run_id=snapshot.run_id,
                    expected_membership_hash=snapshot.membership_hash,
                    expected_item_count=snapshot.expected_item_count,
                    derived_generation_identity=generation.id,
                    now=now,
                )
            except AcquisitionConflictError as exc:
                raise RebuildPublicationConflictError(RebuildRefusalCode.SNAPSHOT_CHANGED) from exc

    async def _require_live_vector_binding(
        self, session: AsyncSession, generation: models.DerivedGeneration
    ) -> None:
        state = await session.get(models.IndexState, self._workspace_id)
        observed_table = None if state is None else state.vector_table
        observed_inventory = None if state is None else state.vector_inventory_digest
        if (
            observed_table != generation.expected_vector_table
            or observed_inventory != generation.expected_vector_inventory_digest
        ):
            raise RebuildPublicationConflictError(RebuildRefusalCode.PUBLICATION_CONFLICT)

    async def _latest_promoted_runs(self, session: AsyncSession) -> list[models.AcquisitionRun]:
        ranked = (
            select(
                models.AcquisitionRun.id.label("run_id"),
                func.row_number()
                .over(
                    partition_by=models.AcquisitionRun.connector_name,
                    order_by=(
                        models.AcquisitionRun.promoted_at.desc(),
                        models.AcquisitionRun.created_at.desc(),
                        models.AcquisitionRun.id.desc(),
                    ),
                )
                .label("scope_rank"),
            )
            .join(
                models.Connector,
                (models.Connector.id == models.AcquisitionRun.connector_id)
                & (models.Connector.workspace_id == models.AcquisitionRun.workspace_id),
            )
            .where(
                models.AcquisitionRun.workspace_id == self._workspace_id,
                models.AcquisitionRun.promoted_at.is_not(None),
                models.AcquisitionRun.superseded_at.is_(None),
                models.Connector.deleted_at.is_(None),
            )
            .subquery()
        )
        return list(
            (
                await session.execute(
                    select(models.AcquisitionRun)
                    .join(ranked, ranked.c.run_id == models.AcquisitionRun.id)
                    .where(ranked.c.scope_rank == 1)
                    .order_by(models.AcquisitionRun.connector_name)
                )
            ).scalars()
        )

    async def _generation_snapshot_rows(
        self, session: AsyncSession, generation_id: str
    ) -> list[models.DerivedGenerationSnapshot]:
        return list(
            (
                await session.execute(
                    select(models.DerivedGenerationSnapshot)
                    .where(models.DerivedGenerationSnapshot.generation_id == generation_id)
                    .order_by(models.DerivedGenerationSnapshot.ordinal)
                )
            ).scalars()
        )

    async def _snapshot_set_is_current(self, session: AsyncSession, run_ids: Sequence[str]) -> bool:
        latest = await self._latest_promoted_runs(session)
        if tuple(row.id for row in latest) != tuple(run_ids):
            return False
        connectors = {row.connector_name for row in latest}
        foreign_live = await session.scalar(
            select(func.count(models.Document.id)).where(
                models.Document.workspace_id == self._workspace_id,
                models.Document.deleted_at.is_(None),
                models.Document.source.not_in(connectors),
            )
        )
        return not foreign_live

    async def _live_chunk_inventory_digest(self, session: AsyncSession) -> str:
        """Recompute #187's exact active-corpus vector inventory after relational mutation."""
        digest = SnapshotChunkDigester()
        after: tuple[str, int, str] | None = None
        while True:
            statement = (
                select(models.Chunk, models.Document.publication_id)
                .join(models.Document, models.Document.id == models.Chunk.document_id)
                .where(
                    models.Document.workspace_id == self._workspace_id,
                    models.Document.deleted_at.is_(None),
                )
            )
            if after is not None:
                document, position, chunk = after
                statement = statement.where(
                    (models.Chunk.document_id > document)
                    | (
                        (models.Chunk.document_id == document)
                        & (
                            (models.Chunk.position > position)
                            | ((models.Chunk.position == position) & (models.Chunk.id > chunk))
                        )
                    )
                )
            rows = (
                await session.execute(
                    statement.order_by(
                        models.Chunk.document_id,
                        models.Chunk.position,
                        models.Chunk.id,
                    ).limit(_INVENTORY_PAGE)
                )
            ).all()
            if not rows:
                return digest.hexdigest()
            for row, publication_id in rows:
                digest.add(
                    SnapshotChunk(
                        chunk=to_chunk(row),
                        vector_id=row.vector_id,
                        publication_id=publication_id,
                        sequence=row.seq,
                        created_at=row.created_at,
                    )
                )
            last = rows[-1][0]
            after = (last.document_id, last.position, last.id)

    async def _publish_item(
        self,
        session: AsyncSession,
        *,
        replacement: DerivedReplacement,
        generation_id: str,
        vector_publication: str,
        connector_name: str,
        target: RebuildTarget,
        snapshot: models.AcquisitionRecord,
    ) -> str:
        document = replacement.document
        expected_id = document_id(self._workspace_id, connector_name, document.source_id)
        if document.id != expected_id or document.source != connector_name:
            raise RebuildPublicationValidationError
        self._require_snapshot_match(replacement, snapshot)
        return await self._publish_member(
            session,
            replacement=replacement,
            vector_publication=vector_publication,
            connector_name=connector_name,
            target=target,
        )

    async def _publish_member(
        self,
        session: AsyncSession,
        *,
        replacement: DerivedReplacement,
        vector_publication: str,
        connector_name: str,
        target: RebuildTarget,
    ) -> str:
        document = replacement.document
        expected_id = document_id(self._workspace_id, connector_name, document.source_id)
        if document.id != expected_id or document.source != connector_name:
            raise RebuildPublicationValidationError
        for member in replacement.members:
            await self._publish_member(
                session,
                replacement=member,
                vector_publication=vector_publication,
                connector_name=connector_name,
                target=target,
            )
        stored = await session.get(models.Document, document.id)
        if stored is None:
            stored = models.Document(id=document.id, workspace_id=self._workspace_id)
            session.add(stored)
        elif (
            stored.workspace_id != self._workspace_id or stored.publication_id == vector_publication
        ):
            raise RebuildPublicationValidationError
        # Populate every non-null document field before the following deletes trigger ORM
        # autoflush. A replacement published into an empty corpus creates this row; leaving it
        # as only id/workspace until after a query makes SQLite correctly refuse the transient
        # invalid shape before ``apply_document`` ever runs.
        apply_document(stored, document)
        stored.publication_id = vector_publication
        stored.parse_fp = replacement.parse_fingerprint
        stored.chunk_fp = target.chunk_fingerprint
        stored.embed_fp = target.embedding_fingerprint
        stored.glossary_fp = target.glossary_fingerprint
        await session.execute(
            delete(models.GlossaryEntry).where(models.GlossaryEntry.document_id == document.id)
        )
        await session.execute(delete(models.Chunk).where(models.Chunk.document_id == document.id))
        await session.flush()
        for chunk in replacement.chunks:
            session.add(from_chunk(chunk, document.id, vector_publication))
        await session.flush()
        aliases: list[tuple[str, str]] = []
        for entry in replacement.glossary:
            entry_id = glossary_entry_id(entry.chunk_id, entry.acronym, entry.expansion)
            session.add(
                models.GlossaryEntry(
                    id=entry_id,
                    document_id=document.id,
                    chunk_id=entry.chunk_id,
                    acronym=entry.acronym,
                    display=entry.display,
                    expansion=entry.expansion,
                    location=entry.location,
                    form=entry.form.value,
                    confidence=entry.confidence,
                )
            )
            aliases.extend((entry_id, alias) for alias in dict.fromkeys(entry.aliases))
        # SQLAlchemy cannot infer an insert dependency between these independently-created ORM
        # rows because the models deliberately expose no relationship collection. Flush every
        # glossary parent before making any alias pending. This is also the evidence-page
        # boundary: the next page query autoflushes, so merely delaying the aliases until the
        # end of this method would still leave SQLite free to see a child before its parent.
        await session.flush()
        for entry_id, alias in aliases:
            session.add(models.GlossaryAlias(entry_id=entry_id, key=alias))
        return document.id

    @staticmethod
    def _digest_part(hasher: _DigestHasher, value: object) -> None:
        encoded = _canonical(value)
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)

    async def _evidence_descriptors(
        self,
        session: AsyncSession,
        generation: models.DerivedGeneration,
    ) -> AsyncIterator[_EvidenceDescriptor]:
        """Stream the ordered evidence inventory without populating the ORM identity map."""
        rows = await session.stream(
            select(
                models.AcquisitionRecord.source_id,
                models.AcquisitionRecord.blob_ref,
                models.AcquisitionRecord.acquired_source,
                models.Blob.algo,
                models.Blob.size_bytes,
                models.Blob.stored_bytes,
                models.Blob.compression,
            )
            .join(models.Blob, models.Blob.hash == models.AcquisitionRecord.blob_ref)
            .join(
                models.DerivedGenerationSnapshot,
                models.DerivedGenerationSnapshot.run_id == models.AcquisitionRecord.run_id,
            )
            .where(
                models.DerivedGenerationSnapshot.generation_id == generation.id,
                models.AcquisitionRecord.workspace_id == self._workspace_id,
                models.AcquisitionRecord.blob_ref.is_not(None),
                models.AcquisitionRecord.acquired_source.is_not(None),
            )
            .order_by(
                models.DerivedGenerationSnapshot.ordinal,
                models.AcquisitionRecord.sequence,
            )
            .execution_options(yield_per=_INVENTORY_PAGE)
        )
        sequence = 0
        async for (
            source_id,
            blob_ref,
            acquired_source,
            algo,
            size,
            stored,
            compression,
        ) in rows:
            try:
                acquired = AcquiredSource.model_validate(acquired_source)
            except ValueError as exc:
                raise RebuildPublicationValidationError from exc
            if (
                not isinstance(blob_ref, str)
                or blob_ref != acquired.content_hash
                or algo != "blake2b"
                or compression not in {"none", "gzip"}
            ):
                raise RebuildPublicationConflictError(RebuildRefusalCode.SNAPSHOT_CHANGED)
            yield _EvidenceDescriptor(
                sequence=sequence,
                source_id=source_id,
                blob_ref=blob_ref,
                acquired_content_hash=acquired.content_hash,
                algo=algo,
                size_bytes=size,
                stored_bytes=stored,
                compression=compression,
            )
            sequence += 1

    async def _evidence_representations(
        self,
        session: AsyncSession,
        generation: models.DerivedGeneration,
    ) -> AsyncIterator[_EvidenceRepresentation]:
        """Stream one descriptor per unique representation in deterministic order."""
        rows = await session.stream(
            select(
                models.AcquisitionRecord.blob_ref,
                models.Blob.algo,
                models.Blob.size_bytes,
                models.Blob.stored_bytes,
                models.Blob.compression,
            )
            .join(models.Blob, models.Blob.hash == models.AcquisitionRecord.blob_ref)
            .join(
                models.DerivedGenerationSnapshot,
                models.DerivedGenerationSnapshot.run_id == models.AcquisitionRecord.run_id,
            )
            .where(
                models.DerivedGenerationSnapshot.generation_id == generation.id,
                models.AcquisitionRecord.workspace_id == self._workspace_id,
                models.AcquisitionRecord.blob_ref.is_not(None),
                models.AcquisitionRecord.acquired_source.is_not(None),
            )
            .distinct()
            .order_by(models.AcquisitionRecord.blob_ref)
            .execution_options(yield_per=_INVENTORY_PAGE)
        )
        async for blob_ref, algo, size, stored, compression in rows:
            if (
                not isinstance(blob_ref, str)
                or algo != "blake2b"
                or compression not in {"none", "gzip"}
            ):
                raise RebuildPublicationConflictError(RebuildRefusalCode.SNAPSHOT_CHANGED)
            yield _EvidenceRepresentation(
                blob_ref=blob_ref,
                algo=algo,
                size_bytes=size,
                stored_bytes=stored,
                compression=compression,
            )

    async def _evidence_inventory_digest(
        self,
        session: AsyncSession,
        generation: models.DerivedGeneration,
    ) -> str:
        """Incrementally bind exact ordered membership with constant resident state."""
        hasher = hashlib.sha256()
        snapshots = await self._generation_snapshot_rows(session, generation.id)
        self._digest_part(
            hasher,
            {
                "version": 2,
                "workspace_id": self._workspace_id,
                "generation_id": generation.id,
                "snapshot_run_ids": [row.run_id for row in snapshots],
                "snapshot_membership_hash": generation.snapshot_membership_hash,
                "expected_item_count": generation.expected_item_count,
            },
        )
        count = 0
        async for item in self._evidence_descriptors(session, generation):
            self._digest_part(
                hasher,
                {
                    "sequence": item.sequence,
                    "source_id": item.source_id,
                    "blob_ref": item.blob_ref,
                    "acquired_content_hash": item.acquired_content_hash,
                    "algo": item.algo,
                    "size_bytes": item.size_bytes,
                    "stored_bytes": item.stored_bytes,
                    "compression": item.compression,
                },
            )
            count += 1
        self._digest_part(hasher, {"evidence_count": count})
        if count != generation.expected_item_count:
            raise RebuildPublicationConflictError(RebuildRefusalCode.SNAPSHOT_CHANGED)
        return hasher.hexdigest()

    async def _evidence_verification_digest(
        self,
        session: AsyncSession,
        generation: models.DerivedGeneration,
        inventory_digest: str,
        *,
        verify_content: bool,
        pins: EvidencePinInventory | None = None,
        final_probe: bool = False,
    ) -> str:
        """Incrementally bind one identity per unique representation."""
        hasher = hashlib.sha256()
        self._digest_part(
            hasher,
            {"version": 2, "inventory_digest": inventory_digest},
        )
        count = 0
        async for item in self._evidence_representations(session, generation):
            if final_probe:
                if pins is None:
                    raise AssertionError("a final evidence probe requires inode pins")
                identity = await pins.validate(
                    item.blob_ref,
                    size_bytes=item.size_bytes,
                    stored_bytes=item.stored_bytes,
                    compression=item.compression,
                )
            elif pins is not None and verify_content:
                identity = await pins.verify(
                    item.blob_ref,
                    size_bytes=item.size_bytes,
                    stored_bytes=item.stored_bytes,
                    compression=item.compression,
                )
            elif pins is not None:
                identity = await pins.pin(
                    item.blob_ref,
                    size_bytes=item.size_bytes,
                    stored_bytes=item.stored_bytes,
                    compression=item.compression,
                )
            else:
                identity = await self._blobs.evidence_identity(
                    item.blob_ref,
                    size_bytes=item.size_bytes,
                    stored_bytes=item.stored_bytes,
                    compression=item.compression,
                    verify_content=verify_content,
                )
            if identity is None:
                raise RebuildPublicationConflictError(RebuildRefusalCode.SNAPSHOT_CHANGED)
            self._digest_part(hasher, {"blob_ref": item.blob_ref, "identity": identity})
            count += 1
        self._digest_part(hasher, {"representation_count": count})
        return hasher.hexdigest()

    async def _record_evidence_verification(self, generation_id: str) -> None:
        """Hash retained bytes without a SQLite writer, then persist one exact fence."""
        async with self._blobs.evidence_fence() as pins:
            async with self._sessions() as session:
                generation = await self._required_generation(session, generation_id)
                if generation.state not in {RebuildState.VALIDATING, RebuildState.PUBLISHED}:
                    raise RebuildPublicationValidationError
                lease_generation = generation.lease_generation
                inventory_digest = await self._evidence_inventory_digest(session, generation)
                verification_digest = await self._evidence_verification_digest(
                    session,
                    generation,
                    inventory_digest,
                    verify_content=True,
                    pins=pins,
                )
            async with self._sessions.begin() as session:
                current = await self._required_generation(session, generation_id)
                if (
                    current.state not in {RebuildState.VALIDATING, RebuildState.PUBLISHED}
                    or current.lease_generation != lease_generation
                ):
                    raise RebuildLeaseConflictError("generation lease changed during verification")
                await self._verify_complete_header(session, current)
                current_inventory = await self._evidence_inventory_digest(session, current)
                if current_inventory != inventory_digest:
                    raise RebuildPublicationConflictError(RebuildRefusalCode.SNAPSHOT_CHANGED)
                current.evidence_inventory_digest = inventory_digest
                current.evidence_verification_digest = verification_digest
                current.evidence_verification_lease_generation = lease_generation
                current.evidence_verified_at = utcnow()
                current.updated_at = utcnow()

    @contextlib.asynccontextmanager
    async def _publication_evidence_fence(
        self, generation_id: str
    ) -> AsyncGenerator[_EvidenceFence]:
        async with self._sessions() as session:
            generation = await self._required_generation(session, generation_id)
            observed_inventory = await self._evidence_inventory_digest(session, generation)
            if (
                generation.evidence_inventory_digest is None
                or generation.evidence_verification_digest is None
                or generation.evidence_inventory_digest != observed_inventory
            ):
                needs_legacy_verification = True
            else:
                needs_legacy_verification = False
        if needs_legacy_verification:
            await self._record_evidence_verification(generation_id)
        async with self._sessions() as session:
            generation = await self._required_generation(session, generation_id)
            inventory_digest = await self._evidence_inventory_digest(session, generation)
            persisted_inventory = generation.evidence_inventory_digest
            persisted_verification = generation.evidence_verification_digest
            persisted_lease = generation.evidence_verification_lease_generation
            lease_generation = generation.lease_generation
        if (
            persisted_inventory is None
            or persisted_verification is None
            or persisted_lease != lease_generation
            or persisted_inventory != inventory_digest
        ):
            raise RebuildPublicationConflictError(RebuildRefusalCode.SNAPSHOT_CHANGED)
        async with self._blobs.evidence_fence() as pins:
            async with self._sessions() as session:
                generation = await self._required_generation(session, generation_id)
                inventory_digest = await self._evidence_inventory_digest(session, generation)
                if inventory_digest != persisted_inventory:
                    raise RebuildPublicationConflictError(RebuildRefusalCode.SNAPSHOT_CHANGED)
                observed = await self._evidence_verification_digest(
                    session,
                    generation,
                    inventory_digest,
                    verify_content=False,
                    pins=pins,
                )
            if observed != persisted_verification:
                raise RebuildPublicationConflictError(RebuildRefusalCode.SNAPSHOT_CHANGED)
            # Full hashing and pin promotion above hold only one digest shard at a time. Once
            # the represented shard set is known, take that fixed (at most 256-lock) set only
            # across cheap canonical-inode probes and the atomic DB publication. Cooperating
            # BlobStore writers/GC can never invalidate an early probe before commit, while
            # unrelated shards remain live during the expensive corpus scan.
            async with pins.publication_fence():
                async with self._sessions() as session:
                    generation = await self._required_generation(session, generation_id)
                    locked_observed = await self._evidence_verification_digest(
                        session,
                        generation,
                        inventory_digest,
                        verify_content=False,
                        pins=pins,
                        final_probe=True,
                    )
                if locked_observed != persisted_verification:
                    raise RebuildPublicationConflictError(RebuildRefusalCode.SNAPSHOT_CHANGED)
                yield _EvidenceFence(
                    inventory_digest=inventory_digest,
                    verification_digest=locked_observed,
                    lease_generation=lease_generation,
                    pins=pins,
                )

    async def _verify_complete_header(
        self,
        session: AsyncSession,
        generation: models.DerivedGeneration,
    ) -> None:
        snapshots = await self._generation_snapshot_rows(session, generation.id)
        if not snapshots:
            raise RebuildPublicationConflictError(RebuildRefusalCode.SNAPSHOT_CHANGED)
        combined = hashlib.sha256(
            _canonical(
                [[row.run_id, row.membership_hash, row.expected_item_count] for row in snapshots]
            )
        ).hexdigest()
        legacy_single = (
            len(snapshots) == 1
            and snapshots[0].membership_hash == generation.snapshot_membership_hash
        )
        if combined != generation.snapshot_membership_hash and not legacy_single:
            raise RebuildPublicationConflictError(RebuildRefusalCode.SNAPSHOT_CHANGED)
        snapshot_count = 0
        evidence_count = 0
        for snapshot in snapshots:
            run = await session.get(models.AcquisitionRun, snapshot.run_id)
            if (
                run is None
                or run.workspace_id != self._workspace_id
                or run.promoted_at is None
                or run.membership_hash != snapshot.membership_hash
                or run.connector_name != snapshot.connector_name
                or run.scope_fingerprint != snapshot.scope_fingerprint
                or not await snapshot_manifest_matches(session, run.id, snapshot.membership_hash)
            ):
                raise RebuildPublicationConflictError(RebuildRefusalCode.SNAPSHOT_CHANGED)
            counts = (
                await session.execute(
                    select(
                        func.count(models.AcquisitionRecord.sequence),
                        func.count(models.AcquisitionRecord.sequence).filter(
                            models.AcquisitionRecord.blob_ref.is_not(None),
                            models.AcquisitionRecord.acquired_source.is_not(None),
                        ),
                        func.count(models.AcquisitionRecord.sequence).filter(
                            models.AcquisitionRecord.snapshot_outcome
                            == SnapshotItemOutcome.OMITTED,
                            models.AcquisitionRecord.blob_ref.is_(None),
                            models.AcquisitionRecord.acquired_source.is_(None),
                        ),
                    ).where(
                        models.AcquisitionRecord.run_id == snapshot.run_id,
                        models.AcquisitionRecord.workspace_id == self._workspace_id,
                    )
                )
            ).one()
            total, retained, omitted = (int(value) for value in counts)
            completeness = (
                None if run.completeness is None else SnapshotCompleteness(run.completeness)
            )
            partial_is_honest = (
                completeness is SnapshotCompleteness.PARTIAL
                and SnapshotPromotionPolicy(run.promotion_policy)
                is SnapshotPromotionPolicy.ALLOW_OMISSIONS
                and omitted == run.omission_count
                and total == retained + omitted
            )
            complete_is_honest = (
                completeness is SnapshotCompleteness.COMPLETE
                and run.omission_count == 0
                and omitted == 0
                and total == retained
            )
            if (
                total != run.discovered_count
                or retained != snapshot.expected_item_count
                or not (partial_is_honest or complete_is_honest)
            ):
                raise RebuildPublicationValidationError
            snapshot_count += total
            evidence_count += retained
        if (
            generation.next_sequence != generation.expected_item_count
            or generation.documents_built != generation.expected_item_count
        ):
            raise RebuildPublicationValidationError
        item_count = (
            await session.execute(
                select(func.count(models.DerivedGenerationItem.sequence)).where(
                    models.DerivedGenerationItem.generation_id == generation.id
                )
            )
        ).scalar_one()
        distinct_item_sources = (
            await session.execute(
                select(
                    func.count(
                        func.distinct(
                            func.json_extract(
                                models.DerivedGenerationItem.payload, "$.document.source"
                            )
                            + "\0"
                            + func.json_extract(
                                models.DerivedGenerationItem.payload, "$.document.source_id"
                            )
                        )
                    )
                ).where(models.DerivedGenerationItem.generation_id == generation.id)
            )
        ).scalar_one()
        if (
            item_count != generation.expected_item_count
            or distinct_item_sources != item_count
            or evidence_count != item_count
            or snapshot_count < evidence_count
        ):
            raise RebuildPublicationValidationError

    async def _evidence_page(
        self,
        session: AsyncSession,
        generation: models.DerivedGeneration,
        *,
        after: int,
    ) -> list[tuple[models.DerivedGenerationItem, models.AcquisitionRecord]]:
        items = list(
            (
                await session.execute(
                    select(models.DerivedGenerationItem)
                    .where(
                        models.DerivedGenerationItem.generation_id == generation.id,
                        models.DerivedGenerationItem.sequence > after,
                    )
                    .order_by(models.DerivedGenerationItem.sequence)
                    .limit(_EVIDENCE_PAGE)
                )
            ).scalars()
        )
        snapshots: list[models.AcquisitionRecord] = []
        for item in items:
            try:
                document = _REPLACEMENT.validate_python(item.payload).document
            except ValueError as exc:
                raise RebuildPublicationValidationError from exc
            snapshot = (
                await session.execute(
                    select(models.AcquisitionRecord)
                    .join(
                        models.DerivedGenerationSnapshot,
                        models.DerivedGenerationSnapshot.run_id == models.AcquisitionRecord.run_id,
                    )
                    .where(
                        models.DerivedGenerationSnapshot.generation_id == generation.id,
                        models.DerivedGenerationSnapshot.connector_name == document.source,
                        models.AcquisitionRecord.workspace_id == self._workspace_id,
                        models.AcquisitionRecord.source_id == document.source_id,
                        models.AcquisitionRecord.blob_ref.is_not(None),
                        models.AcquisitionRecord.acquired_source.is_not(None),
                    )
                )
            ).scalar_one_or_none()
            if snapshot is None:
                raise RebuildPublicationValidationError
            snapshots.append(snapshot)
        expected = list(range(after + 1, after + 1 + len(items)))
        if (
            not items
            or [item.sequence for item in items] != expected
            or len(snapshots) != len(items)
        ):
            raise RebuildPublicationValidationError
        return list(zip(items, snapshots, strict=True))

    def _validated_replacement(
        self,
        row: models.DerivedGenerationItem,
        snapshot: models.AcquisitionRecord,
    ) -> DerivedReplacement:
        digest = hashlib.sha256(_canonical(row.payload)).hexdigest()
        if digest != row.payload_digest:
            raise RebuildPublicationValidationError
        try:
            replacement = _REPLACEMENT.validate_python(row.payload)
            replacement.validate_identity()
            self._require_snapshot_match(replacement, snapshot)
        except RebuildPublicationValidationError:
            raise
        except (RuntimeError, ValueError) as exc:
            raise RebuildPublicationValidationError from exc
        return replacement

    @staticmethod
    def _require_snapshot_match(
        replacement: DerivedReplacement, snapshot: models.AcquisitionRecord
    ) -> None:
        document = replacement.document
        acquired = (
            None
            if snapshot.acquired_source is None
            else AcquiredSource.model_validate(snapshot.acquired_source)
        )
        source = AcquisitionSource.model_validate(snapshot.source_record)
        expected_version = snapshot.fetched_version_token or source.version_token
        if (
            acquired is None
            or snapshot.blob_ref is None
            or document.source_id != snapshot.source_id
            or document.source_id != acquired.source_id
            or document.original_ref != snapshot.blob_ref
            or document.content_hash != acquired.content_hash
            or document.uri != acquired.uri
            or document.media_type != acquired.media_type
            or document.version_token != expected_version
        ):
            raise RebuildPublicationValidationError

    @staticmethod
    def _temporary_cost(
        replacement: DerivedReplacement, target: RebuildTarget, payload_bytes: int
    ) -> int:
        try:
            identity = json.loads(target.embedding_fingerprint)
            dimension = identity["dimension"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("target embedding fingerprint has no dimension") from exc
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("target embedding fingerprint has an invalid dimension")
        chunks = replacement.flattened_chunks()
        vector_bytes = len(chunks) * dimension * 4
        text_bytes = sum(
            len(chunk.text.encode()) + len(chunk.embed_text.encode()) for chunk in chunks
        )
        # SQLite payload + WAL, Lance vector + version, and FTS token/index amplification.
        return payload_bytes * 2 + vector_bytes * 3 + text_bytes * 2

    @staticmethod
    def _require_lease(
        generation: models.DerivedGeneration,
        owner: str,
        lease_generation: int,
        now: datetime,
    ) -> None:
        if (
            generation.lease_owner != owner
            or generation.lease_generation != lease_generation
            or generation.lease_expires_at is None
            or generation.lease_expires_at <= now
        ):
            raise RebuildLeaseConflictError("generation lease changed or expired")

    async def cancel_generation(
        self,
        generation_id: str,
        *,
        owner: str,
        lease_generation: int,
        now: datetime,
    ) -> RebuildCheckpoint:
        async with self._sessions.begin() as session:
            generation = await self._required_generation(session, generation_id)
            if generation.state is RebuildState.PUBLISHED:
                return _checkpoint(generation)
            self._require_lease(generation, owner, lease_generation, now)
            generation.state = RebuildState.CANCELED
            generation.lease_owner = None
            generation.lease_expires_at = None
            generation.updated_at = now
            return _checkpoint(generation)

    async def fail_generation(
        self,
        generation_id: str,
        code: RebuildRefusalCode,
        *,
        owner: str,
        lease_generation: int,
        now: datetime,
    ) -> RebuildCheckpoint:
        async with self._sessions.begin() as session:
            generation = await self._required_generation(session, generation_id)
            self._require_lease(generation, owner, lease_generation, now)
            generation.state = RebuildState.FAILED
            generation.diagnostic_code = code.value
            generation.diagnostic_count = 1
            generation.lease_owner = None
            generation.lease_expires_at = None
            generation.updated_at = now
            return _checkpoint(generation)

    async def release_generation(
        self,
        generation_id: str,
        code: RebuildRefusalCode,
        *,
        owner: str,
        lease_generation: int,
        now: datetime,
    ) -> RebuildCheckpoint:
        """Record why this attempt stopped and drop the lease, leaving the state as it was.

        Every other settlement here is forward-only into a terminal state. This one deliberately
        is not: the generation stays `building` or `validating`, which is what makes the next
        `claim_generation` a takeover rather than a refusal, and what keeps the committed prefix
        worth something after a failure that had nothing to do with it.

        `diagnostic_count` accumulates instead of being set to one. A storage failure that
        happens once is contention; the same one on the fourth attempt is a machine that needs
        an operator, and the count is how that becomes visible without a log.
        """
        async with self._sessions.begin() as session:
            generation = await self._required_generation(session, generation_id)
            self._require_lease(generation, owner, lease_generation, now)
            generation.diagnostic_code = code.value
            generation.diagnostic_count += 1
            generation.lease_owner = None
            generation.lease_expires_at = None
            generation.updated_at = now
            return _checkpoint(generation)

    async def _required_generation(
        self, session: AsyncSession, generation_id: str
    ) -> models.DerivedGeneration:
        row = await session.get(models.DerivedGeneration, generation_id)
        if row is None or row.workspace_id != self._workspace_id:
            raise KeyError(generation_id)
        return row


__all__ = [
    "BlobInventory",
    "GenerationVectorInventory",
    "RebuildLeaseConflictError",
    "RebuildTerminalGenerationError",
    "SqliteRebuildStore",
]
