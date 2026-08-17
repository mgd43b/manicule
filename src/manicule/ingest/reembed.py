"""Resumable shadow-generation re-embedding from an immutable local corpus snapshot.

This module is the storage-independent orchestration contract. The built-in SQLite journal,
durable corpus snapshots, fenced publisher, and named Lance generations live in
:mod:`manicule.storage.reembed`. The application runtime wires those adapters to the local
operator workflow; this module remains independent of every delivery surface.

The protocols expose no connector, parser, or blob fallback. A rebuild reads only stored
documents and their exact ``Chunk.embed_text`` inputs, writes an unpublished generation, and
presents one atomic compare-and-swap request to a publisher. Retrieval's live generation is
never a write target.

Crash recovery rests on an immutable corpus snapshot, a create-if-absent shadow generation,
and an owner token plus monotonically fenced lease generation on every mutation. Publication
returns a durable receipt: retry after a publish-before-journal crash returns that receipt and
cannot roll a newer generation back.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from manicule.core.content import Chunk, Document
from manicule.core.embedding import EmbedFingerprint, Vector
from manicule.core.errors import ManiculeError, PolicyError
from manicule.core.fingerprints import ChunkFingerprint
from manicule.core.protocols import Embedder
from manicule.ingest.embedding import batch_size, embed_chunks

DEFAULT_DOCUMENT_PAGE = 32
DEFAULT_TARGET_BATCH_TOKENS = 16_384
DEFAULT_CHUNKS_PER_SECOND = 20.0
DEFAULT_LEASE_TTL_SECONDS = 30.0
_FLOAT32_BYTES = 4
_ROW_OVERHEAD_BYTES = 512


class ReembedState(StrEnum):
    PLANNED = "planned"
    BUILDING = "building"
    VALIDATING = "validating"
    READY = "ready"
    PUBLISHED = "published"
    SUPERSEDED = "superseded"
    FAILED = "failed"


class PublishOutcome(StrEnum):
    PUBLISHED = "published"
    SUPERSEDED = "superseded"


@dataclass(frozen=True, slots=True)
class LivePublication:
    """The exact live winner a plan is allowed to replace."""

    generation_id: str
    fingerprint: str
    inventory_digest: str | None


@dataclass(frozen=True, slots=True)
class CorpusSnapshot:
    """An immutable, keyset-readable view plus its publication CAS revision."""

    id: str
    revision: str
    live: LivePublication
    workspace_id: str = field(default="", repr=False)


@dataclass(frozen=True, order=True, slots=True)
class ChunkKey:
    position: int
    id: str


@dataclass(frozen=True, slots=True)
class SnapshotDocument:
    """A domain document plus storage-only fields forming its publication identity."""

    workspace_id: str
    document: Document
    original_omitted_reason: str | None = None
    chunk_fingerprint: str | None = None
    embed_fingerprint: str | None = None
    glossary_fingerprint: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_seen_at: datetime | None = None
    deleted_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class SnapshotChunk:
    """A chunk plus physical row identity that the domain value omits."""

    chunk: Chunk
    vector_id: str
    publication_id: str
    sequence: int
    created_at: datetime | None = None


class SnapshotChunkDigester:
    """Incremental physical-inventory digest for a stable keyset-ordered row stream."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self._previous_chunk: tuple[str, int, str] | None = None
        self._previous: tuple[str, int, str, str, str, int] | None = None

    def add(self, stored: SnapshotChunk) -> None:
        chunk_key = (
            stored.chunk.document_id,
            stored.chunk.position,
            stored.chunk.id,
        )
        key = (
            *chunk_key,
            stored.vector_id,
            stored.publication_id,
            stored.sequence,
        )
        if (self._previous_chunk is not None and chunk_key <= self._previous_chunk) or (
            self._previous is not None and key <= self._previous
        ):
            raise ValueError("snapshot chunk rows must be strictly keyset ordered")
        self._previous_chunk = chunk_key
        self._previous = key
        _hash_bytes(self._digest, _canonical_chunk(stored))

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


class SnapshotInventoryDigester:
    """Canonical full-document inventory in the same order used by planning."""

    def __init__(self, revision: str) -> None:
        self._digest = hashlib.sha256()
        self._previous_document_id: str | None = None
        self._active_document_id: str | None = None
        _hash_value(self._digest, revision)

    def add_document(self, stored: SnapshotDocument) -> None:
        document_id = stored.document.id
        if self._previous_document_id is not None and document_id <= self._previous_document_id:
            raise ValueError("snapshot documents must be strictly keyset ordered")
        self._previous_document_id = document_id
        self._active_document_id = document_id
        _hash_bytes(self._digest, _canonical_document(stored))

    def add_chunk(self, stored: SnapshotChunk) -> None:
        if stored.chunk.document_id != self._active_document_id:
            raise ValueError("snapshot chunks must immediately follow their owning document")
        _hash_bytes(self._digest, _canonical_chunk(stored))

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ReembedLease:
    owner_token: str
    generation: int
    expires_at: float


@dataclass(frozen=True, slots=True)
class ReembedPlan:
    """The only dry-run DTO surfaces may serialize: aggregate facts and nothing else."""

    documents: int
    chunks: int
    input_bytes: int
    estimated_seconds: float
    peak_memory_bytes: int
    temporary_disk_bytes: int
    unrepairable_documents: int = 0

    @property
    def runnable(self) -> bool:
        return self.unrepairable_documents == 0

    def public_facts(self) -> Mapping[str, int | float | bool]:
        """Counts and estimates only: never stable document ids or source URLs."""
        return {
            "documents": self.documents,
            "chunks": self.chunks,
            "input_bytes": self.input_bytes,
            "estimated_seconds": self.estimated_seconds,
            "peak_memory_bytes": self.peak_memory_bytes,
            "temporary_disk_bytes": self.temporary_disk_bytes,
            "unrepairable_documents": self.unrepairable_documents,
            "runnable": self.runnable,
        }


@dataclass(frozen=True, slots=True)
class ReembedCommitment:
    """Private resumability inputs. Never serialize this object onto an operator surface."""

    plan: ReembedPlan
    snapshot: CorpusSnapshot = field(repr=False)
    target_fingerprint: str = field(repr=False)
    target_config: str = field(repr=False)
    target_dimension: int = field(repr=False)
    inventory_digest: str = field(repr=False)
    chunk_inventory_digest: str = field(repr=False)
    build_plan: ReembedPlan | None = field(default=None, repr=False)

    @property
    def execution_plan(self) -> ReembedPlan:
        """Private full-installation work; legacy commitments used their public plan."""
        return self.plan if self.build_plan is None else self.build_plan


@dataclass(frozen=True, slots=True)
class PublicationReceipt:
    """Durable outcome of exactly one publication decision."""

    id: str
    run_id: str
    outcome: PublishOutcome
    expected: LivePublication
    observed_winner: LivePublication
    published_generation_id: str | None
    workspace_id: str = field(default="", repr=False)


@dataclass(frozen=True, slots=True)
class ReembedRun:
    id: str
    commitment: ReembedCommitment = field(repr=False)
    workspace_id: str = field(default="", repr=False)
    state: ReembedState = ReembedState.PLANNED
    document_after: str | None = None
    active_document_id: str | None = None
    chunk_after: ChunkKey | None = None
    documents_completed: int = 0
    chunks_completed: int = 0
    workspace_documents_completed: int = 0
    workspace_chunks_completed: int = 0
    shadow_generation_id: str | None = None
    receipt: PublicationReceipt | None = None
    revision: int = 0
    failure: str = ""


@dataclass(frozen=True, slots=True)
class ShadowGeneration:
    id: str
    run_id: str
    fingerprint: str
    inventory_digest: str
    workspace_id: str = field(default="", repr=False)


@dataclass(frozen=True, slots=True)
class ShadowInspection:
    """Measurements recomputed from stored rows, never echoed from preparation arguments.

    ``inventory_digest`` covers every persisted :class:`SnapshotChunk` field. ``lineage_valid``
    additionally verifies backend-specific physical keys point at the target generation and
    logical chunk they claim.
    """

    rows: int
    unique_chunks: int
    dimension: int
    finite: bool
    fingerprint: str
    inventory_digest: str
    lineage_valid: bool
    retrieval_ready: bool
    storage_revision: str = ""


class ReembedError(ManiculeError):
    """A rebuild cannot safely advance."""


class ReembedValidationError(ReembedError):
    """A complete shadow generation failed a publication prerequisite."""


class ReembedCapacityError(ReembedError):
    """Local capacity cannot safely hold the planned shadow generation."""


@dataclass(frozen=True, slots=True)
class ReembedRecovery:
    """Aggregate-safe restart recovery outcome; never carries run ids or error messages."""

    recovered: int = 0
    failures: int = 0
    failure_types: tuple[str, ...] = ()


class ReembedCorpus(Protocol):
    """Stable, bounded local reads. Implementations must never contact a source."""

    async def begin_snapshot(self) -> CorpusSnapshot: ...

    async def discard_snapshot(self, snapshot_id: str) -> None:
        """Delete an unbound snapshot and every private row it owns."""
        ...

    async def documents(
        self, snapshot: CorpusSnapshot, *, after: str | None, limit: int
    ) -> Sequence[SnapshotDocument]:
        """Documents ordered by id, strictly greater than ``after``."""
        ...

    async def document(
        self, snapshot: CorpusSnapshot, document_id: str
    ) -> SnapshotDocument | None: ...

    async def chunks(
        self,
        snapshot: CorpusSnapshot,
        document_id: str,
        *,
        after: ChunkKey | None,
        limit: int,
    ) -> Sequence[SnapshotChunk]:
        """Chunks ordered by ``(position, id)``, strictly greater than ``after``."""
        ...


class ReembedJournal(Protocol):
    """Durable run storage and the authority for fenced leases."""

    async def create(
        self,
        run_id: str,
        commitment: ReembedCommitment,
        *,
        owner_token: str,
        ttl_seconds: float,
    ) -> tuple[ReembedRun, ReembedLease]:
        """Atomically create the run and mint its first fenced lease."""
        ...

    async def create_released(self, run_id: str, commitment: ReembedCommitment) -> ReembedRun:
        """Atomically create an immediately acquirable run with no live lease owner."""
        ...

    async def get(self, run_id: str) -> ReembedRun | None: ...

    async def acquire(
        self, run_id: str, owner_token: str, *, ttl_seconds: float
    ) -> ReembedLease: ...

    async def renew(
        self, run_id: str, lease: ReembedLease, *, ttl_seconds: float
    ) -> ReembedLease: ...

    async def release(self, run_id: str, lease: ReembedLease) -> None:
        """Atomically expire exactly the current fence; stale owners cannot release a winner."""
        ...

    async def save(
        self, run: ReembedRun, *, expected_revision: int, lease: ReembedLease
    ) -> ReembedRun: ...


class ShadowVectorGeneration(Protocol):
    """Non-destructive storage for rows invisible to live retrieval."""

    async def open_or_create(
        self,
        run_id: str,
        *,
        fingerprint: EmbedFingerprint,
        inventory_digest: str,
        lease: ReembedLease,
    ) -> ShadowGeneration:
        """Create if absent; otherwise verify identity and preserve every row."""
        ...

    async def upsert(
        self,
        generation: ShadowGeneration,
        chunks: Sequence[SnapshotChunk],
        vectors: Sequence[Vector],
        *,
        lease: ReembedLease,
    ) -> None:
        """Idempotently write rows only if ``lease`` is current at the mutation boundary.

        Checking before embedding, waiting, or acquiring a backend lock is insufficient: the
        lease can expire and a higher fence can take over during any of them. Concrete storage
        must compare owner, generation, and expiry atomically with the physical row mutation.
        """
        ...

    async def inspect(
        self, generation: ShadowGeneration, *, lease: ReembedLease
    ) -> ShadowInspection:
        """Read and recompute validation facts from physical rows under the lease fence."""
        ...

    async def seal(
        self,
        generation: ShadowGeneration,
        inspection: ShadowInspection,
        *,
        lease: ReembedLease,
    ) -> None:
        """Atomically make the exact inspected generation immutable and publishable."""
        ...


class ReembedPublisher(Protocol):
    """Atomic publication CAS with a durable receipt and lease fence."""

    async def publish(
        self,
        run: ReembedRun,
        generation: ShadowGeneration,
        *,
        expected: LivePublication,
        expected_corpus_revision: str,
        lease: ReembedLease,
    ) -> PublicationReceipt:
        """Return the same receipt forever for ``run.id``.

        With no receipt, compare live generation, fingerprint, inventory, and corpus revision
        in the same transaction as the pointer/fingerprint swap. A mismatch durably records
        ``SUPERSEDED`` without writing. A prior receipt is returned without consulting or
        mutating the current winner.
        """
        ...


async def plan_reembed(
    corpus: ReembedCorpus,
    target: EmbedFingerprint,
    *,
    document_page: int = DEFAULT_DOCUMENT_PAGE,
    target_batch_tokens: int = DEFAULT_TARGET_BATCH_TOKENS,
    chunks_per_second: float = DEFAULT_CHUNKS_PER_SECOND,
) -> ReembedPlan:
    """Price a rebuild without connectors, parsing, embedding, or live-publication mutation.

    A concrete corpus may persist an immutable local snapshot so a later rebuild can resume
    from exactly what was priced.
    """
    commitment = await plan_reembed_commitment(
        corpus,
        target,
        document_page=document_page,
        target_batch_tokens=target_batch_tokens,
        chunks_per_second=chunks_per_second,
    )
    return commitment.plan


async def plan_reembed_commitment(
    corpus: ReembedCorpus,
    target: EmbedFingerprint,
    *,
    document_page: int = DEFAULT_DOCUMENT_PAGE,
    target_batch_tokens: int = DEFAULT_TARGET_BATCH_TOKENS,
    chunks_per_second: float = DEFAULT_CHUNKS_PER_SECOND,
) -> ReembedCommitment:
    """Build the private durable commitment behind a public aggregate plan."""
    _validate_knobs(document_page, target_batch_tokens, chunks_per_second)
    snapshot = await corpus.begin_snapshot()
    try:
        return await _plan_snapshot(
            corpus,
            snapshot,
            target,
            document_page=document_page,
            target_batch_tokens=target_batch_tokens,
            chunks_per_second=chunks_per_second,
        )
    except BaseException as error:
        await _discard_after_failure(corpus, snapshot.id, error)
        raise


async def start_reembed(
    run_id: str,
    *,
    owner_token: str,
    corpus: ReembedCorpus,
    target: EmbedFingerprint,
    journal: ReembedJournal,
    document_page: int = DEFAULT_DOCUMENT_PAGE,
    target_batch_tokens: int = DEFAULT_TARGET_BATCH_TOKENS,
    chunks_per_second: float = DEFAULT_CHUNKS_PER_SECOND,
    lease_ttl_seconds: float = DEFAULT_LEASE_TTL_SECONDS,
    commitment: ReembedCommitment | None = None,
) -> ReembedRun:
    _validate_owner(owner_token, lease_ttl_seconds)
    created_snapshot = commitment is None
    if commitment is None:
        commitment = await plan_reembed_commitment(
            corpus,
            target,
            document_page=document_page,
            target_batch_tokens=target_batch_tokens,
            chunks_per_second=chunks_per_second,
        )
    elif commitment.target_fingerprint != target.canonical():
        raise ReembedError("the prepared commitment belongs to another target embedder")
    if not commitment.execution_plan.runnable:
        error = PolicyError(
            "one or more stored documents have no chunks. "
            "Re-embedding never falls back to a connector or parser; repair local inputs first."
        )
        if created_snapshot:
            await _discard_after_failure(corpus, commitment.snapshot.id, error)
        raise error
    try:
        return await journal.create_released(run_id, commitment)
    except BaseException as error:
        if created_snapshot:
            await _discard_after_failure(corpus, commitment.snapshot.id, error)
        raise


async def resume_reembed(
    run_id: str,
    *,
    owner_token: str,
    corpus: ReembedCorpus,
    embedder: Embedder,
    journal: ReembedJournal,
    shadow: ShadowVectorGeneration,
    publisher: ReembedPublisher,
    document_page: int = DEFAULT_DOCUMENT_PAGE,
    target_batch_tokens: int = DEFAULT_TARGET_BATCH_TOKENS,
    lease_ttl_seconds: float = DEFAULT_LEASE_TTL_SECONDS,
) -> ReembedRun:
    """Resume one run under an explicit fenced lease."""
    _validate_owner(owner_token, lease_ttl_seconds)
    _validate_knobs(document_page, target_batch_tokens, DEFAULT_CHUNKS_PER_SECOND)
    run = await journal.get(run_id)
    if run is None:
        raise ReembedError(f"no re-embedding run {run_id!r} exists")
    if EmbedFingerprint.model_validate_json(run.commitment.target_config) != embedder.fingerprint:
        raise ReembedError(
            "the configured embedder is not the exact target this run planned; model, weights, "
            "dimension, normalization, tokenizer, context limit, and backend must match"
        )
    if run.state in {ReembedState.PUBLISHED, ReembedState.SUPERSEDED}:
        return run
    if run.state is ReembedState.FAILED:
        raise ReembedError(f"re-embedding run {run_id!r} failed validation: {run.failure}")
    lease = await journal.acquire(run.id, owner_token, ttl_seconds=lease_ttl_seconds)
    failure: BaseException | None = None
    try:
        return await _resume_owned(
            run,
            lease=lease,
            corpus=corpus,
            embedder=embedder,
            journal=journal,
            shadow=shadow,
            publisher=publisher,
            document_page=document_page,
            target_batch_tokens=target_batch_tokens,
            lease_ttl_seconds=lease_ttl_seconds,
        )
    except BaseException as error:
        failure = error
        raise
    finally:
        await _release_after_operation(journal, run.id, lease, failure)


async def _resume_owned(
    run: ReembedRun,
    *,
    lease: ReembedLease,
    corpus: ReembedCorpus,
    embedder: Embedder,
    journal: ReembedJournal,
    shadow: ShadowVectorGeneration,
    publisher: ReembedPublisher,
    document_page: int,
    target_batch_tokens: int,
    lease_ttl_seconds: float,
) -> ReembedRun:
    generation = await shadow.open_or_create(
        run.id,
        fingerprint=embedder.fingerprint,
        inventory_digest=run.commitment.chunk_inventory_digest,
        lease=lease,
    )
    if (
        generation.run_id != run.id
        or generation.fingerprint != run.commitment.target_fingerprint
        or generation.inventory_digest != run.commitment.chunk_inventory_digest
    ):
        raise ReembedError("shadow generation identity does not match the durable run")
    if run.shadow_generation_id not in {None, generation.id}:
        raise ReembedError("the durable run names a different immutable shadow generation")
    if run.shadow_generation_id is None or run.state is ReembedState.PLANNED:
        run, lease = await _save(
            journal,
            replace(run, shadow_generation_id=generation.id, state=ReembedState.BUILDING),
            lease,
            lease_ttl_seconds,
        )
    if run.state is ReembedState.BUILDING:
        run, lease = await _build(
            run,
            generation=generation,
            corpus=corpus,
            embedder=embedder,
            journal=journal,
            shadow=shadow,
            document_page=document_page,
            target_batch_tokens=target_batch_tokens,
            lease=lease,
            lease_ttl_seconds=lease_ttl_seconds,
        )
    if run.state is ReembedState.VALIDATING:
        try:
            inspection = await _validate(
                run,
                generation=generation,
                corpus=corpus,
                shadow=shadow,
                document_page=document_page,
                target_batch_tokens=target_batch_tokens,
                lease=lease,
            )
            await shadow.seal(generation, inspection, lease=lease)
        except ReembedValidationError as exc:
            await _save(
                journal,
                replace(run, state=ReembedState.FAILED, failure=str(exc)),
                lease,
                lease_ttl_seconds,
            )
            raise
        run, lease = await _save(
            journal,
            replace(run, state=ReembedState.READY),
            lease,
            lease_ttl_seconds,
        )
    if run.state is ReembedState.READY:
        lease = await journal.renew(run.id, lease, ttl_seconds=lease_ttl_seconds)
        receipt = await publisher.publish(
            run,
            generation,
            expected=run.commitment.snapshot.live,
            expected_corpus_revision=run.commitment.snapshot.revision,
            lease=lease,
        )
        reconciled = await journal.get(run.id)
        if (
            reconciled is None
            or reconciled.receipt != receipt
            or reconciled.state
            not in {
                ReembedState.PUBLISHED,
                ReembedState.SUPERSEDED,
            }
        ):
            raise ReembedError("publication receipt was not reconciled into durable run state")
        run = reconciled
    return run


async def _build(
    run: ReembedRun,
    *,
    generation: ShadowGeneration,
    corpus: ReembedCorpus,
    embedder: Embedder,
    journal: ReembedJournal,
    shadow: ShadowVectorGeneration,
    document_page: int,
    target_batch_tokens: int,
    lease: ReembedLease,
    lease_ttl_seconds: float,
) -> tuple[ReembedRun, ReembedLease]:
    chunk_limit = _chunk_limit(embedder.fingerprint, target_batch_tokens)
    while True:
        if run.active_document_id is None:
            documents = await corpus.documents(
                run.commitment.snapshot, after=run.document_after, limit=document_page
            )
            if not documents:
                return await _save(
                    journal,
                    replace(run, state=ReembedState.VALIDATING),
                    lease,
                    lease_ttl_seconds,
                )
            run, lease = await _save(
                journal,
                replace(run, active_document_id=documents[0].document.id, chunk_after=None),
                lease,
                lease_ttl_seconds,
            )
        active_document_id = run.active_document_id
        if active_document_id is None:  # pragma: no cover - established immediately above
            raise RuntimeError("building without an active document")
        document = await corpus.document(run.commitment.snapshot, active_document_id)
        if document is None:
            raise ReembedError("the immutable snapshot lost a document during resume")
        chunk_fingerprint = None
        if document.chunk_fingerprint is not None:
            try:
                chunk_fingerprint = ChunkFingerprint.model_validate_json(document.chunk_fingerprint)
            except ValueError as exc:
                raise ReembedError(
                    "the immutable snapshot has an invalid chunk fingerprint"
                ) from exc
        stored_chunks = await corpus.chunks(
            run.commitment.snapshot,
            document.document.id,
            after=run.chunk_after,
            limit=chunk_limit,
        )
        if stored_chunks:
            chunks = [stored.chunk for stored in stored_chunks]
            chunk_count = len(chunks)
            last = chunks[-1]
            vectors = await embed_chunks(
                embedder,
                chunks,
                chunk_fingerprint=chunk_fingerprint,
                target_batch_tokens=target_batch_tokens,
            )
            lease = await journal.renew(run.id, lease, ttl_seconds=lease_ttl_seconds)
            await shadow.upsert(generation, stored_chunks, vectors, lease=lease)
            # Do not retain one page's inputs/outputs while the next page is materialized.
            del chunks, stored_chunks, vectors
            run, lease = await _save(
                journal,
                replace(
                    run,
                    chunk_after=ChunkKey(last.position, last.id),
                    chunks_completed=run.chunks_completed + chunk_count,
                    workspace_chunks_completed=(
                        run.workspace_chunks_completed
                        + (chunk_count if document.workspace_id == run.workspace_id else 0)
                    ),
                ),
                lease,
                lease_ttl_seconds,
            )
            continue
        if run.chunk_after is None and document.document.expects_chunks:
            raise ReembedError("a document expected stored chunks but the snapshot has none")
        run, lease = await _save(
            journal,
            replace(
                run,
                document_after=document.document.id,
                active_document_id=None,
                chunk_after=None,
                documents_completed=run.documents_completed + 1,
                workspace_documents_completed=(
                    run.workspace_documents_completed
                    + (1 if document.workspace_id == run.workspace_id else 0)
                ),
            ),
            lease,
            lease_ttl_seconds,
        )


async def _validate(
    run: ReembedRun,
    *,
    generation: ShadowGeneration,
    corpus: ReembedCorpus,
    shadow: ShadowVectorGeneration,
    document_page: int,
    target_batch_tokens: int,
    lease: ReembedLease,
) -> ShadowInspection:
    target = EmbedFingerprint.model_validate_json(run.commitment.target_config)
    current = await _plan_snapshot(
        corpus,
        run.commitment.snapshot,
        target,
        document_page=document_page,
        target_batch_tokens=target_batch_tokens,
        chunks_per_second=DEFAULT_CHUNKS_PER_SECOND,
    )
    inspection = await shadow.inspect(generation, lease=lease)
    failures: list[str] = []
    if current.inventory_digest != run.commitment.inventory_digest:
        failures.append("the immutable inventory does not match its planned digest")
    expected_chunks = run.commitment.execution_plan.chunks
    if inspection.rows != expected_chunks:
        failures.append(f"expected {expected_chunks} rows, found {inspection.rows}")
    if inspection.unique_chunks != expected_chunks:
        failures.append(
            f"expected {expected_chunks} unique chunks, found {inspection.unique_chunks}"
        )
    if inspection.dimension != run.commitment.target_dimension:
        failures.append(
            f"expected dimension {run.commitment.target_dimension}, found {inspection.dimension}"
        )
    if not inspection.finite:
        failures.append("one or more vectors contain non-finite values")
    if inspection.fingerprint != run.commitment.target_fingerprint:
        failures.append("the shadow carries a different embedding fingerprint")
    if inspection.inventory_digest != run.commitment.chunk_inventory_digest:
        failures.append("the shadow rows do not cover the planned physical chunk inventory")
    if not inspection.lineage_valid:
        failures.append("one or more rows have invalid chunk or embedding lineage")
    if not inspection.retrieval_ready:
        failures.append("the shadow failed its retrieval-readiness probe")
    if failures:
        raise ReembedValidationError("; ".join(failures))
    return inspection


async def _plan_snapshot(
    corpus: ReembedCorpus,
    snapshot: CorpusSnapshot,
    target: EmbedFingerprint,
    *,
    document_page: int,
    target_batch_tokens: int,
    chunks_per_second: float,
) -> ReembedCommitment:
    digest = SnapshotInventoryDigester(snapshot.revision)
    chunk_digest = SnapshotChunkDigester()
    documents_count = chunks_count = input_bytes = unrepairable = 0
    public_documents = public_chunks = public_input_bytes = public_unrepairable = 0
    peak_memory = 0
    public_peak_memory = 0
    after: str | None = None
    chunk_limit = _chunk_limit(target, target_batch_tokens)
    while True:
        documents = await corpus.documents(snapshot, after=after, limit=document_page)
        if not documents:
            break
        document_page_bytes = sum(len(_canonical_document(document)) for document in documents)
        peak_memory = max(peak_memory, document_page_bytes)
        for document in documents:
            digest.add_document(document)
            domain_document = document.document
            owned = not snapshot.workspace_id or document.workspace_id == snapshot.workspace_id
            public_document_bytes = len(_canonical_document(document)) if owned else 0
            public_peak_memory = max(public_peak_memory, public_document_bytes)
            documents_count += 1
            public_documents += int(owned)
            chunk_after: ChunkKey | None = None
            found = False
            while True:
                chunks = await corpus.chunks(
                    snapshot,
                    domain_document.id,
                    after=chunk_after,
                    limit=chunk_limit,
                )
                if not chunks:
                    break
                found = True
                for stored_chunk in chunks:
                    digest.add_chunk(stored_chunk)
                    chunk_digest.add(stored_chunk)
                    input_bytes += len(stored_chunk.chunk.embed_text.encode("utf-8"))
                    if owned:
                        public_input_bytes += len(stored_chunk.chunk.embed_text.encode("utf-8"))
                chunks_count += len(chunks)
                if owned:
                    public_chunks += len(chunks)
                chunk_page_bytes = sum(len(_canonical_chunk(chunk)) for chunk in chunks)
                peak_memory = max(
                    peak_memory,
                    document_page_bytes
                    + chunk_page_bytes
                    + len(chunks) * target.dimension * _FLOAT32_BYTES,
                )
                if owned:
                    public_peak_memory = max(
                        public_peak_memory,
                        public_document_bytes
                        + chunk_page_bytes
                        + len(chunks) * target.dimension * _FLOAT32_BYTES,
                    )
                last = chunks[-1].chunk
                chunk_after = ChunkKey(last.position, last.id)
            if not found and domain_document.expects_chunks:
                unrepairable += 1
                public_unrepairable += int(owned)
        after = documents[-1].document.id
    vector_bytes = target.dimension * _FLOAT32_BYTES
    build_plan = ReembedPlan(
        documents=documents_count,
        chunks=chunks_count,
        input_bytes=input_bytes,
        estimated_seconds=chunks_count / chunks_per_second,
        peak_memory_bytes=peak_memory,
        temporary_disk_bytes=chunks_count * (vector_bytes + _ROW_OVERHEAD_BYTES),
        unrepairable_documents=unrepairable,
    )
    plan = ReembedPlan(
        documents=public_documents,
        chunks=public_chunks,
        input_bytes=public_input_bytes,
        estimated_seconds=public_chunks / chunks_per_second,
        peak_memory_bytes=public_peak_memory,
        temporary_disk_bytes=public_chunks * (vector_bytes + _ROW_OVERHEAD_BYTES),
        unrepairable_documents=public_unrepairable,
    )
    return ReembedCommitment(
        plan=plan,
        snapshot=snapshot,
        target_fingerprint=target.canonical(),
        target_config=target.model_dump_json(),
        target_dimension=target.dimension,
        inventory_digest=digest.hexdigest(),
        chunk_inventory_digest=chunk_digest.hexdigest(),
        build_plan=build_plan,
    )


async def _save(
    journal: ReembedJournal,
    proposed: ReembedRun,
    lease: ReembedLease,
    ttl_seconds: float,
) -> tuple[ReembedRun, ReembedLease]:
    lease = await journal.renew(proposed.id, lease, ttl_seconds=ttl_seconds)
    saved = await journal.save(proposed, expected_revision=proposed.revision, lease=lease)
    return saved, lease


async def _discard_after_failure(
    corpus: ReembedCorpus, snapshot_id: str, failure: BaseException
) -> None:
    cleanup = asyncio.create_task(corpus.discard_snapshot(snapshot_id))
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            # Preserve the caller's cancellation, but finish removing private durable rows
            # before it escapes. A second cancellation cannot strand the cleanup task.
            continue
        except Exception:  # noqa: BLE001 - reported on the original failure below
            break
    try:
        cleanup.result()
    except Exception as cleanup_error:  # noqa: BLE001 - annotate and preserve original
        failure.add_note(
            f"discarding the unbound re-embedding snapshot also failed: {cleanup_error}"
        )


async def discard_reembed_snapshot(
    corpus: ReembedCorpus, snapshot_id: str, failure: BaseException | None = None
) -> None:
    """Cancellation-safe cleanup for a commitment that never became a durable run."""
    if failure is not None:
        await _discard_after_failure(corpus, snapshot_id, failure)
        return
    cleanup = asyncio.create_task(corpus.discard_snapshot(snapshot_id))
    canceled: asyncio.CancelledError | None = None
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError as error:
            canceled = error
            continue
    cleanup.result()
    if canceled is not None:
        raise canceled


async def _release_after_operation(
    journal: ReembedJournal,
    run_id: str,
    lease: ReembedLease,
    failure: BaseException | None,
) -> None:
    release = asyncio.create_task(journal.release(run_id, lease))
    while not release.done():
        try:
            await asyncio.shield(release)
        except asyncio.CancelledError:
            continue
        except Exception:  # noqa: BLE001 - released below without masking the primary error
            break
    try:
        release.result()
    except BaseException as release_error:
        if failure is None:
            raise
        failure.add_note(f"releasing the re-embedding lease also failed: {release_error}")


def _validate_knobs(document_page: int, target_batch_tokens: int, rate: float) -> None:
    if document_page < 1 or target_batch_tokens < 1:
        raise ValueError("document_page and target_batch_tokens must be at least 1")
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("chunks_per_second must be a positive finite number")


def _validate_owner(owner_token: str, ttl_seconds: float) -> None:
    if not owner_token:
        raise ValueError("owner_token must not be empty")
    if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
        raise ValueError("lease_ttl_seconds must be a positive finite number")


def _chunk_limit(target: EmbedFingerprint, target_batch_tokens: int) -> int:
    return batch_size(
        budget_tokens=target.max_sequence_length,
        target_batch_tokens=target_batch_tokens,
    )


def _canonical_document(stored: SnapshotDocument) -> bytes:
    document = stored.document
    value = document.model_dump(mode="json", exclude_none=False)
    value["publication_id"] = document.publication_id
    value["workspace_id"] = stored.workspace_id
    value["original_omitted_reason"] = stored.original_omitted_reason
    value["chunk_fingerprint"] = stored.chunk_fingerprint
    value["embed_fingerprint"] = stored.embed_fingerprint
    value["glossary_fingerprint"] = stored.glossary_fingerprint
    value["created_at"] = _datetime(stored.created_at)
    value["updated_at"] = _datetime(stored.updated_at)
    value["last_seen_at"] = _datetime(stored.last_seen_at)
    value["deleted_at"] = _datetime(stored.deleted_at)
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _canonical_chunk(stored: SnapshotChunk) -> bytes:
    value = stored.chunk.model_dump(mode="json", exclude_none=False)
    value["vector_id"] = stored.vector_id
    value["publication_id"] = stored.publication_id
    value["sequence"] = stored.sequence
    value["created_at"] = _datetime(stored.created_at)
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _datetime(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


class _Digest(Protocol):
    def update(self, data: bytes, /) -> None: ...


def _hash_value(digest: _Digest, value: str) -> None:
    _hash_bytes(digest, value.encode("utf-8"))


def _hash_bytes(digest: _Digest, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


__all__ = [
    "DEFAULT_CHUNKS_PER_SECOND",
    "DEFAULT_DOCUMENT_PAGE",
    "DEFAULT_LEASE_TTL_SECONDS",
    "DEFAULT_TARGET_BATCH_TOKENS",
    "ChunkKey",
    "CorpusSnapshot",
    "LivePublication",
    "PublicationReceipt",
    "PublishOutcome",
    "ReembedCapacityError",
    "ReembedCommitment",
    "ReembedCorpus",
    "ReembedError",
    "ReembedJournal",
    "ReembedLease",
    "ReembedPlan",
    "ReembedPublisher",
    "ReembedRecovery",
    "ReembedRun",
    "ReembedState",
    "ReembedValidationError",
    "ShadowGeneration",
    "ShadowInspection",
    "ShadowVectorGeneration",
    "SnapshotChunk",
    "SnapshotChunkDigester",
    "SnapshotDocument",
    "SnapshotInventoryDigester",
    "discard_reembed_snapshot",
    "plan_reembed",
    "plan_reembed_commitment",
    "resume_reembed",
    "start_reembed",
]
