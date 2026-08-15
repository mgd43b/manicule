"""Resumable shadow-generation re-embedding from an immutable local corpus snapshot.

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

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from manicule.core.content import Chunk, Document
from manicule.core.embedding import EmbedFingerprint, Vector
from manicule.core.errors import ManiculeError, PolicyError
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
    inventory_digest: str


@dataclass(frozen=True, slots=True)
class CorpusSnapshot:
    """An immutable, keyset-readable view plus its publication CAS revision."""

    id: str
    revision: str
    live: LivePublication


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
    sequence: int
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ReembedLease:
    owner_token: str
    generation: int
    expires_at: float


@dataclass(frozen=True, slots=True)
class ReembedPlan:
    """Aggregate-only dry-run output and private execution commitments."""

    snapshot: CorpusSnapshot
    target_fingerprint: str
    target_config: str
    target_dimension: int
    documents: int
    chunks: int
    input_bytes: int
    estimated_seconds: float
    peak_memory_bytes: int
    temporary_disk_bytes: int
    inventory_digest: str
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
class PublicationReceipt:
    """Durable outcome of exactly one publication decision."""

    id: str
    run_id: str
    outcome: PublishOutcome
    expected: LivePublication
    observed_winner: LivePublication
    published_generation_id: str | None


@dataclass(frozen=True, slots=True)
class ReembedRun:
    id: str
    plan: ReembedPlan
    state: ReembedState = ReembedState.PLANNED
    document_after: str | None = None
    active_document_id: str | None = None
    chunk_after: ChunkKey | None = None
    documents_completed: int = 0
    chunks_completed: int = 0
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


@dataclass(frozen=True, slots=True)
class ShadowInspection:
    rows: int
    unique_chunks: int
    dimension: int
    finite: bool
    fingerprint: str
    inventory_digest: str
    lineage_valid: bool
    retrieval_ready: bool


class ReembedError(ManiculeError):
    """A rebuild cannot safely advance."""


class ReembedValidationError(ReembedError):
    """A complete shadow generation failed a publication prerequisite."""


class ReembedCorpus(Protocol):
    """Stable, bounded local reads. Implementations must never contact a source."""

    async def begin_snapshot(self) -> CorpusSnapshot: ...

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
        plan: ReembedPlan,
        *,
        owner_token: str,
        ttl_seconds: float,
    ) -> tuple[ReembedRun, ReembedLease]:
        """Atomically create the run and mint its first fenced lease."""
        ...

    async def get(self, run_id: str) -> ReembedRun | None: ...

    async def acquire(
        self, run_id: str, owner_token: str, *, ttl_seconds: float
    ) -> ReembedLease: ...

    async def renew(
        self, run_id: str, lease: ReembedLease, *, ttl_seconds: float
    ) -> ReembedLease: ...

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
        chunks: Sequence[Chunk],
        vectors: Sequence[Vector],
        *,
        lease: ReembedLease,
    ) -> None: ...

    async def inspect(
        self, generation: ShadowGeneration, *, lease: ReembedLease
    ) -> ShadowInspection: ...


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
    """Price a rebuild from an immutable local snapshot without writing or embedding."""
    _validate_knobs(document_page, target_batch_tokens, chunks_per_second)
    snapshot = await corpus.begin_snapshot()
    return await _plan_snapshot(
        corpus,
        snapshot,
        target,
        document_page=document_page,
        target_batch_tokens=target_batch_tokens,
        chunks_per_second=chunks_per_second,
    )


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
) -> ReembedRun:
    _validate_owner(owner_token, lease_ttl_seconds)
    plan = await plan_reembed(
        corpus,
        target,
        document_page=document_page,
        target_batch_tokens=target_batch_tokens,
        chunks_per_second=chunks_per_second,
    )
    if not plan.runnable:
        raise PolicyError(
            f"{plan.unrepairable_documents} document(s) have no stored chunks. Re-embedding "
            "never falls back to a connector or parser; repair local inputs first."
        )
    run, _ = await journal.create(
        run_id,
        plan,
        owner_token=owner_token,
        ttl_seconds=lease_ttl_seconds,
    )
    return run


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
    if EmbedFingerprint.model_validate_json(run.plan.target_config) != embedder.fingerprint:
        raise ReembedError(
            "the configured embedder is not the exact target this run planned; model, weights, "
            "dimension, normalization, tokenizer, context limit, and backend must match"
        )
    if run.state in {ReembedState.PUBLISHED, ReembedState.SUPERSEDED}:
        return run
    if run.state is ReembedState.FAILED:
        raise ReembedError(f"re-embedding run {run_id!r} failed validation: {run.failure}")
    lease = await journal.acquire(run.id, owner_token, ttl_seconds=lease_ttl_seconds)
    generation = await shadow.open_or_create(
        run.id,
        fingerprint=embedder.fingerprint,
        inventory_digest=run.plan.inventory_digest,
        lease=lease,
    )
    if (
        generation.run_id != run.id
        or generation.fingerprint != run.plan.target_fingerprint
        or generation.inventory_digest != run.plan.inventory_digest
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
            await _validate(
                run,
                generation=generation,
                corpus=corpus,
                shadow=shadow,
                document_page=document_page,
                target_batch_tokens=target_batch_tokens,
                lease=lease,
            )
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
            expected=run.plan.snapshot.live,
            expected_corpus_revision=run.plan.snapshot.revision,
            lease=lease,
        )
        terminal = (
            ReembedState.PUBLISHED
            if receipt.outcome is PublishOutcome.PUBLISHED
            else ReembedState.SUPERSEDED
        )
        run, _ = await _save(
            journal,
            replace(run, state=terminal, receipt=receipt),
            lease,
            lease_ttl_seconds,
        )
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
                run.plan.snapshot, after=run.document_after, limit=document_page
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
        document = await corpus.document(run.plan.snapshot, active_document_id)
        if document is None:
            raise ReembedError("the immutable snapshot lost a document during resume")
        stored_chunks = await corpus.chunks(
            run.plan.snapshot,
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
                target_batch_tokens=target_batch_tokens,
            )
            lease = await journal.renew(run.id, lease, ttl_seconds=lease_ttl_seconds)
            await shadow.upsert(generation, chunks, vectors, lease=lease)
            # Do not retain one page's inputs/outputs while the next page is materialized.
            del chunks, vectors
            run, lease = await _save(
                journal,
                replace(
                    run,
                    chunk_after=ChunkKey(last.position, last.id),
                    chunks_completed=run.chunks_completed + chunk_count,
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
) -> None:
    target = EmbedFingerprint.model_validate_json(run.plan.target_config)
    current = await _plan_snapshot(
        corpus,
        run.plan.snapshot,
        target,
        document_page=document_page,
        target_batch_tokens=target_batch_tokens,
        chunks_per_second=DEFAULT_CHUNKS_PER_SECOND,
    )
    inspection = await shadow.inspect(generation, lease=lease)
    failures: list[str] = []
    if current.inventory_digest != run.plan.inventory_digest:
        failures.append("the immutable inventory does not match its planned digest")
    if inspection.rows != run.plan.chunks:
        failures.append(f"expected {run.plan.chunks} rows, found {inspection.rows}")
    if inspection.unique_chunks != run.plan.chunks:
        failures.append(
            f"expected {run.plan.chunks} unique chunks, found {inspection.unique_chunks}"
        )
    if inspection.dimension != run.plan.target_dimension:
        failures.append(
            f"expected dimension {run.plan.target_dimension}, found {inspection.dimension}"
        )
    if not inspection.finite:
        failures.append("one or more vectors contain non-finite values")
    if inspection.fingerprint != run.plan.target_fingerprint:
        failures.append("the shadow carries a different embedding fingerprint")
    if inspection.inventory_digest != run.plan.inventory_digest:
        failures.append("the shadow does not cover the planned inventory")
    if not inspection.lineage_valid:
        failures.append("one or more rows have invalid chunk or embedding lineage")
    if not inspection.retrieval_ready:
        failures.append("the shadow failed its retrieval-readiness probe")
    if failures:
        raise ReembedValidationError("; ".join(failures))


async def _plan_snapshot(
    corpus: ReembedCorpus,
    snapshot: CorpusSnapshot,
    target: EmbedFingerprint,
    *,
    document_page: int,
    target_batch_tokens: int,
    chunks_per_second: float,
) -> ReembedPlan:
    digest = hashlib.sha256()
    _hash_value(digest, snapshot.revision)
    documents_count = chunks_count = input_bytes = unrepairable = 0
    peak_memory = 0
    after: str | None = None
    chunk_limit = _chunk_limit(target, target_batch_tokens)
    while True:
        documents = await corpus.documents(snapshot, after=after, limit=document_page)
        if not documents:
            break
        document_page_bytes = sum(len(_canonical_document(document)) for document in documents)
        peak_memory = max(peak_memory, document_page_bytes)
        for document in documents:
            _hash_bytes(digest, _canonical_document(document))
            domain_document = document.document
            documents_count += 1
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
                    _hash_bytes(digest, _canonical_chunk(stored_chunk))
                    input_bytes += len(stored_chunk.chunk.embed_text.encode("utf-8"))
                chunks_count += len(chunks)
                chunk_page_bytes = sum(len(_canonical_chunk(chunk)) for chunk in chunks)
                peak_memory = max(
                    peak_memory,
                    document_page_bytes
                    + chunk_page_bytes
                    + len(chunks) * target.dimension * _FLOAT32_BYTES,
                )
                last = chunks[-1].chunk
                chunk_after = ChunkKey(last.position, last.id)
            if not found and domain_document.expects_chunks:
                unrepairable += 1
        after = documents[-1].document.id
    vector_bytes = target.dimension * _FLOAT32_BYTES
    return ReembedPlan(
        snapshot=snapshot,
        target_fingerprint=target.canonical(),
        target_config=target.model_dump_json(),
        target_dimension=target.dimension,
        documents=documents_count,
        chunks=chunks_count,
        input_bytes=input_bytes,
        estimated_seconds=chunks_count / chunks_per_second,
        peak_memory_bytes=peak_memory,
        temporary_disk_bytes=chunks_count * (vector_bytes + _ROW_OVERHEAD_BYTES),
        inventory_digest=digest.hexdigest(),
        unrepairable_documents=unrepairable,
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
    "ReembedCorpus",
    "ReembedError",
    "ReembedJournal",
    "ReembedLease",
    "ReembedPlan",
    "ReembedPublisher",
    "ReembedRun",
    "ReembedState",
    "ReembedValidationError",
    "ShadowGeneration",
    "ShadowInspection",
    "ShadowVectorGeneration",
    "SnapshotChunk",
    "SnapshotDocument",
    "plan_reembed",
    "resume_reembed",
    "start_reembed",
]
