"""Plan and resume a corpus-wide embedding rebuild from stored chunk inputs.

This module deliberately knows nothing about connectors, parsers, source blobs, or a concrete
vector database.  A rebuild reads the exact ``Chunk.embed_text`` retained by ingest, writes a
named shadow generation, and asks one publisher to make that validated generation live.  The
small protocols are the safety boundary: an adapter cannot accidentally turn an offline repair
into a crawl, and the orchestration cannot mutate the vector generation retrieval is serving.

The durable journal is external to the process.  Every completed document is checkpointed with
a compare-and-swap revision; if the process dies after a shadow upsert but before its checkpoint,
the same publication-keyed rows are written again on resume.  A correct shadow store therefore
implements upsert, never append.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol

from manicule.core.content import Chunk, Document
from manicule.core.embedding import EmbedFingerprint
from manicule.core.errors import ManiculeError, PolicyError
from manicule.core.protocols import Embedder
from manicule.ingest.embedding import batch_size, embed_chunks

DEFAULT_DOCUMENT_BATCH = 100
DEFAULT_TARGET_BATCH_TOKENS = 16_384
DEFAULT_CHUNKS_PER_SECOND = 20.0
_FLOAT32_BYTES = 4
_ROW_OVERHEAD_BYTES = 512


class ReembedState(StrEnum):
    """Durable lifecycle of one shadow embedding generation."""

    PLANNED = "planned"
    BUILDING = "building"
    VALIDATING = "validating"
    READY = "ready"
    PUBLISHED = "published"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ReembedPlan:
    """A read-only, body-free estimate and inventory commitment for one rebuild."""

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
    unrepairable: tuple[str, ...] = ()

    @property
    def runnable(self) -> bool:
        """Whether every document that promises chunks has local inputs."""
        return not self.unrepairable


@dataclass(frozen=True, slots=True)
class ReembedRun:
    """The checkpoint persisted after every document-sized transaction boundary."""

    id: str
    plan: ReembedPlan
    state: ReembedState = ReembedState.PLANNED
    next_document: int = 0
    documents_completed: int = 0
    chunks_completed: int = 0
    revision: int = 0
    failure: str = ""


@dataclass(frozen=True, slots=True)
class ShadowInspection:
    """Facts a shadow backend measured without trusting the orchestration's counters."""

    rows: int
    unique_chunks: int
    dimension: int
    finite: bool
    fingerprint: str
    inventory_digest: str
    lineage_valid: bool
    retrieval_ready: bool


class ReembedError(ManiculeError):
    """Base class for a rebuild that cannot safely advance."""


class ReembedValidationError(ReembedError):
    """A complete shadow generation failed a publication prerequisite."""


class ReembedCorpus(Protocol):
    """The bounded local reads a re-embed is allowed to perform."""

    async def select_documents(
        self, *, limit: int | None = None, offset: int = 0
    ) -> Sequence[Document]: ...

    async def document_chunks(self, document_id: str) -> Sequence[Chunk]: ...


class ReembedJournal(Protocol):
    """Durable run storage with optimistic concurrency."""

    async def create(self, run_id: str, plan: ReembedPlan) -> ReembedRun: ...

    async def get(self, run_id: str) -> ReembedRun | None: ...

    async def save(self, run: ReembedRun, *, expected_revision: int) -> ReembedRun: ...


class ShadowVectorGeneration(Protocol):
    """A vector generation that is not visible to live reads until publication."""

    async def prepare(self, run_id: str, fingerprint: EmbedFingerprint) -> None: ...

    async def upsert(
        self,
        run_id: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> None: ...

    async def inspect(self, run_id: str) -> ShadowInspection: ...


class ReembedPublisher(Protocol):
    """The single atomic visibility boundary for a validated generation."""

    async def publish(self, run: ReembedRun) -> None:
        """Atomically publish ``run``'s table pointer and embedding fingerprint.

        The operation must be idempotent and must compare the planned inventory at the same
        atomic boundary as it changes the table pointer and fingerprint.  Validation and this
        call are separate awaits; a connector sync can otherwise land between them and make a
        perfectly validated shadow stale before it becomes visible.  A process can also die
        after publication and before the journal records :attr:`ReembedState.PUBLISHED`;
        retrying must observe the same winner.
        """
        ...


class _Digest(Protocol):
    def update(self, data: bytes, /) -> None: ...


async def plan_reembed(
    corpus: ReembedCorpus,
    target: EmbedFingerprint,
    *,
    document_batch: int = DEFAULT_DOCUMENT_BATCH,
    target_batch_tokens: int = DEFAULT_TARGET_BATCH_TOKENS,
    chunks_per_second: float = DEFAULT_CHUNKS_PER_SECOND,
) -> ReembedPlan:
    """Price a rebuild and commit to its current local inventory without writing anything.

    The inventory is streamed a page at a time and reduced into a digest.  No corpus-sized list
    is retained.  Document bodies are never included in the result or a diagnostic; the digest
    sees only revision/lineage identities and a hash of each embedding input.
    """
    if document_batch < 1:
        raise ValueError("document_batch must be at least 1")
    if target_batch_tokens < 1:
        raise ValueError("target_batch_tokens must be at least 1")
    if not math.isfinite(chunks_per_second) or chunks_per_second <= 0:
        raise ValueError("chunks_per_second must be a positive finite number")

    digest = hashlib.sha256()
    digest.update(target.canonical().encode("utf-8"))
    documents = chunks = input_bytes = 0
    peak_memory_bytes = 0
    unrepairable: list[str] = []
    offset = 0
    while True:
        page = await corpus.select_documents(limit=document_batch, offset=offset)
        if not page:
            break
        for document in page:
            stored = await corpus.document_chunks(document.id)
            _digest_document(digest, document, stored)
            documents += 1
            chunks += len(stored)
            input_bytes += sum(len(chunk.embed_text.encode("utf-8")) for chunk in stored)
            peak_memory_bytes = max(
                peak_memory_bytes,
                _peak_batch_memory(stored, target, target_batch_tokens),
            )
            if not stored and document.expects_chunks:
                unrepairable.append(f"{document.id} ({document.uri}): no stored chunks to re-embed")
        offset += len(page)

    vector_bytes = target.dimension * _FLOAT32_BYTES
    return ReembedPlan(
        target_fingerprint=target.canonical(),
        target_config=target.model_dump_json(),
        target_dimension=target.dimension,
        documents=documents,
        chunks=chunks,
        input_bytes=input_bytes,
        estimated_seconds=chunks / chunks_per_second,
        peak_memory_bytes=peak_memory_bytes,
        temporary_disk_bytes=chunks * (vector_bytes + _ROW_OVERHEAD_BYTES),
        inventory_digest=digest.hexdigest(),
        unrepairable=tuple(unrepairable),
    )


async def start_reembed(
    run_id: str,
    *,
    corpus: ReembedCorpus,
    target: EmbedFingerprint,
    journal: ReembedJournal,
    document_batch: int = DEFAULT_DOCUMENT_BATCH,
    target_batch_tokens: int = DEFAULT_TARGET_BATCH_TOKENS,
    chunks_per_second: float = DEFAULT_CHUNKS_PER_SECOND,
) -> ReembedRun:
    """Persist a dry-run plan as a resumable run, refusing absent local inputs."""
    plan = await plan_reembed(
        corpus,
        target,
        document_batch=document_batch,
        target_batch_tokens=target_batch_tokens,
        chunks_per_second=chunks_per_second,
    )
    if not plan.runnable:
        msg = (
            f"{len(plan.unrepairable)} document(s) have no stored chunks. Re-embedding never "
            "falls back to a connector or parser; repair those local inputs first."
        )
        raise PolicyError(msg)
    return await journal.create(run_id, plan)


async def resume_reembed(
    run_id: str,
    *,
    corpus: ReembedCorpus,
    embedder: Embedder,
    journal: ReembedJournal,
    shadow: ShadowVectorGeneration,
    publisher: ReembedPublisher,
    document_batch: int = DEFAULT_DOCUMENT_BATCH,
    target_batch_tokens: int = DEFAULT_TARGET_BATCH_TOKENS,
) -> ReembedRun:
    """Resume, validate, and publish one durable shadow generation.

    Cancellation and arbitrary exceptions are deliberately not converted into ``FAILED``.
    They leave the last committed checkpoint resumable.  Only a measured validation failure is
    terminal, because retrying identical bytes cannot make an invalid generation valid.
    """
    run = await journal.get(run_id)
    if run is None:
        raise ReembedError(f"no re-embedding run {run_id!r} exists")
    planned_target = EmbedFingerprint.model_validate_json(run.plan.target_config)
    if planned_target != embedder.fingerprint:
        raise ReembedError(
            "the configured embedder is not the target this run planned. Resume with the "
            "same model, weights, dimension, normalization, tokenizer, context limit, and "
            "backend configuration."
        )
    if run.state is ReembedState.PUBLISHED:
        return run
    if run.state is ReembedState.FAILED:
        raise ReembedError(f"re-embedding run {run_id!r} failed validation: {run.failure}")

    await shadow.prepare(run.id, embedder.fingerprint)
    if run.state is ReembedState.PLANNED:
        run = await _save(journal, replace(run, state=ReembedState.BUILDING))

    if run.state is ReembedState.BUILDING:
        run = await _build(
            run,
            corpus=corpus,
            embedder=embedder,
            journal=journal,
            shadow=shadow,
            document_batch=document_batch,
            target_batch_tokens=target_batch_tokens,
        )

    if run.state is ReembedState.VALIDATING:
        try:
            await _validate(run, corpus=corpus, shadow=shadow, document_batch=document_batch)
        except ReembedValidationError as exc:
            failed = replace(run, state=ReembedState.FAILED, failure=str(exc))
            await _save(journal, failed)
            raise
        run = await _save(journal, replace(run, state=ReembedState.READY))

    if run.state is ReembedState.READY:
        await publisher.publish(run)
        run = await _save(journal, replace(run, state=ReembedState.PUBLISHED))
    return run


async def _build(
    run: ReembedRun,
    *,
    corpus: ReembedCorpus,
    embedder: Embedder,
    journal: ReembedJournal,
    shadow: ShadowVectorGeneration,
    document_batch: int,
    target_batch_tokens: int,
) -> ReembedRun:
    while True:
        page = await corpus.select_documents(limit=document_batch, offset=run.next_document)
        if not page:
            return await _save(journal, replace(run, state=ReembedState.VALIDATING))
        for document in page:
            chunks = await corpus.document_chunks(document.id)
            if not chunks and document.expects_chunks:
                raise ReembedError(
                    f"document {document.id!r} lost its stored chunks after planning; the "
                    "published generation is unchanged"
                )
            if chunks:
                size = batch_size(
                    budget_tokens=embedder.fingerprint.max_sequence_length,
                    target_batch_tokens=target_batch_tokens,
                )
                for start in range(0, len(chunks), size):
                    batch = chunks[start : start + size]
                    vectors = await embed_chunks(
                        embedder,
                        batch,
                        target_batch_tokens=target_batch_tokens,
                    )
                    await shadow.upsert(run.id, batch, vectors)
            run = await _save(
                journal,
                replace(
                    run,
                    next_document=run.next_document + 1,
                    documents_completed=run.documents_completed + 1,
                    chunks_completed=run.chunks_completed + len(chunks),
                ),
            )


async def _validate(
    run: ReembedRun,
    *,
    corpus: ReembedCorpus,
    shadow: ShadowVectorGeneration,
    document_batch: int,
) -> None:
    current = await plan_reembed(
        corpus,
        # The canonical identity intentionally omits operational fields such as the context
        # limit.  Reconstruct from the complete, validated configuration recorded by the plan.
        EmbedFingerprint.model_validate_json(run.plan.target_config),
        document_batch=document_batch,
    )
    inspection = await shadow.inspect(run.id)
    failures: list[str] = []
    if current.inventory_digest != run.plan.inventory_digest:
        failures.append("the document/chunk inventory changed during the rebuild")
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
        failures.append("the shadow generation carries a different embedding fingerprint")
    if inspection.inventory_digest != run.plan.inventory_digest:
        failures.append("the shadow generation does not cover the planned inventory")
    if not inspection.lineage_valid:
        failures.append("one or more vector rows have invalid chunk or embedding lineage")
    if not inspection.retrieval_ready:
        failures.append("the shadow generation failed its retrieval-readiness probe")
    if failures:
        raise ReembedValidationError("; ".join(failures))


async def _save(journal: ReembedJournal, proposed: ReembedRun) -> ReembedRun:
    return await journal.save(proposed, expected_revision=proposed.revision)


def _digest_document(digest: _Digest, document: Document, chunks: Sequence[Chunk]) -> None:
    # hashlib's concrete types are intentionally private; this is the tiny structural surface
    # both CPython and the type checker can agree on without importing an implementation type.
    update = digest.update
    fields = (
        document.id,
        document.publication_id,
        document.content_hash,
        document.version_token or "",
        document.parse_fp or "",
    )
    for value in fields:
        update(str(value).encode("utf-8"))
        update(b"\0")
    for chunk in chunks:
        update(chunk.id.encode("utf-8"))
        update(b"\0")
        update(hashlib.sha256(chunk.embed_text.encode("utf-8")).digest())


def _peak_batch_memory(
    chunks: Sequence[Chunk], fingerprint: EmbedFingerprint, target_tokens: int
) -> int:
    """Measured inputs plus returned float32 outputs for the largest bounded batch.

    This intentionally does not claim to price model weights or backend scratch allocations;
    those are fixed costs of loading the configured embedder, not temporary growth of this
    migration.  It does include the two values this orchestration owns simultaneously.
    """
    size = batch_size(
        budget_tokens=fingerprint.max_sequence_length,
        target_batch_tokens=target_tokens,
    )
    vector_bytes = fingerprint.dimension * _FLOAT32_BYTES
    return max(
        (
            sum(len(chunk.embed_text.encode("utf-8")) for chunk in chunks[start : start + size])
            + len(chunks[start : start + size]) * vector_bytes
            for start in range(0, len(chunks), size)
        ),
        default=0,
    )


__all__ = [
    "DEFAULT_CHUNKS_PER_SECOND",
    "DEFAULT_DOCUMENT_BATCH",
    "DEFAULT_TARGET_BATCH_TOKENS",
    "ReembedCorpus",
    "ReembedError",
    "ReembedJournal",
    "ReembedPlan",
    "ReembedPublisher",
    "ReembedRun",
    "ReembedState",
    "ReembedValidationError",
    "ShadowInspection",
    "ShadowVectorGeneration",
    "plan_reembed",
    "resume_reembed",
    "start_reembed",
]
