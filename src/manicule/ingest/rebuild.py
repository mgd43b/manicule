"""Offline construction and atomic publication of replacement derived generations.

The runner has no connector parameter and no connector protocol in its import graph. Its only
source is a promoted manifest plus the blob reader, making a network fallback structurally
impossible rather than a convention a caller may accidentally bypass.
"""

from __future__ import annotations

import asyncio
import errno
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from secrets import randbelow
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    NoReturn,
    Protocol,
    cast,
    override,
    runtime_checkable,
)
from uuid import uuid4

from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from manicule.core.content import (
    Chunk,
    Document,
    DocumentStatus,
    ParsedBlock,
    PipelineStage,
    RawDocument,
)
from manicule.core.errors import ManiculeError
from manicule.core.glossary import GlossaryEntry
from manicule.core.ids import content_hash, document_id
from manicule.core.rebuild import (
    DerivedReplacement,
    RebuildCheckpoint,
    RebuildDerivationError,
    RebuildEstimate,
    RebuildLeaseConflictError,
    RebuildLeaseError,
    RebuildPublicationConflictError,
    RebuildPublicationValidationError,
    RebuildRefusalCode,
    RebuildRefusedError,
    RebuildState,
    RebuildStorageBackendError,
    RebuildStorageCause,
    RebuildStorageDiagnostic,
    RebuildStorageError,
    RebuildStorageOperationError,
    RebuildStorageStage,
    RebuildTarget,
    RebuildTerminalError,
    RebuildTerminalGenerationError,
    RebuildValidationError,
    SnapshotRebuildInput,
)
from manicule.ingest.embedding import embed_or_reuse
from manicule.ingest.glossary_lineage import glossary_fingerprint
from manicule.ingest.middleware import MiddlewareRunner
from manicule.ingest.workers import AttemptResult, ParseRunner, StageResult, retained_size
from manicule.parsers.chain import ChainResult, container_result, run_chain
from manicule.parsers.expansion import (
    ExpandedMember,
    MemberFailure,
    member_raw,
    member_title,
)
from manicule.parsers.versions import parse_fingerprint

MAX_MISSING_DETAILS = 1000

_LOG = logging.getLogger(__name__)
_STORAGE_RETRY_ATTEMPTS: Final = 3
_STORAGE_RETRY_BASE_SECONDS: Final = 0.05
_STORAGE_RETRY_MAX_SECONDS: Final = 1.0
_STORAGE_RETRY_MAX_ELAPSED_SECONDS: Final = 5.0


@dataclass(slots=True)
class _LeaseHeartbeat:
    """What the renewal timer has done, for the body it is holding a lease for.

    ``lost`` is what separates the two ways a build stops at a cancellation: the caller asked
    for one, or this worker no longer owns the generation. They arrive as the same exception
    and mean opposite things.
    """

    lost: bool = False
    renewals: int = 0
    storage_error: BaseException | None = None
    irreversible_publication: bool = False


class _StagedStorageError(RuntimeError):
    """Private transport for a raw storage failure plus its bounded operation stage."""

    def __init__(
        self,
        stage: RebuildStorageStage,
        error: BaseException,
        diagnostic: RebuildStorageDiagnostic | None = None,
    ) -> None:
        super().__init__(stage.value)
        self.stage = stage
        self.error = error
        self.diagnostic = diagnostic


def _storage_root(error: BaseException) -> BaseException:
    """Unwrap SQLAlchemy's DBAPI wrapper without ever rendering its message."""
    if isinstance(error, SQLAlchemyError):
        original = getattr(error, "orig", None)
        if isinstance(original, BaseException):
            return original
    return error


def _classify_storage_failure(error: BaseException) -> tuple[RebuildStorageCause, bool]:
    """Return a safe cause and a deliberately narrow retry decision.

    Driver messages are intentionally not consulted except for SQLite's fixed busy/locked
    signals.  A word such as ``timeout`` in arbitrary SQL or a path is not a contract and must
    never turn an unknown failure into an unattended retry loop.
    """
    cause = RebuildStorageCause.UNKNOWN
    retryable = False
    if isinstance(error, RebuildStorageBackendError):
        return RebuildStorageCause.VECTOR_STORAGE, retryable
    if isinstance(error, IntegrityError):
        return RebuildStorageCause.DATABASE_INTEGRITY, retryable
    root = _storage_root(error)
    if isinstance(root, RebuildStorageBackendError):
        cause = RebuildStorageCause.VECTOR_STORAGE
    elif isinstance(root, OSError):
        if root.errno == errno.ENOSPC:
            cause = RebuildStorageCause.CAPACITY
        elif root.errno in {errno.EACCES, errno.EPERM, errno.EROFS}:
            cause = RebuildStorageCause.PERMISSION
        else:
            cause = RebuildStorageCause.IO
            retryable = root.errno in {
                errno.EAGAIN,
                errno.EINTR,
                errno.ETIMEDOUT,
                errno.ECONNABORTED,
                errno.ECONNRESET,
                errno.ENETDOWN,
                errno.ENETUNREACH,
            }
    elif isinstance(error, SQLAlchemyError):
        # sqlite3 exposes numeric result codes without needing its unbounded diagnostic text.
        code = getattr(root, "sqlite_errorcode", None)
        if isinstance(code, int) and (code & 0xFF) in {5, 6}:  # SQLITE_BUSY / SQLITE_LOCKED
            cause = RebuildStorageCause.BUSY
            retryable = True
    return cause, retryable


def _storage_hint(cause: RebuildStorageCause, *, retryable: bool) -> str:
    if retryable:
        return "Retry is delayed; the live lease and staged namespace remain in use."
    if cause is RebuildStorageCause.BUSY:
        return "Automatic retry was not scheduled; wait for storage contention to clear."
    return {
        RebuildStorageCause.CAPACITY: "Free durable storage, then resume the same generation.",
        RebuildStorageCause.PERMISSION: (
            "Restore storage permissions, then resume the same generation."
        ),
        RebuildStorageCause.DATABASE_INTEGRITY: (
            "Inspect durable storage integrity before resuming this generation."
        ),
        RebuildStorageCause.VECTOR_STORAGE: (
            "Repair the vector-store backend before resuming this generation."
        ),
        RebuildStorageCause.IO: "Check durable storage and I/O health before resuming.",
        RebuildStorageCause.UNKNOWN: (
            "Inspect storage health before resuming; no retry was scheduled."
        ),
    }[cause]


LEASE_RENEWALS_PER_LEASE: Final = 3
"""Renewals attempted per lease duration, over everything :meth:`_renewing_lease` covers.

That is the whole build — a takeover's checkpoint replay and the document loop after it — rather
than either one alone, because both are unbounded in aggregate and either can outlast the lease
its generation was claimed under.

Three rather than two, so a single failed or delayed renewal is survivable: renewals land at a
third, two thirds and the whole of the lease, so losing the first still leaves one before the
expiry. At two the first miss is already the last chance, and a heartbeat that cannot tolerate
one bad round is a heartbeat that turns a slow database into a lost generation. The same ratio
the acquisition heartbeat and the adaptive enumeration recorder use, kept identical because an
operator reasoning about one lease should not have to learn a second cadence.
"""

if TYPE_CHECKING:
    from manicule.core.embedding import Vector
    from manicule.core.fingerprints import ChunkFingerprint
    from manicule.core.protocols import Chunker, Embedder, VectorStore
    from manicule.parsers.chain import Attempt, ParserChain


@runtime_checkable
class RebuildStore(Protocol):
    """Durable shadow-generation operations required by the offline runner."""

    async def plan_rebuild(
        self,
        snapshot_run_id: str,
        target: RebuildTarget,
        *,
        missing_limit: int,
        persist: bool = True,
    ) -> RebuildEstimate: ...

    async def checkpoint(self, generation_id: str) -> RebuildCheckpoint: ...

    async def claim_generation(
        self, generation_id: str, owner: str, *, now: datetime, expires_at: datetime
    ) -> RebuildCheckpoint: ...

    async def renew_generation(
        self,
        generation_id: str,
        owner: str,
        lease_generation: int,
        *,
        now: datetime,
        expires_at: datetime,
    ) -> RebuildCheckpoint: ...

    async def assert_generation_lease(
        self,
        generation_id: str,
        owner: str,
        lease_generation: int,
        *,
        now: datetime,
    ) -> None: ...

    async def copy_checkpointed_vectors(
        self,
        generation_id: str,
        source_publication_id: str,
        *,
        owner: str,
        lease_generation: int,
        now: datetime,
        cancel: asyncio.Event | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None: ...

    async def snapshot_inputs(
        self, generation_id: str, *, after_sequence: int, limit: int
    ) -> Sequence[SnapshotRebuildInput]: ...

    async def check_capacity(
        self, generation_id: str, replacements: Sequence[DerivedReplacement]
    ) -> None: ...

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
        """Atomically stage a batch and advance its sequence CAS.

        Implementations must treat an already-identical batch as success and reject different
        output at the same sequence. This is the crash/retry idempotency boundary.
        """
        ...

    async def begin_validation(
        self,
        generation_id: str,
        *,
        owner: str,
        lease_generation: int,
        now: datetime,
    ) -> RebuildCheckpoint: ...

    async def validate_generation(
        self,
        generation_id: str,
        *,
        owner: str,
        lease_generation: int,
        now: datetime,
        cancel: asyncio.Event | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None: ...

    async def publish_generation(
        self,
        generation_id: str,
        *,
        owner: str,
        lease_generation: int,
        now: datetime,
    ) -> RebuildCheckpoint:
        """Flip document, chunk, FTS, glossary and generation pointers in one transaction."""
        ...

    async def cancel_generation(
        self,
        generation_id: str,
        *,
        owner: str,
        lease_generation: int,
        now: datetime,
    ) -> RebuildCheckpoint: ...

    async def fail_generation(
        self,
        generation_id: str,
        code: RebuildRefusalCode,
        *,
        owner: str,
        lease_generation: int,
        now: datetime,
    ) -> RebuildCheckpoint: ...

    async def release_generation(
        self,
        generation_id: str,
        code: RebuildRefusalCode,
        *,
        owner: str,
        lease_generation: int,
        now: datetime,
    ) -> RebuildCheckpoint:
        """Record why this attempt stopped and give the generation up, without ending it.

        The other settlement, :meth:`fail_generation`, is terminal: a generation it marks can
        never be claimed again, and its committed prefix is only reachable through cleanup and
        a fresh plan. That is right for a refusal the run has diagnosed — a corrupt manifest, a
        replacement that does not validate — where retrying the same work would fail the same
        way.

        It is wrong for contention. A writer holding SQLite past the busy timeout, a disk that
        was briefly full, a filesystem error that cleared: those say nothing about the work
        already committed, and ending the generation over one of them discards every document
        derived so far. This records the diagnostic and releases the lease, so status can say
        what happened and to whom it happened — nobody — while the next run takes the
        generation over and resumes from its checkpoint.
        """
        ...


@runtime_checkable
class RebuildBlobSource(Protocol):
    """Read-only access to retained bytes; deliberately has no retain or fetch method."""

    async def get_bounded(self, digest: str, *, max_bytes: int) -> bytes | None: ...


@runtime_checkable
class RebuildStorageDiagnosticRecorder(Protocol):
    """Optional capability for a store that can persist safe retry observations."""

    async def record_storage_diagnostic(
        self,
        generation_id: str,
        diagnostic: RebuildStorageDiagnostic,
        *,
        owner: str,
        lease_generation: int,
        now: datetime,
    ) -> None: ...


@runtime_checkable
class OfflineDeriver(Protocol):
    """Parser/routing/chunker/tokenizer pipeline used for one retained source envelope."""

    async def prepare(
        self,
        raw: RawDocument,
        target: RebuildTarget,
        *,
        generation_id: str,
        blob_ref: str,
        title: str,
        version_token: str | None,
        connector: str | None = None,
    ) -> PreparedReplacement: ...

    async def stage(self, prepared: PreparedReplacement, *, publication_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class PreparedReplacement:
    """A derived document and vectors held in memory but not yet persisted."""

    replacement: DerivedReplacement
    vectors: tuple[Vector, ...]
    memory_bytes: int
    temporary_bytes: int


@runtime_checkable
class RelationalDeriver(Protocol):
    """Offline parser/router/chunker/glossary half of a generation derivation."""

    async def derive(
        self,
        raw: RawDocument,
        target: RebuildTarget,
        *,
        generation_id: str,
        blob_ref: str,
        title: str,
        version_token: str | None,
        connector: str | None = None,
    ) -> DerivedReplacement: ...


class BoundedParseRunner(ParseRunner, Protocol):
    """Production parse runner that caps the child reply before parent allocation."""

    @override
    async def run_attempt(
        self,
        name: str,
        raw: RawDocument,
        *,
        max_output_bytes: int | None = None,
        memory_limit_bytes: int | None = None,
    ) -> AttemptResult: ...

    async def open_stage_session(self, *, memory_limit_bytes: int) -> BoundedStageSession: ...


class BoundedStageSession(Protocol):
    """One document's stateful isolated middleware/chunker lifetime."""

    async def run_before_parse(
        self, raw: RawDocument, *, max_output_bytes: int, memory_limit_bytes: int
    ) -> StageResult: ...

    async def run_after_parse_and_chunk(
        self,
        document: Document,
        blocks: list[ParsedBlock],
        *,
        max_output_bytes: int,
        memory_limit_bytes: int,
        title: str,
        media_type: str,
        detect_glossary: bool,
    ) -> StageResult: ...

    async def aclose(self) -> None: ...


class ParserChunkerRelationalDeriver:
    """Production offline parser-routing, chunking and glossary derivation."""

    def __init__(
        self,
        *,
        workspace_id: str,
        source: str,
        parser_chain: ParserChain,
        routing_identity: str,
        chunker: Chunker,
        middleware: MiddlewareRunner | None = None,
        parse_runner: BoundedParseRunner | None = None,
        detect_glossary: bool = True,
    ) -> None:
        self._workspace_id = workspace_id
        self._source = source
        self._parser_chain = parser_chain
        self._routing_identity = routing_identity
        self._chunker = chunker
        self._middleware = middleware or MiddlewareRunner(())
        self._parse_runner = parse_runner
        self._detect_glossary = detect_glossary

    @staticmethod
    def _require_stage_bound(value: object, budget: int) -> None:
        if retained_size(value) > budget:
            raise ValueError(RebuildRefusalCode.MEMORY_BOUND.value)

    async def _parse(
        self, raw: RawDocument, target: RebuildTarget, *, budget: int
    ) -> tuple[ChainResult, tuple[object, ...]]:
        if self._parse_runner is None:
            return await self._parser_chain.run(raw), ()

        parse_runner = self._parse_runner
        captured: tuple[object, ...] = ()

        async def attempt(name: str, document: RawDocument) -> tuple[list[ParsedBlock], Attempt]:
            nonlocal captured
            output_budget = max(0, min(budget, (target.max_memory_bytes - 4096) // 6))
            outcome = await parse_runner.run_attempt(
                name,
                document,
                max_output_bytes=output_budget,
                memory_limit_bytes=target.max_memory_bytes,
            )
            if outcome.attempt.reason == RebuildRefusalCode.MEMORY_BOUND.value:
                raise ValueError(RebuildRefusalCode.MEMORY_BOUND.value)
            if outcome.members:
                member_bytes = 0
                for member in outcome.members:
                    if isinstance(member, ExpandedMember):
                        content = member.raw.content
                        member_bytes += (
                            len(content) if isinstance(content, bytes) else len(content.encode())
                        )
                if member_bytes * 6 + 4096 > target.max_memory_bytes:
                    raise ValueError(RebuildRefusalCode.MEMORY_BOUND.value)
                captured = outcome.members
            return outcome.blocks, outcome.attempt

        result = await run_chain(self._parser_chain.resolve(raw.media_type), raw, attempt)
        return (container_result(len(captured)) if captured else result), captured

    async def derive(
        self,
        raw: RawDocument,
        target: RebuildTarget,
        *,
        generation_id: str,
        blob_ref: str,
        title: str,
        version_token: str | None,
        connector: str | None = None,
    ) -> DerivedReplacement:
        source = connector or self._source
        root, members = await self._derive_one(
            raw,
            target,
            generation_id=generation_id,
            blob_ref=blob_ref,
            title=title,
            version_token=version_token,
            source=source,
            budget=target.max_memory_bytes,
        )
        nodes = [root]
        children: dict[int, list[int]] = {0: []}
        queue: list[tuple[int, object]] = [(0, member) for member in members]
        self._require_stage_bound((nodes, children, queue), target.max_memory_bytes)
        while queue:
            parent, member = queue.pop(0)
            retained = retained_size((nodes, children, queue, member))
            remaining = target.max_memory_bytes - retained
            if remaining <= 0:
                raise ValueError(RebuildRefusalCode.MEMORY_BOUND.value)
            if isinstance(member, ExpandedMember):
                child, deeper = await self._derive_one(
                    member_raw(member),
                    target,
                    generation_id=generation_id,
                    blob_ref="",
                    title=member_title(member),
                    version_token=None,
                    budget=remaining,
                    source=source,
                )
            elif isinstance(member, MemberFailure):
                diagnostic = ChainResult(
                    blocks=[],
                    status=member.status,
                    status_detail=member.reason,
                    failed_stage=(
                        PipelineStage.PARSE if member.status is DocumentStatus.FAILED else None
                    ),
                )
                child = DerivedReplacement(
                    document=Document(
                        id=document_id(self._workspace_id, source, member.source_id),
                        publication_id=generation_id,
                        source=source,
                        source_id=member.source_id,
                        uri=member.uri,
                        title="",
                        content_hash=content_hash(member.uri),
                        media_type="application/octet-stream",
                        status=member.status,
                        status_detail=member.reason,
                        failed_stage=diagnostic.failed_stage,
                        metadata={**dict(member.metadata), **diagnostic.metadata},
                    )
                )
                deeper = ()
            else:
                raise TypeError("parser runner returned an invalid container member")
            index = len(nodes)
            nodes.append(child)
            children.setdefault(parent, []).append(index)
            children[index] = []
            queue.extend((index, nested) for nested in deeper)
            del member
            self._require_stage_bound((nodes, children, queue), target.max_memory_bytes)

        for index in range(len(nodes) - 1, -1, -1):
            nodes[index] = nodes[index].model_copy(
                update={"members": tuple(nodes[child] for child in children[index])}
            )
        return nodes[0]

    async def _derive_one(
        self,
        raw: RawDocument,
        target: RebuildTarget,
        *,
        generation_id: str,
        blob_ref: str,
        title: str,
        version_token: str | None,
        budget: int,
        source: str,
    ) -> tuple[DerivedReplacement, tuple[object, ...]]:
        runner = self._parse_runner
        if runner is None:
            raise RuntimeError("offline derivation requires an isolated stage runner")
        session = await runner.open_stage_session(memory_limit_bytes=target.max_memory_bytes)
        try:
            return await self._derive_one_in_session(
                raw,
                target,
                generation_id=generation_id,
                blob_ref=blob_ref,
                title=title,
                version_token=version_token,
                budget=budget,
                session=session,
                source=source,
            )
        finally:
            await session.aclose()

    async def _derive_one_in_session(  # noqa: PLR0912, PLR0915 - ordered state machine
        self,
        raw: RawDocument,
        target: RebuildTarget,
        *,
        generation_id: str,
        blob_ref: str,
        title: str,
        version_token: str | None,
        budget: int,
        session: BoundedStageSession,
        source: str,
    ) -> tuple[DerivedReplacement, tuple[object, ...]]:
        if target.parser_routing != self._routing_identity:
            raise ValueError("target parser-routing identity does not match the configured chain")
        original = raw
        raw_bytes = retained_size(raw)
        if raw_bytes >= budget:
            raise ValueError(RebuildRefusalCode.MEMORY_BOUND.value)
        members: tuple[object, ...] = ()
        before = await session.run_before_parse(
            raw,
            max_output_bytes=budget - raw_bytes,
            memory_limit_bytes=target.max_memory_bytes,
        )
        self._raise_stage_failure(before)
        transformed = before.value
        if transformed is not None and not isinstance(transformed, RawDocument):
            raise TypeError("before_parse stage returned an invalid value")
        if transformed is None:
            result = ChainResult(
                blocks=[],
                status=DocumentStatus.SKIPPED,
                status_detail="a middleware hook excluded this document before parsing",
            )
            raw = original
        else:
            raw = transformed
            parse_budget = budget - retained_size((original, raw))
            if parse_budget <= 0:
                raise ValueError(RebuildRefusalCode.MEMORY_BOUND.value)
            result, members = await self._parse(raw, target, budget=parse_budget)
        if transformed is None:
            members = ()
        if result.status is DocumentStatus.FAILED:
            raise ValueError(RebuildRefusalCode.DERIVATION_FAILED.value)
        identifier = document_id(self._workspace_id, source, raw.source_id)
        document = Document(
            id=identifier,
            publication_id=generation_id,
            source=source,
            source_id=raw.source_id,
            uri=raw.uri,
            title=title,
            content_hash=content_hash(original.as_bytes()),
            version_token=version_token,
            original_ref=blob_ref or None,
            media_type=raw.media_type,
            status=result.status,
            status_detail=result.status_detail or None,
            failed_stage=result.failed_stage,
            metadata={**dict(raw.metadata), **result.metadata},
        )
        self._require_stage_bound((original, raw, result, members, document), budget)
        lineage = parse_fingerprint(result.parser_used) if result.parser_used else None
        canonical_lineage = lineage.canonical() if lineage is not None else None
        if canonical_lineage is not None and canonical_lineage not in target.parser_set:
            raise ValueError("installed parser identity is absent from the rebuild target")
        chunks: tuple[Chunk, ...] = ()
        if result.status is DocumentStatus.PARSED:
            retained = retained_size((original, raw, result, members, document))
            remaining = budget - retained
            if remaining <= 0:
                raise ValueError(RebuildRefusalCode.MEMORY_BOUND.value)
            staged = await session.run_after_parse_and_chunk(
                document,
                result.blocks,
                max_output_bytes=remaining,
                memory_limit_bytes=target.max_memory_bytes,
                title=title,
                media_type=raw.media_type,
                detect_glossary=self._detect_glossary,
            )
            self._raise_stage_failure(staged)
            value = staged.value
            if not isinstance(value, tuple):
                raise TypeError("chunk stage returned an invalid value")
            pair = cast("tuple[object, ...]", value)
            if len(pair) != 2:  # noqa: PLR2004 - pair contract
                raise TypeError("chunk stage returned an invalid value")
            chunks_value, entries_value = pair
            if not isinstance(chunks_value, tuple) or not isinstance(entries_value, tuple):
                raise TypeError("chunk stage returned an invalid value")
            candidate = cast("tuple[object, ...]", chunks_value)
            if not all(isinstance(chunk, Chunk) for chunk in candidate):
                raise TypeError("chunk stage returned an invalid value")
            chunks = cast("tuple[Chunk, ...]", candidate)
            entry_candidates = cast("tuple[object, ...]", entries_value)
            if not all(isinstance(entry, GlossaryEntry) for entry in entry_candidates):
                raise TypeError("glossary stage returned an invalid value")
            entries = cast("tuple[GlossaryEntry, ...]", entry_candidates)
            self._require_stage_bound(
                (original, raw, result, members, document, chunks, entries), budget
            )
        else:
            entries = ()
        status = DocumentStatus.INDEXED if chunks else result.status
        if result.status is DocumentStatus.PARSED and not chunks:
            status = DocumentStatus.NO_EXTRACTABLE_TEXT
        settled = document.model_copy(
            update={
                "status": status,
                "status_detail": (
                    "the configured chunker produced no retrievable chunks"
                    if status is DocumentStatus.NO_EXTRACTABLE_TEXT
                    and result.status is DocumentStatus.PARSED
                    else document.status_detail
                ),
                "failed_stage": None,
            }
        )
        installed_glossary = glossary_fingerprint(
            enabled=self._detect_glossary, middleware=self._middleware.chain()
        ).canonical()
        if installed_glossary != target.glossary_fingerprint:
            raise ValueError("target glossary identity does not match the installed detector")
        replacement = DerivedReplacement(
            document=settled,
            chunks=chunks,
            glossary=entries,
            parse_fingerprint=canonical_lineage,
        )
        self._require_stage_bound((replacement, members), budget)
        return replacement, members

    @staticmethod
    def _raise_stage_failure(result: StageResult) -> None:
        if result.reason is None:
            return
        code = (
            RebuildRefusalCode.MEMORY_BOUND
            if result.reason.value == RebuildRefusalCode.MEMORY_BOUND.value
            else RebuildRefusalCode.DERIVATION_FAILED
        )
        raise ValueError(code.value)


class EmbeddingOfflineDeriver:
    """Bind relational derivation to identity-safe vector reuse and staging."""

    def __init__(
        self,
        *,
        relational: RelationalDeriver,
        embedder: Embedder,
        vectors: VectorStore,
        chunk_fingerprint: ChunkFingerprint,
        previous_inputs: Mapping[str, str] | None = None,
    ) -> None:
        self._relational = relational
        self._embedder = embedder
        self._vectors = vectors
        self._chunk_fingerprint = chunk_fingerprint
        self._previous_inputs = previous_inputs

    async def prepare(
        self,
        raw: RawDocument,
        target: RebuildTarget,
        *,
        generation_id: str,
        blob_ref: str,
        title: str,
        version_token: str | None,
        connector: str | None = None,
    ) -> PreparedReplacement:
        if target.embedding_fingerprint != self._embedder.fingerprint.canonical():
            raise ValueError("target embedding identity does not match the configured embedder")
        if target.chunk_fingerprint != self._chunk_fingerprint.canonical():
            raise ValueError("target chunk identity does not match the configured chunker")
        replacement = await self._relational.derive(
            raw,
            target,
            generation_id=generation_id,
            blob_ref=blob_ref,
            title=title,
            version_token=version_token,
            connector=connector,
        )
        if any(item.vector_reused or item.vector_embedded for item in replacement.flattened()):
            raise ValueError("relational derivation must not claim vector work")
        payload_bytes = len(replacement.model_dump_json().encode())
        text_bytes = sum(
            len(chunk.text.encode()) + len(chunk.embed_text.encode())
            for chunk in replacement.flattened_chunks()
        )
        vector_bytes = (
            len(replacement.flattened_chunks()) * self._embedder.fingerprint.dimension * 4
        )
        # Raw + parser/container objects + serialized replacement + chunk text + model input
        # and output scratch. The factor is intentionally charged before the embedder call.
        memory_bytes = len(raw.as_bytes()) * 3 + payload_bytes + text_bytes * 2 + vector_bytes * 2
        temporary_bytes = payload_bytes * 2 + text_bytes * 2 + vector_bytes * 3
        if memory_bytes > target.max_memory_bytes:
            raise ValueError(RebuildRefusalCode.MEMORY_BOUND.value)
        if temporary_bytes > target.max_temporary_bytes:
            raise ValueError(RebuildRefusalCode.TEMP_DISK_BOUND.value)
        stored = await self._vectors.fingerprint()
        if stored is None:
            await self._vectors.ensure_ready(
                self._embedder.fingerprint,
                embed_text_middleware=self._chunk_fingerprint.embed_text_middleware,
            )
        else:
            self._embedder.fingerprint.require_match(stored)
        staged_vectors: list[Vector] = []

        async def complete(item: DerivedReplacement) -> DerivedReplacement:
            vectors, work = await embed_or_reuse(
                self._embedder,
                item.chunks,
                vectors=self._vectors,
                chunk_fingerprint=self._chunk_fingerprint,
                previous=self._previous_inputs,
            )
            staged_vectors.extend(vectors)
            members = tuple([await complete(member) for member in item.members])
            return item.model_copy(
                update={
                    "members": members,
                    "vector_reused": work.reused,
                    "vector_embedded": work.embedded,
                }
            )

        completed = await complete(replacement)
        return PreparedReplacement(
            replacement=completed,
            vectors=tuple(staged_vectors),
            memory_bytes=memory_bytes,
            temporary_bytes=temporary_bytes,
        )

    async def stage(self, prepared: PreparedReplacement, *, publication_id: str) -> None:
        await self._vectors.upsert(
            prepared.replacement.flattened_chunks(),
            prepared.vectors,
            publication_id=publication_id,
        )


class OfflineGenerationRebuilder:
    """Bounded, resumable executor over one workspace set of promoted snapshots."""

    def __init__(
        self,
        *,
        store: RebuildStore,
        blobs: RebuildBlobSource,
        deriver: OfflineDeriver,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = store
        self._blobs = blobs
        self._deriver = deriver
        self._clock = clock or (lambda: datetime.now(UTC))

    def _storage_diagnostic(
        self,
        checkpoint: RebuildCheckpoint | None,
        *,
        stage: RebuildStorageStage,
        error: BaseException,
        correlation_id: str,
        retry_count: int,
        next_retry_at: datetime | None = None,
        event: Literal["primary", "settlement"] = "primary",
        retryable_override: bool | None = None,
    ) -> RebuildStorageDiagnostic:
        cause, classified_retryable = _classify_storage_failure(error)
        retryable = classified_retryable if retryable_override is None else retryable_override
        return RebuildStorageDiagnostic(
            event=event,
            stage=stage,
            cause=cause,
            retryable=retryable and event == "primary",
            namespace_usable=retryable and event == "primary",
            replayed_items=0 if checkpoint is None else checkpoint.replayed_items,
            replayed_vectors=0 if checkpoint is None else checkpoint.replayed_vectors,
            validated_items=0 if checkpoint is None else checkpoint.validated_items,
            validated_vectors=0 if checkpoint is None else checkpoint.validated_vectors,
            occurred_at=self._clock(),
            correlation_id=correlation_id,
            retry_count=retry_count,
            next_retry_at=next_retry_at,
            operator_hint=_storage_hint(cause, retryable=retryable and event == "primary"),
        )

    @staticmethod
    def _emit_storage_diagnostic(diagnostic: RebuildStorageDiagnostic) -> None:
        """Log only the reviewed model, never a driver exception or its formatted traceback."""
        _LOG.warning(
            "rebuild storage diagnostic",
            extra={"rebuild_storage": diagnostic.model_dump(mode="json")},
        )

    async def _record_storage_diagnostic(
        self,
        checkpoint: RebuildCheckpoint,
        owner: str,
        diagnostic: RebuildStorageDiagnostic,
    ) -> None:
        """Best-effort durable evidence; unsupported alternate stores retain log evidence."""
        if not isinstance(self._store, RebuildStorageDiagnosticRecorder):
            return
        try:
            await self._store.record_storage_diagnostic(
                checkpoint.generation_id,
                diagnostic,
                owner=owner,
                lease_generation=checkpoint.lease_generation,
                now=self._clock(),
            )
        except (
            SQLAlchemyError,
            OSError,
            RebuildStorageBackendError,
            RebuildLeaseConflictError,
            KeyError,
        ) as exc:
            settlement = self._storage_diagnostic(
                checkpoint,
                stage=RebuildStorageStage.RELEASE,
                error=exc,
                correlation_id=diagnostic.correlation_id,
                retry_count=diagnostic.retry_count,
                event="settlement",
            )
            self._emit_storage_diagnostic(settlement)

    async def _renew_generation(
        self, checkpoint: RebuildCheckpoint, owner: str, *, lease_seconds: int
    ) -> RebuildCheckpoint:
        renewed_at = self._clock()
        try:
            return await self._store.renew_generation(
                checkpoint.generation_id,
                owner,
                checkpoint.lease_generation,
                now=renewed_at,
                expires_at=renewed_at + timedelta(seconds=lease_seconds),
            )
        except (SQLAlchemyError, OSError) as exc:
            raise _StagedStorageError(RebuildStorageStage.LEASE_RENEWAL, exc) from exc

    async def _assert_generation_lease(self, checkpoint: RebuildCheckpoint, owner: str) -> None:
        try:
            await self._store.assert_generation_lease(
                checkpoint.generation_id,
                owner,
                checkpoint.lease_generation,
                now=self._clock(),
            )
        except (SQLAlchemyError, OSError, RebuildStorageBackendError) as exc:
            raise _StagedStorageError(RebuildStorageStage.LEASE_ASSERTION, exc) from exc

    async def _checkpoint_for_diagnostic(self, checkpoint: RebuildCheckpoint) -> RebuildCheckpoint:
        """Prefer the most recently committed aggregate evidence without masking the failure."""
        try:
            return await self._store.checkpoint(checkpoint.generation_id)
        except (
            SQLAlchemyError,
            OSError,
            RebuildStorageBackendError,
            RebuildLeaseConflictError,
            KeyError,
        ):
            return checkpoint

    async def _retry_storage_operation(
        self,
        checkpoint: RebuildCheckpoint,
        owner: str,
        *,
        stage: RebuildStorageStage,
        operation: Callable[[], Awaitable[None]],
        cancel: asyncio.Event | None,
    ) -> None:
        """Retry only a classified transient, fenced operation under this unchanged lease."""
        correlation_id = uuid4().hex
        attempts = 0
        started = asyncio.get_running_loop().time()
        while True:
            try:
                await operation()
            except RebuildStorageOperationError as failure:
                failed_stage = failure.stage
                error = (
                    failure.__cause__ if isinstance(failure.__cause__, BaseException) else failure
                )
            except _StagedStorageError as failure:
                failed_stage = failure.stage
                error = failure.error
            except (SQLAlchemyError, OSError, RebuildStorageBackendError) as exc:
                failed_stage = stage
                error = exc
            else:
                return

            _, retryable = _classify_storage_failure(error)
            attempts += 1
            remaining = _STORAGE_RETRY_MAX_ELAPSED_SECONDS - (
                asyncio.get_running_loop().time() - started
            )
            can_retry = (
                retryable
                and attempts < _STORAGE_RETRY_ATTEMPTS
                and remaining > 0
                and (cancel is None or not cancel.is_set())
            )
            delay = 0.0
            next_retry_at: datetime | None = None
            if can_retry:
                nominal = min(
                    _STORAGE_RETRY_BASE_SECONDS * (2 ** (attempts - 1)),
                    _STORAGE_RETRY_MAX_SECONDS,
                    remaining,
                )
                # Jitter is intentionally generated without reusing the process-global PRNG:
                # retry timing is operational metadata, but no deterministic test should make
                # coincident writers march in lockstep.
                jitter = 0.75 + randbelow(501) / 1000
                delay = min(nominal * jitter, remaining)
                next_retry_at = self._clock() + timedelta(seconds=delay)
            diagnostic_checkpoint = await self._checkpoint_for_diagnostic(checkpoint)
            diagnostic = self._storage_diagnostic(
                diagnostic_checkpoint,
                stage=failed_stage,
                error=error,
                correlation_id=correlation_id,
                retry_count=attempts,
                next_retry_at=next_retry_at,
                retryable_override=can_retry,
            )
            self._emit_storage_diagnostic(diagnostic)
            await self._record_storage_diagnostic(checkpoint, owner, diagnostic)
            if not can_retry:
                raise _StagedStorageError(failed_stage, error, diagnostic) from error

            if cancel is not None:
                try:
                    await asyncio.wait_for(cancel.wait(), timeout=delay)
                except TimeoutError:
                    pass
                else:
                    raise asyncio.CancelledError
            else:
                await asyncio.sleep(delay)
            # Retrying a page is only safe if this worker still owns exactly the same fenced
            # generation.  The heartbeat remains active throughout the delay.
            await self._assert_generation_lease(checkpoint, owner)

    @staticmethod
    async def _join_irreversible[T](work: asyncio.Task[T]) -> T:
        """Join an atomic publication before propagating task cancellation."""
        current = asyncio.current_task()
        cancellation: asyncio.CancelledError | None = None
        while not work.done():
            try:
                await asyncio.shield(work)
            except asyncio.CancelledError as error:
                cancellation = error
                if current is not None:
                    current.uncancel()
        result = work.result()
        if cancellation is not None:
            raise cancellation
        return result

    async def _fail_build_conflict(
        self,
        checkpoint: RebuildCheckpoint,
        owner: str,
        error: RebuildPublicationConflictError,
    ) -> NoReturn:
        await self._store.fail_generation(
            checkpoint.generation_id,
            error.code,
            owner=owner,
            lease_generation=checkpoint.lease_generation,
            now=self._clock(),
        )
        raise RebuildLeaseError from error

    async def _fail_build_validation(
        self,
        checkpoint: RebuildCheckpoint,
        owner: str,
        error: RebuildPublicationValidationError,
    ) -> NoReturn:
        await self._store.fail_generation(
            checkpoint.generation_id,
            error.code,
            owner=owner,
            lease_generation=checkpoint.lease_generation,
            now=self._clock(),
        )
        raise RebuildValidationError from error

    async def dry_run(
        self, snapshot_run_id: str, target: RebuildTarget, *, missing_limit: int = 100
    ) -> RebuildEstimate:
        if missing_limit <= 0 or missing_limit > MAX_MISSING_DETAILS:
            raise ValueError("missing_limit must be between 1 and 1000")
        return await self._store.plan_rebuild(
            snapshot_run_id, target, missing_limit=missing_limit, persist=False
        )

    async def run(
        self,
        snapshot_run_id: str,
        target: RebuildTarget,
        *,
        missing_limit: int = 100,
        cancel: asyncio.Event | None = None,
        owner: str | None = None,
        lease_seconds: int = 300,
    ) -> RebuildCheckpoint:
        """Execute with one bounded public failure vocabulary at the durable boundary."""
        try:
            return await self._run(
                snapshot_run_id,
                target,
                missing_limit=missing_limit,
                cancel=cancel,
                owner=owner,
                lease_seconds=lease_seconds,
            )
        except RebuildTerminalGenerationError as exc:
            raise RebuildTerminalError from exc
        except RebuildLeaseConflictError as exc:
            raise RebuildLeaseError from exc
        except RebuildStorageBackendError as exc:
            diagnostic = self._storage_diagnostic(
                None,
                stage=RebuildStorageStage.PLAN,
                error=exc,
                correlation_id=uuid4().hex,
                retry_count=0,
            )
            self._emit_storage_diagnostic(diagnostic)
            raise RebuildStorageError(diagnostic) from exc
        except RebuildPublicationValidationError as exc:
            raise RebuildValidationError from exc
        except RebuildStorageOperationError as exc:
            error = exc.__cause__ if isinstance(exc.__cause__, BaseException) else exc
            diagnostic = self._storage_diagnostic(
                None,
                stage=exc.stage,
                error=error,
                correlation_id=uuid4().hex,
                retry_count=0,
            )
            self._emit_storage_diagnostic(diagnostic)
            raise RebuildStorageError(diagnostic) from exc
        except _StagedStorageError as exc:
            diagnostic = exc.diagnostic or self._storage_diagnostic(
                None,
                stage=exc.stage,
                error=exc.error,
                correlation_id=uuid4().hex,
                retry_count=0,
            )
            if exc.diagnostic is None:
                self._emit_storage_diagnostic(diagnostic)
            raise RebuildStorageError(diagnostic) from exc.error
        except (SQLAlchemyError, OSError) as exc:
            # Driver messages can contain statement text and bound parameter values. The cause
            # is retained for server diagnostics but can never cross the application boundary.
            diagnostic = self._storage_diagnostic(
                None,
                stage=RebuildStorageStage.PLAN,
                error=exc,
                correlation_id=uuid4().hex,
                retry_count=0,
            )
            self._emit_storage_diagnostic(diagnostic)
            raise RebuildStorageError(diagnostic) from exc

    async def _run(  # noqa: PLR0912 - explicit lifecycle/error boundary
        self,
        snapshot_run_id: str,
        target: RebuildTarget,
        *,
        missing_limit: int = 100,
        cancel: asyncio.Event | None = None,
        owner: str | None = None,
        lease_seconds: int = 300,
    ) -> RebuildCheckpoint:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        owner = owner or f"offline-rebuild-{uuid4()}"
        estimate = await self._store.plan_rebuild(
            snapshot_run_id, target, missing_limit=missing_limit, persist=True
        )
        if not estimate.runnable:
            raise RebuildRefusedError(
                estimate.refusal or RebuildRefusalCode.MISSING_LOCAL_INPUT, estimate
            )
        if estimate.estimated_peak_memory_bytes > target.max_memory_bytes:
            raise RebuildRefusedError(RebuildRefusalCode.MEMORY_BOUND, estimate)
        if estimate.estimated_temporary_bytes > target.max_temporary_bytes:
            raise RebuildRefusedError(RebuildRefusalCode.TEMP_DISK_BOUND, estimate)

        now = self._clock()
        try:
            checkpoint = await self._store.claim_generation(
                estimate.generation_id,
                owner,
                now=now,
                expires_at=now + timedelta(seconds=lease_seconds),
            )
        except (SQLAlchemyError, OSError, RebuildStorageBackendError) as exc:
            raise _StagedStorageError(RebuildStorageStage.CLAIM, exc) from exc
        if checkpoint.state is RebuildState.PUBLISHED:
            return checkpoint
        if checkpoint.state in {RebuildState.FAILED, RebuildState.CANCELED}:
            raise RebuildTerminalError
        try:
            async with self._renewing_lease(
                checkpoint, owner=owner, lease_seconds=lease_seconds
            ) as lease:
                try:
                    return await self._build(
                        checkpoint,
                        estimate,
                        target,
                        heartbeat=lease,
                        owner=owner,
                        snapshot_run_id=snapshot_run_id,
                        missing_limit=missing_limit,
                        cancel=cancel,
                        lease_seconds=lease_seconds,
                    )
                except asyncio.CancelledError:
                    # The caller's own cancellation is checked first, because it is the one the
                    # caller asked for: a run canceled while its heartbeat happened to fail is
                    # still a cancellation, and reporting it as a lost lease would send somebody
                    # looking for a competing worker that never existed.
                    if cancel is not None and cancel.is_set():
                        raise
                    if not lease.lost:
                        raise
                    # The cancellation was the heartbeat's own request and has now been turned
                    # into a result, so the count that records it is cleared. Left standing it
                    # would tell an enclosing `asyncio.timeout` or `TaskGroup` that this task is
                    # still unwinding a cancellation nobody outside ever asked for — which is
                    # how a lost lease would come back reported as somebody else's timeout.
                    current = asyncio.current_task()
                    if current is not None:
                        current.uncancel()
                    if lease.storage_error is not None:
                        raise _StagedStorageError(
                            RebuildStorageStage.LEASE_RENEWAL, lease.storage_error
                        ) from lease.storage_error
                    raise RebuildLeaseConflictError(
                        "the generation lease could not be renewed while the rebuild ran"
                    ) from None
        except RebuildStorageOperationError as exc:
            error = exc.__cause__ if isinstance(exc.__cause__, BaseException) else exc
            diagnostic = self._storage_diagnostic(
                checkpoint,
                stage=exc.stage,
                error=error,
                correlation_id=uuid4().hex,
                retry_count=0,
            )
            self._emit_storage_diagnostic(diagnostic)
            await self._settle_storage_failure(checkpoint, owner, diagnostic)
            raise RebuildStorageError(diagnostic) from exc
        except _StagedStorageError as exc:
            diagnostic = exc.diagnostic or self._storage_diagnostic(
                checkpoint,
                stage=exc.stage,
                error=exc.error,
                correlation_id=uuid4().hex,
                retry_count=0,
            )
            if exc.diagnostic is None:
                self._emit_storage_diagnostic(diagnostic)
            await self._settle_storage_failure(checkpoint, owner, diagnostic)
            raise RebuildStorageError(diagnostic) from exc.error
        except (SQLAlchemyError, OSError, RebuildStorageBackendError) as exc:
            # Every refusal the build anticipates settles the generation itself. This catches
            # the ones nobody anticipated: a driver or filesystem failure from any store call
            # between the claim and publication, which used to unwind past every handler and
            # leave the row `building` with a lease that then quietly expired — work reported
            # as running with no worker, no diagnostic and nothing to resume from but a guess.
            diagnostic = self._storage_diagnostic(
                checkpoint,
                stage=RebuildStorageStage.BUILD_CHECKPOINT,
                error=exc,
                correlation_id=uuid4().hex,
                retry_count=0,
            )
            self._emit_storage_diagnostic(diagnostic)
            await self._settle_storage_failure(checkpoint, owner, diagnostic)
            raise RebuildStorageError(diagnostic) from exc

    async def _settle_storage_failure(
        self,
        checkpoint: RebuildCheckpoint,
        owner: str,
        diagnostic: RebuildStorageDiagnostic,
    ) -> None:
        """Record ``storage_failed`` and give the generation up, best effort, before re-raising.

        Released rather than failed, because a storage error is the one class of failure that
        says nothing about the work: the documents already committed are still correct, and a
        writer that held SQLite for six seconds is not a reason to derive ten thousand of them
        again. The next run claims the same generation and resumes from its checkpoint, while
        status reports it as nobody's — `incomplete` — for as long as that has not happened.

        Best effort is the design rather than a shortcut. Only a takeover's claim moves the
        lease generation, so the claimed checkpoint stays the right key for as long as this
        worker runs. What can also be true is that the store is the thing that broke, or that
        the lease expired and another owner holds the row now — and that owner alone may mutate
        it. Both are settlements this worker is not entitled to make, while the original failure
        is what the caller needs to see. So the attempt is suppressed rather than chained: the
        run reports :class:`RebuildStorageError` either way, and a row this worker could not
        settle is left to lease recovery rather than to a stale owner.
        """
        await self._record_storage_diagnostic(checkpoint, owner, diagnostic)
        try:
            await self._store.release_generation(
                checkpoint.generation_id,
                RebuildRefusalCode.STORAGE_FAILED,
                owner=owner,
                lease_generation=checkpoint.lease_generation,
                now=self._clock(),
            )
        except (SQLAlchemyError, OSError, RebuildStorageBackendError) as exc:
            settlement = self._storage_diagnostic(
                checkpoint,
                stage=RebuildStorageStage.RELEASE,
                error=exc,
                correlation_id=diagnostic.correlation_id,
                retry_count=diagnostic.retry_count,
                event="settlement",
            )
            self._emit_storage_diagnostic(settlement)
        except (RebuildLeaseConflictError, KeyError) as exc:
            # A successor alone owns this row.  The primary record survives in local evidence;
            # deliberately do not turn a lost settlement right into a stale write.
            settlement = self._storage_diagnostic(
                checkpoint,
                stage=RebuildStorageStage.RELEASE,
                error=exc,
                correlation_id=diagnostic.correlation_id,
                retry_count=diagnostic.retry_count,
                event="settlement",
            )
            self._emit_storage_diagnostic(settlement)
            return

    @asynccontextmanager
    async def _renewing_lease(
        self,
        checkpoint: RebuildCheckpoint,
        *,
        owner: str,
        lease_seconds: int,
    ) -> AsyncGenerator[_LeaseHeartbeat]:
        """Hold this worker's lease for as long as the body runs, however long that is.

        **Build duration is a property of the corpus, and the lease is not.** Two stretches of a
        rebuild are unbounded in aggregate and were covered by nothing. A takeover replays every
        vector its predecessor committed before it may stage anything of its own, in bounded
        pages but an unbounded number of them. And one document's preparation — parse, chunk,
        exact token counts, embedding — sits between the renewal that precedes it and the next
        one, so a single large document is unbounded too. Either could outlast the lease its
        generation was claimed under, and then the first fenced call after the expiry lost a
        lease no competing worker wanted, leaving the generation resumable and unable to advance
        because every retry began the same way.

        **Renewal is on a timer, not on the work.** The loop below also renews per document,
        which is a cadence only as good as documents are uniform. This one renews every
        ``lease_seconds / 3`` no matter what the body is doing — the same cadence the acquisition
        heartbeat uses, and for the same reason it exists there: preparation that never yields
        long enough to renew is preparation that quietly ages out its own ownership.

        **Renewing is not fencing, and this does not weaken it.** ``renew_generation`` moves the
        expiry only for an unchanged owner and lease generation, and every fenced call still
        makes its own assertion. What the heartbeat adds is noticing promptly: a real takeover
        increments the lease generation, the renewal is refused, and the worker is canceled
        there rather than continuing to write into a namespace it no longer owns.
        """
        work = asyncio.current_task()
        if work is None:  # pragma: no cover - every awaited coroutine has an owning task
            msg = "a rebuild lease heartbeat requires an owning task"
            raise RuntimeError(msg)
        state = _LeaseHeartbeat()

        async def heartbeat() -> None:
            while True:
                await asyncio.sleep(lease_seconds / LEASE_RENEWALS_PER_LEASE)
                renewed_at = self._clock()
                try:
                    await self._store.renew_generation(
                        checkpoint.generation_id,
                        owner,
                        checkpoint.lease_generation,
                        now=renewed_at,
                        expires_at=renewed_at + timedelta(seconds=lease_seconds),
                    )
                except (SQLAlchemyError, OSError, RebuildStorageBackendError) as exc:
                    if state.irreversible_publication:
                        # Publication is one fenced atomic store operation. SQLite may hold its
                        # writer lock on the publication connection long enough for this second
                        # connection to report busy, or the renewal may resume after the row is
                        # already terminal. In either case publication itself is the authority:
                        # it must commit or reject its fence, and a heartbeat must not turn a
                        # committed generation into a reported storage failure.
                        return
                    _, retryable = _classify_storage_failure(exc)
                    if retryable:
                        # Three attempts per lease deliberately leave room for one transient
                        # busy/interrupt round. The next fenced store operation still rejects a
                        # lease that truly expired or changed owners.
                        continue
                    # Preserve the private cause for the outer storage boundary.  A real lease
                    # conflict remains distinct: only the latter says another worker won.
                    state.storage_error = exc
                    state.lost = True
                    work.cancel()
                    return
                except Exception:  # noqa: BLE001 - a conflict or plugin failure loses the lease
                    if state.irreversible_publication:
                        # The atomic publication's own fence also decides a real takeover. A
                        # renewal refusal observed concurrently cannot safely overrule its
                        # committed result.
                        return
                    # Broad on purpose, and the one place in this file where that is right. A
                    # refused renewal is a real takeover and arrives as
                    # `RebuildLeaseConflictError`, a `RuntimeError` subclass that a list of
                    # storage exceptions would miss entirely; a storage failure arrives as a
                    # driver error; a plugin store may raise something this file has never
                    # heard of. All three mean the same actionable thing — this worker no
                    # longer holds the generation — and a list that missed one would leave the
                    # heartbeat dead with an unhandled task exception while the superseded
                    # worker kept writing.
                    #
                    # Nothing is carried out of the task, deliberately. It is joined inside a
                    # `finally`, where re-raising would mask whatever the body was already
                    # failing with, and a driver's message is not something to put in front of
                    # somebody on the way to saying the lease is gone.
                    state.lost = True
                    work.cancel()
                    return
                state.renewals += 1

        beat = asyncio.create_task(
            heartbeat(), name=f"rebuild:{checkpoint.generation_id}-lease-heartbeat"
        )
        try:
            yield state
        finally:
            beat.cancel()
            with suppress(asyncio.CancelledError):
                await beat

    async def _build(  # noqa: PLR0912, PLR0915 - explicit refusal/lease stages
        self,
        checkpoint: RebuildCheckpoint,
        estimate: RebuildEstimate,
        target: RebuildTarget,
        *,
        heartbeat: _LeaseHeartbeat,
        owner: str,
        snapshot_run_id: str,
        missing_limit: int,
        cancel: asyncio.Event | None,
        lease_seconds: int,
    ) -> RebuildCheckpoint:
        """Derive every remaining document and publish, under a lease this worker holds."""
        if checkpoint.predecessor_vector_publication_id is not None:
            try:
                await self._store.copy_checkpointed_vectors(
                    checkpoint.generation_id,
                    checkpoint.predecessor_vector_publication_id,
                    owner=owner,
                    lease_generation=checkpoint.lease_generation,
                    now=self._clock(),
                    cancel=cancel,
                    clock=self._clock,
                )
            except RebuildPublicationConflictError as exc:
                await self._fail_build_conflict(checkpoint, owner, exc)
            except RebuildStorageBackendError as exc:
                raise _StagedStorageError(RebuildStorageStage.TAKEOVER_REPLAY_WRITE, exc) from exc
            except RebuildPublicationValidationError as exc:
                await self._fail_build_validation(checkpoint, owner, exc)
            except (SQLAlchemyError, OSError) as exc:
                raise _StagedStorageError(RebuildStorageStage.TAKEOVER_REPLAY_READ, exc) from exc

        while checkpoint.next_sequence < estimate.documents:
            if cancel is not None and cancel.is_set():
                return await self._store.cancel_generation(
                    checkpoint.generation_id,
                    owner=owner,
                    lease_generation=checkpoint.lease_generation,
                    now=self._clock(),
                )
            try:
                inputs = await self._store.snapshot_inputs(
                    checkpoint.generation_id,
                    after_sequence=checkpoint.next_sequence - 1,
                    limit=target.batch_documents,
                )
                expected_sequences = list(
                    range(checkpoint.next_sequence, checkpoint.next_sequence + len(inputs))
                )
            except RebuildPublicationConflictError as exc:
                await self._fail_build_conflict(checkpoint, owner, exc)
            except RebuildPublicationValidationError as exc:
                await self._fail_build_validation(checkpoint, owner, exc)
            if not inputs or [item.sequence for item in inputs] != expected_sequences:
                await self._fail_build_conflict(
                    checkpoint,
                    owner,
                    RebuildPublicationConflictError(RebuildRefusalCode.SNAPSHOT_CHANGED),
                )
            for item in inputs:
                if item.source.byte_length * 6 + 4096 > target.max_memory_bytes:
                    await self._store.fail_generation(
                        checkpoint.generation_id,
                        RebuildRefusalCode.MEMORY_BOUND,
                        owner=owner,
                        lease_generation=checkpoint.lease_generation,
                        now=self._clock(),
                    )
                    raise RebuildRefusedError(RebuildRefusalCode.MEMORY_BOUND, estimate)
                data = await self._blobs.get_bounded(
                    item.blob_ref, max_bytes=target.max_memory_bytes
                )
                if data is None:
                    # Planning checked this. Refusing now prevents a source crawl and prevents
                    # publishing a generation from a manifest that changed under the run.
                    refreshed = await self.dry_run(
                        snapshot_run_id, target, missing_limit=missing_limit
                    )
                    await self._store.fail_generation(
                        checkpoint.generation_id,
                        RebuildRefusalCode.MISSING_LOCAL_INPUT,
                        owner=owner,
                        lease_generation=checkpoint.lease_generation,
                        now=self._clock(),
                    )
                    raise RebuildRefusedError(RebuildRefusalCode.MISSING_LOCAL_INPUT, refreshed)
                try:
                    raw = item.source.raw(data)
                    if item.connector:
                        prepared = await self._deriver.prepare(
                            raw,
                            target,
                            generation_id=checkpoint.generation_id,
                            blob_ref=item.blob_ref,
                            title=item.title,
                            version_token=item.version_token,
                            connector=item.connector,
                        )
                    else:
                        # Preserve the protocol's pre-workspace shape for callers that supply
                        # synthetic or legacy single-snapshot inputs without a connector.
                        prepared = await self._deriver.prepare(
                            raw,
                            target,
                            generation_id=checkpoint.generation_id,
                            blob_ref=item.blob_ref,
                            title=item.title,
                            version_token=item.version_token,
                        )
                except ValueError as exc:
                    try:
                        code = RebuildRefusalCode(str(exc))
                    except ValueError:
                        code = RebuildRefusalCode.DERIVATION_FAILED
                    await self._store.fail_generation(
                        checkpoint.generation_id,
                        code,
                        owner=owner,
                        lease_generation=checkpoint.lease_generation,
                        now=self._clock(),
                    )
                    raise RebuildRefusedError(code, estimate) from exc
                except ManiculeError as exc:
                    await self._store.fail_generation(
                        checkpoint.generation_id,
                        RebuildRefusalCode.DERIVATION_FAILED,
                        owner=owner,
                        lease_generation=checkpoint.lease_generation,
                        now=self._clock(),
                    )
                    raise RebuildDerivationError from exc
                try:
                    prepared.replacement.validate_identity()
                except ValueError as exc:
                    await self._store.fail_generation(
                        checkpoint.generation_id,
                        RebuildRefusalCode.INVALID_REPLACEMENT,
                        owner=owner,
                        lease_generation=checkpoint.lease_generation,
                        now=self._clock(),
                    )
                    raise RebuildValidationError from exc
                if prepared.memory_bytes > target.max_memory_bytes:
                    await self._store.fail_generation(
                        checkpoint.generation_id,
                        RebuildRefusalCode.MEMORY_BOUND,
                        owner=owner,
                        lease_generation=checkpoint.lease_generation,
                        now=self._clock(),
                    )
                    raise RebuildRefusedError(RebuildRefusalCode.MEMORY_BOUND, estimate)
                try:
                    await self._store.check_capacity(
                        checkpoint.generation_id, [prepared.replacement]
                    )
                except RuntimeError as exc:
                    await self._store.fail_generation(
                        checkpoint.generation_id,
                        RebuildRefusalCode.TEMP_DISK_BOUND,
                        owner=owner,
                        lease_generation=checkpoint.lease_generation,
                        now=self._clock(),
                    )
                    raise RebuildRefusedError(RebuildRefusalCode.TEMP_DISK_BOUND, estimate) from exc
                checkpoint = await self._renew_generation(
                    checkpoint, owner, lease_seconds=lease_seconds
                )
                await self._assert_generation_lease(checkpoint, owner)
                await self._deriver.stage(prepared, publication_id=checkpoint.vector_publication_id)
                await self._assert_generation_lease(checkpoint, owner)
                try:
                    checkpoint = await self._store.stage_replacements(
                        checkpoint.generation_id,
                        [(item.sequence, prepared.replacement)],
                        expected_next_sequence=checkpoint.next_sequence,
                        owner=owner,
                        lease_generation=checkpoint.lease_generation,
                        now=self._clock(),
                    )
                except RebuildPublicationConflictError as exc:
                    await self._fail_build_conflict(checkpoint, owner, exc)
                except RebuildPublicationValidationError as exc:
                    await self._fail_build_validation(checkpoint, owner, exc)
                checkpoint = await self._renew_generation(
                    checkpoint, owner, lease_seconds=lease_seconds
                )

        if cancel is not None and cancel.is_set():
            return await self._store.cancel_generation(
                checkpoint.generation_id,
                owner=owner,
                lease_generation=checkpoint.lease_generation,
                now=self._clock(),
            )
        try:
            checkpoint = await self._store.begin_validation(
                checkpoint.generation_id,
                owner=owner,
                lease_generation=checkpoint.lease_generation,
                now=self._clock(),
            )
        except RebuildPublicationConflictError as exc:
            await self._fail_build_conflict(checkpoint, owner, exc)
        except RebuildPublicationValidationError as exc:
            await self._fail_build_validation(checkpoint, owner, exc)
        if cancel is not None and cancel.is_set():
            return await self._store.cancel_generation(
                checkpoint.generation_id,
                owner=owner,
                lease_generation=checkpoint.lease_generation,
                now=self._clock(),
            )
        try:
            await self._assert_generation_lease(checkpoint, owner)
        except RebuildPublicationConflictError as exc:
            await self._fail_build_conflict(checkpoint, owner, exc)
        try:
            await self._retry_storage_operation(
                checkpoint,
                owner,
                stage=RebuildStorageStage.VALIDATION_EVIDENCE_READ,
                operation=lambda: self._store.validate_generation(
                    checkpoint.generation_id,
                    owner=owner,
                    lease_generation=checkpoint.lease_generation,
                    now=self._clock(),
                    cancel=cancel,
                    clock=self._clock,
                ),
                cancel=cancel,
            )
        except RebuildPublicationConflictError as exc:
            await self._fail_build_conflict(checkpoint, owner, exc)
        except RebuildPublicationValidationError as exc:
            await self._store.fail_generation(
                checkpoint.generation_id,
                exc.code,
                owner=owner,
                lease_generation=checkpoint.lease_generation,
                now=self._clock(),
            )
            raise RebuildValidationError from exc
        except (RebuildStorageOperationError, _StagedStorageError):
            raise
        except (RuntimeError, ValueError) as exc:
            await self._store.fail_generation(
                checkpoint.generation_id,
                RebuildRefusalCode.INVALID_REPLACEMENT,
                owner=owner,
                lease_generation=checkpoint.lease_generation,
                now=self._clock(),
            )
            raise RebuildValidationError from exc
        try:
            await self._assert_generation_lease(checkpoint, owner)
        except RebuildPublicationConflictError as exc:
            await self._fail_build_conflict(checkpoint, owner, exc)
        if cancel is not None and cancel.is_set():
            return await self._store.cancel_generation(
                checkpoint.generation_id,
                owner=owner,
                lease_generation=checkpoint.lease_generation,
                now=self._clock(),
            )
        heartbeat.irreversible_publication = True
        publish = asyncio.create_task(
            self._store.publish_generation(
                checkpoint.generation_id,
                owner=owner,
                lease_generation=checkpoint.lease_generation,
                now=self._clock(),
            )
        )
        try:
            return await self._join_irreversible(publish)
        except RebuildPublicationConflictError as exc:
            await self._store.fail_generation(
                checkpoint.generation_id,
                exc.code,
                owner=owner,
                lease_generation=checkpoint.lease_generation,
                now=self._clock(),
            )
            raise RebuildLeaseError from exc
        except RebuildLeaseConflictError:
            # The publication transaction can discover that this worker no longer owns the
            # lease. The new owner alone may mutate the durable generation from here.
            raise
        except RebuildStorageBackendError as exc:
            raise _StagedStorageError(RebuildStorageStage.PUBLICATION, exc) from exc
        except RebuildPublicationValidationError as exc:
            await self._store.fail_generation(
                checkpoint.generation_id,
                exc.code,
                owner=owner,
                lease_generation=checkpoint.lease_generation,
                now=self._clock(),
            )
            raise RebuildValidationError from exc
        except (RebuildStorageOperationError, _StagedStorageError):
            raise
        except (SQLAlchemyError, OSError) as exc:
            raise _StagedStorageError(RebuildStorageStage.PUBLICATION, exc) from exc
        except (RuntimeError, ValueError) as exc:
            # The store's publication protocol reports bounded invariant failures as its typed
            # validation error. RuntimeError remains a compatibility boundary for third-party
            # stores implementing the protocol: its arbitrary text stays chained locally and
            # never crosses an unattended surface.
            await self._store.fail_generation(
                checkpoint.generation_id,
                RebuildRefusalCode.INVALID_REPLACEMENT,
                owner=owner,
                lease_generation=checkpoint.lease_generation,
                now=self._clock(),
            )
            raise RebuildValidationError from exc
        finally:
            heartbeat.irreversible_publication = False


def build_offline_rebuilder(
    *,
    store: RebuildStore,
    blobs: RebuildBlobSource,
    workspace_id: str,
    source: str,
    parser_chain: ParserChain,
    routing_identity: str,
    chunker: Chunker,
    embedder: Embedder,
    vectors: VectorStore,
    chunk_fingerprint: ChunkFingerprint,
    middleware: MiddlewareRunner,
    parse_runner: BoundedParseRunner,
    detect_glossary: bool = True,
    clock: Callable[[], datetime] | None = None,
) -> OfflineGenerationRebuilder:
    """Construct the production offline stack without a connector capability."""
    relational = ParserChunkerRelationalDeriver(
        workspace_id=workspace_id,
        source=source,
        parser_chain=parser_chain,
        routing_identity=routing_identity,
        chunker=chunker,
        middleware=middleware,
        parse_runner=parse_runner,
        detect_glossary=detect_glossary,
    )
    return OfflineGenerationRebuilder(
        store=store,
        blobs=blobs,
        deriver=EmbeddingOfflineDeriver(
            relational=relational,
            embedder=embedder,
            vectors=vectors,
            chunk_fingerprint=chunk_fingerprint,
        ),
        clock=clock,
    )


__all__ = [
    "EmbeddingOfflineDeriver",
    "OfflineDeriver",
    "OfflineGenerationRebuilder",
    "ParserChunkerRelationalDeriver",
    "PreparedReplacement",
    "RebuildBlobSource",
    "RebuildStore",
    "RelationalDeriver",
    "build_offline_rebuilder",
]
