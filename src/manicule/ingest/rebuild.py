"""Offline construction and atomic publication of replacement derived generations.

The runner has no connector parameter and no connector protocol in its import graph. Its only
source is a promoted manifest plus the blob reader, making a network fallback structurally
impossible rather than a convention a caller may accidentally bypass.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol, cast, override, runtime_checkable
from uuid import uuid4

from manicule.core.content import (
    Chunk,
    Document,
    DocumentStatus,
    ParsedBlock,
    PipelineStage,
    RawDocument,
)
from manicule.core.glossary import GlossaryEntry
from manicule.core.ids import content_hash, document_id
from manicule.core.rebuild import (
    DerivedReplacement,
    RebuildCheckpoint,
    RebuildEstimate,
    RebuildRefusalCode,
    RebuildRefusedError,
    RebuildState,
    RebuildTarget,
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

    async def validate_generation(self, generation_id: str) -> None: ...

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


@runtime_checkable
class RebuildBlobSource(Protocol):
    """Read-only access to retained bytes; deliberately has no retain or fetch method."""

    async def get_bounded(self, digest: str, *, max_bytes: int) -> bytes | None: ...


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
    ) -> DerivedReplacement:
        root, members = await self._derive_one(
            raw,
            target,
            generation_id=generation_id,
            blob_ref=blob_ref,
            title=title,
            version_token=version_token,
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
                        id=document_id(self._workspace_id, self._source, member.source_id),
                        publication_id=generation_id,
                        source=self._source,
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
        identifier = document_id(self._workspace_id, self._source, raw.source_id)
        document = Document(
            id=identifier,
            publication_id=generation_id,
            source=self._source,
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
    """Bounded, resumable executor over one promoted source snapshot."""

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

    async def dry_run(
        self, snapshot_run_id: str, target: RebuildTarget, *, missing_limit: int = 100
    ) -> RebuildEstimate:
        if missing_limit <= 0 or missing_limit > MAX_MISSING_DETAILS:
            raise ValueError("missing_limit must be between 1 and 1000")
        return await self._store.plan_rebuild(
            snapshot_run_id, target, missing_limit=missing_limit, persist=False
        )

    async def run(  # noqa: PLR0912, PLR0915 - explicit terminal/refusal/lease stages
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
        checkpoint = await self._store.claim_generation(
            estimate.generation_id,
            owner,
            now=now,
            expires_at=now + timedelta(seconds=lease_seconds),
        )
        if checkpoint.state is RebuildState.PUBLISHED:
            return checkpoint
        if checkpoint.state in {RebuildState.FAILED, RebuildState.CANCELED}:
            raise RuntimeError(f"generation is terminal: {checkpoint.state.value}")
        if checkpoint.predecessor_vector_publication_id is not None:
            await self._store.copy_checkpointed_vectors(
                checkpoint.generation_id,
                checkpoint.predecessor_vector_publication_id,
                owner=owner,
                lease_generation=checkpoint.lease_generation,
                now=self._clock(),
                cancel=cancel,
            )

        while checkpoint.next_sequence < estimate.documents:
            if cancel is not None and cancel.is_set():
                return await self._store.cancel_generation(
                    checkpoint.generation_id,
                    owner=owner,
                    lease_generation=checkpoint.lease_generation,
                    now=self._clock(),
                )
            inputs = await self._store.snapshot_inputs(
                checkpoint.generation_id,
                after_sequence=checkpoint.next_sequence - 1,
                limit=target.batch_documents,
            )
            if not inputs:
                raise RuntimeError("promoted snapshot changed or has a non-contiguous manifest")
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
                prepared.replacement.validate_identity()
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
                renewed_at = self._clock()
                checkpoint = await self._store.renew_generation(
                    checkpoint.generation_id,
                    owner,
                    checkpoint.lease_generation,
                    now=renewed_at,
                    expires_at=renewed_at + timedelta(seconds=lease_seconds),
                )
                await self._store.assert_generation_lease(
                    checkpoint.generation_id,
                    owner,
                    checkpoint.lease_generation,
                    now=self._clock(),
                )
                await self._deriver.stage(prepared, publication_id=checkpoint.vector_publication_id)
                await self._store.assert_generation_lease(
                    checkpoint.generation_id,
                    owner,
                    checkpoint.lease_generation,
                    now=self._clock(),
                )
                checkpoint = await self._store.stage_replacements(
                    checkpoint.generation_id,
                    [(item.sequence, prepared.replacement)],
                    expected_next_sequence=checkpoint.next_sequence,
                    owner=owner,
                    lease_generation=checkpoint.lease_generation,
                    now=self._clock(),
                )
                renewed_at = self._clock()
                checkpoint = await self._store.renew_generation(
                    checkpoint.generation_id,
                    owner,
                    checkpoint.lease_generation,
                    now=renewed_at,
                    expires_at=renewed_at + timedelta(seconds=lease_seconds),
                )

        if cancel is not None and cancel.is_set():
            return await self._store.cancel_generation(
                checkpoint.generation_id,
                owner=owner,
                lease_generation=checkpoint.lease_generation,
                now=self._clock(),
            )
        checkpoint = await self._store.begin_validation(
            checkpoint.generation_id,
            owner=owner,
            lease_generation=checkpoint.lease_generation,
            now=self._clock(),
        )
        if cancel is not None and cancel.is_set():
            return await self._store.cancel_generation(
                checkpoint.generation_id,
                owner=owner,
                lease_generation=checkpoint.lease_generation,
                now=self._clock(),
            )
        await self._store.assert_generation_lease(
            checkpoint.generation_id,
            owner,
            checkpoint.lease_generation,
            now=self._clock(),
        )
        await self._store.validate_generation(checkpoint.generation_id)
        await self._store.assert_generation_lease(
            checkpoint.generation_id,
            owner,
            checkpoint.lease_generation,
            now=self._clock(),
        )
        if cancel is not None and cancel.is_set():
            return await self._store.cancel_generation(
                checkpoint.generation_id,
                owner=owner,
                lease_generation=checkpoint.lease_generation,
                now=self._clock(),
            )
        publish = asyncio.create_task(
            self._store.publish_generation(
                checkpoint.generation_id,
                owner=owner,
                lease_generation=checkpoint.lease_generation,
                now=self._clock(),
            )
        )
        return await self._join_irreversible(publish)


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
