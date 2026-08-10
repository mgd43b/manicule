"""discover → fetch → parse → chunk → embed → store, one document at a time.

**The unit of work is one document. A batch is a scheduling artefact with no semantics of its
own.** That is what makes "one bad document never aborts a batch" a structural property rather
than a promise: there is no batch-level transaction to abort, and no batch-level state a
document can corrupt. Every failure this module catches is attributed to a document, recorded,
and left behind.

Two rules govern what gets written, and both exist because of failures that are otherwise
invisible.

**A failed re-ingest must not demote a working document.** If a document is ``indexed`` and a
re-ingest fails at any stage, it stays ``indexed`` with its existing chunks and vectors, and
the failure is recorded in metadata. Setting ``pending`` before parsing and ``failed``
afterwards — the obvious shape — means a transient network error during a routine re-sync
silently removes a working document from the index, while its chunks and vectors sit in both
stores, intact and unreachable.

**A terminal determination about new bytes does replace it.** The exception, and it is not a
softening of the rule: ``no_extractable_text``, ``unsupported_media_type`` and ``container``
are conclusions about content that genuinely changed, not failures to reach one. Continuing to
serve chunks derived from bytes the source no longer has would cite text the document does not
contain, which is the one thing this project will not do. ``failed`` is the case where we do
not know, and not knowing is the case that must not destroy a working answer.

The write order and its crash windows belong to ``docs/storage.md`` §8.2 and are honoured
rather than restated: chunks, then vectors, then ``indexed`` last, in the transaction that is
the commit point.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from manicule.core.content import (
    SETTLED,
    Document,
    DocumentStatus,
    PipelineStage,
    RawDocument,
    Retention,
)
from manicule.core.errors import (
    ChunkingError,
    ContextOverflowError,
    MiddlewareViolationError,
)
from manicule.core.ids import content_hash, document_id
from manicule.ingest.embedding import embed_chunks
from manicule.ingest.ports import SupportsWatermark
from manicule.ingest.workers import AttemptResult
from manicule.parsers.chain import Attempt, ChainResult, Outcome, container_result, run_chain
from manicule.parsers.expansion import ExpandedMember, MemberFailure

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from manicule.core.content import Chunk, Metadata
    from manicule.core.fingerprints import ChunkFingerprint
    from manicule.core.protocols import Chunker, Connector, Embedder, VectorStore
    from manicule.core.sources import DiscoveredDoc, DocRef
    from manicule.ingest.middleware import MiddlewareRunner
    from manicule.ingest.ports import IngestStore
    from manicule.ingest.workers import ParseRunner
    from manicule.parsers.expansion import MemberOutcome


class _StageError(Exception):
    """One stage's failure, carrying where it happened.

    Internal, and never seen outside this module: it exists so that every stage records its
    failure through one path, which is the property that keeps "one bad document never aborts
    a batch" true as stages are added.
    """

    def __init__(self, stage: PipelineStage, detail: str) -> None:
        super().__init__(detail)
        self.stage = stage
        self.detail = detail


@runtime_checkable
class BlobSink(Protocol):
    """What the pipeline needs of a blob store, and no more.

    Two methods, because ingest writes bytes and re-parse reads them back. Stated as a
    protocol so that the pipeline imports no storage: ``BlobStore`` satisfies it structurally,
    and ``tests/test_import_boundary.py`` fails the build if that stops being true.
    """

    async def retain(self, data: bytes, media_type: str | None = None) -> Retention: ...

    async def get(self, digest: str) -> bytes | None: ...


class NoRetention:
    """The blob sink that keeps nothing, and says so.

    Retention is configurable off, and "off" must be a real, exercised path rather than a
    branch nobody runs: a document ingests identically with and without retained bytes, and
    only its repair options differ. Naming the reason rather than leaving a silent ``NULL`` is
    the same rule as :class:`~manicule.core.anchors.Unlocated` — absent with a stated reason,
    visible in diagnostics, never a silent partial success.
    """

    async def retain(self, data: bytes, media_type: str | None = None) -> Retention:
        del data, media_type
        return Retention(omitted_reason="original bytes not retained: retention is disabled")

    async def get(self, digest: str) -> bytes | None:
        del digest
        return None


@dataclass(frozen=True, slots=True)
class DocumentOutcome:
    """What became of one document, and enough to count it."""

    source_id: str
    status: DocumentStatus
    document_id: str = ""
    detail: str = ""
    chunks: int = 0
    skipped: str = ""
    """``version`` or ``hash`` when change detection stopped early, otherwise empty."""

    members: tuple[str, ...] = ()
    """Source ids of documents found inside this one, queued rather than recursed into."""


@dataclass
class RunReport:
    """Counters for one run, by outcome.

    No ``runs`` table. Run history is diagnostic, not relational, and a table that only ever
    grows needs a retention policy nobody has asked for. The last run's counters live on the
    connector row, where they are overwritten rather than accumulated — which is the correct
    retention policy for a diagnostic.
    """

    connector: str = ""
    discovered: int = 0
    """Documents the connector reported. Members found inside them are counted separately."""

    expanded: int = 0
    """Documents found *inside* others. Kept apart from ``discovered`` because they are not
    what the source enumerated — and because ``discovered`` is what a ``--limit`` bounds, and
    one archive of five hundred members must not consume a limit of ten."""

    skipped_version: int = 0
    skipped_hash: int = 0
    by_status: dict[str, int] = field(default_factory=dict[str, int])
    error: str = ""

    @property
    def indexed(self) -> int:
        return self.by_status.get(DocumentStatus.INDEXED.value, 0)

    @property
    def clean(self) -> bool:
        """Whether the run finished. Only a clean run advances a watermark."""
        return not self.error

    def record(self, outcome: DocumentOutcome, *, expanded: bool = False) -> None:
        if expanded:
            self.expanded += 1
        else:
            self.discovered += 1
        if outcome.skipped == "version":
            self.skipped_version += 1
            return
        if outcome.skipped == "hash":
            self.skipped_hash += 1
            return
        key = outcome.status.value
        self.by_status[key] = self.by_status.get(key, 0) + 1

    def as_metadata(self) -> Metadata:
        return {
            "last_run": {
                "discovered": self.discovered,
                "expanded": self.expanded,
                "skipped_version": self.skipped_version,
                "skipped_hash": self.skipped_hash,
                "by_status": dict(self.by_status),
                "error": self.error,
            }
        }


class IngestPipeline:
    """Runs one connector's documents through every stage, and never lets one stop the rest."""

    def __init__(
        self,
        *,
        store: IngestStore,
        chunker: Chunker,
        embedder: Embedder,
        vectors: VectorStore,
        runner: ParseRunner,
        resolve_chain: Callable[[str], Sequence[str]],
        middleware: MiddlewareRunner,
        chunk_fingerprint: ChunkFingerprint,
        workspace: str = "default",
        blobs: BlobSink | None = None,
        fetch_concurrency: int = 8,
        max_fetch_bytes: int = 256 * 1024 * 1024,
        target_batch_tokens: int = 16_384,
        max_embed_batch: int = 64,
    ) -> None:
        self._store = store
        self._chunker = chunker
        self._embedder = embedder
        self._vectors = vectors
        self._runner = runner
        self._resolve_chain = resolve_chain
        self._middleware = middleware
        self._chunk_fingerprint = chunk_fingerprint
        self._workspace = workspace
        self._blobs = blobs or NoRetention()
        self._max_fetch_bytes = max_fetch_bytes
        self._target_batch_tokens = target_batch_tokens
        self._max_embed_batch = max_embed_batch
        self._fetching = asyncio.Semaphore(max(1, fetch_concurrency))
        self._embedding = asyncio.Lock()

    # --- a run ---------------------------------------------------------------------------

    async def run(self, connector: Connector, *, limit: int | None = None) -> RunReport:
        """Ingest everything a connector reports as changed since its watermark.

        The watermark advances only on a clean run, which is the whole of resumability: an
        interrupted sync re-enumerates from the last good point, change detection makes
        re-enumeration cheap, and the recovery sweep requeues anything caught in flight. There
        is no checkpoint file, no resume token, and nothing to corrupt. Resume is: run it
        again.
        """
        report = RunReport(connector=connector.name)
        watermark = await self._store.get_watermark(connector.name)
        stream = connector.discover(watermark)
        try:
            async for discovered in stream:
                for position, outcome in enumerate(await self.ingest(connector, discovered)):
                    # The first outcome is the discovered document; anything after it came out
                    # of the inside of it.
                    report.record(outcome, expanded=position > 0)
                if limit is not None and report.discovered >= limit:
                    break
        except Exception as exc:  # noqa: BLE001 - an enumeration failure is not a crash
            report.error = f"{type(exc).__name__}: {exc}"
        finally:
            closer = getattr(stream, "aclose", None)
            if closer is not None:
                await closer()
        if report.clean:
            await self._advance_watermark(connector)
        await self._store.record_connector_metadata(connector.name, report.as_metadata())
        return report

    async def _advance_watermark(self, connector: Connector) -> None:
        """Record how far a clean run got, if the connector can say.

        **Only on a clean run**, and that is the whole of resumability. An interrupted sync
        re-enumerates from the last good point; change detection (§4) makes re-enumeration
        cheap, because already-ingested documents skip at level 1 without a fetch; and the
        recovery sweep requeues anything caught in flight. So resume is: run it again. There is
        no checkpoint file, no resume token, and nothing to corrupt.

        The position is asked for rather than required, on the same principle as the lifecycle
        hooks in :mod:`manicule.core.lifecycle`: a connector that has one implements
        :class:`~manicule.ingest.ports.SupportsWatermark` and a connector that does not writes
        nothing. Forcing it onto the ``Connector`` protocol would make every source invent a
        position, and a source with no change signal inventing one is worse than a source that
        re-enumerates — the invented one is believed.
        """
        if not isinstance(connector, SupportsWatermark):
            return
        reached = connector.watermark()
        if reached is not None:
            await self._store.set_watermark(connector.name, reached)

    async def ingest(
        self, connector: Connector, discovered: DiscoveredDoc
    ) -> list[DocumentOutcome]:
        """One discovered document, from the change check to the commit.

        Returns a list because a container is several documents: the archive itself, then
        every member it expanded into. Never raises for anything a document did. A connector
        that cannot fetch it, a parser that hangs, a hook that misbehaves and a model that
        refuses are all recorded against that document and returned.
        """
        source = connector.name
        source_id = discovered.source_id
        existing = await self._store.find_document(source, source_id)

        if self._unchanged_by_token(existing, discovered):
            await self._store.record_seen(existing.id)  # pyright: ignore[reportOptionalMemberAccess]
            return [
                DocumentOutcome(
                    source_id=source_id,
                    status=existing.status,  # pyright: ignore[reportOptionalMemberAccess]
                    document_id=existing.id,  # pyright: ignore[reportOptionalMemberAccess]
                    skipped="version",
                )
            ]

        await self._advance(existing, DocumentStatus.FETCHING)
        try:
            raw = await self._fetch(connector, discovered.ref)
        except Exception as exc:  # noqa: BLE001 - one source's failure is one document's
            return [
                await self._fail(
                    existing, source, source_id, PipelineStage.FETCH, f"{type(exc).__name__}: {exc}"
                )
            ]

        return await self.ingest_raw(
            raw,
            source=source,
            version_token=discovered.version_token,
            title=discovered.title,
            existing=existing,
        )

    async def ingest_raw(
        self,
        raw: RawDocument,
        *,
        source: str,
        version_token: str | None = None,
        title: str = "",
        existing: Document | None = None,
        force: bool = False,
    ) -> list[DocumentOutcome]:
        """Everything from fetched bytes onwards, including anything found inside.

        The path shared by a connector sync, a member of a container, and
        ``reindex --re-parse`` reading retained bytes. One implementation, so a re-parse cannot
        drift into producing something a first ingest would not have produced.

        **Members are queued, never recursed into.** A parser expands one level and this drains
        the queue breadth-first, so a wide archive cannot starve a batch by descending into one
        branch — and a nested container's members join the back of the same queue rather than
        jumping it.

        ``force`` skips change detection for the *top-level* document only. Re-parse exists
        precisely to run a new parser over unchanged bytes, so the hash check would refuse the
        one operation being asked for. Members are not forced: a container whose bytes are
        unchanged still expands to the same members, and re-parsing those is a decision about
        each of them rather than a consequence of touching the archive.
        """
        outcome, members = await self._ingest_one(
            raw,
            source=source,
            version_token=version_token,
            title=title,
            existing=existing,
            force=force,
        )
        outcomes = [outcome]
        queue: list[MemberOutcome] = list(members)
        while queue:
            member = queue.pop(0)
            if isinstance(member, MemberFailure):
                outcomes.append(await self._record_member_failure(member, source))
                continue
            inner, deeper = await self._ingest_one(
                _member_raw(member), source=source, title=_member_title(member)
            )
            outcomes.append(inner)
            queue.extend(deeper)
        return outcomes

    async def _ingest_one(
        self,
        raw: RawDocument,
        *,
        source: str,
        version_token: str | None = None,
        title: str = "",
        existing: Document | None = None,
        force: bool = False,
    ) -> tuple[DocumentOutcome, tuple[MemberOutcome, ...]]:
        """One document, and whatever it turned out to contain."""
        source_bytes = raw.as_bytes()
        digest = content_hash(source_bytes)
        if existing is None:
            existing = await self._store.find_document(source, raw.source_id)

        if not force and self._unchanged_by_hash(existing, digest):
            await self._store.record_seen(existing.id, version_token=version_token)  # pyright: ignore[reportOptionalMemberAccess]
            return (
                DocumentOutcome(
                    source_id=raw.source_id,
                    status=existing.status,  # pyright: ignore[reportOptionalMemberAccess]
                    document_id=existing.id,  # pyright: ignore[reportOptionalMemberAccess]
                    skipped="hash",
                ),
                (),
            )

        identifier = (
            existing.id if existing else document_id(self._workspace, source, raw.source_id)
        )

        # Retention happens **before** any hook runs, and that ordering is load-bearing twice
        # over. `documents.content_hash` is the hash of what the connector returned
        # (`storage.md` §4.2), so retaining post-hook bytes would leave the reference and the
        # hash describing different content. And re-parse feeds retained bytes back through
        # this same path, hooks included — so retaining the transformed bytes would apply
        # `before_parse` twice, and a hook that is not idempotent would compound on every
        # repair. What is kept is the original, exactly as fetched.
        retention = await self._retain(raw, source_bytes)

        try:
            transformed = await self._middleware.before_parse(raw)
        except Exception as exc:  # noqa: BLE001 - a hook's failure is this document's
            failed = await self._fail(
                existing,
                source,
                raw.source_id,
                PipelineStage.MIDDLEWARE,
                f"before_parse: {type(exc).__name__}: {exc}",
                raw=raw,
                digest=digest,
                version_token=version_token,
                title=title,
            )
            return failed, ()
        if transformed is None:
            skipped = await self._settle(
                ChainResult(
                    blocks=[],
                    status=DocumentStatus.SKIPPED,
                    status_detail="a middleware hook excluded this document before parsing",
                ),
                raw=raw,
                source=source,
                digest=digest,
                version_token=version_token,
                title=title,
                identifier=identifier,
                existing=existing,
            )
            return skipped, ()
        raw = transformed

        await self._advance(existing, DocumentStatus.PARSING)

        result, members = await self._parse(raw)
        if members:
            result = container_result(len(members))

        document = await self._store_record(
            result,
            raw=raw,
            source=source,
            digest=digest,
            version_token=version_token,
            title=title,
            identifier=identifier,
            existing=existing,
            retention=retention,
        )
        if result.status is not DocumentStatus.PARSED:
            await self._observe(document)
            return (
                DocumentOutcome(
                    source_id=raw.source_id,
                    status=document.status,
                    document_id=document.id,
                    detail=result.status_detail,
                    members=tuple(member.source_id for member in members),
                ),
                members,
            )

        return await self._finish(result, document, raw=raw, existing=existing), ()

    async def _record_member_failure(self, member: MemberFailure, source: str) -> DocumentOutcome:
        """Store a member that could not become a document, with the reason it could not.

        Dropping it instead would make an archive's member set depend silently on what the
        parser felt able to read, and "the archive had 200 files and we indexed 197" is not a
        fact anybody would ever discover.
        """
        placeholder = RawDocument(
            source_id=member.source_id,
            uri=member.uri,
            media_type="application/octet-stream",
            content=b"",
            metadata=dict(member.metadata),
        )
        return await self._settle(
            ChainResult(
                blocks=[],
                status=member.status,
                status_detail=member.reason,
                failed_stage=(
                    PipelineStage.PARSE if member.status is DocumentStatus.FAILED else None
                ),
            ),
            raw=placeholder,
            source=source,
            digest=content_hash(member.uri),
            version_token=None,
            title="",
            identifier=document_id(self._workspace, source, member.source_id),
            existing=await self._store.find_document(source, member.source_id),
        )

    # --- stages --------------------------------------------------------------------------

    async def _fetch(self, connector: Connector, ref: DocRef) -> RawDocument:
        async with self._fetching:
            raw = await connector.fetch(ref)
        size = len(raw.as_bytes())
        if size > self._max_fetch_bytes:
            msg = (
                f"{ref.uri} is {size} bytes, above the {self._max_fetch_bytes}-byte fetch cap. "
                f"Raise ingest.max_fetch_bytes to index it, or exclude it at the connector."
            )
            raise ValueError(msg)
        return raw

    async def _parse(self, raw: RawDocument) -> tuple[ChainResult, tuple[MemberOutcome, ...]]:
        """Run the resolved chain, remembering whether the winner was a container.

        The chain is resolved **once, before the first attempt**, and recorded as it proceeds.
        Resolving lazily would let a configuration reload mid-chain produce a chain that never
        existed — and a result nobody could explain months later.
        """
        chain = list(self._resolve_chain(raw.media_type))
        captured: list[AttemptResult] = []

        async def attempt(name: str, document: RawDocument) -> tuple[list[object], Attempt]:
            result = await self._runner.run_attempt(name, document)
            captured.append(result)
            return result.blocks, result.attempt  # pyright: ignore[reportReturnType]

        result = await run_chain(chain, raw, attempt)  # pyright: ignore[reportArgumentType]
        won = captured[-1] if captured else None
        members = won.members if won is not None and won.attempt.outcome is Outcome.PARSED else ()
        return result, tuple(members)  # pyright: ignore[reportReturnType]

    async def _finish(
        self,
        result: ChainResult,
        document: Document,
        *,
        raw: RawDocument,
        existing: Document | None,
    ) -> DocumentOutcome:
        """Chunk, embed and commit a document the chain produced text for.

        Each stage's failure is turned into one :class:`_StageError` and caught once, rather
        than returned from six places. The shape matters more than the line count: with six
        exits it is possible to add a seventh that forgets to record the failure, and a stage
        that fails without recording is a document that quietly stops being re-tried.
        """
        try:
            chunks = await self._prepare(result, document)
            if not chunks:
                return await self._nothing_to_index(result, document, raw=raw)
            await self._advance(existing, DocumentStatus.EMBEDDING)
            async with self._embedding:
                vectors = await embed_chunks(
                    self._embedder,
                    chunks,
                    chunk_fingerprint=self._chunk_fingerprint,
                    target_batch_tokens=self._target_batch_tokens,
                    maximum=self._max_embed_batch,
                )
        except _StageError as failure:
            return await self._demote(document, existing, failure.stage, failure.detail)
        except ContextOverflowError as exc:
            return await self._demote(document, existing, PipelineStage.EMBED, str(exc))
        except Exception as exc:  # noqa: BLE001 - a model failure is this document's
            return await self._demote(
                document, existing, PipelineStage.EMBED, f"{type(exc).__name__}: {exc}"
            )

        return await self._commit(document, chunks, vectors, raw=raw, existing=existing)

    async def _prepare(self, result: ChainResult, document: Document) -> list[Chunk]:
        """Blocks through ``after_parse``, into chunks, through ``after_chunk``.

        Raises:
            _StageError: Naming the stage, so the caller records one thing in one place.
        """
        try:
            blocks = await self._middleware.after_parse(document, result.blocks)
        except MiddlewareViolationError as exc:
            raise _StageError(PipelineStage.MIDDLEWARE, str(exc)) from exc
        except Exception as exc:
            detail = f"after_parse: {type(exc).__name__}: {exc}"
            raise _StageError(PipelineStage.MIDDLEWARE, detail) from exc

        try:
            chunks = self._chunker.chunk(document, blocks)
        except ChunkingError as exc:
            raise _StageError(PipelineStage.CHUNK, str(exc)) from exc

        try:
            return await self._middleware.after_chunk(document, chunks)
        except MiddlewareViolationError as exc:
            raise _StageError(PipelineStage.MIDDLEWARE, str(exc)) from exc
        except Exception as exc:
            detail = f"after_chunk: {type(exc).__name__}: {exc}"
            raise _StageError(PipelineStage.MIDDLEWARE, detail) from exc

    async def _nothing_to_index(
        self, result: ChainResult, document: Document, *, raw: RawDocument
    ) -> DocumentOutcome:
        """A document whose blocks survived parsing and produced no chunk.

        ``no_extractable_text`` rather than ``failed``, because nothing broke: the tooling
        worked and there is nothing to index. Stored rather than dropped, so it is countable,
        skippable on the next sync, and reachable by a re-parse the day that changes.
        """
        settled = ChainResult(
            blocks=[],
            status=DocumentStatus.NO_EXTRACTABLE_TEXT,
            status_detail=(
                "the parser chain produced blocks but chunking produced nothing to index"
            ),
            parser_used=result.parser_used,
            attempts=result.attempts,
        )
        stored = await self._store.upsert_document(_with_status(document, settled))
        await self._store.replace_chunks(stored.id, [])
        await self._observe(stored)
        return DocumentOutcome(
            source_id=raw.source_id,
            status=stored.status,
            document_id=stored.id,
            detail=settled.status_detail,
        )

    async def _commit(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        *,
        raw: RawDocument,
        existing: Document | None,
    ) -> DocumentOutcome:
        """Write in the one order that survives a crash at any point.

        1. Chunks. The document is not ``indexed``, so nothing is served.
        2. Vectors, upserted by chunk id, so re-running step 2 is free.
        3. ``indexed``, last, in the transaction that is the commit point.

        A crash between 1 and 2 leaves chunks with no vectors and nothing served; the repair
        re-embeds those chunk ids. A crash between 2 and 3 leaves vectors for a document that
        is not ``indexed``; the repair re-runs 2 idempotently and then 3.
        """
        try:
            await self._store.replace_chunks(document.id, chunks)
            await self._vectors.upsert(chunks, vectors)
        except Exception as exc:  # noqa: BLE001 - a store failure is this document's
            return await self._demote(
                document, existing, PipelineStage.STORE, f"{type(exc).__name__}: {exc}"
            )

        indexed = await self._store.upsert_document(
            document.model_copy(
                update={
                    "status": DocumentStatus.INDEXED,
                    "status_detail": None,
                    "failed_stage": None,
                }
            )
        )
        await self._store.set_lineage(
            indexed.id,
            chunk_fp=self._chunk_fingerprint.canonical(),
            embed_fp=self._embedder.fingerprint.canonical(),
        )
        await self._observe(indexed)
        return DocumentOutcome(
            source_id=raw.source_id,
            status=DocumentStatus.INDEXED,
            document_id=indexed.id,
            chunks=len(chunks),
        )

    # --- change detection ------------------------------------------------------------------

    @staticmethod
    def _unchanged_by_token(existing: Document | None, discovered: DiscoveredDoc) -> bool:
        """Level 1: the source says nothing changed, and we believe it without fetching.

        The token is opaque and connector-defined — a git blob SHA, a Confluence version
        number, an S3 ETag. It is compared for equality and never interpreted: no ordering, no
        parsing, no "is this newer". A connector wanting ordering implements it in ``discover``.

        Only a **settled** document may skip. One requeued after a crash carries a token and a
        hash from an ingest that never finished, and skipping on those would strand it forever.
        """
        return (
            existing is not None
            and existing.status in SETTLED
            and discovered.version_token is not None
            and existing.version_token == discovered.version_token
        )

    @staticmethod
    def _unchanged_by_hash(existing: Document | None, digest: str) -> bool:
        """Level 2: the bytes are identical, whatever the source claimed.

        Level 1 can lie — a source that touches its modification date on every save reports a
        new token for an unchanged body — and this catches it before the expensive part, which
        is parse, chunk and embed rather than the fetch.
        """
        return (
            existing is not None and existing.status in SETTLED and existing.content_hash == digest
        )

    # --- records ---------------------------------------------------------------------------

    async def _retain(self, raw: RawDocument, source_bytes: bytes) -> Retention:
        """Keep the connector's bytes, or record why they were not kept.

        Failing to retain never fails a document: the document is still indexable, and what is
        lost is a repair option rather than content. The reason is recorded so the set of
        documents for which a re-crawl is the only repair stays a query.
        """
        try:
            return await self._blobs.retain(source_bytes, raw.media_type)
        except Exception as exc:  # noqa: BLE001 - failing to keep bytes must not fail a document
            return Retention(omitted_reason=f"retention failed: {type(exc).__name__}: {exc}")

    async def _store_record(
        self,
        result: ChainResult,
        *,
        raw: RawDocument,
        source: str,
        digest: str,
        version_token: str | None,
        title: str,
        identifier: str,
        existing: Document | None,
        retention: Retention,
    ) -> Document:
        """Write the document row for whatever the chain concluded.

        **A chunk-less terminal status still stores the document.** Storing the failure is what
        makes it re-queryable, skippable on the next sync, and reachable by a re-parse the day
        the missing capability arrives. An unstored failure is re-fetched on every sync, absent
        from every listing, and invisible to any repair.
        """
        keep_status = self._keeps_status(existing, result.status)
        metadata: Metadata = {**(existing.metadata if existing else {}), **result.metadata}
        if keep_status:
            # The document keeps everything a reader can see, and the failure still goes on the
            # record. It simply does not cost anybody a document that was working.
            metadata["last_ingest_error"] = {
                "stage": PipelineStage.PARSE.value,
                "detail": result.status_detail,
            }
        settled = existing if keep_status and existing else None
        document = Document(
            id=identifier,
            source=source,
            source_id=raw.source_id,
            uri=raw.uri,
            title=title or (existing.title if existing else ""),
            content_hash=settled.content_hash if settled else digest,
            version_token=version_token,
            original_ref=settled.original_ref if settled else retention.ref,
            media_type=raw.media_type,
            status=settled.status if settled else result.status,
            status_detail=(settled.status_detail if settled else result.status_detail) or None,
            failed_stage=settled.failed_stage if settled else result.failed_stage,
            metadata=metadata,
        )
        stored = await self._store.upsert_document(document)
        if settled is None:
            await self._store.set_original(
                stored.id, ref=retention.ref, omitted_reason=retention.omitted_reason
            )
        if result.status in {DocumentStatus.CONTAINER, DocumentStatus.NO_EXTRACTABLE_TEXT}:
            await self._store.replace_chunks(stored.id, [])
        return stored

    @staticmethod
    def _keeps_status(existing: Document | None, proposed: DocumentStatus) -> bool:
        """Whether an indexed document keeps its status rather than taking the new one.

        Only a ``failed`` outcome is refused, and only for a document that is currently
        servable. Everything else is a conclusion about the new bytes rather than a failure to
        reach one.
        """
        return (
            existing is not None
            and existing.status is DocumentStatus.INDEXED
            and proposed is DocumentStatus.FAILED
        )

    async def _settle(
        self,
        result: ChainResult,
        *,
        raw: RawDocument,
        source: str,
        digest: str,
        version_token: str | None,
        title: str,
        identifier: str,
        existing: Document | None,
    ) -> DocumentOutcome:
        document = await self._store_record(
            result,
            raw=raw,
            source=source,
            digest=digest,
            version_token=version_token,
            title=title,
            identifier=identifier,
            existing=existing,
            retention=Retention(omitted_reason="not retained: the document was skipped"),
        )
        await self._observe(document)
        return DocumentOutcome(
            source_id=raw.source_id,
            status=document.status,
            document_id=document.id,
            detail=result.status_detail,
        )

    async def _advance(self, existing: Document | None, status: DocumentStatus) -> None:
        """Record an in-flight status, unless doing so would unserve a working document.

        A document with no servable content loses nothing by being marked in flight, and gains
        a recovery sweep that can find it. An ``indexed`` one has everything to lose: it would
        stop being returned the moment a re-sync began, and stay that way if the re-sync failed.
        """
        if existing is None or existing.status is DocumentStatus.INDEXED:
            return
        await self._store.set_status(existing.id, status)

    async def _observe(self, document: Document) -> None:
        """Let hooks see a committed document. Their failure is theirs, not the document's."""
        try:
            await self._middleware.after_store(document)
        except Exception as exc:  # noqa: BLE001 - the document is already committed
            await self._store.annotate(
                document.id, {"last_after_store_error": f"{type(exc).__name__}: {exc}"}
            )

    async def _fail(
        self,
        existing: Document | None,
        source: str,
        source_id: str,
        stage: PipelineStage,
        detail: str,
        *,
        raw: RawDocument | None = None,
        digest: str = "",
        version_token: str | None = None,
        title: str = "",
    ) -> DocumentOutcome:
        """Record a failure that happened before there was anything to store."""
        if existing is not None:
            return await self._demote(existing, existing, stage, detail)
        if raw is None:
            # Nothing was fetched, so there is no content hash and no row to write. The
            # document is simply not indexed, and the next sync rediscovers it.
            return DocumentOutcome(source_id=source_id, status=DocumentStatus.FAILED, detail=detail)
        return await self._settle(
            ChainResult(
                blocks=[],
                status=DocumentStatus.FAILED,
                status_detail=detail,
                failed_stage=stage,
            ),
            raw=raw,
            source=source,
            digest=digest or content_hash(raw.as_bytes()),
            version_token=version_token,
            title=title,
            identifier=document_id(self._workspace, source, source_id),
            existing=None,
        )

    async def _demote(
        self,
        document: Document,
        existing: Document | None,
        stage: PipelineStage,
        detail: str,
    ) -> DocumentOutcome:
        """Record a failure against a document that already has a row.

        A document that was ``indexed`` keeps its status, its chunks and its vectors. The
        failure still goes on the record, in metadata, so nothing is quiet about it — it simply
        does not cost anybody a document that was working five minutes ago.
        """
        was_indexed = existing is not None and existing.status is DocumentStatus.INDEXED
        await self._store.annotate(
            document.id, {"last_ingest_error": {"stage": stage.value, "detail": detail}}
        )
        if was_indexed:
            return DocumentOutcome(
                source_id=document.source_id,
                status=DocumentStatus.INDEXED,
                document_id=document.id,
                detail=detail,
            )
        await self._store.upsert_document(
            document.model_copy(
                update={
                    "status": DocumentStatus.FAILED,
                    "status_detail": detail,
                    "failed_stage": stage,
                }
            )
        )
        return DocumentOutcome(
            source_id=document.source_id,
            status=DocumentStatus.FAILED,
            document_id=document.id,
            detail=detail,
        )


def _with_status(document: Document, result: ChainResult) -> Document:
    return document.model_copy(
        update={
            "status": result.status,
            "status_detail": result.status_detail or None,
            "failed_stage": result.failed_stage,
            "metadata": {**document.metadata, **result.metadata},
        }
    )


def _member_raw(member: ExpandedMember) -> RawDocument:
    """A member's bytes, addressed by the identity the container gave it.

    The identity is taken from the member rather than from the bytes it wrapped, because a
    file inside an archive is identified by its path within that archive — two copies of the
    same PDF at two paths are two documents, and one of them is not a duplicate of the other.
    """
    return member.raw.model_copy(update={"source_id": member.source_id, "uri": member.uri})


def _member_title(member: ExpandedMember) -> str:
    title = member.metadata.get("title")
    return title if isinstance(title, str) else ""


__all__ = [
    "BlobSink",
    "DocumentOutcome",
    "IngestPipeline",
    "NoRetention",
    "RunReport",
]
