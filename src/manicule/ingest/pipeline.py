"""discover → fetch → parse → chunk → embed → store, several documents at a time.

**The unit of work is one document. A batch is a scheduling artifact with no semantics of its
own.** That is what makes "one bad document never aborts a batch" a structural property rather
than a promise: there is no batch-level transaction to abort, and no batch-level state a
document can corrupt. Every failure this module catches is attributed to a document, recorded,
and left behind.

**A run is three stages joined by bounded hand-offs**, and every bound comes from configuration
rather than from how many tasks happen to exist. Discovery fills a hand-off; ``fetch_concurrency``
fetch workers drain it and fill the next; ``parse_workers + 1`` ingest workers drain that one and
carry a document the rest of the way. Nothing anywhere gathers a task per document, because a
task per document is the same thing as no bound at all — see :meth:`IngestPipeline.run` and
``docs/ingest.md`` §8.3.

**What concurrency is not allowed to touch is the write sequence.** One document's record,
chunks, glossary and vectors are published under the keyed lock in :meth:`IngestPipeline._mutating`
and guarded by the compare-and-swap in :meth:`IngestPipeline._commit`, exactly as they were when
one document went through at a time. Running several documents at once makes those guards more
reachable, not less needed, so the concurrency is between documents and never inside one.

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

The write order and its crash windows belong to ``docs/storage.md`` §8.2 and are honored
rather than restated: chunks, then vectors, then ``indexed`` last, in the transaction that is
the commit point.
"""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from enum import StrEnum
from importlib.metadata import PackageNotFoundError
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from manicule.core.content import (
    SETTLED,
    Document,
    DocumentRevision,
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
from manicule.core.provenance import Provenance
from manicule.ingest.embedding import EmbeddingWork, embed_or_reuse
from manicule.ingest.glossary import detect_entries
from manicule.ingest.glossary_lineage import glossary_fingerprint
from manicule.ingest.ports import GlossaryWriter
from manicule.ingest.refusals import require_measured
from manicule.ingest.stages import Conveyor, CountedLock, Gauge, StageReport
from manicule.ingest.workers import AttemptResult, default_worker_count
from manicule.parsers.chain import (
    Attempt,
    ChainResult,
    Outcome,
    classify,
    container_result,
    run_chain,
)
from manicule.parsers.expansion import ExpandedMember, MemberFailure
from manicule.parsers.versions import parse_fingerprint

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Sequence

    from manicule.core.content import Chunk, Metadata
    from manicule.core.fingerprints import ChunkFingerprint, ParseFingerprint
    from manicule.core.protocols import Chunker, Connector, Embedder, VectorStore
    from manicule.core.sources import DiscoveredDoc, DocRef, Watermark
    from manicule.ingest.middleware import MiddlewareRunner
    from manicule.ingest.ports import IngestStore
    from manicule.ingest.workers import ParseRunner
    from manicule.parsers.expansion import MemberOutcome


type Watching = Callable[[str], None]
"""How a run says what it has done so far, for a caller streaming progress to a person.

Called from inside an ingest worker after each document reaches a terminal outcome, so it must
not block and must not raise. The one implementation appends a string to a list; the socket
turns that into a frame on its own tick, which is what keeps a slow reader from becoming
backpressure on the pipeline.
"""


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


class _SupersededError(Exception):
    """A guarded write found the document had moved on. Internal, like :class:`_StageError`.

    An exception rather than a returned sentinel for the reason that one is: the guard fires at
    two different depths — the record write and the commit — and threading an "or it moved"
    return type up from both would put the decision in two places. There is one handler, in
    :meth:`IngestPipeline._ingest_one`, and it is the only thing that builds a superseded
    outcome.
    """

    def __init__(self, stored: Document | None) -> None:
        super().__init__("the stored document moved past the revision this work was derived from")
        self.stored = stored


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


class Change(StrEnum):
    """What differs between a stored document and what a connector just fetched.

    Four axes, kept apart because they change independently and cost different things to repair.
    A single "changed" boolean would answer whether to re-ingest and destroy the answer to why —
    and the second is the question somebody watching an unexpected re-ingest of a whole corpus is
    actually asking.
    """

    CONTENT = "content"
    """The source bytes differ. A re-parse, re-chunk and re-embed."""

    METADATA = "metadata"
    """The source record differs while the bytes may not. A mirrored page whose manifest was
    corrected: the citation changes, the text does not."""

    LINEAGE = "lineage"
    """The parser that produced the stored text is not the one installed now
    (``docs/storage.md`` §6.4). Neither the bytes nor the metadata moved; what we make of them
    did."""

    ROUTING = "routing"
    """The source now declares a different media type, so a **different parser** would read it.

    Distinct from :attr:`LINEAGE`, and the distinction is not academic. Lineage asks whether the
    parser that ran has since changed *version*, and it answers by looking up ``parser_used`` — so
    when a document is re-routed to a different parser entirely, lineage compares the old parser
    against itself, finds it unchanged, and reports the document current. Nothing else notices
    either: the bytes are identical and the source record has not moved.

    The consequence is a corpus that keeps text produced by a parser nothing routes to any more,
    for ever, with no signal — and unlike a version bump there is no number to move that would fix
    it, because the parser's *identity* changed rather than its version. Introducing a media type is
    exactly the operation that does this, which is why this axis exists before the first one is
    introduced rather than after.
    """


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

    superseded: bool = False
    """Whether this document moved past the revision the work was derived from, mid-flight.

    Expected concurrency rather than a failure, and kept off ``status`` for that reason: the
    stored document is *fine*, it is simply newer than what this operation had in hand, and
    nothing was written. Only an operation that supplied an expected revision can produce it —
    a connector sync carries the newest bytes there are and never asks to be guarded against
    somebody else's.
    """

    glossary_detail: str = ""
    """Why detection did not produce this document's entries, when it did not.

    **A field of its own rather than a second use of ``detail``**, and the separation is
    load-bearing rather than tidy. :func:`~manicule.ingest.reindex.re_parse` reads an
    ``indexed`` outcome carrying a ``detail`` as "the parse failed and the previous version was
    kept", which is exactly the right reading of that field — and putting a detector failure
    there would make a re-parse of a perfectly rebuilt document report a failure, skip the
    lineage read-back that keeps its sweep cursor honest, and count the document as unrepaired.
    A detector failure is not a parse failure. It costs this document's glossary and nothing
    else.
    """

    members: tuple[str, ...] = ()
    """Source ids of documents found inside this one, queued rather than recursed into."""

    embedding: EmbeddingWork = field(default_factory=EmbeddingWork)
    """What the embed stage cost: what was reused, what was embedded, and how many batches.

    Zero everywhere for a document change detection skipped, because a document that was not
    re-indexed did not reach the embedder — which is a different statement from a document
    whose vectors were all reusable, and the reason this is a whole record rather than a
    number.
    """


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

    limited: bool = False
    """Whether ``--limit`` stopped discovery before the source was exhausted.

    **A limited run is bounded, not clean, and not an error.** Nothing went wrong and every
    accepted document was carried to a terminal outcome, so ``error`` stays empty and the
    counters mean what they say — but discovery stopped early, so the position the connector
    reports describes an enumeration that was never finished. Reporting it as clean is how a
    ``--limit 10`` on a corpus of ten thousand silently skips the other nine thousand nine
    hundred and ninety on every subsequent sync.
    """

    unrecorded: int = 0
    """Accepted documents that reached no durable record at all.

    Exactly one shape produces this: a fetch that failed for a document the index had never
    seen, so there is no row to write the failure against and nothing to re-query. Every other
    failure stores something — ``failed`` with a stage, or an ``indexed`` document keeping the
    content it had — and is therefore findable afterwards.

    It is counted because it is the one outcome a watermark must not be advanced past. The
    document exists at the source, was enumerated once, and is in no index; if the position
    moves beyond it, no later sync enumerates it again and nothing anywhere reports a problem.
    """

    glossary_failures: list[str] = field(default_factory=list[str])
    """One line per document whose definitions the detector could not read.

    Here rather than nowhere, because "does not advance the fingerprint" is only half of failing
    closed. The other half is that somebody finds out: the document keeps the entries it had and
    keeps its stale lineage, so the next survey names it — but a run that hit a detector bug on
    every document and reported a clean sweep of green counters would be a system that had
    stopped detecting anything and said nothing about it.
    """

    stages: StageReport = field(default_factory=StageReport)
    """What the stages did: queue depths, stage occupancy, and peak retained bodies.

    Counts only, never content. It is what makes "the bound held" checkable by a test and by
    ``doctor`` (``docs/ingest.md`` §14) rather than a claim about code somebody has read.
    """

    @property
    def indexed(self) -> int:
        return self.by_status.get(DocumentStatus.INDEXED.value, 0)

    @property
    def clean(self) -> bool:
        """Whether the run finished without an enumeration failure."""
        return not self.error

    @property
    def complete(self) -> bool:
        """Whether this run enumerated the whole source and landed everything it accepted.

        **The watermark gate, and it is three conditions rather than one.** A clean run is only
        the first: a run stopped by ``--limit`` enumerated a prefix, and a run that lost a
        document to a fetch failure before the index had ever seen it left something behind that
        the position would hide forever. Each of the three is a way for the connector's reported
        position to describe more than what actually landed, and the watermark is a promise that
        it does not.
        """
        return self.clean and not self.limited and self.unrecorded == 0

    def record(self, outcome: DocumentOutcome, *, expanded: bool = False) -> None:
        """Count one outcome. Called from every fetch and ingest worker of a staged run.

        **Nothing in here awaits, and that is what makes it safe from all of them.** A coroutine
        is only descheduled at an ``await``, so every read-modify-write below completes before
        another worker can run — which is the same reason the refcount in
        :meth:`IngestPipeline._mutating` needs no lock of its own. Adding an ``await`` here would
        silently start losing counts under concurrency, and a lost count in a report is a run
        that says it did less work than it did.
        """
        if expanded:
            self.expanded += 1
        else:
            self.discovered += 1
            if not outcome.document_id:
                # No row anywhere: a fetch that failed for a document nothing had stored. The
                # only outcome with nothing to find afterwards, and therefore the only one that
                # holds the watermark back.
                self.unrecorded += 1
        if outcome.glossary_detail:
            self.glossary_failures.append(
                f"{outcome.document_id or outcome.source_id}: {outcome.glossary_detail}"
            )
        if outcome.skipped == "version":
            self.skipped_version += 1
            return
        if outcome.skipped == "hash":
            self.skipped_hash += 1
            return
        key = outcome.status.value
        self.by_status[key] = self.by_status.get(key, 0) + 1

    def settle(self) -> None:
        """Put the order-sensitive parts into an order that does not depend on who finished first.

        Counters do not care in what order they were incremented, but ``glossary_failures`` is a
        list, and under concurrency the order documents complete in is not the order they were
        discovered in. Two identical runs would then produce two different reports, which makes
        a diagnostic impossible to diff and a test pass or fail on scheduling. Sorted, because
        each line begins with the document id and a document id is derived rather than assigned.
        """
        self.glossary_failures.sort()

    def as_metadata(self) -> Metadata:
        return {
            "last_run": {
                "discovered": self.discovered,
                "expanded": self.expanded,
                "skipped_version": self.skipped_version,
                "skipped_hash": self.skipped_hash,
                "by_status": dict(self.by_status),
                "error": self.error,
                "limited": self.limited,
                "unrecorded": self.unrecorded,
                "glossary_failures": list(self.glossary_failures),
                "stages": self.stages.as_metadata(),
            }
        }


@dataclass(frozen=True, slots=True)
class _Fetched:
    """What crosses the hand-off between the fetch stage and the ingest stage.

    The bytes, and the two things the fetch stage already read that the ingest stage would
    otherwise read again: what the source said about this document, and what the index holds for
    it. Re-reading the stored document on the far side would be a second round trip *and* a
    second snapshot, and the two snapshots disagreeing is the shape of a lost update.
    """

    raw: RawDocument
    discovered: DiscoveredDoc
    existing: Document | None


@dataclass
class _Sync:
    """One run's moving parts, in one object rather than eight arguments down five methods.

    Internal, and deliberately not the report: the report is what a caller reads afterwards,
    and this is what the stages write to each other while they work.
    """

    connector: Connector
    report: RunReport
    limit: int | None
    watermark: Watermark | None
    refs: Conveyor[DiscoveredDoc]
    """Discovery to fetch. Carries references, so its depth costs metadata rather than bodies."""

    bodies: Conveyor[_Fetched]
    """Fetch to ingest. Carries bytes, so its depth is what bounds a run's memory."""

    watching: Watching | None = None
    """Where to say what has happened so far, or ``None`` when nobody is watching."""

    accepted: int = 0
    """Top-level documents handed to the fetch stage. What ``--limit`` bounds, counted where
    the bound is applied rather than derived afterwards from what finished."""

    bodies_held: Gauge = field(default_factory=lambda: Gauge("bodies"))
    """Fetched bodies in memory: queued, and held by an ingest worker.

    **Per run rather than on the pipeline**, unlike the parse and embed gauges. Those count
    process-wide resources — one pool, one accelerator — and a second operation sharing this
    pipeline genuinely is inside them. A body belongs to the run that fetched it, and a run that
    was canceled with items still queued would otherwise leave a count nothing will ever release,
    inflating every later run's report by a number that only ever grows.
    """

    stop: asyncio.Event = field(default_factory=asyncio.Event)
    """Set on cancellation. Discovery is the only stage that reads it, because stopping
    discovery is what brings every stage behind it down in order."""


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
        parse_workers: int = 0,
        queue_depth_factor: int = 2,
        shutdown_grace_s: float = 30.0,
        max_fetch_bytes: int = 256 * 1024 * 1024,
        target_batch_tokens: int = 16_384,
        max_embed_batch: int = 64,
        parse_fingerprints: Callable[[str], ParseFingerprint | None] = parse_fingerprint,
        glossary: GlossaryWriter | None = None,
        detect_glossary: bool = True,
    ) -> None:
        # Second of the two places this is refused, and not a redundant one.
        # `check_before_run` is the once-per-run boundary and is what an operator meets; this
        # one is the boundary in *code*, because a pipeline is constructible without going
        # through that function and everything it writes is permanent. A chunker counting with
        # a stand-in vocabulary must not be able to reach a store at all.
        require_measured(chunk_fingerprint)
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
        self._parse_fingerprints = parse_fingerprints
        # Two conditions, and the second is not configuration. Detection is switchable, *and*
        # the store has to be able to hold the result — a pipeline that detected definitions
        # and had nowhere to put them would spend the work on every document and produce
        # nothing, which reads from the outside as a detector that finds nothing.
        self._glossary = glossary if glossary is not None else _writer_of(store)
        self._detect_glossary = detect_glossary and self._glossary is not None
        # Read once per pipeline rather than per document: it digests two source files and the
        # answer cannot change under a running process. `None` where the store cannot hold
        # entries at all — there is no glossary state in that index, so there is no lineage to
        # claim about it, and stamping one would describe rows that have nowhere to live.
        self._glossary_lineage = (
            None
            if self._glossary is None
            else glossary_fingerprint(
                enabled=detect_glossary, middleware=middleware.chain()
            ).canonical()
        )
        # --- how many of each stage a run may occupy, all derived here and nowhere else ------
        #
        # Read once, at construction, so that every bound in one run comes from one reading of
        # one configuration. Deriving a queue size at the moment it is needed is how a hand-off
        # ends up sized against a number that has since moved.
        self._fetch_workers = max(1, fetch_concurrency)
        # One more than the parse pool, so a document waiting for the embedding lock does not
        # leave a parse worker idle behind it. Not two more: past the pool size the extra
        # workers only queue for the same accelerator, and each of them is holding a fetched
        # body in memory while it waits.
        self._parse_workers = parse_workers if parse_workers > 0 else default_worker_count()
        self._ingest_workers = self._parse_workers + 1
        # `docs/ingest.md` §8.3: twice the consumer's parallelism. Deep enough that a consumer
        # never idles waiting for the stage in front of it to produce one more item, shallow
        # enough that "how much is in memory" stays a small multiple of the worker count.
        self._queue_depth_factor = max(1, queue_depth_factor)
        self._shutdown_grace_s = max(0.0, shutdown_grace_s)
        # Retained for callers of `ingest` and `ingest_raw` that are not a staged run. The staged
        # run bounds fetches structurally, by running exactly this many fetch workers, so inside
        # it this semaphore is never contended — which is the right relationship between a
        # structural bound and the check that it holds.
        self._fetching = asyncio.Semaphore(self._fetch_workers)
        self._embedding = CountedLock("embed")
        self._parsing = Gauge("parse")
        self._fetches = Gauge("fetch")
        self._mutations: dict[str, tuple[asyncio.Lock, int]] = {}

    # --- the two locks, and what each one is for ------------------------------------------

    @asynccontextmanager
    async def _mutating(self, document_id: str) -> AsyncGenerator[None]:
        """Exclude other work on **this** document for the length of one document's writes.

        **Keyed, because the thing that must not interleave is one document's write sequence
        and nothing wider.** A pipeline-wide lock would serialize a sweep against a sync over
        entirely unrelated pages, which is a throughput cost paid to fix a correctness problem
        neither of them has. The three writes that publish a document — its record, its chunks
        and glossary, its vectors — are not one statement and cannot be made one, so what makes
        them indivisible is holding this from before the first to after the last.

        **This is not the durable guard, and must not be mistaken for one.** It is an
        ``asyncio.Lock``: it holds within one event loop in one process, and a second process
        opened on the same data directory knows nothing about it. That case is the exclusive
        lock on ``<data_dir>/manicule.lock`` (:class:`~manicule.ingest.recovery.InstanceLock`),
        and the invariant that survives both being absent is the compare-and-swap at the commit
        — see :meth:`~manicule.ingest.ports.IngestStore.commit_document`.

        **Distinct from** ``self._embedding``, which serializes *the model* across every
        document because there is one accelerator. Two documents may be mutated at once and
        must not be embedded at once; one document may be neither. Sharing one lock for both
        would make each of them wrong in the other's direction.

        The entry is dropped when the last holder leaves, so a sweep over a corpus does not
        accumulate one lock per document it has finished with. The refcount is safe without a
        lock of its own because nothing between reading it and writing it back awaits.
        """
        lock, holders = self._mutations.get(document_id, (asyncio.Lock(), 0))
        self._mutations[document_id] = (lock, holders + 1)
        try:
            async with lock:
                yield
        finally:
            lock, holders = self._mutations[document_id]
            if holders > 1:
                self._mutations[document_id] = (lock, holders - 1)
            else:
                del self._mutations[document_id]

    @property
    def glossary_lineage(self) -> str | None:
        """The detector identity this pipeline stamps, or ``None`` if it stamps none.

        Public because it is the thing that has to agree with what
        :meth:`~manicule.app.ports.Ingesting.glossary_fingerprint` reports and what the repair
        selects against, and an agreement nothing can read is an agreement nobody can check.
        ``None`` only where the store cannot hold entries at all.
        """
        return self._glossary_lineage

    # --- a run: three stages, two bounded hand-offs -----------------------------------------

    async def run(
        self,
        connector: Connector,
        *,
        limit: int | None = None,
        watching: Watching | None = None,
    ) -> RunReport:
        """Ingest everything a connector reports as changed since its watermark.

        **Three stages, joined by two bounded hand-offs, and every bound derived from
        configuration:**

        .. code-block:: text

            discover
               │  fetch hand-off, depth = queue_depth_factor x fetch_concurrency
               ▼
            fetch x fetch_concurrency          change detection, then the network
               │  parse hand-off, depth = queue_depth_factor x ingest workers
               ▼
            ingest x (parse_workers + 1)       parse in the pool, chunk, embed under
                                               one lock, commit under the document's

        Discovery is the only stage with nothing in front of it, so it is where backpressure
        has to arrive. It does: putting a document into a full hand-off waits, and every stage
        downstream propagates that wait upward, so a slow embedder eventually stops the source
        being paged. That is a correctness requirement rather than a memory one — a connector
        that races ahead of durable progress exhausts its pagination cursors and fails a sync
        that had nothing wrong with it (``docs/ingest.md`` §8.3).

        **What the concurrency does not change.** Each document still travels the same path it
        did one at a time, and the per-document lock still spans its record, chunks, glossary
        and vectors. Two documents may be in the ingest stage at once; one document is never in
        two places.

        **``limit`` bounds acceptance, not completion.** Discovery stops after handing ``limit``
        top-level documents downstream, and everything already accepted is carried to a terminal
        outcome before this returns. Members found inside a container do not count against it —
        one archive of five hundred files must not exhaust a limit of ten. A run stopped this
        way is *bounded*: no error, and no watermark.

        **Cancellation.** ``Ctrl-C`` stops discovery, gives what is already accepted
        ``shutdown_grace_s`` to reach a terminal outcome, then cancels the stages and re-raises.
        No watermark is written on either path and no task outlives this call. A second
        ``Ctrl-C`` inside the grace window skips the wait, which is what makes the impatient
        case safe rather than different (``docs/ingest.md`` §13.3).

        Args:
            connector: The source to pull.
            limit: Stop discovery after this many top-level documents.
            watching: Called with one sentence each time a document reaches a terminal outcome,
                for a caller that is streaming progress to somebody. It is called from inside an
                ingest worker, so it must not block and must not raise — the one implementation
                appends to a list. ``None`` is the ordinary case and costs a branch per document.
        """
        run = _Sync(
            connector=connector,
            report=RunReport(connector=connector.name),
            limit=limit,
            watching=watching,
            watermark=await self._store.get_watermark(connector.name),
            refs=Conveyor(
                name="fetch",
                capacity=self._queue_depth_factor * self._fetch_workers,
                consumers=self._fetch_workers,
            ),
            bodies=Conveyor(
                name="parse",
                capacity=self._queue_depth_factor * self._ingest_workers,
                consumers=self._ingest_workers,
                producers=self._fetch_workers,
            ),
        )
        # Peaks only. The active counts belong to whoever is inside the stage right now, and a
        # second operation sharing this pipeline is one of them.
        for gauge in (self._fetches, self._parsing, self._embedding.gauge):
            gauge.rebase()

        stages = asyncio.create_task(self._drive(run), name=f"ingest:{connector.name}")
        try:
            # Shielded so that a cancellation arriving here does not tear the stages down before
            # anything has had a chance to finish the document it is holding. The stages are
            # still a child task of this call and are joined on every path below, so nothing
            # survives this method.
            await asyncio.shield(stages)
        except asyncio.CancelledError:
            await self._stop_within_grace(run, stages)
            raise
        except ExceptionGroup as failures:
            # A stage failed for a reason that is not a document's: the document store went
            # away, or this module has a defect. The task group has already stopped and joined
            # the other stages, so what is left is to say so and leave the watermark alone.
            # Documents that were mid-write when it happened are in a non-terminal status, which
            # is what the recovery sweep (`docs/ingest.md` §6.4) exists to finish.
            run.report.error = _first_failure(failures)
        except BaseException:
            stages.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stages
            raise

        run.report.stages = self._stage_report(run)
        run.report.settle()
        if run.report.complete:
            await self._advance_watermark(connector)
        await self._store.record_connector_metadata(connector.name, run.report.as_metadata())
        return run.report

    async def _drive(self, run: _Sync) -> None:
        """Start every stage, and return when discovery is spent and every acceptance is done.

        One :class:`asyncio.TaskGroup`, so the stages are children of this call: it returns only
        when all of them have, and canceling it cancels and joins all of them. There is no
        detached task and nothing to leak.

        A stage task raising cancels its siblings, and that is deliberate rather than tolerated:
        everything a document can do is already an outcome by the time it reaches here, so what
        is left to raise is the store or a defect, and neither is survivable by carrying on.
        """
        refs, bodies = run.refs, run.bodies
        async with asyncio.TaskGroup() as stages:
            stages.create_task(self._discover_into(run, refs), name="discover")
            for worker in range(self._fetch_workers):
                stages.create_task(self._fetch_into(run, refs, bodies), name=f"fetch-{worker}")
            for worker in range(self._ingest_workers):
                stages.create_task(self._ingest_from(run, bodies), name=f"ingest-{worker}")

    async def _discover_into(self, run: _Sync, refs: Conveyor[DiscoveredDoc]) -> None:
        """Pull the source, hand each document to the fetch stage, and stop when told to.

        The three ways it ends are all recorded, because each means something different to the
        watermark: exhausted (may advance), stopped at ``limit`` (bounded, may not), and raised
        (unclean, may not).
        """
        stream = run.connector.discover(run.watermark)
        try:
            async for discovered in stream:
                if run.stop.is_set():
                    break
                # Before the counter, so a document is counted as accepted only once it is
                # somewhere a worker will find it. The wait inside `put` is the backpressure.
                await refs.put(discovered)
                run.accepted += 1
                if run.limit is not None and run.accepted >= run.limit:
                    run.report.limited = True
                    break
        except Exception as exc:  # noqa: BLE001 - an enumeration failure is not a crash
            run.report.error = f"{type(exc).__name__}: {exc}"
        finally:
            closer = getattr(stream, "aclose", None)
            if closer is not None:
                await closer()
            # However discovery ended, the stage in front of it has to be told, or every fetch
            # worker waits for an item that is never coming and the run never returns.
            refs.finish()

    async def _fetch_into(
        self, run: _Sync, refs: Conveyor[DiscoveredDoc], bodies: Conveyor[_Fetched]
    ) -> None:
        """One fetch worker: change detection, then the network, then the next stage.

        Level-1 change detection lives here rather than in discovery because it is what avoids
        the fetch, and it costs a store read that has no business blocking the source's paging.
        A document that skips never reaches the parse hand-off at all, which is why an unchanged
        corpus flows through this stage at the speed of the store rather than of the model.
        """
        try:
            while (discovered := await refs.take()) is not None:
                accepted = await self._accept(run.connector, discovered)
                if isinstance(accepted, DocumentOutcome):
                    run.report.record(accepted)
                    # Reported here as well as in the ingest stage, because a document that
                    # skips never reaches the ingest stage at all — and a resync of a corpus
                    # nobody has touched is *entirely* skips. Without this, the longest quiet
                    # run there is is the one that says nothing: an unchanged ten-thousand-page
                    # corpus walked at the speed of the store, reporting for the first time when
                    # it finished.
                    _report_progress(run)
                    continue
                # Entered here and left in the ingest stage, because what is being counted is
                # how many fetched bodies are held in memory at once — which spans the queue
                # they wait in as well as the worker that has one. A block cannot express that,
                # so the pair is written out.
                #
                # **A `put` canceled before it lands leaves this entry unpaired, and that costs
                # nothing.** Only the peak is reported, and the peak was already right the moment
                # `enter` ran: this worker *is* holding a body while it waits. What goes stale is
                # the running count, which is per run (see `_Sync.bodies_held`) and is read by
                # nothing — and the only thing that cancels a fetch worker is the run ending.
                # Undoing it here would be a guard with no failure behind it; what has to stay
                # true is that this gauge belongs to the run rather than to the pipeline.
                run.bodies_held.enter()
                await bodies.put(accepted)
        finally:
            bodies.finish()

    async def _ingest_from(self, run: _Sync, bodies: Conveyor[_Fetched]) -> None:
        """One ingest worker: parse, chunk, embed and commit one document, then the next.

        The stage that holds a document's whole write sequence, unchanged from when there was
        one of these. What bounds it is how many of these workers exist, and what serializes the
        parts that must not overlap is the parse pool, the embedding lock and the per-document
        lock — each of which was already the bound for the resource it names.

        **Nothing broad is caught here, and that is the same decision the sequential loop made.**
        Everything a *document* can do — a parser that raises, a hook that misbehaves, a model
        that refuses, a vector store that will not take an upsert — is already turned into a
        recorded outcome inside :meth:`ingest_raw`, and those documents fail alone. What is left
        to escape is the document store itself going away, and a run that carried on through
        that would report every remaining document as failed, finish clean, and advance a
        watermark past a corpus it never wrote. So it propagates, the task group stops the other
        stages, and :meth:`run` records it as the run's error.
        """
        while (fetched := await bodies.take()) is not None:
            try:
                outcomes = await self.ingest_raw(
                    fetched.raw,
                    source=run.connector.name,
                    version_token=fetched.discovered.version_token,
                    title=fetched.discovered.title,
                    existing=fetched.existing,
                )
            finally:
                # The other half of the pair entered in the fetch stage: this body is no longer
                # held. In a `finally`, so a document that failed still releases its accounting —
                # the deadlock this whole design has to avoid is a permit that a failure keeps.
                run.bodies_held.leave()
            for position, outcome in enumerate(outcomes):
                # The first outcome is the discovered document; anything after it came out of
                # the inside of it.
                run.report.record(outcome, expanded=position > 0)
            # After the whole document, not per outcome: a container that expanded into five
            # hundred members is one thing that happened to somebody watching, and five hundred
            # lines of it is not progress.
            _report_progress(run)

    async def _stop_within_grace(self, run: _Sync, stages: asyncio.Task[None]) -> None:
        """Stop pulling the source, let what is accepted finish, cancel the rest at the deadline.

        ``docs/ingest.md`` §13.3, made true. Stopping discovery is enough to bring the whole run
        down on its own: the fetch hand-off closes, its workers exit on the end-of-stream and
        close the parse hand-off behind them, and the ingest workers finish what is queued. So
        the grace window is a bound on *that*, not a separate drain protocol.

        **A second cancellation skips the wait**, because the wait is itself cancellable — which
        is the documented behavior and the reason the impatient case is safe rather than
        different. Either way the recovery sweep is what finishes the story for a document left
        in flight, and no watermark is written.
        """
        run.stop.set()
        try:
            await asyncio.wait_for(asyncio.shield(stages), self._shutdown_grace_s)
        except (TimeoutError, asyncio.CancelledError):
            stages.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stages

    def _stage_report(self, run: _Sync) -> StageReport:
        """What the stages did, read out of the gauges and the hand-offs once, at the end."""
        return StageReport(
            accepted=run.accepted,
            fetch_queue=run.refs.report(),
            parse_queue=run.bodies.report(),
            peak_fetches=self._fetches.peak,
            peak_parses=self._parsing.peak,
            peak_embeds=self._embedding.gauge.peak,
            peak_bodies=run.bodies_held.peak,
        )

    async def _advance_watermark(self, connector: Connector) -> None:
        """Record how far a complete run got, if the connector can say.

        **Only on a complete run**, and that is the whole of resumability. An interrupted sync
        re-enumerates from the last good point; change detection (§4) makes re-enumeration
        cheap, because already-ingested documents skip at level 1 without a fetch; and the
        recovery sweep requeues anything caught in flight. So resume is: run it again. There is
        no checkpoint file, no resume token, and nothing to corrupt.

        **Complete is three conditions, and concurrency is why saying so matters.** The position
        a connector reports is one value describing a whole enumeration, so it cannot be advanced
        partly — which means it must not be advanced at all unless the enumeration finished *and*
        everything it produced landed. Completion order under a staged run is not discovery
        order, so "the last document finished" says nothing; :attr:`RunReport.complete` is the
        condition, and it is checked after every accepted document has reached a terminal
        outcome rather than as they go.

        ``Connector.watermark`` is the other half of ``discover``, which consumes a position and
        for a while had nowhere to produce the next one. A connector with no change signal
        answers ``None`` and nothing is written — which is the honest outcome, because a source
        that cannot say where it got to re-enumerating is cheaper than one that invents a
        position and is believed.

        **Two checks guard this write and neither is redundant.**
        :func:`manicule.testing.assert_connector_contract` catches a connector that advances its
        watermark as it yields — and it catches it on an *uninterrupted* run, which is the only
        kind anyone looks at. This gate catches a caller that persists a watermark for work that
        was not committed, on a run that has already gone wrong. A connector can pass the first
        and still lose documents through a caller that ignores the second.

        Deleting either one restores a failure whose symptom is documents that exist in the
        source, were enumerated once, and are in no index — permanently, with nothing raised,
        and no later sync fixing it. Two tests that look like they overlap are the shape this
        guarantee has to take; they are checking different halves of it.
        """
        reached = connector.watermark
        if reached is not None:
            await self._store.set_watermark(connector.name, reached)

    async def ingest(
        self, connector: Connector, discovered: DiscoveredDoc
    ) -> list[DocumentOutcome]:
        """One discovered document, from the change check to the commit.

        The two stages a run pipelines, run back to back instead. **One implementation, so the
        direct path and the staged path cannot come to disagree** about when a document skips,
        when it is fetched, or what an outcome for it looks like — which is exactly the drift a
        second copy of change detection would produce, and it would produce it silently.

        Returns a list because a container is several documents: the archive itself, then
        every member it expanded into. Never raises for anything a document did. A connector
        that cannot fetch it, a parser that hangs, a hook that misbehaves and a model that
        refuses are all recorded against that document and returned.
        """
        accepted = await self._accept(connector, discovered)
        if isinstance(accepted, DocumentOutcome):
            return [accepted]
        return await self.ingest_raw(
            accepted.raw,
            source=connector.name,
            version_token=discovered.version_token,
            title=discovered.title,
            existing=accepted.existing,
        )

    async def _accept(
        self, connector: Connector, discovered: DiscoveredDoc
    ) -> DocumentOutcome | _Fetched:
        """Decide whether a discovered document is worth fetching, and fetch it if it is.

        The first stage's whole job, and the reason it is a stage of its own: level-1 change
        detection is what avoids the fetch, so it belongs on the same side of the hand-off as
        the fetch rather than in discovery, where it would make a store read block the source
        being paged.

        Returns:
            The bytes and the change-detection context, or the outcome for a document that
            never needed them — one that skipped, and one whose fetch failed.
        """
        source = connector.name
        source_id = discovered.source_id
        existing = await self._store.find_document(source, source_id)

        if self._unchanged_by_token(existing, discovered):
            await self._store.record_seen(existing.id)  # pyright: ignore[reportOptionalMemberAccess]
            return DocumentOutcome(
                source_id=source_id,
                status=existing.status,  # pyright: ignore[reportOptionalMemberAccess]
                document_id=existing.id,  # pyright: ignore[reportOptionalMemberAccess]
                skipped="version",
            )

        await self._advance(existing, DocumentStatus.FETCHING)
        try:
            raw = await self._fetch(connector, discovered.ref)
        except Exception as exc:  # noqa: BLE001 - one source's failure is one document's
            return await self._fail(
                existing, source, source_id, PipelineStage.FETCH, f"{type(exc).__name__}: {exc}"
            )

        return _Fetched(raw=raw, discovered=discovered, existing=existing)

    async def ingest_raw(
        self,
        raw: RawDocument,
        *,
        source: str,
        version_token: str | None = None,
        title: str = "",
        existing: Document | None = None,
        force: bool = False,
        expected: DocumentRevision | None = None,
    ) -> list[DocumentOutcome]:
        """Everything from fetched bytes onwards, including anything found inside.

        The path shared by a connector sync, a member of a container, and
        a corpus-wide re-parse reading retained bytes. One implementation, so a re-parse cannot
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

        ``expected`` guards the *top-level* document only, and for the same reason ``force``
        does: it is a statement about the snapshot **this caller** read, and a member found
        inside the archive is a document the caller never saw. A caller with newer bytes than
        anything stored — every connector sync — passes nothing and is guarded against nobody.
        """
        outcome, members = await self._ingest_one(
            raw,
            source=source,
            version_token=version_token,
            title=title,
            existing=existing,
            force=force,
            expected=expected,
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
        expected: DocumentRevision | None = None,
    ) -> tuple[DocumentOutcome, tuple[MemberOutcome, ...]]:
        """One document, and whatever it turned out to contain."""
        source_bytes = raw.as_bytes()
        digest = content_hash(source_bytes)
        if existing is None:
            existing = await self._store.find_document(source, raw.source_id)

        if not force and self._unchanged_by_hash(existing, digest, raw):
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

        # Every write this document is about to receive happens inside this, and the two guards
        # inside it are what hold when this does not. Taken here rather than around each write,
        # because a document is published by three of them in sequence — its record, its chunks
        # and glossary, its vectors — and what must not interleave is the sequence.
        async with self._mutating(identifier):
            try:
                return await self._ingest_guarded(
                    raw,
                    source=source,
                    source_bytes=source_bytes,
                    digest=digest,
                    version_token=version_token,
                    title=title,
                    identifier=identifier,
                    existing=existing,
                    expected=expected,
                )
            except _SupersededError as moved:
                # Nothing was written, by construction: the guard fires on the first write this
                # document gets and again on the last, and the sequence between them is inside
                # the lock taken above. There is no derived state to reconcile because none of
                # it ever existed.
                return (
                    DocumentOutcome(
                        source_id=raw.source_id,
                        status=(moved.stored.status if moved.stored else DocumentStatus.PENDING),
                        document_id=identifier,
                        superseded=True,
                    ),
                    (),
                )

    async def _ingest_guarded(
        self,
        raw: RawDocument,
        *,
        source: str,
        source_bytes: bytes,
        digest: str,
        version_token: str | None,
        title: str,
        identifier: str,
        existing: Document | None,
        expected: DocumentRevision | None,
    ) -> tuple[DocumentOutcome, tuple[MemberOutcome, ...]]:
        """The part of one document's ingest that writes, under the lock and the guard.

        Split from :meth:`_ingest_one` rather than indented into it so that there is exactly one
        ``except _SupersededError`` and it is not buried inside eighty lines of stages.

        Raises:
            _SupersededError: The stored document moved past ``expected``. Nothing was written.
        """
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
                expected=expected,
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
            expected=expected,
        )
        if result.status is not DocumentStatus.PARSED:
            glossary_detail = ""
            if result.status is not DocumentStatus.FAILED:
                # A container's members and an "unsupported media type" verdict are both
                # conclusions a parser version reached about these bytes, and both are now
                # what is stored. A failure is not: it is the absence of a conclusion, and
                # claiming lineage for it would mark a document current on the strength of
                # the run that could not read it.
                #
                # The glossary is settled here for the same reason and by the same rule
                # `_nothing_to_index` states: a document with no chunks states no definitions,
                # and saying so explicitly rather than leaving it to the chunk cascade is what
                # keeps it a property of the pipeline instead of a property of one store's
                # schema. `_store_record` above has already emptied the chunks; a store holding
                # them elsewhere would otherwise keep a whole glossary for a page that no longer
                # has any text in it — and, now, keep it behind a lineage nothing ever advances.
                glossary_detail = await self._store_definitions(document, [])
                await self._store.set_lineage(
                    document.id,
                    chunk_fp=None,
                    embed_fp=None,
                    parse_fp=self._parse_lineage_of(document),
                )
            await self._observe(document)
            return (
                DocumentOutcome(
                    source_id=raw.source_id,
                    status=document.status,
                    document_id=document.id,
                    detail=result.status_detail,
                    glossary_detail=glossary_detail,
                    members=tuple(member.source_id for member in members),
                ),
                members,
            )

        return (
            await self._finish(result, document, raw=raw, existing=existing, expected=expected),
            (),
        )

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
            with self._fetches.holding():
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
            # Counted around the runner rather than around the document, because the pool's
            # parallelism is per attempt: this is the number that says whether one connector
            # sync is using more than one parse worker.
            with self._parsing.holding():
                result = await self._runner.run_attempt(name, document)
            captured.append(result)
            return result.blocks, result.attempt  # pyright: ignore[reportReturnType]

        try:
            result = await run_chain(chain, raw, attempt)  # pyright: ignore[reportArgumentType]
        except Exception as exc:  # noqa: BLE001 - the guarantee is worth one broad catch
            # The runner is a seam a plugin or a future backend sits behind, and a batch that
            # ends because one of them raised somewhere unanticipated is the exact failure this
            # pipeline promises not to have. A runner is *expected* to turn every parser
            # problem into a hard-failed attempt; if one ever does not, the document still
            # fails alone.
            attempted = tuple(a.attempt for a in captured)
            reason = f"{type(exc).__name__}: {exc}"
            broken = (*attempted, Attempt(parser="", outcome=Outcome.FAILED, reason=reason))
            return classify(raw, broken), ()
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
        expected: DocumentRevision | None = None,
    ) -> DocumentOutcome:
        """Chunk, embed and commit a document the chain produced text for.

        Each stage's failure is turned into one :class:`_StageError` and caught once, rather
        than returned from six places. The shape matters more than the line count: with six
        exits it is possible to add a seventh that forgets to record the failure, and a stage
        that fails without recording is a document that quietly stops being re-tried.

        Raises:
            _SupersededError: From :meth:`_commit`, and deliberately not caught by the broad
                ``except`` below. A document that moved on is not a document that failed to
                embed, and demoting it would record somebody else's success as this run's
                failure.
        """
        try:
            chunks = await self._prepare(result, document)
            if not chunks:
                return await self._nothing_to_index(result, document, raw=raw)
            await self._advance(existing, DocumentStatus.EMBEDDING)
            # The lock goes to `embed_or_reuse`, which holds it around the model call and
            # nothing else. Both reads here — this one and the vector store's — are about what
            # the index already holds, and taking the one lock every embedder in the process
            # shares while they run would serialize a sweep against a concurrent sync on work
            # neither of them needs the model for.
            previous = await self._previous_inputs(document, existing)
            vectors, work = await embed_or_reuse(
                self._embedder,
                chunks,
                vectors=self._vectors,
                chunk_fingerprint=self._chunk_fingerprint,
                previous=previous,
                target_batch_tokens=self._target_batch_tokens,
                maximum=self._max_embed_batch,
                lock=self._embedding,
            )
        except _StageError as failure:
            return await self._demote(document, existing, failure.stage, failure.detail)
        except ContextOverflowError as exc:
            return await self._demote(document, existing, PipelineStage.EMBED, str(exc))
        except Exception as exc:  # noqa: BLE001 - a model failure is this document's
            return await self._demote(
                document, existing, PipelineStage.EMBED, f"{type(exc).__name__}: {exc}"
            )

        # The embed stage's accounting is attached to what the commit returns rather than
        # passed into it. What it cost to produce these vectors is not one of the things the
        # commit has to get right, and threading it through would put another argument in the
        # one function whose ordering is the crash contract — which now also carries
        # `expected`, and has earned the right to be left alone.
        outcome = await self._commit(
            document, chunks, vectors, raw=raw, existing=existing, expected=expected
        )
        return replace(outcome, embedding=work)

    async def _previous_inputs(
        self, document: Document, existing: Document | None
    ) -> dict[str, str]:
        """What the index already recorded as each stored chunk's embedding input.

        Only used to tell two chunks apart that both have no vector row: one that never had
        one, and one whose row went missing while its embedding input did not change. Both are
        embedded; this decides which the report calls a repair.

        Skipped entirely for a document being ingested for the first time, where there is
        nothing stored and the query would be a round trip to learn that.
        """
        if existing is None:
            return {}
        stored = await self._store.document_chunks(document.id)
        return {chunk.id: chunk.embed_text for chunk in stored}

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
        # A document with no chunks states no definitions. Said explicitly rather than left to
        # the chunk cascade, because the cascade is a property of one store's schema and this
        # is a property of the pipeline: a store that kept its chunks in a separate service
        # would leave a whole glossary behind for a page that no longer has any text in it.
        # It is also a derived empty result, so it records lineage like any other.
        glossary_detail = await self._store_definitions(stored, [])
        # The determination "there is no text in this" is itself the output of a parser
        # version. Without lineage here, a document that yielded nothing would be re-parsed on
        # every sync forever — and, worse, the day a library learns to read it, nothing would
        # say which documents were judged empty by the version that could not.
        await self._store.set_lineage(
            stored.id, chunk_fp=None, embed_fp=None, parse_fp=self._parse_lineage_of(document)
        )
        await self._observe(stored)
        return DocumentOutcome(
            source_id=raw.source_id,
            status=stored.status,
            document_id=stored.id,
            detail=settled.status_detail,
            glossary_detail=glossary_detail,
        )

    async def _commit(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        *,
        raw: RawDocument,
        existing: Document | None,
        expected: DocumentRevision | None = None,
    ) -> DocumentOutcome:
        """Write in the one order that survives a crash at any point.

        1. Chunks. The document is not ``indexed``, so nothing is served.
        2. Vectors, upserted by chunk id, so re-running step 2 is free.
        3. ``indexed``, last, in the transaction that is the commit point.

        A crash between 1 and 2 leaves chunks with no vectors and nothing served; the repair
        re-embeds those chunk ids. A crash between 2 and 3 leaves vectors for a document that
        is not ``indexed``; the repair re-runs 2 idempotently and then 3.

        **``expected`` is verified twice here, and neither one is the other's spare.** Step 3
        is the write that publishes everything the first two staged, so a compare-and-swap
        there is a compare-and-swap on the act of publishing: it is the durable invariant, and
        putting it anywhere earlier would leave a window exactly as wide as the one it closes.
        Step 0 is what makes the *derived* writes safe. Chunks and glossary rows are replaced
        rather than added, so step 1 destroys whatever was there — and by the time step 3 could
        refuse, that has already happened. Verifying again after the model has run and before
        anything is replaced is what keeps a superseded re-parse from producing derived state at
        all, which the alternative — reconciling it afterwards — turns into a second race.

        Raises:
            _SupersededError: The stored document moved past ``expected``. From step 0 that
                means it moved while the document was being chunked and embedded; from step 3,
                while the derived writes were in flight. Neither is reachable from inside this
                process — :meth:`_mutating` is held across all of it — so both mean a second
                process is writing this data directory without the instance lock.
        """
        if expected is not None:
            # Step 0. Written rather than read: a `SELECT` and a later write have a gap between
            # them, and this whole class of bug is what lives in gaps like that.
            await self._publish(document, expected=expected)
        try:
            await self._store.replace_chunks(document.id, chunks)
            # Between the chunks and the vectors, because the entries have a foreign key to the
            # chunks and none to anything written later. The crash window this opens is
            # harmless in the one direction that matters: the document is not yet ``indexed``,
            # and a glossary lookup only ever reads entries of indexed documents — so entries
            # written by a run that died before its vectors are invisible until the repair
            # finishes the job.
            glossary_detail = await self._store_definitions(document, chunks)
            await self._vectors.upsert(chunks, vectors)
        except Exception as exc:  # noqa: BLE001 - a store failure is this document's
            return await self._demote(
                document, existing, PipelineStage.STORE, f"{type(exc).__name__}: {exc}"
            )

        indexed = await self._publish(
            document.model_copy(
                update={
                    "status": DocumentStatus.INDEXED,
                    "status_detail": None,
                    "failed_stage": None,
                }
            ),
            expected=expected,
        )
        await self._store.set_lineage(
            indexed.id,
            chunk_fp=self._chunk_fingerprint.canonical(),
            embed_fp=self._embedder.fingerprint.canonical(),
            parse_fp=self._parse_lineage_of(document),
        )
        await self._observe(indexed)
        return DocumentOutcome(
            source_id=raw.source_id,
            status=DocumentStatus.INDEXED,
            document_id=indexed.id,
            chunks=len(chunks),
            glossary_detail=glossary_detail,
        )

    async def _publish(self, document: Document, *, expected: DocumentRevision | None) -> Document:
        """Write a document row, guarded when the caller brought a revision to be guarded by.

        The one place a document row is written on the ingest path, so that "guarded or not" is
        decided once. A caller with no expectation — every connector sync — writes exactly as it
        always did: it is holding the newest bytes there are, and a guard against a document
        moving under it would be a guard against itself.

        Raises:
            _SupersededError: The stored document is no longer ``expected``, so nothing was written.
        """
        if expected is None:
            return await self._store.upsert_document(document)
        committed = await self._store.commit_document(document, expected=expected)
        if not committed.committed or committed.stored is None:
            raise _SupersededError(committed.stored)
        return committed.stored

    async def _store_definitions(self, document: Document, chunks: Sequence[Chunk]) -> str:
        """Read this document's glossary definitions and make them its stored ones.

        **Unconditionally a replace, including with an empty list.** A document that used to
        define three terms and now defines none has to end up with none — writing only when
        something was found would leave the old three answering queries, cited to a page that
        no longer says them. That is the failure this whole feature could most easily
        introduce: a definition that is wrong, confident, and looks exactly like a right one.

        **The write carries its own lineage**, so the entries and the statement of what produced
        them are one transaction. That is what makes an empty glossary a derived result rather
        than an absence: a document with no entries and a recorded fingerprint has been read by
        the current detector, and one with no entries and no fingerprint has not.

        **A detector failure fails closed and says so.** ``detect_entries`` is regular
        expressions over lines and has no model to be unavailable, so it raising means a bug in
        this repository — and the two things that must not happen then are the two that would be
        automatic. Nothing is written, so the entries a working detector produced stay exactly
        where they are and stay servable; and the fingerprint is not advanced, so this document
        remains selected by ``document reindex --stale-glossary`` and by ``doctor`` until the
        fix ships. The rest of the ingest is unaffected: chunks, vectors and the other three
        lineages are this document's index, and a glossary bug is not allowed to cost them.

        Returns:
            Why detection did not produce this document's entries, or the empty string when it
            did — or when there was nothing for it to do.
        """
        if self._glossary is None or self._glossary_lineage is None:
            return ""
        if not self._detect_glossary:
            # Detection is switched off. The rows are left exactly as they are, which is what
            # `rag.glossary.detect_on_ingest` promises an operator investigating a detector that
            # is producing rubbish — and the lineage records that no detector ran, which is a
            # value in the column rather than an absence somebody has to interpret. Switching
            # detection back on changes the installed fingerprint, so every document stamped
            # this way is selected by the next survey.
            await self._store.set_lineage(
                document.id, chunk_fp=None, embed_fp=None, glossary_fp=self._glossary_lineage
            )
            return ""
        try:
            entries = detect_entries(chunks, title=document.title, media_type=document.media_type)
        except Exception as exc:  # noqa: BLE001 - a detector bug costs this document's glossary
            return f"glossary detection failed: {type(exc).__name__}: {exc}"
        await self._glossary.replace_glossary_entries(
            document.id, entries, fingerprint=self._glossary_lineage
        )
        return ""

    # --- change detection ------------------------------------------------------------------

    def _unchanged_by_token(self, existing: Document | None, discovered: DiscoveredDoc) -> bool:
        """Level 1: the source says nothing changed, and we believe it without fetching.

        The token is opaque and connector-defined — a git blob SHA, a Confluence version
        number, an S3 ETag. It is compared for equality and never interpreted: no ordering, no
        parsing, no "is this newer". A connector wanting ordering implements it in ``discover``.

        Only a **settled** document may skip. One requeued after a crash carries a token and a
        hash from an ingest that never finished, and skipping on those would strand it forever.

        **Routing is checked here and not only in :meth:`changes_since`**, for the same reason
        lineage is. A well-behaved connector reports an unchanged token for a page nobody has
        edited, so this level answers and the fetch never happens — and a check placed only at
        level 2 would never run at all on exactly the corpora that are best behaved. Discovery
        already carries the declared media type, so this costs no fetch to answer.
        """
        return (
            existing is not None
            and existing.status in SETTLED
            and discovered.version_token is not None
            and existing.version_token == discovered.version_token
            and self._parse_lineage_is_current(existing)
            and self._routing_is_current(existing, discovered.media_type)
        )

    def _unchanged_by_hash(
        self, existing: Document | None, digest: str, raw: RawDocument | None = None
    ) -> bool:
        """Level 2: the bytes are identical, whatever the source claimed.

        Level 1 can lie — a source that touches its modification date on every save reports a
        new token for an unchanged body — and this catches it before the expensive part, which
        is parse, chunk and embed rather than the fetch.

        ``raw`` is the fetched document, so that the source record can be compared as well; it is
        optional only for callers that have no fetch in hand, and those cannot have a changed
        record either.

        Expressed in terms of :meth:`changes_since` rather than repeating its comparisons, so that
        "what changed" and "may this be skipped" cannot come to disagree — and so the classifier is
        exercised by every ingest rather than being a reporting path nobody runs.
        """
        return (
            existing is not None
            and existing.status in SETTLED
            and not self.changes_since(existing, digest, raw)
        )

    def changes_since(
        self, existing: Document, digest: str, raw: RawDocument | None = None
    ) -> frozenset[Change]:
        """What differs between the stored document and what was just fetched.

        **Content and metadata are detected independently**, which is a requirement rather than a
        nicety: they change independently and they cost different things to repair. A body edited
        with its manifest untouched needs a re-parse; a manifest corrected over an unchanged body
        needs only the record rewritten, and reports a version the reader can trust. Collapsing
        them into one boolean answers "should this be re-ingested" and destroys the answer to
        "why", which is the question an operator staring at a re-ingest of ten thousand pages
        actually has.

        Returns an empty set when nothing changed, which is the skip condition — so the set is
        load-bearing rather than diagnostic, and cannot rot into a description of a decision made
        somewhere else.
        """
        found: set[Change] = set()
        if existing.content_hash != digest:
            found.add(Change.CONTENT)
        if not self._source_record_is_current(existing, raw):
            found.add(Change.METADATA)
        if not self._parse_lineage_is_current(existing):
            found.add(Change.LINEAGE)
        if not self._routing_is_current(existing, raw.media_type if raw is not None else None):
            found.add(Change.ROUTING)
        return frozenset(found)

    @staticmethod
    def _routing_is_current(existing: Document, declared: str | None) -> bool:
        """Whether what the source declares now still reaches the parser that produced the text.

        One helper for both levels, so "may this be skipped" cannot come to disagree with "what
        changed" — the same reason :meth:`_unchanged_by_hash` is expressed in terms of
        :meth:`changes_since`. Level 1 has the declaration from discovery and level 2 from the
        fetch; neither has to interpret it, because equality is the whole question.

        **No declaration is agreement, not ignorance.** A connector that supplies no media type
        at discovery, or a caller with no fetch in hand, has said nothing about routing — and
        treating silence as a change would re-ingest every such corpus on every sync to learn
        nothing, which is the failure the same rule avoids in :meth:`_source_record_is_current`.
        """
        return declared is None or existing.media_type == declared

    @staticmethod
    def _source_record_is_current(existing: Document, raw: RawDocument | None) -> bool:
        """Whether the stored source record is the one this fetch just brought back.

        **Identical bytes are not an unchanged document when the metadata is what moved**, and
        this is the same trap ``_parse_lineage_is_current`` exists for, one field along. A
        mirrored page whose manifest is corrected — a title fixed, a new source version declared
        — has byte-for-byte the same body, so the hash agrees and level 2 skips. Worse than
        merely skipping: the skip path calls ``record_seen`` with the *new* version token, so the
        corrected record is never read again on any later sync either. The corpus cites a version
        it was told about and then declined to look at, and nothing anywhere reports a problem.

        Compared through the validating accessor on both sides, so an unusable record on either
        one reads as absent and compares equal to another absent one — which is right: two
        documents about which nothing authoritative is known are not different documents.

        A fetch that brings no record leaves a stored one alone rather than counting as a change,
        matching the assignment rule in ``_store_record``. Otherwise a connector that supplies
        metadata on only some paths would re-ingest its whole corpus on every run.
        """
        if raw is None:
            return True
        fresh = Provenance.from_metadata(raw.metadata)
        if fresh is None:
            return True
        return fresh == existing.provenance

    def _parse_lineage_is_current(self, existing: Document) -> bool:
        """Whether the stored text was produced by the parser version installed now.

        **Both change-detection levels ask this, and both have to.** Unchanged bytes are the
        reason a parser bump is silent: ``content_hash`` is taken over what the connector
        returned, a new ``pypdfium2`` does not touch those bytes, so nothing already stored is
        ever re-read — while a newly ingested document with *identical* bytes parses
        differently, and the corpus ends up holding two generations of extracted text. Level 1
        needs it as much as level 2 and for a sharper reason: a source whose version token has
        not moved never even reaches the hash comparison, so a check placed only at level 2
        would leave every well-behaved connector's corpus permanently stale.

        The comparison is against **the parser this document actually used**, read from
        ``parser_used`` in its metadata. Not against every installed parser: a ``pypdfium2``
        bump must re-parse the PDFs and leave the Markdown alone, which is the whole of what
        ``docs/storage.md`` §6.4 calls set-valued invalidation.

        ``None`` on both sides is agreement rather than ignorance, and the two ways it arises
        both mean "no version to compare". A parser manicule does not ship has no version this
        repository can read, so nothing was recorded and nothing is expected; a document that
        never reached a parser at all — excluded by middleware, claimed by no chain — has no
        parse to be stale. Treating either as changed would re-parse a plugin corpus on every
        sync, forever, to learn nothing.
        """
        return existing.parse_fp == self._parse_lineage_of(existing)

    def _parse_lineage_of(self, document: Document) -> str | None:
        """The canonical parse fingerprint this document's parser would produce today.

        ``None`` where there is no version to read: no parser ran, the one that did is not one
        manicule ships and therefore not one it can version, or its library has since been
        uninstalled. Recording ``None`` leaves any stored lineage untouched, which is what
        makes this safe to call on every commit.

        **An uninstalled library is caught here rather than allowed to end a run.**
        :func:`~manicule.parsers.versions.parse_fingerprint` raises for a distribution that is
        not present, which is right where a repair is being planned — a partial set of current
        fingerprints is a repair that cannot succeed. It is wrong here. This runs inside
        discovery, once per document, and an exception escaping it would abort the enumeration
        and take every *other* document in the batch with it, which is the one failure mode
        this pipeline exists not to have. Answering ``None`` instead makes the document
        ineligible for a skip, so the chain runs, fails on the missing library alone, and is
        recorded against that document.

        **Written only where the stored content is now the output of that parse**, which is
        narrower than "wherever a parse happened" and deliberately so. A document that parsed
        cleanly and then failed to embed keeps its previous chunks — the pipeline refuses to
        demote a working document — so recording the new fingerprint there would claim the
        stored text is current when it is a generation behind, and the next sync, seeing the
        lineage agree, would skip it forever. That is the silent inconsistency this field
        exists to end, arriving through the field itself.
        """
        used = document.metadata.get("parser_used")
        if not isinstance(used, str) or not used:
            return None
        try:
            current = self._parse_fingerprints(used)
        except PackageNotFoundError:
            return None
        return current.canonical() if current is not None else None

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
        expected: DocumentRevision | None = None,
    ) -> Document:
        """Write the document row for whatever the chain concluded.

        **A chunk-less terminal status still stores the document.** Storing the failure is what
        makes it re-queryable, skippable on the next sync, and reachable by a re-parse the day
        the missing capability arrives. An unstored failure is re-fetched on every sync, absent
        from every listing, and invisible to any repair.

        **The first write a document gets, and therefore the first place ``expected`` is
        verified.** Not because one guard here would be enough — the document can still move
        between this and the commit, which is why :meth:`_commit` carries the same guard — but
        because failing here is what keeps a superseded re-parse from ever *producing* a chunk,
        a glossary row or a vector. The spec's alternative, reconciling stale derived state
        after the fact, is where the second race lives.

        Raises:
            _SupersededError: The stored document is no longer ``expected``, so nothing was written.
        """
        if existing is not None and self._keeps_status(existing, result.status):
            # A failed parse reached no conclusion about the newly fetched bytes. Preserve the
            # indexed source revision as one indivisible snapshot: adopting even its new token
            # would make the next sync skip bytes this document does not hold. The error is the
            # only fact this attempt established, so it is the only field merged into the row.
            metadata = dict(existing.metadata)
            metadata["last_ingest_error"] = {
                "stage": PipelineStage.PARSE.value,
                "detail": result.status_detail,
            }
            return await self._publish(
                existing.model_copy(update={"metadata": metadata}), expected=expected
            )

        # **The connector's own metadata reaches the document, and it is not decoration.** The
        # chunker builds its breadcrumb from `document.metadata["ancestors"]`, so a pipeline that
        # dropped what the connector attached to the fetched bytes would leave every breadcrumb
        # empty — and an empty breadcrumb is not a visible failure, it is a section called
        # "Configuration" that nobody can retrieve.
        #
        # **What a connector just fetched beats what was stored last time, and the parse stage
        # beats both.** The order used to put `existing.metadata` over `raw.metadata`, which
        # protected accumulated per-document state at the cost of freezing everything a connector
        # re-derives on every fetch. Two instances made the cost concrete: a source record whose
        # version had moved, and a page's labels and content status. In each case the stored copy
        # won, so the fact never updated and nothing looked wrong.
        #
        # The failure shape is what decided it. This is not a document that looks stale — it is a
        # document that looks **freshly synced** while carrying superseded facts, because the
        # fields that do refresh (its version token, its content hash) sit beside the ones that do
        # not. A page archived and deprecated at the source reads as `current` for ever, and an
        # operator filtering on that gets a retired runbook back under a version number asserting
        # the sync is up to date.
        #
        # Accumulated state still survives, and for a reason that needs no special case: a key
        # absent from `raw.metadata` overrides nothing. `annotate` writes
        # `last_after_store_error` and `last_ingest_error`, which no connector supplies, so they
        # are untouched by the reorder.
        metadata: Metadata = {
            **(existing.metadata if existing else {}),
            **dict(raw.metadata),
            **result.metadata,
        }
        # **What a citation shows comes from the record when there is one.** Read back out of
        # `metadata` rather than from `fresh`, so that the record which decides the citation is
        # by construction the record that gets stored — reading one and storing the other is how
        # a corpus ends up citing a title nothing in it holds. The local facts are not lost:
        # `source_id` is still the path this connector fetched by, `content_hash` still digests
        # these bytes, and the snapshot's location is in the record's own snapshot half.
        # `raw.uri` is deliberately left alone upstream of here, so on the **fetch** path a
        # parser's diagnostics still name the artifact it actually read rather than a web page
        # nobody can open locally. That is a property of this path only, and the exception is
        # worth stating rather than discovering: `reindex.re_parse` rebuilds a `RawDocument` from
        # `document.uri`, which by then *is* the canonical address — so a re-parse diagnostic
        # names the document rather than the bytes. Defensible, because on that path the bytes
        # came from the blob store and neither URI describes where they were read from, but not
        # the same claim.
        record = Provenance.from_metadata(metadata)
        canonical = record.source if record is not None else None
        document = Document(
            id=identifier,
            source=source,
            source_id=raw.source_id,
            uri=(canonical.canonical_uri if canonical and canonical.canonical_uri else raw.uri),
            title=(
                canonical.title
                if canonical and canonical.title
                else title or (existing.title if existing else "")
            ),
            content_hash=digest,
            version_token=version_token,
            original_ref=retention.ref,
            media_type=raw.media_type,
            status=result.status,
            status_detail=result.status_detail or None,
            failed_stage=result.failed_stage,
            metadata=metadata,
        )
        stored = await self._publish(document, expected=expected)
        await self._store.set_original(
            stored.id, ref=retention.ref, omitted_reason=retention.omitted_reason
        )
        if result.status in {DocumentStatus.CONTAINER, DocumentStatus.NO_EXTRACTABLE_TEXT}:
            await self._store.replace_chunks(stored.id, [])
        return stored

    @staticmethod
    def _keeps_status(existing: Document, proposed: DocumentStatus) -> bool:
        """Whether an indexed document keeps its status rather than taking the new one.

        Only a ``failed`` outcome is refused, and only for a document that is currently
        servable. Everything else is a conclusion about the new bytes rather than a failure to
        reach one.
        """
        return existing.status is DocumentStatus.INDEXED and proposed is DocumentStatus.FAILED

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
        expected: DocumentRevision | None = None,
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
            expected=expected,
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


def _report_progress(run: _Sync) -> None:
    """Tell the watcher where the run has got to, if anybody is watching.

    **One function for both stages**, because a document that skips is reported from the fetch
    stage and a document that lands is reported from the ingest stage — and two call sites that
    worded the same thing differently would make a sync's output change shape halfway through
    depending on what the corpus happened to contain.

    Counters rather than a narration of one document. The number somebody wants from a long sync
    is how far through it is, and counters make each line **supersede** the one before it, which
    is what lets a caller show only the newest.
    """
    if run.watching is None:
        return
    unchanged = run.report.skipped_version + run.report.skipped_hash
    settled = run.report.indexed + unchanged
    run.watching(
        f"{run.connector.name}: {settled} of {run.report.discovered} settled "
        f"({run.report.indexed} indexed, {unchanged} unchanged)"
    )


def _first_failure(failures: BaseExceptionGroup[Exception]) -> str:
    """One line for what stopped the stages, from however many of them noticed.

    Several ingest workers meeting the same dead store produce several identical exceptions, and
    a report that concatenated them would say the same sentence four times and bury how many
    stages were affected. The first is named because it is the one that happened; the count is
    kept because "four stages" and "one stage" are different diagnoses.
    """
    flattened = _leaves(failures)
    first = flattened[0]
    detail = f"{type(first).__name__}: {first}"
    return detail if len(flattened) == 1 else f"{detail} (and {len(flattened) - 1} more)"


def _leaves(failures: BaseExceptionGroup[Exception]) -> list[Exception]:
    """Every exception in a group, however deeply the task group nested them."""
    found: list[Exception] = []
    for failure in failures.exceptions:
        if isinstance(failure, BaseExceptionGroup):
            found.extend(_leaves(failure))
        else:
            found.append(failure)
    return found


def _writer_of(store: object) -> GlossaryWriter | None:
    """The store itself, when it can hold glossary entries.

    Structural rather than configured, and taken from the store the pipeline was already given
    rather than accepted as a second handle. Two handles onto one database is how a definition
    ends up committed against a workspace the document is not in — the store carries the
    workspace, so taking the writer from it makes the two agree by construction.
    """
    return store if isinstance(store, GlossaryWriter) else None


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
    "Change",
    "DocumentOutcome",
    "IngestPipeline",
    "NoRetention",
    "RunReport",
]
