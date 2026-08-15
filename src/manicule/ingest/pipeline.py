"""discover → fetch → parse → chunk → embed → store, several documents at a time.

**The unit of work is one document. A batch is a scheduling artifact with no semantics of its
own.** That is what makes "one bad document never aborts a batch" a structural property rather
than a promise: there is no batch-level transaction to abort, and no batch-level state a
document can corrupt. Every failure this module catches is attributed to a document, recorded,
and left behind.

**With acquisition journaling enabled, a run enumerates first, then has three local stages joined
by bounded hand-offs.** Each source identity is committed to the journal before discovery
advances. A bounded journal reader then fills the fetch hand-off; ``fetch_concurrency`` fetch
workers fill the next; and ``parse_workers + 1`` ingest workers carry documents the rest of the
way. The compatibility fallback feeds its bounded fetch hand-off directly from discovery.
Neither mode gathers a task or an in-memory record per document — see
:meth:`IngestPipeline.run` and ``docs/ingest.md`` §8.3.

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

The publication order and its crash windows belong to ``docs/storage.md`` §8.2 and are honored
rather than restated: versioned vectors are staged first, then one SQLite transaction flips the
document, chunks, glossary and lineage from the old publication to the new one.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from importlib.metadata import PackageNotFoundError
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable
from uuid import uuid4

from manicule.connectors.errors import (
    BodyUnavailableError,
    NotFoundError,
    RemoteError,
    SessionExpiredError,
)
from manicule.core.acquisition import (
    AcquiredSource,
    AcquisitionDiagnostic,
    AcquisitionFailureCode,
    AcquisitionFence,
    AcquisitionRecord,
    AcquisitionRecordState,
    AcquisitionRunState,
    AcquisitionSource,
    AcquisitionStage,
    SnapshotCompleteness,
    SnapshotPromotionPolicy,
)
from manicule.core.content import (
    SETTLED,
    Document,
    DocumentRevision,
    DocumentStatus,
    PipelineStage,
    RawDocument,
    Retention,
)
from manicule.core.embedding import canonical_stored_vector
from manicule.core.errors import (
    AcquisitionLeaseLostError,
    ChunkingError,
    ContextOverflowError,
    MiddlewareViolationError,
)
from manicule.core.ids import content_hash, document_id
from manicule.core.provenance import Provenance
from manicule.core.sources import DiscoveredDoc
from manicule.ingest.embedding import EmbeddingWork, embed_or_reuse
from manicule.ingest.glossary import detect_entries
from manicule.ingest.glossary_lineage import glossary_fingerprint
from manicule.ingest.ports import AcquisitionStore, FencedIngestStore, GlossaryWriter
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
    from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine, Sequence

    from manicule.core.content import Chunk, Metadata
    from manicule.core.fingerprints import ChunkFingerprint, ParseFingerprint
    from manicule.core.glossary import GlossaryEntry
    from manicule.core.protocols import Chunker, Connector, Embedder, VectorStore
    from manicule.core.sources import DocRef, Watermark
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

type _PublicationFence = Callable[[], Awaitable[AcquisitionFence]]


_publication_fence: ContextVar[_PublicationFence | None] = ContextVar(
    "acquisition_publication_fence", default=None
)


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


class _AcquisitionRetentionError(RuntimeError):
    """Fetched bytes could not become a durable snapshot input."""


class _MissingAcquisitionSnapshotError(RuntimeError):
    """A journal record no longer has the local snapshot it promises."""


class _CorruptAcquisitionSnapshotError(RuntimeError):
    """A journal-owned local snapshot failed integrity or envelope validation."""


class _UnprovenSourceRevisionError(RuntimeError):
    """Fetched bytes do not prove they are at least the discovered source revision."""


def _acquisition_diagnostic(exc: Exception) -> AcquisitionDiagnostic:
    """Reduce source failures to a bounded, non-sensitive persisted vocabulary."""
    if isinstance(exc, SessionExpiredError) or (
        isinstance(exc, RemoteError) and exc.status_code in {401, 403}
    ):
        code = AcquisitionFailureCode.AUTHENTICATION
    elif isinstance(exc, NotFoundError):
        code = AcquisitionFailureCode.SOURCE_DELETED
    elif isinstance(exc, _UnprovenSourceRevisionError):
        code = AcquisitionFailureCode.STALE_BODY
    elif isinstance(exc, BodyUnavailableError):
        code = (
            AcquisitionFailureCode.STALE_BODY
            if "stale" in str(exc).lower() or "discovered at version" in str(exc).lower()
            else AcquisitionFailureCode.MISSING_BODY
        )
    elif isinstance(exc, _AcquisitionRetentionError):
        code = AcquisitionFailureCode.CAPACITY
    else:
        code = AcquisitionFailureCode.FETCH_FAILED
    return AcquisitionDiagnostic(stage=AcquisitionStage.ACQUISITION, code=code)


def _snapshot_diagnostic(exc: Exception) -> AcquisitionDiagnostic:
    """Classify failures reopening already-acquired bytes as local indexing failures."""
    code = (
        AcquisitionFailureCode.SNAPSHOT_MISSING
        if isinstance(exc, _MissingAcquisitionSnapshotError)
        else AcquisitionFailureCode.SNAPSHOT_CORRUPT
    )
    return AcquisitionDiagnostic(stage=AcquisitionStage.INDEXING, code=code)


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

    async def retain_acquisition(
        self, key: str, raw: RawDocument
    ) -> tuple[Retention, AcquiredSource]: ...

    async def resume_acquisition(self, key: str) -> tuple[Retention, AcquiredSource] | None: ...

    async def complete_acquisition(self, key: str) -> None: ...

    async def reconcile_acquisition_markers(self) -> bool: ...


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

    async def retain_acquisition(
        self, key: str, raw: RawDocument
    ) -> tuple[Retention, AcquiredSource]:
        del key
        return await self.retain(raw.as_bytes(), raw.media_type), AcquiredSource.from_raw(raw)

    async def resume_acquisition(self, key: str) -> tuple[Retention, AcquiredSource] | None:
        del key
        return None

    async def complete_acquisition(self, key: str) -> None:
        del key

    async def reconcile_acquisition_markers(self) -> bool:
        return True


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
    failed_stage: PipelineStage | None = None
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
    error_type: str = ""
    error_message: str = ""

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

    enumeration_completed: bool = True
    """Whether discovery exhausted the source rather than stopping or raising."""

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

    watermark_advanced: bool = False
    """Whether this run persisted a new connector position."""

    pending_derivation: bool = False
    """Whether retained local work remains after this invocation returned normally."""

    snapshot_completeness: Literal["", "complete", "partial"] = ""
    snapshot_omissions: int = 0
    snapshot_omission_reasons: dict[str, int] = field(default_factory=dict[str, int])

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
        return not self.retry_required and not self.limited

    @property
    def retry_required(self) -> bool:
        """Whether repeating this operation is required to finish its durable work."""
        strict_snapshot_incomplete = (
            self.snapshot_omissions > 0 and self.snapshot_completeness != "partial"
        )
        return bool(
            self.error or self.unrecorded or self.pending_derivation or strict_snapshot_incomplete
        )

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
                "error_type": self.error_type,
                "error_message": self.error_message,
                "limited": self.limited,
                "unrecorded": self.unrecorded,
                "outcome": (
                    "incomplete"
                    if self.retry_required
                    else "bounded"
                    if self.limited
                    else "complete"
                ),
                "enumeration_completed": self.enumeration_completed,
                "watermark_advanced": self.watermark_advanced,
                "snapshot_completeness": self.snapshot_completeness,
                "snapshot_omissions": self.snapshot_omissions,
                "snapshot_omission_reasons": dict(self.snapshot_omission_reasons),
                "retry_required": self.retry_required,
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
    acquisition_record: AcquisitionRecord | None = None
    retention: Retention | None = None
    force: bool = False


@dataclass(frozen=True, slots=True)
class _AcquisitionStart:
    run_id: str = ""
    owner: str = ""
    generation: int = 0
    expires_at: datetime | None = None
    resume_completed: bool = False
    candidate_watermark: Watermark | None = None
    accepted: int = 0
    watermark: Watermark | None = None
    state: AcquisitionRunState | None = None
    source_scope: str = ""
    scope_fingerprint: str = ""
    promotion_policy: SnapshotPromotionPolicy = SnapshotPromotionPolicy.REQUIRE_COMPLETE
    completeness: SnapshotCompleteness | None = None
    omission_count: int = 0
    omission_reasons: dict[AcquisitionFailureCode, int] = field(
        default_factory=dict[AcquisitionFailureCode, int]
    )


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
    refs: Conveyor[DiscoveredDoc | AcquisitionRecord]
    """Journal reader to fetch. Carries references, so depth costs metadata rather than bodies."""

    bodies: Conveyor[_Fetched]
    """Fetch to ingest. Carries bytes, so its depth is what bounds a run's memory."""

    acquisitions: AcquisitionStore | None = None
    acquisition_run_id: str = ""
    lease_owner: str = ""
    lease_generation: int = 0
    lease_expires_at: datetime | None = None
    resume_completed: bool = False
    candidate_watermark: Watermark | None = None
    acquisition_state: AcquisitionRunState | None = None
    source_scope: str = ""
    scope_fingerprint: str = ""
    promotion_policy: SnapshotPromotionPolicy = SnapshotPromotionPolicy.REQUIRE_COMPLETE
    snapshot_completeness: SnapshotCompleteness | None = None
    snapshot_omission_count: int = 0
    snapshot_omission_reasons: dict[AcquisitionFailureCode, int] = field(
        default_factory=dict[AcquisitionFailureCode, int]
    )
    reusable_snapshot_checked: bool = False
    reusable_snapshot_run_id: str | None = None
    discovery_records_held: Gauge = field(default_factory=lambda: Gauge("discovery-records"))

    watching: Watching | None = None
    """Where to say what has happened so far, or ``None`` when nobody is watching."""

    accepted: int = 0
    """Top-level documents committed by discovery. What ``--limit`` bounds, counted where the
    bound is applied rather than derived afterwards from what finished."""

    bodies_held: Gauge = field(default_factory=lambda: Gauge("bodies"))
    """Fetched bodies in memory: queued, and held by an ingest worker.

    **Per run rather than on the pipeline**, unlike the parse and embed gauges. Those count
    process-wide resources — one pool, one accelerator — and a second operation sharing this
    pipeline genuinely is inside them. A body belongs to the run that fetched it, and a run that
    was canceled with items still queued would otherwise leave a count nothing will ever release,
    inflating every later run's report by a number that only ever grows.
    """

    stop: asyncio.Event = field(default_factory=asyncio.Event)
    """Set on cancellation so discovery or journal reading stops admitting new work."""


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
        acquisitions: AcquisitionStore | None = None,
        acquisition_lease_s: float = 300.0,
        acquisition_clock: Callable[[], datetime] | None = None,
        acquisition_history_s: float = 30 * 24 * 3600.0,
        acquisition_cleanup_batch: int = 100,
        snapshot_policy: SnapshotPromotionPolicy = SnapshotPromotionPolicy.REQUIRE_COMPLETE,
    ) -> None:
        # Second of the two places this is refused, and not a redundant one.
        # `check_before_run` is the once-per-run boundary and is what an operator meets; this
        # one is the boundary in *code*, because a pipeline is constructible without going
        # through that function and everything it writes is permanent. A chunker counting with
        # a stand-in vocabulary must not be able to reach a store at all.
        require_measured(chunk_fingerprint)
        self._store = store
        # Explicit protocol injection keeps the legacy bounded path available to protocol-only
        # stores and unit tests. Production wires the SQLite acquisition store here, making the
        # blob-backed snapshot path the normal connector topology rather than structural guesswork.
        self._acquisitions = acquisitions
        self._fenced_store = (
            store if acquisitions is not None and isinstance(store, FencedIngestStore) else None
        )
        if acquisitions is not None and self._fenced_store is None:
            msg = "durable acquisition requires transaction-fenced document publication"
            raise TypeError(msg)
        self._acquisition_lease_s = max(1.0, acquisition_lease_s)
        self._acquisition_clock = acquisition_clock or (lambda: datetime.now(UTC))
        self._acquisition_history_s = max(0.0, acquisition_history_s)
        self._acquisition_cleanup_batch = max(1, acquisition_cleanup_batch)
        self._snapshot_policy = snapshot_policy
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

    async def _start_acquisition(
        self, connector: Connector, watermark: Watermark | None
    ) -> _AcquisitionStart:
        acquisitions = self._acquisitions
        if acquisitions is None:
            return _AcquisitionStart(watermark=watermark)
        source_scope, scope_fingerprint = _snapshot_scope(connector)
        owner = f"pipeline:{uuid4().hex}"
        now = self._acquisition_clock()
        claimed = await acquisitions.claim_or_create_acquisition_run(
            connector.name,
            uuid4().hex,
            owner,
            source_scope=source_scope,
            scope_fingerprint=scope_fingerprint,
            promotion_policy=self._snapshot_policy,
            now=now,
            expires_at=now + timedelta(seconds=self._acquisition_lease_s),
        )
        if claimed is None:
            msg = f"an active acquisition run for connector {connector.name!r} could not be claimed"
            raise RuntimeError(msg)
        if claimed.state not in {
            AcquisitionRunState.ENUMERATING,
            AcquisitionRunState.ACQUIRING,
            AcquisitionRunState.INDEXING,
        }:
            msg = f"acquisition run {claimed.id!r} is not resumable from {claimed.state}"
            raise RuntimeError(msg)
        return _AcquisitionStart(
            run_id=claimed.id,
            owner=owner,
            generation=claimed.lease_generation,
            expires_at=claimed.lease_expires_at,
            resume_completed=claimed.enumeration_completed_at is not None,
            candidate_watermark=claimed.candidate_watermark,
            accepted=claimed.discovered_count,
            watermark=(
                claimed.base_watermark
                if claimed.base_watermark_scope_fingerprint == claimed.scope_fingerprint
                else None
            ),
            state=claimed.state,
            source_scope=claimed.source_scope,
            scope_fingerprint=claimed.scope_fingerprint,
            promotion_policy=claimed.promotion_policy,
            completeness=claimed.completeness,
            omission_count=claimed.omission_count,
            omission_reasons=dict(claimed.omission_reasons),
        )

    async def run(
        self,
        connector: Connector,
        *,
        limit: int | None = None,
        watching: Watching | None = None,
    ) -> RunReport:
        """Ingest everything a connector reports as changed since its watermark.

        **Durable enumeration, then three bounded local stages:**

        .. code-block:: text

            discover → committed acquisition journal
               │  bounded journal reader
               │  fetch hand-off, depth = queue_depth_factor x fetch_concurrency
               ▼
            fetch x fetch_concurrency          change detection, then the network
               │  parse hand-off, depth = queue_depth_factor x ingest workers
               ▼
            ingest x (parse_workers + 1)       parse in the pool, chunk, embed under
                                               one lock, commit under the document's

        Discovery acknowledges one identity only after its journal transaction commits. Local
        fetching, parsing and embedding begin after enumeration, so their backpressure cannot
        age a live source cursor. The journal reader and both local hand-offs remain bounded, so
        decoupling cursor lifetime does not turn into corpus-sized memory (``docs/ingest.md``
        §8.3).

        **What the concurrency does not change.** Each document still travels the same path it
        did one at a time, and the per-document lock still spans its record, chunks, glossary
        and vectors. Two documents may be in the ingest stage at once; one document is never in
        two places.

        **``limit`` bounds acceptance, not completion.** Discovery stops after committing
        ``limit`` top-level journal records, and everything already accepted is carried to a
        terminal outcome before this returns. Members found inside a container do not count
        against it — one archive of five hundred files must not exhaust a limit of ten. A run
        stopped this way is *bounded*: no error, no completion marker, and no watermark.

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
        if self._acquisitions is not None:
            # Settled journal rows are diagnostic history, not recovery input. Bound this pass
            # so starting one connector cannot monopolize the workspace writer. Active work on
            # the authoritative run never ages out; superseded work may, because its incremented
            # generation makes it impossible to resume. Blob collection remains its own
            # mark-and-sweep operation.
            now = self._acquisition_clock()
            if await self._blobs.reconcile_acquisition_markers():
                await self._acquisitions.cleanup_acquisition_history(
                    now - timedelta(seconds=self._acquisition_history_s),
                    limit=self._acquisition_cleanup_batch,
                )
            _, scope_fingerprint = _snapshot_scope(connector)
            watermark = await self._acquisitions.get_acquisition_watermark(
                connector.name, scope_fingerprint
            )
        else:
            watermark = await self._store.get_watermark(connector.name)
        acquisition = await self._start_acquisition(connector, watermark)

        ref_capacity = self._queue_depth_factor * self._fetch_workers
        run = _Sync(
            connector=connector,
            report=RunReport(connector=connector.name),
            limit=limit,
            watching=watching,
            watermark=acquisition.watermark,
            refs=Conveyor(
                name="fetch",
                capacity=ref_capacity,
                consumers=self._fetch_workers,
            ),
            bodies=Conveyor(
                name="parse",
                capacity=self._queue_depth_factor * self._ingest_workers,
                consumers=self._ingest_workers,
                producers=self._fetch_workers,
            ),
            acquisitions=self._acquisitions,
            acquisition_run_id=acquisition.run_id,
            lease_owner=acquisition.owner,
            lease_generation=acquisition.generation,
            lease_expires_at=acquisition.expires_at,
            resume_completed=acquisition.resume_completed,
            candidate_watermark=acquisition.candidate_watermark,
            acquisition_state=acquisition.state,
            source_scope=acquisition.source_scope,
            scope_fingerprint=acquisition.scope_fingerprint,
            promotion_policy=acquisition.promotion_policy,
            snapshot_completeness=acquisition.completeness,
            snapshot_omission_count=acquisition.omission_count,
            snapshot_omission_reasons=dict(acquisition.omission_reasons),
            accepted=acquisition.accepted,
        )
        if run.snapshot_completeness is not None:
            run.report.snapshot_completeness = run.snapshot_completeness.value
            run.report.snapshot_omissions = run.snapshot_omission_count
            run.report.snapshot_omission_reasons = {
                code.value: count for code, count in run.snapshot_omission_reasons.items()
            }
        # Peaks only. The active counts belong to whoever is inside the stage right now, and a
        # second operation sharing this pipeline is one of them.
        for gauge in (self._fetches, self._parsing, self._embedding.gauge):
            gauge.rebase()

        stages = asyncio.create_task(self._drive(run), name=f"ingest:{connector.name}")
        crashed = False
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
            first, detail = _first_failure(failures)
            crashed = True
            run.report.error_type = type(first).__name__
            run.report.error_message = str(first)
            run.report.error = detail
        except BaseException:
            stages.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stages
            raise

        run.report.stages = self._stage_report(run)
        run.report.settle()
        if run.acquisitions is None and run.report.complete:
            try:
                run.report.watermark_advanced = await self._advance_watermark(run)
            except Exception as exc:  # noqa: BLE001 - a checkpoint failure makes this run retry
                run.report.error_type = type(exc).__name__
                run.report.error_message = str(exc)
                run.report.error = f"{type(exc).__name__}: {exc}"
        await self._record_run_completion(run, crashed=crashed)
        return run.report

    async def _record_run_completion(self, run: _Sync, *, crashed: bool) -> None:
        """Publish diagnostics only while this invocation still owns their durable order."""
        acquisitions = run.acquisitions
        if acquisitions is None:
            await self._store.record_connector_metadata(
                run.connector.name, run.report.as_metadata()
            )
            return
        recorded = True
        try:
            recorded = await acquisitions.record_acquisition_run_metadata(
                run.acquisition_run_id,
                run.lease_owner,
                run.lease_generation,
                now=self._acquisition_clock(),
                updates=run.report.as_metadata(),
                release=not crashed,
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics must not hide the run outcome
            if not crashed:
                run.report.error_type = type(exc).__name__
                run.report.error_message = str(exc)
                run.report.error = f"{type(exc).__name__}: {exc}"
        if not recorded and not crashed:
            msg = "the acquisition generation changed before diagnostics and release"
            run.report.error_type = AcquisitionLeaseLostError.__name__
            run.report.error_message = msg
            run.report.error = f"{AcquisitionLeaseLostError.__name__}: {msg}"

    async def _drive(self, run: _Sync) -> None:
        """Start every stage, and return when discovery is spent and every acceptance is done.

        One :class:`asyncio.TaskGroup`, so the stages are children of this call: it returns only
        when all of them have, and canceling it cancels and joins all of them. There is no
        detached task and nothing to leak.

        A stage task raising cancels its siblings, and that is deliberate rather than tolerated:
        everything a document can do is already an outcome by the time it reaches here, so what
        is left to raise is the store or a defect, and neither is survivable by carrying on.
        """
        if run.acquisitions is not None:
            await self._drive_durable(run)
            return
        producer = self._discover_into

        refs, bodies = run.refs, run.bodies
        await self._run_local_stages(run, producer, refs, bodies)

    async def _drive_durable(self, run: _Sync) -> None:  # noqa: PLR0912 - durable state machine
        """Acquire a source snapshot first, then derive only from retained local bytes."""
        acquisitions = run.acquisitions
        if acquisitions is None:  # pragma: no cover - selected structurally
            return
        await self._verify_resumed_snapshot(run, acquisitions)

        async def owned(work: Coroutine[object, object, None], name: str) -> None:
            async with asyncio.TaskGroup() as ownership:
                task = ownership.create_task(work, name=name)
                ownership.create_task(
                    self._heartbeat_acquisition_until_done(run, task), name="lease-heartbeat"
                )

        if run.acquisition_state is AcquisitionRunState.ENUMERATING:
            if run.resume_completed:
                run.report.enumeration_completed = True
            else:
                await owned(self._enumerate_to_journal(run), "journal-enumeration")
            if not run.report.enumeration_completed:
                # A bounded/incomplete run may still make its durable prefix useful, but it can
                # never publish the candidate watermark or claim complete source coverage.
                acquired = await self._owned_acquisition(run, self._acquire_journal(run))
                if run.stop.is_set():
                    return
                if not acquired:
                    await self._report_snapshot_omissions(run)
                await owned(self._index_acquired(run), "journal-indexing")
                await self._mark_pending_derivation(run, acquisitions)
                return
            run.acquisition_state = AcquisitionRunState.ACQUIRING
        else:
            run.report.enumeration_completed = True

        if run.acquisition_state is AcquisitionRunState.ACQUIRING:
            acquired = await self._owned_acquisition(run, self._acquire_journal(run))
            if run.stop.is_set():
                return
            if not acquired:
                await self._report_snapshot_omissions(run)
            if acquired or run.promotion_policy is SnapshotPromotionPolicy.ALLOW_OMISSIONS:
                now = self._acquisition_clock()
                await self._keep_acquisition_lease_live(run, acquisitions, now, force=True)
                await acquisitions.complete_snapshot_acquisition(
                    run.acquisition_run_id,
                    lease_owner=run.lease_owner,
                    lease_generation=run.lease_generation,
                    now=now,
                )
                promoted = await acquisitions.promote_snapshot_and_commit_watermark(
                    run.acquisition_run_id,
                    expected_scope_fingerprint=run.scope_fingerprint,
                    lease_owner=run.lease_owner,
                    lease_generation=run.lease_generation,
                    now=now,
                )
                run.report.watermark_advanced = promoted.watermark_committed_at is not None
                run.report.snapshot_completeness = (
                    "partial"
                    if promoted.completeness is SnapshotCompleteness.PARTIAL
                    else "complete"
                    if promoted.completeness is SnapshotCompleteness.COMPLETE
                    else ""
                )
                run.report.snapshot_omissions = promoted.omission_count
                run.report.snapshot_omission_reasons = {
                    code.value: count for code, count in promoted.omission_reasons.items()
                }
                await acquisitions.transition_acquisition_run(
                    run.acquisition_run_id,
                    AcquisitionRunState.ACQUIRING,
                    AcquisitionRunState.INDEXING,
                    lease_owner=run.lease_owner,
                    lease_generation=run.lease_generation,
                    now=now,
                )
                run.acquisition_state = AcquisitionRunState.INDEXING

        await owned(self._index_acquired(run), "journal-indexing")
        if run.acquisition_state is AcquisitionRunState.INDEXING and not run.stop.is_set():
            if await self._mark_pending_derivation(run, acquisitions):
                # A local document failure is durable work, not permission to call the source
                # again and not a run-level crash. Leave INDEXING visible and let takeover retry
                # the retained snapshot on the next invocation.
                return
            now = self._acquisition_clock()
            await self._keep_acquisition_lease_live(run, acquisitions, now, force=True)
            await acquisitions.transition_acquisition_run(
                run.acquisition_run_id,
                AcquisitionRunState.INDEXING,
                AcquisitionRunState.SETTLED,
                lease_owner=run.lease_owner,
                lease_generation=run.lease_generation,
                now=now,
            )

    async def _mark_pending_derivation(self, run: _Sync, acquisitions: AcquisitionStore) -> bool:
        """Expose work that can continue from retained evidence without source contact."""
        run.report.pending_derivation = False
        after: int | None = None
        while True:
            records = await acquisitions.list_acquisition_records(
                run.acquisition_run_id,
                states=(
                    AcquisitionRecordState.ACQUIRED,
                    AcquisitionRecordState.INDEXING,
                    AcquisitionRecordState.RETRY,
                ),
                after_sequence=after,
                limit=100,
            )
            if not records:
                return False
            for record in records:
                local_retry = record.diagnostic is not None and (
                    record.diagnostic.stage is AcquisitionStage.INDEXING
                )
                retained = record.blob_ref is not None or record.acquired_source is not None
                if record.state is not AcquisitionRecordState.RETRY or retained or local_retry:
                    run.report.pending_derivation = True
                    return True
            after = records[-1].sequence

    async def _report_snapshot_omissions(self, run: _Sync) -> None:
        """Expose a bounded, typed aggregate even when strict policy refuses promotion."""
        acquisitions = run.acquisitions
        if acquisitions is None:  # pragma: no cover - journal path only
            return
        after: int | None = None
        omissions = 0
        reasons: dict[str, int] = {}
        while True:
            records = await acquisitions.list_acquisition_records(
                run.acquisition_run_id,
                after_sequence=after,
                limit=100,
            )
            if not records:
                break
            for record in records:
                if record.blob_ref is not None and record.acquired_source is not None:
                    continue
                omissions += 1
                diagnostic = record.snapshot_diagnostic or record.diagnostic
                code = (
                    diagnostic.code.value
                    if diagnostic is not None
                    else AcquisitionFailureCode.UNKNOWN.value
                )
                reasons[code] = reasons.get(code, 0) + 1
            after = records[-1].sequence
        run.report.snapshot_omissions = omissions
        run.report.snapshot_omission_reasons = reasons

    @staticmethod
    async def _verify_resumed_snapshot(run: _Sync, acquisitions: AcquisitionStore) -> None:
        if run.snapshot_completeness is None:
            return
        if await acquisitions.verify_snapshot_manifest(run.acquisition_run_id):
            return
        msg = "promoted source snapshot failed its canonical evidence verification"
        raise RuntimeError(msg)

    async def _owned_acquisition(self, run: _Sync, work: Awaitable[bool]) -> bool:
        """Run a bool-returning acquisition phase with the same independent lease heartbeat."""
        result = False

        async def capture() -> None:
            nonlocal result
            result = await work

        async with asyncio.TaskGroup() as ownership:
            task = ownership.create_task(capture(), name="journal-acquisition")
            ownership.create_task(
                self._heartbeat_acquisition_until_done(run, task), name="lease-heartbeat"
            )
        return result

    async def _run_local_stages(
        self,
        run: _Sync,
        producer: Callable[[_Sync, Conveyor[DiscoveredDoc | AcquisitionRecord]], Awaitable[None]],
        refs: Conveyor[DiscoveredDoc | AcquisitionRecord],
        bodies: Conveyor[_Fetched],
    ) -> None:
        """Run the bounded local stages while the ownership task fences the whole group."""

        async def produce() -> None:
            await producer(run, refs)

        async with asyncio.TaskGroup() as stages:
            stages.create_task(produce(), name="discover")
            for worker in range(self._fetch_workers):
                stages.create_task(self._fetch_into(run, refs, bodies), name=f"fetch-{worker}")
            for worker in range(self._ingest_workers):
                stages.create_task(self._ingest_from(run, bodies), name=f"ingest-{worker}")

    async def _heartbeat_acquisition_until_done(self, run: _Sync, work: asyncio.Task[None]) -> None:
        """Renew independently of document mutations and stop all work if ownership is lost."""
        acquisitions = run.acquisitions
        if acquisitions is None:  # pragma: no cover - created only for journal runs
            return
        interval = self._acquisition_lease_s / 3
        while not work.done():
            try:
                await asyncio.wait_for(asyncio.shield(work), timeout=interval)
            except TimeoutError:
                await self._keep_acquisition_lease_live(
                    run, acquisitions, self._acquisition_clock(), force=True
                )

    async def _discover_into(
        self, run: _Sync, refs: Conveyor[DiscoveredDoc | AcquisitionRecord]
    ) -> None:
        """Legacy in-memory discovery for stores without an acquisition journal.

        The three ways it ends are all recorded, because each means something different to the
        watermark: exhausted (may advance), stopped at ``limit`` (bounded, may not), and raised
        (unclean, may not).
        """
        run.report.enumeration_completed = False
        if run.limit is not None and run.accepted >= run.limit:
            run.report.limited = True
            return
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
            else:
                run.report.enumeration_completed = True
        except Exception as exc:  # noqa: BLE001 - an enumeration failure is not a crash
            run.report.error_type = type(exc).__name__
            run.report.error_message = str(exc)
            run.report.error = f"{type(exc).__name__}: {exc}"
        finally:
            closer = getattr(stream, "aclose", None)
            if closer is not None:
                await closer()
            # However discovery ended, the stage in front of it has to be told, or every fetch
            # worker waits for an item that is never coming and the run never returns.
            refs.finish()

    async def _enumerate_to_journal(self, run: _Sync) -> None:
        """Commit each source record before asking discovery for the next one.

        There is intentionally no downstream hand-off in this loop. The connector can be
        delayed only by its own work, journal admission, or lease maintenance; parsing and
        embedding do not start until the source iterator has ended and its completion marker is
        durable.
        """
        acquisitions = run.acquisitions
        if acquisitions is None:  # pragma: no cover - caller selects this path structurally
            return
        run.report.enumeration_completed = False
        if run.limit is not None and run.accepted >= run.limit:
            run.report.limited = True
            return
        stream = run.connector.discover(run.watermark)
        try:
            async for discovered in stream:
                if run.stop.is_set():
                    break
                now = self._acquisition_clock()
                await self._keep_acquisition_lease_live(run, acquisitions, now)
                appended = await acquisitions.append_acquisition_record(
                    run.acquisition_run_id,
                    run.accepted,
                    AcquisitionSource.from_discovered(discovered),
                    lease_owner=run.lease_owner,
                    lease_generation=run.lease_generation,
                    now=now,
                )
                if appended.sequence != run.accepted:
                    continue
                run.accepted += 1
                if run.limit is not None and run.accepted >= run.limit:
                    run.report.limited = True
                    break
            else:
                now = self._acquisition_clock()
                await self._keep_acquisition_lease_live(run, acquisitions, now)
                completed = await acquisitions.complete_acquisition_enumeration(
                    run.acquisition_run_id,
                    run.connector.watermark,
                    lease_owner=run.lease_owner,
                    lease_generation=run.lease_generation,
                    now=now,
                )
                run.candidate_watermark = completed.candidate_watermark
                run.report.enumeration_completed = True
        except Exception as exc:  # noqa: BLE001 - an enumeration/admission failure is reported
            run.report.error_type = type(exc).__name__
            run.report.error_message = str(exc)
            run.report.error = f"{type(exc).__name__}: {exc}"
        finally:
            closer = getattr(stream, "aclose", None)
            if closer is not None:
                await closer()

    async def _keep_acquisition_lease_live(
        self,
        run: _Sync,
        acquisitions: AcquisitionStore,
        now: datetime,
        *,
        force: bool = False,
    ) -> None:
        """Renew near expiry; every following journal mutation is generation-fenced."""
        renewal_margin = timedelta(seconds=self._acquisition_lease_s / 3)
        if (
            not force
            and run.lease_expires_at is not None
            and now + renewal_margin < run.lease_expires_at
        ):
            return
        expires_at = now + timedelta(seconds=self._acquisition_lease_s)
        renewed = await acquisitions.renew_acquisition_lease(
            run.acquisition_run_id,
            run.lease_owner,
            run.lease_generation,
            now=now,
            expires_at=expires_at,
        )
        if not renewed:
            _raise_lost_acquisition_lease(run.acquisition_run_id)
        run.lease_expires_at = expires_at

    async def _fence_acquisition_publication(self, run: _Sync) -> AcquisitionFence:
        """Prove ownership immediately before a journal consumer makes bytes servable."""
        acquisitions = run.acquisitions
        if acquisitions is None:  # pragma: no cover - callback exists only for journal runs
            msg = "an acquisition publication fence was requested outside a durable run"
            raise RuntimeError(msg)
        now = self._acquisition_clock()
        await self._keep_acquisition_lease_live(run, acquisitions, now, force=True)
        return AcquisitionFence(
            run_id=run.acquisition_run_id,
            owner=run.lease_owner,
            generation=run.lease_generation,
            now=now,
        )

    async def _acquire_journal(self, run: _Sync) -> bool:
        """Fetch and retain a bounded page stream; return whether every item has coverage."""
        refs: Conveyor[DiscoveredDoc | AcquisitionRecord] = Conveyor(
            name="fetch",
            capacity=self._queue_depth_factor * self._fetch_workers,
            consumers=self._fetch_workers,
        )
        run.refs = refs
        async with asyncio.TaskGroup() as stages:
            stages.create_task(self._acquisition_candidates_into(run, refs), name="journal-reader")
            for worker in range(self._fetch_workers):
                stages.create_task(self._acquire_from_journal(run, refs), name=f"acquire-{worker}")
        acquisitions = run.acquisitions
        if acquisitions is None:  # pragma: no cover - selected structurally
            return False
        after: int | None = None
        while True:
            remaining = await acquisitions.list_acquisition_records(
                run.acquisition_run_id,
                states=(
                    AcquisitionRecordState.DISCOVERED,
                    AcquisitionRecordState.ACQUIRING,
                    AcquisitionRecordState.RETRY,
                ),
                after_sequence=after,
                limit=100,
            )
            if not remaining:
                return True
            if any(record.blob_ref is None for record in remaining):
                return False
            after = remaining[-1].sequence

    async def _acquisition_candidates_into(
        self, run: _Sync, refs: Conveyor[DiscoveredDoc | AcquisitionRecord]
    ) -> None:
        acquisitions = run.acquisitions
        if acquisitions is None:  # pragma: no cover
            refs.finish()
            return
        after: int | None = None
        try:
            while not run.stop.is_set():
                await refs.wait_for_room()
                if run.stop.is_set():
                    return
                records = await acquisitions.list_acquisition_records(
                    run.acquisition_run_id,
                    states=(
                        AcquisitionRecordState.DISCOVERED,
                        AcquisitionRecordState.ACQUIRING,
                        AcquisitionRecordState.RETRY,
                    ),
                    after_sequence=after,
                    limit=1,
                )
                if not records:
                    return
                if run.stop.is_set():
                    return
                record = records[0]
                # A retry carrying a blob belongs to local indexing, never another source call.
                if record.state is AcquisitionRecordState.RETRY and record.blob_ref is not None:
                    after = record.sequence
                    continue
                run.discovery_records_held.enter()
                await refs.put(record)
                after = record.sequence
        finally:
            refs.finish()

    async def _acquire_from_journal(
        self, run: _Sync, refs: Conveyor[DiscoveredDoc | AcquisitionRecord]
    ) -> None:
        acquisitions = run.acquisitions
        if acquisitions is None:  # pragma: no cover
            return
        while (work := await refs.take()) is not None:
            if not isinstance(work, AcquisitionRecord):  # pragma: no cover - private conveyor
                continue
            run.discovery_records_held.leave()
            record = work
            now = self._acquisition_clock()
            await self._keep_acquisition_lease_live(run, acquisitions, now)
            if record.state in {AcquisitionRecordState.DISCOVERED, AcquisitionRecordState.RETRY}:
                record = await acquisitions.transition_acquisition_record(
                    run.acquisition_run_id,
                    record.source.source_id,
                    record.state,
                    AcquisitionRecordState.ACQUIRING,
                    lease_owner=run.lease_owner,
                    lease_generation=run.lease_generation,
                    now=now,
                )

            existing = await self._store.find_document(run.connector.name, record.source.source_id)
            discovered = self._discovered(record)
            if self._unchanged_by_token(existing, discovered):
                reusable = await self._validated_reusable_snapshot(run, record)
                if reusable is not None:
                    await self._keep_acquisition_lease_live(
                        run, acquisitions, self._acquisition_clock(), force=True
                    )
                    await acquisitions.settle_unchanged_acquisition_record(
                        run.acquisition_run_id,
                        record.source.source_id,
                        existing.id,  # pyright: ignore[reportOptionalMemberAccess]
                        lease_owner=run.lease_owner,
                        lease_generation=run.lease_generation,
                        now=self._acquisition_clock(),
                        blob_ref=reusable.blob_ref,
                        acquired_source=reusable.acquired_source,
                        fetched_version_token=reusable.fetched_version_token,
                    )
                    run.report.record(
                        DocumentOutcome(
                            source_id=record.source.source_id,
                            status=existing.status,  # pyright: ignore[reportOptionalMemberAccess]
                            document_id=existing.id,  # pyright: ignore[reportOptionalMemberAccess]
                            skipped="version",
                        )
                    )
                    _report_progress(run)
                    continue

            try:
                stage_key = f"{run.acquisition_run_id}\0{record.source.source_id}"
                staged = await self._blobs.resume_acquisition(stage_key)
                if staged is None:
                    raw = await self._fetch(run.connector, record.source.ref)
                    fetched_version = self._validated_fetched_version(
                        run.connector, record.source, raw
                    )
                    retained, acquired_source = await self._retain_acquisition(
                        stage_key, raw, record.source.source_id
                    )
                else:
                    retained, acquired_source = staged
                    data = await self._blobs.get(retained.ref or "")
                    raw = self._staged_raw(acquired_source, data)
                    fetched_version = self._validated_fetched_version(
                        run.connector, record.source, raw
                    )
            except Exception as exc:  # noqa: BLE001 - persisted as a typed safe diagnostic
                diagnostic = _acquisition_diagnostic(exc)
                await self._keep_acquisition_lease_live(
                    run, acquisitions, self._acquisition_clock(), force=True
                )
                await acquisitions.transition_acquisition_record(
                    run.acquisition_run_id,
                    record.source.source_id,
                    AcquisitionRecordState.ACQUIRING,
                    AcquisitionRecordState.RETRY,
                    lease_owner=run.lease_owner,
                    lease_generation=run.lease_generation,
                    now=self._acquisition_clock(),
                    diagnostic=diagnostic,
                )
                run.report.record(
                    DocumentOutcome(
                        source_id=record.source.source_id,
                        status=(
                            existing.status if existing is not None else DocumentStatus.PENDING
                        ),
                        document_id=(existing.id if existing is not None else ""),
                        detail=diagnostic.code.value,
                    )
                )
                _report_progress(run)
                continue

            await self._keep_acquisition_lease_live(
                run, acquisitions, self._acquisition_clock(), force=True
            )
            await acquisitions.transition_acquisition_record(
                run.acquisition_run_id,
                record.source.source_id,
                AcquisitionRecordState.ACQUIRING,
                AcquisitionRecordState.ACQUIRED,
                lease_owner=run.lease_owner,
                lease_generation=run.lease_generation,
                now=self._acquisition_clock(),
                blob_ref=retained.ref,
                acquired_source=acquired_source,
                fetched_version_token=fetched_version,
            )
            await self._blobs.complete_acquisition(stage_key)

    async def _validated_reusable_snapshot(
        self, run: _Sync, record: AcquisitionRecord
    ) -> AcquisitionRecord | None:
        """Return same-scope retained evidence only while its bytes still validate."""
        acquisitions = run.acquisitions
        if acquisitions is None:  # pragma: no cover - called only by the journal path
            return None
        if not run.reusable_snapshot_checked:
            promoted = await acquisitions.latest_promoted_snapshot(
                run.connector.name, run.scope_fingerprint
            )
            run.reusable_snapshot_run_id = None if promoted is None else promoted.id
            run.reusable_snapshot_checked = True
        if run.reusable_snapshot_run_id is None:
            return None
        reusable = await acquisitions.reusable_record_from_verified_snapshot(
            run.reusable_snapshot_run_id,
            record.source.source_id,
            record.source.version_token,
        )
        if reusable is None or reusable.acquired_source is None:
            return None
        try:
            reused_data = await self._blobs.get(reusable.blob_ref or "")
            if reused_data is None:
                return None
            reusable.acquired_source.raw(reused_data)
        except Exception:  # noqa: BLE001 - corrupt reuse falls back to fresh fetch
            return None
        return reusable

    @staticmethod
    def _discovered(record: AcquisitionRecord) -> DiscoveredDoc:
        source = record.source
        return DiscoveredDoc(
            ref=source.ref,
            version_token=source.version_token,
            title=source.title,
            media_type=source.media_type,
            size_bytes=source.size_bytes,
            metadata=source.metadata,
        )

    async def _retain_acquisition(
        self, key: str, raw: RawDocument, expected_source_id: str
    ) -> tuple[Retention, AcquiredSource]:
        """Validate identity, durably retain bytes, and bind the returned digest."""
        if raw.source_id != expected_source_id:
            msg = "the fetched source identity did not match its journal record"
            raise ValueError(msg)
        retained, acquired_source = await self._blobs.retain_acquisition(key, raw)
        if retained.ref is None:
            msg = "source bytes were not retained"
            raise _AcquisitionRetentionError(msg)
        if retained.ref != acquired_source.content_hash:
            msg = "the blob store returned a reference for different source bytes"
            raise _AcquisitionRetentionError(msg)
        return retained, acquired_source

    @staticmethod
    def _staged_raw(acquired: AcquiredSource, data: bytes | None) -> RawDocument:
        if data is None:
            msg = "the staged source blob is unavailable"
            raise _AcquisitionRetentionError(msg)
        return acquired.raw(data)

    @staticmethod
    def _validated_fetched_version(
        connector: Connector, discovered: AcquisitionSource, raw: RawDocument
    ) -> str | None:
        """Require connector-owned proof that fetched bytes are not older than discovery."""
        fetched = raw.metadata.get("version_token")
        fetched_token = fetched if isinstance(fetched, str) and fetched else None
        expected = discovered.version_token
        if expected is None:
            return fetched_token
        if fetched_token is None:
            msg = "the fetched body carried no revision proof for a versioned discovery record"
            raise _UnprovenSourceRevisionError(msg)
        comparator = getattr(connector, "fetched_revision_at_least", None)
        if comparator is None:
            proven = fetched_token == expected
        else:
            proven = comparator(expected, fetched_token)
        if not proven:
            msg = "the fetched body revision was not proven at least as new as discovery"
            raise _UnprovenSourceRevisionError(msg)
        return fetched_token

    async def _index_acquired(self, run: _Sync) -> None:
        """Derive from blob-backed journal records without consulting the connector."""
        refs: Conveyor[DiscoveredDoc | AcquisitionRecord] = Conveyor(
            name="snapshot",
            capacity=self._queue_depth_factor * self._fetch_workers,
            consumers=self._fetch_workers,
        )
        bodies: Conveyor[_Fetched] = Conveyor(
            name="parse",
            capacity=self._queue_depth_factor * self._ingest_workers,
            consumers=self._ingest_workers,
            producers=self._fetch_workers,
        )
        run.refs = refs
        run.bodies = bodies
        async with asyncio.TaskGroup() as stages:
            stages.create_task(self._index_candidates_into(run, refs), name="snapshot-reader")
            for worker in range(self._fetch_workers):
                stages.create_task(
                    self._snapshot_into(run, refs, bodies), name=f"snapshot-{worker}"
                )
            for worker in range(self._ingest_workers):
                stages.create_task(self._ingest_from(run, bodies), name=f"ingest-{worker}")

    async def _index_candidates_into(
        self, run: _Sync, refs: Conveyor[DiscoveredDoc | AcquisitionRecord]
    ) -> None:
        acquisitions = run.acquisitions
        if acquisitions is None:  # pragma: no cover
            refs.finish()
            return
        after: int | None = None
        try:
            while not run.stop.is_set():
                await refs.wait_for_room()
                if run.stop.is_set():
                    return
                records = await acquisitions.list_acquisition_records(
                    run.acquisition_run_id,
                    states=(
                        AcquisitionRecordState.ACQUIRED,
                        AcquisitionRecordState.INDEXING,
                        AcquisitionRecordState.RETRY,
                    ),
                    after_sequence=after,
                    limit=1,
                )
                if not records:
                    return
                if run.stop.is_set():
                    return
                record = records[0]
                after = record.sequence
                if record.blob_ref is None:
                    continue
                run.discovery_records_held.enter()
                await refs.put(record)
        finally:
            refs.finish()

    async def _snapshot_into(
        self,
        run: _Sync,
        refs: Conveyor[DiscoveredDoc | AcquisitionRecord],
        bodies: Conveyor[_Fetched],
    ) -> None:
        acquisitions = run.acquisitions
        if acquisitions is None:  # pragma: no cover
            bodies.finish()
            return
        try:
            while (work := await refs.take()) is not None:
                if not isinstance(work, AcquisitionRecord):  # pragma: no cover
                    continue
                run.discovery_records_held.leave()
                record = work
                retrying = record.state in {
                    AcquisitionRecordState.INDEXING,
                    AcquisitionRecordState.RETRY,
                }
                now = self._acquisition_clock()
                await self._keep_acquisition_lease_live(run, acquisitions, now)
                if record.state in {
                    AcquisitionRecordState.ACQUIRED,
                    AcquisitionRecordState.RETRY,
                }:
                    record = await acquisitions.transition_acquisition_record(
                        run.acquisition_run_id,
                        record.source.source_id,
                        record.state,
                        AcquisitionRecordState.INDEXING,
                        lease_owner=run.lease_owner,
                        lease_generation=run.lease_generation,
                        now=now,
                    )
                try:
                    raw = await self._raw_from_acquisition(record)
                except Exception as exc:  # noqa: BLE001 - safe typed retry outcome
                    diagnostic = _snapshot_diagnostic(exc)
                    await acquisitions.transition_acquisition_record(
                        run.acquisition_run_id,
                        record.source.source_id,
                        AcquisitionRecordState.INDEXING,
                        AcquisitionRecordState.RETRY,
                        lease_owner=run.lease_owner,
                        lease_generation=run.lease_generation,
                        now=self._acquisition_clock(),
                        diagnostic=diagnostic,
                    )
                    continue
                existing = await self._store.find_document(
                    run.connector.name, record.source.source_id
                )
                run.bodies_held.enter()
                await bodies.put(
                    _Fetched(
                        raw=raw,
                        discovered=DiscoveredDoc(
                            ref=record.source.ref,
                            version_token=record.fetched_version_token,
                            title=record.source.title,
                            media_type=raw.media_type,
                            size_bytes=len(raw.as_bytes()),
                            metadata=record.source.metadata,
                        ),
                        existing=existing,
                        acquisition_record=record,
                        retention=Retention(ref=record.blob_ref),
                        force=retrying,
                    )
                )
        finally:
            bodies.finish()

    async def _raw_from_acquisition(self, record: AcquisitionRecord) -> RawDocument:
        """Load and verify a journal-owned blob before local derivation sees it."""
        if record.blob_ref is None or record.acquired_source is None:
            msg = "the acquired source snapshot is unavailable"
            raise _MissingAcquisitionSnapshotError(msg)
        try:
            data = await self._blobs.get(record.blob_ref)
        except Exception as exc:
            msg = "the acquired source snapshot could not be read"
            raise _CorruptAcquisitionSnapshotError(msg) from exc
        if data is None:
            msg = "the acquired source snapshot is unavailable"
            raise _MissingAcquisitionSnapshotError(msg)
        try:
            return record.acquired_source.raw(data)
        except Exception as exc:
            msg = "the acquired source snapshot failed validation"
            raise _CorruptAcquisitionSnapshotError(msg) from exc

    async def _journal_into(
        self, run: _Sync, refs: Conveyor[DiscoveredDoc | AcquisitionRecord]
    ) -> None:
        """Feed a bounded hand-off from committed journal pages in discovery order."""
        acquisitions = run.acquisitions
        if acquisitions is None:  # pragma: no cover - caller selects this path structurally
            refs.finish()
            return
        after: int | None = None
        try:
            while not run.stop.is_set():
                await refs.wait_for_room()
                if run.stop.is_set():
                    return
                records = await acquisitions.list_acquisition_records(
                    run.acquisition_run_id,
                    states=(
                        AcquisitionRecordState.DISCOVERED,
                        AcquisitionRecordState.ACQUIRING,
                        AcquisitionRecordState.RETRY,
                    ),
                    after_sequence=after,
                    limit=1,
                )
                if not records:
                    return
                if run.stop.is_set():
                    return
                record = records[0]
                run.discovery_records_held.enter()
                await refs.put(record)
                after = record.sequence
        finally:
            refs.finish()

    async def _fetch_into(
        self,
        run: _Sync,
        refs: Conveyor[DiscoveredDoc | AcquisitionRecord],
        bodies: Conveyor[_Fetched],
    ) -> None:
        """One fetch worker: change detection, then the network, then the next stage.

        Level-1 change detection lives here rather than in discovery because it is what avoids
        the fetch, and it costs a store read that has no business blocking the source's paging.
        A document that skips never reaches the parse hand-off at all, which is why an unchanged
        corpus flows through this stage at the speed of the store rather than of the model.
        """
        try:
            while (source_work := await refs.take()) is not None:
                journaled = isinstance(source_work, AcquisitionRecord)
                if journaled:
                    run.discovery_records_held.leave()
                    record = source_work
                    source = record.source
                    discovered = DiscoveredDoc(
                        ref=source.ref,
                        version_token=source.version_token,
                        title=source.title,
                        media_type=source.media_type,
                        size_bytes=source.size_bytes,
                        metadata=source.metadata,
                    )
                    if run.acquisitions is None:  # pragma: no cover - journal records imply it
                        return
                    if record.state in {
                        AcquisitionRecordState.DISCOVERED,
                        AcquisitionRecordState.RETRY,
                    }:
                        now = self._acquisition_clock()
                        await self._keep_acquisition_lease_live(run, run.acquisitions, now)
                        await run.acquisitions.transition_acquisition_record(
                            run.acquisition_run_id,
                            discovered.source_id,
                            record.state,
                            AcquisitionRecordState.ACQUIRING,
                            lease_owner=run.lease_owner,
                            lease_generation=run.lease_generation,
                            now=now,
                        )
                else:
                    discovered = source_work
                fence_token = (
                    _publication_fence.set(lambda: self._fence_acquisition_publication(run))
                    if journaled
                    else None
                )
                try:
                    accepted = await self._accept(run.connector, discovered)
                finally:
                    if fence_token is not None:
                        _publication_fence.reset(fence_token)
                if isinstance(accepted, DocumentOutcome):
                    run.report.record(accepted)
                    if journaled:
                        await self._settle_journal_record(run, accepted)
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

    async def _settle_journal_record(self, run: _Sync, outcome: DocumentOutcome) -> None:
        """Close transitional direct-fetch work without claiming blob-backed acquisition."""
        acquisitions = run.acquisitions
        if acquisitions is None:  # pragma: no cover - caller has a journal record
            return
        if outcome.skipped:
            target = AcquisitionRecordState.UNCHANGED
        elif outcome.document_id:
            # Explicitly feature-gated in __init__. This temporary #176 consumer fetched and
            # published the source directly; it must not pass through ACQUIRED or INDEXING,
            # whose #175 contract requires a retained blob that this path does not produce.
            target = AcquisitionRecordState.SETTLED
        else:
            target = AcquisitionRecordState.RETRY
        now = self._acquisition_clock()
        await self._keep_acquisition_lease_live(run, acquisitions, now)
        await acquisitions.transition_acquisition_record(
            run.acquisition_run_id,
            outcome.source_id,
            AcquisitionRecordState.ACQUIRING,
            target,
            lease_owner=run.lease_owner,
            lease_generation=run.lease_generation,
            now=now,
        )

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
            fence_token = (
                _publication_fence.set(lambda: self._fence_acquisition_publication(run))
                if run.acquisitions is not None
                else None
            )
            try:
                outcomes = await self.ingest_raw(
                    fetched.raw,
                    source=run.connector.name,
                    version_token=fetched.discovered.version_token,
                    title=fetched.discovered.title,
                    existing=fetched.existing,
                    retention=fetched.retention,
                    force=fetched.force,
                    force_members=fetched.force,
                )
            finally:
                if fence_token is not None:
                    _publication_fence.reset(fence_token)
                # The other half of the pair entered in the fetch stage: this body is no longer
                # held. In a `finally`, so a document that failed still releases its accounting —
                # the deadlock this whole design has to avoid is a permit that a failure keeps.
                run.bodies_held.leave()
            for position, outcome in enumerate(outcomes):
                # The first outcome is the discovered document; anything after it came out of
                # the inside of it.
                run.report.record(outcome, expanded=position > 0)
            if fetched.acquisition_record is not None:
                await self._settle_indexed_record(run, fetched.acquisition_record, outcomes)
            # After the whole document, not per outcome: a container that expanded into five
            # hundred members is one thing that happened to somebody watching, and five hundred
            # lines of it is not progress.
            _report_progress(run)

    async def _settle_indexed_record(
        self, run: _Sync, record: AcquisitionRecord, outcomes: Sequence[DocumentOutcome]
    ) -> None:
        """Persist derivation outcome without weakening the independent byte snapshot."""
        acquisitions = run.acquisitions
        if acquisitions is None:  # pragma: no cover
            return
        retryable = [outcome for outcome in outcomes if _retryable_derivation(outcome)]
        outcome = retryable[0] if retryable else outcomes[0]
        target = (
            AcquisitionRecordState.SETTLED
            if outcomes[0].document_id and not retryable
            else AcquisitionRecordState.RETRY
        )
        diagnostic = (
            None
            if target is AcquisitionRecordState.SETTLED
            else AcquisitionDiagnostic(
                stage=AcquisitionStage.INDEXING,
                code=(
                    AcquisitionFailureCode.EMBED_FAILED
                    if outcome.failed_stage is PipelineStage.EMBED
                    else (
                        AcquisitionFailureCode.PARSE_FAILED
                        if outcome.failed_stage
                        in {
                            PipelineStage.MIDDLEWARE,
                            PipelineStage.PARSE,
                            PipelineStage.CHUNK,
                        }
                        else AcquisitionFailureCode.PUBLICATION_FAILED
                    )
                ),
            )
        )
        now = self._acquisition_clock()
        await self._keep_acquisition_lease_live(run, acquisitions, now)
        await acquisitions.transition_acquisition_record(
            run.acquisition_run_id,
            record.source.source_id,
            AcquisitionRecordState.INDEXING,
            target,
            lease_owner=run.lease_owner,
            lease_generation=run.lease_generation,
            now=now,
            diagnostic=diagnostic,
        )

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
            peak_discovery_records=run.discovery_records_held.peak,
        )

    async def _advance_watermark(self, run: _Sync) -> bool:
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
        reached = (
            run.candidate_watermark if run.acquisitions is not None else run.connector.watermark
        )
        if reached is not None:
            await self._store.set_watermark(run.connector.name, reached)
            return True
        return False

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
            await self._record_seen(existing.id)  # pyright: ignore[reportOptionalMemberAccess]
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
        retention: Retention | None = None,
        force_members: bool = False,
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

        ``force_members`` is narrower and used only when retrying a journal snapshot whose
        earlier local derivation failed. It prevents a failed container member from hash-skipping
        forever when the retained parent is expanded again; ordinary re-parse keeps its existing
        per-member change-detection behavior.

        ``expected`` guards the *top-level* document only, and for the same reason ``force``
        does: it is a statement about the snapshot **this caller** read, and a member found
        inside the archive is a document the caller never saw. A caller with newer bytes than
        anything stored — every connector sync — passes nothing and is guarded against nobody.

        ``retention`` identifies bytes the caller already loaded from the blob store. Durable
        acquisition and re-parse pass it so this shared derivation path never retains the same
        top-level snapshot twice. Container members are new derived documents and retain their
        own bytes normally.
        """
        outcome, members = await self._ingest_one(
            raw,
            source=source,
            version_token=version_token,
            title=title,
            existing=existing,
            force=force,
            expected=expected,
            retention=retention,
        )
        outcomes = [outcome]
        queue: list[MemberOutcome] = list(members)
        while queue:
            member = queue.pop(0)
            if isinstance(member, MemberFailure):
                outcomes.append(await self._record_member_failure(member, source))
                continue
            member_raw = _member_raw(member)
            member_existing = await self._store.find_document(source, member_raw.source_id)
            retry_member = force_members and _document_requires_local_retry(
                member_existing, member_raw
            )
            inner, deeper = await self._ingest_one(
                member_raw,
                source=source,
                title=_member_title(member),
                existing=member_existing,
                force=retry_member,
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
        retention: Retention | None = None,
    ) -> tuple[DocumentOutcome, tuple[MemberOutcome, ...]]:
        """One document, and whatever it turned out to contain."""
        source_bytes = raw.as_bytes()
        digest = content_hash(source_bytes)
        if existing is None:
            existing = await self._store.find_document(source, raw.source_id)

        if not force and self._unchanged_by_hash(existing, digest, raw):
            if existing is None:  # pragma: no cover - the predicate rejects absence
                msg = "hash-unchanged requires an existing document"
                raise RuntimeError(msg)
            await self._record_seen(existing.id, version_token=version_token)
            return (
                DocumentOutcome(
                    source_id=raw.source_id,
                    status=existing.status,
                    document_id=existing.id,
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
                    retention=retention,
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
        retention: Retention | None,
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
        if retention is None:
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
                retention=retention,
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
                # schema. The shared publication below empties both chunks and glossary; doing
                # either in a separate write would expose half of the conclusion.
                document, glossary_detail = await self._publish_chunkless(
                    document,
                    expected=expected,
                    retention=retention,
                )
            await self._observe(document)
            return (
                DocumentOutcome(
                    source_id=raw.source_id,
                    status=document.status,
                    document_id=document.id,
                    detail=result.status_detail,
                    failed_stage=result.failed_stage,
                    glossary_detail=glossary_detail,
                    members=tuple(member.source_id for member in members),
                ),
                members,
            )

        return (
            await self._finish(
                result,
                document,
                raw=raw,
                existing=existing,
                retention=retention,
                expected=expected,
            ),
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
        retention: Retention,
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
                return await self._nothing_to_index(
                    result,
                    document,
                    raw=raw,
                    existing=existing,
                    retention=retention,
                    expected=expected,
                )
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
        except _SupersededError:
            raise
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
            document,
            chunks,
            vectors,
            raw=raw,
            existing=existing,
            retention=retention,
            expected=expected,
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
        self,
        result: ChainResult,
        document: Document,
        *,
        raw: RawDocument,
        existing: Document | None,
        retention: Retention,
        expected: DocumentRevision | None,
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
        settled_document = _with_status(document, settled)
        stored, glossary_detail = await self._publish_chunkless(
            settled_document,
            expected=expected,
            retention=retention,
        )
        await self._observe(stored)
        return DocumentOutcome(
            source_id=raw.source_id,
            status=stored.status,
            document_id=stored.id,
            detail=settled.status_detail,
            glossary_detail=glossary_detail,
        )

    async def _publish_chunkless(
        self,
        document: Document,
        *,
        expected: DocumentRevision | None,
        retention: Retention,
    ) -> tuple[Document, str]:
        """Atomically publish every successful conclusion that has no chunks.

        Parser-empty, container, unsupported, middleware-skipped and chunker-empty outcomes
        all replace the same derived state: document, empty chunks, empty glossary and settled
        lineage. Keeping one path is what prevents a newly added terminal status from quietly
        reintroducing the old sequence of independently visible writes.
        """
        publication = self._publication_of(document, [])
        settled = document.model_copy(update={"publication_id": publication})
        entries, glossary_fp, glossary_detail = self._derive_definitions(settled, [])
        fence = await self._publication_authority()
        publisher = self._fenced_store if fence is not None else None
        publish = publisher.fenced_publish_document if publisher is not None else None
        if fence is not None and publish is not None:
            committed = await publish(
                fence,
                settled,
                [],
                expected=expected,
                chunk_fp=None,
                embed_fp=None,
                parse_fp=self._parse_lineage_of(settled),
                glossary_entries=entries,
                glossary_fp=glossary_fp,
                original_omitted_reason=retention.omitted_reason,
            )
        else:
            committed = await self._store.publish_document(
                settled,
                [],
                expected=expected,
                chunk_fp=None,
                embed_fp=None,
                parse_fp=self._parse_lineage_of(settled),
                glossary_entries=entries,
                glossary_fp=glossary_fp,
                original_omitted_reason=retention.omitted_reason,
            )
        if not committed.committed or committed.stored is None:
            raise _SupersededError(committed.stored)
        return committed.stored, glossary_detail

    async def _commit(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        *,
        raw: RawDocument,
        existing: Document | None,
        retention: Retention,
        expected: DocumentRevision | None = None,
    ) -> DocumentOutcome:
        """Stage vectors, then atomically flip every authoritative derived row.

        Vector row ids include ``publication``, so staging cannot overwrite the active
        revision even when a chunk's logical id survives while ``embed_text`` changes. The
        relational store replaces document, chunks, glossary and lineage in one transaction;
        ``documents.publication_id`` is the pointer dense hydration checks. A crash or CAS miss
        before that flip leaves the old publication wholly servable and the staged rows inert.

        Raises:
            _SupersededError: The stored document moved past ``expected`` before the atomic
                relational flip. Its staged vectors remain tombstoned for cleanup.
        """
        try:
            publication = self._publication_of(document, chunks, vectors)
            indexed_document = document.model_copy(
                update={
                    "publication_id": publication,
                    "status": DocumentStatus.INDEXED,
                    "status_detail": None,
                    "failed_stage": None,
                }
            )
            fence = await self._publication_authority()
            publisher = self._fenced_store if fence is not None else None
            if existing is None or existing.publication_id != publication:
                if fence is not None and publisher is not None:
                    await publisher.fenced_stage_vectors(fence, publication, chunks)
                else:
                    await self._store.stage_vectors(publication, chunks)
            await self._check_publication_fence()
            await self._vectors.upsert(chunks, vectors, publication_id=publication)
            entries, glossary_fp, glossary_detail = self._derive_definitions(document, chunks)
            fence = await self._publication_authority()
            publisher = self._fenced_store if fence is not None else None
            if fence is not None and publisher is not None:
                committed = await publisher.fenced_publish_document(
                    fence,
                    indexed_document,
                    chunks,
                    expected=expected,
                    chunk_fp=self._chunk_fingerprint.canonical(),
                    embed_fp=self._embedder.fingerprint.canonical(),
                    parse_fp=self._parse_lineage_of(document),
                    glossary_entries=entries,
                    glossary_fp=glossary_fp,
                    original_omitted_reason=retention.omitted_reason,
                )
            else:
                committed = await self._store.publish_document(
                    indexed_document,
                    chunks,
                    expected=expected,
                    chunk_fp=self._chunk_fingerprint.canonical(),
                    embed_fp=self._embedder.fingerprint.canonical(),
                    parse_fp=self._parse_lineage_of(document),
                    glossary_entries=entries,
                    glossary_fp=glossary_fp,
                    original_omitted_reason=retention.omitted_reason,
                )
        except AcquisitionLeaseLostError:
            raise
        except Exception as exc:  # noqa: BLE001 - a store failure is this document's
            return await self._demote(
                document, existing, PipelineStage.STORE, f"{type(exc).__name__}: {exc}"
            )
        if not committed.committed or committed.stored is None:
            raise _SupersededError(committed.stored)
        indexed = committed.stored
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
        fence = await self._publication_authority()
        publisher = self._fenced_store if fence is not None else None
        if fence is not None and publisher is not None:
            committed = await publisher.fenced_publish_record(fence, document, expected=expected)
        elif expected is None:
            return await self._store.upsert_document(document)
        else:
            committed = await self._store.commit_document(document, expected=expected)
        if not committed.committed or committed.stored is None:
            raise _SupersededError(committed.stored)
        return committed.stored

    async def _publication_authority(self) -> AcquisitionFence | None:
        fence = _publication_fence.get()
        return None if fence is None else await fence()

    async def _check_publication_fence(self) -> None:
        """Run the task-local acquisition fence at the last point before publication."""
        await self._publication_authority()

    async def _record_seen(self, document_id: str, *, version_token: str | None = None) -> None:
        fence = await self._publication_authority()
        publisher = self._fenced_store if fence is not None else None
        if fence is not None and publisher is not None:
            await publisher.fenced_record_seen(fence, document_id, version_token=version_token)
            return
        await self._store.record_seen(document_id, version_token=version_token)

    def _derive_definitions(
        self, document: Document, chunks: Sequence[Chunk]
    ) -> tuple[Sequence[GlossaryEntry] | None, str | None, str]:
        """Compute glossary state without publishing it."""
        if self._glossary is None or self._glossary_lineage is None:
            return None, None, ""
        if not self._detect_glossary:
            return None, self._glossary_lineage, ""
        try:
            entries = detect_entries(chunks, title=document.title, media_type=document.media_type)
        except Exception as exc:  # noqa: BLE001 - a detector bug costs this document's glossary
            return None, None, f"glossary detection failed: {type(exc).__name__}: {exc}"
        return entries, self._glossary_lineage, ""

    def _publication_of(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]] = (),
    ) -> str:
        """Content address for one complete set of derived ingest state."""
        return content_hash(
            "\0".join(
                (
                    document.model_dump_json(exclude={"publication_id", "indexed_at"}),
                    self._chunk_fingerprint.canonical(),
                    self._embedder.fingerprint.canonical(),
                    self._parse_lineage_of(document) or "",
                    self._glossary_lineage or "",
                    *(chunk.model_dump_json() for chunk in chunks),
                    *(
                        repr(self._canonical_vector(chunk, vector))
                        for chunk, vector in zip(chunks, vectors, strict=True)
                    ),
                )
            )
        )

    def _canonical_vector(self, chunk: Chunk, vector: Sequence[float]) -> tuple[float, ...]:
        """Canonicalize with enough attribution to diagnose a backend's invalid output."""
        try:
            return canonical_stored_vector(vector)
        except ValueError as exc:
            fingerprint = self._embedder.fingerprint
            backend = fingerprint.backend or "an unspecified backend"
            msg = (
                f"chunk {chunk.id!r} was offered a vector with non-finite values for "
                f"{fingerprint.describe()} from {backend}. NaN and infinity cannot participate "
                "in cosine distance, so the vector was refused before publication."
            )
            raise ValueError(msg) from exc

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

        A successful parse is constructed here but not written. Its source record joins chunks,
        glossary and lineage only at the atomic publication boundary; until then an existing
        indexed row remains the active revision. Chunk-less terminal conclusions are returned
        unwritten and go through the shared atomic boundary with the caller's guard.

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
        if result.status is not DocumentStatus.FAILED:
            return document
        fence = await self._publication_authority()
        publisher = self._fenced_store if fence is not None else None
        if fence is not None and publisher is not None:
            committed = await publisher.fenced_publish_failure(
                fence,
                document,
                expected=expected,
                original_omitted_reason=retention.omitted_reason,
            )
        else:
            committed = await self._store.publish_failure(
                document,
                expected=expected,
                original_omitted_reason=retention.omitted_reason,
            )
        if not committed.committed or committed.stored is None:
            raise _SupersededError(committed.stored)
        return committed.stored

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
        retention: Retention | None = None,
    ) -> DocumentOutcome:
        retention = retention or Retention(omitted_reason="not retained: the document was skipped")
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
        glossary_detail = ""
        if result.status is not DocumentStatus.FAILED:
            document, glossary_detail = await self._publish_chunkless(
                document,
                expected=expected,
                retention=retention,
            )
        await self._observe(document)
        return DocumentOutcome(
            source_id=raw.source_id,
            status=document.status,
            document_id=document.id,
            detail=result.status_detail,
            failed_stage=result.failed_stage,
            glossary_detail=glossary_detail,
        )

    async def _advance(self, existing: Document | None, status: DocumentStatus) -> None:
        """Record an in-flight status, unless doing so would unserve a working document.

        A document with no servable content loses nothing by being marked in flight, and gains
        a recovery sweep that can find it. An ``indexed`` one has everything to lose: it would
        stop being returned the moment a re-sync began, and stay that way if the re-sync failed.
        """
        if existing is None or existing.status is DocumentStatus.INDEXED:
            return
        fence = await self._publication_authority()
        publisher = self._fenced_store if fence is not None else None
        if fence is not None and publisher is not None:
            await publisher.fenced_set_status(fence, existing.id, status)
        else:
            await self._store.set_status(existing.id, status)

    async def _observe(self, document: Document) -> None:
        """Let hooks see a committed document. Their failure is theirs, not the document's."""
        try:
            await self._middleware.after_store(document)
        except Exception as exc:  # noqa: BLE001 - the document is already committed
            updates: Metadata = {"last_after_store_error": f"{type(exc).__name__}: {exc}"}
            fence = await self._publication_authority()
            publisher = self._fenced_store if fence is not None else None
            if fence is not None and publisher is not None:
                await publisher.fenced_annotate(fence, document.id, updates)
            else:
                await self._store.annotate(document.id, updates)

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
            return DocumentOutcome(
                source_id=source_id,
                status=DocumentStatus.FAILED,
                detail=detail,
                failed_stage=stage,
            )
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
        updates: Metadata = {"last_ingest_error": {"stage": stage.value, "detail": detail}}
        fence = await self._publication_authority()
        publisher = self._fenced_store if fence is not None else None
        if fence is not None and publisher is not None:
            await publisher.fenced_annotate(fence, document.id, updates)
        else:
            await self._store.annotate(document.id, updates)
        if was_indexed:
            return DocumentOutcome(
                source_id=document.source_id,
                status=DocumentStatus.INDEXED,
                document_id=document.id,
                detail=detail,
                failed_stage=stage,
            )
        failed = document.model_copy(
            update={
                "status": DocumentStatus.FAILED,
                "status_detail": detail,
                "failed_stage": stage,
            }
        )
        fence = await self._publication_authority()
        publisher = self._fenced_store if fence is not None else None
        if fence is not None and publisher is not None:
            committed = await publisher.fenced_publish_record(fence, failed, expected=None)
            if not committed.committed:
                raise _SupersededError(committed.stored)
        else:
            await self._store.upsert_document(failed)
        return DocumentOutcome(
            source_id=document.source_id,
            status=DocumentStatus.FAILED,
            document_id=document.id,
            detail=detail,
            failed_stage=stage,
        )


def _retryable_derivation(outcome: DocumentOutcome) -> bool:
    """Whether local work failed while a safe old publication may still be served."""
    return outcome.status is DocumentStatus.FAILED or (
        outcome.status is DocumentStatus.INDEXED and bool(outcome.detail) and not outcome.superseded
    )


def _document_requires_local_retry(document: Document | None, raw: RawDocument) -> bool:
    """Select only failed container members when re-expanding a retained parent snapshot."""
    return (
        document is None
        or document.status is DocumentStatus.FAILED
        or (
            bool(document.metadata.get("last_ingest_error"))
            and document.content_hash != content_hash(raw.as_bytes())
        )
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


def _first_failure(failures: BaseExceptionGroup[Exception]) -> tuple[Exception, str]:
    """One line for what stopped the stages, from however many of them noticed.

    Several ingest workers meeting the same dead store produce several identical exceptions, and
    a report that concatenated them would say the same sentence four times and bury how many
    stages were affected. The first is named because it is the one that happened; the count is
    kept because "four stages" and "one stage" are different diagnoses.
    """
    flattened = _leaves(failures)
    first = flattened[0]
    detail = f"{type(first).__name__}: {first}"
    return first, detail if len(flattened) == 1 else f"{detail} (and {len(flattened) - 1} more)"


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


def _raise_lost_acquisition_lease(run_id: str) -> None:
    msg = f"acquisition lease for run {run_id!r} was lost"
    raise AcquisitionLeaseLostError(msg)


def _snapshot_scope(connector: Connector) -> tuple[str, str]:
    """Read a connector's non-secret scope identity, with a safe whole-instance default."""
    declared = getattr(connector, "source_scope", None)
    source_scope = (
        declared if isinstance(declared, str) and declared else f"instance:{connector.name}"
    )
    declared_fingerprint = getattr(connector, "scope_fingerprint", None)
    if isinstance(declared_fingerprint, str) and declared_fingerprint:
        return source_scope, declared_fingerprint
    fingerprint = hashlib.blake2b(source_scope.encode(), digest_size=20).hexdigest()
    return source_scope, fingerprint


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
