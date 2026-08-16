"""A connector sync is three bounded stages, and this is what holds each bound in place.

**Every assertion here is about an arrival or a maximum, never about elapsed time.** A
concurrency test written with a sleep passes on an idle machine, fails on a loaded runner, and
— the reason it is worth a paragraph — passes against a *sequential* implementation whenever
the sleep happens to be long enough. So the shape throughout is: park the work in a gate, wait
for the stated number of callers to be inside it at once, and let the wait itself be the
assertion. If they never are, :meth:`tests.ingest.fakes.Gate.wait_for` says how many ever were.

**The bounds are written out as arithmetic rather than imported.** Recomputing a queue depth
from the pipeline's own settings would make a test that passes whatever the pipeline derives,
which is the exact way a bound stops being checked. A number here that has to change is a
number somebody has to re-derive and state, which is the point.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast, override

import pytest

from manicule.connectors import CursorExpiredError
from manicule.core.content import Chunk, Document, DocumentStatus, RawDocument
from manicule.core.ids import content_hash
from manicule.ingest.middleware import MiddlewareRunner
from manicule.ingest.pipeline import IngestPipeline
from manicule.ingest.workers import InProcessRunner
from tests.fakes import MEDIA_TYPE, HashEmbedder
from tests.ingest import fakes

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, Sequence

    from manicule.core.protocols import Embedder, Middleware
    from manicule.core.sources import DiscoveredDoc, Watermark
    from manicule.ingest.pipeline import RunReport
    from manicule.ingest.workers import ParseRunner


def build(
    *,
    store: fakes.MemoryIngestStore | None = None,
    vectors: fakes.MemoryVectors | None = None,
    runner: ParseRunner | None = None,
    embedder: Embedder | None = None,
    middleware: Sequence[Middleware] = (),
    parsers: Mapping[str, object] | None = None,
    chain: Sequence[str] = ("lines",),
    fetch_concurrency: int = 4,
    parse_workers: int = 2,
    queue_depth_factor: int = 2,
    shutdown_grace_s: float = 30.0,
) -> tuple[IngestPipeline, fakes.MemoryIngestStore, fakes.MemoryVectors]:
    """A pipeline whose stage bounds are all stated, over in-memory everything.

    Every concurrency knob is an argument, because a test that could not set them would be
    asserting against ``default_worker_count()`` — a number that depends on the machine, which
    is the one thing a bound must not depend on.
    """
    store = store or fakes.MemoryIngestStore()
    vectors = vectors or fakes.MemoryVectors()
    chunker = fakes.BlockChunker()
    pipeline = IngestPipeline(
        store=store,
        chunker=chunker,
        embedder=embedder or HashEmbedder(),
        vectors=vectors,
        runner=runner or InProcessRunner(parsers or {"lines": fakes.LineParser()}),
        resolve_chain=lambda _: list(chain),
        middleware=MiddlewareRunner(middleware),
        chunk_fingerprint=chunker.fingerprint,
        fetch_concurrency=fetch_concurrency,
        parse_workers=parse_workers,
        queue_depth_factor=queue_depth_factor,
        shutdown_grace_s=shutdown_grace_s,
    )
    return pipeline, store, vectors


def corpus(count: int, *, prefix: str = "doc") -> dict[str, str]:
    """``count`` invented documents, each with two lines so each produces two chunks."""
    return {
        f"{prefix}-{number:03d}": f"line one of {number}\nline two of {number}"
        for number in range(count)
    }


# --- the fetch stage reaches, and never exceeds, its configured concurrency --------------------


async def test_a_sync_reaches_exactly_the_configured_fetch_concurrency() -> None:
    """The claim the semaphore was making and the loop was not honoring.

    Before the stages existed, ``fetch_concurrency`` guarded a semaphore that a sequential loop
    never contended: the run awaited one document's whole path before pulling the next, so the
    most fetches that ever overlapped was one however the setting was configured. This is that
    measurement, turned into a gate — four callers must be inside the fetch at once, and the
    test cannot pass by waiting longer.
    """
    pipeline, _, _ = build(fetch_concurrency=4, parse_workers=2)
    connector = fakes.ObservedConnector(corpus(20), park_fetches=True)

    run = asyncio.create_task(pipeline.run(connector))
    await connector.fetching.wait_for(4)
    connector.fetching.open()
    report = await run

    assert connector.fetching.peak == 4, "the fetch stage did not reach its configured width"
    assert report.stages.peak_fetches == 4, "and the pipeline's own gauge agrees with the source"
    assert report.indexed == 20


async def test_a_sync_never_exceeds_the_configured_fetch_concurrency() -> None:
    """The other half, and it fails at the fetch that crossed the line rather than afterwards.

    A bound that is only checked as a peak at the end reports a number; this reports which
    document was the straw, in the task that carried it. The corpus is far larger than the bound
    so that the stage is asked to exceed it continuously rather than once.
    """
    pipeline, _, _ = build(fetch_concurrency=3, parse_workers=2)
    connector = fakes.ObservedConnector(corpus(40), fetch_capacity=3)

    report = await pipeline.run(connector)

    assert connector.fetching.entries == 40, "every document was fetched"
    assert connector.fetching.peak <= 3
    assert report.stages.peak_fetches <= 3


async def test_one_fetch_that_hangs_does_not_stop_the_others_finishing() -> None:
    """Concurrency is worth nothing if the stage still moves at the speed of its slowest member.

    One document parks in the fetch for the length of the run while the rest go all the way to
    ``indexed``. Sequentially this is impossible: the parked fetch is the head of the queue and
    nothing behind it moves.
    """
    slow = fakes.Gate()

    class OneSlowFetch(fakes.ObservedConnector):
        @override
        async def fetch(self, ref: object) -> RawDocument:
            if getattr(ref, "source_id", "") == "doc-000":
                await slow.pass_through()
            return await super().fetch(ref)  # pyright: ignore[reportArgumentType]

    pipeline, store, _ = build(fetch_concurrency=4, parse_workers=2)
    connector = OneSlowFetch(corpus(8))

    run = asyncio.create_task(pipeline.run(connector))
    await slow.wait_for(1)
    # Everything except the parked document must reach the store while it is still parked. The
    # gate is what makes "while it is still parked" a fact rather than a hope.
    for source_id in sorted(corpus(8))[1:]:
        await _until_indexed(store, source_id)
    slow.open()
    report = await run

    assert report.indexed == 8


# --- the parse stage uses more than one worker ------------------------------------------------


async def test_one_connector_sync_uses_more_than_one_parse_worker() -> None:
    """``parse_workers`` was a pool of one user.

    The pool has always been able to serve several attempts at once — it hands out idle workers
    from a queue — and the sequential loop only ever had one document to give it. Two attempts
    inside the runner at the same moment is the whole claim, and the gate is what makes it one.
    """
    runner = fakes.GatedRunner({"lines": fakes.LineParser()})
    pipeline, _, _ = build(runner=runner, fetch_concurrency=4, parse_workers=3)
    connector = fakes.ObservedConnector(corpus(12))

    run = asyncio.create_task(pipeline.run(connector))
    await runner.gate.wait_for(2)
    runner.gate.open()
    report = await run

    assert runner.gate.peak >= 2, "one connector sync still parses one document at a time"
    assert report.stages.peak_parses >= 2
    assert report.indexed == 12


async def test_parse_concurrency_stays_within_the_ingest_stage_width() -> None:
    """The parse stage is bounded by how many ingest workers exist, and that is the bound.

    ``parse_workers + 1`` ingest workers can have at most that many attempts in the pool at
    once, so the pool is kept fed without a second queue in front of it. The ceiling is asserted
    at the moment it would be crossed.
    """
    runner = fakes.GatedRunner({"lines": fakes.LineParser()}, capacity=3)
    runner.gate.open()
    pipeline, _, _ = build(runner=runner, fetch_concurrency=8, parse_workers=2)

    report = await pipeline.run(fakes.ObservedConnector(corpus(30)))

    assert runner.gate.peak <= 3, "parse_workers + 1 ingest workers, so at most three attempts"
    assert report.indexed == 30


# --- the embedder stays serialized ---------------------------------------------------------------


async def test_the_embedder_is_never_called_while_it_is_already_running() -> None:
    """One model, one unified-memory pool, one accelerator: a second batch is contention.

    Asserted by an embedder that raises the moment it is re-entered, over a corpus wide enough
    that four fetch workers and three ingest workers are all busy — so the opportunity for a
    second concurrent call is offered continuously rather than once.

    **Every document here is a different one, and that is deliberate but not sufficient.** A
    test that only ever used two documents would say something true about the accelerator and
    nothing about how many documents were in flight; this one is the accelerator claim, and the
    per-document claims are asserted separately below.
    """
    embedder = fakes.ExclusiveEmbedder()
    pipeline, _, _ = build(embedder=embedder, fetch_concurrency=4, parse_workers=2)

    report = await pipeline.run(fakes.ObservedConnector(corpus(30)))

    assert embedder.overlaps == 0
    assert report.stages.peak_embeds == 1, "the embedding lock had one holder at a time"
    assert report.indexed == 30
    assert embedder.batches, "the embedder was actually used, so zero overlaps means something"


async def test_documents_wait_for_the_embedder_rather_than_queueing_inside_it() -> None:
    """The serialization is a lock in front of the model, not a queue behind it.

    While one document is parked inside the embedder, others reach the lock and stop there. What
    this rules out is an implementation that lets several batches into the model and relies on
    the model to sort them out.
    """
    embedder = fakes.GatedEmbedder()
    pipeline, _, _ = build(embedder=embedder, fetch_concurrency=4, parse_workers=2)
    connector = fakes.ObservedConnector(corpus(12))

    run = asyncio.create_task(pipeline.run(connector))
    await embedder.gate.wait_for(1)
    assert embedder.gate.inside == 1, "only one document is inside the model"
    embedder.gate.open()
    report = await run

    assert embedder.overlaps == 0
    assert report.indexed == 12


async def test_ready_chunks_are_batched_within_the_derived_token_and_count_limits() -> None:
    """Batch size is derived from both fingerprints, and concurrency does not widen it.

    ``target_batch_tokens // budget_tokens``, clamped — so a document of eight chunks against a
    budget of 64 tokens and a target of 128 is batched two at a time. Checked under a staged run
    rather than a direct call, because the thing that could have gone wrong is exactly the stage
    boundary handing the embedder more than one document's worth.
    """
    embedder = fakes.CountingEmbedder()
    store = fakes.MemoryIngestStore()
    chunker = fakes.BlockChunker()
    pipeline = IngestPipeline(
        store=store,
        chunker=chunker,
        embedder=embedder,
        vectors=fakes.MemoryVectors(),
        runner=InProcessRunner({"lines": fakes.LineParser()}),
        resolve_chain=lambda _: ["lines"],
        middleware=MiddlewareRunner(()),
        chunk_fingerprint=chunker.fingerprint,
        fetch_concurrency=4,
        parse_workers=2,
        target_batch_tokens=128,
        max_embed_batch=64,
    )
    wide = {"wide": "\n".join(f"line {number}" for number in range(8))}

    await pipeline.run(fakes.ObservedConnector(wide))

    assert embedder.batches == [2, 2, 2, 2], "128 // 64 == 2, and no batch spans two documents"


# --- backpressure -----------------------------------------------------------------------------

# The pile-up these three tests share, derived once so the arithmetic is stated rather than
# repeated. With `fetch_concurrency=2`, `parse_workers=1` and `queue_depth_factor=1`:
#
#   ingest workers   parse_workers + 1                                     = 2
#   parse hand-off   queue_depth_factor x ingest workers                   = 2
#   fetch workers    fetch_concurrency, each blocked holding a fetched body = 2
#   fetch hand-off   queue_depth_factor x fetch_concurrency                = 2
#
# Eight in all. So eight documents are in the system and discovery is blocked inside the ninth
# `put`. Six of the eight are fetched bodies; the two waiting in the fetch hand-off are still
# references, which is why that hand-off is the cheap one and this one is not.
BLOCKED_IN_SYSTEM = 8
BLOCKED_YIELDS = 9
BLOCKED_BODIES = 6


async def test_the_legacy_nonjournal_source_still_obeys_the_local_queue_bound() -> None:
    """Protocol-only stores retain the old bounded fallback.

    Production SQLite has the durable journal boundary. In-memory protocol implementations that
    lack the optional acquisition surface still use direct discovery and must remain bounded
    while callers migrate.

    **The stop is proven, not timed.** Once the embedder is parked, every stage behind it is
    blocked on a full hand-off, and discovery is inside a ``put`` that cannot return until the
    test opens the gate. So a tenth document is not merely unlikely — there is no scheduling in
    which one is produced, which is why the count is asserted rather than sampled.
    """
    embedder = fakes.GatedEmbedder()
    pipeline, _, _ = build(
        embedder=embedder, fetch_concurrency=2, parse_workers=1, queue_depth_factor=1
    )
    connector = fakes.ObservedConnector(corpus(40))

    run = asyncio.create_task(pipeline.run(connector))
    await embedder.gate.wait_for(1)
    for _ in range(BLOCKED_YIELDS):
        await connector.yielded.acquire()

    assert connector.yields == BLOCKED_YIELDS, (
        "the nonjournal fallback was paged further than its local stages can hold"
    )

    embedder.gate.open()
    report = await run

    assert report.indexed == 40
    assert report.stages.fetch_queue.blocked_puts > 0, (
        "the fallback never reached the configured queue bound"
    )


async def test_downstream_backpressure_expires_the_cursor_before_the_next_page() -> None:
    """Characterize the coupling that durable source acquisition must remove.

    The first page has more records than the bounded stages can hold. Once embedding is parked,
    discovery is suspended while holding the cursor for page two. Opening the embed gate makes
    each completed document advance a manual clock by 50 ms, proving that ordinary downstream
    backpressure becomes a typed source-enumeration failure. No scheduler timing participates.

    When issue #175's durable journal boundary lands, this test should be extended to prove that
    admission continues independently of indexing and inverted to require a complete run.
    """
    clock = fakes.ManualClock()
    embedder = fakes.ClockedGatedEmbedder(clock, seconds_per_document=0.05)
    pipeline, store, _ = build(
        embedder=embedder, fetch_concurrency=2, parse_workers=1, queue_depth_factor=1
    )
    connector = fakes.ExpiringCursorConnector(
        corpus(1_000, prefix="synthetic-doc"),
        clock=clock,
        page_size=100,
        cursor_lifetime_seconds=0.5,
    )

    run = asyncio.create_task(pipeline.run(connector))
    await connector.cursor_issued.wait()
    await embedder.gate.wait_for(1)
    for _ in range(BLOCKED_YIELDS):
        await connector.yielded.acquire()

    assert connector.yields == BLOCKED_YIELDS
    embedder.gate.open()
    report = await run

    assert report.error_type == "CursorExpiredError"
    assert report.error_message == "source enumeration failed"
    assert not report.enumeration_completed
    assert not report.watermark_advanced
    assert report.stages.fetch_queue.blocked_puts > 0
    assert store.watermarks == {}
    assert connector.cursors_issued == 1


async def test_the_documents_held_in_memory_stay_within_the_configured_bounds() -> None:
    """ "Do not accumulate the complete fetched corpus in memory", as a number.

    The bound is the parse hand-off plus everyone holding a body: the ingest workers that have
    one, and the fetch workers blocked trying to hand one over. Forty documents go through and
    at most six bodies exist at once.
    """
    embedder = fakes.GatedEmbedder()
    pipeline, _, _ = build(
        embedder=embedder, fetch_concurrency=2, parse_workers=1, queue_depth_factor=1
    )
    connector = fakes.ObservedConnector(corpus(40))

    run = asyncio.create_task(pipeline.run(connector))
    await embedder.gate.wait_for(1)
    for _ in range(BLOCKED_YIELDS):
        await connector.yielded.acquire()
    embedder.gate.open()
    report = await run

    assert report.stages.peak_bodies <= BLOCKED_BODIES
    assert report.stages.parse_queue.peak_depth <= report.stages.parse_queue.capacity
    assert report.stages.fetch_queue.peak_depth <= report.stages.fetch_queue.capacity
    assert report.indexed == 40


async def test_a_blocked_embedder_leaves_the_hand_offs_at_their_capacity_and_no_further() -> None:
    """The queues are bounded by their configuration rather than by how fast anything is.

    The distinction matters because an unbounded queue looks identical to a bounded one whenever
    the consumer keeps up. Blocking the consumer is what tells them apart, and the capacities
    here are two and two rather than forty.
    """
    embedder = fakes.GatedEmbedder()
    pipeline, _, _ = build(
        embedder=embedder, fetch_concurrency=2, parse_workers=1, queue_depth_factor=1
    )
    connector = fakes.ObservedConnector(corpus(40))

    run = asyncio.create_task(pipeline.run(connector))
    await embedder.gate.wait_for(1)
    for _ in range(BLOCKED_YIELDS):
        await connector.yielded.acquire()
    embedder.gate.open()
    report = await run

    assert report.stages.fetch_queue.capacity == 2
    assert report.stages.parse_queue.capacity == 2
    assert report.stages.accepted == 40


# --- one document's failure is one document's ---------------------------------------------------


async def test_several_concurrent_fetch_failures_leave_every_other_document_indexed() -> None:
    """A 503 on three documents while five others are in flight costs three documents."""
    pipeline, store, _ = build(fetch_concurrency=4, parse_workers=2)
    connector = fakes.ObservedConnector(corpus(8))
    connector.fail_fetch = {"doc-001", "doc-003", "doc-005"}

    report = await pipeline.run(connector)

    assert report.indexed == 5
    assert report.by_status[DocumentStatus.FAILED.value] == 3
    assert report.discovered == 8
    for source_id in ("doc-000", "doc-002", "doc-004", "doc-006", "doc-007"):
        document = await store.find_document("memory", source_id)
        assert document is not None
        assert document.status is DocumentStatus.INDEXED


async def test_a_parse_worker_killed_on_one_document_leaves_the_batch_alone() -> None:
    """A killed worker is a hard failure for its document and nothing else.

    The pool replaces the worker and the run continues, which was already true; what is new is
    that the documents beside it were in flight at the time rather than waiting behind it.
    """
    runner = fakes.BrokenRunner({"lines": fakes.LineParser()}, kill="doc-002")
    pipeline, store, _ = build(runner=runner, fetch_concurrency=4, parse_workers=2)

    report = await pipeline.run(fakes.ObservedConnector(corpus(8)))

    assert runner.killed == ["doc-002"]
    assert report.indexed == 7
    killed = await store.find_document("memory", "doc-002")
    assert killed is not None
    assert killed.status is DocumentStatus.FAILED
    assert "worker killed" in (killed.status_detail or "")


async def test_an_embedder_that_fails_every_batch_still_reaches_a_terminal_outcome_for_each() -> (
    None
):
    """A failure must release its place in the stages, or the run never returns.

    The corpus is larger than every bound put together, so a document whose failure kept its
    accounting would leave the hand-off in front of it permanently full and this test would hang
    rather than fail. That is what the deadline is for.
    """
    pipeline, store, _ = build(
        embedder=fakes.RefusingEmbedder(),
        fetch_concurrency=2,
        parse_workers=1,
        queue_depth_factor=1,
    )

    async with asyncio.timeout(20):
        report = await pipeline.run(fakes.ObservedConnector(corpus(30)))

    assert report.discovered == 30
    assert report.by_status[DocumentStatus.FAILED.value] == 30
    assert len(store.documents) == 30, "every one of them is a row, so every one is re-queryable"


async def test_a_commit_that_fails_does_not_strand_a_place_in_the_stages() -> None:
    """The same claim one stage later: a store refusing vectors must not wedge the pipeline."""
    pipeline, store, _ = build(
        vectors=fakes.RefusingVectors(),
        fetch_concurrency=2,
        parse_workers=1,
        queue_depth_factor=1,
    )

    async with asyncio.timeout(20):
        report = await pipeline.run(fakes.ObservedConnector(corpus(30)))

    assert report.by_status[DocumentStatus.FAILED.value] == 30
    assert all(document.status is DocumentStatus.FAILED for document in store.documents.values()), (
        "a document that could not commit is failed, not left in flight"
    )


async def test_a_connector_whose_cleanup_raises_ends_the_run_instead_of_hanging_it() -> None:
    """Closing somebody else's async generator is the one thing discovery does that can fail.

    A connector's own ``finally`` is code this repository did not write, and it runs while
    discovery is holding the only thing that tells the fetch stage no more is coming. Raising
    there skips that, so every fetch worker is left waiting for an item that will never arrive.

    **What makes the run return anyway is the task group**, not the ordering inside the
    ``finally``. Discovery raising is a stage failure, so the group cancels the workers where
    they are blocked and the command ends with the failure recorded. That is worth pinning
    precisely because it is not obvious: the sibling case one stage later — a fetch worker
    exiting *normally* without closing the parse hand-off — has no exception to trigger the
    cancellation, and it hangs.

    ``--limit`` is what reaches this path: discovery breaks with the generator still suspended,
    so the close is a real ``aclose`` rather than an exhausted iterator running its own
    ``finally``. The deadline is the assertion.
    """

    class BadCleanup(_Positioned):
        @override
        async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
            try:
                async for found in super().discover(watermark):
                    yield found
            finally:
                msg = "the connector's cursor teardown raised"
                raise RuntimeError(msg)

    pipeline, store, _ = build(fetch_concurrency=2, parse_workers=1, queue_depth_factor=1)

    async with asyncio.timeout(20):
        report = await pipeline.run(BadCleanup(corpus(20)), limit=2)

    assert not report.clean
    assert "cursor teardown" in report.error
    assert store.watermarks == {}


async def test_a_dead_document_store_ends_the_run_rather_than_being_absorbed() -> None:
    """The one failure that is *not* one document's, and it must not be counted as one.

    Every stage already turns what a document can do into a recorded outcome, so what reaches
    the task group is the store itself going away. A run that carried on would report every
    remaining document as failed, finish clean, and advance a watermark past a corpus it never
    wrote — which is the silent-loss shape this whole file exists around.
    """

    class DeadStore(fakes.MemoryIngestStore):
        @override
        async def upsert_document(self, document: Document) -> Document:
            msg = "the document store went away mid-sync"
            raise RuntimeError(msg)

    pipeline, _, _ = build(store=DeadStore(), fetch_concurrency=2, parse_workers=1)

    report = await pipeline.run(_Positioned(corpus(10)))

    assert not report.clean
    assert "went away" in report.error
    assert not report.complete


# --- watermarks -------------------------------------------------------------------------------


class _Positioned(fakes.ObservedConnector):
    """A connector that can say how far it got, so a watermark write is observable."""

    def __init__(self, documents: Mapping[str, str], *, name: str = "memory") -> None:
        super().__init__(documents, name=name)
        self.position = "position-1"
        self.fail_after: int | None = None

    @property
    @override
    def watermark(self) -> Watermark:
        from datetime import UTC, datetime  # noqa: PLC0415 - one test helper, one import

        from manicule.core.sources import Watermark as Mark  # noqa: PLC0415

        return Mark(value=self.position, observed_at=datetime.now(UTC))

    @override
    async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
        emitted = 0
        async for found in super().discover(watermark):
            if self.fail_after is not None and emitted >= self.fail_after:
                msg = "the search cursor expired"
                raise CursorExpiredError(msg)
            emitted += 1
            yield found


def _last_run(store: fakes.MemoryIngestStore) -> dict[str, Any]:
    return cast("dict[str, Any]", store.connector_meta["memory"]["last_run"])


async def test_a_complete_run_advances_the_watermark_however_the_documents_finished() -> None:
    """Completion order is not discovery order, and the position is still the one write.

    The watermark is a single value describing a whole enumeration, so there is no partial
    version of it to get wrong — which is exactly why it must not be written unless the whole
    enumeration landed. The baseline for the three tests below.
    """
    pipeline, store, _ = build(fetch_concurrency=4, parse_workers=2)

    report = await pipeline.run(_Positioned(corpus(12)))

    assert report.complete
    assert report.watermark_advanced
    assert store.watermarks["memory"].value == "position-1"


async def test_a_partial_discovery_does_not_advance_the_watermark() -> None:
    """Documents that were enumerated once and never stored must stay re-enumerable."""
    pipeline, store, _ = build(fetch_concurrency=4, parse_workers=2)
    connector = _Positioned(corpus(12))
    connector.fail_after = 4

    report = await pipeline.run(connector)

    assert not report.clean
    assert not report.watermark_advanced
    assert report.error == "CursorExpiredError: source enumeration failed"
    recorded = _last_run(store)
    assert recorded["outcome"] == "incomplete"
    assert recorded["enumeration_completed"] is False
    assert recorded["retry_required"] is True
    assert recorded["error_type"] == "CursorExpiredError"
    assert store.watermarks == {}, "a run that did not finish enumerating has no position to save"


async def test_a_complete_retry_after_cursor_expiry_advances_the_watermark_once() -> None:
    """Partial documents stay durable, and the next full walk is the one that checkpoints."""

    class CountingStore(fakes.MemoryIngestStore):
        def __init__(self) -> None:
            super().__init__()
            self.watermark_writes = 0

        @override
        async def set_watermark(self, connector: str, watermark: Watermark) -> None:
            self.watermark_writes += 1
            await super().set_watermark(connector, watermark)

    store = CountingStore()
    pipeline, _, _ = build(store=store, fetch_concurrency=4, parse_workers=2)
    connector = _Positioned(corpus(12))
    connector.fail_after = 4

    failed = await pipeline.run(connector)
    assert failed.indexed == 4
    assert store.watermark_writes == 0
    assert len(store.documents) == 4

    connector.fail_after = None
    completed = await pipeline.run(connector)
    assert completed.complete
    assert completed.watermark_advanced
    assert store.watermark_writes == 1
    assert len(store.documents) == 12
    recorded = _last_run(store)
    assert recorded["outcome"] == "complete"
    assert recorded["retry_required"] is False


async def test_a_watermark_store_failure_is_an_incomplete_result_with_partial_counters() -> None:
    class RefusingCheckpoint(fakes.MemoryIngestStore):
        @override
        async def set_watermark(self, connector: str, watermark: Watermark) -> None:
            del connector, watermark
            msg = "the checkpoint store is unavailable"
            raise OSError(msg)

    store = RefusingCheckpoint()
    pipeline, _, _ = build(store=store, fetch_concurrency=4, parse_workers=2)

    report = await pipeline.run(_Positioned(corpus(4)))

    assert report.indexed == 4
    assert report.error_type == "OSError"
    assert not report.complete
    recorded = _last_run(store)
    assert recorded["outcome"] == "incomplete"
    assert recorded["enumeration_completed"] is True
    assert recorded["retry_required"] is True
    assert recorded["watermark_advanced"] is False


async def test_a_document_that_left_no_record_at_all_holds_the_watermark_back() -> None:
    """The one failure a watermark can hide for ever, and the reason ``unrecorded`` exists.

    A fetch that fails for a document the index has never seen writes nothing: there is no row
    to record the failure against, so it is not in any listing and no repair can select it. If
    the position advanced past it, no later sync would enumerate it either — the document exists
    at the source, was seen once, and is nowhere, with nothing raised.

    A document that failed *and left a row* is a different case and does not hold the watermark:
    it is stored, countable and repairable, which is what makes it findable at all.
    """
    pipeline, store, _ = build(fetch_concurrency=4, parse_workers=2)
    connector = _Positioned(corpus(6))
    connector.fail_fetch = {"doc-003"}

    report = await pipeline.run(connector)

    assert report.clean, "a fetch failure is not an enumeration failure"
    assert report.unrecorded == 1
    assert not report.complete
    recorded = _last_run(store)
    assert recorded["outcome"] == "incomplete"
    assert recorded["enumeration_completed"] is True
    assert recorded["retry_required"] is True
    assert store.watermarks == {}


async def test_a_failure_that_did_leave_a_row_does_not_hold_the_watermark_back() -> None:
    """The other side of the same rule, so ``unrecorded`` cannot quietly become "any failure".

    A parse failure stores the document as ``failed`` with its stage. It is in every listing, it
    is selectable by ``document reindex``, and the next sync compares it again — so nothing is
    lost by moving the position past it, and holding the position back for it would make one
    permanently broken document re-enumerate a whole corpus for ever.
    """
    pipeline, store, _ = build(
        parsers={"lines": fakes.ExplodingParser()}, fetch_concurrency=4, parse_workers=2
    )

    report = await pipeline.run(_Positioned(corpus(6)))

    assert report.by_status[DocumentStatus.FAILED.value] == 6
    assert report.unrecorded == 0
    assert report.complete
    assert report.watermark_advanced
    assert store.watermarks["memory"].value == "position-1"


# --- `--limit` ---------------------------------------------------------------------------------


async def test_a_limit_accepts_no_more_top_level_documents_than_it_asked_for() -> None:
    """The limit bounds acceptance, because under concurrency completion is too late to ask.

    A sequential loop could check the count after each document and stop; a staged run has
    several in flight when the tenth is accepted, so the bound has to be applied where discovery
    hands work downstream. Everything accepted is still carried to a terminal outcome.
    """
    pipeline, store, _ = build(fetch_concurrency=4, parse_workers=2)
    connector = _Positioned(corpus(40))

    report = await pipeline.run(connector, limit=10)

    assert report.stages.accepted == 10
    assert report.discovered == 10, "and every acceptance produced exactly one top-level outcome"
    assert report.indexed == 10
    assert len(store.documents) == 10, "nothing was abandoned in a queue"


async def test_a_run_stopped_by_a_limit_is_bounded_rather_than_clean() -> None:
    """A prefix of a corpus is not a corpus, and the watermark is what would believe it was.

    ``--limit 10`` against ten thousand documents that advanced a position would skip the other
    nine thousand nine hundred and ninety on every subsequent sync, silently.
    """
    pipeline, store, _ = build(fetch_concurrency=4, parse_workers=2)

    report = await pipeline.run(_Positioned(corpus(40)), limit=10)

    assert report.clean, "nothing went wrong"
    assert report.limited, "but discovery stopped before the source was exhausted"
    assert not report.complete
    assert store.watermarks == {}
    recorded = _last_run(store)
    assert recorded["outcome"] == "bounded"
    assert recorded["enumeration_completed"] is False
    assert recorded["retry_required"] is False
    assert recorded["watermark_advanced"] is False


async def test_members_found_inside_a_container_do_not_consume_the_limit() -> None:
    """One archive of five hundred members must not exhaust a limit of ten.

    Checked with the limit actually binding rather than with a report that happens to separate
    the counters, because the failure being ruled out is a limit consuming acceptances it should
    not have counted.
    """
    pipeline, _, _ = build(
        parsers={"archive": fakes.FakeArchive(), "lines": fakes.LineParser()},
        chain=("archive", "lines"),
        fetch_concurrency=4,
        parse_workers=2,
    )
    bundles = {f"bundle-{n}": f"one=alpha {n}\ntwo=beta {n}\nthree=gamma {n}" for n in range(6)}
    connector = fakes.ObservedConnector(bundles)
    for source_id in bundles:
        connector.media_types[source_id] = fakes.CONTAINER_MEDIA_TYPE

    report = await pipeline.run(connector, limit=2)

    assert report.discovered == 2, "two archives, and the limit counted archives"
    assert report.expanded == 6, "and their six members were all carried to an outcome"
    assert report.stages.accepted == 2


# --- cancellation and shutdown -------------------------------------------------------------------


async def test_pressing_control_c_cancels_and_joins_every_stage() -> None:
    """No background task may survive the command that created it.

    The stages are children of :meth:`~manicule.ingest.pipeline.IngestPipeline.run`, so a
    cancellation that returned while a fetch worker was still running would be a task writing to
    a store after the command that opened it had gone. Counted rather than argued: the set of
    tasks alive afterwards is the set that was alive before.
    """
    embedder = fakes.GatedEmbedder()
    pipeline, store, _ = build(
        embedder=embedder,
        fetch_concurrency=2,
        parse_workers=1,
        queue_depth_factor=1,
        shutdown_grace_s=0.0,
    )
    connector = _Positioned(corpus(40))
    before = len(asyncio.all_tasks())

    run = asyncio.create_task(pipeline.run(connector))
    await embedder.gate.wait_for(1)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert len(asyncio.all_tasks()) == before, "a stage outlived the run that created it"
    assert store.watermarks == {}, "a canceled run is an incomplete run"


async def test_a_graceful_stop_carries_what_was_already_accepted_to_a_terminal_outcome() -> None:
    """``shutdown_grace_s``, made true: a document mid-embed finishes rather than being redone.

    The setting has been configurable and read by nothing. What it buys is that work already
    accepted lands instead of being requeued — so what this asserts is that nothing accepted is
    left in a non-terminal status, which is the difference between draining and abandoning.
    """
    embedder = fakes.GatedEmbedder()
    pipeline, store, _ = build(
        embedder=embedder,
        fetch_concurrency=2,
        parse_workers=1,
        queue_depth_factor=1,
        shutdown_grace_s=30.0,
    )
    connector = _Positioned(corpus(40))

    run = asyncio.create_task(pipeline.run(connector))
    await embedder.gate.wait_for(1)
    run.cancel()
    # The grace window is what lets the parked document out, so the gate opens after the
    # cancellation rather than before it: this is the drain, not the happy path.
    embedder.gate.open()
    with pytest.raises(asyncio.CancelledError):
        await run

    settled = {DocumentStatus.INDEXED, DocumentStatus.FAILED, DocumentStatus.NO_EXTRACTABLE_TEXT}
    assert store.documents, "the grace window drained nothing at all"
    assert all(document.status in settled for document in store.documents.values()), (
        "a document was accepted, given the grace window, and still left in flight"
    )
    assert store.watermarks == {}


async def test_a_canceled_run_stops_pulling_the_source_even_while_it_drains() -> None:
    """The grace window finishes accepted work; it is not a license to keep reading the source.

    Without this, ``Ctrl-C`` on a large sync would keep enumerating and keep fetching for the
    whole grace window — a canceled command still hammering a rate-limited API, which is the
    opposite of what a person pressing ``Ctrl-C`` is asking for.

    **The sequence is what makes it airtight.** The stages are wedged behind a parked embedder
    with discovery blocked inside a ``put``, so nothing more can be produced until room appears.
    The cancellation goes in first and the gate opens second, so room appears *after* the stop —
    and every document produced from then on is one the stop failed to prevent.
    """
    embedder = fakes.GatedEmbedder()
    pipeline, _, _ = build(
        embedder=embedder,
        fetch_concurrency=2,
        parse_workers=1,
        queue_depth_factor=1,
        shutdown_grace_s=30.0,
    )
    connector = _Positioned(corpus(40))

    run = asyncio.create_task(pipeline.run(connector))
    await embedder.gate.wait_for(1)
    for _ in range(BLOCKED_YIELDS):
        await connector.yielded.acquire()
    run.cancel()
    embedder.gate.open()
    with pytest.raises(asyncio.CancelledError):
        await run

    # One more than the wedged count: discovery was suspended inside the ninth `put`, and it
    # reads the stop at the top of the loop, so the tenth document is produced by the source and
    # then dropped rather than never produced. Abandoning one enumerated document costs nothing —
    # no watermark was written, so the next sync sees it again.
    assert connector.yields <= BLOCKED_YIELDS + 1, (
        f"the source was paged {connector.yields} times after the run was canceled"
    )


async def test_a_second_cancellation_inside_the_grace_window_stops_waiting() -> None:
    """The impatient case is safe rather than different, and the recovery path is the same.

    Nothing is released here, so the run can only end by the second cancellation cutting the
    grace short. A test that instead waited out a long grace would pass by elapsing rather than
    by the behavior it names.
    """
    embedder = fakes.GatedEmbedder()
    pipeline, store, _ = build(
        embedder=embedder,
        fetch_concurrency=2,
        parse_workers=1,
        queue_depth_factor=1,
        shutdown_grace_s=3600.0,
    )

    run = asyncio.create_task(pipeline.run(_Positioned(corpus(40))))
    await embedder.gate.wait_for(1)
    run.cancel()
    await asyncio.sleep(0)  # let the first cancellation reach the grace wait
    run.cancel()
    async with asyncio.timeout(20):
        with pytest.raises(asyncio.CancelledError):
            await run

    assert store.watermarks == {}


# --- the guards that concurrency must not weaken ---------------------------------------------


async def test_two_different_documents_may_be_written_at_the_same_time() -> None:
    """The mutation lock is keyed, and this is what the key is worth.

    A pipeline-wide lock would make every write in the installation queue behind every other,
    which is a throughput cost paid to fix a correctness problem that two unrelated documents do
    not have. Two documents inside the write sequence at once is the property, and it is checked
    through a hook that runs after chunking — inside the lock, before the commit.
    """
    inside = fakes.Gate()

    class ParkingHook(fakes.PassThrough):
        name = "parking"

        @override
        async def after_chunk(self, document: Document, chunks: list[Chunk]) -> list[Chunk]:
            del document
            await inside.pass_through()
            return chunks

    pipeline, _, _ = build(
        middleware=(ParkingHook(),),
        fetch_concurrency=4,
        parse_workers=3,
    )
    connector = fakes.ObservedConnector(corpus(12))

    run = asyncio.create_task(pipeline.run(connector))
    await inside.wait_for(2)
    inside.open()
    report = await run

    assert inside.peak >= 2, "two unrelated documents serialized on one another's writes"
    assert report.indexed == 12


async def test_one_document_is_never_written_by_two_operations_at_once() -> None:
    """The other half of the same lock: the same document id excludes itself.

    A document is published by three writes in sequence — its record, its chunks and glossary,
    its vectors — and what must not interleave is the sequence. Two ingests of the *same* id run
    concurrently here and the gate proves only one is ever inside.
    """
    inside = fakes.Gate(capacity=1, opened=True)

    class WatchingHook(fakes.PassThrough):
        name = "watching"

        @override
        async def after_chunk(self, document: Document, chunks: list[Chunk]) -> list[Chunk]:
            del document
            async with inside.holding():
                await asyncio.sleep(0)  # a real await, so an unlocked implementation interleaves
            return chunks

    pipeline, _, _ = build(middleware=(WatchingHook(),), fetch_concurrency=4, parse_workers=3)
    raw = RawDocument(
        source_id="shared", uri="memory://shared", media_type=MEDIA_TYPE, content="alpha\nbeta"
    )

    async with asyncio.TaskGroup() as both:
        for _ in range(4):
            both.create_task(pipeline.ingest_raw(raw, source="memory", force=True))

    assert inside.entries == 4, "all four ran"
    assert inside.peak == 1, "and never two at once on one document"


async def test_stale_work_cannot_overwrite_a_document_that_moved_while_it_ran() -> None:
    """The compare-and-swap, under concurrency, where it is most reachable.

    A guarded caller — every re-parse — derives its work from a snapshot. A connector sync
    committing newer bytes for the same document while that work is in flight must not lose to
    it. The guard is in the write rather than before it, so this is checked by letting the sync
    land first and then asking the stale caller to commit.
    """
    pipeline, store, _ = build(fetch_concurrency=4, parse_workers=2)
    connector = fakes.ObservedConnector({"page": "first version\nsecond line"})
    await pipeline.run(connector)

    stored = await store.find_document("memory", "page")
    assert stored is not None
    stale = stored.revision

    connector.documents["page"] = "newer version from the source\nsecond line"
    await pipeline.run(connector)

    outcomes = await pipeline.ingest_raw(
        RawDocument(
            source_id="page",
            uri="memory://page",
            media_type=MEDIA_TYPE,
            content="a re-parse of the old bytes",
        ),
        source="memory",
        force=True,
        expected=stale,
    )

    assert outcomes[0].superseded, "the stale write was not refused"
    current = await store.find_document("memory", "page")
    assert current is not None
    assert current.content_hash == content_hash("newer version from the source\nsecond line")
    assert any("newer version" in chunk.text for chunk in store.chunks[current.id]), (
        "the corpus kept the newer text, which is the whole point of refusing the older one"
    )


async def test_glossary_lineage_is_written_inside_the_guarded_sequence_under_concurrency() -> None:
    """Entries and the claim about what produced them stay one transaction when runs overlap.

    A document with entries and no recorded detector is the state versioning them exists to make
    unreachable, and a failed document must not be stamped at all — its stored text is not what
    the current detector read.
    """
    store = fakes.MemoryGlossaryStore()
    pipeline, _, _ = build(store=store, fetch_concurrency=4, parse_workers=2)
    definitions = {
        f"page-{n}": f"NOW - Network Operations Workspace {n}\nThe scheduler restarts nightly."
        for n in range(12)
    }

    report = await pipeline.run(fakes.ObservedConnector(definitions))

    assert report.indexed == 12
    assert pipeline.glossary_lineage is not None
    for document in store.documents.values():
        assert store.glossary[document.id], "a document stating a definition recorded none"
        assert store.glossary_lineage_by_id[document.id] == pipeline.glossary_lineage


async def test_a_failed_document_is_never_stamped_with_a_detector_that_did_not_read_it() -> None:
    """Lineage withheld from a ``FAILED`` document, with other documents in flight beside it."""
    store = fakes.MemoryGlossaryStore()
    runner = fakes.BrokenRunner({"lines": fakes.LineParser()}, kill="doc-002")
    pipeline, _, _ = build(store=store, runner=runner, fetch_concurrency=4, parse_workers=2)

    await pipeline.run(fakes.ObservedConnector(corpus(8)))

    failed = await store.find_document("memory", "doc-002")
    assert failed is not None
    assert failed.status is DocumentStatus.FAILED
    assert failed.id not in store.glossary_lineage_by_id, (
        "a failure is the absence of a conclusion, and claiming lineage for it marks the "
        "document current on the strength of the run that could not read it"
    )
    indexed = await store.find_document("memory", "doc-003")
    assert indexed is not None
    assert indexed.id in store.glossary_lineage_by_id


# --- reuse counters under out-of-order completion -------------------------------------------------


async def test_durable_vector_reuse_counters_stay_correct_when_documents_finish_out_of_order() -> (
    None
):
    """Reuse is per document and per embedding input, so completion order cannot reach it.

    Worth checking rather than assuming: the counters are assembled inside
    :func:`~manicule.ingest.embedding.embed_or_reuse` from a vector-store read taken *outside*
    the embedding lock, which is exactly the sort of read a concurrent design invites into a
    race. The second sync changes one document out of twelve.
    """
    store = fakes.MemoryIngestStore()
    vectors = fakes.MemoryVectors()
    pipeline, _, _ = build(store=store, vectors=vectors, fetch_concurrency=4, parse_workers=3)
    connector = fakes.ObservedConnector(corpus(12))

    first = await pipeline.run(connector)
    assert first.indexed == 12
    rows_after_first = dict(vectors.rows)

    connector.documents["doc-006"] = "line one of 6 rewritten\nline two of 6"
    connector.tokens["doc-006"] = "moved"
    second = await pipeline.run(connector)

    assert second.skipped_version == 11, "eleven documents did not move at all"
    assert second.indexed == 1
    changed = await store.find_document("memory", "doc-006")
    assert changed is not None
    untouched = [
        chunk_id
        for chunk_id, row in vectors.rows.items()
        if row.document_id != changed.id and rows_after_first.get(chunk_id) == row
    ]
    assert len(untouched) == 22, "the other eleven documents kept both their vectors each"


async def test_a_second_sync_of_an_unchanged_corpus_touches_the_network_not_at_all() -> None:
    """Level-1 change detection lives in the fetch stage, so a skip still costs no fetch.

    Moving change detection across a stage boundary is precisely where this guarantee could
    have been lost — a design that fetched first and compared afterwards would look identical in
    every counter and cost a whole corpus of requests per sync.
    """
    pipeline, _, _ = build(fetch_concurrency=4, parse_workers=2)
    connector = fakes.ObservedConnector(corpus(12))
    await pipeline.run(connector)

    connector.fetches.clear()
    connector.fetching.entries = 0
    report = await pipeline.run(connector)

    assert report.skipped_version == 12
    assert connector.fetches == []
    assert connector.fetching.entries == 0


# --- reporting ---------------------------------------------------------------------------------


class _ReverseFinishing(fakes.PassThrough):
    """Makes each document finish only once the document discovered *after* it has.

    A deterministic scrambler, which is a thing concurrency tests otherwise cannot have: it does
    not make completion order *likely* to differ from discovery order, it makes it exactly the
    reverse, every time. The chain is what does it — the last document is the only one waiting
    for nobody, so it goes first and each earlier one is released by its successor.

    Requires every document to be in the ingest stage at once, so a run using this needs as many
    ingest workers as there are documents. A run that does not deadlocks, which is a loud enough
    way to find out.
    """

    name = "reverse-finishing"

    def __init__(self, order: Sequence[str]) -> None:
        self._order = list(order)
        self._done = {source_id: asyncio.Event() for source_id in self._order}

    @override
    async def after_chunk(self, document: Document, chunks: list[Chunk]) -> list[Chunk]:
        position = self._order.index(document.source_id)
        if position + 1 < len(self._order):
            await self._done[self._order[position + 1]].wait()
        self._done[document.source_id].set()
        return chunks


async def test_a_report_says_the_same_thing_however_the_documents_finished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two runs of one corpus, finishing in opposite orders, produce the same report.

    Counters do not care in what order they were incremented, so most of a report is
    order-independent for free. ``glossary_failures`` is a list, and under a staged run the order
    documents complete in is not the order they were discovered in — so without a deliberate
    ordering the same corpus produces two different reports, a diagnostic cannot be diffed, and
    a test asserting on one passes or fails on scheduling.

    **Every document fails detection here on purpose**, because a corpus with no glossary
    failures produces an empty list and an empty list is in every order at once. A test that
    used ordinary documents would assert this and check nothing.
    """

    def unreadable(*_: object, **__: object) -> object:
        msg = "the detector could not read this"
        raise RuntimeError(msg)

    monkeypatch.setattr("manicule.ingest.pipeline.detect_entries", unreadable)
    documents = corpus(6)
    reports: list[RunReport] = []
    for reversing in (False, True):
        store = fakes.MemoryGlossaryStore()
        middleware = (_ReverseFinishing(sorted(documents)),) if reversing else ()
        pipeline, _, _ = build(
            store=store,
            middleware=middleware,
            fetch_concurrency=6,
            parse_workers=5,
        )
        reports.append(await pipeline.run(fakes.ObservedConnector(documents)))

    forward, backward = reports
    assert len(forward.glossary_failures) == 6, "the detector failed on every document"
    assert forward.by_status == backward.by_status
    assert (forward.discovered, forward.expanded) == (backward.discovered, backward.expanded)
    assert forward.glossary_failures == backward.glossary_failures, (
        "one corpus produced two different reports depending on which document finished first"
    )


async def test_the_stage_counters_reach_the_connector_row_without_any_document_content() -> None:
    """``doctor`` wants queue depth and stage occupancy; it must not get a corpus with them."""
    pipeline, store, _ = build(fetch_concurrency=4, parse_workers=2)

    await pipeline.run(fakes.ObservedConnector(corpus(12)))

    recorded = store.connector_meta["memory"]["last_run"]
    assert isinstance(recorded, dict)
    stages = recorded["stages"]
    assert isinstance(stages, dict)
    assert stages["accepted"] == 12
    assert stages["peak_embeds"] == 1
    assert set(stages) == {
        "accepted",
        "fetch_queue",
        "parse_queue",
        "peak_fetches",
        "peak_parses",
        "peak_embeds",
        "peak_bodies",
        "peak_discovery_records",
    }, "the stage report grew a field, and every field here is read by an operator"


async def _until_indexed(store: fakes.MemoryIngestStore, source_id: str) -> None:
    """Wait for one document to reach ``indexed``, without ever sleeping on a guess.

    Yields to the loop until the store holds the answer. Bounded by an outer deadline so a
    pipeline that never gets there fails the test rather than hanging it.
    """
    async with asyncio.timeout(10):
        while True:
            document = await store.find_document("memory", source_id)
            if document is not None and document.status is DocumentStatus.INDEXED:
                return
            await asyncio.sleep(0)
