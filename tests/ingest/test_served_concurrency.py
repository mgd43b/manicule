"""The four guards, re-proven with a scheduler and a proxied command line driving one pipeline.

A served manicule syncs on its own configuration *and* answers write commands that arrive on the
control socket. Both reach the same :class:`~manicule.ingest.pipeline.IngestPipeline`, so the two
can be inside it at once — which is exactly the interleaving the compare-and-swap in #119, the
keyed per-document lock, the embedding partition in #120 and the glossary lineage write in #122
were built for. #138 proved each of those under staged concurrency *within* one run. This proves
each again with two runs, started by two different things, overlapping.

**And a third thing, now that one process serves MCP as well: a client reading.** Each of the
four is proven a third time with an MCP client on the network surface issuing searches
throughout — see :class:`Reading` and the section at the end. That interleaving is different in
kind from the other two rather than more of the same: the reads go through the *same event loop*
and the same ``ApplicationService`` as the syncs, so a guard that held only because the two
writers were the only things scheduled would come apart here. The reads are also the case where
a regression would be least visible, because nothing about a search reports that a write went
wrong beside it — which is why the assertion is always the guard's own, with the reader's answer
count beside it to show it was genuinely running.

**Nothing here is a fake pipeline.** The guards under test are in the pipeline, so an ingestion
port that recorded calls would prove nothing about them: :class:`PipelineIngestion` drives the
real one, and the scheduler and the socket reach it through the real
:class:`~manicule.app.service.ApplicationService`.

**Every assertion is an arrival or a maximum, never an elapsed time** — the rule
``tests/ingest/test_concurrency.py`` states and for the same reason. A test that waited would
pass against a *sequential* implementation whenever the wait happened to be long enough, which
is the one outcome that would make this whole file worthless.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

import pytest
from fastmcp import Client

from manicule.app import control
from manicule.app.served import ControlHandler, Scheduler
from manicule.app.service import ApplicationService
from manicule.config.settings import ConnectorSettings
from manicule.connectors.sessions import SessionVault
from manicule.core.content import Chunk, Document, RawDocument
from manicule.core.ids import content_hash
from manicule.core.sources import DocRef
from manicule.ingest.middleware import MiddlewareRunner
from manicule.ingest.pipeline import IngestPipeline
from manicule.ingest.workers import InProcessRunner
from tests.api.live import serving
from tests.app.fakes import FakeBackend
from tests.fakes import MEDIA_TYPE, HashEmbedder
from tests.ingest import fakes

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

    from manicule.core.content import ParsedBlock
    from manicule.core.protocols import Chunker, Connector, Embedder, Middleware
    from manicule.ingest.pipeline import RunReport, Watching


@pytest.fixture
def socket_for() -> Iterator[Callable[[], Path]]:
    made: list[Path] = []

    def build() -> Path:
        path = control.socket_path(Path(f"/manicule-suite/{uuid.uuid4()}"))
        made.append(path)
        return path

    yield build
    for path in made:
        path.unlink(missing_ok=True)


class PipelineIngestion:
    """The ingest port, over a real pipeline and a fixed set of connectors.

    Stands in for :class:`~manicule.app.runtime._Ingestion`, which needs a container, a database
    and a plugin registry to build one connector. What it must **not** stand in for is the
    pipeline: every guard this file is about lives there, so it is the real class with its real
    bounds.
    """

    def __init__(self, pipeline: IngestPipeline, connectors: dict[str, Connector]) -> None:
        self.pipeline = pipeline
        self.connectors = connectors

    async def sync(
        self,
        connector: str,
        *,
        limit: int | None = None,
        watching: Watching | None = None,
        acquire_only: bool = False,
        retain_source_bytes: bool | None = None,
    ) -> RunReport:
        return await self.pipeline.run(
            self.connectors[connector],
            limit=limit,
            watching=watching,
            acquire_only=acquire_only,
            retain_source_bytes=retain_source_bytes,
        )

    async def connector(self, name: str) -> Connector:
        return self.connectors[name]


def served(
    connectors: dict[str, Connector],
    *,
    store: fakes.MemoryIngestStore | None = None,
    embedder: Embedder | None = None,
    middleware: tuple[Middleware, ...] = (),
    fetch_concurrency: int = 4,
    parse_workers: int = 3,
    chunker: Chunker | None = None,
) -> tuple[ApplicationService, PipelineIngestion, fakes.MemoryIngestStore]:
    """A service whose syncs go through one real pipeline, with every stage bound stated.

    The bounds are arguments for the reason ``test_concurrency.py`` gives: a test that took
    ``default_worker_count()`` would be asserting against a number that depends on the machine,
    which is the one thing a bound must not depend on.
    """
    store = store or fakes.MemoryIngestStore()
    chunker = chunker or fakes.BlockChunker()
    pipeline = IngestPipeline(
        store=store,
        chunker=chunker,
        embedder=embedder or HashEmbedder(),
        vectors=fakes.MemoryVectors(),
        runner=InProcessRunner({"lines": fakes.LineParser()}),
        resolve_chain=lambda _: ["lines"],
        middleware=MiddlewareRunner(middleware),
        chunk_fingerprint=chunker.fingerprint,
        fetch_concurrency=fetch_concurrency,
        parse_workers=parse_workers,
        queue_depth_factor=2,
        shutdown_grace_s=30.0,
    )
    ingestion = PipelineIngestion(pipeline, connectors)
    backend = FakeBackend()
    backend.ingestion_ = ingestion  # type: ignore[assignment]
    service = ApplicationService(backend)
    # This harness deliberately has no BlobSink. Match its service-level policy to that
    # capability so a direct warm-up sync and a proxied sync exercise the same retention mode.
    service.settings.storage.retain_source_bytes = False
    service.settings.connectors.clear()
    for name in connectors:
        service.settings.connectors[name] = ConnectorSettings.model_validate(
            {"type": "filesystem", "options": {"root": "."}}
        )
    return service, ingestion, store


class Overlap:
    """A scheduler and a proxied command line, both driving one pipeline, held open together.

    The whole point is the *overlap*, so this does not simply start two syncs and hope. The
    caller supplies a gate that both runs pass through, and :meth:`both_inside` returns only
    once the stated number of callers are in it at the same time — which is the assertion that
    the interleaving under test actually happened, rather than two runs that took turns.
    """

    def __init__(
        self, service: ApplicationService, path: Path, *, release: Callable[[], None] | None = None
    ) -> None:
        self._service = service
        self._path = path
        self._release = release
        self._clock = _Clock()
        self.scheduler = Scheduler(service, {"scheduled": 600}, sleep=self._clock.sleep)
        self.server = control.ControlServer(path, ControlHandler(service, SessionVault()))
        self._proxied: asyncio.Task[dict[str, Any]] | None = None

    async def __aenter__(self) -> Overlap:
        await self.server.start()
        self.scheduler.start()
        # The scheduler's loop has to reach its first sleep before a tick can mean anything.
        await asyncio.wait_for(self._clock.arrived.wait(), timeout=5)
        return self

    async def __aexit__(self, *exc: object) -> None:
        del exc
        # Anything parked in a gate is let go **first**, and this is load-bearing rather than
        # tidy. `ControlServer.aclose` waits for the connections in flight — deliberately, since
        # each is a write command and tearing one down mid-write is how a document ends up half
        # written — so a run still parked inside a hook would make shutdown wait for ever.
        #
        # That is not hypothetical: it is what disabling the keying did. The test failed
        # correctly, at `wait_for`, and then hung in teardown, which turns a clear diagnosis into
        # a suite that has to be killed.
        if self._release is not None:
            self._release()
        if self._proxied is not None and not self._proxied.done():
            self._proxied.cancel()
            await asyncio.gather(self._proxied, return_exceptions=True)
        await self.scheduler.aclose()
        await self.server.aclose()

    def start_scheduled(self) -> None:
        """Release the scheduler's wait, so its sync begins."""
        self._clock.arrived.clear()
        self._clock.release()

    def start_proxied(self, source: str = "proxied") -> None:
        """Send a sync over the real control socket, as a command line would."""
        self._proxied = asyncio.create_task(
            control.connect(
                self._path,
                control.Invoke(op="connector_sync", arguments={"name": source, "limit": None}),
                on_progress=lambda _: None,
            ),
            name="proxied-sync",
        )

    async def proxied_result(self) -> dict[str, Any]:
        assert self._proxied is not None
        return await asyncio.wait_for(self._proxied, timeout=30)

    async def scheduled_finished(self) -> None:
        """Return once the scheduled sync has completed, not merely started.

        **The waiting is the assertion, and it is an arrival rather than a duration.** The
        scheduler's loop syncs and then asks for its next sleep, so the clock being asked again
        *is* the run having finished — there is no counter to poll and no interval to guess at.
        A run that never finishes fails on the timeout instead of on a count that had not caught
        up yet.
        """
        await asyncio.wait_for(self._clock.arrived.wait(), timeout=30)


class _Clock:
    """``asyncio.sleep`` a test releases by hand, so a scheduled sync starts when asked."""

    def __init__(self) -> None:
        self.arrived = asyncio.Event()
        self._release = asyncio.Event()

    async def sleep(self, seconds: float) -> None:
        del seconds
        self.arrived.set()
        await self._release.wait()
        self._release.clear()

    def release(self) -> None:
        self._release.set()


class GatedConnector(fakes.ObservedConnector):
    """A source that parks in ``fetch`` until a test lets it through.

    **The only place outside the per-document lock that a test can hold two runs.** Every
    middleware hook — ``before_parse`` included — runs *inside*
    :meth:`~manicule.ingest.pipeline.IngestPipeline._mutating`, so a gate in one of them can
    never hold two runs of the same document at once: that is precisely what the lock prevents,
    and a test that tried would deadlock rather than assert.

    Fetching happens before the lock is taken. Parking both runs here and releasing them on one
    tick is what puts them into the write sequence together — which is what makes the
    capacity-one gate inside it a statement about exclusion rather than about scheduling luck.

    **This was not a refinement.** Without it the same-document test passed with the lock
    removed: the two runs simply did not overlap, so nothing was excluded and nothing said so.
    """

    def __init__(self, documents: Mapping[str, str], *, name: str, gate: fakes.Gate) -> None:
        super().__init__(documents, name=name)
        self._gate = gate

    @override
    async def fetch(self, ref: DocRef) -> RawDocument:
        fetched = await super().fetch(ref)
        await self._gate.pass_through()
        return fetched


def _open_both(first: fakes.Gate, second: fakes.Gate) -> Callable[[], None]:
    """Release two gates on shutdown, as one callable with nothing to return."""

    def release() -> None:
        first.open()
        second.open()

    return release


def corpus(count: int, *, prefix: str) -> dict[str, str]:
    return {
        f"{prefix}-{number:03d}": f"line one of {number}\nline two of {number}"
        for number in range(count)
    }


class Reading:
    """An MCP client on the network surface, searching in a loop until the block ends.

    The third interleaving. It is a *client* over a real socket rather than a call into the
    service, because that is what makes it a third scheduling participant: the request arrives on
    the transport, is handled in the same event loop the two writers are running in, and answers
    through the same ``ApplicationService`` they are writing through.

    ``answered`` is asserted by every test that uses this, and the reason is the reason every
    negative assertion needs a control: a reader that failed to connect, or that stopped after
    its first search, would make "the guard held with a reader running" a statement about no
    reader at all — and the guard assertions would all still pass.
    """

    def __init__(self, backend: FakeBackend) -> None:
        self._backend = backend
        self._stack = AsyncExitStack()
        self._task: asyncio.Task[None] | None = None
        self._answers = asyncio.Semaphore(0)
        self.answered = 0
        self.refused: list[str] = []
        """Any envelope that came back ``ok: false``, so a silent failure is not silent."""

    async def __aenter__(self) -> Reading:
        live = await self._stack.enter_async_context(serving(self._backend, web=False))
        client = await self._stack.enter_async_context(live.mcp())
        # One search before the writers start, so "the reader was connected" is established
        # rather than hoped for — a connection that failed later is a different fact from one
        # that never worked.
        await self._search(client)
        # That first answer is proof of a connection rather than proof of overlap, so it is
        # consumed here: everything `answered_while` counts afterwards happened during the run.
        await self._answers.acquire()
        self._task = asyncio.create_task(self._loop(client), name="mcp-reader")
        return self

    async def __aexit__(self, *exc: object) -> None:
        del exc
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        await self._stack.aclose()

    async def answered_while(self, count: int, *, patience_s: float = 30.0) -> None:
        """Return once ``count`` more searches have been answered since this was last called.

        **This is where the overlap becomes an assertion rather than a hope.** Called at a point
        where the writers are known to be mid-run — parked in a gate, or with both syncs in
        flight — it returns only when the reader has been answered *during* that, and fails
        saying how many it managed if it never was. Waiting a fixed time and looking afterwards
        would pass against a server that answered nothing until the writes had finished.
        """
        try:
            async with asyncio.timeout(patience_s):
                for _ in range(count):
                    await self._answers.acquire()
        except TimeoutError:
            msg = (
                f"waited for {count} search(es) to be answered while the writers were running; "
                f"{self.answered} have been answered in total, and {self.refused} were refused"
            )
            raise AssertionError(msg) from None

    async def _loop(self, client: Client[Any]) -> None:
        while True:
            await self._search(client)

    async def _search(self, client: Client[Any]) -> None:
        envelope = (
            await client.call_tool("search", {"query": "retry", "limit": 1})
        ).structured_content or {}
        if envelope.get("ok"):
            self.answered += 1
            self._answers.release()
        else:
            self.refused.append(str(envelope.get("error")))


# --- progress, from the real pipeline ---------------------------------------------------------


async def test_a_proxied_sync_reports_progress_from_the_real_pipeline(
    socket_for: Callable[[], Path],
) -> None:
    """The message a person watching a long sync reads, produced by the pipeline that produces it.

    **This test exists because writing it found a defect.** The first version of the progress
    line read counters off ``RunReport`` that are not on it, so the ``AttributeError`` propagated
    out of an ingest worker, ended the run through the task group, and left a sync that had
    indexed 4 of 20 documents reporting ``ok``. A fake ingestion cannot catch that — it formats
    its own string — so the assertion has to be made against a real run, and it has to be about
    the *content* of the message rather than about a message having arrived.
    """
    path = socket_for()
    service, _, store = served(
        {"proxied": fakes.ObservedConnector(corpus(12, prefix="doc"), name="handbook")},
        fetch_concurrency=2,
        parse_workers=2,
    )
    server = control.ControlServer(path, ControlHandler(service, SessionVault()))
    await server.start()
    seen: list[str] = []
    try:
        answered = await control.connect(
            path,
            control.Invoke(op="connector_sync", arguments={"name": "proxied", "limit": None}),
            on_progress=seen.append,
        )
    finally:
        await server.aclose()

    assert answered["ok"] is True
    data = answered["data"]
    assert isinstance(data, dict)
    assert data["error"] == "", f"the run recorded an error: {data['error']}"
    assert data["ingested"] == 12, "the run did not index the whole corpus"
    assert len(store.documents) == 12
    assert seen, "a twelve-document sync reported no progress at all"
    assert seen[-1] == "handbook: 12 of 12 settled (12 indexed, 0 unchanged)", seen
    assert all("handbook" in line for line in seen), seen


async def test_a_sync_that_changes_nothing_still_reports_progress(
    socket_for: Callable[[], Path],
) -> None:
    """The longest quiet run there is, and the one that reported nothing at all.

    **Found by running it.** A document that skips on change detection never reaches the ingest
    stage — that is the whole point of putting the check in the fetch stage — so a resync of a
    corpus nobody has touched settled every document and said nothing until it finished. On a
    ten-thousand-page corpus that is precisely the long silent sync progress exists to prevent,
    and it is the *commonest* shape a scheduled resync takes.
    """
    path = socket_for()
    service, ingestion, _ = served(
        {"proxied": fakes.ObservedConnector(corpus(8, prefix="doc"), name="handbook")},
        fetch_concurrency=2,
        parse_workers=2,
    )
    # Indexed once, so the run under test finds every document unchanged.
    await ingestion.sync("proxied")

    server = control.ControlServer(path, ControlHandler(service, SessionVault()))
    await server.start()
    seen: list[str] = []
    try:
        answered = await control.connect(
            path,
            control.Invoke(op="connector_sync", arguments={"name": "proxied", "limit": None}),
            on_progress=seen.append,
        )
    finally:
        await server.aclose()

    data = answered["data"]
    assert isinstance(data, dict)
    assert data["ingested"] == 0, "the second run should have found nothing to do"
    assert data["skipped"] == 8
    assert seen, "a run that skipped eight documents reported nothing at all"
    assert seen[-1] == "handbook: 8 of 8 settled (0 indexed, 8 unchanged)", seen


# --- the keyed per-document lock ------------------------------------------------------------------


async def test_two_documents_from_two_runs_may_be_written_at_the_same_time(
    socket_for: Callable[[], Path],
) -> None:
    """The key is worth what it was worth before, across runs as well as within one.

    A pipeline-wide lock would make a scheduled sync and a proxied one queue behind each other
    entirely — a throughput cost paid to fix a correctness problem that two *unrelated*
    documents do not have. What must hold is that two documents are inside the write sequence at
    once; what must not is that one document is.
    """
    inside = fakes.Gate()

    class Parking(fakes.PassThrough):
        name = "parking"

        @override
        async def after_chunk(self, document: Document, chunks: list[Chunk]) -> list[Chunk]:
            del document
            await inside.pass_through()
            return chunks

    service, _, _ = served(
        {
            "scheduled": fakes.ObservedConnector(corpus(6, prefix="sched"), name="scheduled"),
            "proxied": fakes.ObservedConnector(corpus(6, prefix="prox"), name="proxied"),
        },
        middleware=(Parking(),),
    )
    async with Overlap(service, socket_for(), release=inside.open) as both:
        both.start_scheduled()
        both.start_proxied()
        await inside.wait_for(2)
        inside.open()
        answered = await both.proxied_result()

    assert inside.peak >= 2, "a scheduled sync and a proxied one serialized on one another's writes"
    assert answered["ok"] is True


async def test_one_document_is_never_written_by_the_scheduler_and_the_proxy_at_once(
    socket_for: Callable[[], Path],
) -> None:
    """The other half of the same lock, and the case a served manicule makes reachable.

    Both runs are of **the same source**, which is the realistic collision: a schedule comes
    round while somebody types ``manicule connector sync handbook``. The corpus is deliberately
    **one document**, so that every arrival at the gate is an arrival at *that* document's write
    sequence and a capacity of one is a statement about it.

    Two documents would make this test wrong in the direction that looks right: the lock is
    keyed, so unrelated documents are supposed to overlap, and a shared capacity-1 gate over a
    corpus of eight would fail on exactly the behavior the previous test asserts. That is not a
    hypothetical — it is what the first version of this file did.
    """
    inside = fakes.Gate(capacity=1, opened=True)
    fetched = fakes.Gate()

    class Watching(fakes.PassThrough):
        name = "watching"

        @override
        async def after_chunk(self, document: Document, chunks: list[Chunk]) -> list[Chunk]:
            del document
            async with inside.holding():
                await asyncio.sleep(0)  # a real await, so an unlocked implementation interleaves
            return chunks

    # One document, and the two runs report **different bytes for it**. Identical bytes would
    # make whichever run arrived second skip on change detection, so only one would ever reach
    # the write sequence and a capacity-1 gate would pass against no exclusion at all.
    service, _, _ = served(
        {
            "scheduled": GatedConnector(
                {"the-one-page": "as the schedule found it"}, name="handbook", gate=fetched
            ),
            "proxied": GatedConnector(
                {"the-one-page": "as the command line found it"}, name="handbook", gate=fetched
            ),
        },
        middleware=(Watching(),),
    )
    async with Overlap(service, socket_for(), release=_open_both(fetched, inside)) as both:
        both.start_scheduled()
        both.start_proxied()
        # Both runs have fetched and neither has taken the lock. Releasing them on one tick is
        # what makes the next thing they do a race for it.
        await fetched.wait_for(2)
        fetched.open()
        answered = await both.proxied_result()
        await both.scheduled_finished()

    assert answered["ok"] is True
    assert inside.entries == 2, "the two runs did not both reach the write sequence"
    assert inside.peak == 1, "one document was inside its write sequence twice at once"


# --- the compare-and-swap -------------------------------------------------------------------------


async def test_a_stale_guarded_write_loses_to_a_scheduled_sync_that_moved_the_document(
    socket_for: Callable[[], Path],
) -> None:
    """#119's guard, with the newer bytes committed by the scheduler rather than by a caller.

    A guarded caller — every re-parse — derives its work from a snapshot. Once a server exists,
    the thing most likely to move that document underneath it is a sync nobody typed. The guard
    is in the write rather than before it, so this lets the scheduled sync land and then asks
    the stale caller to commit.

    **Two of the guard's three points are shown by this test and the third is not**, which is
    worth recording rather than leaving to be discovered. Disabling the record write's
    ``expected``, or the ``indexed`` write's, turns this red with "the stale write was not
    refused". Disabling the third — the guarded publish at the head of ``_commit`` — turns
    nothing red here, in ``tests/ingest/test_concurrency.py``, or in
    ``tests/ingest/test_reindex.py``. That point catches a document moving *between* the record
    write and the commit, and :meth:`~manicule.ingest.pipeline.IngestPipeline._mutating` makes
    that unreachable from inside this process: its own docstring says reaching it means a second
    process is writing the data directory without the instance lock. It is defense in depth
    against a state this process cannot produce, so it has no test, and that is a gap in the
    suite rather than in the guard.
    """
    connector = fakes.ObservedConnector({"page": "first version\nsecond line"}, name="handbook")
    service, ingestion, store = served(
        {"scheduled": connector, "proxied": connector}, fetch_concurrency=2, parse_workers=2
    )

    async with Overlap(service, socket_for()) as both:
        both.start_scheduled()
        # The whole run, not the first row it wrote. Reading a revision out of a sync still in
        # flight would make the snapshot this test is about a different one each time it ran.
        await both.scheduled_finished()
        stored = await store.find_document("handbook", "page")
        assert stored is not None
        stale = stored.revision

        # The connector's own copy, not the dictionary it was built from: `DictConnector`
        # copies, so mutating the original changes nothing the source will report.
        connector.documents["page"] = "newer version from the source\nsecond line"
        both.start_proxied()
        answered = await both.proxied_result()
        assert answered["ok"] is True

        outcomes = await ingestion.pipeline.ingest_raw(
            RawDocument(
                source_id="page",
                uri="memory://page",
                media_type=MEDIA_TYPE,
                content="a re-parse of the old bytes",
            ),
            source="handbook",
            force=True,
            expected=stale,
        )

    assert outcomes[0].superseded, "the stale write was not refused"
    current = await store.find_document("handbook", "page")
    assert current is not None
    assert current.content_hash == content_hash("newer version from the source\nsecond line")
    assert any("newer version" in chunk.text for chunk in store.chunks[current.id]), (
        "the corpus kept the newer text, which is the whole point of refusing the older one"
    )


# --- the embedding lock ---------------------------------------------------------------------------


async def test_the_embedder_is_never_re_entered_with_two_runs_offering_it_work(
    socket_for: Callable[[], Path],
) -> None:
    """#120's partition, under the pressure a server actually creates.

    :class:`~tests.ingest.fakes.GatedEmbedder` raises on re-entry rather than reporting a peak
    afterwards, so a single overlap fails the run instead of being averaged away — and it parks
    *inside the model*, which is what makes the re-entry reachable at all.

    **The parking is not decoration.** Without it this test passed with the embedding lock
    removed: the fake embedder has no ``await`` that yields, so two callers could never
    interleave inside it and "no overlap" said nothing about the lock. Holding one caller inside
    the model while twenty more documents are ready behind it is what turns the assertion into a
    statement about exclusion.

    Twenty documents on each side, four fetch workers and four ingest workers, so work is
    offered continuously. Every document is a different one; the per-document claims are
    asserted separately above rather than inferred from this.
    """
    embedder = fakes.GatedEmbedder()
    service, _, store = served(
        {
            "scheduled": fakes.ObservedConnector(corpus(20, prefix="sched"), name="scheduled"),
            "proxied": fakes.ObservedConnector(corpus(20, prefix="prox"), name="proxied"),
        },
        embedder=embedder,
        fetch_concurrency=4,
        parse_workers=3,
    )
    async with Overlap(service, socket_for(), release=embedder.gate.open) as both:
        both.start_scheduled()
        both.start_proxied()
        # One caller is held inside the model. Everything else that reaches the embedder while
        # it is there is a re-entry, and there is a great deal of it behind them.
        await embedder.gate.wait_for(1)
        embedder.gate.open()
        answered = await both.proxied_result()
        await both.scheduled_finished()

    assert answered["ok"] is True
    assert len(store.documents) == 40, "both runs did not finish, so overlap was never offered"
    # The gate counts arrivals *inside* the model, so this is "the embedder was entered many
    # times", which is the precondition for re-entry being reachable. `batches` is not used:
    # `GatedEmbedder` reaches past `CountingEmbedder` to the hash, so it stays empty.
    assert embedder.gate.entries > 2, "the embedder was entered too few times to prove anything"
    assert embedder.overlaps == 0, "the embedding lock had one holder at a time"


# --- the glossary lineage write ---------------------------------------------------------------


async def test_glossary_lineage_is_written_inside_the_guarded_sequence_across_two_runs(
    socket_for: Callable[[], Path],
) -> None:
    """#122's write, with a scheduled run and a proxied one publishing at the same time.

    A document with entries and no recorded detector is the state versioning them exists to make
    unreachable. Entries and the claim about what produced them are one transaction, and two
    runs overlapping is where a write that had drifted outside the guarded sequence would show
    it.
    """
    store = fakes.MemoryGlossaryStore()
    definitions = {
        f"page-{n}": f"NOW - Network Operations Workspace {n}\nThe scheduler restarts nightly."
        for n in range(10)
    }
    service, ingestion, _ = served(
        {
            "scheduled": fakes.ObservedConnector(dict(definitions), name="scheduled"),
            "proxied": fakes.ObservedConnector(dict(definitions), name="proxied"),
        },
        store=store,
        fetch_concurrency=4,
        parse_workers=2,
    )
    async with Overlap(service, socket_for()) as both:
        both.start_scheduled()
        both.start_proxied()
        answered = await both.proxied_result()
        await both.scheduled_finished()

    assert answered["ok"] is True
    assert ingestion.pipeline.glossary_lineage is not None
    for document in store.documents.values():
        assert store.glossary[document.id], "a document stating a definition recorded none"
        assert store.glossary_lineage_by_id[document.id] == ingestion.pipeline.glossary_lineage


# --- the same four, with an MCP client reading throughout -----------------------------------------


async def test_one_document_is_never_written_twice_while_an_mcp_client_reads(
    socket_for: Callable[[], Path],
) -> None:
    """The keyed per-document lock, with a third participant in the loop.

    The corpus is one document and the two runs carry different bytes for it, exactly as the
    two-writer version above — so every arrival at the capacity-one gate is an arrival at *that*
    document's write sequence. What is added is a client searching the whole time, which is the
    scheduling pressure a served manicule actually has and the two-writer test does not.
    """
    inside = fakes.Gate(capacity=1, opened=True)
    fetched = fakes.Gate()

    class Watching(fakes.PassThrough):
        name = "watching"

        @override
        async def after_chunk(self, document: Document, chunks: list[Chunk]) -> list[Chunk]:
            del document
            async with inside.holding():
                await asyncio.sleep(0)
            return chunks

    service, _, _ = served(
        {
            "scheduled": GatedConnector(
                {"the-one-page": "as the schedule found it"}, name="handbook", gate=fetched
            ),
            "proxied": GatedConnector(
                {"the-one-page": "as the command line found it"}, name="handbook", gate=fetched
            ),
        },
        middleware=(Watching(),),
    )

    async with (
        Reading(_backend(service)) as reader,
        Overlap(service, socket_for(), release=_open_both(fetched, inside)) as both,
    ):
        both.start_scheduled()
        both.start_proxied()
        await fetched.wait_for(2)
        # Both writers are parked between fetch and the write sequence. Anything the reader
        # is answered now is answered *during* the interleaving under test.
        await reader.answered_while(2)
        fetched.open()
        answered = await both.proxied_result()
        await both.scheduled_finished()

    assert answered["ok"] is True
    assert inside.entries == 2, "the two runs did not both reach the write sequence"
    assert inside.peak == 1, "one document was inside its write sequence twice at once"
    _reader_was_running(reader)


async def test_a_stale_guarded_write_still_loses_while_an_mcp_client_reads(
    socket_for: Callable[[], Path],
) -> None:
    """#119's compare-and-swap, with a reader in the loop.

    The same two of the guard's three points as the two-writer version: the record write's
    ``expected`` and the ``indexed`` write's. The third — the guarded publish at the head of
    ``_commit`` — is unreachable from inside one process and has no test here either, which that
    version's docstring says at length and this one does not repeat.
    """
    connector = fakes.ObservedConnector({"page": "first version\nsecond line"}, name="handbook")
    service, ingestion, store = served(
        {"scheduled": connector, "proxied": connector}, fetch_concurrency=2, parse_workers=2
    )

    async with (
        Reading(_backend(service)) as reader,
        Overlap(service, socket_for()) as both,
    ):
        both.start_scheduled()
        await both.scheduled_finished()
        stored = await store.find_document("handbook", "page")
        assert stored is not None
        stale = stored.revision

        connector.documents["page"] = "newer version from the source\nsecond line"
        both.start_proxied()
        await reader.answered_while(2)
        assert (await both.proxied_result())["ok"] is True

        outcomes = await ingestion.pipeline.ingest_raw(
            RawDocument(
                source_id="page",
                uri="memory://page",
                media_type=MEDIA_TYPE,
                content="a re-parse of the old bytes",
            ),
            source="handbook",
            force=True,
            expected=stale,
        )

    assert outcomes[0].superseded, "the stale write was not refused"
    current = await store.find_document("handbook", "page")
    assert current is not None
    assert current.content_hash == content_hash("newer version from the source\nsecond line")
    _reader_was_running(reader)


async def test_the_embedder_is_never_re_entered_while_an_mcp_client_reads(
    socket_for: Callable[[], Path],
) -> None:
    """#120's partition, with the reader adding work to the loop that is not embedding.

    The reader is the interesting part rather than a bystander: it is scheduled between the
    embedder's awaits, which is precisely the window a lock that yielded without excluding would
    let a second caller into. :class:`~tests.ingest.fakes.GatedEmbedder` raises on re-entry, so
    one overlap fails the run rather than being averaged away.
    """
    embedder = fakes.GatedEmbedder()
    service, _, store = served(
        {
            "scheduled": fakes.ObservedConnector(corpus(20, prefix="sched"), name="scheduled"),
            "proxied": fakes.ObservedConnector(corpus(20, prefix="prox"), name="proxied"),
        },
        embedder=embedder,
        fetch_concurrency=4,
        parse_workers=3,
    )

    async with (
        Reading(_backend(service)) as reader,
        Overlap(service, socket_for(), release=embedder.gate.open) as both,
    ):
        both.start_scheduled()
        both.start_proxied()
        await embedder.gate.wait_for(1)
        # One caller is held inside the model, with twenty more documents behind it on each
        # side. A search answered here is answered while the embedder is occupied.
        await reader.answered_while(2)
        embedder.gate.open()
        answered = await both.proxied_result()
        await both.scheduled_finished()

    assert answered["ok"] is True
    assert len(store.documents) == 40, "both runs did not finish, so overlap was never offered"
    assert embedder.gate.entries > 2, "the embedder was entered too few times to prove anything"
    assert embedder.overlaps == 0, "the embedding lock had one holder at a time"
    _reader_was_running(reader)


async def test_glossary_lineage_stays_inside_the_guarded_sequence_while_an_mcp_client_reads(
    socket_for: Callable[[], Path],
) -> None:
    """#122's write, with a reader in the loop.

    A document with entries and no recorded detector is the state versioning them exists to make
    unreachable. Entries and the claim about what produced them are one transaction, and three
    participants in one loop is where a write that had drifted outside the guarded sequence would
    show it.
    """
    store = fakes.MemoryGlossaryStore()
    definitions = {
        f"page-{n}": f"NOW - Network Operations Workspace {n}\nThe scheduler restarts nightly."
        for n in range(10)
    }
    service, ingestion, _ = served(
        {
            "scheduled": fakes.ObservedConnector(dict(definitions), name="scheduled"),
            "proxied": fakes.ObservedConnector(dict(definitions), name="proxied"),
        },
        store=store,
        fetch_concurrency=4,
        parse_workers=2,
    )

    async with (
        Reading(_backend(service)) as reader,
        Overlap(service, socket_for()) as both,
    ):
        both.start_scheduled()
        both.start_proxied()
        await reader.answered_while(2)
        answered = await both.proxied_result()
        await both.scheduled_finished()

    assert answered["ok"] is True
    assert ingestion.pipeline.glossary_lineage is not None
    for document in store.documents.values():
        assert store.glossary[document.id], "a document stating a definition recorded none"
        assert store.glossary_lineage_by_id[document.id] == ingestion.pipeline.glossary_lineage
    _reader_was_running(reader)


def _backend(service: ApplicationService) -> FakeBackend:
    """The fake this service was built over, for the reader to be served from.

    Through the public accessor rather than the attribute the helper set, because ``backend`` is
    what the service itself offers and a test reaching past it would be reading a different
    object from the one under test.
    """
    backend = service.backend
    assert isinstance(backend, FakeBackend)
    return backend


def _reader_was_running(reader: Reading) -> None:
    """The control every one of the four needs.

    Without it, a reader that failed to connect makes "the guard held with a client reading" a
    statement about no client at all — and every guard assertion beside it would still be green,
    because they are statements about the writers.
    """
    assert reader.refused == [], f"the reader's searches were refused: {reader.refused}"
    assert reader.answered > 1, (
        f"the reader answered {reader.answered} search(es), which is not enough to have been "
        f"running alongside the writes rather than only before them"
    )


# --- a client that goes away while one document is being prepared ------------------------------


HELD_STAGE_TIMEOUT_S = 30.0
"""How long a held stage waits to be released before it fails loudly rather than opening."""


class HeldChunker(fakes.BlockChunker):
    """A chunker that blocks synchronously, the way the production one does under a big block."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.exited = threading.Event()
        self.timed_out = False

    @override
    def chunk(self, document: Document, blocks: Iterable[ParsedBlock]) -> list[Chunk]:
        self.entered.set()
        try:
            return self._held(document, blocks)
        finally:
            # Set on every path, because a run that loses its lease is torn down without
            # waiting for this thread — so a test that read ``timed_out`` straight after the
            # run returned would read it before this thread had finished deciding. Waiting on
            # ``exited`` is what makes that flag worth asserting.
            self.exited.set()

    def _held(self, document: Document, blocks: Iterable[ParsedBlock]) -> list[Chunk]:
        if not self.release.wait(timeout=HELD_STAGE_TIMEOUT_S):
            # A gate that was never opened means the interleaving under test did not happen.
            # Proceeding anyway would let the test carry on and assert against a *different*
            # ordering — passing or failing for reasons that have nothing to do with what it
            # claims to check — so it is recorded and raised. Every test below reads
            # ``timed_out`` before its own assertions, so the failure names this rather than
            # the misleading consequence.
            self.timed_out = True
            msg = (
                f"the held preparation stage was never released within "
                f"{HELD_STAGE_TIMEOUT_S}s; the test did not reach the point where it opens "
                f"the gate"
            )
            raise AssertionError(msg)
        return super().chunk(document, blocks)


async def test_a_client_that_disconnects_during_held_preparation_does_not_strand_the_sync(
    socket_for: Callable[[], Path],
) -> None:
    """The disconnect case, taken through the stage that used to own the event loop.

    ``tests/app/test_control.py`` already proves the socket keeps server-owned work after its
    client goes, and it proves it against a handler parked on an ``asyncio.Event``. That is the
    easy half: a coroutine parked on an event is one the loop can schedule around, so the
    connection task notices the disconnect immediately and the two never interact.

    Synchronous preparation is the half that could not be asked before. It held the loop
    outright, so a close arriving on the socket was not read until the document finished — the
    server appeared to survive the disconnect only because it had not yet noticed it. Now that
    preparation runs on a worker thread the close is genuinely processed *while* the stage is
    held, which is the interleaving to pin: the sync must still finish, settle the corpus, and
    leave a socket that answers the next request.

    Raw ``asyncio.open_unix_connection`` rather than ``control.connect`` for the reason the
    control test gives — ``connect`` owns the whole exchange and cannot be abandoned partway.
    """
    path = socket_for()
    held = HeldChunker()
    documents = corpus(3, prefix="held")
    service, ingestion, store = served(
        {"proxied": fakes.ObservedConnector(documents, name="handbook")},
        chunker=held,
        fetch_concurrency=2,
        parse_workers=1,
    )

    # An arrival rather than a wait: the run's own return is what says it survived the
    # disconnect, and nothing else in the process announces it once the caller that asked for
    # it has gone.
    finished = asyncio.Event()
    running = ingestion.sync

    async def watched(*args: Any, **kwargs: Any) -> Any:
        try:
            return await running(*args, **kwargs)
        finally:
            finished.set()

    ingestion.sync = watched

    server = control.ControlServer(path, ControlHandler(service, SessionVault()))
    await server.start()
    try:
        reader, writer = await asyncio.open_unix_connection(path)
        del reader
        writer.write(
            control.Invoke(
                op="connector_sync", arguments={"name": "proxied", "limit": None}
            ).to_line()
        )
        await writer.drain()
        entered = await asyncio.wait_for(
            asyncio.to_thread(held.entered.wait, HELD_STAGE_TIMEOUT_S / 3),
            timeout=HELD_STAGE_TIMEOUT_S,
        )
        assert entered, "preparation was never reached, so the close had nothing to race"

        writer.close()
        await writer.wait_closed()
        assert not finished.is_set(), (
            "the sync finished before the client was closed, so this asserted nothing about a "
            "disconnect arriving while preparation was held"
        )

        held.release.set()
        await asyncio.wait_for(finished.wait(), timeout=30)

        await asyncio.wait_for(
            asyncio.to_thread(held.exited.wait, HELD_STAGE_TIMEOUT_S * 2),
            timeout=HELD_STAGE_TIMEOUT_S * 3,
        )
        assert not held.timed_out
        assert len(store.documents) == len(documents), (
            "the sync was abandoned when its client went away: the server owns accepted work, "
            "and a document left half-derived is what the recovery sweep exists to avoid"
        )
        answered = await control.connect(
            path, control.Invoke(op="connector_list"), on_progress=lambda _: None
        )
        assert answered["ok"] is True, "the disconnect poisoned the socket for the next caller"
    finally:
        held.release.set()
        await server.aclose()


async def test_an_http_client_that_disconnects_during_held_preparation_keeps_the_sync() -> None:
    """The socket test's guarantee, over a port, on the one network surface that can start a sync.

    ``tests/api/test_both_surfaces.py`` proves a disconnected client does not cancel a *search*,
    which parks in the retriever on the event loop. A sync is the case with something to lose:
    it holds a lease, it publishes, and the stage it spends its time in is synchronous
    preparation on a worker thread. A transport closing under that must not take the run with
    it.

    **There is deliberately no MCP half of this.** The served MCP server is built
    ``read_only=True``, so ``connector_sync`` is not registered on it at all — not hidden behind
    a permission, absent from the dispatch table — and a client there cannot start a sync to
    then disconnect from. ``tests/mcp/test_transports.py`` holds that exclusion; asserting it
    again here would be testing that test. The surfaces table marks ``connector_sync`` as having
    an MCP tool because it does over stdio, where a person is present.

    The call is abandoned as rudely as a client can manage — request task canceled and the
    transport closed under it — while preparation is held.
    """
    held = HeldChunker()
    documents = corpus(3, prefix="held")
    service, ingestion, store = served(
        {"proxied": fakes.ObservedConnector(documents, name="handbook")},
        chunker=held,
        fetch_concurrency=2,
        parse_workers=1,
    )

    finished = asyncio.Event()
    running = ingestion.sync

    async def watched(*args: Any, **kwargs: Any) -> Any:
        try:
            return await running(*args, **kwargs)
        finally:
            finished.set()

    ingestion.sync = watched

    async with serving(_backend(service), web=False) as live:
        # An exit stack rather than `async with`, because the client has to be closed *while a
        # call is in flight* — the reason `tests/api/test_both_surfaces.py` gives.
        attached = AsyncExitStack()
        http = await attached.enter_async_context(live.http())
        call = asyncio.create_task(http.post("/api/v1/admin/connectors/proxied/sync", json={}))

        entered = await asyncio.wait_for(
            asyncio.to_thread(held.entered.wait, HELD_STAGE_TIMEOUT_S / 3),
            timeout=HELD_STAGE_TIMEOUT_S,
        )
        assert entered, "preparation was never reached, so the disconnect had nothing to race"

        call.cancel()
        await asyncio.gather(call, return_exceptions=True)
        await asyncio.gather(attached.aclose(), return_exceptions=True)
        assert not finished.is_set(), "the sync finished before the client was even gone"

        held.release.set()
        await asyncio.wait_for(finished.wait(), timeout=HELD_STAGE_TIMEOUT_S)

    await asyncio.wait_for(
        asyncio.to_thread(held.exited.wait, HELD_STAGE_TIMEOUT_S * 2),
        timeout=HELD_STAGE_TIMEOUT_S * 3,
    )
    assert not held.timed_out
    assert len(store.documents) == len(documents), (
        "the HTTP client going away abandoned the sync it started: the server owns accepted "
        "work, and a half-derived corpus is what the recovery sweep exists to avoid"
    )
