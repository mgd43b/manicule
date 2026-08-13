"""The corpus-wide re-parse: what it selects, what it costs, and what it refuses to do.

``select(parse_fingerprints=...)`` and ``re_parse`` composed into one sweep. Everything here
drives the **real** pipeline over an in-memory store, so a claim about what the sweep does is a
measurement of what it did rather than a reading of a signature — which matters most for the
central claim, that a repair after a parser upgrade never asks the source for anything.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, override

import pytest

from manicule.core.anchors import LineAnchor
from manicule.core.content import BlockKind, DocumentStatus, ParsedBlock, RawDocument
from manicule.core.ids import content_hash
from manicule.core.provenance import PROVENANCE_KEY, Provenance, SourceMetadata
from manicule.ingest.reindex import plan_stale, re_parse_stale, select
from tests.fakes import MEDIA_TYPE
from tests.ingest import fakes
from tests.ingest.test_pipeline import build, parse_versions

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Collection, Sequence

    from pydantic import JsonValue

    from manicule.core.content import Document
    from manicule.core.embedding import Vector
    from manicule.core.fingerprints import ParseFingerprint
    from manicule.core.sources import DocRef
    from manicule.ingest.pipeline import IngestPipeline

MARKER = "~"
"""The character version two of the parser stops emitting.

A document containing it comes out of the upgrade with different text; a document without it
comes out byte-identical. That is the shape of a real rules bump — every document the parser
produced is stale, and only some of them move — and it is what makes ``changed`` and
``unchanged`` two different numbers rather than one number and a zero.
"""


class TrimmingLineParser(fakes.LineParser):
    """Version two: the marker is structure, not text, so it stops reaching a block."""

    @override
    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        for number, line in enumerate(raw.as_text().splitlines(), start=1):
            if line.strip():
                yield ParsedBlock(
                    kind=BlockKind.PROSE,
                    text=line.replace(MARKER, ""),
                    anchor=LineAnchor(start=number, end=number),
                )


class BoundedSelects(fakes.MemoryIngestStore):
    """A store that fails rather than lets a sweep spin, and the reason it has to exist.

    The sweep's loop is the one thing here that can go wrong by never ending, and a deadline
    does not catch it: every ``await`` in that loop is over an in-memory fake and completes
    without yielding, so the event loop never gets control and ``asyncio.wait_for`` never fires
    its timer. A test written that way hangs forever instead of failing — which is exactly the
    shape of failure it was supposed to convert into a red build.

    So the bound is on the thing that would grow without limit: how many times the selection is
    asked for. It is checked synchronously, inside the call, so a loop that stops making
    progress fails on its next query rather than on a clock.
    """

    def __init__(self, *, ceiling: int) -> None:
        super().__init__()
        self.ceiling = ceiling
        self.selects = 0

    @override
    async def select_documents(
        self,
        *,
        source: str | None = None,
        statuses: Collection[DocumentStatus] | None = None,
        media_types: Collection[str] | None = None,
        chunk_fp_other_than: str | None = None,
        parse_fp_current: Collection[str] | None = None,
        glossary_fp_other_than: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> Sequence[Document]:
        self.selects += 1
        if self.selects > self.ceiling:
            msg = (
                f"the sweep asked for the selection {self.selects} times, past the ceiling of "
                f"{self.ceiling}. Its cursor has stopped moving past the documents it cannot "
                f"repair, so it is reading the same page for ever."
            )
            raise AssertionError(msg)
        return await super().select_documents(
            source=source,
            statuses=statuses,
            media_types=media_types,
            chunk_fp_other_than=chunk_fp_other_than,
            parse_fp_current=parse_fp_current,
            glossary_fp_other_than=glossary_fp_other_than,
            limit=limit,
            offset=offset,
        )


def fingerprints(library: str) -> tuple[ParseFingerprint, ...]:
    """What ``lines`` would produce with that version of its library installed."""
    produced = parse_versions(lines=library)("lines")
    assert produced is not None, "the fake lookup must know this parser, or nothing is stale"
    return (produced,)


async def corpus(
    pages: dict[str, str],
    *,
    embedder: fakes.CountingEmbedder | None = None,
    blobs: fakes.MemoryBlobs | None = None,
    store: fakes.MemoryIngestStore | None = None,
) -> tuple[
    fakes.MemoryIngestStore,
    fakes.MemoryVectors,
    fakes.MemoryBlobs,
    fakes.DictConnector,
    fakes.CountingEmbedder,
]:
    """An indexed corpus, built by version one of the parser and its connector kept alive.

    The connector is returned rather than dropped on purpose. Every claim below about the
    network is a comparison of ``fetches`` taken across the sweep, with the connector present
    and reachable throughout — an absent connector would prove only that it was absent.
    """
    store = store or fakes.MemoryIngestStore()
    vectors = fakes.MemoryVectors()
    blobs = blobs or fakes.MemoryBlobs()
    embedder = embedder or fakes.CountingEmbedder()
    # What ``app.runtime`` does before it builds a pipeline, done here for the same reason: a
    # store that has not been told which vector space it holds cannot say whether a stored
    # vector is still a chunk's, so a sweep over an unprepared store would measure no reuse and
    # prove nothing about the sweep.
    await vectors.ensure_ready(embedder.fingerprint)
    pipeline, _, _ = build(
        store=store,
        vectors=vectors,
        blobs=blobs,
        embedder=embedder,
        parse_fingerprints=parse_versions(lines="1"),
    )
    connector = fakes.DictConnector(pages)
    await pipeline.run(connector)
    return store, vectors, blobs, connector, embedder


def upgraded(
    store: fakes.MemoryIngestStore,
    vectors: fakes.MemoryVectors,
    blobs: fakes.MemoryBlobs,
    embedder: fakes.CountingEmbedder,
) -> IngestPipeline:
    """The same corpus, read by version two of the parser."""
    pipeline, _, _ = build(
        store=store,
        vectors=vectors,
        blobs=blobs,
        embedder=embedder,
        parsers={"lines": TrimmingLineParser()},
        parse_fingerprints=parse_versions(lines="2"),
    )
    return pipeline


# --- the claim the command exists to make -----------------------------------------------------


async def test_a_parser_bump_selects_every_document_and_repairs_them_without_the_connector() -> (
    None
):
    """The sweep, measured rather than asserted from signatures.

    Three documents indexed under version one, the library bumped to version two, and the
    connector **still there** and never asked for anything. ``fetches`` is captured after the
    sync and compared after the sweep, so "no network" is a count that did not move rather
    than an inference from the fact that ``re_parse_stale`` takes no connector.
    """
    store, vectors, blobs, connector, embedder = await corpus(
        {"a": "alpha", "b": f"be{MARKER}ta", "c": "gamma"}
    )
    downloads = list(connector.fetches)
    assert downloads, "the fixture must have fetched something, or the comparison is vacuous"
    current = fingerprints("2")
    assert len(await select(store, parse_fingerprints=current)) == 3, (
        "a bump makes every document that parser produced stale, whether or not its text moves"
    )

    sweep = await re_parse_stale(
        store=store,
        pipeline=upgraded(store, vectors, blobs, embedder),
        blobs=blobs,
        parse_fingerprints=current,
        batch=2,
    )

    assert connector.fetches == downloads, "the source was not asked for the bytes a second time"
    assert (sweep.selected, sweep.reparsed, sweep.failed, sweep.unrepairable) == (3, 3, 0, 0)
    assert (sweep.changed, sweep.unchanged) == (1, 2), (
        "only the document holding the marker comes out different; the bump is narrow and the "
        "report has to say so, or an operator cannot tell a broad change from a narrow one"
    )
    assert await select(store, parse_fingerprints=current) == [], (
        "every repaired document records the fingerprint the installed parser produces"
    )
    rebuilt = await store.find_document("memory", "b")
    assert rebuilt is not None
    assert [chunk.text for chunk in store.chunks[rebuilt.id]] == ["beta"], (
        "and the stored text is what the current parser produces, not what the stale "
        "fingerprint claimed was current"
    )


async def test_an_unchanged_chunk_keeps_its_id_and_the_vector_row_stored_against_it() -> None:
    """Compared as rows, in both directions, because one direction passes vacuously.

    A test that only looked at the untouched document would also pass if the sweep had done
    nothing at all. So the moved document is asserted in the same breath: its old chunk id is
    gone from the new set, and the row that answers to it is the thing a citation would have
    resolved through.
    """
    store, vectors, blobs, _, embedder = await corpus({"a": "alpha", "b": f"be{MARKER}ta"})
    steady = await store.find_document("memory", "a")
    moved = await store.find_document("memory", "b")
    assert steady is not None
    assert moved is not None
    steady_ids = {chunk.id for chunk in store.chunks[steady.id]}
    moved_ids = {chunk.id for chunk in store.chunks[moved.id]}
    before = dict(vectors.rows)

    sweep = await re_parse_stale(
        store=store,
        pipeline=upgraded(store, vectors, blobs, embedder),
        blobs=blobs,
        parse_fingerprints=fingerprints("2"),
    )

    assert {chunk.id for chunk in store.chunks[steady.id]} == steady_ids
    assert {identifier: vectors.rows[identifier] for identifier in steady_ids} == {
        identifier: before[identifier] for identifier in steady_ids
    }, "an unchanged chunk's vector row is the row it already was"
    assert {chunk.id for chunk in store.chunks[moved.id]}.isdisjoint(moved_ids), (
        "a chunk whose text moved gets a new id, which is what stops a stale vector answering "
        "for text the document no longer has"
    )
    assert (sweep.chunks_kept, sweep.chunks_new) == (1, 1)


async def test_a_second_run_selects_nothing_and_asks_the_embedder_for_nothing() -> None:
    """Idempotence at the level that costs money, not at the level of a count.

    The selection being empty is the cheap half. The half worth asserting is that no batch
    reached the model: a sweep that re-selected nothing and still walked the corpus would show
    up here and nowhere else.
    """
    store, vectors, blobs, _, embedder = await corpus({"a": "alpha", "b": f"be{MARKER}ta"})
    current = fingerprints("2")
    pipeline = upgraded(store, vectors, blobs, embedder)

    first = await re_parse_stale(
        store=store, pipeline=pipeline, blobs=blobs, parse_fingerprints=current
    )
    embedder.batches.clear()
    second = await re_parse_stale(
        store=store, pipeline=pipeline, blobs=blobs, parse_fingerprints=current
    )

    assert first.reparsed == 2
    assert (second.selected, second.reparsed) == (0, 0)
    assert embedder.batches == [], "the second run did no embedding work of any kind"


# --- what it cannot repair, and what it refuses to break ---------------------------------------


async def test_a_document_with_no_retained_bytes_is_named_while_the_others_complete() -> None:
    """One document that cannot be repaired is not a reason to stop repairing the corpus.

    The reason and the remedy are both in the line, because "unrepairable" on its own is a
    status nobody can act on — and the remedy is the one operation on the ladder that reaches
    the network, so it has to be a decision rather than something the sweep does by itself.
    """
    store, vectors, blobs, _, embedder = await corpus({"a": "alpha", "b": "beta", "c": "gamma"})
    stranded = await store.find_document("memory", "b")
    assert stranded is not None
    assert stranded.original_ref is not None
    del blobs.data[stranded.original_ref]

    sweep = await re_parse_stale(
        store=store,
        pipeline=upgraded(store, vectors, blobs, embedder),
        blobs=blobs,
        parse_fingerprints=fingerprints("2"),
        batch=2,
    )

    assert (sweep.selected, sweep.reparsed, sweep.unrepairable) == (3, 2, 1)
    assert len(sweep.unrepairable_documents) == 1
    assert stranded.id in sweep.unrepairable_documents[0]
    assert "re-sync" in sweep.unrepairable_documents[0], "the line has to name the remedy"


async def test_the_sweep_ends_on_a_corpus_where_nothing_can_be_repaired() -> None:
    """The loop's termination argument, run against the case that would hang it.

    Every document is unrepairable and there are more of them than fit in a page, so a cursor
    that counted pages, or one that restarted at zero each time, would read the same first page
    for ever. The failure is a query count rather than a clock, for the reason
    :class:`BoundedSelects` gives.
    """
    bounded = BoundedSelects(ceiling=8)
    store, vectors, blobs, _, embedder = await corpus(
        {name: f"text for {name}" for name in "abcde"}, store=bounded
    )
    blobs.data.clear()

    sweep = await re_parse_stale(
        store=store,
        pipeline=upgraded(store, vectors, blobs, embedder),
        blobs=blobs,
        parse_fingerprints=fingerprints("2"),
        batch=2,
    )

    assert bounded.selects == 4, "three pages of two, and the empty one that ends it"
    assert (sweep.selected, sweep.unrepairable, sweep.reparsed) == (5, 5, 0)
    assert len(sweep.unrepairable_documents) == 5, "each named once, not once per page"


async def test_a_document_no_installed_parser_versions_is_swept_once_and_stays_selectable() -> None:
    """The one document class for which the sweep is not idempotent, pinned rather than found.

    ``parse_fingerprint`` returns ``None`` for a parser manicule does not ship, and
    ``select`` treats a null lineage as eligible on purpose — no recorded fingerprint is no
    evidence the stored text is current, and the alternative is a plugin corpus that no repair
    can reach. The cost is that such a document is selected by every sweep. It must still be
    re-parsed exactly **once** per run: the same property that makes the loop terminate is what
    stops one unversioned document being rebuilt on every page for the length of the corpus.
    """
    store = BoundedSelects(ceiling=6)
    vectors = fakes.MemoryVectors()
    blobs = fakes.MemoryBlobs()
    embedder = fakes.CountingEmbedder()
    unversioned = build(
        store=store,
        vectors=vectors,
        blobs=blobs,
        embedder=embedder,
        parse_fingerprints=lambda parser: None,
    )[0]
    await unversioned.run(fakes.DictConnector({"a": "alpha", "b": "beta"}))
    assert all(document.parse_fp is None for document in store.documents.values())
    store.selects = 0

    sweep = await re_parse_stale(
        store=store,
        pipeline=unversioned,
        blobs=blobs,
        parse_fingerprints=fingerprints("2"),
        batch=1,
    )

    assert (sweep.selected, sweep.reparsed) == (2, 2), "each swept once, however small the page"
    assert len(await select(store, parse_fingerprints=fingerprints("2"))) == 2, (
        "and still selectable afterwards, because there is still no version to compare"
    )


# --- the dry run ------------------------------------------------------------------------------


async def test_a_dry_run_reports_the_same_selection_and_writes_nothing_anywhere() -> None:
    """Every store the sweep can reach, snapshotted and compared.

    The database, the chunks, the vectors, the blobs and the lineage. A plan that advanced a
    fingerprint would be the worst of the five: it would look right, and the documents it
    claimed it *would* repair would never be selected again.

    A plan takes no pipeline and no blob store, so it cannot write to two of those five even by
    mistake. They are still compared, because "cannot" is a reading of a signature and the
    selection itself runs against a live store — and because the whole point of the assertion
    is that it keeps holding when somebody changes what a plan is allowed to look at.
    """
    store, vectors, blobs, _, embedder = await corpus({"a": "alpha", "b": f"be{MARKER}ta"})
    current = fingerprints("2")
    documents = {key: value.model_copy(deep=True) for key, value in store.documents.items()}
    chunks = {key: list(value) for key, value in store.chunks.items()}
    rows = dict(vectors.rows)
    retained = dict(blobs.data)
    lineage = dict(store.lineage)
    parse_lineage = dict(store.parse_lineage)
    embedder.batches.clear()

    planned = await plan_stale(store=store, parse_fingerprints=current, batch=1)

    assert planned.dry_run is True
    assert (planned.reparsed, planned.changed, planned.unchanged) == (0, 0, 0), (
        "a dry run reports what it would touch, never what it did"
    )
    assert store.documents == documents
    assert store.chunks == chunks
    assert vectors.rows == rows
    assert blobs.data == retained
    assert store.lineage == lineage
    assert store.parse_lineage == parse_lineage, "a dry run that advanced a fingerprint is a lie"
    assert embedder.batches == [], "and it ran no model"

    # Compared afterwards, so the numbers come from a run that still had the whole corpus to do.
    performed = await re_parse_stale(
        store=store,
        pipeline=upgraded(store, vectors, blobs, embedder),
        blobs=blobs,
        parse_fingerprints=current,
    )

    assert planned.selected == performed.selected == 2, "the plan is the run's own selection"
    assert performed.reparsed == 2


async def test_a_dry_run_names_the_documents_that_have_no_bytes_to_re_parse() -> None:
    """The half of the plan an operator has to act on before starting the real run.

    The document is stranded the way documents actually are stranded — the retention cap
    refused its bytes at ingest — rather than by editing a column afterwards. A dry run reaches
    this conclusion from ``original_ref`` alone, which is why it can reach it without reading a
    single blob.
    """
    store, _, _, _, embedder = await corpus(
        {"a": "alpha", "b": "beta beta beta beta"}, blobs=fakes.MemoryBlobs(max_bytes=8)
    )
    stranded = await store.find_document("memory", "b")
    assert stranded is not None
    assert stranded.original_ref is None, (
        "the cap must have refused this one, or nothing is under test"
    )
    batches = list(embedder.batches)

    planned = await plan_stale(store=store, parse_fingerprints=fingerprints("2"))

    assert (planned.selected, planned.unrepairable) == (2, 1)
    assert stranded.id in planned.unrepairable_documents[0]
    assert "re-sync" in planned.unrepairable_documents[0], "the plan names the remedy too"
    assert embedder.batches == batches, "a plan runs no model"


# --- stopping and starting again ---------------------------------------------------------------


class InterruptingBlobs(fakes.MemoryBlobs):
    """Retained bytes that stop being readable partway through, the way ``Ctrl-C`` does.

    The blob read is the first thing a document's repair does, so raising here interrupts the
    sweep exactly between two documents — which is the boundary the whole resumability claim
    rests on. Raising inside the pipeline instead would test the pipeline's crash windows,
    which ``repair`` already owns.
    """

    def __init__(self, *, after: int) -> None:
        super().__init__()
        self.after = after
        self.reads = 0

    @override
    async def get(self, digest: str) -> bytes | None:
        if self.reads >= self.after:
            raise asyncio.CancelledError
        self.reads += 1
        return await super().get(digest)


async def test_an_interrupted_sweep_leaves_what_it_finished_consistent_and_resumes() -> None:
    """Stop it in the middle, look at everything, then run it again.

    A document is the transaction boundary, so there is nothing to roll back and no resume
    token to keep: what was committed is complete, what was not is still selected, and running
    the command again is the resume. The second run is asserted to pick up **exactly** the
    remainder, because a resume that started over would also leave a consistent corpus.
    """
    blobs = InterruptingBlobs(after=2)
    store, vectors, _, _, embedder = await corpus(
        {"a": "alpha", "b": "beta", "c": "gamma", "d": "delta"}, blobs=blobs
    )
    current = fingerprints("2")
    pipeline = upgraded(store, vectors, blobs, embedder)
    blobs.reads = 0

    with pytest.raises(asyncio.CancelledError):
        await re_parse_stale(
            store=store, pipeline=pipeline, blobs=blobs, parse_fingerprints=current, batch=4
        )

    repaired = [
        identifier
        for identifier, recorded in store.parse_lineage.items()
        if recorded == current[0].canonical()
    ]
    assert len(repaired) == 2, "the two the sweep reached before it was stopped"
    for identifier in repaired:
        document = await store.get_document(identifier)
        assert document is not None
        assert document.status is DocumentStatus.INDEXED
        assert store.chunks[identifier], "a committed document has its chunks"
        assert all(chunk.id in vectors.rows for chunk in store.chunks[identifier]), (
            "and a vector for every one of them"
        )

    blobs.after = 99
    resumed = await re_parse_stale(
        store=store, pipeline=pipeline, blobs=blobs, parse_fingerprints=current, batch=4
    )

    assert (resumed.selected, resumed.reparsed) == (2, 2), (
        "the resume picks up the remainder rather than starting the corpus again"
    )
    assert await select(store, parse_fingerprints=current) == []


# --- concurrency, and the contract it obeys ----------------------------------------------------


class ObservingEmbedder(fakes.CountingEmbedder):
    """Records the largest number of callers inside ``embed`` at once."""

    def __init__(self) -> None:
        super().__init__()
        self.inside = 0
        self.concurrent = 0

    @override
    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        self.inside += 1
        self.concurrent = max(self.concurrent, self.inside)
        try:
            # A real forward pass is not instantaneous, and a test whose two tasks never
            # actually overlap in time would report serialisation it never observed.
            await asyncio.sleep(0)
            return await super().embed(texts)
        finally:
            self.inside -= 1


async def test_a_sweep_and_a_sync_running_together_never_reach_the_model_at_once() -> None:
    """The existing concurrency contract, which the sweep inherits by using the same pipeline.

    ``IngestPipeline`` holds one ``asyncio.Lock`` around the embed stage because the embedder
    is in-process: one model, one accelerator, and two concurrent batches produce contention
    rather than throughput. The sweep gets that for free by running through the pipeline the
    runtime already built rather than assembling one of its own — which is the reason
    ``Ingesting.reparse_stale`` is a port method and not something a surface composes.

    **This is a test about the model and about nothing else, and it is narrow on purpose.** The
    two documents here have different ids, and all it observes is that two batches were never
    inside the embedder at once. It passed throughout the whole life of a lost-update bug on the
    *same* document, which is the shape of test that reports safety it never checked — so it is
    left exactly as narrow as its name and the tests below it check the other thing. See
    ``docs/ingest.md`` §8.4 for why serialising the accelerator says nothing about the database.
    """
    embedder = ObservingEmbedder()
    store, vectors, blobs, _, _ = await corpus(
        {"a": "alpha", "b": "beta", "c": "gamma"}, embedder=embedder
    )
    pipeline = upgraded(store, vectors, blobs, embedder)
    later = fakes.DictConnector({"d": "delta", "e": "epsilon"}, name="memory")

    await asyncio.gather(
        re_parse_stale(
            store=store,
            pipeline=pipeline,
            blobs=blobs,
            parse_fingerprints=fingerprints("2"),
            batch=1,
        ),
        pipeline.run(later),
    )

    assert embedder.concurrent == 1, (
        "two batches were inside the embedder at once; the sweep is not going through the "
        "pipeline's embedding lock"
    )
    assert embedder.batches, "the fixture must have embedded something, or nothing was observed"


# --- the same document, from both directions at once -------------------------------------------

OLD = "SDR — Stale Document Record"
"""What the corpus holds before a sync moves it, in a shape the glossary detector recognises.

An em-dash definition whose initials spell the term, so one line of one document produces a
document row, a chunk, a vector, a source record *and* a glossary entry. Every one of those is
derived from the same bytes and written by a different call, which is what makes a single
document enough to check that a superseded re-parse reverted none of them.
"""

NEW = "SDR — Synced Document Record"
"""And what the connector has instead. One word apart, so the expansion is the only difference."""

MARKED = f"SDR — Sta{MARKER}le Document Record"
"""The same document, holding the character version two of the parser stops emitting.

**For the tests that have to tell "the sweep wrote nothing" from "the sweep wrote something
identical".** A re-parse of :data:`OLD` under version two produces :data:`OLD` again, so a
document built from it comes out of a completed re-parse and an abandoned one looking exactly
the same — and an assertion that its chunks did not move would hold whether or not the guard
under test did anything. Built from this instead, version one stores it verbatim and a version
two re-parse produces :data:`OLD`, so the two outcomes are two different strings.
"""


class GatedBlobs(fakes.MemoryBlobs):
    """Retained bytes that stop the sweep at the instant it has the old snapshot in hand.

    **A gate rather than a delay, and it is on the blob read rather than in the parser.** The
    blob read is the moment the sweep has committed to a snapshot: the document row it selected
    is in a local variable and the bytes that row points at are in another, and everything it
    does from here is derived from those two. Pausing here and letting a whole sync commit is
    therefore the exact interleaving in question, reproduced without a single ``sleep`` and
    without depending on which task the event loop happens to run first.

    A sync reads no blobs — it fetches from its connector and *writes* through ``retain`` — so
    a gate here catches the sweep and nothing else.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gate: str | None = None
        self.reached = asyncio.Event()
        self.resume = asyncio.Event()

    @override
    async def get(self, digest: str) -> bytes | None:
        data = await super().get(digest)
        if digest == self.gate:
            self.reached.set()
            await self.resume.wait()
        return data


async def superseded_corpus() -> tuple[
    fakes.MemoryGlossaryStore,
    fakes.MemoryVectors,
    GatedBlobs,
    fakes.CountingEmbedder,
    Document,
]:
    """One indexed document holding :data:`OLD`, with its blob read gated and its row in hand."""
    blobs = GatedBlobs()
    store = fakes.MemoryGlossaryStore()
    _, vectors, _, _, embedder = await corpus({"a": OLD}, blobs=blobs, store=store)
    stale = await store.find_document("memory", "a")
    assert stale is not None, "the fixture must have indexed the document under test"
    assert stale.original_ref is not None, "and retained its bytes, or there is nothing to gate"
    assert store.glossary[stale.id], "and detected a definition, or one surface is not covered"
    blobs.gate = stale.original_ref
    return store, vectors, blobs, embedder, stale


async def describes_the_sync(
    store: fakes.MemoryGlossaryStore, vectors: fakes.MemoryVectors, document: Document
) -> None:
    """Assert every stored trace of one document is the sync's, not the re-parse's.

    Written once and called from both orderings, because the claim is the same claim: whichever
    of the two operations was still running, what is stored describes the newest source
    revision. Seven surfaces, and the reason for all seven rather than the obvious one is that
    the bug reverted all seven — a test that looked only at the chunks would have gone green on
    a fix that left the content hash saying something else.
    """
    stored = store.documents[document.id]
    assert stored.content_hash == content_hash(NEW), "the document's own hash is the new bytes'"
    assert stored.version_token == content_hash(NEW), "and its version token the new token"
    assert stored.original_ref == content_hash(NEW), "and its retained reference the new blob"
    assert stored.status is DocumentStatus.INDEXED
    assert stored.title == SYNCED_TITLE, (
        "the source record the sync brought reaches the row it is stored on; a re-parse that "
        "wrote its snapshot's metadata back would undo a correction over unchanged bytes"
    )
    chunks = store.chunks[document.id]
    assert [chunk.text for chunk in chunks] == [NEW]
    assert [entry.expansion for entry in store.glossary[document.id]] == [
        "Synced Document Record"
    ], "the glossary is replaced per document, so a reverted document reverts its definitions"
    assert all(chunk.id in vectors.rows for chunk in chunks), "every current chunk has a row"
    embedded = await fakes.CountingEmbedder().embed([chunk.embed_text for chunk in chunks])
    assert [list(vectors.rows[chunk.id].vector) for chunk in chunks] == embedded, (
        "and the row against it is the embedding of the text it has now. A chunk id is derived "
        "from its text, so a reverted document would have reverted its chunk ids too and this "
        "would be reading rows written for the losing parse"
    )


SYNCED_TITLE = "Synced Document Record"
"""The title the sync's source record declares, so the record is a *changed* record.

Without it the two revisions would differ only in their bytes, and the assertion that a
re-parse cannot write a stale source record back would pass without ever being tested.
"""


def source_record() -> dict[str, JsonValue]:
    """The corrected manifest a sync brings back, in the shape the pipeline reads it from."""
    return {
        PROVENANCE_KEY: Provenance(
            source=SourceMetadata(title=SYNCED_TITLE, version="2")
        ).model_dump(mode="json")
    }


def syncing(pages: dict[str, str]) -> fakes.DictConnector:
    """A connector over the same source name, carrying an authoritative record per page."""
    connector = fakes.DictConnector(pages, name="memory")
    for source_id in pages:
        connector.metadata[source_id] = source_record()
    return connector


async def test_a_sweep_holding_old_bytes_does_not_overwrite_the_sync_that_finished_first() -> None:
    """The lost update, reproduced deterministically and then refused.

    The sweep selects the document, reads its retained bytes, and stops there. A connector sync
    for the same page then runs from end to end and commits newer bytes — cleanly, reporting
    one indexed document. Only then does the sweep continue, into a parse, a chunking, an
    embedding and a commit, every one of them derived from a snapshot that is now two
    generations behind.

    **Before the commit-time compare-and-swap this test failed on its first assertion**, with
    the stored content hash still the old one, and the chunks, glossary, source record and
    retained reference all reverted with it. The embedding lock the sweep shares with the sync
    does not help: it serialises the *model*, and both operations went through it politely, one
    after the other, on their way to writing over each other.
    """
    store, vectors, blobs, embedder, stale = await superseded_corpus()
    pipeline = upgraded(store, vectors, blobs, embedder)

    sweeping = asyncio.create_task(
        re_parse_stale(
            store=store, pipeline=pipeline, blobs=blobs, parse_fingerprints=fingerprints("2")
        )
    )
    await blobs.reached.wait()
    synced = await pipeline.run(syncing({"a": NEW}))
    assert (synced.clean, synced.indexed) == (True, 1), (
        "the sync has to have completed before the sweep resumes, or the ordering under test "
        "is not the one that happened"
    )
    blobs.resume.set()
    sweep = await sweeping

    await describes_the_sync(store, vectors, stale)
    assert (sweep.superseded, sweep.reparsed) == (1, 0), (
        "the document was not re-parsed and must not be counted as though it were"
    )
    assert (sweep.failed, sweep.unrepairable) == (0, 0), "and it is not a failure either"
    assert len(sweep.superseded_documents) == 1
    assert stale.id in sweep.superseded_documents[0], "the line names the document"
    assert not any(
        text in sweep.superseded_documents[0] for text in (OLD, NEW, "Document Record")
    ), (
        "the third reporting path composes its line out of identifiers like the other two. "
        "Both revisions are checked, because the losing one is in hand at the moment the line "
        "is written and is the one that would be reached for"
    )


async def test_a_sync_that_starts_first_still_owns_the_document_the_sweep_was_holding() -> None:
    """The inverse ordering, and the same outcome, because the outcome is about revisions.

    The sync fetches first and is held inside its own fetch; the sweep then runs to completion
    over the bytes that are still stored, and *succeeds* — nothing has moved yet, so there is
    nothing to refuse. The sync then commits on top. The final corpus is the newest source
    revision either way, which is the property, and it is worth checking in this direction
    because a fix that simply made the sweep always lose would pass the other test and leave
    this one describing a re-parse that never happened.
    """
    store, vectors, blobs, embedder, stale = await superseded_corpus()
    blobs.gate = None
    pipeline = upgraded(store, vectors, blobs, embedder)
    connector = syncing({"a": NEW})
    fetching = asyncio.Event()
    released = asyncio.Event()
    fetch = connector.fetch

    async def held(ref: DocRef) -> RawDocument:
        fetching.set()
        await released.wait()
        return await fetch(ref)

    connector.fetch = held
    syncing_task = asyncio.create_task(pipeline.run(connector))
    await fetching.wait()

    sweep = await re_parse_stale(
        store=store, pipeline=pipeline, blobs=blobs, parse_fingerprints=fingerprints("2")
    )
    assert (sweep.reparsed, sweep.superseded) == (1, 0), (
        "nothing had moved when the sweep committed, so it is an ordinary repair"
    )
    released.set()
    synced = await syncing_task

    assert (synced.clean, synced.indexed) == (True, 1)
    await describes_the_sync(store, vectors, stale)


async def test_a_correction_to_the_source_record_alone_is_enough_to_supersede_a_re_parse() -> None:
    """The half of the revision that is not a column, isolated so that it decides on its own.

    Every other test here moves the bytes, which moves the content hash, the version token and
    the retained reference together — so all four columns disagree at once and the guard would
    fire on any one of them. This moves **only the source record**: the sync fetches the same
    bytes under the same token, so the hash, the token and the retained reference are all
    identical, and it goes through the *old* parser so the recorded lineage does not move
    either. The corrected manifest is the only difference there is.

    That case is worth its own test because it is the one a column comparison silently misses. A
    mirrored page whose title is fixed over an unchanged body is a real edit — it is what a
    citation shows — and a re-parse that wrote its snapshot's metadata back would undo it while
    reporting a successful repair.

    The corrected fetch goes in through ``ingest_raw`` rather than through a connector run,
    which is the same code path one document of a sync takes once its bytes are in hand. A whole
    run would not reach it: the token has not moved and the old parser finds its own lineage
    current, so level 1 answers "unchanged" and never fetches — which is correct behaviour, and
    is why a real installation meets this case through the connector that *did* notice.
    """
    store, vectors, blobs, embedder, stale = await superseded_corpus()
    assert stale.provenance is None, "the corpus must start with no record, or nothing moves"
    sweeping_pipeline = upgraded(store, vectors, blobs, embedder)
    behind, _, _ = build(
        store=store,
        vectors=vectors,
        blobs=blobs,
        embedder=embedder,
        parse_fingerprints=parse_versions(lines="1"),
    )

    sweeping = asyncio.create_task(
        re_parse_stale(
            store=store,
            pipeline=sweeping_pipeline,
            blobs=blobs,
            parse_fingerprints=fingerprints("2"),
        )
    )
    await blobs.reached.wait()
    outcomes = await behind.ingest_raw(
        RawDocument(
            source_id="a",
            uri="memory://a",
            media_type=MEDIA_TYPE,
            content=OLD,
            metadata=source_record(),
        ),
        source="memory",
        version_token=stale.version_token,
    )
    assert [outcome.status for outcome in outcomes] == [DocumentStatus.INDEXED]
    corrected = store.documents[stale.id]
    assert (corrected.content_hash, corrected.version_token, corrected.original_ref) == (
        stale.content_hash,
        stale.version_token,
        stale.original_ref,
    ), "the sync moved nothing but the record, or the guard is being asked an easier question"
    assert corrected.parse_fp == stale.parse_fp, "and not the lineage either"
    blobs.resume.set()
    sweep = await sweeping

    assert (sweep.superseded, sweep.reparsed) == (1, 0)
    assert store.documents[stale.id].title == SYNCED_TITLE, (
        "the correction survives; a re-parse that committed would have written the empty title "
        "its snapshot was carrying back over it"
    )
    assert store.documents[stale.id].provenance is not None


JOINING = 2.0
"""How long one document waits inside the parser for another to join it, before giving up.

A bound on a failure rather than a delay in a success: when the two parses do overlap both
arrive at once and nothing waits at all. The first to give up breaks the barrier, so the second
gives up immediately and a red run costs this once.
"""


class Rendezvous(TrimmingLineParser):
    """Version two of the parser, which reports whether two documents were inside it together.

    **A barrier rather than a clock.** "Unrelated documents are not serialised" is a statement
    about two things happening at once, and the only honest way to observe it is to make each
    one wait for the other: if the pipeline serialises them, the first waits alone. A test that
    instead measured elapsed time would pass on a fast machine and fail on a loaded one while
    the code did the same thing both times.
    """

    def __init__(self, parties: int) -> None:
        self.barrier = asyncio.Barrier(parties)
        self.together = 0
        self.arrived: list[str] = []

    @override
    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        self.arrived.append(raw.source_id)
        # Suppressed rather than raised: a parser that raises fails its document, and the
        # failure this is watching for would arrive as one line inside a sweep report instead
        # of as the assertion below. Both documents still ingest; only `together` stays at zero.
        with contextlib.suppress(TimeoutError, asyncio.BrokenBarrierError):
            await asyncio.wait_for(self.barrier.wait(), JOINING)
            self.together += 1
        async for block in super().parse(raw):
            yield block


async def test_a_sweep_and_a_sync_over_different_documents_are_not_serialised_by_the_guard() -> (
    None
):
    """The keyed guard, checked for the thing a keyed guard is *for*.

    A per-document lock and a pipeline-wide one are indistinguishable from every correctness
    test in this file: both make the same corpus come out right. What separates them is a page
    nobody is contending for, and the cost of getting it wrong is a sweep that stops every sync
    in the installation for as long as it runs.

    So: one document re-parsed by the sweep, a different one ingested for the first time by a
    concurrent sync, and a parser both of them have to reach before either may leave. Both
    documents also have to come out correct, because a guard that let unrelated work overlap by
    not guarding anything would satisfy the first assertion alone.
    """
    store, vectors, blobs, embedder, stale = await superseded_corpus()
    blobs.gate = None
    parser = Rendezvous(parties=2)
    pipeline, _, _ = build(
        store=store,
        vectors=vectors,
        blobs=blobs,
        embedder=embedder,
        parsers={"lines": parser},
        parse_fingerprints=parse_versions(lines="2"),
    )

    sweep, synced = await asyncio.gather(
        re_parse_stale(
            store=store, pipeline=pipeline, blobs=blobs, parse_fingerprints=fingerprints("2")
        ),
        pipeline.run(syncing({"b": "beta"})),
    )

    assert parser.together == 2, (
        f"only one document was ever inside the parser; the two arrived separately as "
        f"{parser.arrived}. Work on unrelated documents is being serialised, so the mutation "
        f"guard is not keyed by document id"
    )
    assert (sweep.reparsed, sweep.superseded, sweep.failed) == (1, 0, 0)
    assert [chunk.text for chunk in store.chunks[stale.id]] == [OLD], (
        "the swept document holds version two's reading of its own retained bytes — which is "
        "what MARKED and OLD being different strings is for, so this cannot pass on a sweep "
        "that did nothing — and nothing about the other document's sync reached it"
    )
    fresh = await store.find_document("memory", "b")
    assert fresh is not None
    assert fresh.status is DocumentStatus.INDEXED
    assert synced.indexed == 1
    assert all(chunk.id in vectors.rows for chunk in store.chunks[fresh.id])
    # Read through the private on purpose. A lock per document is only affordable if the entry
    # goes away with the document that needed it, and a sweep over a corpus is exactly the
    # caller that would otherwise accumulate one per row it has already finished with.
    assert pipeline._mutations == {}, (  # pyright: ignore[reportPrivateUsage]
        "a lock was left behind for a document nothing is working on any more"
    )


class CancellingEmbedder(fakes.CountingEmbedder):
    """Stops the run where an interrupt most often lands: inside the model.

    Between the record write and the commit, which is the window the spec asks about — the
    document has been parsed and its row written, and none of its chunks, vectors or glossary
    rows have been replaced yet.
    """

    def __init__(self) -> None:
        super().__init__()
        self.armed = False

    @override
    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        if self.armed:
            raise asyncio.CancelledError
        return await super().embed(texts)


async def test_a_sweep_cancelled_before_its_commit_serves_nothing_it_half_wrote() -> None:
    """``Ctrl-C`` in the middle of one document, and then the same command again.

    Everything derived is compared against a snapshot taken before the cancelled run, in all
    four stores, because the failure this guards against is not a crash — it is a corpus that
    looks fine and answers with half of one parse and half of another.
    """
    embedder = CancellingEmbedder()
    blobs = GatedBlobs()
    store = fakes.MemoryGlossaryStore()
    _, vectors, _, _, _ = await corpus({"a": MARKED}, blobs=blobs, store=store, embedder=embedder)
    document = await store.find_document("memory", "a")
    assert document is not None
    chunks = list(store.chunks[document.id])
    assert [chunk.text for chunk in chunks] == [MARKED], (
        "version one stored the marker verbatim, so a completed version-two re-parse would "
        "leave a different string here and this comparison can tell the two apart"
    )
    rows = dict(vectors.rows)
    entries = list(store.glossary[document.id])
    pipeline = upgraded(store, vectors, blobs, embedder)
    embedder.armed = True

    with pytest.raises(asyncio.CancelledError):
        await re_parse_stale(
            store=store, pipeline=pipeline, blobs=blobs, parse_fingerprints=fingerprints("2")
        )

    interrupted = await store.get_document(document.id)
    assert interrupted is not None
    assert interrupted.status is not DocumentStatus.INDEXED, (
        "a document caught mid-repair must not be servable while it is one"
    )
    assert store.chunks[document.id] == chunks, "and nothing derived from the new parse landed"
    assert vectors.rows == rows
    assert store.glossary[document.id] == entries

    embedder.armed = False
    resumed = await re_parse_stale(
        store=store, pipeline=pipeline, blobs=blobs, parse_fingerprints=fingerprints("2")
    )

    assert (resumed.selected, resumed.reparsed, resumed.superseded) == (1, 1, 0), (
        "the interrupted document was still selected, and running the command again is the "
        "whole of the resume"
    )
    finished = await store.get_document(document.id)
    assert finished is not None
    assert finished.status is DocumentStatus.INDEXED
    assert [chunk.text for chunk in store.chunks[document.id]] == [OLD], (
        "and the resumed run is what produced the version-two reading, so the first run really "
        "had written none of it"
    )
    assert await select(store, parse_fingerprints=fingerprints("2")) == []


class OvertakingEmbedder(fakes.CountingEmbedder):
    """A second writer that lands while the model is running, from outside the pipeline.

    **Writes into the store directly, and that is the case being modelled.** The per-document
    mutation lock is an ``asyncio.Lock``: it holds inside one event loop in one process, and a
    second process opened on the same data directory takes no part in it. From in here that is
    indistinguishable from a row changing underneath with no warning — which is what this does,
    at the one moment that matters, after the vectors exist and before anything has been
    replaced.
    """

    def __init__(self, store: fakes.MemoryGlossaryStore, document_id: str) -> None:
        super().__init__()
        self.store = store
        self.document_id = document_id
        self.armed = False

    @override
    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        vectors = await super().embed(texts)
        if self.armed:
            self.armed = False
            stored = self.store.documents[self.document_id]
            self.store.documents[self.document_id] = stored.model_copy(
                update={
                    "content_hash": content_hash(NEW),
                    "version_token": content_hash(NEW),
                    "original_ref": content_hash(NEW),
                    "status": DocumentStatus.INDEXED,
                }
            )
        return vectors


async def test_a_document_overtaken_after_its_vectors_were_built_writes_no_derived_row() -> None:
    """The guard at the head of the commit, which is what makes "no cleanup" true.

    The compare-and-swap that ends the commit is the durable invariant, and on its own it would
    fire too late: chunks and glossary rows are *replaced*, so by the time a final guard could
    refuse, the ones that were there are already gone. This is the other one — after the model
    has run, before the first thing is replaced — and this test is the reason it exists rather
    than a reconciliation pass afterwards, which would be a second race in the same place.

    Only reachable from outside the process, so it is reached that way: the overtaking write
    goes straight into the store rather than through the pipeline, which is what a second
    process holding no instance lock looks like from in here.
    """
    blobs = GatedBlobs()
    store = fakes.MemoryGlossaryStore()
    document_id = ""
    _, vectors, _, _, _ = await corpus(
        {"a": MARKED}, blobs=blobs, store=store, embedder=fakes.CountingEmbedder()
    )
    document = await store.find_document("memory", "a")
    assert document is not None
    document_id = document.id
    embedder = OvertakingEmbedder(store, document_id)
    chunks = list(store.chunks[document_id])
    assert [chunk.text for chunk in chunks] == [MARKED], (
        "the re-parse under way produces a different string from the one stored, so replacing "
        "the chunk set would be visible here. Built from OLD the two would be identical and "
        "this test would pass against an implementation that wrote all of it"
    )
    rows = dict(vectors.rows)
    entries = list(store.glossary[document_id])
    pipeline = upgraded(store, vectors, blobs, embedder)
    embedder.armed = True

    sweep = await re_parse_stale(
        store=store, pipeline=pipeline, blobs=blobs, parse_fingerprints=fingerprints("2")
    )

    assert (sweep.superseded, sweep.reparsed, sweep.failed) == (1, 0, 0)
    assert embedder.batches, "the model must have run, or the miss was not forced after it"
    assert store.chunks[document_id] == chunks, (
        "the chunks the overtaken re-parse produced were never written, so the ones that were "
        "there are still there"
    )
    assert vectors.rows == rows, "and no vector row was upserted for a chunk nothing holds"
    assert store.glossary[document_id] == entries
    assert store.documents[document_id].content_hash == content_hash(NEW), (
        "the row belongs to whoever overtook it; this sweep left it alone"
    )


async def test_a_sweep_run_again_after_a_supersession_converges_with_no_manual_cleanup() -> None:
    """The document the sweep declined to touch, on the next run of the same command.

    The sync here goes through the **old** parser, so the document it commits is stale the
    instant it lands: it is still the sweep's to repair, and the second run has to find it and
    finish it. That is the harder half of "remains eligible for the appropriate later repair" —
    the easy half is a sync that happened to leave the document current, which needs no repair
    and proves nothing about eligibility.
    """
    store, vectors, blobs, embedder, stale = await superseded_corpus()
    sweeping_pipeline = upgraded(store, vectors, blobs, embedder)
    behind, _, _ = build(
        store=store,
        vectors=vectors,
        blobs=blobs,
        embedder=embedder,
        parse_fingerprints=parse_versions(lines="1"),
    )

    sweeping = asyncio.create_task(
        re_parse_stale(
            store=store,
            pipeline=sweeping_pipeline,
            blobs=blobs,
            parse_fingerprints=fingerprints("2"),
        )
    )
    await blobs.reached.wait()
    assert (await behind.run(syncing({"a": NEW}))).indexed == 1
    blobs.resume.set()
    first = await sweeping

    assert (first.superseded, first.reparsed) == (1, 0)
    assert len(await select(store, parse_fingerprints=fingerprints("2"))) == 1, (
        "the sync left the document stale, so it is still selected and nothing has been lost"
    )

    blobs.gate = None
    second = await re_parse_stale(
        store=store, pipeline=sweeping_pipeline, blobs=blobs, parse_fingerprints=fingerprints("2")
    )

    assert (second.selected, second.reparsed, second.superseded) == (1, 1, 0)
    await describes_the_sync(store, vectors, stale)
    assert await select(store, parse_fingerprints=fingerprints("2")) == [], (
        "converged, with nothing done to it by hand in between"
    )


async def test_a_supersession_does_not_step_the_cursor_past_a_document_nobody_looked_at() -> None:
    """The paging arithmetic, for the outcome that was not in it when it was written.

    The cursor is the count of documents a pass **left behind in the selection**, which is what
    makes it right in both directions while the set shrinks underneath it. A superseded document
    is the awkward case: the sweep did nothing to it, so the reflex is to count it as left
    behind — but the sync that overtook it usually left it *current*, so it is gone from the
    selection, and counting it steps the offset one place past the document that moved up into
    its slot. With pages of one and two stale documents, that document is the entire remainder.

    The symptom would be a sweep that reported doing less than it selected and stopped, with the
    skipped page still stale and nothing anywhere naming it.
    """
    blobs = GatedBlobs()
    store = fakes.MemoryGlossaryStore()
    _, vectors, _, _, embedder = await corpus({"a": OLD, "b": "beta"}, blobs=blobs, store=store)
    overtaken = await store.find_document("memory", "a")
    behind = await store.find_document("memory", "b")
    assert overtaken is not None
    assert behind is not None
    blobs.gate = overtaken.original_ref
    pipeline = upgraded(store, vectors, blobs, embedder)

    sweeping = asyncio.create_task(
        re_parse_stale(
            store=store,
            pipeline=pipeline,
            blobs=blobs,
            parse_fingerprints=fingerprints("2"),
            batch=1,
        )
    )
    await blobs.reached.wait()
    assert (await pipeline.run(syncing({"a": NEW}))).indexed == 1
    blobs.resume.set()
    sweep = await sweeping

    assert (sweep.superseded, sweep.selected) == (1, 2), (
        "the second document was still in the selection and the sweep has to have reached it"
    )
    assert sweep.reparsed == 1
    assert await select(store, parse_fingerprints=fingerprints("2")) == [], (
        "and repaired it; a cursor one place too far leaves it stale for ever"
    )


# --- what the report is allowed to say ---------------------------------------------------------


async def test_the_report_names_documents_without_quoting_what_is_in_them() -> None:
    """Requirement met by construction, checked because the whole subject is retained content.

    Every line the sweep produces is an id, a URI and a reason. The retained bytes carry a
    string that appears nowhere else, and the assertion is that it appears nowhere in the
    report — through both reporting paths, the unrepairable one and the failed one.

    **What this does not cover, said plainly.** The sweep passes a failing stage's own ``detail``
    through, so a parser that raised with the text it choked on *in the message* would put that
    text here and this would go red for the right reason — but nothing stops such a parser being
    written, and this test is not that guard. What it holds is the sweep itself to composing its
    report out of identifiers: found by making it append the document's stored chunk text and
    watching this go red, which an earlier version of the break — appending ``metadata``, which
    carries no content — did not.
    """
    marker = "sparrow-hyacinth-marginalia"
    store, vectors, blobs, _, embedder = await corpus({"a": f"alpha {marker}", "b": "beta"})
    stranded = await store.find_document("memory", "b")
    assert stranded is not None
    assert stranded.original_ref is not None
    del blobs.data[stranded.original_ref]
    pipeline, _, _ = build(
        store=store,
        vectors=vectors,
        blobs=blobs,
        embedder=embedder,
        parsers={"lines": fakes.ExplodingParser()},
        parse_fingerprints=parse_versions(lines="2"),
    )

    sweep = await re_parse_stale(
        store=store, pipeline=pipeline, blobs=blobs, parse_fingerprints=fingerprints("2")
    )

    assert (sweep.failed, sweep.unrepairable) == (1, 1), "both reporting paths were exercised"
    printed = " ".join([*sweep.unrepairable_documents, *sweep.failures])
    assert marker not in printed, "the sweep put a document's own content into its report"
    assert any(marker in data.decode() for data in blobs.data.values()), (
        "the fixture must still hold the marker, or this proves nothing"
    )
