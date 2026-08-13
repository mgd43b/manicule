"""The corpus-wide re-parse: what it selects, what it costs, and what it refuses to do.

``select(parse_fingerprints=...)`` and ``re_parse`` composed into one sweep. Everything here
drives the **real** pipeline over an in-memory store, so a claim about what the sweep does is a
measurement of what it did rather than a reading of a signature — which matters most for the
central claim, that a repair after a parser upgrade never asks the source for anything.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, override

import pytest

from manicule.core.anchors import LineAnchor
from manicule.core.content import BlockKind, DocumentStatus, ParsedBlock
from manicule.ingest.reindex import plan_stale, re_parse_stale, select
from tests.ingest import fakes
from tests.ingest.test_pipeline import build, parse_versions

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Collection, Sequence

    from manicule.core.content import Document, RawDocument
    from manicule.core.embedding import Vector
    from manicule.core.fingerprints import ParseFingerprint
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

    Across processes the contract is the exclusive lock on ``<data_dir>/manicule.lock``
    (``docs/ingest.md`` §6.5). It is not this module's to take: it is held for a process
    lifetime by whoever opens the data directory, and a command that took one of its own would
    be a second answer to a question that already has one.
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
