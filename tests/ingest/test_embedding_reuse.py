"""What a re-parse actually costs at the model, counted rather than inferred.

Every assertion here is a count taken from :class:`~tests.ingest.fakes.CountingEmbedder`, which
records one entry per batch it is handed. ``sum(batches)`` is chunks embedded and
``len(batches)`` is forward passes, and neither is derivable from any row identity — which is
the whole point, because the claim this module exists to hold is a claim about work avoided and
the obvious evidence for it is evidence for something else.

Three facts that look alike and are not, kept apart deliberately:

- **A chunk id surviving.** ``chunks.id`` is derived from ``text``, so a chunk a re-parse did
  not move comes back with the id it had and keeps every citation that resolved to it.
- **A vector surviving.** A vector is produced from ``embed_text``, which carries the heading
  breadcrumb. A chunk can keep its id while the string that produced its vector changes.
- **A forward pass avoided.** Only this one costs accelerator time, and only a counting
  embedder can see it.

The in-memory embedding cache is deliberately not in play anywhere here. It is a bounded LRU
over exact duplicate text and it cannot answer any of these questions: a fresh process with a
corpus larger than it evicts every entry before anything could be reused.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import pytest

from manicule.config.settings import EmbeddingSettings
from manicule.core.anchors import LineAnchor
from manicule.core.content import BlockKind, Chunk, Document, ParsedBlock
from manicule.core.embedding import (
    UNRECORDED_IDENTITY,
    VectorState,
    embedding_input_identity,
)
from manicule.core.errors import ContextOverflowError
from manicule.core.ids import chunk_id
from manicule.ingest.embedding import EmbeddingWork, embed_or_reuse
from manicule.ingest.reindex import re_parse_stale, repair, select
from tests.fakes import PassThroughMiddleware
from tests.ingest import fakes
from tests.ingest.test_pipeline import build, parse_versions
from tests.ingest.test_reindex_sweep import MARKER, TrimmingLineParser, fingerprints

CACHE_CAPACITY = EmbeddingSettings().cache_entries
"""The embedding cache's configured size, read rather than restated.

A corpus larger than this is the only shape in which a reuse measurement is about the
persisted identity: at or below it, an LRU over duplicate text could account for the same
number and nothing would distinguish the two explanations.
"""

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable, Sequence

    from manicule.core.content import RawDocument
    from manicule.core.protocols import Chunker, Middleware
    from manicule.ingest.pipeline import IngestPipeline


class RebreadcrumbingChunker(fakes.BlockChunker):
    """A chunker whose breadcrumb moves while the text it packs does not.

    The fixture requirement 7 is about, and it cannot be produced by changing a parser: a chunk
    id is derived from ``text``, so anything that moves the text moves the id and the question
    stops being interesting. Moving the *breadcrumb* leaves every id where it was and changes
    every embedding input, which is exactly the case an optimisation keyed on the chunk id gets
    wrong.
    """

    def __init__(self, breadcrumb: str) -> None:
        self.breadcrumb = breadcrumb

    @override
    def chunk(self, document: Document, blocks: Iterable[ParsedBlock]) -> list[Chunk]:
        return [
            chunk.model_copy(update={"embed_text": f"{self.breadcrumb} > {chunk.text}"})
            for chunk in super().chunk(document, blocks)
        ]


class RefusingChunkMiddleware(PassThroughMiddleware):
    """Raises in ``after_chunk``, which is the last thing before the reuse partition runs.

    A parser that explodes never enters ``_finish``; this does, and fails one line before
    ``embed_or_reuse`` would have been called. The two reach the same outcome by different
    routes, and only this one passes through the code that attaches the embed-stage accounting.
    """

    name = "refusing-chunk"

    @override
    async def after_chunk(self, document: Document, chunks: list[Chunk]) -> list[Chunk]:
        del document, chunks
        msg = "the chunk hook is unavailable"
        raise RuntimeError(msg)


class ReversingLineParser(fakes.LineParser):
    """Version two: the same lines, in the opposite order.

    Every chunk keeps its text and therefore its embedding input; every chunk changes position,
    and a chunk id is derived from position as well as text — so the ids move and the vectors
    must not. Requirement 8, with nothing else moving alongside it.
    """

    @override
    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        lines = [line for line in raw.as_text().splitlines() if line.strip()]
        for number, line in reversed(list(enumerate(lines, start=1))):
            yield ParsedBlock(
                kind=BlockKind.PROSE, text=line, anchor=LineAnchor(start=number, end=number)
            )


async def indexed(
    pages: dict[str, str],
    *,
    chunker: Chunker | None = None,
) -> tuple[fakes.MemoryIngestStore, fakes.MemoryVectors, fakes.MemoryBlobs, fakes.CountingEmbedder]:
    """A corpus indexed by version one of the parser, with its vector store prepared."""
    store = fakes.MemoryIngestStore()
    vectors = fakes.MemoryVectors()
    blobs = fakes.MemoryBlobs()
    embedder = fakes.CountingEmbedder()
    await vectors.ensure_ready(embedder.fingerprint)
    pipeline, _, _ = build(
        store=store,
        vectors=vectors,
        blobs=blobs,
        embedder=embedder,
        chunker=chunker,
        parse_fingerprints=parse_versions(lines="1"),
    )
    await pipeline.run(fakes.DictConnector(pages))
    return store, vectors, blobs, embedder


def rebuilt(
    store: fakes.MemoryIngestStore,
    vectors: fakes.MemoryVectors,
    blobs: fakes.MemoryBlobs,
    embedder: fakes.CountingEmbedder,
    *,
    parser: object | None = None,
    chunker: Chunker | None = None,
    middleware: Sequence[Middleware] = (),
    library: str = "2",
) -> IngestPipeline:
    """The same corpus, read by a version the installed parse lineage no longer matches."""
    pipeline, _, _ = build(
        store=store,
        vectors=vectors,
        blobs=blobs,
        embedder=embedder,
        chunker=chunker,
        middleware=middleware,
        parsers={"lines": parser or TrimmingLineParser()},
        parse_fingerprints=parse_versions(lines=library),
    )
    return pipeline


# --- the three-way partition ------------------------------------------------------------------


async def test_a_narrow_parser_change_embeds_only_the_chunk_whose_input_moved() -> None:
    """The claim the whole change exists to make, in forward passes.

    Four documents, one of which the bump moves. Before this change the sweep embedded every
    chunk of every document it re-parsed — so the number below was four, and every one of the
    three unchanged documents paid a forward pass for producing exactly the bytes it already
    had. The assertion is on batches taken across the sweep, not on rows, because a row can
    look identical while having been recomputed.
    """
    store, vectors, blobs, embedder = await indexed(
        {"a": "alpha", "b": f"be{MARKER}ta", "c": "gamma", "d": "delta"}
    )
    steady: dict[str, Document] = {}
    for source in ("a", "c", "d"):
        document = await store.find_document("memory", source)
        assert document is not None, f"the fixture must have indexed {source}"
        steady[source] = document
    steady_ids = {
        source: {chunk.id for chunk in store.chunks[document.id]}
        for source, document in steady.items()
    }
    before = dict(vectors.rows)
    embedder.batches.clear()

    sweep = await re_parse_stale(
        store=store,
        pipeline=rebuilt(store, vectors, blobs, embedder),
        blobs=blobs,
        parse_fingerprints=fingerprints("2"),
    )

    assert sweep.reparsed == 4, "every document that parser produced is stale and is rebuilt"
    assert sum(embedder.batches) == 1, (
        "only the chunk whose embedding input changed reached the model. Every other chunk of "
        "every other document produced the bytes it already had, and paying a forward pass "
        "for that is the cost this change removes"
    )
    assert sweep.embedding.forward_calls == len(embedder.batches) == 1, (
        "the report's forward-call count is the number of times the model was called, not a "
        "restatement of the batching arithmetic"
    )
    assert (sweep.embedding.reused, sweep.embedding.embedded) == (3, 1)
    assert (sweep.embedding.input_changed, sweep.embedding.repaired) == (1, 0)

    for source, identifiers in steady_ids.items():
        assert {chunk.id for chunk in store.chunks[steady[source].id]} == identifiers, (
            f"document {source} kept every chunk id it had"
        )
        assert {key: vectors.rows[key] for key in identifiers} == {
            key: before[key] for key in identifiers
        }, f"document {source}'s vector rows are the rows they already were, unchanged"


async def test_an_entirely_unchanged_document_costs_no_forward_pass_and_advances_its_lineage() -> (
    None
):
    """A bump whose output is identical everywhere. Zero inputs, zero calls, lineage moved.

    The two halves have to be asserted together. Zero calls on its own is also what a sweep
    that silently did nothing would report, and advancing lineage on its own says nothing about
    what it cost.
    """
    store, vectors, blobs, embedder = await indexed({"a": "alpha\nbeta", "b": "gamma\ndelta"})
    current = fingerprints("2")
    assert len(await select(store, parse_fingerprints=current)) == 2
    embedder.batches.clear()

    # Version two of the *library*, with the same parser behind it: every document is stale and
    # none of them comes out different.
    sweep = await re_parse_stale(
        store=store,
        pipeline=rebuilt(store, vectors, blobs, embedder, parser=fakes.LineParser()),
        blobs=blobs,
        parse_fingerprints=current,
    )

    assert (sweep.reparsed, sweep.unchanged, sweep.changed) == (2, 2, 0)
    assert embedder.batches == [], "no batch reached the model"
    assert (sweep.embedding.embedded, sweep.embedding.forward_calls) == (0, 0)
    assert sweep.embedding.reused == 4, "every chunk of both documents reused its vector"
    assert await select(store, parse_fingerprints=current) == [], (
        "parse lineage advanced, so a second sweep selects nothing — reuse must not be bought "
        "by skipping the repair"
    )


async def test_a_moved_vector_row_is_repaired_and_counted_apart_from_a_changed_input() -> None:
    """Requirement 3's third group, which is the one that gets forgotten.

    A vector row deleted under a chunk whose embedding input never changed. Identity metadata
    is not consulted about whether a vector exists — the row is read — so this is embedded
    again, and it is reported as a repair rather than as a changed input, because an operator
    told "one input changed" about a corpus in which none did would go looking for a parser
    bug that is not there.
    """
    store, vectors, blobs, embedder = await indexed({"a": "alpha\nbeta\ngamma"})
    document = await store.find_document("memory", "a")
    assert document is not None
    victim = store.chunks[document.id][1]
    del vectors.rows[victim.id]
    embedder.batches.clear()

    sweep = await re_parse_stale(
        store=store,
        pipeline=rebuilt(store, vectors, blobs, embedder, parser=fakes.LineParser()),
        blobs=blobs,
        parse_fingerprints=fingerprints("2"),
    )

    assert sum(embedder.batches) == 1, "exactly the chunk whose vector went missing"
    assert (sweep.embedding.reused, sweep.embedding.embedded) == (2, 1)
    assert (sweep.embedding.repaired, sweep.embedding.input_changed) == (1, 0), (
        "a missing vector is a repair, not a changed input"
    )
    assert (sweep.embedding.vectors_new, sweep.embedding.vectors_replaced) == (1, 0)
    assert victim.id in vectors.rows, "and the row is back"


async def test_the_partition_reports_all_three_groups_from_one_sweep() -> None:
    """Reused, re-embedded for a changed input, and repaired — in one pass, told apart.

    A report that could not separate these would be a report an operator cannot act on: the
    remedy for a changed input is nothing, and the remedy for a corrupt vector is finding out
    what damaged the directory.
    """
    store, vectors, blobs, embedder = await indexed(
        {"a": "alpha\nbeta", "b": f"ga{MARKER}mma\ndelta"}
    )
    quiet = await store.find_document("memory", "a")
    assert quiet is not None
    # A row of the wrong dimension: present, claiming a current identity, and unusable.
    damaged = store.chunks[quiet.id][0]
    vectors.rows[damaged.id].vector = (0.5, 0.5)
    embedder.batches.clear()

    sweep = await re_parse_stale(
        store=store,
        pipeline=rebuilt(store, vectors, blobs, embedder),
        blobs=blobs,
        parse_fingerprints=fingerprints("2"),
    )

    assert (sweep.embedding.reused, sweep.embedding.embedded) == (2, 2)
    assert (sweep.embedding.input_changed, sweep.embedding.repaired) == (1, 1)
    assert sum(embedder.batches) == 2
    assert len(vectors.rows[damaged.id].vector) == embedder.fingerprint.dimension, (
        "the row of the wrong dimension was rebuilt rather than left in place"
    )


# --- the distinctions the spec is built around --------------------------------------------------


async def test_a_chunk_whose_breadcrumb_moved_re_embeds_although_its_id_did_not() -> None:
    """Requirement 7, and the reason reuse is not keyed on the chunk id.

    Nothing about any chunk's ``text`` moves here, so every id survives and every citation
    still resolves. Every ``embed_text`` moves, so every stored vector describes a string the
    corpus no longer contains — and a "reuse when the id survives" optimisation would keep all
    of them, silently, for ever.
    """
    store, vectors, blobs, embedder = await indexed(
        {"a": "alpha\nbeta"}, chunker=RebreadcrumbingChunker("Handbook")
    )
    document = await store.find_document("memory", "a")
    assert document is not None
    identifiers = {chunk.id for chunk in store.chunks[document.id]}
    identities = {chunk.id: vectors.rows[chunk.id].identity for chunk in store.chunks[document.id]}
    embedder.batches.clear()

    sweep = await re_parse_stale(
        store=store,
        pipeline=rebuilt(
            store,
            vectors,
            blobs,
            embedder,
            parser=fakes.LineParser(),
            chunker=RebreadcrumbingChunker("Runbook"),
        ),
        blobs=blobs,
        parse_fingerprints=fingerprints("2"),
    )

    assert {chunk.id for chunk in store.chunks[document.id]} == identifiers, (
        "the chunk ids survive: the identity contract is about text, and no text moved"
    )
    assert sum(embedder.batches) == 2, "and both chunks were embedded anyway"
    assert (sweep.embedding.reused, sweep.embedding.input_changed) == (0, 2)
    assert sweep.embedding.vectors_replaced == 2
    for identifier in identifiers:
        assert vectors.rows[identifier].identity != identities[identifier], (
            "the embedding-input identity moved with the breadcrumb, so a later sweep compares "
            "against what is stored now rather than against what it replaced"
        )


async def test_a_chunk_that_only_moved_position_reuses_its_vector() -> None:
    """Requirement 8: position is not part of the embedding input.

    The ids all change, because a chunk id is derived from position as well as text. Not one
    embedding input changes. A reuse rule keyed on the id — the naive optimisation in its other
    direction — would re-embed the whole document for a reordering that moved no text.
    """
    store, vectors, blobs, embedder = await indexed({"a": "alpha\nbeta\ngamma"})
    document = await store.find_document("memory", "a")
    assert document is not None
    identifiers = {chunk.id for chunk in store.chunks[document.id]}
    embedder.batches.clear()

    sweep = await re_parse_stale(
        store=store,
        pipeline=rebuilt(store, vectors, blobs, embedder, parser=ReversingLineParser()),
        blobs=blobs,
        parse_fingerprints=fingerprints("2"),
    )

    after = {chunk.id for chunk in store.chunks[document.id]}
    assert after != identifiers, "the fixture must move the ids, or it is testing nothing"
    assert embedder.batches == [], "and not one of them cost a forward pass"
    assert (sweep.embedding.reused, sweep.embedding.embedded) == (3, 0)
    assert sweep.embedding.vectors_new == 0, (
        "`vectors_new` counts where the *embedder's* output went, and the embedder produced "
        "none. The rows written under the ids the chunks now have are new rows, and they are "
        "counted under `reused` because the vectors in them came out of the store"
    )


async def test_a_changed_embedding_fingerprint_reuses_nothing() -> None:
    """Requirement 9, at the level the identity decides it.

    The fingerprint is one of the three inputs to the identity, so every stored identity stops
    matching the moment the model does — with every chunk's text, position and id untouched.
    That is checked here directly rather than through the run-level refusal, because the
    refusal is a property of the *pipeline* and this is a property of the *identity*: a repair
    verb that did not go through ``check_before_run`` would otherwise reuse a vector from
    another model's space.
    """
    store, vectors, _, embedder = await indexed({"a": "alpha\nbeta\ngamma"})
    document = await store.find_document("memory", "a")
    assert document is not None
    chunks = list(store.chunks[document.id])

    verdicts = await vectors.stored_vectors(chunks)
    assert all(verdict.state is VectorState.READABLE for verdict in verdicts.values())

    other = fakes.CountingEmbedder()
    other.fingerprint = embedder.fingerprint.model_copy(update={"model_id": "other/embedder"})
    swapped = fakes.MemoryVectors()
    swapped.rows = dict(vectors.rows)
    await swapped.ensure_ready(other.fingerprint)

    verdicts = await swapped.stored_vectors(chunks)
    assert all(verdict.state is VectorState.STALE for verdict in verdicts.values()), (
        "not one vector survives a change of model, however unchanged the text that produced it"
    )
    assert not any(verdict.vector for verdict in verdicts.values())


async def test_identity_metadata_that_contradicts_its_own_row_is_repaired_not_trusted() -> None:
    """The other half of requirement 3's third group: metadata that claims a match.

    A row whose recorded identity says it embeds this chunk's current input, beside a chunk
    that says it embeds something else. There is no way to tell which half is right, so the row
    is rebuilt — because the failure of believing it is a stale vector answering for current
    text, and the cost of not believing it is one forward pass.
    """
    store, vectors, blobs, embedder = await indexed({"a": "alpha\nbeta"})
    document = await store.find_document("memory", "a")
    assert document is not None
    chunks = list(store.chunks[document.id])
    row = vectors.rows[chunks[0].id]
    row.embed_text = "something else entirely"

    verdicts = await vectors.stored_vectors(chunks)
    assert verdicts[chunks[0].id].state is VectorState.CORRUPT, (
        "a row that says two different things about what it embedded is not evidence of "
        "anything, and the one thing it must not be is evidence that a vector is current"
    )
    assert verdicts[chunks[1].id].state is VectorState.READABLE, "and the row beside it is fine"

    embedder.batches.clear()
    sweep = await re_parse_stale(
        store=store,
        pipeline=rebuilt(store, vectors, blobs, embedder, parser=fakes.LineParser()),
        blobs=blobs,
        parse_fingerprints=fingerprints("2"),
    )
    assert sum(embedder.batches) == 1
    assert (sweep.embedding.repaired, sweep.embedding.reused) == (1, 1)


# --- migration, idempotence, and the cache -----------------------------------------------------


async def test_a_row_with_no_recorded_identity_is_reconstructed_rather_than_re_embedded() -> None:
    """The migration of an existing ``vectors/`` directory, priced.

    Every row written before the identity column carries nothing, and the conservative reading
    — distrust it — would re-embed a whole corpus to learn what the row already says. The
    embedding input is reconstructible from the chunk the row was written with, in the same
    call, from the same object. So the upgrade costs no forward pass, and the rows it
    reconstructs are counted so the one-time backfill is visible rather than silent.
    """
    store, vectors, blobs, embedder = await indexed({"a": "alpha\nbeta\ngamma"})
    for row in vectors.rows.values():
        row.identity = UNRECORDED_IDENTITY
    embedder.batches.clear()

    sweep = await re_parse_stale(
        store=store,
        pipeline=rebuilt(store, vectors, blobs, embedder, parser=fakes.LineParser()),
        blobs=blobs,
        parse_fingerprints=fingerprints("2"),
    )

    assert embedder.batches == [], "a directory that predates the column keeps every vector"
    assert (sweep.embedding.reused, sweep.embedding.vectors_backfilled) == (3, 3)
    assert all(row.identity != UNRECORDED_IDENTITY for row in vectors.rows.values()), (
        "and the identities are recorded by the write, so the backfill happens once"
    )


async def test_a_second_sweep_makes_no_model_call_and_replaces_no_vector() -> None:
    """Idempotence, at the level that costs money and the level that costs writes."""
    store, vectors, blobs, embedder = await indexed({"a": "alpha", "b": f"be{MARKER}ta"})
    current = fingerprints("2")
    pipeline = rebuilt(store, vectors, blobs, embedder)

    first = await re_parse_stale(
        store=store, pipeline=pipeline, blobs=blobs, parse_fingerprints=current
    )
    assert first.reparsed == 2
    embedder.batches.clear()
    settled = dict(vectors.rows)

    second = await re_parse_stale(
        store=store, pipeline=pipeline, blobs=blobs, parse_fingerprints=current
    )

    assert (second.selected, second.reparsed) == (0, 0)
    assert embedder.batches == []
    assert second.embedding.forward_calls == 0
    assert vectors.rows == settled, "and no vector row moved"


async def test_reuse_survives_a_corpus_larger_than_the_embedding_cache() -> None:
    """Requirement 10: the behaviour must not be the LRU's.

    ``EmbeddingCache`` is a bounded LRU over exact duplicate text at a default capacity of
    10 000, and a corpus larger than it evicts every entry before anything could be reused —
    which is why the measured cost of a sweep over 20 000 distinct chunks was 20 000 forward
    passes twice. This is that shape: every chunk distinct, more of them than the cache holds,
    and a store that has never seen any of them before the first pass.

    The cache is not constructed anywhere in this test. That is the point: nothing here can be
    absorbing anything, so the zero below is the persisted identity's and nobody else's.
    """
    lines = "\n".join(f"distinct line {number}" for number in range(CACHE_CAPACITY + 1))
    store, vectors, blobs, embedder = await indexed({"a": lines})
    first_pass = sum(embedder.batches)
    assert first_pass > CACHE_CAPACITY, (
        "the fixture must be larger than the cache, or this measures the LRU rather than the "
        "persisted identity"
    )
    embedder.batches.clear()

    sweep = await re_parse_stale(
        store=store,
        pipeline=rebuilt(store, vectors, blobs, embedder, parser=fakes.LineParser()),
        blobs=blobs,
        parse_fingerprints=fingerprints("2"),
    )

    assert sum(embedder.batches) == 0, (
        f"the first pass embedded {first_pass} chunks and the re-parse embedded none. Before "
        f"this change the second number was the first number"
    )
    assert sweep.embedding.reused == first_pass


# --- the repair verb ---------------------------------------------------------------------------


async def test_repair_after_a_crash_embeds_only_the_chunks_with_no_vector() -> None:
    """The crash window between chunks written and vectors written, priced honestly.

    ``repair`` re-embeds stored chunks, so before this change a document interrupted with one
    vector missing cost a full re-embed of itself to write that one vector. The stored chunks
    are exactly what the index recorded, so every vector that survived the crash is reusable
    and the report calls the rest repairs rather than changes.
    """
    store, vectors, _, embedder = await indexed({"a": "alpha\nbeta\ngamma\ndelta"})
    document = await store.find_document("memory", "a")
    assert document is not None
    chunks = list(store.chunks[document.id])
    del vectors.rows[chunks[2].id]
    embedder.batches.clear()

    report = await repair(
        [document],
        store=store,
        embedder=embedder,
        vectors=vectors,
        chunk_fingerprint=fakes.BlockChunker().fingerprint,
    )

    assert sum(embedder.batches) == 1, "three vectors survived the crash and one did not"
    assert (report.embedding.reused, report.embedding.repaired) == (3, 1)
    assert report.embedding.input_changed == 0, (
        "nothing about this document changed; it was interrupted"
    )
    assert chunks[2].id in vectors.rows


# --- the identity itself -------------------------------------------------------------------------


def a_chunk(text: str = "alpha", embed_text: str = "Doc > alpha") -> Chunk:
    return Chunk(
        id=chunk_id("doc", 0, text),
        document_id="doc",
        text=text,
        embed_text=embed_text,
        anchor=LineAnchor(start=1, end=1),
        position=0,
        token_count=1,
    )


def identity(text: str = "Doc > alpha", *, document: str = "doc-1", **overrides: object) -> str:
    """An embedding-input identity, with one input at a time varied by the caller."""
    embedder = fakes.CountingEmbedder()
    fields: dict[str, object] = {"embed": embedder.fingerprint, "document_id": document}
    fields.update(overrides)
    return embedding_input_identity(text, **fields)  # pyright: ignore[reportArgumentType]


def test_the_identity_separates_every_input_it_takes() -> None:
    """Four inputs, four ways to change the answer, and no way to collide them."""
    embedder = fakes.CountingEmbedder()

    assert identity() == identity(), (
        "the same four inputs produce the same identity, or nothing is ever reused"
    )
    assert identity() != identity("Doc > beta")
    assert identity() != identity(
        embed=embedder.fingerprint.model_copy(update={"model_id": "other/embedder"})
    )
    assert identity() != identity(middleware=("redact@1",))
    assert identity() != identity(document="doc-2"), (
        "the same string in two documents is two identities, which is what keeps a lookup by "
        "identity from reaching across a tenancy boundary the vector table has no column for"
    )
    assert identity(middleware=("b@1", "a@1")) == identity(middleware=("a@1", "b@1")), (
        "a declaration set that differs only in order is the same declaration"
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        # A separator that a naive concatenation would let one field impersonate another with.
        ("a", '","'),
        ("alpha", "alpha "),
        # NFC and NFD of the same word. They tokenise differently and embed differently, so
        # they are two inputs; a normalising identity would reuse one's vector for the other.
        ("café", "café"),
    ],
)
def test_the_identity_never_conflates_two_different_embedding_inputs(left: str, right: str) -> None:
    """Different strings, different identities — including the ones that look the same.

    The Unicode pair is the case worth stating: a reader who saw "normalized embedding input"
    and reached for :func:`unicodedata.normalize` would make these one input, and the model
    does not. What is normalised is the *serialisation* the digest is taken over, never the
    text.
    """
    assert left != right, "the fixture must supply two different strings"
    assert identity(left) != identity(right)


async def test_a_store_asked_about_a_chunk_it_has_never_held_answers_absent() -> None:
    """The answer is total, so a caller never decides what a missing key meant."""
    vectors = fakes.MemoryVectors()
    embedder = fakes.CountingEmbedder()
    await vectors.ensure_ready(embedder.fingerprint)
    chunks: Sequence[Chunk] = [a_chunk(), a_chunk(text="beta", embed_text="Doc > beta")]

    verdicts = await vectors.stored_vectors(chunks)

    assert set(verdicts) == {chunk.id for chunk in chunks}
    assert all(verdict.state is VectorState.ABSENT for verdict in verdicts.values())


async def test_a_document_with_nothing_to_embed_still_refuses_an_oversized_chunk() -> None:
    """The refusal must not become reachable only when there is embedding work to do.

    ``require_within_context`` is the one thing standing between a *lowered*
    ``max_sequence_length`` and a corpus of silently truncated vectors: the limit is excluded
    from the embed fingerprint, so lowering it changes no fingerprint and fires no comparison.
    Run it over the chunks that survive the partition and a document all of whose vectors are
    reusable would sail past it — and a corpus-wide sweep is exactly the operation made of such
    documents, so the configuration would stay broken and silent for as long as nobody edited
    anything.
    """
    embedder = fakes.CountingEmbedder()
    vectors = fakes.MemoryVectors()
    await vectors.ensure_ready(embedder.fingerprint)
    oversized = a_chunk().model_copy(
        update={"token_count": embedder.fingerprint.max_sequence_length + 1}
    )
    await vectors.upsert([oversized], [[0.0] * embedder.fingerprint.dimension])
    assert (await vectors.stored_vectors([oversized]))[oversized.id].is_reusable, (
        "the fixture must be entirely reusable, or the refusal is reached the ordinary way"
    )

    with pytest.raises(ContextOverflowError):
        await embed_or_reuse(embedder, [oversized], vectors=vectors)

    assert embedder.batches == [], "and it refused before asking the model for anything"


async def test_the_partition_adds_up_however_the_work_falls() -> None:
    """The arithmetic `EmbeddingWork` claims, over a corpus holding all four verdicts at once.

    A report whose parts do not sum to its whole is one an operator cannot reason from, and the
    failure is quiet: every number looks plausible on its own. Asserted here rather than left to
    the class docstring, over one document arranged to contain a reused chunk, a chunk whose
    input moved, a chunk whose row went missing and a chunk that never had one.
    """
    store, vectors, _, embedder = await indexed({"a": "alpha\nbeta\ngamma"})
    document = await store.find_document("memory", "a")
    assert document is not None
    stored = list(store.chunks[document.id])
    vectors.rows.pop(stored[1].id)
    vectors.rows[stored[2].id].embed_text = "no longer what this chunk says"
    fresh = stored[0].model_copy(
        update={"id": "brand-new", "text": "delta", "embed_text": "Doc > delta", "position": 3}
    )
    chunks = [*stored, fresh]

    _, work = await embed_or_reuse(
        embedder,
        chunks,
        vectors=vectors,
        previous={chunk.id: chunk.embed_text for chunk in stored},
    )

    assert work.chunks == len(chunks)
    assert work.reused + work.embedded == work.chunks
    assert work.input_changed + work.first_seen + work.repaired == work.embedded
    assert work.vectors_new + work.vectors_replaced == work.embedded
    assert (work.reused, work.repaired, work.input_changed) == (1, 2, 1), (
        "one untouched, one whose row went missing, one whose row contradicts itself, and one "
        "chunk the index has never seen"
    )
    assert work.first_seen == 0, (
        "the index holds chunks for this document, so nothing here is growth — the unmatched "
        "chunk is a change, and calling it new would price a narrow bump as a first sync"
    )


async def test_a_document_lost_at_the_store_still_reports_what_it_spent_at_the_model() -> None:
    """A failure after the model has run does not refund the forward passes.

    The embed stage runs before the store is touched, so a document whose vectors could not be
    written cost exactly what a document whose vectors were written cost. Counting only the
    documents a sweep managed to rebuild would report it as costing *less* the worse the run
    went — wrong in the one direction nobody checks, because the run being priced is the one
    that failed.

    The counting embedder is the arbiter, not the report: the assertion is that the two agree,
    not that two of the report's own numbers do.
    """
    store, _, blobs, embedder = await indexed({"a": "alpha\nbeta\ngamma"})
    refusing = fakes.RefusingVectors()
    await refusing.ensure_ready(embedder.fingerprint)
    embedder.batches.clear()

    sweep = await re_parse_stale(
        store=store,
        pipeline=rebuilt(store, refusing, blobs, embedder, parser=ReversingLineParser()),
        blobs=blobs,
        parse_fingerprints=fingerprints("2"),
    )

    assert (sweep.failed, sweep.reparsed) == (1, 0), "the fixture must fail at the store"
    assert sum(embedder.batches) == 3, "and it must have reached the model before failing"
    assert sweep.embedding.forward_calls == len(embedder.batches)
    assert sweep.embedding.embedded == sum(embedder.batches), (
        "what the sweep spent is reported even though none of it was committed"
    )


async def test_a_vector_is_never_reused_across_a_document_or_a_workspace_boundary() -> None:
    """The identity-keyed lookup is a read no filter scopes, so the scope is in the key.

    The vector table has no ``workspace_id`` column and by design never will — tenancy lives on
    ``documents``, and a copy in a derived store is a value that can disagree. So a lookup
    keyed on the embedding input alone would be the one vector read in the codebase that is not
    workspace-scoped, and it would stay that way silently: nothing about a query that matches
    too much looks wrong.

    Folding the document into the identity closes it by construction, the same way
    ``document_id(workspace_id, …)`` does one level up. A document id is derived from its
    workspace, so a cross-tenant match cannot be *expressed* rather than merely being unlikely
    to be written. Two chunks with byte-identical text and byte-identical breadcrumbs stand in
    for the two tenants here, because that is the only case in which the question arises.
    """
    embedder = fakes.CountingEmbedder()
    vectors = fakes.MemoryVectors()
    await vectors.ensure_ready(embedder.fingerprint)
    theirs = a_chunk().model_copy(update={"document_id": "their-document", "id": "their-chunk"})
    await vectors.upsert([theirs], [[0.5] * embedder.fingerprint.dimension])

    mine = a_chunk().model_copy(update={"document_id": "my-document", "id": "my-chunk"})
    assert (mine.text, mine.embed_text) == (theirs.text, theirs.embed_text), (
        "the fixture must be byte-identical, or the boundary is not what is being tested"
    )

    verdicts = await vectors.stored_vectors([mine])

    assert verdicts[mine.id].state is VectorState.ABSENT, (
        "the other document's vector is not reachable, by identity or by any other route"
    )
    assert verdicts[mine.id].vector == ()


async def test_a_document_that_never_reached_the_model_reports_no_reuse() -> None:
    """Failing closed is only half of it; the other half is that the number says so.

    #122 makes the same argument about glossary lineage — a document whose detector could not
    read it keeps its stale fingerprint *and* is named, because a run that failed everywhere and
    reported green counters is a system that stopped working quietly. The embed stage's
    accounting owes the same thing one field along: a document lost before the model ran has not
    reused anything, and a report crediting it with reuse would be the accounting defect this
    branch already fixed once, in a new place.

    Nothing is credited here because the accounting is attached to what the commit returns, and
    a stage that raised before the commit never gets that far. Asserted rather than left to that
    structure, because the structure is one `replace` call away from being wrong.
    """
    store, vectors, blobs, embedder = await indexed({"a": "alpha\nbeta"})

    for stage, pipeline in (
        # Lost in the parser, before `_finish` is entered at all.
        ("parse", rebuilt(store, vectors, blobs, embedder, parser=fakes.ExplodingParser())),
        # Lost *inside* `_finish`, one line before the partition would have run. The two take
        # different routes to the same outcome, and only the second passes through the code
        # that attaches this accounting — so a test with only the first would report a guard it
        # does not have.
        (
            "middleware",
            rebuilt(
                store,
                vectors,
                blobs,
                embedder,
                parser=fakes.LineParser(),
                middleware=(RefusingChunkMiddleware(),),
            ),
        ),
    ):
        embedder.batches.clear()
        sweep = await re_parse_stale(
            store=store,
            pipeline=pipeline,
            blobs=blobs,
            parse_fingerprints=fingerprints("2"),
        )

        assert sweep.failed == 1, f"the {stage} fixture must fail before the embed stage"
        assert embedder.batches == [], f"and the {stage} fixture must not reach the model"
        assert sweep.embedding == EmbeddingWork(), (
            f"every field is zero for the {stage} failure, including `reused`: a document lost "
            f"before the model has not had a vector reused, and crediting it would report "
            f"avoided work that was never faced"
        )


async def test_growth_and_change_are_counted_apart() -> None:
    """A first ingest and a corpus-wide bump must not price the same.

    Both send every chunk to the model, so ``embedded`` alone cannot tell them apart — and an
    operator reading one number would conclude a narrow parser change had rewritten the corpus.
    ``first_seen`` is the document having held nothing; ``input_changed`` is it having held
    something that no longer matches.

    The trap this encodes is the definition I got wrong first: "the chunk id is new" looks like
    the same question and is not, because an id is derived from its text, so **every** chunk
    whose text moved arrives with an id the index has never seen. Keyed that way, the re-parse
    below would report growth rather than change.
    """
    store, vectors, blobs, embedder = await indexed({"a": f"alpha\nbe{MARKER}ta"})
    document = await store.find_document("memory", "a")
    assert document is not None
    stored = list(store.chunks[document.id])

    _, fresh = await embed_or_reuse(embedder, stored, vectors=fakes.MemoryVectors(), previous={})
    assert (fresh.first_seen, fresh.input_changed) == (len(stored), 0), (
        "an index holding nothing for this document is growth, all of it"
    )

    sweep = await re_parse_stale(
        store=store,
        pipeline=rebuilt(store, vectors, blobs, embedder),
        blobs=blobs,
        parse_fingerprints=fingerprints("2"),
    )

    assert sweep.embedding.embedded == 1, "the bump moves one chunk's text and no other"
    assert (sweep.embedding.input_changed, sweep.embedding.first_seen) == (1, 0), (
        "and that chunk is a change even though its id is one the index has never seen, "
        "because its text is what the id is derived from"
    )


class PrefixingEmbedMiddleware(PassThroughMiddleware):
    """Rewrites ``embed_text`` and declares it, which is the only legal way to do so.

    A real middleware of the kind the identity has to account for: display text untouched, so
    every chunk id survives and every citation still resolves, and the string the model sees
    prefixed with something the corpus does not contain.
    """

    name = "prefixing-embed"
    mutates_embedded_text = True

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    @override
    async def after_chunk(self, document: Document, chunks: list[Chunk]) -> list[Chunk]:
        del document
        return [
            chunk.model_copy(update={"embed_text": f"{self.prefix}{chunk.embed_text}"})
            for chunk in chunks
        ]


async def test_a_middleware_that_rewrites_embed_text_re_embeds_and_keeps_every_id() -> None:
    """Requirement 3 of the durable-reuse spec, driven end to end rather than at the digest.

    Asserting that the *declaration* changes the identity is a claim about a hash function.
    This is the claim that matters: a middleware that actually rewrites ``embed_text`` runs
    through the pipeline, every chunk id survives because display text never moved, and every
    vector is rebuilt anyway because the string the model sees did.

    Both halves are needed. Re-embedding alone would also be what a system that had given up
    on reuse does; the surviving ids are what make it the interesting case.
    """
    store, vectors, blobs, embedder = await indexed({"a": "alpha\nbeta"})
    document = await store.find_document("memory", "a")
    assert document is not None
    identifiers = {chunk.id for chunk in store.chunks[document.id]}
    texts = {chunk.id: chunk.text for chunk in store.chunks[document.id]}
    embedder.batches.clear()

    sweep = await re_parse_stale(
        store=store,
        pipeline=rebuilt(
            store,
            vectors,
            blobs,
            embedder,
            parser=fakes.LineParser(),
            middleware=(PrefixingEmbedMiddleware("Runbook > "),),
        ),
        blobs=blobs,
        parse_fingerprints=fingerprints("2"),
    )

    rebuilt_chunks = store.chunks[document.id]
    assert {chunk.id for chunk in rebuilt_chunks} == identifiers, (
        "display text never moved, so every id survives and every citation still resolves"
    )
    assert {chunk.id: chunk.text for chunk in rebuilt_chunks} == texts
    assert all(chunk.embed_text.startswith("Runbook > ") for chunk in rebuilt_chunks), (
        "the fixture must actually have rewritten the embedded string"
    )
    assert sum(embedder.batches) == 2, "and both chunks were embedded despite keeping their ids"
    assert (sweep.embedding.reused, sweep.embedding.input_changed) == (0, 2)


async def test_reuse_survives_a_process_restart() -> None:
    """Durable means durable: a second process reuses what the first stored.

    The distinction the whole change turns on. An in-memory cache would report the same zero
    inside one warm process and could not report it here, because nothing in this test carries
    state across the boundary except the store's own directory.

    Simulated by discarding every component except the stores and building the pipeline again —
    which is what a restart is from the corpus's point of view, and is why the 20,000-chunk
    figure in the documents is taken in a fresh interpreter.
    """
    store, vectors, blobs, first_embedder = await indexed({"a": "alpha\nbeta\ngamma"})
    assert sum(first_embedder.batches) == 3, "the first process embedded the corpus"

    # A new embedder, so no in-memory cache anywhere can be carrying the answer, and a new
    # pipeline built from nothing but the stores.
    second_embedder = fakes.CountingEmbedder()
    assert second_embedder is not first_embedder

    sweep = await re_parse_stale(
        store=store,
        pipeline=rebuilt(store, vectors, blobs, second_embedder, parser=fakes.LineParser()),
        blobs=blobs,
        parse_fingerprints=fingerprints("2"),
    )

    assert second_embedder.batches == [], (
        "the second process made no model call at all, and it holds no cache that could have "
        "absorbed one — what it read was the identity the first process persisted"
    )
    assert (sweep.reparsed, sweep.embedding.reused) == (1, 3)
    assert sweep.embedding.cache_hits == 0, "and nothing was served by a warm cache either"


async def test_a_mixed_batch_returns_every_vector_against_the_chunk_it_belongs_to() -> None:
    """Ordering, asserted rather than assumed, on the shape most able to break it.

    Reused vectors come from a store lookup and embedded ones come back from the model in a
    different call; the two are woven together by position. Get that wrong and every citation
    from the seam onward resolves to somebody else's text — with no error, because a vector is
    a vector.

    The fixture alternates hit and miss so a naive concatenation, which would pass on any run
    where the two groups happen to be contiguous, cannot.
    """
    embedder = fakes.CountingEmbedder()
    vectors = fakes.MemoryVectors()
    await vectors.ensure_ready(embedder.fingerprint)
    chunks = [
        a_chunk(text=f"line {index}", embed_text=f"Doc > line {index}").model_copy(
            update={"id": f"chunk-{index}", "position": index}
        )
        for index in range(8)
    ]
    # Every other chunk already stored, so the partition alternates.
    stored = chunks[::2]
    known = await embedder.embed([chunk.embed_text for chunk in stored])
    await vectors.upsert(stored, known)
    embedder.batches.clear()

    produced, work = await embed_or_reuse(embedder, chunks, vectors=vectors)

    assert (work.reused, work.embedded) == (4, 4), "the fixture must alternate, not clump"
    expected = await embedder.embed([chunk.embed_text for chunk in chunks])
    assert [list(vector) for vector in produced] == [list(vector) for vector in expected], (
        "every position holds the embedding of the chunk at that position, whether it came "
        "from the store or from the model"
    )
