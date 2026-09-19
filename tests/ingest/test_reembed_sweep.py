"""The whole-corpus re-embed: a model whose vectors moved while its fingerprint did not.

Everything here drives the **real** pipeline over an in-memory store. The embedder standing in
for the upgraded runtime keeps the original's fingerprint exactly and returns different numbers,
which is the whole premise: every identity the index checks agrees, so the only way to tell an
old vector from a new one is to compare the numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from manicule.core.content import DocumentStatus
from manicule.ingest.reindex import NO_RETAINED_BYTES, plan_re_embed, re_embed_all, re_parse
from manicule.parsers.expansion import MemberFailure
from tests.ingest import fakes
from tests.ingest.test_pipeline import build, parse_versions
from tests.ingest.test_reindex_sweep import corpus

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from manicule.core.content import RawDocument
    from manicule.core.embedding import Vector
    from manicule.ingest.pipeline import IngestPipeline
    from manicule.parsers.expansion import MemberOutcome

PAGES = {"a": "alpha\nbeta", "b": "gamma\ndelta\nepsilon", "c": "zeta"}
"""Three documents, so a batch of two pages the selection once and then stops on a short page."""

DRIFT = 0.5
"""How far every component moves. Large, so no comparison below depends on float tolerance."""


class DriftedEmbedder(fakes.CountingEmbedder):
    """The same fingerprint and different numbers: a runtime upgraded under unchanged weights."""

    @override
    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        vectors = await super().embed(texts)
        return [[value + DRIFT for value in vector] for vector in vectors]


def drifted_pipeline(
    store: fakes.MemoryIngestStore,
    vectors: fakes.MemoryVectors,
    blobs: fakes.MemoryBlobs,
    embedder: fakes.CountingEmbedder,
) -> IngestPipeline:
    """The same corpus and the same parser, with the upgraded runtime behind the embedder."""
    pipeline, _, _ = build(
        store=store,
        vectors=vectors,
        blobs=blobs,
        embedder=embedder,
        parse_fingerprints=parse_versions(lines="1"),
    )
    return pipeline


def stored(vectors: fakes.MemoryVectors) -> dict[str, tuple[float, ...]]:
    return {chunk_id: tuple(row.vector) for chunk_id, row in vectors.rows.items()}


async def test_the_fixture_is_the_premise_one_fingerprint_two_sets_of_numbers() -> None:
    """If the drifted embedder had its own fingerprint, every test below would prove nothing.

    A different fingerprint is refused by the pipeline and handled by ``manicule reembed``. What
    this verb is for is the case where the fingerprint cannot tell them apart.
    """
    original, drifted = fakes.CountingEmbedder(), DriftedEmbedder()
    assert drifted.fingerprint == original.fingerprint
    assert await drifted.embed(["same text"]) != await original.embed(["same text"])


async def test_a_sync_under_a_drifted_model_keeps_every_old_vector() -> None:
    """Why the verb exists: the ordinary path cannot see the drift, and reuses its way past it."""
    store, vectors, blobs, connector, _ = await corpus(PAGES)
    before = stored(vectors)

    await drifted_pipeline(store, vectors, blobs, DriftedEmbedder()).run(connector)

    assert stored(vectors) == before


async def test_so_does_a_re_parse_that_reuses_what_it_finds() -> None:
    """``--stale``'s path with reuse on, the one ``--re-embed`` switches off. It changes nothing."""
    store, vectors, blobs, _, _ = await corpus(PAGES)
    before = stored(vectors)

    report = await re_parse(
        list(store.documents.values()),
        pipeline=drifted_pipeline(store, vectors, blobs, DriftedEmbedder()),
        blobs=blobs,
    )

    assert report.embedding.reused == len(before)
    assert stored(vectors) == before


async def test_every_vector_is_replaced_and_none_is_reused() -> None:
    """The claim the command makes, measured against the embedder that should have produced it."""
    store, vectors, blobs, _, _ = await corpus(PAGES)
    before = stored(vectors)
    drifted = DriftedEmbedder()

    sweep = await re_embed_all(
        store=store,
        pipeline=drifted_pipeline(store, vectors, blobs, drifted),
        blobs=blobs,
        batch=2,
    )

    assert sweep.selected == len(PAGES)
    assert sweep.reembedded == len(PAGES), "a batch of two must still reach the third document"
    assert (sweep.unrepairable, sweep.failed, sweep.superseded) == (0, 0, 0)
    assert sweep.chunks == len(before)
    assert sweep.embedding.reused == 0
    assert sweep.embedding.refreshed == len(before), "every stored vector looked current"
    assert sweep.embedding.input_changed == 0, "and no chunk's text moved"
    assert sweep.embedding.forward_calls == len(drifted.batches)
    reference = DriftedEmbedder()
    for chunk_id, row in vectors.rows.items():
        assert row.vector != before[chunk_id], f"{chunk_id} still holds its old vector"
        assert list(row.vector) == (await reference.embed([row.embed_text]))[0]


async def test_each_document_is_published_anew_rather_than_rewritten() -> None:
    """New vectors are a new content address, which is how the commit swaps them atomically.

    A document's publication id is derived from its vectors, so a document this re-embedded and
    a document it only claimed to have re-embedded are told apart by whether the id moved.
    """
    store, vectors, blobs, _, _ = await corpus(PAGES)
    before = {document.id: document.publication_id for document in store.documents.values()}

    await re_embed_all(
        store=store,
        pipeline=drifted_pipeline(store, vectors, blobs, DriftedEmbedder()),
        blobs=blobs,
    )

    for document in store.documents.values():
        assert document.status is DocumentStatus.INDEXED
        assert document.publication_id != before[document.id], f"{document.id} was not republished"


async def test_a_second_run_with_no_drift_between_changes_nothing() -> None:
    """Running it again is safe: the same vectors are the same publication, written again."""
    store, vectors, blobs, _, _ = await corpus(PAGES)
    pipeline = drifted_pipeline(store, vectors, blobs, DriftedEmbedder())
    await re_embed_all(store=store, pipeline=pipeline, blobs=blobs)
    after_first = stored(vectors)
    publications = {document.id: document.publication_id for document in store.documents.values()}

    again = await re_embed_all(store=store, pipeline=pipeline, blobs=blobs)

    assert again.reembedded == len(PAGES)
    assert stored(vectors) == after_first
    assert {
        document.id: document.publication_id for document in store.documents.values()
    } == publications


async def test_a_plan_prices_exactly_what_the_run_embeds_and_writes_nothing() -> None:
    """Chunks are the price, and a plan that disagreed with its run would be a guess."""
    store, vectors, blobs, _, _ = await corpus(PAGES)
    before = stored(vectors)

    plan = await plan_re_embed(store=store, batch=2)

    assert plan.dry_run is True
    assert plan.selected == len(PAGES)
    assert plan.chunks == len(before)
    assert plan.reembedded == 0
    assert stored(vectors) == before

    run = await re_embed_all(
        store=store,
        pipeline=drifted_pipeline(store, vectors, blobs, DriftedEmbedder()),
        blobs=blobs,
        batch=2,
    )
    assert run.chunks == plan.chunks


async def test_a_document_with_no_retained_bytes_is_named_by_the_plan_and_the_run_alike() -> None:
    """The one document only a re-sync can reach, named in one sentence by both.

    The plan finds it from the row and the run finds it again before reading a blob, so the two
    are compared line for line. The rest of the corpus is still re-embedded.
    """
    store, vectors, blobs, _, _ = await corpus(PAGES)
    orphan = next(iter(store.documents.values()))
    store.documents[orphan.id] = orphan.model_copy(update={"original_ref": None})
    kept = {
        chunk_id: vector
        for chunk_id, vector in stored(vectors).items()
        if vectors.rows[chunk_id].document_id == orphan.id
    }

    plan = await plan_re_embed(store=store)
    run = await re_embed_all(
        store=store,
        pipeline=drifted_pipeline(store, vectors, blobs, DriftedEmbedder()),
        blobs=blobs,
    )

    assert run.unrepairable == plan.unrepairable == 1
    assert run.unrepairable_documents == plan.unrepairable_documents
    assert NO_RETAINED_BYTES in run.unrepairable_documents[0]
    assert run.reembedded == len(PAGES) - 1
    assert run.chunks == plan.chunks, "the plan does not price the document it cannot reach"
    assert {chunk_id: stored(vectors)[chunk_id] for chunk_id in kept} == kept


async def test_only_documents_search_is_serving_are_selected() -> None:
    """A failed document has no published vectors to be wrong."""
    store, vectors, blobs, _, _ = await corpus(PAGES)
    failed = next(iter(store.documents.values()))
    store.documents[failed.id] = failed.model_copy(update={"status": DocumentStatus.FAILED})

    plan = await plan_re_embed(store=store)
    run = await re_embed_all(
        store=store,
        pipeline=drifted_pipeline(store, vectors, blobs, DriftedEmbedder()),
        blobs=blobs,
    )

    assert plan.selected == run.selected == len(PAGES) - 1


class ReadsAndExpands(fakes.LineParser):
    """Text of its own and one member that cannot be read: a message with an encrypted attachment.

    The shape that is ``indexed`` *and* has members, so the sweep selects it and its re-parse
    comes back as two outcomes. A pure container never reaches the sweep; its status is
    ``container``.
    """

    async def expand(self, raw: RawDocument) -> AsyncIterator[MemberOutcome]:
        yield MemberFailure(
            source_id=f"{raw.source_id}!/sealed",
            uri=f"fake:{raw.uri}!/sealed",
            status=DocumentStatus.FAILED,
            reason="member is encrypted",
            depth=1,
        )


async def test_a_member_that_fails_does_not_count_its_document_as_failed() -> None:
    """The document is judged by its own outcome; the failed member is named, not counted."""
    store, vectors, blobs = fakes.MemoryIngestStore(), fakes.MemoryVectors(), fakes.MemoryBlobs()
    original = fakes.CountingEmbedder()
    await vectors.ensure_ready(original.fingerprint)

    def pipeline_for(embedder: fakes.CountingEmbedder) -> IngestPipeline:
        pipeline, _, _ = build(
            store=store,
            vectors=vectors,
            blobs=blobs,
            embedder=embedder,
            parsers={"lines": ReadsAndExpands()},
            parse_fingerprints=parse_versions(lines="1"),
        )
        return pipeline

    await pipeline_for(original).run(fakes.DictConnector({"message": "a body\nof two lines"}))
    message = await store.find_document("memory", "message")
    assert message is not None
    assert message.status is DocumentStatus.INDEXED, "the fixture must be selected by the sweep"
    before = stored(vectors)

    sweep = await re_embed_all(store=store, pipeline=pipeline_for(DriftedEmbedder()), blobs=blobs)

    assert sweep.selected == 1
    assert (sweep.reembedded, sweep.failed) == (1, 0)
    assert any("member is encrypted" in line for line in sweep.failures), "still named"
    assert all(stored(vectors)[chunk_id] != vector for chunk_id, vector in before.items())
