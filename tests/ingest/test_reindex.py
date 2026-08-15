"""Re-ingest from what is on disk, and the check the re-embed path most needs."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import override

import pytest

from manicule.core.anchors import Unlocated
from manicule.core.content import Chunk, DocumentStatus, ParsedBlock, RawDocument, Retention
from manicule.core.embedding import Vector
from manicule.core.fingerprints import ChunkFingerprint, ParseFingerprint
from manicule.ingest.reindex import re_embed, re_parse, repair, select
from manicule.parsers.config import CONFLUENCE_MEDIA_TYPE, ConfluenceConfig, WebConfig
from manicule.parsers.confluence import ConfluenceStorageParser
from manicule.parsers.versions import current_parse_fingerprints, parse_fingerprint
from manicule.parsers.web import WebParser
from manicule.testing import assert_refuses_oversized_chunks
from tests.fakes import HashEmbedder, make_chunks, make_document
from tests.ingest import fakes
from tests.ingest.test_pipeline import build


async def test_the_re_embed_path_refuses_a_chunk_the_model_would_truncate() -> None:
    """The conformance suite, run against the path it was written for.

    Re-embed reads stored ``embed_text`` and does not re-chunk, so the chunker's budget refusal
    never runs — and ``max_sequence_length`` is excluded from fingerprint identity, so a limit
    that *fell* fires no comparison either. Without this check one command truncates a corpus
    in silence.
    """
    store = fakes.MemoryIngestStore()
    vectors = fakes.MemoryVectors()
    embedder = HashEmbedder()
    document = make_document()
    store.documents[document.id] = document

    async def embed_batch(chunks: Sequence[Chunk]) -> object:
        await store.replace_chunks(document.id, list(chunks))
        report = await re_embed(
            [document],
            store=store,
            embedder=embedder,
            vectors=vectors,
            chunk_fingerprint=fakes.BlockChunker.fingerprint,
        )
        if report.failures:
            from manicule.core.errors import ContextOverflowError  # noqa: PLC0415

            raise ContextOverflowError(report.failures[0])
        return report

    await assert_refuses_oversized_chunks(embed_batch, embedder)


async def test_re_embed_rebuilds_vectors_without_a_parser_or_the_network() -> None:
    """Rung 2. The whole return on storing ``embed_text`` rather than only ``text``."""
    store = fakes.MemoryIngestStore()
    vectors = fakes.MemoryVectors()
    document = make_document()
    store.documents[document.id] = document
    await store.replace_chunks(document.id, make_chunks(document, count=3))

    report = await re_embed(
        [document],
        store=store,
        embedder=HashEmbedder(),
        vectors=vectors,
        chunk_fingerprint=fakes.BlockChunker.fingerprint,
    )

    assert report.documents == 1
    assert report.chunks == 3
    assert len(vectors.rows) == 3


async def test_re_embed_does_not_claim_a_new_chunk_lineage() -> None:
    """It did not re-chunk, so saying otherwise makes a later repair query answer wrongly."""
    store = fakes.MemoryIngestStore()
    document = make_document()
    store.documents[document.id] = document
    await store.replace_chunks(document.id, make_chunks(document))
    await store.set_lineage(document.id, chunk_fp="old-chunker", embed_fp="old-model")

    await re_embed(
        [document],
        store=store,
        embedder=HashEmbedder(),
        vectors=fakes.MemoryVectors(),
        chunk_fingerprint=fakes.BlockChunker.fingerprint,
    )

    chunk_fp, embed_fp = store.lineage[document.id]
    assert chunk_fp == "old-chunker"
    assert embed_fp == HashEmbedder().fingerprint.canonical()


async def test_repair_finishes_a_document_an_interrupted_run_left_half_written() -> None:
    """Crash between writing chunks and writing vectors: the chunks are the repair's input."""
    store = fakes.MemoryIngestStore()
    vectors = fakes.MemoryVectors()
    document = make_document().model_copy(
        update={"status": DocumentStatus.PENDING, "status_detail": None}
    )
    store.documents[document.id] = document
    await store.replace_chunks(document.id, make_chunks(document, count=2))

    report = await repair(
        [document],
        store=store,
        embedder=HashEmbedder(),
        vectors=vectors,
        chunk_fingerprint=fakes.BlockChunker.fingerprint,
    )

    assert report.documents == 1
    assert store.documents[document.id].status is DocumentStatus.INDEXED
    assert len(vectors.rows) == 2


async def test_repair_names_a_document_it_cannot_finish_rather_than_failing_the_run() -> None:
    """A document with no chunks needs a re-parse, and an operator needs to be told which."""
    store = fakes.MemoryIngestStore()
    document = make_document()
    store.documents[document.id] = document

    report = await repair(
        [document],
        store=store,
        embedder=HashEmbedder(),
        vectors=fakes.MemoryVectors(),
        chunk_fingerprint=fakes.BlockChunker.fingerprint,
    )

    assert report.documents == 0
    assert len(report.unrepairable) == 1
    assert "re-parse" in report.unrepairable[0]


async def test_re_parse_runs_the_current_chain_over_retained_bytes() -> None:
    """Rung 3, and the reason retaining bytes pays for itself.

    The bytes are unchanged, so change detection would ordinarily skip this document — which
    is exactly the operation being asked for, and why the re-parse path forces past it.
    """
    blobs = fakes.MemoryBlobs()
    pipeline, store, _ = build(blobs=blobs)
    await pipeline.run(fakes.DictConnector({"a": "alpha\nbeta"}))
    document = await store.find_document("memory", "a")
    assert document is not None
    before = [chunk.id for chunk in store.chunks[document.id]]

    report = await re_parse([document], pipeline=pipeline, blobs=blobs)

    assert report.documents == 1
    assert [chunk.id for chunk in store.chunks[document.id]] == before, (
        "a chunk that survives a re-parse unchanged keeps its id, and therefore its vector"
    )


async def test_re_parse_reuses_retention_without_requesting_more_capacity() -> None:
    class RefusingRetentionBlobs(fakes.MemoryBlobs):
        refusing = False

        @override
        async def retain(self, data: bytes, media_type: str | None = None) -> Retention:
            if self.refusing:
                raise AssertionError("re-parse must not retain an already durable snapshot again")
            return await super().retain(data, media_type)

    blobs = RefusingRetentionBlobs()
    pipeline, store, _ = build(blobs=blobs)
    await pipeline.run(fakes.DictConnector({"private-source": "private body"}))
    document = await store.find_document("memory", "private-source")
    assert document is not None
    blobs.refusing = True

    report = await re_parse([document], pipeline=pipeline, blobs=blobs)

    assert report.documents == 1
    assert not report.failures


async def test_a_parser_rules_bump_re_parses_its_documents_without_the_connector() -> None:
    """The claim a rules bump makes, run rather than asserted.

    Bumping ``PARSERS["confluence"].rules`` is only worth doing if something acts on it, and the
    something is this: ``select`` finds the documents whose recorded lineage is not one an
    installed parser would produce now, and ``re_parse`` rebuilds them from retained bytes. The
    connector is present throughout and is never asked for anything — ``fetches`` is captured
    after the sync and compared afterwards, so "no network" is a measurement rather than an
    inference from ``re_parse``'s signature.

    The stale lineage is written as ``confluence`` rules **1**, which is the version the macro
    paragraph rule moved away from. A later bump makes that "an earlier version" rather than
    "the previous one", which is still exactly what this case needs, so it does not have to be
    edited every time the parser moves.
    """
    source = (
        '<ac:structured-macro ac:name="expand">'
        "<ac:rich-text-body><p>First paragraph.</p><p>Second paragraph.</p>"
        "</ac:rich-text-body></ac:structured-macro>"
    )
    blobs = fakes.MemoryBlobs()
    pipeline, store, _ = build(
        blobs=blobs,
        parsers={"confluence": ConfluenceStorageParser(ConfluenceConfig())},
        chain=("confluence",),
    )
    connector = fakes.DictConnector({"page": source})
    connector.media_types["page"] = CONFLUENCE_MEDIA_TYPE
    await pipeline.run(connector)
    document = await store.find_document("memory", "page")
    assert document is not None
    installed = parse_fingerprint("confluence")
    assert installed is not None
    assert document.parse_fp == installed.canonical(), (
        "first ingest records what the installed parser produced"
    )
    downloads = list(connector.fetches)

    superseded = ParseFingerprint(
        parser="confluence", version="1", libraries=dict(installed.libraries)
    )
    await store.set_lineage(
        document.id, chunk_fp=None, embed_fp=None, parse_fp=superseded.canonical()
    )
    stale = await select(store, parse_fingerprints=current_parse_fingerprints())
    assert [chosen.id for chosen in stale] == [document.id], (
        "the bump selects this document, and the selector is the complement of what is installed"
    )

    report = await re_parse(stale, pipeline=pipeline, blobs=blobs)

    assert report.documents == 1
    assert report.unrepairable == []
    assert report.failures == []
    assert connector.fetches == downloads, "the source was not asked for the bytes a second time"
    rebuilt = "\n".join(chunk.text for chunk in store.chunks[document.id])
    assert "First paragraph.\n\nSecond paragraph." in rebuilt, (
        "and the rebuilt text is what the current parser produces, not what the stale "
        "fingerprint claimed was current"
    )


_BREAK_PAGE = """<html><head><title>Signal routing</title></head><body>
<h1 id="routing">Signal routing</h1>
<p>The first paragraph, which holds no break and must not move.</p>
<p>primary endpoint<br/>secondary endpoint</p>
<p>The last paragraph, which holds no break either.</p>
</body></html>"""


class _GluedWebParser(WebParser):
    """The HTML parser as ``html`` rules **2** read a page: a ``<br>`` contributed nothing.

    A stand-in for the previous version rather than a copy of it, and the difference it
    reproduces is the only one there is — under version 2 the break contributed no character,
    so the fragments either side were glued. This fixture's prose blocks carry newlines from
    nothing else, so stripping them reproduces exactly what was stored.
    """

    @override
    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        async for block in super().parse(raw):
            yield block.model_copy(update={"text": block.text.replace("\n", "")})


class _CountingEmbedder(HashEmbedder):
    """Records every text it was asked to embed, so "re-embedded" can be counted."""

    def __init__(self) -> None:
        super().__init__()
        self.embedded: list[str] = []

    @override
    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        self.embedded.extend(texts)
        return await super().embed(texts)


async def test_the_break_bump_rebuilds_from_retained_bytes_and_keeps_unmoved_chunk_identity() -> (
    None
):
    """The migration a ``<br>`` becoming a newline costs, run end to end.

    A document stored under the superseded fingerprint is selected; it is rebuilt from retained
    bytes with the connector present and never asked; the one chunk whose text moved gets a new
    id and a new vector; and the chunks that did not move keep both, because a chunk id is
    derived from its content and re-parse reconciles against the stored set. Keeping the id is
    what keeps every stored citation pointing where it pointed.

    **What is not kept is the embedding work**, and it is asserted here because reading the code
    suggests otherwise. ``re_parse`` runs the ordinary ingest path, which embeds every chunk it
    is handed; the vector an unchanged chunk ends up with is identical, but it was recomputed to
    get there. Measured on a four-chunk document whose text did not move at all: four chunks
    embedded. ``docs/parsing.md`` §4.5 said the opposite until this case was written.

    **Four chunks, one ``embed()`` call.** The assertion below counts texts, because
    :class:`_CountingEmbedder` extends by ``texts`` — which is the number that matters, since a
    forward pass per chunk is what a re-embed costs. It is not a count of calls: the embedder
    batches, and this document's four chunks go in one. Written down because the docstring said
    "four calls" and the number it named was the other one.

    The old parser is stood in for rather than checked out, so this states what changed between
    the two versions in one line of code instead of pinning the whole of the previous one.
    """
    blobs = fakes.MemoryBlobs()
    store = fakes.MemoryIngestStore()
    vectors = fakes.MemoryVectors()
    embedder = _CountingEmbedder()
    connector = fakes.DictConnector({"page": _BREAK_PAGE})
    connector.media_types["page"] = "text/html"

    stale_pipeline, _, _ = build(
        store=store,
        vectors=vectors,
        blobs=blobs,
        embedder=embedder,
        parsers={"html": _GluedWebParser(WebConfig())},
        chain=("html",),
    )
    await stale_pipeline.run(connector)
    document = await store.find_document("memory", "page")
    assert document is not None
    stored = {chunk.id: chunk.text for chunk in store.chunks[document.id]}
    assert "primary endpointsecondary endpoint" in stored.values(), (
        "the corpus this migration repairs: two fragments glued into a word the page never had"
    )
    kept = {
        chunk.id: vectors.rows[chunk.id]
        for chunk in store.chunks[document.id]
        if "endpoint" not in chunk.text
    }
    downloads = list(connector.fetches)

    installed = parse_fingerprint("html")
    assert installed is not None
    superseded = ParseFingerprint(parser="html", version="2", libraries=dict(installed.libraries))
    await store.set_lineage(
        document.id, chunk_fp=None, embed_fp=None, parse_fp=superseded.canonical()
    )
    selected = await select(store, parse_fingerprints=current_parse_fingerprints())
    assert [chosen.id for chosen in selected] == [document.id], (
        "the bump selects this document: the selector is the complement of what is installed"
    )

    current_pipeline, _, _ = build(
        store=store,
        vectors=vectors,
        blobs=blobs,
        embedder=embedder,
        parsers={"html": WebParser(WebConfig())},
        chain=("html",),
    )
    embedder.embedded.clear()
    report = await re_parse(selected, pipeline=current_pipeline, blobs=blobs)

    assert report.documents == 1
    assert report.failures == []
    assert connector.fetches == downloads, "the source was not asked for the bytes a second time"
    rebuilt = {chunk.id: chunk.text for chunk in store.chunks[document.id]}
    assert "primary endpoint\nsecondary endpoint" in rebuilt.values(), (
        "the rebuilt text is what the current parser produces, break included"
    )
    assert set(kept) <= set(rebuilt), "a chunk whose text did not move keeps its id"
    assert all(vectors.rows[chunk_id] == vector for chunk_id, vector in kept.items()), (
        "and therefore its stored vector, which is what keeps its citations pointing where "
        "they pointed"
    )
    moved = set(rebuilt) - set(stored)
    assert len(moved) == 1, "exactly one chunk's text moved, so exactly one id is new"
    assert rebuilt[moved.pop()] == "primary endpoint\nsecondary endpoint"
    assert len(embedder.embedded) == len(rebuilt), (
        "and every chunk was re-embedded to get there, unchanged ones included: `re_parse` "
        "runs the ordinary ingest path, which has no skip for a chunk whose id it already has"
    )


async def test_a_document_with_no_retained_bytes_is_named_rather_than_failed() -> None:
    """Those are the only documents for which a re-crawl is the only repair."""
    pipeline, store, _ = build()
    await pipeline.run(fakes.DictConnector({"a": "alpha"}))
    document = await store.find_document("memory", "a")
    assert document is not None

    report = await re_parse([document], pipeline=pipeline, blobs=fakes.MemoryBlobs())

    assert report.documents == 0
    assert len(report.unrepairable) == 1
    assert "re-sync" in report.unrepairable[0]


async def test_selection_is_a_query_over_lineage_rather_than_a_scan() -> None:
    """A grammar upgrade invalidates code documents and nothing else, and saying so is a
    ``WHERE`` clause rather than a rebuild."""
    store = fakes.MemoryIngestStore()
    old = make_document().model_copy(update={"id": "old", "source_id": "old"})
    current = make_document().model_copy(update={"id": "current", "source_id": "current"})
    store.documents.update({old.id: old, current.id: current})
    now = fakes.BlockChunker.fingerprint
    await store.set_lineage(old.id, chunk_fp="a-previous-chunker", embed_fp=None)
    await store.set_lineage(current.id, chunk_fp=now.canonical(), embed_fp=None)

    chosen = await select(store, chunk_fingerprint=now)

    assert [document.id for document in chosen] == ["old"]


async def test_a_library_bump_selects_its_own_documents_and_no_others() -> None:
    """The re-parse selector, checked in both directions.

    ``--re-parse`` is what closes the window between an upgrade and the next sync, during
    which every anchor stored under the old version is being resolved by the new one. It takes
    a *set* of current fingerprints, because parsing has no single corpus-wide identity: a
    ``pypdfium2`` release makes the PDFs stale and says nothing about the Markdown.
    """
    store = fakes.MemoryIngestStore()
    pdf_old = ParseFingerprint(parser="pdf", version="1", libraries={"pypdfium2": "5.12.1"})
    pdf_new = ParseFingerprint(parser="pdf", version="1", libraries={"pypdfium2": "5.13.0"})
    markdown = ParseFingerprint(parser="markdown", version="1", libraries={"markdown-it-py": "4"})
    for name, lineage in (("stale", pdf_old), ("fresh", pdf_new), ("other", markdown)):
        document = make_document().model_copy(update={"id": name, "source_id": name})
        store.documents[document.id] = document
        await store.set_lineage(
            document.id, chunk_fp=None, embed_fp=None, parse_fp=lineage.canonical()
        )
    unversioned = make_document().model_copy(update={"id": "plugin", "source_id": "plugin"})
    store.documents[unversioned.id] = unversioned

    chosen = await select(store, parse_fingerprints=[pdf_new, markdown])

    assert sorted(document.id for document in chosen) == ["plugin", "stale"], (
        "the stale PDF and the document with no recorded lineage, and nothing else"
    )


async def test_selecting_on_parse_lineage_is_opt_in() -> None:
    """``None`` means "do not filter"; an empty set means "nothing is current".

    Collapsing the two with a falsy test would make an installation with no parsers select
    every document — or, the other way round, make the ordinary unfiltered call select none.
    """
    store = fakes.MemoryIngestStore()
    document = make_document()
    store.documents[document.id] = document

    assert len(await select(store)) == 1
    assert len(await select(store, parse_fingerprints=[])) == 1


@pytest.mark.parametrize("budget", [16, 512])
async def test_the_batch_size_is_derived_from_the_budget_rather_than_constant(
    budget: int,
) -> None:
    """A constant is wrong in both directions.

    Thirty-two chunks of 512 tokens is a very different allocation from thirty-two of 8 000,
    and the second is where an in-process embedder runs the machine out of memory.
    """
    from manicule.ingest.embedding import batch_size, embed_chunks  # noqa: PLC0415

    embedder = fakes.CountingEmbedder()
    fingerprint = ChunkFingerprint(
        chunker="block",
        version="1",
        max_tokens=budget,
        overlap_tokens=0,
        tokenizer_id="whitespace",
    )
    chunks = [
        Chunk(
            id=f"c{n}",
            document_id="d",
            text="x",
            embed_text="x",
            anchor=Unlocated(reason="synthetic"),
            position=n,
            token_count=1,
        )
        for n in range(64)
    ]

    await embed_chunks(embedder, chunks, chunk_fingerprint=fingerprint, target_batch_tokens=1024)

    expected = batch_size(budget_tokens=budget, target_batch_tokens=1024)
    assert embedder.batches[0] == expected
