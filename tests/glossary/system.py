"""A whole retrieval system over the synthetic glossary corpus.

The shipped :class:`~manicule.retrieval.retriever.Retriever`, the shipped stages and a migrated
SQLite database. Only the embedder and the vector store are doubles, and the vector store
really ranks by cosine — a fixture that stubbed the ranking could not demonstrate anything
about ranking.

Glossary entries are written through :func:`~manicule.ingest.glossary.detect_entries`, the same
function ingest calls, rather than constructed by hand. A fixture that hand-wrote its entries
would pass whether or not detection worked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from manicule.core.retrieval import Filter
from manicule.ingest.glossary import detect_entries
from manicule.retrieval.assembly import ContextAssembler
from manicule.retrieval.cache import L1QueryCache
from manicule.retrieval.dense import DenseStage
from manicule.retrieval.expansion import ExpansionPolicy
from manicule.retrieval.fusion import RRFStage
from manicule.retrieval.lexical import LexicalStage
from manicule.retrieval.profile import Profiles
from manicule.retrieval.retriever import Retriever
from manicule.retrieval.runner import PipelineRunner
from manicule.retrieval.tokens import ContextTokenCounter
from tests.evaluation.fakes import CosineVectorStore
from tests.glossary import corpus
from tests.storage_helpers import make_chunk, make_document

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manicule.core.content import Chunk, Document
    from manicule.core.protocols import Embedder
    from manicule.retrieval.ports import GlossarySource
    from manicule.storage.docstore import SqliteDocStore

WORKSPACE = "default"
SCOPE = frozenset({WORKSPACE})

PROFILE_OVERRIDES = {"candidates": 20, "final_top_k": 10, "min_score": 0.0}
"""A floor of zero, because the doubles are not calibrated to any similarity scale.

A ``min_score`` tuned for a real model would discard a stand-in embedder's candidates before
the assertion saw them, which would make a test pass for the wrong reason.
"""


async def index(
    store: SqliteDocStore,
    source_id: str,
    title: str,
    texts: Sequence[str],
    *,
    workspace_id: str = WORKSPACE,
    detect: bool = True,
) -> tuple[Document, list[Chunk]]:
    """Store one document whose chunks are ``texts``, and its detected definitions.

    One chunk per element, because the fixture's whole argument is about what a *chunk*
    contains: the glossary page is one chunk holding twenty-five entries, and each other
    passage is a chunk of its own.
    """
    document = make_document(
        source="fixture",
        source_id=source_id,
        workspace_id=workspace_id,
        title=title,
        uri=f"file:///{source_id}.md",
        body="\n".join(texts).encode(),
    )
    await store.upsert_document(document)
    chunks = [
        make_chunk(document, position, text, heading_path=(title,))
        for position, text in enumerate(texts)
    ]
    await store.replace_chunks(document.id, chunks)
    entries = detect_entries(chunks, title=title) if detect else []
    await store.replace_glossary_entries(document.id, entries)
    return document, chunks


async def build_corpus(store: SqliteDocStore, *, workspace_id: str = WORKSPACE) -> list[Chunk]:
    """The whole fixture: one glossary page, forty-five ordinary uses, fifteen usages."""
    chunks: list[Chunk] = []
    _, glossary = await index(
        store,
        "glossary",
        corpus.GLOSSARY_TITLE,
        [corpus.glossary_page()],
        workspace_id=workspace_id,
    )
    chunks.extend(glossary)
    _, handbook = await index(
        store,
        "handbook",
        "Operations handbook",
        corpus.passages(),
        workspace_id=workspace_id,
    )
    chunks.extend(handbook)
    return chunks


async def retriever_over(
    store: SqliteDocStore,
    embedder: Embedder,
    chunks: Sequence[Chunk],
    *,
    policy: ExpansionPolicy | None = None,
    glossary: GlossarySource | bool = True,
    cache: L1QueryCache | None = None,
) -> Retriever:
    """The shipped retriever, dense + lexical + RRF, over ``chunks``.

    ``glossary=False`` builds the **baseline**: the identical pipeline with no glossary source
    wired at all. That is what makes the before-and-after comparison a comparison of one thing
    — a baseline built by also changing the stage list, the profile or the store would not be.

    ``glossary=<a source>`` substitutes one, which is how the leaky one is exercised. It is
    matched with ``is True`` rather than by truthiness, and that is a fix rather than fussiness:
    the first version tested ``if glossary``, so passing a source silently selected the real
    store instead and a tenancy test passed while checking the wrong object entirely.
    """
    vectors = CosineVectorStore()
    await vectors.ensure_ready(embedder.fingerprint)
    await vectors.upsert(chunks, await embedder.embed([chunk.embed_text for chunk in chunks]))

    profiles = Profiles(PROFILE_OVERRIDES)
    dense = DenseStage(embedder=embedder, vectors=vectors, docstore=store, profiles=profiles)
    lexical = LexicalStage(docstore=store, profiles=profiles)
    fusion = RRFStage()
    return Retriever(
        runner=PipelineRunner([dense, lexical, fusion], docstore=store),
        docstore=store,
        assembler=ContextAssembler(counter=ContextTokenCounter(), profiles=profiles),
        profiles=profiles,
        cache=cache,
        legs=fusion.legs,
        rrf_k=fusion.k,
        embed_fingerprint=embedder.fingerprint.canonical(),
        glossary=_source(store, glossary),
        expansion=policy or ExpansionPolicy(),
    )


def _source(store: SqliteDocStore, glossary: GlossarySource | bool) -> GlossarySource | None:
    if glossary is True:
        return store
    if glossary is False:
        return None
    return glossary


def query_filter(*, workspace_id: str = WORKSPACE, **fields: object) -> Filter:
    return Filter(workspace_ids=frozenset({workspace_id}), **fields)  # pyright: ignore[reportArgumentType]


def rank_of(candidates: Sequence[object], chunk_id: str) -> int | None:
    """One-based position of ``chunk_id``, or ``None`` when it is not there at all."""
    for position, candidate in enumerate(candidates, start=1):
        if getattr(getattr(candidate, "chunk", None), "id", None) == chunk_id:
            return position
    return None


__all__ = [
    "PROFILE_OVERRIDES",
    "SCOPE",
    "WORKSPACE",
    "build_corpus",
    "index",
    "query_filter",
    "rank_of",
    "retriever_over",
]
