"""A real corpus and a real dense-only pipeline, for the checks that must not use a stub.

The chance-level guard is only evidence if it runs against the code that will actually be
used. So these build the shipped :class:`~manicule.retrieval.retriever.Retriever` over the
shipped :class:`~manicule.retrieval.dense.DenseStage` against a migrated SQLite store, and
change exactly one thing between the two systems under comparison: the embedder.

**Dense-only, deliberately.** A hybrid pipeline retrieves through BM25 as well, so it would
find a document by its title however meaningless its vectors were — and would be right to pass
the probe, because the *system* retrieves. What must be catchable is a pipeline whose only
retrieval mechanism has no semantic content, and that is a dense leg on its own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from manicule.core.protocols import Embedder
from manicule.retrieval.assembly import ContextAssembler
from manicule.retrieval.cache import L1QueryCache
from manicule.retrieval.dense import DenseStage
from manicule.retrieval.profile import Profiles
from manicule.retrieval.retriever import Retriever
from manicule.retrieval.runner import PipelineRunner
from manicule.retrieval.tokens import ContextTokenCounter
from tests.evaluation.fakes import CosineVectorStore
from tests.storage_helpers import make_chunk, make_document

if TYPE_CHECKING:
    from manicule.core.content import Chunk
    from manicule.storage.docstore import SqliteDocStore

WORKSPACE = "default"
SCOPE = frozenset({WORKSPACE})

SUBJECTS = (
    "aurora ledger",
    "basalt gateway",
    "cinder scheduler",
    "delta warehouse",
    "ember indexer",
    "fathom router",
    "glacier archive",
    "harbour notifier",
    "isotope planner",
    "juniper cache",
    "krypton balancer",
    "lantern collector",
    "mistral compactor",
    "nimbus resolver",
    "obsidian broker",
    "pumice migrator",
    "quartz allocator",
    "ravine sequencer",
    "sierra transcoder",
    "tundra reconciler",
)
"""Two-word subjects, none sharing a word with another.

Distinctness is load-bearing in one direction only: a corpus where two documents are about the
same thing would make the known answer ambiguous, and a working system would then score misses
on questions that have two right answers.
"""

ASPECTS = ("configuration", "failure handling", "capacity limits")
"""Each subject gets one document per aspect, which is how the corpus reaches a size where
chance is small enough to measure against."""


PROFILE_OVERRIDES = {"candidates": 10, "final_top_k": 10, "min_score": 0.0}
"""A floor of zero, because these embedders are not calibrated to any similarity scale.

A ``min_score`` tuned for a real model would discard the meaningless embedder's candidates
before the probe ever saw them — which would make the probe pass for the wrong reason and, far
worse, would make it look like the guard fired when what actually happened is that nothing was
retrieved at all.
"""


async def build_corpus(store: SqliteDocStore) -> list[Chunk]:
    """Index one document per subject-and-aspect, each with a distinctive title.

    Returns the chunks in a stable order so a vector store can be filled from them.
    """
    chunks: list[Chunk] = []
    for subject in SUBJECTS:
        for aspect in ASPECTS:
            title = f"{subject} {aspect}"
            document = make_document(
                source="fixture",
                source_id=title.replace(" ", "-"),
                title=title,
                uri=f"file:///{title.replace(' ', '-')}.md",
                body=title.encode(),
            )
            await store.upsert_document(document)
            text = (
                f"{title}. This page documents the {aspect} of the {subject} component, "
                f"including how {subject} behaves when {aspect} changes."
            )
            chunk = make_chunk(document, 0, text, heading_path=("Reference",))
            await store.replace_chunks(document.id, [chunk])
            chunks.append(chunk)
    return chunks


async def dense_only_retriever(
    store: SqliteDocStore,
    embedder: Embedder,
    chunks: list[Chunk],
    *,
    cache: L1QueryCache | None = None,
) -> Retriever:
    """The shipped retriever with one stage: the shipped dense leg over ``embedder``.

    ``cache`` exists so that a retriever whose cache can hit can be built on purpose, which is
    what lets the adapter's refusal be demonstrated rather than asserted.
    """
    vectors = CosineVectorStore()
    await vectors.ensure_ready(embedder.fingerprint)
    await vectors.upsert(chunks, await embedder.embed([chunk.embed_text for chunk in chunks]))

    profiles = Profiles(PROFILE_OVERRIDES)
    stage = DenseStage(embedder=embedder, vectors=vectors, docstore=store, profiles=profiles)
    return Retriever(
        runner=PipelineRunner([stage], docstore=store),
        docstore=store,
        assembler=ContextAssembler(counter=ContextTokenCounter(), profiles=profiles),
        profiles=profiles,
        cache=cache,
        legs=(),
        embed_fingerprint=embedder.fingerprint.canonical(),
    )


__all__ = [
    "ASPECTS",
    "PROFILE_OVERRIDES",
    "SCOPE",
    "SUBJECTS",
    "WORKSPACE",
    "build_corpus",
    "dense_only_retriever",
]
