"""The same fixture through the shipped embedder, because the argument rests on a measurement.

``tests/glossary/test_retrieval.py`` proves the behaviour deterministically, with a bag-of-words
stand-in. That is the right tool for a guard — it runs in milliseconds and cannot drift — but it
cannot settle the question this feature was justified by, which is what **BGE-M3** does with this
corpus. A stand-in embedder can be made to fail by choosing the fixture, and the accusation
would be fair.

So these run against the real weights and skip when they are not on the machine. The numbers in
the docstrings are what they produced on Apple Silicon through the MLX backend; the assertions
are the *claims*, stated loosely enough to survive a backend change and tightly enough to fail
if the effect goes away.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from manicule.core.retrieval import Query
from manicule.retrieval.expansion import ExpansionPolicy
from tests.embedding_support import FULL_MODEL, require_model, requires_mlx
from tests.glossary import corpus, system

if TYPE_CHECKING:
    from manicule.core.protocols import Embedder
    from manicule.storage.docstore import SqliteDocStore

LIMIT = 10


async def _embedder() -> Embedder:
    requires_mlx(FULL_MODEL)
    require_model(FULL_MODEL, mlx=True)
    from manicule.embedding.cards import read_card  # noqa: PLC0415 - an embeddings extra
    from manicule.embedding.runtimes.mlx_backend import MlxEmbedder  # noqa: PLC0415

    built = MlxEmbedder(read_card(FULL_MODEL))
    await built.setup()
    return built


def _ask(text: str) -> Query:
    return Query(text=text, limit=LIMIT, filter=system.query_filter())


async def test_the_real_embedder_buries_the_definition_and_the_glossary_recovers_it(
    store: SqliteDocStore,
) -> None:
    """The measurement the feature is justified by, end to end.

    Measured with BGE-M3 over the 61-chunk fixture:

    ==========================================  =========================  ==================
    Query                                       Definition's rank, before  after
    ==========================================  =========================  ==================
    ``What is NOW?``                            absent from a 10-passage   1
                                                context (15 of 61 overall)
    ``what is now?``                            absent (13 of 61)          1
    ``What is Now?``                            absent                     1
    ``What is N.O.W.?``                         6                          1
    ==========================================  =========================  ==================

    The definition's own cosine to ``What is NOW?`` is 0.4655 — below the 0.54 noise floor —
    because it is one line of a twenty-five entry page. That is why the recovery comes from the
    *lookup* rather than from the expanded embedding: the expansion alone ranks the glossary 8
    of 61, which would not have reached a ten-passage context either.
    """
    embedder = await _embedder()
    chunks = await system.build_corpus(store)
    entries = await store.glossary_entries(chunks[0].document_id)
    definition = next(entry.chunk_id for entry in entries if entry.acronym == corpus.ACRONYM)

    baseline = await system.retriever_over(store, embedder, chunks, glossary=False)
    retriever = await system.retriever_over(store, embedder, chunks)

    before = await baseline.retrieve(_ask(corpus.QUERY_ACRONYM))
    after = await retriever.retrieve(_ask(corpus.QUERY_ACRONYM))

    assert system.rank_of(before.context.passages, definition) is None, (
        "the fixture no longer reproduces the bug against the real embedder"
    )
    assert system.rank_of(after.context.passages, definition) == 1
    assert after.expansion is not None
    assert after.expansion.matches[0].entry.expansion == corpus.EXPANSION


async def test_the_real_embedder_leaves_an_ordinary_use_of_the_word_alone(
    store: SqliteDocStore,
) -> None:
    """Acceptance criterion 4, measured rather than argued.

    ``should I restart the daemon now …`` scored 0.4957 ``medium`` before and after, with an
    identical ranking. That is the number that says the feature can be left switched on.
    """
    embedder = await _embedder()
    chunks = await system.build_corpus(store)
    baseline = await system.retriever_over(store, embedder, chunks, glossary=False)
    retriever = await system.retriever_over(store, embedder, chunks)

    before = await baseline.retrieve(_ask(corpus.QUERY_ORDINARY_USE))
    after = await retriever.retrieve(_ask(corpus.QUERY_ORDINARY_USE))

    assert [passage.chunk.id for passage in after.context.passages] == [
        passage.chunk.id for passage in before.context.passages
    ]
    assert before.confidence is not None
    assert after.confidence is not None
    assert after.confidence.score == pytest.approx(before.confidence.score)


async def test_the_real_embedder_reports_a_calibrated_confidence_for_the_acronym_question(
    store: SqliteDocStore,
) -> None:
    """Acceptance criterion 2, and the honest limit of it.

    ``What is NOW?`` reports **0.1121 low** on this corpus, before and after — non-zero, and
    unchanged by the feature. It is unchanged by design: a promoted passage carries no leg
    score, so it can neither manufacture evidence nor destroy it.

    The number is low because the passage that defines the term is one line of a twenty-five
    entry page, and a chunk that is mostly about other terms *is* weak evidence. Making the
    number larger would mean letting a detection confidence stand in for a cosine, which is the
    substitution ``docs/retrieval.md`` §8 refuses everywhere else.
    """
    embedder = await _embedder()
    chunks = await system.build_corpus(store)
    baseline = await system.retriever_over(store, embedder, chunks, glossary=False)
    retriever = await system.retriever_over(store, embedder, chunks)

    before = await baseline.retrieve(_ask(corpus.QUERY_ACRONYM))
    after = await retriever.retrieve(_ask(corpus.QUERY_ACRONYM))

    assert before.confidence is not None
    assert after.confidence is not None
    assert after.confidence.score > 0.0
    assert after.confidence.score >= before.confidence.score


async def test_switching_the_feature_off_restores_the_measured_baseline(
    store: SqliteDocStore,
) -> None:
    """The disable switch, against the real embedder rather than only the stand-in."""
    embedder = await _embedder()
    chunks = await system.build_corpus(store)
    baseline = await system.retriever_over(store, embedder, chunks, glossary=False)
    disabled = await system.retriever_over(
        store, embedder, chunks, policy=ExpansionPolicy(enabled=False)
    )

    before = await baseline.retrieve(_ask(corpus.QUERY_ACRONYM))
    after = await disabled.retrieve(_ask(corpus.QUERY_ACRONYM))

    assert [passage.chunk.id for passage in after.context.passages] == [
        passage.chunk.id for passage in before.context.passages
    ]
