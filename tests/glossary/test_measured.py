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

from manicule.core.retrieval import ConfidenceBand, Query
from manicule.retrieval.confidence import DEFINITION_CITED, NOTHING_RESEMBLES
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

    Measured with BGE-M3 over the 62-chunk fixture:

    ==========================================  =========================  ==================
    Query                                       Definition's rank, before  after
    ==========================================  =========================  ==================
    ``What is NOW?``                            absent from a 10-passage   1
                                                context
    ``what is now?``                            absent                     1
    ``What is Now?``                            absent                     1
    ``What is N.O.W.?``                         7                          1
    ==========================================  =========================  ==================

    The dotted spelling ranked **6** while the corpus was 61 chunks and ranks 7 now that a
    second glossary page has been added; nothing else in the table moved. The overall ranks the
    first two rows used to carry are gone rather than updated: they were produced by a
    measurement this test does not perform, they did not reproduce when it was attempted, and a
    number nobody can re-derive is worse than no number.

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

    ``What is NOW?`` reports **0.2117 low** on this corpus, before and after — non-zero, and
    unchanged by the feature. It is unchanged by design: a promoted passage carries no leg
    score, so it can neither manufacture evidence nor destroy it.

    It read **0.1121 low** until the glossary supplement was added, and the move is the
    supplement rather than any change to scoring: a second page that is itself a glossary is
    genuinely somewhat on topic for "what is X", so the dense leg finds real evidence it did not
    have before. Both figures are the same band, and re-measuring the 61-chunk corpus still
    produces 0.1121 exactly — which is how the cause was established rather than assumed.

    The number is low because the passage that defines the term is one line of a twenty-five
    entry page, and a chunk that is mostly about other terms *is* weak evidence. Making the
    number larger would mean letting a detection confidence stand in for a cosine, which is the
    substitution ``docs/retrieval.md`` §8 refuses everywhere else — and which the
    ``explicit_definition`` classification exists specifically to avoid needing.
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


async def test_the_real_embedder_stops_calling_a_cited_definition_nothing_resembling(
    store: SqliteDocStore,
) -> None:
    """The contradiction, against real weights rather than a stand-in.

    ``What is Now?`` is the spelling that lands in the ``none`` band on this corpus — 0.0000,
    because the defining chunk is one line of a twenty-five entry page and nothing else the
    query reaches clears the noise floor. That is the exact state in which the baseline says
    "nothing in this corpus resembles your question" while the definition of the queried term
    sits at rank 1 of what it is about to show.

    The score is asserted **equal**, not merely non-decreasing. A test that allowed it to rise
    would pass for an unconditional boost, and the whole argument of this change is that no
    number moved.
    """
    embedder = await _embedder()
    chunks = await system.build_corpus(store)
    definition = (await system.entries_by_acronym(store, chunks))[corpus.ACRONYM].chunk_id

    baseline = await system.retriever_over(store, embedder, chunks, glossary=False)
    retriever = await system.retriever_over(store, embedder, chunks)

    before = await baseline.retrieve(_ask(corpus.QUERY_MIXED_CASE))
    after = await retriever.retrieve(_ask(corpus.QUERY_MIXED_CASE))

    assert before.confidence is not None
    assert after.confidence is not None
    assert before.confidence.band is ConfidenceBand.NONE
    assert before.confidence.reason == NOTHING_RESEMBLES, (
        "the real embedder no longer reproduces the contradiction on this fixture"
    )

    assert system.rank_of(after.context.passages, definition) == 1
    assert after.confidence.explicit_definition
    assert after.confidence.reason == DEFINITION_CITED
    assert after.confidence.score == pytest.approx(before.confidence.score)
    assert after.confidence.band is before.confidence.band


async def test_the_real_embedder_recovers_a_definition_that_carried_a_description(
    store: SqliteDocStore,
) -> None:
    """The motivating example against real weights: ``What is NOVA?``.

    Before extraction learned where a description begins, the line was thirteen words on the
    right of the dash and no entry existed — so there was nothing to promote, nothing to expand
    with, and nothing to classify. What is asserted is the trimmed expansion rather than the
    rank alone: an entry carrying the whole right-hand side would promote the same passage and
    would then splice nine words of prose into the second query.
    """
    embedder = await _embedder()
    chunks = await system.build_corpus(store)
    retriever = await system.retriever_over(store, embedder, chunks)

    result = await retriever.retrieve(_ask(corpus.QUERY_DESCRIBED))

    assert result.expansion is not None
    assert result.expansion.matches, "no entry fired, so extraction did not produce one"
    match = result.expansion.matches[0]
    assert match.entry.expansion == corpus.DESCRIBED_EXPANSION
    assert system.rank_of(result.context.passages, match.entry.chunk_id) == 1
    assert result.confidence is not None
    assert result.confidence.explicit_definition


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
