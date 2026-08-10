"""The cross-encoder: it raises, it truncates, and it names its model."""

from __future__ import annotations

import pytest

from manicule.core.protocols import Reranker
from manicule.core.retrieval import Candidate
from manicule.retrieval.rerank import CrossEncoderReranker
from manicule.retrieval.runtimes.cross_encoder import CrossEncoderScorer, RerankerUnavailableError
from manicule.retrieval.trace import RerankReport, installed
from manicule.testing import assert_retrieval_stage_contract
from tests.retrieval.fakes import ExplodingScorer, FixedScorer, ShortScorer, a_query, profiles
from tests.storage_helpers import make_chunk, make_document

DOCUMENT = make_document()


def fused(count: int) -> list[Candidate]:
    """Candidates carrying fusion's scores, which are on the order of 0.016."""
    return [
        Candidate(
            chunk=make_chunk(DOCUMENT, position, f"passage {position}"),
            score=0.016,
            scores={"rrf": 0.016},
        )
        for position in range(count)
    ]


def a_reranker(scorer: FixedScorer, **overrides: object) -> CrossEncoderReranker:
    return CrossEncoderReranker(scorer=scorer, profiles=profiles(**overrides))


async def test_it_is_structurally_a_stage() -> None:
    await assert_retrieval_stage_contract(
        a_reranker(FixedScorer({"passage 0": 2.0})), a_query(), fused(3)
    )


def test_it_satisfies_the_reranker_protocol() -> None:
    """A recorded result that cannot name its reranker cannot be reproduced."""
    stage = a_reranker(FixedScorer())
    assert isinstance(stage, Reranker)
    assert stage.model_id == "fake/cross-encoder"


async def test_it_returns_only_what_it_scored(store: object) -> None:
    """No mixed-scale tail.

    Concatenating an unreranked remainder onto the rescored head puts fusion's sums — on the
    order of 0.016 — in one list beside unbounded logits, so every comparison across the join
    is between numbers that differ by nearly two orders of magnitude.
    """
    del store
    stage = a_reranker(FixedScorer(), candidates=3, final_top_k=3)

    produced = await stage.run(a_query(limit=3), fused(10))

    assert len(produced) == 3
    assert all("rerank" in candidate.scores for candidate in produced)


async def test_a_failed_rerank_is_the_query_s_failure() -> None:
    """It raises; it does not return its input.

    A profile that says it reranks and produced an unreranked list has misreported which
    pipeline ran, and nothing downstream — including an evaluation harness — can see the
    difference.
    """
    with pytest.raises(RuntimeError, match="forward pass failed"):
        await a_reranker(ExplodingScorer()).run(a_query(), fused(3))


async def test_a_scorer_that_returns_the_wrong_number_of_scores_is_refused() -> None:
    """Pairs and scores are positional, so a mismatch ranks a passage by another's score."""
    with pytest.raises(ValueError, match="scored 2 of 3 pairs"):
        await a_reranker(ShortScorer(), candidates=3, final_top_k=3).run(a_query(limit=3), fused(3))


async def test_the_head_is_reordered_by_the_new_scores() -> None:
    """The fused list is a good candidate set and a mediocre final ordering.

    ``K = 60`` compresses within-leg ordering almost flat on purpose, so that appearing in both
    legs dominates appearing high in one. Producing a good *ordering* from that set is the job
    the cross-encoder is here for.
    """
    scorer = FixedScorer({"passage 0": -3.0, "passage 1": 5.0, "passage 2": 1.0})
    produced = await a_reranker(scorer, candidates=3, final_top_k=3).run(a_query(limit=3), fused(3))

    assert [candidate.chunk.text for candidate in produced] == [
        "passage 1",
        "passage 2",
        "passage 0",
    ]


async def test_it_scores_the_citable_text_and_not_the_breadcrumb() -> None:
    """``embed_text`` carries a heading breadcrumb that exists to make a passage findable.

    Judging it as though it were part of the answer scores the retrieval scaffolding rather
    than the passage.
    """
    scorer = FixedScorer()
    await a_reranker(scorer, candidates=1, final_top_k=1).run(a_query(limit=1), fused(1))

    (pairs,) = scorer.calls
    assert pairs == [("authentication", "passage 0")]


async def test_it_records_the_model_and_the_pairs() -> None:
    with installed() as frame:
        await a_reranker(FixedScorer(), candidates=2, final_top_k=2).run(a_query(limit=2), fused(5))
        report = RerankReport.model_validate(frame.take_diagnostics())

    assert report.model_id == "fake/cross-encoder"
    assert report.pairs == 2
    assert report.truncated_from == 5


async def test_an_empty_ranking_scores_nothing_and_returns_nothing() -> None:
    scorer = FixedScorer()
    assert await a_reranker(scorer).run(a_query(), []) == []
    assert scorer.calls == []


async def test_a_scorer_asked_before_its_weights_loaded_refuses() -> None:
    """Never a silent zero. A logit of 0 is indistinguishable from a genuine 'irrelevant'."""
    scorer = CrossEncoderScorer("BAAI/bge-reranker-v2-m3")
    with pytest.raises(RerankerUnavailableError, match="before its weights were loaded"):
        await scorer.score([("q", "p")])


async def test_a_scorer_with_no_pairs_needs_no_model() -> None:
    """The one call that is free, so a reranker in a pipeline with nothing to rank is free."""
    assert await CrossEncoderScorer("BAAI/bge-reranker-v2-m3").score([]) == []
