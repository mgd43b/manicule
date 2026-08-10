"""Reciprocal rank fusion: ranks in, one ordering out, and no magnitudes anywhere."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from manicule.core.retrieval import Candidate
from manicule.retrieval.config import FusionConfig
from manicule.retrieval.fusion import RRFStage
from manicule.retrieval.trace import FusionReport, installed
from manicule.testing import assert_retrieval_stage_contract
from tests.retrieval.fakes import a_query
from tests.storage_helpers import make_chunk, make_document

if TYPE_CHECKING:
    from manicule.core.content import Chunk

DOCUMENT = make_document()


def chunk(position: int) -> Chunk:
    return make_chunk(DOCUMENT, position, f"passage {position}")


def scored(position: int, **scores: float) -> Candidate:
    effective = next(iter(scores.values())) if scores else 0.0
    return Candidate(chunk=chunk(position), score=effective, scores=dict(scores))


async def test_a_stage_contract_holds_for_fusion() -> None:
    """Uniformity is what lets two pipelines be compared by configuration."""
    await assert_retrieval_stage_contract(
        RRFStage(), a_query(), [scored(0, dense=0.9), scored(1, lexical=3.2)]
    )


async def test_cross_leg_agreement_beats_a_single_strong_opinion() -> None:
    """The property RRF is chosen for, spelled out in the arithmetic.

    A candidate first in one leg and absent from the other scores ``1/61``. One placed third
    and fourth in *both* scores ``1/63 + 1/64`` and outranks it. That looks like a bug and is
    the entire point: two independent methods agreeing is stronger evidence than one method
    being certain.
    """
    only_dense = scored(0, dense=1.0)
    both = scored(1, dense=0.5, lexical=0.5)
    others = [scored(position, dense=0.9 - position * 0.01) for position in range(2, 5)]
    # Place `both` third in the dense ladder and fourth in the lexical one.
    ladder = [only_dense, *others, both, scored(5, lexical=0.9), scored(6, lexical=0.8)]

    fused = await RRFStage().run(a_query(), ladder)

    ranking = [candidate.chunk.id for candidate in fused]
    assert ranking.index(both.chunk.id) < ranking.index(only_dense.chunk.id)


async def test_fusion_reads_the_legs_it_was_configured_with_and_no_other_key() -> None:
    """The indirection that lets a leg be replaced without touching this stage.

    The lexical store writes a ``bm25`` key describing the algorithm; the stage writes its own
    name over the top. Fusion must read the configured names, so swapping the lexical leg for a
    learned-sparse one is a configuration edit rather than a change here.
    """
    stage = RRFStage(config=FusionConfig(legs=("dense", "sparse")))
    candidates = [scored(0, dense=0.1, bm25=99.0), scored(1, dense=0.9, sparse=0.5)]

    fused = await stage.run(a_query(), candidates)

    # The second candidate appears in both configured legs; the first appears in one, and its
    # enormous `bm25` value contributes nothing at all.
    assert fused[0].chunk.id == candidates[1].chunk.id


async def test_a_candidate_no_leg_scored_sorts_last_and_is_not_dropped() -> None:
    """Fusion orders candidates; it does not filter them.

    A stage that quietly removed what it did not understand would make the pipeline
    order-dependent, and an order-dependent pipeline cannot be compared with another one.
    """
    orphan = scored(9)
    fused = await RRFStage().run(a_query(), [orphan, scored(0, dense=0.5)])

    assert [candidate.chunk.id for candidate in fused][-1] == orphan.chunk.id
    assert len(fused) == 2


async def test_a_short_leg_shortens_its_ladder_and_needs_no_padding() -> None:
    """Nothing imputes a worst rank to a candidate a leg never saw.

    With six lexical candidates against twenty dense ones, the tail of the fused ordering is
    decided by the dense leg alone — which is the correct answer to "only one leg had an
    opinion about these".
    """
    candidates = [scored(position, dense=1.0 - position * 0.01) for position in range(20)]
    candidates = [
        candidate.scored_by("lexical", 1.0 - index * 0.01) if index < 6 else candidate
        for index, candidate in enumerate(candidates)
    ]

    fused = await RRFStage().run(a_query(), candidates)

    tail = [candidate for candidate in fused if "lexical" not in candidate.scores]
    assert [candidate.scores["dense"] for candidate in tail] == sorted(
        (candidate.scores["dense"] for candidate in tail), reverse=True
    )


async def test_a_leg_that_returned_nothing_makes_the_run_incomparable() -> None:
    """A degraded run is part of that run's recorded identity.

    Without this, an evaluation harness cannot tell a metric that moved because the corpus is
    hard from one that moved because the lexical index threw. Both are legitimate outcomes of a
    query and only one is a legitimate input to a measurement.
    """
    with installed() as frame:
        await RRFStage().run(a_query(), [scored(0, dense=0.9)])

    assert frame.incomparable
    report = FusionReport.model_validate(frame.take_diagnostics())
    assert report.degraded
    assert report.per_leg == {"dense": 1, "lexical": 0}


async def test_two_runs_of_one_pipeline_fuse_identically() -> None:
    """Ties break on chunk id, so nothing is incomparable for want of a stable sort."""
    tied = [scored(position, dense=0.5, lexical=0.5) for position in range(5)]
    first = await RRFStage().run(a_query(), tied)
    second = await RRFStage().run(a_query(), list(reversed(tied)))

    assert [c.chunk.id for c in first] == [c.chunk.id for c in second]


def test_a_repeated_leg_is_refused() -> None:
    """One ladder counted twice weights that leg without saying so."""
    with pytest.raises(ValueError, match="same stage twice"):
        FusionConfig(legs=("dense", "dense"))


async def test_the_fused_score_is_bounded_by_two_over_sixty_one() -> None:
    """Why a ``[0, 1]`` floor on the fused score would empty every result set.

    With two legs and ``K = 60`` the maximum reachable score is ``2/61``, so a ``min_score`` of
    0.3 applied here discards every candidate in every profile and returns an empty list that
    looks exactly like a corpus with nothing in it.
    """
    best = scored(0, dense=1.0, lexical=1.0)
    fused = await RRFStage().run(a_query(), [best])

    assert fused[0].scores["rrf"] == pytest.approx(2 / 61)
    assert fused[0].scores["rrf"] < 0.15  # below the most permissive shipped floor
