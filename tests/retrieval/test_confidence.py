"""Confidence: a statement about the retrieval, with no fallback term anywhere."""

from __future__ import annotations

from manicule.core.retrieval import Candidate, ConfidenceBand, PipelineIdentity
from manicule.retrieval.confidence import (
    AGREEMENT,
    RERANK,
    SIMILARITY,
    WEIGHTS,
    score_confidence,
)
from tests.storage_helpers import make_chunk, make_document

FIRST = make_document(source_id="one")
SECOND = make_document(source_id="two")
THIRD = make_document(source_id="three")


def passage(document: object, position: int, **scores: float) -> Candidate:
    chunk = make_chunk(document, position, f"passage {position}")  # pyright: ignore[reportArgumentType]
    return Candidate(chunk=chunk, score=1.0, scores=dict(scores))


def test_the_weights_sum_to_one_and_are_never_renormalised() -> None:
    assert sum(WEIGHTS.values()) == 1.0


def test_a_pipeline_with_no_reranker_cannot_reach_high() -> None:
    """Intended behaviour, not an artefact.

    ``fast`` is the profile that skips the verification step; it must not be able to claim it
    verified. Arithmetically its ceiling is 0.70, which is below the ``high`` band, and that is
    the most concrete difference between the profiles a user ever sees.
    """
    passages = [
        passage(FIRST, 0, dense=1.0, lexical=1.0),
        passage(SECOND, 1, dense=1.0, lexical=1.0),
        passage(THIRD, 2, dense=1.0, lexical=1.0),
    ]

    confidence = score_confidence(passages, rerank_stage=None)

    assert confidence.ceiling == 0.70
    assert confidence.score <= 0.70
    assert confidence.band is not ConfidenceBand.HIGH
    assert RERANK in confidence.suppressed


def test_turning_the_reranker_off_never_raises_the_reported_confidence() -> None:
    """The bug this rule exists to avoid, stated as a comparison.

    Substituting the retrieval average into the reranker's slot when none ran makes retrieval
    count for 0.70 instead of 0.40 — so the weaker pipeline claims more, for identical
    retrieval. Renormalising the remaining weights has the same effect by a more respectable
    route.
    """
    passages = [
        passage(FIRST, 0, dense=1.0, lexical=1.0, rerank=10.0),
        passage(SECOND, 1, dense=1.0, lexical=1.0, rerank=10.0),
        passage(THIRD, 2, dense=1.0, lexical=1.0, rerank=10.0),
    ]

    with_reranker = score_confidence(passages, rerank_stage="rerank")
    without = score_confidence(passages, rerank_stage=None)

    assert without.score < with_reranker.score


def test_a_degraded_leg_suppresses_agreement_rather_than_scoring_it_zero() -> None:
    """Confidence never blames the corpus for a fault in the pipeline.

    If the lexical leg returned nothing, no passage can carry both scores — so an agreement
    term computed as zero would report lower confidence *because the lexical index threw*,
    which is a statement about the system dressed up as a statement about the evidence.
    """
    passages = [passage(FIRST, 0, dense=0.9), passage(SECOND, 1, dense=0.9)]

    degraded = score_confidence(passages, degraded_legs=["lexical"], rerank_stage=None)
    healthy = score_confidence(
        [
            passage(FIRST, 0, dense=0.9, lexical=1.0),
            passage(SECOND, 1, dense=0.9, lexical=1.0),
        ],
        rerank_stage=None,
    )

    assert AGREEMENT in degraded.suppressed
    assert AGREEMENT not in degraded.components
    assert degraded.ceiling < healthy.ceiling
    assert degraded.score <= healthy.score


def test_a_degraded_dense_leg_suppresses_similarity_by_the_same_rule() -> None:
    """The rule generalises, and it has to: the reasoning is about the *cause*, not the term.

    A similarity term computed as zero because the dense leg contributed nothing reports weak
    evidence for a pipeline fault, which is exactly what the agreement rule forbids.
    """
    passages = [passage(FIRST, 0, lexical=3.0), passage(SECOND, 1, lexical=2.0)]

    confidence = score_confidence(passages, degraded_legs=["dense"], rerank_stage=None)

    assert SIMILARITY in confidence.suppressed
    assert confidence.components.get(SIMILARITY) is None


def test_negative_similarities_clamp_to_zero() -> None:
    """A cosine below zero is not evidence against; it is simply no support."""
    confidence = score_confidence([passage(FIRST, 0, dense=-0.4, lexical=1.0)], rerank_stage=None)
    assert confidence.components[SIMILARITY] == 0.0


def test_support_breadth_rewards_independent_documents() -> None:
    """One document saying so and three saying so are different claims."""
    one = score_confidence([passage(FIRST, 0, dense=1.0, lexical=1.0)], rerank_stage=None)
    three = score_confidence(
        [
            passage(FIRST, 0, dense=1.0, lexical=1.0),
            passage(SECOND, 1, dense=1.0, lexical=1.0),
            passage(THIRD, 2, dense=1.0, lexical=1.0),
        ],
        rerank_stage=None,
    )
    assert three.score > one.score


def test_nothing_retrieved_is_the_none_band_with_a_reason() -> None:
    """Distinct from *absent*: "we did not look" and "we looked and there is nothing" are
    different claims, and a single zero conflates them."""
    confidence = score_confidence([])
    assert confidence.band is ConfidenceBand.NONE
    assert confidence.reason
    assert confidence.score == 0.0


def test_a_leg_stopped_by_its_budget_caps_the_band_at_medium() -> None:
    """The retrieval is known to be a floor rather than a result."""
    passages = [
        passage(FIRST, 0, dense=1.0, lexical=1.0, rerank=10.0),
        passage(SECOND, 1, dense=1.0, lexical=1.0, rerank=10.0),
        passage(THIRD, 2, dense=1.0, lexical=1.0, rerank=10.0),
    ]

    uncapped = score_confidence(passages, rerank_stage="rerank")
    capped = score_confidence(passages, rerank_stage="rerank", exhausted_budget=True)

    assert uncapped.band is ConfidenceBand.HIGH
    assert capped.band is ConfidenceBand.MEDIUM
    assert capped.reason


def test_the_scalar_is_never_reported_alone() -> None:
    """A number that cannot say why it is 0.62 is a number nobody can act on, and one that
    cannot say what produced it is one nobody can compare."""
    identity = PipelineIdentity(stages=("dense", "lexical", "rrf"), rrf_k=60)
    confidence = score_confidence(
        [passage(FIRST, 0, dense=0.8, lexical=1.0)], identity=identity, rerank_stage=None
    )

    assert confidence.components
    assert confidence.pipeline == identity


def test_the_fused_score_and_bm25_are_not_admissible() -> None:
    """A rank artefact bounded by ``2/61`` and a corpus-relative unbounded number have no
    absolute meaning to combine with anything."""
    with_noise = score_confidence(
        [passage(FIRST, 0, dense=0.5, lexical=1.0, rrf=0.03, bm25=97.0)], rerank_stage=None
    )
    without = score_confidence([passage(FIRST, 0, dense=0.5, lexical=1.0)], rerank_stage=None)

    assert with_noise.score == without.score


def test_a_pipeline_that_declares_no_legs_suppresses_similarity_rather_than_zeroing_it() -> None:
    """The same rule once more, for the shape a pipeline with no fusion produces.

    Nothing here has a similarity to average, so a zero would report weak evidence for what is
    a property of the pipeline rather than of the corpus.
    """
    confidence = score_confidence([passage(FIRST, 0, rerank=4.0)], legs=(), rerank_stage="rerank")

    assert SIMILARITY in confidence.suppressed
    assert SIMILARITY not in confidence.components
    assert confidence.ceiling < 1.0
