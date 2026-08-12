"""Confidence: a statement about the retrieval, with no fallback term anywhere."""

from __future__ import annotations

import pytest

from manicule.core.retrieval import Candidate, ConfidenceBand, PipelineIdentity
from manicule.retrieval.confidence import (
    AGREEMENT,
    NOISE_SIMILARITY,
    NOTHING_RESEMBLES,
    RERANK,
    SIMILARITY,
    WEIGHTS,
    rescale_similarity,
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

    assert confidence.ceiling == pytest.approx(0.70)
    assert confidence.score <= confidence.ceiling
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


def test_a_leg_that_matched_nothing_is_scored_not_excused() -> None:
    """The inversion this whole module was rewritten for.

    Suppression is for a leg that *failed*. A leg that ran and matched nothing has reported a
    fact about the query, and waiving its component rewards precisely the queries that deserve
    it least — nonsense matches no keywords, so under the old rule it paid no agreement penalty
    at all, while a real question that matched some paid one in full.
    """
    matched_nothing = score_confidence(
        [passage(FIRST, 0, dense=0.9), passage(SECOND, 1, dense=0.9)],
        rerank_stage=None,
    )
    agreed = score_confidence(
        [
            passage(FIRST, 0, dense=0.9, lexical=1.0),
            passage(SECOND, 1, dense=0.9, lexical=1.0),
        ],
        rerank_stage=None,
    )

    assert AGREEMENT not in matched_nothing.suppressed
    assert matched_nothing.components[AGREEMENT] == 0.0
    # Same ceiling: nothing about the *pipeline* differed between these two runs, so they are
    # measured on the same scale — which is what makes the comparison below meaningful at all.
    assert matched_nothing.ceiling == pytest.approx(agreed.ceiling)
    assert matched_nothing.score < agreed.score


def test_agreement_is_suppressed_only_when_the_pipeline_declares_one_leg() -> None:
    """Suppression tracks the shape of the pipeline, never what one query happened to match."""
    confidence = score_confidence(
        [passage(FIRST, 0, dense=0.9)], legs=("dense",), rerank_stage=None
    )

    assert AGREEMENT in confidence.suppressed
    assert AGREEMENT not in confidence.components


def test_a_passage_the_dense_leg_never_scored_is_absent_not_zero() -> None:
    """Not-measured and measured-zero are different claims, and averaging conflates them.

    A passage the lexical leg found and the dense leg never ranked carries no similarity. Under
    ``.get(leg, 0.0)`` it was averaged in as a zero, which asserts the dense leg looked and found
    it orthogonal to the query. It never looked.
    """
    both_scored = score_confidence(
        [passage(FIRST, 0, dense=0.75), passage(SECOND, 1, dense=0.75)], rerank_stage=None
    )
    one_unscored = score_confidence(
        [
            passage(FIRST, 0, dense=0.75),
            passage(SECOND, 1, dense=0.75),
            passage(THIRD, 2, lexical=7.0),
        ],
        rerank_stage=None,
    )

    assert one_unscored.components[SIMILARITY] == pytest.approx(both_scored.components[SIMILARITY])


def test_no_passage_carrying_a_dense_score_suppresses_similarity() -> None:
    """When nothing was measured there is nothing to average, and zero would be a claim."""
    confidence = score_confidence(
        [passage(FIRST, 0, lexical=3.0), passage(SECOND, 1, lexical=2.0)], rerank_stage=None
    )

    assert SIMILARITY in confidence.suppressed
    assert confidence.components.get(SIMILARITY) is None


def test_negative_similarities_clamp_to_zero() -> None:
    """A cosine below zero is not evidence against; it is simply no support."""
    confidence = score_confidence([passage(FIRST, 0, dense=-0.4, lexical=1.0)], rerank_stage=None)
    assert confidence.components[SIMILARITY] == 0.0


def test_scattering_across_documents_does_not_raise_confidence() -> None:
    """The term that used to run backwards, asserted in the direction it actually runs.

    Counting distinct documents was meant to read as corroboration and read as diffusion
    instead: noise scatters across a corpus and a good answer concentrates in one place, so the
    term paid the worse result. Spread is now not evidence in either direction — only how well
    the passages match, and whether the two legs agreed, are.
    """
    concentrated = score_confidence(
        [
            passage(FIRST, 0, dense=0.7, lexical=1.0),
            passage(FIRST, 1, dense=0.7, lexical=1.0),
            passage(FIRST, 2, dense=0.7, lexical=1.0),
        ],
        rerank_stage=None,
    )
    scattered = score_confidence(
        [
            passage(FIRST, 0, dense=0.7, lexical=1.0),
            passage(SECOND, 1, dense=0.7, lexical=1.0),
            passage(THIRD, 2, dense=0.7, lexical=1.0),
        ],
        rerank_stage=None,
    )

    assert scattered.score == pytest.approx(concentrated.score)


def test_noise_scores_below_a_weaker_but_real_match() -> None:
    """The reported defect, reduced to its arithmetic.

    Measured cosines: an unrelated query's passages sat around 0.45 and spread over four
    documents; a real question's sat around 0.60 in a single document. The old scoring put the
    first above the second. Nothing may do that again.
    """
    noise = score_confidence(
        [
            passage(FIRST, 0, dense=0.459),
            passage(SECOND, 1, dense=0.458),
            passage(THIRD, 2, dense=0.447),
        ],
        rerank_stage=None,
    )
    signal = score_confidence(
        [
            passage(FIRST, 0, dense=0.633),
            passage(FIRST, 1, dense=0.619),
            passage(FIRST, 2, dense=0.602),
        ],
        rerank_stage=None,
    )

    assert noise.score < signal.score
    assert noise.band is ConfidenceBand.NONE
    assert noise.reason == NOTHING_RESEMBLES


def test_confidence_is_monotonic_in_similarity() -> None:
    """A better-matched query is never reported as less confident than a worse one."""
    scores = [
        score_confidence([passage(FIRST, 0, dense=cosine, lexical=1.0)], rerank_stage=None).score
        for cosine in (0.30, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
    ]

    assert scores == sorted(scores)


def test_a_cosine_at_or_below_the_corpus_noise_level_contributes_nothing() -> None:
    """Further from the query than unrelated text is not a finer grade of relevance."""
    assert rescale_similarity(NOISE_SIMILARITY, noise=NOISE_SIMILARITY, strong=0.75) == 0.0
    assert rescale_similarity(0.20, noise=NOISE_SIMILARITY, strong=0.75) == 0.0
    assert rescale_similarity(0.75, noise=NOISE_SIMILARITY, strong=0.75) == 1.0
    assert rescale_similarity(0.95, noise=NOISE_SIMILARITY, strong=0.75) == 1.0


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
