"""Confidence: a statement about the retrieval, with no fallback term anywhere."""

from __future__ import annotations

import pytest

from manicule.core.retrieval import Candidate, ConfidenceBand, PipelineIdentity
from manicule.retrieval.confidence import (
    AGREEMENT,
    BANDS,
    DEFINITION_CITED,
    NOISE_SIMILARITY,
    NOTHING_RESEMBLES,
    RERANK,
    SIMILARITY,
    STRONG_SIMILARITY,
    WEIGHTS,
    explain_confidence,
    rescale_similarity,
    score_confidence,
)
from tests.storage_helpers import make_chunk, make_document

FIRST = make_document(source_id="one")
SECOND = make_document(source_id="two")
THIRD = make_document(source_id="three")

NONSENSE_COSINES = (0.4588, 0.4576, 0.4472, 0.4456, 0.4439)
"""What the query ``zzzqqq unrelated nonsense xyzzy`` actually returned, over seven documents.

Kept as measured rather than rounded to a story: the whole point is that these sit just *above*
a raw reading of the noise level and just *below* the rescale's, which is why the defect was
invisible until the two populations were measured against each other."""


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


def test_scattered_noise_does_not_raise_confidence() -> None:
    """The distinction the retired support-breadth term could not make.

    Breadth counted distinct documents and so read *diffusion* as corroboration: noise scatters
    across a corpus and a good answer concentrates in one place, and the term paid the worse
    result. Spread across documents now raises confidence **only when each document independently
    clears the noise floor** — which is what corroboration means and what a count never captured.
    Passages below the floor contribute nothing however many documents they are spread over.

    These are the cosines an unrelated query actually returned, over seven documents.
    """
    concentrated = score_confidence(
        [passage(FIRST, index, dense=cosine) for index, cosine in enumerate(NONSENSE_COSINES)],
        legs=("dense",),
        rerank_stage=None,
    )
    scattered = score_confidence(
        [
            passage(document, index, dense=cosine)
            for index, (document, cosine) in enumerate(
                zip((FIRST, SECOND, THIRD, FIRST, SECOND), NONSENSE_COSINES, strict=True)
            )
        ],
        legs=("dense",),
        rerank_stage=None,
    )

    assert scattered.band is ConfidenceBand.NONE
    assert concentrated.band is ConfidenceBand.NONE
    assert scattered.score < BANDS[-1][0]


def test_spreading_real_evidence_across_documents_does_raise_confidence() -> None:
    """The other half of the same rule, so neither direction can drift unnoticed.

    Two documents that each independently answer the question *are* better support than one, and
    saying so is the job the breadth term was reaching for and getting backwards.
    """
    one = score_confidence([passage(FIRST, 0, dense=0.55)], legs=("dense",), rerank_stage=None)
    two = score_confidence(
        [passage(FIRST, 0, dense=0.55), passage(SECOND, 1, dense=0.55)],
        legs=("dense",),
        rerank_stage=None,
    )

    assert two.score > one.score


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


# --- the glossary corpus: one definition, one distractor, filler, one unsupported question ----
#
# Every cosine below was measured with the shipped embedder against a synthetic public glossary
# defining `NOW - Network Operations Workspace`, not invented to make the arithmetic work. The
# corpus is one definition, one semantically related distractor about scheduling ("...admitted
# to the queue now...") and eight irrelevant operational passages.

GLOSSARY_HIT = 0.6228
"""Cosine of the defining passage for the question "What is NOW?". Rank 1 of ten."""
GLOSSARY_FILLER = (0.4108, 0.3218, 0.3065, 0.2722)
"""The four passages behind it, which `final_top_k` puts in the context regardless."""
UNSUPPORTED = (0.3928, 0.3780, 0.3734, 0.3099, 0.3087)
"""The whole context for a question the corpus cannot answer."""


def _context(*cosines: float) -> list[Candidate]:
    """One passage per cosine, spread over distinct documents.

    Spread rather than concentrated because that is the shape the reported defect had: a correct
    passage at rank 1 and unrelated filler from elsewhere in the corpus behind it.
    """
    documents = (FIRST, SECOND, THIRD, FIRST, SECOND)
    return [
        passage(documents[index % len(documents)], index, dense=cosine)
        for index, cosine in enumerate(cosines)
    ]


def test_a_correct_rank_one_passage_is_not_reported_as_no_evidence() -> None:
    """The reported defect. A perfect retrieval scored 0.0 and the band ``none``.

    "What is NOW?" put the defining passage at rank 1 with cosine 0.623 — and the mean over the
    context, dragged down by four filler passages the pipeline is obliged to return, fell to
    0.387. That is *below* the noise floor, so the strongest possible dense evidence reported
    exactly no evidence.
    """
    confidence = score_confidence(
        _context(GLOSSARY_HIT, *GLOSSARY_FILLER), legs=("dense",), rerank_stage=None
    )

    assert confidence.score > 0.0
    assert confidence.band is not ConfidenceBand.NONE
    assert confidence.reason != NOTHING_RESEMBLES


def test_filler_behind_a_correct_hit_does_not_lower_confidence() -> None:
    """The mechanism, isolated: filler is not evidence against the answer in front of it.

    An average said otherwise, and the context is *guaranteed* to contain filler — the pipeline
    fills to ``final_top_k`` whether or not the corpus holds that many relevant passages, so a
    narrow question was penalised precisely for being narrow.
    """
    alone = score_confidence(_context(GLOSSARY_HIT), legs=("dense",), rerank_stage=None)
    padded = score_confidence(
        _context(GLOSSARY_HIT, *GLOSSARY_FILLER), legs=("dense",), rerank_stage=None
    )

    assert padded.score == pytest.approx(alone.score)


def test_an_unsupported_question_still_reports_no_answer() -> None:
    """The property won earlier and defended here: nothing may make noise look like evidence."""
    confidence = score_confidence(_context(*UNSUPPORTED), legs=("dense",), rerank_stage=None)

    assert confidence.score == 0.0
    assert confidence.band is ConfidenceBand.NONE
    assert confidence.reason == NOTHING_RESEMBLES


def test_independent_documents_supporting_the_same_answer_raise_confidence() -> None:
    """Monotone in added independent evidence, which is what "combined" has to mean."""
    one = score_confidence([passage(FIRST, 0, dense=0.60)], legs=("dense",), rerank_stage=None)
    two = score_confidence(
        [passage(FIRST, 0, dense=0.60), passage(SECOND, 1, dense=0.60)],
        legs=("dense",),
        rerank_stage=None,
    )
    three = score_confidence(
        [
            passage(FIRST, 0, dense=0.60),
            passage(SECOND, 1, dense=0.60),
            passage(THIRD, 2, dense=0.60),
        ],
        legs=("dense",),
        rerank_stage=None,
    )

    assert one.score < two.score < three.score


def test_duplicate_evidence_from_one_document_does_not_multiply_confidence() -> None:
    """Ten chunks of one page are one observation, not ten.

    Without this, a single well-matched document manufactures certainty by being chunked
    finely — a property of the ingest configuration reported as a property of the evidence.
    """
    single = score_confidence([passage(FIRST, 0, dense=0.60)], legs=("dense",), rerank_stage=None)
    chunked = score_confidence(
        [passage(FIRST, index, dense=0.60) for index in range(10)],
        legs=("dense",),
        rerank_stage=None,
    )

    assert chunked.score == pytest.approx(single.score)


def test_one_strong_passage_alone_is_not_treated_as_proof() -> None:
    """Rank 1 is evidence, not certainty.

    A perfectly-matched passage takes the *dense* component in full, which is honest — that is
    what the component measures. What stops top-1 becoming proof is that dense evidence is 0.55
    of the number: the lexical leg has to have found it too, and a reranker has to have agreed,
    before anything can reach ``high``. So a lone strong passage in a full two-leg pipeline
    lands at ``medium`` and cannot climb further on its own.
    """
    confidence = score_confidence([passage(FIRST, 0, dense=STRONG_SIMILARITY)], rerank_stage=None)

    assert confidence.components[SIMILARITY] == pytest.approx(WEIGHTS[SIMILARITY])
    assert confidence.band is not ConfidenceBand.HIGH
    assert confidence.score < BANDS[0][0]


def test_corroboration_is_measured_over_the_passages_that_carry_evidence() -> None:
    """Filler must not dilute agreement either, for the same reason it must not dilute evidence.

    A query with one strong, doubly-confirmed answer used to score 1/5 for perfect
    corroboration, because the denominator counted four passages nobody claimed were relevant.

    Corroboration is then **scaled by the evidence**, because you cannot corroborate more than
    you have: the component is the corroborated fraction times the evidence level, so a fully
    corroborated strong answer approaches the full weight and a fully corroborated *weak* one
    does not.
    """
    corroborated = score_confidence(
        [
            passage(FIRST, 0, dense=GLOSSARY_HIT, lexical=9.0),
            *(
                passage(SECOND, index + 1, dense=cosine)
                for index, cosine in enumerate(GLOSSARY_FILLER)
            ),
        ],
        rerank_stage=None,
    )
    uncorroborated = score_confidence(
        [
            passage(FIRST, 0, dense=GLOSSARY_HIT),
            *(
                passage(SECOND, index + 1, dense=cosine)
                for index, cosine in enumerate(GLOSSARY_FILLER)
            ),
        ],
        rerank_stage=None,
    )

    evidence = corroborated.components[SIMILARITY] / WEIGHTS[SIMILARITY]
    assert corroborated.components[AGREEMENT] == pytest.approx(WEIGHTS[AGREEMENT] * evidence)
    assert uncorroborated.components[AGREEMENT] == 0.0
    assert corroborated.score > uncorroborated.score


def test_corroborating_weak_evidence_pays_less_than_corroborating_strong_evidence() -> None:
    """You cannot corroborate more than you have.

    Counting alone handed a nonsense query the full agreement weight: exactly one passage cleared
    the floor, barely, both legs happened to touch it, and 1/1 paid out in full — 0.15 of a number
    whose whole job in that case is to say the corpus holds nothing.
    """
    weak = score_confidence(
        [passage(FIRST, 0, dense=NOISE_SIMILARITY + 0.01, lexical=9.0)], rerank_stage=None
    )
    strong = score_confidence(
        [passage(FIRST, 0, dense=STRONG_SIMILARITY, lexical=9.0)], rerank_stage=None
    )

    assert weak.components[AGREEMENT] < strong.components[AGREEMENT]
    assert weak.band is ConfidenceBand.NONE


def test_the_thresholds_remain_configurable() -> None:
    """A corpus whose noise sits elsewhere must be able to say so without editing this module."""
    passages = [passage(FIRST, 0, dense=0.50)]

    default = score_confidence(passages, legs=("dense",), rerank_stage=None)
    shifted = score_confidence(
        passages,
        legs=("dense",),
        rerank_stage=None,
        noise_similarity=0.20,
        strong_similarity=0.40,
    )

    assert default.score < shifted.score


# --- the diagnostic ---------------------------------------------------------------------------


def test_the_diagnostic_reports_every_input_to_the_score() -> None:
    """ "The right passage was rank 1 and confidence said none" must be answerable in one call."""
    diagnostics = explain_confidence(
        _context(GLOSSARY_HIT, *GLOSSARY_FILLER), legs=("dense",), rerank_stage=None
    )

    assert diagnostics.noise_similarity == NOISE_SIMILARITY
    assert diagnostics.strong_similarity == STRONG_SIMILARITY
    assert diagnostics.weights == dict(WEIGHTS)
    assert diagnostics.bands == tuple((value, band.value) for value, band in BANDS)
    assert diagnostics.components
    assert len(diagnostics.passages) == 1 + len(GLOSSARY_FILLER)
    assert diagnostics.evidence_documents == 1
    # The passage that carried the answer is the one counted, and the filler is visibly not.
    counted = [item for item in diagnostics.passages if item.counted]
    assert len(counted) == 1
    assert counted[0].raw_similarity == pytest.approx(GLOSSARY_HIT)
    assert counted[0].evidence > 0.0


def test_the_diagnostic_reports_a_cited_definition_without_giving_it_a_weight() -> None:
    """Requirement: the diagnostics explain every component, including the one that is not one.

    ``explicit_definition`` appears as a field of its own and is deliberately **absent** from
    both ``components`` and ``suppressed``. A component has a weight and contributes; a suppressed
    component has a weight and was prevented from contributing, which is why it lowers the
    ceiling. This has no weight at all, and listing it beside things that do would invite exactly
    the reading it exists to prevent — that a definition is worth some number of points.

    The weighted arithmetic is asserted **identical** to the same run without the flag, so a
    diagnostic that quietly started scoring it would fail here rather than in a band somewhere.
    """
    passages = _context(NOISE_SIMILARITY - 0.05, NOISE_SIMILARITY - 0.10)

    plain = explain_confidence(passages, legs=("dense",), rerank_stage=None)
    cited = explain_confidence(
        passages, legs=("dense",), rerank_stage=None, explicit_definition=True
    )

    assert not plain.explicit_definition
    assert plain.reason == NOTHING_RESEMBLES
    assert cited.explicit_definition
    assert cited.reason == DEFINITION_CITED

    assert "explicit_definition" not in cited.components
    assert "explicit_definition" not in cited.suppressed
    assert cited.components == plain.components
    assert cited.score == plain.score
    assert cited.band is plain.band
    assert cited.ceiling == plain.ceiling
    assert cited.weights == plain.weights


def test_the_diagnostic_agrees_with_the_score_it_explains() -> None:
    """It runs the scorer rather than reimplementing it; a second implementation would drift."""
    passages = _context(GLOSSARY_HIT, *GLOSSARY_FILLER)

    scored = score_confidence(passages, legs=("dense",), rerank_stage=None)
    explained = explain_confidence(passages, legs=("dense",), rerank_stage=None)

    assert explained.score == scored.score
    assert explained.band is scored.band
    assert explained.ceiling == scored.ceiling
    assert explained.components == scored.components


def test_the_diagnostic_never_carries_passage_text() -> None:
    """Diagnostics travel into logs, bug reports and screenshots; corpus text must not.

    Asserted over the serialised form rather than the attributes, because the risk is what gets
    written out — a field added later that happens to hold text would pass an attribute check
    and still leak.
    """
    body = "CORPUS-TEXT-THAT-MUST-NOT-APPEAR-IN-A-DIAGNOSTIC"
    chunk = make_chunk(FIRST, 0, body)
    candidate = Candidate(chunk=chunk, score=1.0, scores={"dense": GLOSSARY_HIT})

    serialised = explain_confidence(
        [candidate], legs=("dense",), rerank_stage=None
    ).model_dump_json()

    assert body not in serialised
    assert chunk.id in serialised, "the chunk id is how an entitled reader finds the passage"


def test_the_diagnostic_shows_duplicate_evidence_being_refused() -> None:
    """Ten chunks of one document appear, and exactly one of them is counted."""
    diagnostics = explain_confidence(
        [passage(FIRST, index, dense=0.60) for index in range(10)],
        legs=("dense",),
        rerank_stage=None,
    )

    assert len(diagnostics.passages) == 10
    assert sum(1 for item in diagnostics.passages if item.counted) == 1
    assert diagnostics.evidence_documents == 1


NONSENSE_STRONGEST = 0.5194
"""The highest cosine the unrelated query ``zzzqqq unrelated nonsense xyzzy`` reached.

On a single passage, over this project's own documentation — and both retrieval legs found it.
It is the reason the noise floor is 0.52 rather than a rounder number below it: at 0.45 or 0.50
this passage counted as evidence, and because two legs had touched it the agreement term paid
out as well, carrying a query about nothing into the ``low`` band.
"""


def test_the_strongest_measured_noise_passage_is_not_evidence() -> None:
    """The floor is where it is because of this number, so this number is what pins it.

    A floor calibrated on context *means* looked safe at 0.45. Per passage the populations sit
    far closer, and moving to a per-passage statistic without re-measuring the constant is what
    let a nonsense query reach ``low``.
    """
    confidence = score_confidence(
        [passage(FIRST, 0, dense=NONSENSE_STRONGEST, lexical=8.0)], rerank_stage=None
    )

    assert confidence.components[SIMILARITY] == 0.0
    assert confidence.score == 0.0
    assert confidence.band is ConfidenceBand.NONE


WEAKEST_ANSWERABLE = 0.5598
"""The weakest top-1 cosine any answerable question produced across the measured corpora."""


def test_the_weakest_answerable_passage_is_still_evidence() -> None:
    """The other side of the same boundary, so the floor cannot drift upward unnoticed either.

    The weakest *top* passage any answerable question produced across the measured corpora. A
    floor above this would start reporting real answers as no answer, which is the failure the
    previous floor had in the other direction. Together with the test above these two numbers
    bracket the floor: it must sit between them, and it sits in the middle.
    """
    confidence = score_confidence(
        [passage(FIRST, 0, dense=WEAKEST_ANSWERABLE, lexical=8.0)], rerank_stage=None
    )

    assert confidence.components[SIMILARITY] > 0.0
    assert confidence.band is not ConfidenceBand.NONE
