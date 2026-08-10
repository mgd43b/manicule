"""How well-supported an answer is by the corpus.

Three things it is not, stated first because each is a way the number gets misread:

**Not a probability that the answer is correct.** Nothing in this pipeline is calibrated
against anything, and an uncalibrated score presented as a probability is the kind of claim
that gets believed.

**Not about the answer at all.** It is computed before generation, from the retrieval, and it
says how much supporting evidence was found and how strongly two independent methods of
finding it agreed. An answer can be wrong at high confidence — the evidence was there and the
model misread it — and that is not a bug in this number.

**Not comparable across configurations.** A score computed under one reranker and one computed
under none are different measurements, which is why the pipeline identity travels with it.

Only quantities with a defined scale are admissible. The fused RRF score is a rank artefact
bounded by ``2/61`` and means nothing absolute; BM25 is corpus-relative and unbounded; and
substring keyword coverage does no stemming and no IDF weighting, so a query for
``authenticate`` scores zero against a passage containing ``authentication`` — the precise
failure the lexical index's stemming tokenizer exists to avoid. Cross-leg agreement replaces
it, which is the same idea asked of the leg that already solved it.

**No fallback term.** When a component did not run, its weight is not redistributed and the
remaining weights are not renormalised: the reachable ceiling drops instead. Substituting the
retrieval average into the reranker's slot — the obvious-looking fix — makes retrieval count
for 0.70 instead of 0.40, so *turning the reranker off raises the reported confidence* for
identical retrieval, and the weaker pipeline claims more. Renormalising has the same effect by
a more respectable route.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

from manicule.core.retrieval import Confidence, ConfidenceBand, PipelineIdentity

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from manicule.core.retrieval import Candidate

SIMILARITY: Final = "similarity"
AGREEMENT: Final = "agreement"
BREADTH: Final = "breadth"
RERANK: Final = "rerank"

WEIGHTS: Final[Mapping[str, float]] = {
    SIMILARITY: 0.40,
    AGREEMENT: 0.15,
    BREADTH: 0.15,
    RERANK: 0.30,
}
"""What each admissible component is worth. They sum to 1.0 and are never renormalised."""

BREADTH_TARGET: Final = 3
"""Distinct documents at which support breadth is considered full.

Three, not because three is special, but because the term is meant to distinguish "one
document said so" from "several independent places said so", and beyond a handful the
distinction stops carrying information.
"""

BANDS: Final[tuple[tuple[float, ConfidenceBand], ...]] = (
    (0.75, ConfidenceBand.HIGH),
    (0.45, ConfidenceBand.MEDIUM),
    (0.20, ConfidenceBand.LOW),
)

NOTHING_RETRIEVED: Final = "retrieval ran and found no supporting passage"
BUDGET_CAPPED: Final = (
    "a leg stopped at its own budget rather than at the end of the corpus, so this retrieval "
    "is a floor rather than a result"
)


def band_for(score: float) -> ConfidenceBand:
    """Which band a score falls in."""
    for threshold, band in BANDS:
        if score >= threshold:
            return band
    return ConfidenceBand.NONE


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sigmoid(value: float) -> float:
    """Squash an unbounded reranker logit into ``[0, 1]``.

    Reranker scores are model-specific logits, comparable only within one model's rankings.
    Squashing makes them combinable with the other components' scales; it does **not** make two
    models' confidences comparable, which is what the pipeline identity is for.
    """
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponentiated = math.exp(value)
    return exponentiated / (1.0 + exponentiated)


def score_confidence(
    passages: Sequence[Candidate],
    *,
    identity: PipelineIdentity | None = None,
    legs: Sequence[str] = ("dense", "lexical"),
    degraded_legs: Sequence[str] = (),
    rerank_stage: str | None = None,
    exhausted_budget: bool = False,
) -> Confidence:
    """Score the retrieval that produced ``passages``.

    Args:
        passages: The candidates that reached the context. Confidence is about what will be
            shown, not about everything that was considered.
        identity: What produced the ranking. Travels with the score, because two scores from
            different pipelines are not two measurements of one thing.
        legs: The retrieval legs whose agreement is being measured.
        degraded_legs: Legs that contributed nothing. Their components are **suppressed**, not
            scored zero — reporting lower confidence because the lexical index threw is a
            statement about the system dressed up as a statement about the evidence.
        rerank_stage: The stage name a reranker scored under, or ``None`` if none ran.
        exhausted_budget: Whether a leg stopped at its own caps. Caps the band at ``medium``.

    Returns:
        A :class:`~manicule.core.retrieval.Confidence` carrying the band, every component that
        contributed, every component that was suppressed and why, and the ceiling those
        suppressions left.
    """
    pipeline = identity or PipelineIdentity()
    if not passages:
        return Confidence(
            score=0.0,
            band=ConfidenceBand.NONE,
            ceiling=1.0,
            reason=NOTHING_RETRIEVED,
            pipeline=pipeline,
        )

    healthy = [leg for leg in legs if leg not in degraded_legs]
    components: dict[str, float] = {}
    suppressed: dict[str, str] = {}

    dense_leg = legs[0] if legs else None
    if dense_leg is not None and dense_leg in degraded_legs:
        suppressed[SIMILARITY] = (
            f"the {dense_leg!r} leg contributed nothing, so no passage carries a similarity "
            f"this run could average"
        )
    else:
        similarities = [
            max(candidate.scores.get(dense_leg or "", 0.0), 0.0) for candidate in passages
        ]
        components[SIMILARITY] = _mean(similarities)

    if len(healthy) < len(legs) or len(legs) < 2:  # noqa: PLR2004 - agreement needs two legs
        suppressed[AGREEMENT] = (
            "fewer than two legs contributed, so no passage could carry both scores and a "
            "zero here would blame the corpus for a fault in the pipeline"
        )
    else:
        agreeing = sum(1 for candidate in passages if all(leg in candidate.scores for leg in legs))
        components[AGREEMENT] = agreeing / len(passages)

    documents = {candidate.chunk.document_id for candidate in passages}
    components[BREADTH] = min(len(documents) / BREADTH_TARGET, 1.0)

    if rerank_stage is None:
        suppressed[RERANK] = (
            "no reranker ran, so there is no verification step to report. Its weight is not "
            "redistributed: a pipeline that skipped the check must not be able to claim it"
        )
    else:
        logits = [
            _sigmoid(candidate.scores[rerank_stage])
            for candidate in passages
            if rerank_stage in candidate.scores
        ]
        if logits:
            components[RERANK] = _mean(logits)
        else:
            suppressed[RERANK] = (
                f"the {rerank_stage!r} stage ran but scored none of the passages that reached "
                f"the context"
            )

    score = sum(WEIGHTS[name] * value for name, value in components.items())
    ceiling = sum(WEIGHTS[name] for name in components)
    band = band_for(score)
    reason = ""
    if exhausted_budget and band is ConfidenceBand.HIGH:
        band = ConfidenceBand.MEDIUM
        reason = BUDGET_CAPPED
    elif exhausted_budget:
        reason = BUDGET_CAPPED

    return Confidence(
        score=score,
        band=band,
        components={name: WEIGHTS[name] * value for name, value in components.items()},
        suppressed=suppressed,
        ceiling=ceiling,
        reason=reason,
        pipeline=pipeline,
    )


__all__ = [
    "AGREEMENT",
    "BANDS",
    "BREADTH",
    "BREADTH_TARGET",
    "BUDGET_CAPPED",
    "NOTHING_RETRIEVED",
    "RERANK",
    "SIMILARITY",
    "WEIGHTS",
    "band_for",
    "score_confidence",
]
