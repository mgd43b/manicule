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

**Cosine is rescaled against the corpus's noise level, because raw cosine is not centred on
zero.** A query with no relationship to anything indexed still returns its nearest neighbours,
and those neighbours are not far away in absolute terms: measured over a 604-chunk corpus,
unrelated questions topped out at 0.467 while real ones started at 0.560
(``docs/retrieval.md`` §8.4). Read raw, those two are a tenth of a point apart on a scale whose
bands are cut at 0.20 and 0.45, so noise and signal land in the same band. Subtracting the
measured noise level and dividing by the distance to a strong match is what makes "nothing in
this corpus resembles your question" expressible at all — the same move the evaluation harness
makes when it reports a hit rate against a chance rate rather than bare.

**There is no support-breadth term, and removing it was a measured correction rather than a
simplification.** Counting distinct documents was meant to separate "one document said so" from
"several independent places said so". It cannot do that, because noise scatters across a corpus
and a good answer concentrates in it — so the term ran *backwards*. On the pair that exposed
this, a nonsense query reached four documents and took the full 0.15 while a correctly-focused
answer reached one and took 0.05: a tenth of a point handed to the worse result, which was most
of the reason the worse result outranked it. Telling corroboration from diffusion requires
knowing whether the documents *agree*, which is an entailment check and not a count.

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
RERANK: Final = "rerank"

WEIGHTS: Final[Mapping[str, float]] = {
    SIMILARITY: 0.55,
    AGREEMENT: 0.15,
    RERANK: 0.30,
}
"""What each admissible component is worth. They sum to 1.0 and are never renormalised.

Similarity carries the weight support breadth used to hold, rather than that weight being
dropped or spread. Breadth was removed at design time because it measured the wrong direction
(see the module docstring); its 0.15 goes to the one component that survived contact with a
measurement. Retiring a component and *renormalising after a component is suppressed at run
time* are different acts — the second is forbidden below and remains so.

Keeping the total at 1.0 preserves the property the profiles were built around: a pipeline
with no reranker reaches at most 0.70, so ``fast`` still cannot claim a verification step it
skipped.
"""

NOISE_SIMILARITY: Final = 0.45
"""Cosine at which this embedder's output stops carrying information about the query.

**Measured, not inherited.** Over manicule's own documentation — 13 documents, 604 chunks,
BGE-M3 — 22 questions drawn from subjects the corpus does not cover averaged 0.353 to 0.457
across the passages that reached a context; 16 questions the corpus does answer averaged 0.531
to 0.641. 0.45 sits just under the top of the noise range, so an unanswerable question scores
essentially zero while the weakest real question keeps a clear margin.

Both constants are calibrated against the **mean over the passages in the context**, which is
the quantity :func:`score_confidence` actually rescales — not against a top-1 cosine, which runs
roughly 0.08 higher and would put the whole scale out by that much.

This is a property of **the embedder and the corpus**, not a universal constant: BGE-M3's
similarities are not centred on zero for unrelated text, and a different model or a much more
topically concentrated corpus puts the noise level somewhere else. Re-measure it the way it was
measured — ask questions the corpus demonstrably cannot answer and read where their similarities
land — rather than adjusting it until an example looks right.
"""

STRONG_SIMILARITY: Final = 0.65
"""Mean context cosine at which a retrieval is treated as fully on-topic.

The top of the measured range rather than 1.0. Cosine 1.0 against a chunk means the query *is*
that chunk, which no question is; and a *mean* over several passages is pulled below even the
best passage by the ones beneath it. The best on-corpus mean observed was 0.641, so scaling to
1.0 — or to the 0.724 best *top-1*, which is a different statistic — would cap a flawless
retrieval at about half the component and report well-answered questions as weakly supported.
"""

BANDS: Final[tuple[tuple[float, ConfidenceBand], ...]] = (
    (0.75, ConfidenceBand.HIGH),
    (0.45, ConfidenceBand.MEDIUM),
    (0.10, ConfidenceBand.LOW),
)
"""Where the bands are cut on the rescaled scale.

The ``none`` boundary moved from 0.20 to 0.10, and the reason is the measurement rather than
taste. Once similarity is rescaled against the noise level the two populations separate with a
wide empty gap between them: over 604 chunks, unanswerable questions scored at most 0.032 and
real ones at least 0.162, across all three profiles. A boundary at 0.20 sat *inside* the real
population, so the deepest profile reported ``none`` — "nothing here resembles your question" —
for a question the corpus answers. That is the original defect wearing the other mask. 0.10 sits
in the gap, where a boundary belongs: every negative measured falls below it and every positive
above it.

``high`` and ``medium`` are unchanged, so a pipeline with no reranker still tops out below
``high``.
"""

NOTHING_RETRIEVED: Final = "retrieval ran and found no supporting passage"
BUDGET_CAPPED: Final = (
    "a leg stopped at its own budget rather than at the end of the corpus, so this retrieval "
    "is a floor rather than a result"
)
NOTHING_RESEMBLES: Final = (
    "every passage retrieved sits at or below the level this corpus returns for a question it "
    "has no answer to, so these are its nearest passages rather than relevant ones"
)
UNREACHABLE_BAND: Final = (
    "this pipeline's suppressed components put a higher band out of reach regardless of the "
    "evidence, so the band reports the ceiling rather than the corpus"
)


def band_for(score: float) -> ConfidenceBand:
    """Which band a score falls in.

    Deliberately **not** normalised by the run's ceiling. Dividing a score through by what the
    arithmetic permitted would let a pipeline with no reranker reach ``high``, which is exactly
    the claim ``fast`` must not be able to make: the profile that skips the verification step
    must not be able to report that it verified. The ceiling is used instead by
    :func:`reachable_band`, to say when a band was *unreachable* rather than unearned.
    """
    for threshold, band in BANDS:
        if score >= threshold:
            return band
    return ConfidenceBand.NONE


def reachable_band(ceiling: float) -> ConfidenceBand:
    """The best band this run's arithmetic allowed, whatever the corpus held.

    Read beside the band actually reported. "The evidence was mediocre" and "this pipeline
    cannot report better than mediocre" are different statements that produce the same number,
    and a reader deciding whether to trust an answer needs to know which one they have.
    """
    return band_for(ceiling)


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


def rescale_similarity(cosine: float, *, noise: float, strong: float) -> float:
    """Where ``cosine`` sits between "this corpus's noise" and "a strong match", in ``[0, 1]``.

    The one transformation that lets an absolute cosine mean something to a reader. Raw cosine
    understates how good a good match is and wildly overstates how good a bad one is, because
    the scale does not start at zero: this embedder returns 0.4-odd for text with no relation to
    the query at all. Below ``noise`` the answer is 0.0 — not a small number, because "further
    from the query than unrelated text" is not a finer grade of relevance.
    """
    if strong <= noise:  # pragma: no cover - refused where the values are configured
        msg = f"strong similarity {strong!r} must exceed noise similarity {noise!r}"
        raise ValueError(msg)
    return min(1.0, max(0.0, (cosine - noise) / (strong - noise)))


def score_confidence(
    passages: Sequence[Candidate],
    *,
    identity: PipelineIdentity | None = None,
    legs: Sequence[str] = ("dense", "lexical"),
    rerank_stage: str | None = None,
    exhausted_budget: bool = False,
    noise_similarity: float = NOISE_SIMILARITY,
    strong_similarity: float = STRONG_SIMILARITY,
) -> Confidence:
    """Score the retrieval that produced ``passages``.

    Args:
        passages: The candidates that reached the context. Confidence is about what will be
            shown, not about everything that was considered.
        identity: What produced the ranking. Travels with the score, because two scores from
            different pipelines are not two measurements of one thing.
        legs: The retrieval legs **this pipeline declares**. Whether a leg found anything for
            this particular query is evidence and is scored; it is not a reason to suppress.
        rerank_stage: The stage name a reranker scored under, or ``None`` if none ran.
        exhausted_budget: Whether a leg stopped at its own caps. Caps the band at ``medium``.
        noise_similarity: Cosine at or below which a passage carries no information.
        strong_similarity: Cosine at which a passage is fully on-topic.

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

    components: dict[str, float] = {}
    suppressed: dict[str, str] = {}

    dense_leg = legs[0] if legs else None
    # Only the passages the dense leg actually scored. `.get(leg, 0.0)` was the bug this
    # replaces: a passage the lexical leg found and the dense leg never ranked has *no*
    # measured similarity, and reading the absent key as 0.0 asserts the dense leg looked at it
    # and found it orthogonal to the query. It did not look. On the query that exposed this,
    # BM25's top hit was averaged in as a zero and cost the best answer in the corpus a twentieth
    # of a point — the not-measured-versus-measured-zero conflation this module forbids one
    # component along, committed here in a default argument.
    scored = (
        [candidate.scores[dense_leg] for candidate in passages if dense_leg in candidate.scores]
        if dense_leg is not None
        else []
    )
    if dense_leg is None:
        # A pipeline with no fusion names no legs, so nothing here has a similarity to average.
        # Suppressed rather than scored zero: a zero would report weak evidence for what is a
        # property of the pipeline.
        suppressed[SIMILARITY] = (
            "this pipeline declares no retrieval legs to read a similarity from"
        )
    elif not scored:
        suppressed[SIMILARITY] = (
            f"no passage in the context carries a {dense_leg!r} score, so there is no similarity "
            f"this run could average — absent, not zero"
        )
    else:
        components[SIMILARITY] = rescale_similarity(
            _mean([max(value, 0.0) for value in scored]),
            noise=noise_similarity,
            strong=strong_similarity,
        )

    # Suppressed on the shape of the *pipeline*, never on what this query happened to match. A
    # lexical leg that ran and matched nothing has reported a fact about the query, and waiving
    # the agreement term for it rewards exactly the queries that deserve it least: a nonsense
    # query matches no keywords, so it would pay no agreement penalty while a real question that
    # matched some would pay one in full.
    if len(legs) < 2:  # noqa: PLR2004 - agreement needs two legs to compare
        suppressed[AGREEMENT] = (
            "this pipeline declares fewer than two retrieval legs, so no passage could carry "
            "two scores and a zero here would describe the pipeline rather than the evidence"
        )
    else:
        agreeing = sum(1 for candidate in passages if all(leg in candidate.scores for leg in legs))
        components[AGREEMENT] = agreeing / len(passages)

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
    # Ordered least to most specific, so the most informative reason is the one that survives
    # into a single `reason` field. A run that both stopped at its budget and retrieved nothing
    # resembling the query should say the second: "we did not finish looking" matters far less
    # than "what we found is what this corpus returns for a question it cannot answer", and the
    # budget shortfall is still on the trace either way.
    if band is not ConfidenceBand.NONE and reachable_band(ceiling) is band:
        reason = UNREACHABLE_BAND
    if exhausted_budget and band is ConfidenceBand.HIGH:
        band = ConfidenceBand.MEDIUM
        reason = BUDGET_CAPPED
    elif exhausted_budget:
        reason = BUDGET_CAPPED
    # The `none` band now *means* this, whenever similarity was actually measured: the rescale
    # puts a passage at the corpus's noise level at zero, so a run that lands here retrieved
    # nothing that stands above what an unanswerable question returns.
    if band is ConfidenceBand.NONE and SIMILARITY in components:
        reason = NOTHING_RESEMBLES

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
    "BUDGET_CAPPED",
    "NOISE_SIMILARITY",
    "NOTHING_RESEMBLES",
    "NOTHING_RETRIEVED",
    "RERANK",
    "SIMILARITY",
    "STRONG_SIMILARITY",
    "UNREACHABLE_BAND",
    "WEIGHTS",
    "band_for",
    "reachable_band",
    "rescale_similarity",
    "score_confidence",
]
