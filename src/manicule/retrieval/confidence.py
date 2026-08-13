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
unrelated questions averaged at most 0.457 across their context passages while real ones
averaged at least 0.531 (``docs/retrieval.md`` §8.4). Read raw, the two populations sit a
fifteenth of a point apart and land in the same band, which is exactly what was reported.
Subtracting the measured noise level and dividing by the distance to a strong match is what
makes "nothing in this corpus resembles your question" expressible at all — the same move the
evaluation harness makes when it reports a hit rate against a chance rate rather than bare.

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

from pydantic import BaseModel, ConfigDict, Field

from manicule.core.retrieval import Confidence, ConfidenceBand, PipelineIdentity

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

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

NOISE_SIMILARITY: Final = 0.54
"""Cosine at or below which **one passage** carries no information about the query.

**Measured, and re-measured when the statistic changed.** Both constants describe a single
passage, because :func:`rescale_similarity` is applied per passage and the results combined
afterwards — which is the only thing a cosine ever described. They were first calibrated against
the *mean* over a context, and that pairing was wrong in a way only a narrow query exposed (see
:func:`_similarity`).

Per passage the two populations sit much closer than their context means did, and that is the
number this constant has to respect. The measured boundary, over two corpora:

* The strongest passage any unanswerable query reached was **0.5194** — and both legs had found
  it, so at a floor of 0.45 or 0.50 it counted as evidence *and* collected the agreement term,
  carrying a query about nothing into the ``low`` band.
* The weakest top passage any answerable query produced was **0.5598**.

0.54 is the middle of that gap rather than either edge. Sitting mid-gap is deliberate and is the
same argument the retrieval floor makes in ``docs/retrieval.md`` §4.5: MLX and ONNX agree to
cosine 0.9999, worth about 0.01 of movement in a query-passage cosine, so a constant placed
against one edge could be crossed by a backend change. Platform may change throughput; it must
never change output.

This is a property of **the embedder and the corpus**, not a universal constant: BGE-M3's
similarities are not centred on zero for unrelated text, and a different model or a much more
topically concentrated corpus puts the noise level somewhere else. Re-measure it the way it was
measured — ask questions the corpus demonstrably cannot answer and read where their similarities
land — rather than adjusting it until an example looks right.

This is a property of **the embedder and the corpus**, not a universal constant: BGE-M3's
similarities are not centred on zero for unrelated text, and a different model or a much more
topically concentrated corpus puts the noise level somewhere else. Re-measure it the way it was
measured — ask questions the corpus demonstrably cannot answer and read where their similarities
land — rather than adjusting it until an example looks right.
"""

STRONG_SIMILARITY: Final = 0.65
"""Cosine at which one passage is treated as fully answering the query.

The top of the measured range rather than 1.0: cosine 1.0 against a chunk means the query *is*
that chunk, which no question is, so scaling to 1.0 would cap a flawless retrieval at roughly
two thirds of the component for a reason that has nothing to do with the evidence. Measured
top-1 cosines ran to 0.724 over this project's own documentation and 0.747 against a synthetic
glossary, so a passage at or above 0.65 has answered the question as well as this embedder ever
signals.
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
DEFINITION_CITED: Final = (
    "the context contains a glossary entry that explicitly defines a term this question asked "
    "the meaning of, and the citation resolves to it. The score stays where the similarity put "
    "it — a detector's confidence that some text is a definition is not a cosine and is not "
    "added to one — so this reports what was found rather than a number that was adjusted"
)
"""What is said instead of :data:`NOTHING_RESEMBLES` when a definition was cited.

**The correctness fix, and it needs no calibration.** The two sentences cannot both be true.
:data:`NOTHING_RESEMBLES` is a claim about the *corpus* — that it holds nothing addressing the
question — and a glossary entry defining the exact term the question named is a
counter-example to it sitting in the context the reader is looking at. Reporting both is not a
threshold set too high; it is two paths that never spoke, one reading a cosine and the other
reading a lookup.

Note what does **not** change: the score, the band, the components and the ceiling. A question
whose answer is one line of a twenty-five entry page really does have weak dense evidence, and
saying so is right. What was wrong was the sentence next to it.
"""
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


def combine_evidence(strengths: Iterable[float]) -> float:
    """Combine independent pieces of evidence for one answer, in ``[0, 1]``.

    ``1 - Π(1 - e_i)`` — the probability that *at least one* of them is real, if each were
    independent. It is chosen for four properties the alternatives do not have together:

    * **Passages carrying nothing cost nothing.** A term at 0 multiplies by 1. This is the whole
      correction: the pipeline always fills the context to ``final_top_k`` whether or not the
      corpus holds that many relevant passages, so a narrow question with one right answer is
      *guaranteed* filler — and an average made that filler count against the answer.
    * **More independent support raises it, and never lowers it.** Monotone non-decreasing in
      every term.
    * **It saturates rather than overflowing.** Ten good passages approach 1.0 without a clamp
      having to hide an out-of-range number.
    * **One strong passage is not proof.** A single piece of evidence at 0.86 yields 0.86, so
      rank 1 alone lands short of certainty and a second independent source is what closes the
      gap. Confidence is a statement about evidence, and one source is one source.
    """
    absent = 1.0
    for strength in strengths:
        absent *= 1.0 - min(1.0, max(0.0, strength))
    return 1.0 - absent


def evidence_per_passage(
    passages: Sequence[Candidate], dense_leg: str | None, *, noise: float, strong: float
) -> list[float | None]:
    """How much evidence each passage carries, positionally, or ``None`` where none was measured.

    The single definition of that question. Three callers need it — the component, the
    corroboration term and the diagnostic — and computing it three times is how the diagnostic
    ends up explaining a number nobody computed. ``None`` is not zero: it means the dense leg
    never ranked this passage, so there is nothing to rescale.
    """
    if dense_leg is None:
        return [None] * len(passages)
    return [
        rescale_similarity(max(candidate.scores[dense_leg], 0.0), noise=noise, strong=strong)
        if dense_leg in candidate.scores
        else None
        for candidate in passages
    ]


def strongest_per_document(
    passages: Sequence[Candidate], strengths: Sequence[float | None]
) -> dict[str, float]:
    """The best evidence each document offered, which is what may be combined.

    Chunks of one document are not independent observations, so only the strongest counts —
    otherwise a finely-chunked document manufactures certainty by being chunked finely, which is
    a property of the ingest configuration reported as a property of the evidence.
    """
    best: dict[str, float] = {}
    for candidate, strength in zip(passages, strengths, strict=True):
        if strength is None:
            continue
        document = candidate.chunk.document_id
        best[document] = max(best.get(document, 0.0), strength)
    return best


def _similarity(
    passages: Sequence[Candidate], dense_leg: str | None, *, noise: float, strong: float
) -> tuple[float, str]:
    """The dense evidence component, or why there is none.

    Returns the value and an empty string, or ``0.0`` and the reason it is unavailable. Absent
    and zero are different answers here and the caller must not be able to confuse them, which
    is why the reason rather than the number carries the signal.

    **The strongest passage per document, combined by :func:`combine_evidence`.** This replaced a
    mean, which was wrong in a way that only a narrow query exposes. Measured on a synthetic
    glossary: the question "What is NOW?" put the defining passage at rank 1 with cosine 0.623,
    and the four filler passages behind it pulled the mean to 0.387 — *below the noise floor*, so
    a perfect retrieval reported 0.0 and the band ``none``. The full-expansion phrasing scored a
    near-ideal 0.702 at rank 1 and also reported 0.0. An average answers "how on-topic is the
    typical passage shown", and nobody asked that; the question is how well supported the answer
    is, and filler behind a correct hit is not evidence against it.

    **Best per document, not per passage**, so that a document split into ten chunks is one piece
    of evidence rather than ten. Chunks of one page are not independent observations, and letting
    them compound is how a single well-matched document manufactures certainty.

    **Only the passages the dense leg actually scored count.** ``.get(leg, 0.0)`` was an earlier
    defect here: a passage the lexical leg found and the dense leg never ranked has *no* measured
    similarity, and reading the absent key as 0.0 asserts the dense leg looked at it and judged it
    orthogonal to the query. It never looked.
    """
    if dense_leg is None:
        # A pipeline with no fusion names no legs, so nothing here has a similarity to read.
        # Suppressed rather than scored zero: a zero would report weak evidence for what is a
        # property of the pipeline.
        return 0.0, "this pipeline declares no retrieval legs to read a similarity from"
    best = strongest_per_document(
        passages, evidence_per_passage(passages, dense_leg, noise=noise, strong=strong)
    )
    if not best:
        return 0.0, (
            f"no passage in the context carries a {dense_leg!r} score, so there is no similarity "
            f"this run could read — absent, not zero"
        )
    return combine_evidence(best.values()), ""


def _agreement(
    passages: Sequence[Candidate],
    legs: Sequence[str],
    dense_leg: str | None,
    *,
    evidence: float,
    noise: float,
    strong: float,
) -> float:
    """How much of the evidence both legs found, over the passages that carry evidence.

    **The denominator is the evidence-bearing passages, not every passage in the context**, and
    that is the same correction :func:`_similarity` needed for the same reason. Filler is
    guaranteed — the context is filled to ``final_top_k`` regardless of how much the corpus
    actually holds — and dividing by it meant a query with one strong, doubly-confirmed answer
    scored 1/5 for perfect corroboration. Asking whether the lexical leg also found the *filler*
    is not a question anybody wants answered.

    With no evidence-bearing passage there is nothing to corroborate, so this is 0.0 and the
    similarity component is already 0.0 for the same reason — a run with no evidence should not
    collect an agreement bonus for two legs concurring about noise.
    """
    if dense_leg is None:
        return 0.0
    strengths = evidence_per_passage(passages, dense_leg, noise=noise, strong=strong)
    bearing = [
        candidate
        for candidate, strength in zip(passages, strengths, strict=True)
        if strength is not None and strength > 0.0
    ]
    if not bearing:
        return 0.0
    agreeing = sum(1 for candidate in bearing if all(leg in candidate.scores for leg in legs))
    # Scaled by the evidence, not merely counted over it. **You cannot corroborate more than you
    # have.** Counting alone gave a nonsense query the full agreement weight: exactly one passage
    # cleared the floor, barely, both legs happened to touch it, and 1/1 paid out in full — 0.15
    # of a number that is supposed to say the corpus holds nothing. Multiplying by the evidence
    # level makes weak corroboration of weak evidence weak, which is the only reading that is
    # ever true.
    return (agreeing / len(bearing)) * evidence


def _rerank(passages: Sequence[Candidate], stage: str | None) -> tuple[float, str]:
    """The reranker component, or why there is none. Same absent-versus-zero contract as
    :func:`_similarity`."""
    if stage is None:
        return 0.0, (
            "no reranker ran, so there is no verification step to report. Its weight is not "
            "redistributed: a pipeline that skipped the check must not be able to claim it"
        )
    logits = [
        _sigmoid(candidate.scores[stage]) for candidate in passages if stage in candidate.scores
    ]
    if not logits:
        return 0.0, (
            f"the {stage!r} stage ran but scored none of the passages that reached the context"
        )
    return _mean(logits), ""


def score_confidence(
    passages: Sequence[Candidate],
    *,
    identity: PipelineIdentity | None = None,
    legs: Sequence[str] = ("dense", "lexical"),
    rerank_stage: str | None = None,
    exhausted_budget: bool = False,
    explicit_definition: bool = False,
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
        explicit_definition: Whether the context holds a glossary entry defining a term this
            question asked the meaning of. **Changes no number** — see :data:`DEFINITION_CITED`.
            It is passed in rather than derived here because deciding it needs the query text
            and the glossary lookup, neither of which this module may see: everything here works
            from candidates and scales, and reaching for a query would make the score depend on
            the wording of the question as well as on what was retrieved.
        noise_similarity: Cosine at or below which a passage carries no information.
        strong_similarity: Cosine at which a passage is fully on-topic.

    Returns:
        A :class:`~manicule.core.retrieval.Confidence` carrying the band, every component that
        contributed, every component that was suppressed and why, and the ceiling those
        suppressions left.
    """
    pipeline = identity or PipelineIdentity()
    if not passages:
        # No context means no passage to have been the definition, whatever the lookup found.
        # `explicit_definition` is deliberately not carried through here: the caller sets it from
        # a chunk being *present*, so this branch is unreachable with it true, and honouring it
        # would define behaviour for a state that cannot exist.
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
    measured, unavailable = _similarity(
        passages, dense_leg, noise=noise_similarity, strong=strong_similarity
    )
    if unavailable:
        suppressed[SIMILARITY] = unavailable
    else:
        components[SIMILARITY] = measured

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
        components[AGREEMENT] = _agreement(
            passages,
            legs,
            dense_leg,
            evidence=components.get(SIMILARITY, 0.0),
            noise=noise_similarity,
            strong=strong_similarity,
        )

    verified, unverifiable = _rerank(passages, rerank_stage)
    if unverifiable:
        suppressed[RERANK] = unverifiable
    else:
        components[RERANK] = verified

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
    #
    # Unless a definition of the queried term is sitting in the context, in which case the claim
    # is simply false and a different true thing is said instead. The *score* is untouched either
    # way — the branch chooses a sentence, never a number.
    if band is ConfidenceBand.NONE and SIMILARITY in components:
        reason = DEFINITION_CITED if explicit_definition else NOTHING_RESEMBLES

    return Confidence(
        score=score,
        band=band,
        components={name: WEIGHTS[name] * value for name, value in components.items()},
        suppressed=suppressed,
        ceiling=ceiling,
        reason=reason,
        explicit_definition=explicit_definition,
        pipeline=pipeline,
    )


class PassageEvidence(BaseModel):
    """What one passage contributed, without saying what it said.

    **No passage text, ever.** Diagnostics travel further than results — into logs, bug reports
    and screenshots — and a diagnostic that carries corpus text turns every one of those into a
    disclosure. The chunk id is enough to find the passage for anyone entitled to read it, and
    useless to anyone who is not.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    document_id: str
    raw_similarity: float | None = Field(
        default=None, description="The dense leg's cosine, or None if it never ranked this passage."
    )
    evidence: float = Field(
        ge=0.0, le=1.0, description="That cosine rescaled against the noise level. 0.0 is noise."
    )
    legs: tuple[str, ...] = Field(default=(), description="Which legs scored it, sorted.")
    counted: bool = Field(
        description="Whether it reached the combination at all. False for a passage whose "
        "document contributed a stronger one, which is how duplicate evidence is refused."
    )


class ConfidenceDiagnostics(BaseModel):
    """Every input to a confidence score, so the number can be argued with.

    Built on request rather than always, because it costs a second pass and most callers want
    the number. What it is *for* is the class of report this module keeps producing: "the right
    passage was rank 1 and confidence said none". Answering that took a measurement harness and
    a day; with this it is one call, and the answer is in the components rather than inferred
    from them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    score: float
    band: ConfidenceBand
    ceiling: float
    reason: str = ""
    components: dict[str, float] = Field(default_factory=dict)
    suppressed: dict[str, str] = Field(default_factory=dict)
    noise_similarity: float
    strong_similarity: float
    bands: tuple[tuple[float, str], ...] = Field(
        default=(), description="The thresholds this run was cut on, best band first."
    )
    weights: dict[str, float] = Field(default_factory=dict)
    passages: tuple[PassageEvidence, ...] = ()
    evidence_documents: int = Field(
        default=0, ge=0, description="Documents that cleared the noise floor and were combined."
    )
    explicit_definition: bool = Field(
        default=False,
        description="Whether a glossary entry defining the queried term reached the context. "
        "Appears here rather than in :attr:`components` or :attr:`suppressed` because it is "
        "neither: a component has a weight and contributes, a suppressed component has a weight "
        "and was prevented from contributing and so lowers :attr:`ceiling`, and this has no "
        "weight at all. Listing it beside the weighted components would invite exactly the "
        "reading it exists to prevent — that a definition is worth some number of points.",
    )


def explain_confidence(
    passages: Sequence[Candidate],
    *,
    legs: Sequence[str] = ("dense", "lexical"),
    rerank_stage: str | None = None,
    exhausted_budget: bool = False,
    explicit_definition: bool = False,
    noise_similarity: float = NOISE_SIMILARITY,
    strong_similarity: float = STRONG_SIMILARITY,
) -> ConfidenceDiagnostics:
    """Score a retrieval and show the working.

    Runs the same :func:`score_confidence` the pipeline runs rather than reimplementing it, so
    the explanation cannot drift from the number it explains — a diagnostic that computes its own
    answer is a second implementation, and the one that disagrees is always the one nobody reads.
    """
    confidence = score_confidence(
        passages,
        legs=legs,
        rerank_stage=rerank_stage,
        exhausted_budget=exhausted_budget,
        explicit_definition=explicit_definition,
        noise_similarity=noise_similarity,
        strong_similarity=strong_similarity,
    )
    dense_leg = legs[0] if legs else None
    strengths = evidence_per_passage(
        passages, dense_leg, noise=noise_similarity, strong=strong_similarity
    )
    strongest = strongest_per_document(passages, strengths)

    detail: list[PassageEvidence] = []
    claimed: set[str] = set()
    for candidate, measured in zip(passages, strengths, strict=True):
        raw = candidate.scores.get(dense_leg) if dense_leg is not None else None
        strength = measured or 0.0
        document = candidate.chunk.document_id
        # A passage at zero evidence contributed nothing, so it is not "counted" however it
        # compares to its document's best — otherwise every filler passage in a context of noise
        # reports as having been used, which is the opposite of what this field exists to show.
        counted = strength > 0.0 and document not in claimed and strongest.get(document) == strength
        if counted:
            claimed.add(document)
        detail.append(
            PassageEvidence(
                chunk_id=candidate.chunk.id,
                document_id=document,
                raw_similarity=raw,
                evidence=strength,
                legs=tuple(sorted(candidate.scores)),
                counted=counted,
            )
        )

    return ConfidenceDiagnostics(
        score=confidence.score,
        band=confidence.band,
        ceiling=confidence.ceiling,
        reason=confidence.reason,
        components=dict(confidence.components),
        suppressed=dict(confidence.suppressed),
        noise_similarity=noise_similarity,
        strong_similarity=strong_similarity,
        bands=tuple((threshold, band.value) for threshold, band in BANDS),
        weights=dict(WEIGHTS),
        passages=tuple(detail),
        evidence_documents=sum(1 for value in strongest.values() if value > 0.0),
        explicit_definition=confidence.explicit_definition,
    )


__all__ = [
    "AGREEMENT",
    "BANDS",
    "BUDGET_CAPPED",
    "DEFINITION_CITED",
    "NOISE_SIMILARITY",
    "NOTHING_RESEMBLES",
    "NOTHING_RETRIEVED",
    "RERANK",
    "SIMILARITY",
    "STRONG_SIMILARITY",
    "UNREACHABLE_BAND",
    "WEIGHTS",
    "ConfidenceDiagnostics",
    "PassageEvidence",
    "band_for",
    "combine_evidence",
    "evidence_per_passage",
    "explain_confidence",
    "reachable_band",
    "rescale_similarity",
    "score_confidence",
    "strongest_per_document",
]
