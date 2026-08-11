"""The check that makes every other number in this package mean something.

**A system that retrieves at chance cannot be reported as anything but useless.**

That sentence is the whole design. An evaluation harness is a measuring instrument, and the
characteristic failure of a measuring instrument is not reading wrong — it is reading
plausibly while measuring nothing. A retrieval evaluation built on a component with no
semantic content produces well-formed reports, sensible-looking win rates and confident
conclusions about features that were never distinguishable from noise, and nothing anywhere in
the output says so. Every conclusion drawn from it is unfalsifiable, and the damage compounds:
each feature added on top of it is now justified by a number, and the numbers all came from
the same place.

So before any preference is recorded, each side must demonstrate that it retrieves at all.

**How.** A probe is a set of questions whose answer is known by construction — no relevance
judgements, no labelling session. The default construction uses document titles: searching for
a document's title should return that document. That is the least a retrieval system can be
asked to do, which is exactly the property wanted; a probe that only a good system passes
would be a quality benchmark, and this is a liveness check.

**The verdict is a hypothesis test, not a threshold.** "It got 6 of 20" says nothing until it
is compared against what guessing would do. With ``k`` results drawn from a corpus of ``N``
documents, chance is ``k / N`` per item, hits are binomial, and the probe reports the
probability that chance alone would have done at least this well. Below ``alpha`` the system
discriminates; at or above it, the system is at chance and its preferences are noise.

**And the probe refuses when it could not have told the difference.** Three refusals, each
guarding a way of producing a verdict that was never in doubt:

- A corpus so small that ``k`` results cover most of it. Chance approaches certainty and every
  system passes.
- A system that cannot say how many documents it is choosing between. Chance is then unknown
  and any p-value is invented.
- Too few probe items for even a *perfect* run to reach significance. A check whose failing
  verdict is unconditional is not a check — it would report a flawless system as being at
  chance, which is the mirror image of the failure this module exists to prevent, and just as
  useless.

The refusals are raises rather than caveats. A caveat lives in a field somebody has to read;
this way there is no report at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from manicule.core.retrieval import Filter
from manicule.evaluation.errors import (
    AtChanceError,
    ProbeUnusableError,
    UnderpoweredProbeError,
)
from manicule.evaluation.statistics import binomial_tail, smallest_detectable_sample
from manicule.evaluation.systems import ResultItem, SystemUnderComparison

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from manicule.core.protocols import DocStore

DEFAULT_ALPHA = 0.01
"""How surprising a result must be before "it retrieves" is admitted.

Stricter than the 0.05 convention, on purpose and in one direction only: this gate protects
against admitting a system that does not retrieve, and the cost of the stricter threshold is
that a genuinely working system needs a few more probe items. Those are free — they are
derived from the corpus.
"""

DEFAULT_K = 5
"""Results examined per probe item. The question is "did it find the document at all", so a
small window rather than one: a system that ranks the right document third is retrieving."""

MIN_TITLE_WORDS = 2
"""How many words a title needs before it can be a probe query.

One-word titles collide across documents and make the known answer ambiguous, which would
show up as a working system failing the probe.
"""

TITLE_PAGE = 200
"""Documents read per round trip when deriving probe items."""

ARITHMETIC_TOLERANCE = 1e-9
"""How far a recorded figure may sit from its recomputation before the outcome is refused.

Floating-point slack rather than a margin of judgement: the check re-runs the same
computations that produced the numbers, so anything larger means they came from somewhere else.
"""


class ProbeItem(BaseModel):
    """One question whose correct answer is known without anybody judging anything."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1)
    document_ids: frozenset[str] = Field(
        default=frozenset(),
        description="Documents that count as a hit. The preferred form, and the only one that "
        "is exact.",
    )
    contains: str = Field(
        default="",
        description="A distinctive span the correct passage contains. For a system whose "
        "document ids are not manicule's — the text is the only identifier two independent "
        "systems share. The span must be distinctive, or chance stops being ``k/N``.",
    )

    @model_validator(mode="after")
    def _has_a_known_answer(self) -> Self:
        if not self.document_ids and not self.contains:
            msg = (
                f"probe item {self.text!r} declares no correct answer. A probe item without one "
                f"is scored as a miss for every system, which drags a working system towards "
                f"chance — the exact reading this probe exists to make impossible"
            )
            raise ValueError(msg)
        return self


class ProbeOutcome(BaseModel):
    """What the probe measured, in the form that makes the verdict checkable.

    Every input to :attr:`discriminates` is recorded beside it, **and the arithmetic is re-done
    whenever one of these is constructed.** Recording the inputs is not enough on its own: a
    verdict nobody recomputes is a verdict that has to be trusted, and this type is read back
    off disk by :func:`~manicule.evaluation.report.build_report`, where "trusted" means
    "whatever the file says". A record claiming a decisive ``p_value`` beside one hit in
    twenty-four would otherwise pass every guard in this package and launder noise into a rate.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    config_label: str
    trials: int = Field(ge=1)
    hits: int = Field(ge=0)
    k: int = Field(ge=1)
    pool_size: int = Field(ge=1)
    chance_rate: float = Field(ge=0.0, le=1.0)
    hit_rate: float = Field(ge=0.0, le=1.0)
    p_value: float = Field(ge=0.0, le=1.0)
    alpha: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _the_arithmetic_holds(self) -> Self:
        """Recompute every derived field and refuse the outcome if any disagrees.

        The tolerance is floating-point slack, not a margin: these are the same computations
        that produced the numbers, so anything beyond rounding means the record was not
        produced by a probe.
        """
        if self.hits > self.trials:
            msg = f"{self.hits} hits in {self.trials} trials is not possible"
            raise ValueError(msg)
        checks: tuple[tuple[str, float, float], ...] = (
            ("hit_rate", self.hit_rate, self.hits / self.trials),
            ("chance_rate", self.chance_rate, self.k / self.pool_size),
            ("p_value", self.p_value, binomial_tail(self.hits, self.trials, self.chance_rate)),
        )
        for name, recorded, recomputed in checks:
            if abs(recorded - recomputed) > ARITHMETIC_TOLERANCE:
                msg = (
                    f"{name} is recorded as {recorded!r} but recomputing it from this outcome's "
                    f"own numbers gives {recomputed!r}. A probe outcome whose arithmetic does "
                    f"not hold was not produced by a probe, and the verdict it carries decides "
                    f"whether a whole file of judgements counts"
                )
                raise ValueError(msg)
        return self

    @property
    def discriminates(self) -> bool:
        """Whether this system retrieves better than guessing would.

        ``False`` is not a soft finding. A side that fails this may not have preferences
        recorded for it, and records already carrying a failure may not be summarised.
        """
        return self.p_value <= self.alpha

    def describe(self) -> str:
        """One line, phrased so it cannot be quoted as a quality result."""
        verdict = "discriminates" if self.discriminates else "AT CHANCE"
        return (
            f"{self.config_label}: {verdict} — found the known answer in "
            f"{self.hits}/{self.trials} probes ({self.hit_rate:.0%}) against a chance rate of "
            f"{self.chance_rate:.1%} (p = {self.p_value:.2e}, alpha = {self.alpha})"
        )


class DiscriminationProbe:
    """Runs the known-answer probe and issues — or refuses to issue — a verdict."""

    def __init__(
        self,
        items: Iterable[ProbeItem],
        *,
        k: int = DEFAULT_K,
        alpha: float = DEFAULT_ALPHA,
    ) -> None:
        self._items = tuple(items)
        self._k = k
        self._alpha = alpha
        if not self._items:
            msg = (
                "a discrimination probe with no items would admit every system, including one "
                "that returns nothing. Derive items from the corpus with probe_from_titles, or "
                "supply your own"
            )
            raise ProbeUnusableError(msg)
        if k < 1:
            msg = f"k must be at least 1, got {k}"
            raise ProbeUnusableError(msg)
        if not 0.0 < alpha < 1.0:
            msg = f"alpha must be strictly between 0 and 1, got {alpha!r}"
            raise ProbeUnusableError(msg)

    @property
    def items(self) -> tuple[ProbeItem, ...]:
        return self._items

    async def run(self, system: SystemUnderComparison) -> ProbeOutcome:
        """Measure the system. Raises only when no honest verdict is available.

        ``k / N`` treats the ``k`` results as ``k`` distinct documents. A system returning
        several chunks of one document examines fewer documents than that, so the real chance
        of a hit is lower and this null is an over-estimate. That error runs in the safe
        direction and only that direction: an over-stated null makes the test harder to pass,
        so it can make a working system look worse and can never admit one that is guessing.

        Raises:
            ProbeUnusableError: The corpus size is unknown, or is too small for ``k`` results
                to be distinguishable from the whole corpus.
            UnderpoweredProbeError: Even a perfect run could not reach ``alpha`` with this many
                items, so the verdict would have been "at chance" whatever happened.
        """
        pool_size = system.corpus_version.document_count
        if pool_size is None:
            msg = (
                f"{system.config_label} does not report how many documents it holds, so there "
                f"is no chance rate to compare against and any verdict would be invented. Set "
                f"CorpusVersion.document_count on the adapter"
            )
            raise ProbeUnusableError(msg)
        chance = self._chance_rate(pool_size, system.config_label)
        self._require_power(chance, system.config_label)

        hits = 0
        for item in self._items:
            result = await system.search(item.text, limit=self._k)
            if _is_hit(item, result.items[: self._k]):
                hits += 1

        trials = len(self._items)
        return ProbeOutcome(
            config_label=system.config_label,
            trials=trials,
            hits=hits,
            k=self._k,
            pool_size=pool_size,
            chance_rate=chance,
            hit_rate=hits / trials,
            p_value=binomial_tail(hits, trials, chance),
            alpha=self._alpha,
        )

    async def certify(self, system: SystemUnderComparison) -> ProbeOutcome:
        """Measure the system and refuse it if it is at chance.

        Raises:
            AtChanceError: The system retrieves no better than guessing. Its preferences would
                be a measurement of noise, so none are collected.
        """
        outcome = await self.run(system)
        if not outcome.discriminates:
            msg = (
                f"{outcome.describe()}. A system that cannot be distinguished from guessing has "
                f"nothing to say about retrieval quality, so no preferences will be recorded "
                f"for it. Check the embedder, the index and the workspace scope before reading "
                f"this as a finding about the pipeline"
            )
            raise AtChanceError(msg)
        return outcome

    def _chance_rate(self, pool_size: int, label: str) -> float:
        """``k`` results drawn from ``pool_size`` documents, under a ranking that ignores the
        query.

        An empty corpus is separated from a merely small one, and not for tidiness: computing
        the guessing rate for a message about it divides by zero, so the diagnosis this method
        exists to give would be replaced by a ``ZeroDivisionError`` from inside an f-string. An
        empty index is also the single most likely reason somebody is reading this error, and
        "nothing is indexed" is a different instruction from "index more".
        """
        if pool_size == 0:
            msg = (
                f"{label} reports an empty corpus, so there is nothing to retrieve and no "
                f"chance rate to compare against. Index some documents, or check that the "
                f"workspace scope names the workspace the corpus is in"
            )
            raise ProbeUnusableError(msg)
        if pool_size <= self._k:
            msg = (
                f"{label} holds {pool_size} documents and the probe examines {self._k} results, "
                f"so guessing would score {min(1.0, self._k / pool_size):.0%} and every system "
                f"passes. Index more documents or lower k"
            )
            raise ProbeUnusableError(msg)
        return self._k / pool_size

    def _require_power(self, chance: float, label: str) -> None:
        """Refuse a probe that could not have detected a perfect system."""
        best_possible = binomial_tail(len(self._items), len(self._items), chance)
        if best_possible > self._alpha:
            needed = smallest_detectable_sample(chance, self._alpha)
            msg = (
                f"{len(self._items)} probe items cannot distinguish {label} from chance at "
                f"alpha = {self._alpha}: a system that found the known answer every single time "
                f"would still score p = {best_possible:.3g}. At a chance rate of {chance:.1%} "
                f"this probe needs at least {needed} items. It refuses to run rather than "
                f"report 'at chance' for a system it could not have cleared"
            )
            raise UnderpoweredProbeError(msg)


def _is_hit(item: ProbeItem, results: Sequence[ResultItem]) -> bool:
    """Whether the known answer appears in what came back.

    Document id first, because it is exact. Text containment second, because two systems that
    were never built together share no identifiers and the passage text is the only thing they
    both have.
    """
    needle = item.contains.casefold()
    for result in results:
        if result.document_id in item.document_ids:
            return True
        if needle and needle in result.text.casefold():
            return True
    return False


async def probe_from_titles(
    docstore: DocStore,
    *,
    workspace_ids: frozenset[str],
    limit: int = 100,
) -> tuple[ProbeItem, ...]:
    """Derive probe items from the corpus, with no labelling and no authoring.

    A document's own title, used as the query, with that document as the known answer. This is
    a deliberately low bar — a title is the most retrievable text a document has — and a low
    bar is what a liveness check needs. Anything that fails it is not retrieving.

    Two exclusions, both because they would make a working system look broken rather than
    because they are inconvenient:

    - **Titles shorter than :data:`MIN_TITLE_WORDS` words.** One word collides across
      documents, so the "correct" answer is ambiguous and a correct result scores as a miss.
    - **Titles that are not unique in the corpus.** Same reason, arrived at from the other
      direction, and the version that actually bites: two documents named ``README`` make each
      other's probe unanswerable.

    Ordered by document id so the same corpus yields the same probe on every run. A probe that
    resampled would make two runs of the same configuration disagree for a reason that has
    nothing to do with retrieval.
    """
    scope = Filter(workspace_ids=workspace_ids)
    titles: dict[str, list[str]] = {}
    offset = 0
    while True:
        page = await docstore.list_documents(scope, limit=TITLE_PAGE, offset=offset)
        if not page:
            break
        for document in page:
            title = (document.title or "").strip()
            if len(title.split()) >= MIN_TITLE_WORDS:
                titles.setdefault(title, []).append(document.id)
        if len(page) < TITLE_PAGE:
            break
        offset += TITLE_PAGE

    unique = sorted((title, ids[0]) for title, ids in titles.items() if len(ids) == 1)
    return tuple(
        ProbeItem(text=title, document_ids=frozenset({document_id}))
        for title, document_id in unique[:limit]
    )


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_K",
    "MIN_TITLE_WORDS",
    "TITLE_PAGE",
    "DiscriminationProbe",
    "ProbeItem",
    "ProbeOutcome",
    "probe_from_titles",
]
