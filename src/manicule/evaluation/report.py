"""Reading a file of judgments, per intent category, without overclaiming.

Three properties, and each is a refusal or a disclosure rather than a calculation.

**Per category, never one average.** A change that helps explanatory questions and destroys
exact-identifier lookups reads as a small overall win. The categories are reported as rows and
the overall figure is one row among them rather than the headline.

**Every rate carries its interval.** Seven wins from ten is ``0.70``, and its 95% interval
runs from ``0.40`` to ``0.89`` — visibly containing the point where neither system is better.
A rate quoted without that is the single easiest way for a harness to manufacture a finding.

**A report says what it is.** An example query set produces a report whose first line says the
numbers are illustrative, and :attr:`PreferenceReport.is_evidence` is ``False``. Not a
convention about how to present results — a field, set from the query set's declared
provenance, that the rendering reads.

And it refuses more than it reports. Records from different systems, different corpora or
different provenance are not summarized together, and records carrying a side that was at
chance are not summarized at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from manicule.evaluation.corpus import CorpusVersion
from manicule.evaluation.errors import AtChanceError, IncomparableRecordsError
from manicule.evaluation.preference import Preference, PreferenceRecord
from manicule.evaluation.probe import ProbeOutcome
from manicule.evaluation.queries import Intent, Provenance
from manicule.evaluation.statistics import sign_test, wilson_interval

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

ILLUSTRATIVE = (
    "ILLUSTRATIVE ONLY — this query set declares itself an example. These numbers demonstrate "
    "the harness and are not a measurement of retrieval quality."
)
"""The banner an example-provenance report leads with.

Its wording is deliberately unquotable as a result: any sentence lifted out of it says what it
is not.
"""

UNVERIFIED_CORPUS = (
    "corpus identity asserted by label only — at least one side could not produce a content "
    "digest, so 'the same documents' is a claim rather than a check"
)


class IntentSummary(BaseModel):
    """One category's row, with everything needed to judge how much it is worth."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent: Intent | None = Field(
        default=None, description="``None`` is the row over every category."
    )
    judged: int = Field(ge=0, description="Admissible records in this category.")
    left_wins: int = Field(ge=0)
    right_wins: int = Field(ge=0)
    ties: int = Field(ge=0)
    neither: int = Field(ge=0)

    @property
    def label(self) -> str:
        return "overall" if self.intent is None else self.intent.value

    @property
    def decided(self) -> int:
        """Judgments that expressed a preference. Ties and ``neither`` are not among them."""
        return self.left_wins + self.right_wins

    @property
    def left_win_rate(self) -> float | None:
        """Left's share of the decided judgments, or ``None`` when none were decided.

        ``None`` rather than 0.5, because "nothing was decided" and "they split evenly" are
        different findings and only one of them is about the systems.
        """
        return None if self.decided == 0 else self.left_wins / self.decided

    @property
    def interval(self) -> tuple[float, float]:
        """A 95% Wilson interval on :attr:`left_win_rate`."""
        return wilson_interval(self.left_wins, self.decided)

    @property
    def p_value(self) -> float:
        """Two-sided sign test. How often a coin would split this lopsidedly."""
        return sign_test(self.left_wins, self.right_wins)

    @property
    def separates(self) -> bool:
        """Whether this row distinguishes the two systems at the 5% level.

        A property of the row, read by the rendering. Nothing here decides what to do about a
        row that does not separate; it says so and stops.
        """
        significance = 0.05
        return self.decided > 0 and self.p_value < significance


class PreferenceReport(BaseModel):
    """A whole file of judgments, summarized and labeled."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    left_label: str
    right_label: str
    query_set: str
    provenance: Provenance
    left_corpus: CorpusVersion
    right_corpus: CorpusVersion
    corpus_verified: bool
    left_probe: ProbeOutcome
    right_probe: ProbeOutcome
    total: int = Field(ge=0, description="Records read, admissible or not.")
    excluded: tuple[tuple[str, int], ...] = Field(
        default=(),
        description="Reasons a record was not counted, and how many carried each. Reported "
        "rather than dropped: a run where most pairings were excluded is a finding about the "
        "run, and a rate computed from the remainder without saying so is not.",
    )
    overall: IntentSummary
    by_intent: tuple[IntentSummary, ...] = ()
    stage_delta: tuple[str, ...] = Field(
        default=(),
        description="Stages present on one side only, or configured differently on the two. "
        "The comparison's method is two pipelines differing in exactly one place, and this is "
        "what says whether that held.",
    )

    @property
    def is_evidence(self) -> bool:
        """Whether these numbers may be described as a measurement."""
        return self.provenance is not Provenance.EXAMPLE

    @property
    def judged(self) -> int:
        return self.overall.judged

    def render(self) -> str:
        """The report as text, leading with whatever weakens it."""
        lines: list[str] = []
        if not self.is_evidence:
            lines.extend([ILLUSTRATIVE, ""])
        lines.append(f"{self.left_label}  vs  {self.right_label}")
        lines.append(f"query set: {self.query_set} ({self.provenance.value})")
        lines.append(
            f"corpus: {self.left_corpus.label} "
            f"({'verified by digest' if self.corpus_verified else 'label only'})"
        )
        if not self.corpus_verified:
            lines.append(f"  note: {UNVERIFIED_CORPUS}")
        lines.append(f"probe: {self.left_probe.describe()}")
        lines.append(f"probe: {self.right_probe.describe()}")
        if self.stage_delta:
            lines.append(f"stages differing between the sides: {', '.join(self.stage_delta)}")
        lines.append(f"records: {self.total} read, {self.overall.judged} counted")
        for reason, count in self.excluded:
            lines.append(f"  excluded {count}: {reason}")
        lines.append("")
        lines.append(
            f"{'category':<20} {'n':>4} {'left':>5} {'right':>5} {'tie':>4} {'nei':>4} "
            f"{'left win rate':>28} {'p':>8}"
        )
        for row in (*self.by_intent, self.overall):
            lines.append(_render_row(row, self.left_label))
        return "\n".join(lines) + "\n"


def _render_row(row: IntentSummary, left_label: str) -> str:
    rate = row.left_win_rate
    if rate is None:
        shown = "nothing decided"
    else:
        low, high = row.interval
        shown = f"{rate:.0%} for {left_label} [{low:.0%}, {high:.0%}]"
    return (
        f"{row.label:<20} {row.judged:>4} {row.left_wins:>5} {row.right_wins:>5} "
        f"{row.ties:>4} {row.neither:>4} {shown:>28} {row.p_value:>8.3f}"
    )


def build_report(records: Iterable[PreferenceRecord]) -> PreferenceReport:
    """Summarize a set of records, or refuse to.

    Raises:
        IncomparableRecordsError: There are no records, or they span more than one pair of
            systems, more than one corpus, or more than one query-set provenance. Averaging
            across any of those produces a figure nobody can attribute to anything.
        AtChanceError: A record names a side that did not beat chance. Such a record should
            not exist, and summarizing one would launder noise into a rate.
    """
    collected = list(records)
    if not collected:
        msg = "no preference records to report on"
        raise IncomparableRecordsError(msg)

    first = collected[0]
    _require_one_comparison(collected, first)
    _require_discriminating(collected)

    admissible = [record for record in collected if record.admissible]
    excluded: dict[str, int] = {}
    for record in collected:
        for reason in record.inadmissible_because:
            excluded[reason] = excluded.get(reason, 0) + 1

    by_intent = tuple(
        _summarize([r for r in admissible if r.intent is intent], intent=intent)
        for intent in sorted({record.intent for record in admissible})
    )
    return PreferenceReport(
        left_label=first.left.config_label,
        right_label=first.right.config_label,
        query_set=first.query_set,
        provenance=first.provenance,
        left_corpus=first.left.corpus_version,
        right_corpus=first.right.corpus_version,
        corpus_verified=first.left.corpus_version.agrees_verifiably_with(
            first.right.corpus_version
        ),
        left_probe=first.left_probe,
        right_probe=first.right_probe,
        total=len(collected),
        excluded=tuple(sorted(excluded.items())),
        overall=_summarize(admissible, intent=None),
        by_intent=by_intent,
        stage_delta=_stage_delta(admissible),
    )


def _require_one_comparison(records: Sequence[PreferenceRecord], first: PreferenceRecord) -> None:
    pairs = {(record.left.config_label, record.right.config_label) for record in records}
    if len(pairs) > 1:
        listed = ", ".join(f"{left} vs {right}" for left, right in sorted(pairs))
        msg = (
            f"these records cover more than one comparison ({listed}). Summarizing them "
            f"together would average across different systems"
        )
        raise IncomparableRecordsError(msg)
    provenances = {record.provenance for record in records}
    if len(provenances) > 1:
        listed = ", ".join(sorted(p.value for p in provenances))
        msg = (
            f"these records come from query sets with different provenance ({listed}). An "
            f"example set mixed into a real one produces a report that is neither"
        )
        raise IncomparableRecordsError(msg)
    for record in records:
        for side, reference in (
            (record.left.corpus_version, first.left.corpus_version),
            (record.right.corpus_version, first.right.corpus_version),
        ):
            moved = reference.disagreement_with(side)
            if moved is not None:
                msg = (
                    f"these records were made against different corpora, so they are not "
                    f"repeated measurements of one thing: {moved}"
                )
                raise IncomparableRecordsError(msg)


def _require_discriminating(records: Sequence[PreferenceRecord]) -> None:
    """The reporting-time half of the chance-level guard.

    The harness already refuses to *record* a preference for a side at chance, and
    :class:`~manicule.evaluation.preference.PreferenceRecord` refuses to construct one. This is
    the third place the same rule is enforced, and it is not redundant: records arrive from a
    file, files outlive the process that wrote them, and a file is exactly where a record
    written by some other tool would enter. The rule has to hold at the point the number is
    produced, not only at the point the judgment was made.
    """
    failing = sorted(
        {
            probe.config_label
            for record in records
            for probe in (record.left_probe, record.right_probe)
            if not probe.discriminates
        }
    )
    if failing:
        msg = (
            f"these records name a side that retrieves at chance ({', '.join(failing)}), so "
            f"they are judgments about noise. No rate will be computed from them"
        )
        raise AtChanceError(msg)


def _summarize(records: Sequence[PreferenceRecord], *, intent: Intent | None) -> IntentSummary:
    counts = dict.fromkeys(Preference, 0)
    for record in records:
        counts[record.preference] += 1
    return IntentSummary(
        intent=intent,
        judged=len(records),
        left_wins=counts[Preference.LEFT],
        right_wins=counts[Preference.RIGHT],
        ties=counts[Preference.TIE],
        neither=counts[Preference.NEITHER],
    )


def _stage_delta(records: Sequence[PreferenceRecord]) -> tuple[str, ...]:
    """Which stages the two sides did not share, across the whole run.

    Per-stage attribution, and it costs nothing: a stage's name and declared configuration
    arrive on every result, so the difference between the two pipelines is a set operation
    rather than something anybody has to remember to write down. A comparison whose method is
    "two pipelines differing in exactly one place" can then be checked against what ran.
    """
    differing: set[str] = set()
    for record in records:
        left = {stage.name: stage.config for stage in record.left.stages}
        right = {stage.name: stage.config for stage in record.right.stages}
        differing |= set(left) ^ set(right)
        differing |= {name for name in set(left) & set(right) if left[name] != right[name]}
    return tuple(sorted(differing))


__all__ = [
    "ILLUSTRATIVE",
    "UNVERIFIED_CORPUS",
    "IntentSummary",
    "PreferenceReport",
    "build_report",
]
