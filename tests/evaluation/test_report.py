"""What a report says, what it refuses to say, and what it will not summarise at all."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from manicule.evaluation.corpus import CorpusVersion
from manicule.evaluation.errors import AtChanceError, IncomparableRecordsError
from manicule.evaluation.preference import Preference, PreferenceRecord, assign_slots
from manicule.evaluation.probe import ProbeOutcome
from manicule.evaluation.queries import Intent, Provenance
from manicule.evaluation.report import ILLUSTRATIVE, UNVERIFIED_CORPUS, build_report
from manicule.evaluation.statistics import binomial_tail
from manicule.evaluation.systems import StageObservation, SystemResult
from tests.evaluation.fakes import an_item

if TYPE_CHECKING:
    from collections.abc import Sequence

VERSION = CorpusVersion(label="fixture", digest="sha256:aaa", document_count=60)


def an_outcome(label: str, *, hits: int = 20) -> ProbeOutcome:
    """A probe outcome whose numbers agree with each other.

    ``p_value`` is computed rather than supplied: the model recomputes it, so a fixture that
    asserted one would be a fixture testing whether the arithmetic guard is switched on rather
    than whatever the test is about.
    """
    trials, k, pool_size = 20, 3, 60
    chance = k / pool_size
    return ProbeOutcome(
        config_label=label,
        trials=trials,
        hits=hits,
        k=k,
        pool_size=pool_size,
        chance_rate=chance,
        hit_rate=hits / trials,
        p_value=binomial_tail(hits, trials, chance),
        alpha=0.01,
    )


def a_result(label: str, **overrides: object) -> SystemResult:
    payload: dict[str, object] = {
        "config_label": label,
        "configuration": {"stages": ["dense"]},
        "corpus_version": VERSION,
        "items": (an_item("d1"),),
    }
    payload.update(overrides)
    return SystemResult.model_validate(payload)


def a_record(index: int = 0, **overrides: object) -> PreferenceRecord:
    payload: dict[str, object] = {
        "recorded_at": datetime.now(UTC),
        "query_id": f"q{index}",
        "query_text": "a question",
        "intent": Intent.LOOKUP,
        "query_set": "fixture",
        "provenance": Provenance.AUTHORED,
        "left": a_result("alpha"),
        "right": a_result("beta"),
        "left_probe": an_outcome("alpha"),
        "right_probe": an_outcome("beta"),
        "slots": assign_slots(f"q{index}"),
        "preference": Preference.LEFT,
        "judge": "test",
    }
    payload.update(overrides)
    return PreferenceRecord.model_validate(payload)


def records_with(preferences: Sequence[Preference], **overrides: object) -> list[PreferenceRecord]:
    return [
        a_record(index, preference=preference, **overrides)
        for index, preference in enumerate(preferences)
    ]


def test_categories_are_reported_as_rows_rather_than_averaged_into_one_number() -> None:
    """A change that helps one category and ruins another reads as a small win when averaged."""
    helped = records_with([Preference.LEFT] * 8)
    hurt = [
        a_record(100 + i, preference=Preference.RIGHT, intent=Intent.EXACT_IDENTIFIER)
        for i in range(6)
    ]

    report = build_report([*helped, *hurt])

    rows = {row.label: row for row in report.by_intent}
    assert rows["lookup"].left_wins == 8
    assert rows["exact_identifier"].right_wins == 6
    assert report.overall.left_wins == 8
    assert report.overall.right_wins == 6
    assert not report.overall.separates, "the averaged row hides both effects, and says so"
    assert rows["lookup"].separates


def test_a_small_win_carries_an_interval_that_contains_no_difference() -> None:
    """Seven of ten is the number this package exists to stop being quoted as a result."""
    report = build_report(records_with([Preference.LEFT] * 7 + [Preference.RIGHT] * 3))

    low, high = report.overall.interval
    assert low < 0.5 < high
    assert not report.overall.separates
    assert "[" in report.render(), "the interval must appear in the rendering, not only the model"


def test_ties_and_neither_are_counted_apart() -> None:
    """A set on which both systems fail must not read as a dead heat."""
    report = build_report(records_with([Preference.TIE] * 3 + [Preference.NEITHER] * 5))

    assert report.overall.ties == 3
    assert report.overall.neither == 5
    assert report.overall.decided == 0
    assert report.overall.left_win_rate is None
    assert "nothing decided" in report.render()


def test_a_report_from_an_example_query_set_says_so_in_its_first_line() -> None:
    """The example ships to demonstrate the harness. This is what keeps it from being quoted."""
    report = build_report(records_with([Preference.LEFT] * 8, provenance=Provenance.EXAMPLE))

    assert not report.is_evidence
    assert report.render().startswith(ILLUSTRATIVE)


def test_an_unverified_corpus_is_labelled_on_the_face_of_the_report() -> None:
    """One side could not produce a digest, so sameness is a claim rather than a check."""
    report = build_report(
        records_with(
            [Preference.LEFT] * 4,
            right=a_result("beta", corpus_version=CorpusVersion(label="fixture")),
        )
    )

    assert not report.corpus_verified
    assert UNVERIFIED_CORPUS in report.render()


def test_excluded_pairings_are_counted_and_named_rather_than_dropped() -> None:
    """A run where most pairings were excluded is a finding about the run."""
    clean = records_with([Preference.LEFT] * 2)
    degraded = [
        a_record(50 + i, right=a_result("beta", incomparable=("degraded leg",))) for i in range(3)
    ]

    report = build_report([*clean, *degraded])

    assert report.total == 5
    assert report.overall.judged == 2
    assert report.excluded == (("beta: degraded leg", 3),)
    assert "excluded 3" in report.render()


def test_the_stages_the_two_sides_did_not_share_are_named() -> None:
    """Per-stage attribution: the method is two pipelines differing in exactly one place."""
    report = build_report(
        records_with(
            [Preference.LEFT] * 4,
            right=a_result(
                "beta",
                stages=(
                    StageObservation(name="dense", wall_ms=1.0, candidates_in=0, candidates_out=5),
                    StageObservation(name="rerank", wall_ms=9.0, candidates_in=5, candidates_out=5),
                ),
            ),
            left=a_result(
                "alpha",
                stages=(
                    StageObservation(name="dense", wall_ms=1.0, candidates_in=0, candidates_out=5),
                ),
            ),
        )
    )

    assert report.stage_delta == ("rerank",)
    assert "rerank" in report.render()


def test_a_stage_configured_differently_on_the_two_sides_counts_as_a_difference() -> None:
    """Same stage, different knob, is exactly the comparison this harness is run for."""
    report = build_report(
        records_with(
            [Preference.LEFT] * 4,
            left=a_result(
                "alpha",
                stages=(
                    StageObservation(
                        name="dense",
                        wall_ms=1.0,
                        candidates_in=0,
                        candidates_out=5,
                        config={"min_score": 0.3},
                    ),
                ),
            ),
            right=a_result(
                "beta",
                stages=(
                    StageObservation(
                        name="dense",
                        wall_ms=1.0,
                        candidates_in=0,
                        candidates_out=5,
                        config={"min_score": 0.15},
                    ),
                ),
            ),
        )
    )

    assert report.stage_delta == ("dense",)


def test_records_naming_a_side_at_chance_are_not_summarised() -> None:
    """The third place the rule is enforced, and the one that guards a file read back later.

    Records arrive from disk, and a file outlives the process that wrote it — so the rule has
    to hold where the number is produced, not only where the judgement was made. Constructed
    here through the model's own back door being closed, which is why the fixture builds the
    record with a passing probe and the report is handed a doctored copy.
    """
    doctored = a_record().model_copy(update={"right_probe": an_outcome("beta", hits=1)})

    with pytest.raises(AtChanceError, match="beta"):
        build_report([doctored])


def test_records_from_two_different_comparisons_are_not_summarised_together() -> None:
    with pytest.raises(IncomparableRecordsError, match="more than one comparison"):
        build_report([a_record(), a_record(1, right=a_result("gamma"))])


def test_records_from_two_different_corpora_are_not_summarised_together() -> None:
    """As the corpus grows, last month's records stop being about the same thing."""
    grown = a_result("alpha", corpus_version=CorpusVersion(label="fixture", digest="sha256:bbb"))

    with pytest.raises(IncomparableRecordsError, match="different corpora"):
        build_report([a_record(), a_record(1, left=grown)])


def test_an_example_set_mixed_into_a_real_one_is_refused() -> None:
    with pytest.raises(IncomparableRecordsError, match="provenance"):
        build_report([a_record(), a_record(1, provenance=Provenance.EXAMPLE)])


def test_an_empty_set_of_records_produces_no_report() -> None:
    with pytest.raises(IncomparableRecordsError, match="no preference records"):
        build_report([])
