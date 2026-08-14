"""Blinding, storage, and the record's own refusal to describe a side that was at chance."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from manicule.evaluation.corpus import CorpusVersion
from manicule.evaluation.errors import PreferenceRecordError
from manicule.evaluation.preference import (
    Pairing,
    Preference,
    PreferenceRecord,
    PreferenceStore,
    Side,
    Slot,
    assign_slots,
    build_record,
)
from manicule.evaluation.probe import ProbeOutcome
from manicule.evaluation.queries import EvalQuery, Intent, Provenance
from manicule.evaluation.statistics import binomial_tail
from manicule.evaluation.systems import SystemResult
from tests.evaluation.fakes import an_item

if TYPE_CHECKING:
    from pathlib import Path

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


def a_result(
    label: str, *, documents: tuple[str, ...] = ("d1",), **overrides: object
) -> SystemResult:
    payload: dict[str, object] = {
        "config_label": label,
        "configuration": {"stages": ["dense"]},
        "corpus_version": VERSION,
        "items": tuple(an_item(document) for document in documents),
    }
    payload.update(overrides)
    return SystemResult.model_validate(payload)


def a_pairing(query_id: str = "q1", **overrides: object) -> Pairing:
    payload: dict[str, object] = {
        "query": EvalQuery(id=query_id, text="how does sync work", intent=Intent.HOW_DOES_X_WORK),
        "left": a_result("alpha"),
        "right": a_result("beta"),
        "slots": assign_slots(query_id),
    }
    payload.update(overrides)
    return Pairing.model_validate(payload)


def a_record(**overrides: object) -> PreferenceRecord:
    """Built through validation every time.

    ``model_copy`` would be shorter and does not re-run validators, so a fixture using it could
    produce a record the model forbids — and a suite whose fixtures can build the impossible
    proves nothing about what the model refuses.
    """
    payload: dict[str, object] = {
        "recorded_at": datetime.now(UTC),
        "query_id": "q1",
        "query_text": "how does sync work",
        "intent": Intent.HOW_DOES_X_WORK,
        "query_set": "fixture",
        "provenance": Provenance.AUTHORED,
        "left": a_result("alpha"),
        "right": a_result("beta"),
        "left_probe": an_outcome("alpha"),
        "right_probe": an_outcome("beta"),
        "slots": assign_slots("q1"),
        "preference": Preference.LEFT,
        "judge": "test",
    }
    payload.update(overrides)
    return PreferenceRecord.model_validate(payload)


def test_slot_assignment_is_stable_for_a_query_and_split_across_a_set() -> None:
    """Keyed by query id, so adding a query does not reshuffle the rest of the session.

    And it must actually vary: an assignment that always put ``left`` in slot A would leave
    position bias fully intact while looking blinded.
    """
    assert assign_slots("q1") == assign_slots("q1")
    layouts = {assign_slots(f"q{i}") for i in range(50)}
    assert layouts == {(Side.LEFT, Side.RIGHT), (Side.RIGHT, Side.LEFT)}


def test_a_different_seed_reshuffles_the_layout_and_nothing_else() -> None:
    ids = [f"q{i}" for i in range(50)]
    default = [assign_slots(i) for i in ids]
    reseeded = [assign_slots(i, seed="another") for i in ids]

    assert default != reseeded


def test_a_slot_resolves_to_whichever_side_was_behind_it() -> None:
    """The one place blinding could be undone incorrectly, and it would be invisible."""
    pairing = a_pairing(slots=(Side.RIGHT, Side.LEFT))

    assert pairing.shown_as(Slot.A).config_label == "beta"
    assert pairing.resolve(Slot.A) is Preference.RIGHT
    assert pairing.resolve(Slot.B) is Preference.LEFT


def test_a_record_may_not_be_built_for_a_side_that_was_at_chance() -> None:
    """The second of the three places the chance-level rule is enforced.

    The harness refuses to run one, and this refuses to construct one — so a record written by
    some other path still cannot exist. A stored judgment outlives the knowledge that the
    system producing it was noise.
    """
    with pytest.raises(ValueError, match="at chance"):
        build_record(
            a_pairing(),
            preference=Preference.LEFT,
            query_set="fixture",
            provenance=Provenance.AUTHORED,
            left_probe=an_outcome("alpha"),
            right_probe=an_outcome("beta", hits=1),
            judge="test",
        )


def test_a_pairing_where_a_run_was_degraded_is_kept_but_not_counted() -> None:
    """Kept, because it is evidence about the run. Not counted, because it is not a comparison."""
    record = a_record(right=a_result("beta", incomparable=("degraded leg",)))

    assert not record.admissible
    assert record.inadmissible_because == ("beta: degraded leg",)


def test_records_round_trip_through_the_store(tmp_path: Path) -> None:
    store = PreferenceStore(tmp_path / "nested" / "preferences.jsonl")

    store.append(a_record())
    store.append(a_record(query_id="q2"))

    assert len(store) == 2
    assert [record.query_id for record in store.records()] == ["q1", "q2"]


def test_an_absent_file_reads_as_no_records_rather_than_an_error(tmp_path: Path) -> None:
    assert list(PreferenceStore(tmp_path / "none.jsonl").records()) == []


def test_a_line_this_build_cannot_read_is_refused_rather_than_skipped(tmp_path: Path) -> None:
    """A reader that drops what it does not understand reports a rate over an unknown subset."""
    path = tmp_path / "preferences.jsonl"
    store = PreferenceStore(path)
    store.append(a_record())
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"schema_version": 99}\n')

    with pytest.raises(PreferenceRecordError, match="preference record"):
        list(store.records())


def test_a_naive_timestamp_is_refused() -> None:
    """A judgment whose time has no defined meaning cannot be placed against a corpus version."""
    with pytest.raises(ValueError, match="timezone-aware"):
        a_record(recorded_at=datetime.now(UTC).replace(tzinfo=None))
