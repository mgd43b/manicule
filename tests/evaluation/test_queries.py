"""The query set format, and the four things it refuses to read.

Each refusal corresponds to a way a query set silently stops being what it claims: an unknown
schema version loaded with fields missing, an export whose extra column is dropped, two
queries sharing an id so one judgement overwrites another, and a set with no declared
provenance whose numbers are later quoted as measurements.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from manicule.evaluation.errors import QuerySetError
from manicule.evaluation.queries import (
    QUERY_SET_SCHEMA_VERSION,
    EvalQuery,
    Intent,
    Provenance,
    QuerySet,
    Thumbs,
    dump_query_set,
    load_query_set,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

EXAMPLE_SET = Path(__file__).resolve().parents[2] / "docs" / "evaluation" / "example-queries.json"


def a_set(**overrides: object) -> QuerySet:
    payload: dict[str, object] = {
        "name": "fixture",
        "provenance": Provenance.EXPORTED,
        "queries": (
            EvalQuery(id="q1", text="how does the gateway authenticate", intent=Intent.LOOKUP),
            EvalQuery(id="q2", text="ERR_TOKEN_EXPIRED", intent=Intent.EXACT_IDENTIFIER),
        ),
    }
    payload.update(overrides)
    return QuerySet.model_validate(payload)


def test_a_set_groups_its_queries_by_intent() -> None:
    """Reporting per category needs the categories recorded when the set was written."""
    grouped = a_set().by_intent()

    assert set(grouped) == {Intent.LOOKUP, Intent.EXACT_IDENTIFIER}
    assert [q.id for q in grouped[Intent.LOOKUP]] == ["q1"]


def test_a_category_nobody_wrote_queries_for_is_absent_rather_than_empty() -> None:
    """An empty row would report a coverage gap as a result."""
    assert Intent.COMPARISON not in a_set().by_intent()


def test_a_schema_version_this_build_does_not_know_is_refused() -> None:
    """A newer format read by an older build loses fields with nothing saying so."""
    with pytest.raises(ValueError, match="schema version"):
        a_set(schema_version=QUERY_SET_SCHEMA_VERSION + 1)


def test_two_queries_sharing_an_id_are_refused() -> None:
    """Preferences key on the id, so a repeat overwrites a judgement instead of adding one."""
    with pytest.raises(ValueError, match="duplicate query ids"):
        a_set(
            queries=(
                EvalQuery(id="q1", text="first"),
                EvalQuery(id="q1", text="second"),
            )
        )


def test_an_unknown_field_is_refused_rather_than_dropped() -> None:
    """An export gaining a column must fail loudly, not lose it.

    The thumbs signal is exactly the kind of field that would disappear this way — present in
    the export, absent from the model, and nothing anywhere reporting that the set being
    measured is thinner than the file it came from.
    """
    with pytest.raises(ValueError, match=r"Extra inputs"):
        EvalQuery.model_validate({"id": "q1", "text": "hello", "sentiment": "positive"})


def test_a_naive_export_timestamp_is_refused() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        a_set(exported_at="2026-01-01T00:00:00")


def test_a_set_must_declare_where_it_came_from() -> None:
    """Provenance is the field that keeps an example from being quoted as a measurement."""
    with pytest.raises(ValueError, match="provenance"):
        QuerySet.model_validate({"name": "n", "queries": [{"id": "q", "text": "t"}]})


def test_an_example_set_is_not_evidence_and_an_exported_one_is() -> None:
    assert not a_set(provenance=Provenance.EXAMPLE).is_evidence
    assert a_set(provenance=Provenance.EXPORTED).is_evidence
    assert a_set(provenance=Provenance.AUTHORED).is_evidence


def test_a_set_survives_a_round_trip_through_a_file(tmp_path: Path) -> None:
    """The writer exists so an export script has a target rather than a format to copy."""
    original = a_set(
        queries=(EvalQuery(id="q1", text="why", thumbs=Thumbs.DOWN, note="from the log"),)
    )
    path = tmp_path / "queries.json"

    dump_query_set(original, path)

    assert load_query_set(path) == original


def test_a_file_that_is_not_json_names_itself(tmp_path: Path) -> None:
    path = tmp_path / "queries.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(QuerySetError, match="not valid JSON"):
        load_query_set(path)


def test_a_missing_file_is_a_query_set_error_rather_than_an_os_error(tmp_path: Path) -> None:
    with pytest.raises(QuerySetError, match="could not read"):
        load_query_set(tmp_path / "absent.json")


def test_the_shipped_example_set_loads_and_declares_itself_an_example() -> None:
    """The example ships so the harness can be demonstrated, never so it can be quoted.

    Checked here rather than trusted: the file is the one thing in this package that could be
    mistaken for results, and the only thing stopping that is the provenance field it declares.
    """
    query_set = load_query_set(EXAMPLE_SET)

    assert query_set.provenance is Provenance.EXAMPLE
    assert not query_set.is_evidence
    assert len(query_set.queries) >= len(Intent) - 1
    assert {q.intent for q in query_set.queries} >= {
        Intent.LOOKUP,
        Intent.HOW_DOES_X_WORK,
        Intent.COMPARISON,
        Intent.EXACT_IDENTIFIER,
    }


def test_the_shipped_example_set_is_the_current_schema() -> None:
    """A checked-in file at an old version would load through compatibility nobody wrote."""
    payload: Mapping[str, object] = json.loads(EXAMPLE_SET.read_text(encoding="utf-8"))

    assert payload["schema_version"] == QUERY_SET_SCHEMA_VERSION
