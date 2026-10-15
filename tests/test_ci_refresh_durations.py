"""The workflow that re-measures ``.test_durations`` has to measure the whole suite, or not commit.

``.github/workflows/refresh-test-durations.yml`` cannot be run before it is on ``main``, and what
it does is write a file that decides how CI shards the suite. These pin the properties that, if
they quietly changed, would make it commit a wrong measurement without anything failing: a run of
one shard would replace the file with a third of it, a failed suite must not be committed, and the
guard must sit between the measurement and the commit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

WORKFLOW = (
    Path(__file__).resolve().parents[1] / ".github" / "workflows" / "refresh-test-durations.yml"
)


def _workflow() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _steps() -> list[dict[str, Any]]:
    return _workflow()["jobs"]["refresh"]["steps"]


def _step(prefix: str) -> dict[str, Any]:
    (found,) = [step for step in _steps() if str(step.get("name", "")).startswith(prefix)]
    return found


def test_it_measures_the_whole_suite_and_never_a_shard() -> None:
    """``--store-durations`` rewrites the file from what ran, so a shard would replace it whole."""
    command = _step("measure")["run"]

    assert "--store-durations" in command
    assert "--splits" not in command
    assert "--group" not in command


def test_it_drops_the_entries_of_tests_that_no_longer_exist() -> None:
    assert "--clean-durations" in _step("measure")["run"]


def test_the_refusal_guard_sits_between_the_measurement_and_the_commit() -> None:
    names = [str(step.get("name", "")) for step in _steps()]

    measure = names.index(_step("measure")["name"])
    guard = names.index(_step("refuse")["name"])
    commit = names.index(_step("commit")["name"])

    assert measure < guard < commit
    assert "if" not in _step("commit"), "a conditional could skip past a failed guard"


def test_it_runs_monthly_and_on_demand_and_nothing_else() -> None:
    # YAML 1.1 reads the key `on` as the boolean True.
    triggers: dict[str, Any] = cast("Any", _workflow())[True]

    assert set(triggers) == {"schedule", "workflow_dispatch"}


def test_it_writes_only_where_it_must() -> None:
    workflow = _workflow()

    assert workflow["permissions"] == {}
    assert workflow["jobs"]["refresh"]["permissions"] == {"contents": "write"}


def test_it_measures_main_and_pushes_to_main() -> None:
    checkout = next(step for step in _steps() if "actions/checkout" in str(step.get("uses", "")))

    assert checkout["with"]["ref"] == "main"
    assert 'git push origin "HEAD:main"' in _step("commit")["run"]
