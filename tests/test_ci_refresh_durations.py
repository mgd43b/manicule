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


def test_the_token_is_checked_before_the_half_hour_measurement() -> None:
    """A missing secret found afterwards loses the measurement, which is what the first run did."""
    names = [str(step.get("name", "")) for step in _steps()]

    assert names.index(_step("check the token")["name"]) == 0
    assert names.index(_step("check the token")["name"]) < names.index(_step("measure")["name"])
    assert "DURATIONS_REFRESH_TOKEN" in _step("check the token")["run"]


def test_the_measurement_is_kept_before_anything_can_refuse_or_fail_to_deliver_it() -> None:
    names = [str(step.get("name", "")) for step in _steps()]

    keep = names.index(_step("keep the measurement")["name"])

    assert names.index(_step("measure")["name"]) < keep
    assert keep < names.index(_step("refuse")["name"])
    assert keep < names.index(_step("open or update")["name"])
    assert _step("keep the measurement")["with"]["path"] == ".test_durations"


def test_the_refusal_guard_sits_between_the_measurement_and_the_delivery() -> None:
    names = [str(step.get("name", "")) for step in _steps()]

    measure = names.index(_step("measure")["name"])
    guard = names.index(_step("refuse")["name"])
    deliver = names.index(_step("open or update")["name"])

    assert measure < guard < deliver
    assert "if" not in _step("open or update"), "a conditional could skip past a failed guard"


def test_it_runs_monthly_and_on_demand_and_nothing_else() -> None:
    # YAML 1.1 reads the key `on` as the boolean True.
    triggers: dict[str, Any] = cast("Any", _workflow())[True]

    assert set(triggers) == {"schedule", "workflow_dispatch"}


def test_it_holds_no_write_permission_of_its_own() -> None:
    """Everything that writes does so with the secret's token, so the job's own token only reads."""
    workflow = _workflow()

    assert workflow["permissions"] == {}
    assert workflow["jobs"]["refresh"]["permissions"] == {"contents": "read"}


def test_it_measures_main_and_proposes_on_a_branch_never_to_main() -> None:
    """``main`` refused the push once; the proposal goes to a branch and a pull request."""
    checkout = next(step for step in _steps() if "actions/checkout" in str(step.get("uses", "")))
    delivery = _step("open or update")["run"]

    assert checkout["with"]["ref"] == "main"
    assert "HEAD:main" not in delivery
    assert 'git push --force origin "${branch}"' in delivery
    assert "gh pr create" in delivery


def test_the_token_comes_from_the_secret_and_not_from_github_token() -> None:
    """``github.token`` starts no workflows, so the pull request's checks would never run."""
    job = _workflow()["jobs"]["refresh"]
    checkout = next(step for step in _steps() if "actions/checkout" in str(step.get("uses", "")))
    expression = "${{ secrets.DURATIONS_REFRESH_TOKEN }}"  # a workflow expression, not a credential

    assert job["env"]["GH_TOKEN"] == expression
    assert checkout["with"]["token"] == expression
    executable = [
        line
        for line in WORKFLOW.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    ]
    forbidden = ("${{ github.token }}", "secrets.GITHUB_TOKEN")
    assert not any(form in line for line in executable for form in forbidden)
