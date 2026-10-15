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

    assert names.index(_step("check the token before")["name"]) == 0
    assert names.index(_step("check the token before")["name"]) < names.index(
        _step("measure")["name"]
    )
    assert "DURATIONS_REFRESH_TOKEN" in _step("check the token before")["run"]


def test_the_token_is_proved_able_to_write_before_the_half_hour_measurement() -> None:
    """A role check cannot see a token's scope; only the server refusing a push can."""
    names = [str(step.get("name", "")) for step in _steps()]

    push_check = names.index(_step("check the token can push")["name"])

    assert names.index(_step("measure")["name"]) > push_check
    assert "push --dry-run --force" in _step("check the token can push")["run"], (
        "the delivery force-pushes, so the check must too or a leftover branch fails it"
    )
    assert "chore/refresh-test-durations" in _step("check the token can push")["run"]


def test_the_measurement_is_kept_before_anything_can_refuse_or_fail_to_deliver_it() -> None:
    names = [str(step.get("name", "")) for step in _steps()]

    keep = names.index(_step("keep the measurement")["name"])

    assert names.index(_step("measure")["name"]) < keep
    assert keep < names.index(_step("refuse")["name"])
    assert keep < names.index(_step("open or update")["name"])
    kept = _step("keep the measurement")["with"]
    assert kept["path"] == ".test_durations"
    # A dotfile is skipped by default, and a missing file only warns by default: together they
    # make this step succeed having kept nothing.
    assert kept["include-hidden-files"] is True
    assert kept["if-no-files-found"] == "error"


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
    assert 'push --force origin "${branch}"' in delivery
    assert "gh pr create" in delivery


EXPRESSION = "${{ secrets.DURATIONS_REFRESH_TOKEN }}"  # a workflow expression, not a credential
TOKEN_STEPS = ("check the token before", "check the token can push", "open or update")


def test_the_token_comes_from_the_secret_and_not_from_github_token() -> None:
    """``github.token`` starts no workflows, so the pull request's checks would never run."""
    for prefix in TOKEN_STEPS:
        assert _step(prefix)["env"]["GH_TOKEN"] == EXPRESSION
    executable = [
        line
        for line in WORKFLOW.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    ]
    forbidden = ("${{ github.token }}", "secrets.GITHUB_TOKEN")
    assert not any(form in line for line in executable for form in forbidden)


def test_the_token_is_visible_to_the_three_steps_that_use_it_and_to_nothing_else() -> None:
    """The measurement runs every test and every dependency for half an hour.

    Anything in the job's environment or in git's stored credentials is readable by all of it, so
    the token is not in the job's ``env``, is not persisted by checkout, and appears only in the
    steps that talk to GitHub.
    """
    job = _workflow()["jobs"]["refresh"]
    checkout = next(step for step in _steps() if "actions/checkout" in str(step.get("uses", "")))
    holders = [step["name"] for step in _steps() if "GH_TOKEN" in (step.get("env") or {})]

    assert "env" not in job
    assert checkout["with"]["persist-credentials"] is False
    assert "token" not in checkout["with"]
    assert holders == [_step(prefix)["name"] for prefix in TOKEN_STEPS]
    mentions = [
        line
        for line in WORKFLOW.read_text(encoding="utf-8").splitlines()
        if "secrets.DURATIONS_REFRESH_TOKEN" in line and not line.lstrip().startswith("#")
    ]
    assert len(mentions) == len(TOKEN_STEPS), "the secret is referenced outside its three steps"


def test_an_unchanged_measurement_closes_a_proposal_that_is_still_open() -> None:
    """Auto-merge stays requested on an open PR, which would later merge an older file."""
    delivery = _step("open or update")["run"]
    unchanged = delivery[: delivery.index("exit 0")]

    assert "gh pr close" in unchanged
    assert unchanged.index("branch=") < unchanged.index("git diff --quiet")


def test_only_proposals_from_this_repository_are_looked_up_or_closed() -> None:
    """``gh pr list --head`` matches the branch name alone, so a fork's PR could pass for ours."""
    delivery = "\n".join(
        line
        for line in _step("open or update")["run"].splitlines()
        if not line.lstrip().startswith("#")
    )

    assert delivery.count("gh pr list") == 1, "every lookup goes through the one filtered helper"
    assert "select(.isCrossRepository == false)" in delivery
    assert "--base main" in delivery
    assert delivery.count("our_proposals") >= 3, (
        "defined once, used by the close loop and the check"
    )
