"""The link check, and the two ways it can report success having checked nothing.

The `docs links` job is the only thing standing between this repository and a renumbered
heading silently invalidating every reference to it. It went red three times in an hour, and
the reason was not a broken link — the checker failed while downloading *itself* from a GitHub
release, and the step that checks links was skipped. #85 removed exactly that fragility from
the image build; this is the same defect in the same file.

Failing loudly, as it did, is the *good* outcome. The bad one is one configuration change away
and is what this module defends:

- **A checker that could not run, reporting success.** `continue-on-error` on the download, or
  on the run, and the job goes green having verified nothing. Nobody would learn that the links
  were unchecked, because a green tick is exactly what a working link check looks like.
- **A checker that ran and checked nothing.** Measured rather than assumed: lychee pointed at a
  glob matching no files prints one `[WARN]` line and **exits 0**. So a renamed directory, or an
  `--exclude` that grew, produces a passing job that has established nothing at all.

Both are the same failure this repository keeps having — a check whose name is broader than
what it verified — and neither is caught by asserting on an exit status, which is why the gate
is the *report* instead. A report is an artefact only a real run produces.

The tests below come in two halves: what `tools/check_link_report.py` decides, exercised
against reports rather than by reading it, and whether the workflow is still wired to it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def workflow() -> dict[str, Any]:
    """The workflow CI actually runs, parsed. ``ruamel.yaml`` as in test_ci_switches.py."""
    from ruamel.yaml import YAML  # noqa: PLC0415 - test-only, see tests/test_ci_switches.py

    return cast(
        "dict[str, Any]",
        YAML(typ="safe").load(WORKFLOW.read_text(encoding="utf-8")),  # pyright: ignore[reportUnknownMemberType]
    )


def docs_job() -> dict[str, Any]:
    """The job that checks the documentation's links."""
    return cast("dict[str, Any]", workflow()["jobs"]["docs"])


def docs_steps() -> list[dict[str, Any]]:
    """Its steps, in order."""
    return cast("list[dict[str, Any]]", docs_job()["steps"])


def report(**overrides: Any) -> dict[str, Any]:
    """A lychee report, shaped like the real thing and healthy unless told otherwise.

    The keys and their spellings are copied from an actual ``--format json`` run against this
    repository rather than invented, so a test passing here is a test about the file lychee
    writes.
    """
    base: dict[str, Any] = {
        "total": 281,
        "unique": 66,
        "successful": 154,
        "excludes": 127,
        "errors": 0,
        "error_map": {},
    }
    return base | overrides


# --- what the checker decides ------------------------------------------------------------


def test_a_healthy_report_passes() -> None:
    """The everyday case, so that the tests below are about the guard rather than the fixture."""
    from tools.check_link_report import check  # noqa: PLC0415 - a CI script, not runtime

    assert check(report()) == []


def test_a_run_that_found_no_links_at_all_is_a_failure() -> None:
    """lychee exits 0 here, which is the whole problem.

    A glob that matches nothing, a renamed docs directory, a `workingDirectory` that moved: all
    of them produce this report, and all of them are indistinguishable from a clean run if the
    exit status is the only thing consulted.
    """
    from tools.check_link_report import EMPTY_REPORT, check  # noqa: PLC0415 - a CI script

    problems = check(report(total=0, unique=0, successful=0, excludes=0))

    assert len(problems) == 1
    assert EMPTY_REPORT in problems[0]


def test_a_run_that_verified_nothing_is_a_failure_even_with_links_in_it() -> None:
    """The subtler half, and the one an errors-only check cannot see.

    If every link is excluded the report looks busy — a healthy ``total``, no errors — and not
    one link was resolved. An `--exclude` that grew to cover the repository would land here, and
    would otherwise be a permanent green tick over an unchecked corpus.
    """
    from tools.check_link_report import NOTHING_VERIFIED, check  # noqa: PLC0415 - a CI script

    problems = check(report(successful=0, excludes=281, errors=0))

    assert len(problems) == 1
    assert NOTHING_VERIFIED in problems[0]


def test_broken_links_are_named_with_the_document_and_the_reason() -> None:
    """A count sends somebody to the log; a name sends them to the line.

    The upstream exit status says only that there were errors, so this is the only place the
    two useful strings — which document, and why — are put in front of a reader.
    """
    from tools.check_link_report import check  # noqa: PLC0415 - a CI script, not runtime

    problems = check(
        report(
            successful=1,
            errors=1,
            error_map={
                "docs/parsing.md": [
                    {
                        "url": "file:///repo/docs/storage.md#no-such-heading",
                        "status": {"text": "Cannot find fragment"},
                        "span": {"line": 412, "column": 3},
                    }
                ]
            },
        )
    )

    assert len(problems) == 1
    assert "docs/parsing.md:412" in problems[0]
    assert "Cannot find fragment" in problems[0]
    assert "no-such-heading" in problems[0]


def test_a_report_full_of_broken_links_is_not_also_reported_as_unverified() -> None:
    """Two true statements, one of which buries the other.

    A run where every link is broken did verify none of them, and saying so would put a sentence
    about exclusions above the list of what to fix. The unverified clause exists for the run that
    would otherwise *pass*, so it stays quiet whenever something else already fails.
    """
    from tools.check_link_report import NOTHING_VERIFIED, check  # noqa: PLC0415 - a CI script

    problems = check(report(successful=0, excludes=0, errors=2, error_map={}))

    assert len(problems) == 1
    assert NOTHING_VERIFIED not in problems[0]


def test_a_missing_report_says_the_check_never_ran(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """ "It never ran" must not read as "it found nothing wrong".

    This is what makes the guard undisableable: the report is an artefact only a completed run
    produces, so a download that failed — however quietly — cannot satisfy it.

    The message is asserted rather than just the exit status, and that is not fussiness. Reading
    a file that is not there fails on its own, so the status alone stays correct with the check
    deleted and the operator is handed ``[Errno 2] No such file or directory`` — which reads as
    a broken script. The whole value here is telling somebody that *nothing about this
    repository's links has been established*, which is a different sentence from "cannot open a
    file" and is the one that gets the job looked at.
    """
    from tools.check_link_report import main  # noqa: PLC0415 - a CI script, not runtime

    assert main([str(tmp_path / "absent.json")]) == 1

    assert "did not run to completion" in capsys.readouterr().err


def test_the_checker_exits_non_zero_on_a_report_it_rejects(tmp_path: Path) -> None:
    """The exit status is what the workflow reads, so the decision has to reach it."""
    from tools.check_link_report import main  # noqa: PLC0415 - a CI script, not runtime

    healthy = tmp_path / "good.json"
    healthy.write_text(json.dumps(report()), encoding="utf-8")
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps(report(total=0, successful=0, excludes=0)), encoding="utf-8")

    assert main([str(healthy)]) == 0
    assert main([str(empty)]) == 1


# --- whether the workflow is still wired to it -----------------------------------------------


def test_the_report_the_job_writes_is_the_one_it_checks() -> None:
    """Two paths in two steps, and nothing but convention holding them equal.

    Point them at different files and the checker reads a report that is not there, which fails —
    loudly, and for a reason that reads like a bug in the checker rather than a typo in the
    workflow. Caught here instead, where the message says which.
    """
    commands = [str(step.get("run", "")) for step in docs_steps()]
    written = [
        word
        for command in commands
        for word in command.split()
        if command.count("--output") and word.endswith(".json")
    ]
    checked = [
        word
        for command in commands
        if "check_link_report.py" in command
        for word in command.split()
        if word.endswith(".json")
    ]

    assert written, "no step of the docs job writes a machine-readable lychee report"
    assert checked, "no step of the docs job runs tools/check_link_report.py over a report"
    assert set(checked) <= set(written), (
        f"the docs job checks {checked} and writes {written}. The report being asserted on is "
        f"not the report the run produces"
    )


def test_nothing_in_the_docs_job_is_allowed_to_fail_quietly() -> None:
    """The one-line change that would turn this job into a permanent green tick.

    ``continue-on-error`` on the install, or on the run, and a checker that never executed
    reports success — which is indistinguishable from a working link check and is the reason
    anybody looked at this job at all: it failed *loudly*, and that is the only thing that made
    the outage visible.
    """
    job = docs_job()
    offenders = [
        step.get("name") or step.get("uses") or "<unnamed step>"
        for step in docs_steps()
        if step.get("continue-on-error")
    ]

    assert not job.get("continue-on-error"), (
        "the docs job is marked continue-on-error, so a link check that could not run reports "
        "success and nobody learns the documentation was never checked"
    )
    assert offenders == [], (
        f"{offenders} are marked continue-on-error in the docs job. A failure there stops being "
        f"a failure, and this job's whole value is that it fails when it cannot check"
    )


def test_the_check_is_the_last_word_even_when_lychee_itself_failed() -> None:
    """Unconditional, so the artefact is what decides rather than the step before it.

    Without this the assertion is skipped exactly when it is most useful — after a run that
    exited non-zero, or one somebody made non-fatal — and the job's verdict comes from the step
    whose reliability is in question.
    """
    guarded = [
        str(step.get("if", ""))
        for step in docs_steps()
        if "check_link_report.py" in str(step.get("run", ""))
    ]

    assert guarded, "the docs job no longer runs the link report check at all"
    assert all("cancelled()" in condition for condition in guarded), (
        f"the link report check runs under {guarded}, so it is skipped whenever an earlier step "
        f"failed — including the download whose failure is the reason it exists"
    )


def test_the_link_checker_is_pinned_and_the_pin_is_what_gets_cached() -> None:
    """A floating version cannot be a cache key, and a mismatched one is a stale binary.

    The version has to be one string used in three places — the download URL, the cache key and
    the assertion before the run — or the cache holds one checker while the job believes it has
    another.
    """
    job = docs_job()
    version = str(cast("dict[str, Any]", job.get("env") or {}).get("LYCHEE_VERSION", ""))
    keys = [
        str(cast("dict[str, Any]", step.get("with") or {}).get("key", ""))
        for step in docs_steps()
        if "cache" in str(step.get("uses", ""))
    ]
    downloads = [
        str(step.get("run", ""))
        for step in docs_steps()
        if "releases/download" in str(step.get("run", ""))
    ]

    assert version.startswith("v"), (
        "the docs job does not pin LYCHEE_VERSION, so the checker it installs can change "
        "without a diff and the cache cannot be keyed on it"
    )
    assert keys, "the link checker is downloaded on every run and never cached"
    assert all("LYCHEE_VERSION" in key for key in keys), (
        f"the link checker's cache keys are {keys} and none names the pinned version, so a "
        f"version bump would restore the previous checker and skip installing the new one"
    )
    assert downloads, "no step of the docs job downloads the link checker"
    assert all("LYCHEE_VERSION" in command for command in downloads), (
        "the download does not use the pinned version, so the cache key and the binary it "
        "describes can disagree"
    )
