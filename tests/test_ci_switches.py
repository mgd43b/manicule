"""The ``REQUIRE_*`` switches, and the two ways one of them silently stops running.

Several suites in this repository **skip** when a machine resource is absent — grammars, model
weights, BPE vocabularies, a macOS Keychain. Skipping is right on a developer's first checkout
and worthless in CI, where it means the suite reports green having checked nothing. So each has
an environment switch that turns its skips into failures, and CI sets it.

That mechanism has now failed twice, in both of the ways it can:

- **A switch that nothing reads.** The first grammar switch lived inside manicule's ``MANICULE_``
  namespace, which the test environment fixture scrubs before every test — so it was deleted
  before it could be read and the job went green having skipped everything.
- **A switch that is set for a run that does not include the file reading it.** #71 found the
  Keychain cases had never run in CI at all: the macOS job runs a *named list* of test files and
  that list did not include the one holding them. The switch was correct, declared, and armed —
  and pointed at a run that could not reach it.

Three PRs added a switch to the same job in one day and two of them conflicted over it, which is
what makes this worth a test rather than a convention. **A resolution that keeps two of three
switches looks exactly like a correct one**, and so does a file list that lost an entry.

Read out of the workflow file CI actually runs, not out of a copy of the list kept here — a
second list of switches would drift from the first and the drift would be invisible, which is
the whole defect under test.

**What this does not catch, said plainly, because a check whose name outruns what it verifies
is the thing being defended against.** A switch armed in two jobs and dropped from one of them
still passes here: the remaining job satisfies every assertion below. That matters concretely —
both bundle switches are armed on ubuntu *and* on macOS, and the macOS arming exists because
the behavior differs by platform, so losing it would silently stop checking the platform the
suite was written for. Catching it would need a per-platform expectation table, which would be
a second list of switches kept here — exactly what the paragraph above refuses. The defense
against that one remains a reviewer reading the diff.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
TESTS = REPO_ROOT / "tests"

SWITCH = re.compile(r"\bREQUIRE_[A-Z0-9_]+\b")
"""What a switch is called. The prefix is the convention and it is deliberately **outside**
manicule's ``MANICULE_`` namespace, because the environment fixture deletes everything in that
namespace before every test — which is how the first one of these was disarmed."""


def _workflow() -> dict[str, Any]:
    """The workflow, parsed. ``ruamel.yaml`` because it is the YAML reader already declared."""
    from ruamel.yaml import YAML  # noqa: PLC0415 - test-only, see tests/parsers/test_versions.py

    return cast(
        "dict[str, Any]",
        YAML(typ="safe").load(WORKFLOW.read_text(encoding="utf-8")),  # pyright: ignore[reportUnknownMemberType]
    )


def workflow_switches() -> dict[str, list[str]]:
    """Every ``REQUIRE_*`` the workflow sets, mapped to the commands it arms.

    Job-level ``env`` is read as well as step-level. A switch set for a whole job arms every
    step in it, and reading only step-level ``env`` would report such a switch as unused.
    """
    armed: dict[str, list[str]] = {}
    for job in cast("dict[str, Any]", _workflow()["jobs"]).values():
        steps = cast("list[dict[str, Any]]", job.get("steps", []))
        job_env = cast("dict[str, Any]", job.get("env") or {})
        commands = [str(step.get("run", "")) for step in steps]
        for name in job_env:
            if SWITCH.fullmatch(name):
                armed.setdefault(name, []).extend(commands)
        for step in steps:
            for name in cast("dict[str, Any]", step.get("env") or {}):
                if SWITCH.fullmatch(name):
                    armed.setdefault(name, []).append(str(step.get("run", "")))
    return armed


def suite_switches() -> dict[str, set[Path]]:
    """Every ``REQUIRE_*`` the suite reads, mapped to the files that read it.

    A switch is a **string literal**, because that is the only form the environment is ever
    asked for. A bare identifier is excluded when the same file defines it — those are the
    constants holding a switch name (``REQUIRE_BUNDLE_ENV = "REQUIRE_GRAMMAR_BUNDLE"``), and
    counting them would report a switch named after a variable that no CI job could ever set.
    """
    found: dict[str, set[Path]] = {}
    for path in sorted(TESTS.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        defined = set(re.findall(r"^([A-Z0-9_]+)\s*(?::[^=]+)?=", source, re.MULTILINE))
        for name in set(re.findall(r'"(REQUIRE_[A-Z0-9_]+)"', source)) - defined:
            found.setdefault(name, set()).add(path)
    return found


GATE = re.compile(r"^def ([a-z_]+)\(.*?(?=^def |\Z)", re.MULTILINE | re.DOTALL)
"""A top-level function in a support module, with its body, so the body can be looked at."""


def gates(support: Path) -> set[str]:
    """The functions in ``support`` that turn an absent resource into a skip or a failure.

    Importing *something* from a support module is not the same as being governed by its
    switch, and the difference is not academic: ``tests/test_embedding_embedder.py`` imports a
    synthetic-model builder from the same file that holds the model gate, and no amount of
    ``REQUIRE_EMBEDDING_MODELS`` changes what it does. A coarser rule reported that as a hole
    in CI, and "fixing" it would have added a file to a job's list for no reason — which is
    the careless half of the same mistake this module exists to catch.

    So a gate is a function that actually calls ``pytest.skip`` or ``pytest.fail``. Those are
    the ones whose behavior the switch changes.
    """
    source = support.read_text(encoding="utf-8")
    return {
        match.group(1)
        for match in GATE.finditer(source)
        if "pytest.skip" in match.group(0) or "pytest.fail" in match.group(0)
    }


def affected_modules(readers: set[Path]) -> set[Path]:
    """The test modules a switch actually governs.

    A switch is usually read by a *support* module rather than by a test file — the support
    module is what turns a skip into a failure — so "which files does this arm" is the set of
    test modules that call one of its gates, not the file the literal appears in. Getting that
    wrong is precisely #71's defect from the other side.
    """
    modules: set[Path] = {path for path in readers if path.name.startswith("test_")}
    named = {name for path in readers if path not in modules for name in gates(path)}
    if not named:
        return modules
    for path in TESTS.rglob("test_*.py"):
        source = path.read_text(encoding="utf-8")
        if any(re.search(rf"\b{name}\b", source) for name in named):
            modules.add(path)
    return modules


def selected_by(command: str, module: Path) -> bool:
    """Whether ``command`` would collect ``module``.

    A ``pytest`` invocation naming no path under ``tests/`` runs everything, which is the
    ubuntu matrix's whole-suite run. One that names paths runs only what is under them.
    """
    named = [Path(word) for word in command.split() if word.startswith("tests/")]
    if not named:
        return True
    relative = module.relative_to(REPO_ROOT)
    return any(relative == path or path in relative.parents for path in named)


def test_every_switch_the_workflow_arms_is_one_the_suite_reads() -> None:
    """A switch nothing reads is a job that believes it is enforcing something.

    It costs nothing and reports nothing, which is worse than its absence: the workflow says
    the suite is armed and the suite has never heard of it.
    """
    unread = sorted(set(workflow_switches()) - set(suite_switches()))

    assert unread == [], (
        f"{unread} are set by .github/workflows/ci.yml and read by no test. Either the suite "
        f"stopped reading them or they were renamed on one side only"
    )


def test_every_switch_the_suite_reads_is_armed_by_the_workflow() -> None:
    """The direction that goes silently green.

    A suite whose switch nobody sets skips in CI exactly as it does on a laptop, and a skipped
    conformance suite certifies nothing. This is the assertion a careless merge conflict
    resolution trips over: dropping one of three switches from the same job leaves a workflow
    that parses, a job that passes, and a guard that no longer runs.
    """
    unarmed = sorted(set(suite_switches()) - set(workflow_switches()))

    assert unarmed == [], (
        f"{unarmed} turn a suite's skips into failures and no CI job sets them, so that suite "
        f"skips in CI exactly as it does on a laptop. Arm them in .github/workflows/ci.yml"
    )


def test_each_switch_is_armed_for_a_run_that_actually_collects_what_it_governs() -> None:
    """#71's defect, as a test: the switch was right and the file list could not reach it.

    ``REQUIRE_KEYCHAIN`` was declared, exported, read, and set on the only platform where the
    behavior exists — and the job ran a named list of test files that did not include the one
    holding the Keychain cases. Every part of the mechanism was correct except the one that
    decided whether the code ran at all, and nothing about the job's output said so.
    """
    armed = workflow_switches()
    wrong: list[str] = []
    for name, readers in sorted(suite_switches().items()):
        commands = armed.get(name, [])
        for module in sorted(affected_modules(readers)):
            if not any(selected_by(command, module) for command in commands):
                wrong.append(f"{name} does not reach {module.relative_to(REPO_ROOT)}")

    assert wrong == [], (
        f"{wrong}. A switch that turns skips into failures only does so for a run that "
        f"collects the tests it governs; a named file list that omits one leaves those cases "
        f"skipping in CI with the job green"
    )
