"""The shard count, which is one decision written in two places that cannot see each other.

The test job splits the suite across runners with ``pytest-split``: the job's ``pytest`` command
passes ``--splits N``, and the matrix lists the groups to run. Nothing in GitHub Actions ties
those together — ``strategy:`` cannot read the ``env`` context, so the count cannot be named once
and referenced twice — and the two can disagree in either direction:

- ``--splits`` **larger than the matrix**: the extra groups are never dispatched. ``--splits 4``
  against a three-entry matrix runs three quarters of the suite, and all six jobs pass. This is
  the one that matters, and it is the ordinary result of raising the count for speed and
  forgetting the list — or of a merge that took one side of each.
- ``--splits`` **smaller than the matrix**: ``--group`` is 1-indexed and bounded, so the surplus
  entry fails outright rather than going green. Caught here anyway, because a job that cannot
  run is still cheaper to find in a test than in a CI log.

So this is the same defect ``tests/test_ci_switches.py`` exists for, arriving through the split
rather than through a switch: a job that reports green having run less than it claims.

**Read out of the parsed workflow, never out of its text.** ``tests/test_ci_grammar_cache.py``
records two guards that were caught passing on prose — a flag discussed in a comment kept a
substring check true after the flag itself was deleted — and the commentary around this job
names ``--splits 4`` in the course of explaining what it would break. A search over the raw YAML
would find that and be satisfied by it. The safe loader drops comments, so reading
``step["run"]`` is prose-free by construction rather than by care.

**What this does not catch, said plainly.** That ``.test_durations`` is *accurate*. Existence and
shape are checked below; freshness is not, and cannot be cheaply — it would mean collecting the
suite and comparing node IDs, and the answer would still only be "these tests were once measured"
rather than "these measurements are still true". The failure mode that leaves is bounded and
loud in the right way: an unmeasured test is given the average duration rather than dropped, so
stale durations cost *balance* and can never cost coverage. A shard drifting slow says so in the
``--durations=25`` output the job already prints.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
DURATIONS = REPO_ROOT / ".test_durations"


def _workflow() -> dict[str, Any]:
    """The workflow, parsed. ``ruamel.yaml`` because it is the YAML reader already declared."""
    from ruamel.yaml import YAML  # noqa: PLC0415 - test-only, see tests/test_ci_switches.py

    return cast(
        "dict[str, Any]",
        YAML(typ="safe").load(WORKFLOW.read_text(encoding="utf-8")),  # pyright: ignore[reportUnknownMemberType]
    )


def suite_job() -> dict[str, Any]:
    """The matrix job that runs the suite."""
    jobs = cast("dict[str, Any]", _workflow()["jobs"])
    assert "test" in jobs, (
        "no `test` job in .github/workflows/ci.yml. If the suite is run by a job under another "
        "name now, this module is describing something gone"
    )
    return cast("dict[str, Any]", jobs["test"])


def split_command() -> list[str]:
    """The job's ``pytest`` invocation, as words.

    The step is found by the flag it carries rather than by its ``name``, so renaming the step
    cannot quietly turn this module into a check of nothing.
    """
    steps = cast("list[dict[str, Any]]", suite_job()["steps"])
    commands = [shlex.split(str(step.get("run", ""))) for step in steps]
    sharded = [words for words in commands if "--splits" in words]
    assert len(sharded) == 1, (
        f"expected exactly one `--splits` command in the test job, found {len(sharded)}. The "
        f"shard count is asserted against the matrix below, and neither zero commands nor two "
        f"of them leaves that assertion meaning anything"
    )
    return sharded[0]


def flag(command: list[str], name: str) -> str:
    """The value of ``name`` in ``command``, in either the ``--flag value`` or ``--flag=value``."""
    for index, word in enumerate(command):
        if word == name and index + 1 < len(command):
            return command[index + 1]
        if word.startswith(f"{name}="):
            return word.split("=", 1)[1]
    message = f"no {name} in the test job's pytest command: {' '.join(command)}"
    raise AssertionError(message)


def test_the_split_count_and_the_shard_matrix_are_the_same_number() -> None:
    """The disagreement that runs part of the suite and reports green.

    ``--splits`` decides how many pieces the suite is cut into; the matrix decides how many of
    those pieces anybody runs. Raising one without the other is not a broken workflow — it is a
    working one that silently stops running the remainder.
    """
    shards = cast("list[int]", suite_job()["strategy"]["matrix"]["shard"])
    splits = int(flag(split_command(), "--splits"))

    assert splits == len(shards), (
        f"the test job splits the suite into {splits} groups and its matrix runs {len(shards)} "
        f"of them ({shards}). Every group must be dispatched by exactly one matrix entry, or "
        f"the shards that are missing simply never run and every job still passes"
    )


def test_every_group_is_run_exactly_once() -> None:
    """The count agreeing is not the same as the groups being the right ones.

    ``[1, 2, 2]`` is three entries against three splits and runs two thirds of the suite while
    doing a third of it twice. ``[0, 1, 2]`` is three entries that pass the count check and
    fail at run time, because ``--group`` is 1-indexed. Both are what a hand-edited list looks
    like after somebody changed the number.
    """
    shards = cast("list[int]", suite_job()["strategy"]["matrix"]["shard"])
    splits = int(flag(split_command(), "--splits"))

    assert sorted(shards) == list(range(1, splits + 1)), (
        f"the shard matrix is {shards}; for --splits {splits} it has to be "
        f"{list(range(1, splits + 1))}. `--group` is 1-indexed, so a duplicate runs one shard "
        f"twice and leaves another unrun, and a 0 fails the job outright"
    )


def test_the_durations_the_split_balances_on_are_committed() -> None:
    """Without the file the split still works, and stops being worth doing.

    ``pytest-split`` gives an unmeasured test the average duration, so with no durations at all
    every test looks identical and the split degrades to equal *test counts*. For this suite
    that is close to the worst available answer: the 25 slowest tests are roughly half the run
    time, so one shard inherits the parser corpus round-trips and the other two finish early.
    Nothing fails, CI just quietly goes back to being as slow as the slowest third.
    """
    assert DURATIONS.is_file(), (
        f"{DURATIONS.name} is missing. The test job balances its shards on it; without it the "
        f"split falls back to equal test counts, which for this suite means one shard gets the "
        f"parser corpus and the sharding buys nothing. Regenerate it with "
        f"`uv run pytest --store-durations`"
    )

    parsed: object = json.loads(DURATIONS.read_text(encoding="utf-8"))

    assert isinstance(parsed, dict), (
        f"{DURATIONS.name} is not an object of test id to seconds, so pytest-split will read no "
        f"durations from it and balance on nothing"
    )
    recorded = cast("dict[str, object]", parsed)
    assert recorded, (
        f"{DURATIONS.name} is empty, which pytest-split treats exactly as a missing file: every "
        f"test gets the same assumed duration and the split becomes an equal-count one"
    )
    assert all(isinstance(seconds, (int, float)) for seconds in recorded.values()), (
        f"{DURATIONS.name} has non-numeric durations in it"
    )
