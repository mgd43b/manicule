"""Running under a supervisor: the plist manicule ships, and the exit status it depends on.

Two halves, and the second is what makes the first mean anything.

**The plist is read as a plist**, not as text, so a template that is well-formed XML and not a
property list fails here rather than at ``launchctl bootstrap``. What it asserts is the three
decisions in it: start at login and keep running, throttle the respawn, and carry no credential.

**The exit status is observed from a real process.** ``ThrottleInterval`` alone does nothing —
launchd throttles every respawn, and a job that exits ``1`` on a lock somebody else holds is
still a job that restarts every thirty seconds for ever. What makes the pairing work is that
manicule distinguishes "try again later" from "this will fail identically next time", and the
only way to know it does is to run it against a held directory and look at what it exits with.
"""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from manicule.app.daemon import STOP_GRACE_S
from manicule.cli.serving import EX_TEMPFAIL
from manicule.ingest.recovery import InstanceLock

if TYPE_CHECKING:
    from collections.abc import Iterator

pytest.importorskip("fcntl", reason="this platform has no flock, so there is no refusal to test")

REPO_ROOT = Path(__file__).resolve().parents[2]

PLIST = REPO_ROOT / "tools" / "launchd" / "com.manicule.server.plist"

MARKER = "REPLACE"
"""What the template puts in every value an operator has to supply.

One word, in the value rather than in a comment beside it, so that a plist loaded without being
edited fails immediately with a path that says why — rather than starting a server pointed at
somebody's root directory.
"""

HELD = "held"
"""What the holder prints once it owns the directory. A gate, so no test waits on a clock."""

_HOLDER = """
import sys, time
from pathlib import Path
from manicule.ingest.recovery import InstanceLock
InstanceLock(Path(sys.argv[1])).acquire()
print("held", flush=True)
try:
    time.sleep(120)
except KeyboardInterrupt:
    pass
"""
"""A process that takes the directory and keeps it, as ``tests/app/test_process_exclusion.py``
does and for the same reason: the guarantee is about processes, so the holder is one."""


@pytest.fixture
def plist() -> dict[str, Any]:
    """The shipped template, parsed."""
    assert PLIST.is_file(), f"the launchd template is missing from {PLIST}"
    parsed: dict[str, Any] = plistlib.loads(PLIST.read_bytes())
    return parsed


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    made = tmp_path / "data"
    made.mkdir()
    return made


@pytest.fixture
def holder(data_dir: Path) -> Iterator[subprocess.Popen[str]]:
    """A second process holding the data directory, stopped however the test leaves it."""
    process = subprocess.Popen(  # noqa: S603 - this interpreter, a literal script
        [sys.executable, "-c", _HOLDER, str(data_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    line = process.stdout.readline().strip()
    if line != HELD:  # pragma: no cover - only on a holder that could not start
        process.kill()
        assert process.stderr is not None
        pytest.fail(f"the holder never took the lock: {process.stderr.read()}")
    try:
        yield process
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()


# --- the plist ---------------------------------------------------------------------------------


def test_the_agent_starts_at_login_and_is_kept_running(plist: dict[str, Any]) -> None:
    """Both, because either alone is a different arrangement.

    ``RunAtLoad`` without ``KeepAlive`` is a server that starts once and stays down after the
    first crash; ``KeepAlive`` without ``RunAtLoad`` is one that does not come up until something
    else happens to it.
    """
    assert plist["Label"] == "com.manicule.server"
    assert plist["RunAtLoad"] is True
    assert plist["KeepAlive"] is True


def test_the_agent_serves_over_a_socket_rather_than_stdio(plist: dict[str, Any]) -> None:
    """A supervised server has no client on its stdin, so stdio would serve nobody.

    Asserted over the argument list rather than over the whole string, so reordering the flags or
    adding one does not fail a test about which transport was chosen.
    """
    arguments = plist["ProgramArguments"]
    assert arguments[1] == "serve", arguments
    assert "--transport" in arguments, arguments
    assert "http" in arguments, arguments
    assert "--mcp-only" not in arguments, (
        "--mcp-only serves MCP alone, so the browser surface and the JSON API would be absent "
        "from a server started for an operator to use"
    )


def test_the_respawn_is_throttled_by_more_than_a_shutdown_takes(plist: dict[str, Any]) -> None:
    """The number that stops a restart becoming a spin, checked against what it has to outlast.

    manicule exits :data:`~manicule.cli.serving.EX_TEMPFAIL` while another process holds the data
    directory, and the commonest reason for that is an outgoing server that has not finished
    letting go. So the throttle has to be longer than a stop takes, or launchd's retry lands
    inside the same window and the two chase each other.

    Compared against :data:`~manicule.app.daemon.STOP_GRACE_S` rather than against a literal,
    because that constant is what ``manicule stop`` is willing to wait for and is therefore the
    project's own answer to "how long can stopping take".
    """
    throttle = plist["ThrottleInterval"]
    assert throttle > STOP_GRACE_S, (
        f"ThrottleInterval is {throttle}s and a clean stop is allowed {STOP_GRACE_S}s, so launchd "
        f"can respawn into a directory the outgoing server still holds"
    )


def test_the_agent_carries_no_credential(plist: dict[str, Any]) -> None:
    """No environment block, so there is nowhere in this file for a session to be written.

    A plist is world-readable and every process the job starts inherits its environment, so a
    token here is a token in both. manicule reads no session from the environment — that path was
    removed with the keychain — and this is the assertion that stops one being added back by
    somebody solving the "a restart signs me out" problem the wrong way.
    """
    assert "EnvironmentVariables" not in plist, (
        "the plist declares an environment block. Sessions are captured with "
        "`manicule connector login <name> --browser` and live in the server's memory; a "
        "credential here would be readable by anything that can read this file."
    )


def test_every_value_an_operator_must_supply_is_marked(plist: dict[str, Any]) -> None:
    """Three of them, and a template that started anyway would be worse than one that refuses.

    ``WorkingDirectory`` is the one worth being explicit about: launchd's default is ``/``, and a
    relative connector root would then resolve against the root of the disk — a source pointed at
    the wrong tree, silently, with a sync that appears to work.
    """
    supplied = {
        "ProgramArguments": plist["ProgramArguments"][0],
        "WorkingDirectory": plist["WorkingDirectory"],
        "StandardOutPath": plist["StandardOutPath"],
        "StandardErrorPath": plist["StandardErrorPath"],
    }
    unmarked = sorted(key for key, value in supplied.items() if MARKER not in value)
    assert unmarked == [], (
        f"these values look like real paths rather than placeholders: {unmarked}. A template with "
        f"somebody's home directory baked into it is a template that half works for everybody else."
    )


def test_the_documentation_names_the_file_that_exists() -> None:
    """The path in ``docs/deployment.md`` §6.3 is the path the file is at.

    The commonest way an install instruction rots: the file moves and the sentence does not.
    """
    deployment = (REPO_ROOT / "docs" / "deployment.md").read_text(encoding="utf-8")
    relative = PLIST.relative_to(REPO_ROOT).as_posix()
    assert relative in deployment, f"docs/deployment.md does not name {relative}"


# --- the exit status the plist depends on --------------------------------------------------------


WIDE_ENOUGH = 400
"""A terminal width no refusal here will wrap at. See :func:`_serve`."""

MANICULE = Path(sys.executable).with_name("manicule")
"""The console script beside the interpreter running the suite.

The **command an operator types**, rather than ``python -m manicule.cli.main`` — which is not the
same thing and does not even run: the module defines ``main`` and calls nothing, so ``-m`` on it
imports and exits 0. A test written that way passes the exit-status assertion by measuring the
exit status of nothing, which is precisely the failure this file exists to not have.
"""


def test_the_console_script_this_suite_drives_is_installed() -> None:
    """A missing entry point fails here, saying so, rather than as a confusing ENOENT below.

    ``uv sync`` installs it beside the interpreter. A suite that reported "No such file or
    directory" from inside a ``subprocess.run`` would send somebody looking at the test rather
    than at their environment.
    """
    assert MANICULE.is_file(), (
        f"{MANICULE} is not there, so the environment has manicule importable but not installed "
        f"as a command. Run `uv sync --all-groups`."
    )


def _serve(data_dir: Path, *extra: str, wide: bool = False) -> subprocess.CompletedProcess[str]:
    """Run the real ``manicule serve`` against ``data_dir`` and return what it did.

    Stdio, with a closed stdin, for every case here: the refusal happens before any transport
    starts, and a test that bound a port would fail when something else was listening on it.

    Args:
        data_dir: The directory to serve, passed the way an operator would set it — through the
            environment, which is also how the launchd job would.
        extra: Further arguments, for the case that needs ``--transport http``.
        wide: Configure a non-loopback bind, for the refusal that is permanent.
    """
    environment = {
        **os.environ,
        "MANICULE_DATA_DIR": str(data_dir),
        # Wide enough that Rich does not wrap the refusal. Without it the message is folded at
        # the runner's width — mid-path, since a data directory under a temporary directory is
        # long — and an assertion that the message names the directory fails on a message that
        # names it perfectly well. Pinned here rather than in a fixture for the reason
        # `tests/conftest.py` gives: a width chosen for everybody becomes the width every layout
        # assertion was silently written against.
        "COLUMNS": str(WIDE_ENOUGH),
    }
    if wide:
        environment["MANICULE_SECURITY__TRANSPORT__BIND_HOST"] = "0.0.0.0"  # noqa: S104
    return subprocess.run(  # noqa: S603 - this project's own console script
        [str(MANICULE), "serve", *extra],
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=120.0,
    )


def test_a_server_refused_because_another_holds_the_directory_asks_to_be_retried(
    data_dir: Path, holder: subprocess.Popen[str]
) -> None:
    """The status launchd sees, from a real process, with the directory genuinely held.

    ``EX_TEMPFAIL`` rather than ``1``, and the difference is the whole reason a supervised
    manicule works: this is the refusal that fixes itself, so a supervisor should come back to
    it. A refusal that will be identical in thirty seconds — a bind this configuration forbids, a
    plugin that will not import — exits ``1`` and is checked below, because a status that means
    "retry" for everything means nothing.

    The message is asserted too. A status is what a supervisor reads; the sentence is what an
    operator reads in the log the supervisor wrote, and "exit 75" on its own names no cause.
    """
    outcome = _serve(data_dir)

    assert outcome.returncode == EX_TEMPFAIL, (
        f"serve exited {outcome.returncode} with the directory held by pid {holder.pid}. "
        f"stdout={outcome.stdout!r} stderr={outcome.stderr!r}"
    )
    said = outcome.stdout + outcome.stderr
    assert str(data_dir) in said, "the refusal does not name the directory it is about"
    assert f"pid {holder.pid}" in said, "the refusal does not name the process holding it"


def test_a_refusal_that_will_not_fix_itself_is_not_asking_to_be_retried(data_dir: Path) -> None:
    """The control, without which the status above is "manicule exits 75" and not a distinction.

    An unauthenticated non-loopback bind is refused by configuration and will be refused
    identically for ever, so it exits ``1`` — the status a supervisor should treat as a job that
    is not going to come up. Nothing holds the directory here, so the two tests differ in exactly
    the thing being classified.
    """
    outcome = _serve(data_dir, "--transport", "http", wide=True)

    assert outcome.returncode == 1, (
        f"a permanently refused bind exited {outcome.returncode}; only the temporary refusal may "
        f"use {EX_TEMPFAIL}. stdout={outcome.stdout!r} stderr={outcome.stderr!r}"
    )
    assert "bind_host" in outcome.stdout + outcome.stderr, (
        "the refusal has to name the setting, or exit 1 is a status with no cause attached"
    )


def test_the_directory_is_free_the_moment_the_holder_goes(
    data_dir: Path, holder: subprocess.Popen[str]
) -> None:
    """A restart takes the lock back with nothing to clean up first.

    The half of a launchd restart the plist cannot express: the outgoing process ends, and the
    incoming one has to be able to take the directory immediately and without deleting anything.
    ``flock`` is held by an open file description, so the kernel releases it however the process
    died — and the lock *file* being left behind is a diagnostic rather than a lock.

    Waited for rather than polled: the claim is about the state **after** the holder has gone, and
    taking the lock before it has would be testing a race instead of the property.
    """
    holder.terminate()
    holder.wait(timeout=10)

    with InstanceLock(data_dir):
        pass
    with InstanceLock(data_dir):
        pass
