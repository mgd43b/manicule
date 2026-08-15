"""One writer per data directory, tested with real processes rather than two objects.

**Two lock objects in one interpreter prove almost nothing here.** ``flock`` is held by an open
file description, and the guarantee that matters — a second *process* is refused, and the kernel
takes the lock back however the first one died — is a guarantee about processes. So the holder
in every test below is a real subprocess, and the thing it holds is the production
:class:`~manicule.ingest.recovery.InstanceLock` rather than a stand-in.

The classification is checked here too, and by enumeration rather than by inspection: every
operation the command line can emit has to be named as a reader or as a writer, so an operation
added later cannot quietly inherit a default nobody looked at.
"""

from __future__ import annotations

import ast
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from manicule.app.dispatch import READ_ONLY_OPS, writes
from manicule.core.errors import InstanceLockedError
from manicule.ingest.recovery import LOCK_FILENAME, InstanceLock

if TYPE_CHECKING:
    from collections.abc import Iterator

pytest.importorskip("fcntl", reason="this platform has no flock, so there is no lock to test")
"""Skip the module where the primitive genuinely is not there, rather than fail collecting it.

``importorskip`` rather than a ``skipif`` over ``__import__``. The obvious spelling —
``skipif(not hasattr(__import__("fcntl"), "flock"))`` — evaluates the import *while deciding
whether to skip*, so on a platform with no ``fcntl`` it raises ``ModuleNotFoundError`` before
the skip can apply and takes collection of the whole file down with it. A guard that crashes on
exactly the platform it exists to spare is worse than no guard, because the failure looks like a
broken test suite rather than an unsupported primitive.
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
"""A process that takes the directory and keeps it until it is told to stop.

It uses the production lock, and it sleeps rather than exiting, because every claim below is
about what happens *while* somebody else holds the directory.

``KeyboardInterrupt`` is caught so that a ``SIGINT`` reaches the interpreter's ordinary shutdown
rather than the default handler — which is what a real ``Ctrl-C`` on a manicule command does,
and the case the release-on-graceful-shutdown test is about.
"""


def _reap(process: subprocess.Popen[str], *, expect: str) -> None:
    """Wait for a process that has already been asked to stop, and close its pipes.

    **It does not kill.** The tests that signal a holder are asserting that the *signal* ended
    it, and a helper that quietly killed anything still running would make those tests pass
    against a process that ignored the signal entirely — which is the whole property.
    """
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover - only on a holder that ignored it
        process.kill()
        process.wait(timeout=10)
        pytest.fail(f"the holder was still running ten seconds after {expect}")
    finally:
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()


def _stop(process: subprocess.Popen[str]) -> None:
    """End a holder and close its pipes.

    The pipes matter: this suite turns ``ResourceWarning`` into an error, and a ``Popen`` whose
    streams are left to the garbage collector raises one from whichever unrelated test happens
    to be running when the collector gets to it.
    """
    if process.poll() is None:
        process.kill()
    process.wait(timeout=10)
    for pipe in (process.stdout, process.stderr):
        if pipe is not None:
            pipe.close()


def _holder(data_dir: Path) -> subprocess.Popen[str]:
    """Start a process holding ``data_dir`` and wait until it actually holds it."""
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
    return process


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """A data directory of this test's own.

    A subdirectory rather than ``tmp_path`` itself, because the suite's own fixtures put a fake
    home and a working directory in there — and one assertion below is that a refused process
    left the directory holding *nothing but* the lock file.
    """
    made = tmp_path / "data"
    made.mkdir()
    return made


@pytest.fixture
def held(data_dir: Path) -> Iterator[subprocess.Popen[str]]:
    """A second process holding the data directory, stopped however the test leaves it."""
    process = _holder(data_dir)
    try:
        yield process
    finally:
        _stop(process)


# --- the exclusion itself -----------------------------------------------------------------


def test_a_second_writer_is_refused_while_another_process_holds_the_directory(
    data_dir: Path, held: subprocess.Popen[str]
) -> None:
    """The whole guarantee, across a process boundary, in the direction that matters.

    The refusal has to name enough to act on: **where** the lock is, **who** has it, and what to
    do instead. A message saying only that something is locked leaves an operator with a
    directory and no next step, which is the failure this replaces rather than a smaller version
    of it.
    """
    with pytest.raises(InstanceLockedError) as refused:
        InstanceLock(data_dir).acquire()

    message = str(refused.value)
    assert str(data_dir) in message, "the refusal names the directory it is about"
    assert f"pid {held.pid}" in message, "and the process that has it, read from the lock itself"
    assert "different data directory" in message, "and what to do instead of waiting"


def test_two_different_data_directories_are_held_at_the_same_time(tmp_path: Path) -> None:
    """The exclusion is per directory, which is what makes it usable at all.

    A lock that was really a global mutex would pass every test above it and make two unrelated
    corpora on one machine mutually exclusive.
    """
    first, second = tmp_path / "one", tmp_path / "two"
    holder = _holder(first)
    try:
        with InstanceLock(second):
            assert (first / LOCK_FILENAME).exists()
            assert (second / LOCK_FILENAME).exists()
    finally:
        _stop(holder)  # this one was never asked to stop, so killing it is the only way


def test_a_writer_that_exited_normally_leaves_the_directory_available(data_dir: Path) -> None:
    """Release on ordinary exit, observed from outside rather than asserted from inside."""
    holder = _holder(data_dir)
    holder.terminate()
    _reap(holder, expect="SIGTERM")

    with InstanceLock(data_dir):
        pass


@pytest.mark.parametrize("sent", [signal.SIGINT, signal.SIGTERM])
def test_a_signaled_writer_gives_the_directory_up_once_it_has_stopped(
    data_dir: Path, *, sent: signal.Signals
) -> None:
    """``Ctrl-C`` and a supervisor's ``SIGTERM``, which are the two ways a server is stopped.

    Waited for rather than polled: the claim is that the directory is free **after the process
    has gone**, and taking the lock before it has gone would be testing a race instead of the
    property.
    """
    holder = _holder(data_dir)
    holder.send_signal(sent)
    _reap(holder, expect=sent.name)

    with InstanceLock(data_dir):
        pass


def test_a_writer_killed_outright_needs_no_file_deleted_before_the_next_one_starts(
    data_dir: Path,
) -> None:
    """The reason this is an ``flock`` and not a PID file, stated as a test.

    ``SIGKILL`` gives a process no chance to tidy up, so the lock **file** is still there with a
    dead process's number in it. A scheme that read that number would refuse forever, or would
    have to guess when a number is old enough to ignore — and guessing wrong deletes the lock of
    a process that is running. The kernel drops the lock when the process goes, so the file being
    left behind is a diagnostic and nothing more.
    """
    holder = _holder(data_dir)
    holder.kill()
    _reap(holder, expect="SIGKILL")
    stale = data_dir / LOCK_FILENAME
    assert stale.exists(), "the file outlives the process, which is exactly the trap"
    assert stale.read_text().strip() == str(holder.pid), "still naming a process that is gone"

    with InstanceLock(data_dir):
        pass

    assert stale.exists(), "and taking it again does not require deleting anything"


# --- who has to take it -------------------------------------------------------------------


def _cli_ops() -> set[str]:
    """Every operation the command line can emit, read out of the command line itself.

    Parsed rather than listed. A list in a test is a second copy of the truth and goes stale
    silently; this one cannot, because it is derived from the calls that produce the envelopes.
    """
    import manicule.cli.main as cli  # noqa: PLC0415 - only this test needs the module

    tree = ast.parse(Path(cli.__file__).read_text(encoding="utf-8"))
    return {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"emit", "run_op", "_execute"}
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }


WRITERS: frozenset[str] = frozenset(
    {
        "auth_create_key",
        "auth_revoke_key",
        "collection_add",
        "collection_create",
        "collection_delete",
        "collection_remove",
        "collection_rename",
        "collection_update",
        "config_set",
        "connector_sidecar",
        "connector_sync",
        "document_delete",
        "document_redetect_glossary",
        "document_reindex",
        "document_reindex_stale",
        "import",
        "index_path",
        "plugin_add",
        "plugin_remove",
        "reembed_abandon",
        "reembed_cleanup",
        "reembed_plan",
        "reembed_resume",
        "reembed_start",
        "reset_index",
        "restore",
        "upgrade",
        "workspace_switch",
    }
)
"""Every command-line operation that takes the directory, written out on purpose.

**Two operations left this list when the server took over the writing**, and both left for the
same reason rather than for convenience. ``init`` writes the configuration file and pre-seeds the
cache; ``connector_login`` used to write a keychain item and now hands a session to a server over
the control socket. Neither touches the data directory, so neither takes its lock — and both had
to move, because a write command with no server refuses, and requiring a server in order to run
``manicule init`` is a cycle.

**The point of writing it out is that neither set may be the default.** The code defaults to
"writer", which is the safe direction and is what stops a command added in a hurry from indexing
beside a running sweep. But a default is also how a *reader* silently becomes a writer and starts
refusing during a sync, so this list and :data:`~manicule.app.dispatch.READ_ONLY_OPS` are
compared against the command line's own operations below: a new command lands in neither, and the
test says so by name.
"""


def test_every_command_line_operation_is_classified_as_a_reader_or_a_writer() -> None:
    """The enumeration, checked against the surface rather than against memory.

    This is the test that makes the classification maintainable. Adding a command without
    deciding what it does to the data directory is a decision by omission, and the omission is
    invisible in a diff — the command works, and the consequence shows up as a corrupted index
    on somebody else's machine months later.
    """
    ops = _cli_ops()
    assert ops, "the parse must find the operations, or this test asserts nothing at all"

    unclassified = ops - WRITERS - READ_ONLY_OPS
    assert not unclassified, (
        f"these command-line operations are in neither set: {sorted(unclassified)}. Decide "
        f"whether each one writes the data directory, then add it to WRITERS here or to "
        f"READ_ONLY_OPS in manicule.app.dispatch."
    )
    both = WRITERS & READ_ONLY_OPS
    assert not both, f"classified as both a reader and a writer: {sorted(both)}"
    assert all(writes(op) for op in WRITERS)
    assert not any(writes(op) for op in READ_ONLY_OPS)


def test_an_operation_nobody_classified_takes_the_lock_rather_than_going_without() -> None:
    """The default, pinned. Forgetting has to fail closed.

    The test above makes forgetting a red build, which is the first line of defense. This is the
    second: on the day somebody adds a command and skips the test, the behavior they get is a
    refusal they can read rather than a race they cannot.
    """
    assert writes("an_operation_that_does_not_exist_yet")


def test_every_long_lived_writer_takes_the_directory_before_it_starts_serving() -> None:
    """The four entry points that outlive one operation, checked as code rather than as prose.

    A server, a REPL and watch mode each hold one runtime for their whole life and can all
    write, so each has to take the directory on the way in — before a port is bound, before a
    banner, before the first change is noticed. Asserted structurally because the alternative is
    starting four servers in a test suite, and what would go wrong is a call being dropped in an
    edit rather than the lock failing to work.
    """
    for module in (
        "manicule/cli/serving.py",
        "manicule/cli/repl.py",
        "manicule/cli/watch.py",
        "manicule/mcp/__main__.py",
    ):
        import manicule  # noqa: PLC0415 - to locate the installed package

        source = (Path(manicule.__file__).parent.parent / module).read_text(encoding="utf-8")
        assert "runtime.acquire()" in source, (
            f"{module} opens a runtime that outlives one operation and never takes the data "
            f"directory, so a second one would start beside it"
        )


# --- what a refused process is not allowed to have done -------------------------------------


_REFUSED = """
import sys
from pathlib import Path
from manicule.core.errors import InstanceLockedError
from manicule.ingest.recovery import InstanceLock
try:
    InstanceLock(Path(sys.argv[1])).acquire()
except InstanceLockedError:
    print("refused", flush=True)
    raise SystemExit(3)
print("acquired", flush=True)
"""


def test_a_refused_writer_creates_no_database_and_runs_no_migration(
    data_dir: Path, held: subprocess.Popen[str]
) -> None:
    """Refused *before* anything, which is the whole reason the lock is taken where it is.

    A lock acquired after the schema migration, or after the recovery sweep has requeued the
    other process's in-flight documents, has protected nothing that mattered. The directory
    holds the lock file and nothing else, and that is checked by looking at the directory rather
    than by trusting the ordering of two calls.
    """
    before = sorted(p.name for p in data_dir.iterdir())

    outcome = subprocess.run(  # noqa: S603 - this interpreter, a literal script
        [sys.executable, "-c", _REFUSED, str(data_dir)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert outcome.returncode == 3, outcome.stderr
    assert outcome.stdout.strip() == "refused"
    assert sorted(p.name for p in data_dir.iterdir()) == before == [LOCK_FILENAME], (
        "a refused process left something behind, so it had already begun initializing storage "
        "by the time it found out it was not allowed to"
    )


def test_the_lock_file_is_private_to_the_account_that_created_it(data_dir: Path) -> None:
    """The directory holds an index of whatever it was pointed at; its lock is not public.

    Checked because the mode is passed to ``os.open`` as a request, and a request is subject to
    the umask — so the only honest way to know what was created is to look at what was created.
    """
    with InstanceLock(data_dir):
        mode = (data_dir / LOCK_FILENAME).stat().st_mode
    assert not mode & 0o077, f"the lock file is readable or writable by others: {mode:o}"


def test_the_holder_writes_its_own_process_id_and_nothing_else(data_dir: Path) -> None:
    """The file's contents are a diagnostic, and are checked to be only that.

    It exists so the refusal can name a holder. Anything more in it would be state somebody
    would eventually reason from, and reasoning from the file rather than from the lock is the
    mistake the whole design avoids.
    """
    with InstanceLock(data_dir):
        assert (data_dir / LOCK_FILENAME).read_text().strip() == str(os.getpid())


def test_taking_the_same_directory_twice_in_one_process_is_still_refused(data_dir: Path) -> None:
    """Belt and braces on the in-process case, which ``flock`` does not promise by itself.

    ``flock`` is held per open file description, so two ``open`` calls in one process are two
    descriptions and the second is refused. Worth pinning: were it re-entrant, a runtime that
    took the lock twice would look fine here and let a second *process* in, because the
    behavior under test would be about descriptions rather than about processes.
    """
    with InstanceLock(data_dir), pytest.raises(InstanceLockedError):
        InstanceLock(data_dir).acquire()


def test_a_released_lock_can_be_taken_again_immediately(data_dir: Path) -> None:
    """No cool-down, no timestamp, nothing to wait for. Release is release."""
    started = time.monotonic()
    for _ in range(3):
        with InstanceLock(data_dir):
            pass
    assert time.monotonic() - started < 5, "taking and releasing must not be waiting on anything"


# --- the runtime, which is what actually takes it ------------------------------------------


async def test_a_writer_runtime_takes_the_directory_and_a_reader_runtime_does_not(
    data_dir: Path, held: subprocess.Popen[str]
) -> None:
    """The wiring, rather than the lock. Both halves, because each is a way of being wrong.

    The lock working and the runtime taking it are two different claims, and the tests above
    only make the first: every one of them would still pass against a `Runtime` that never
    called `acquire`. This is the one that fails if the wiring is removed.

    The reader half is not a nicety either. A lock that stopped `doctor` running during the
    sweep somebody wanted diagnosed would be worse than no lock, and "readers are unaffected" is
    a claim that has to be checked while a writer is genuinely holding the directory.
    """
    del held
    from manicule.app.runtime import Runtime  # noqa: PLC0415 - the whole runtime, only here

    with pytest.raises(InstanceLockedError):
        async with Runtime.open(data_dir=data_dir):
            pass

    async with Runtime.open(writer=False, data_dir=data_dir):
        pass


async def test_a_runtime_gives_the_directory_back_when_it_closes(data_dir: Path) -> None:
    """Released on the way out, so the next command is not refused by a process that has gone.

    Checked from outside the runtime — by taking the lock afterwards — rather than by reading a
    field on it. A release that set the attribute and left the descriptor open would satisfy any
    assertion made from the inside.
    """
    from manicule.app.runtime import Runtime  # noqa: PLC0415

    async with Runtime.open(data_dir=data_dir):
        with pytest.raises(InstanceLockedError):
            InstanceLock(data_dir).acquire()

    with InstanceLock(data_dir):
        pass


NO_RUNTIME_COMMANDS: frozenset[str] = frozenset(
    {
        # Prints a shell script from a table in this process. Opens nothing.
        "completion",
        # Reads the loaded configuration and prints it. No data directory is touched.
        "config show",
        # Signals a server whose pid is in a file and waits for it to go. It must **not** take
        # the directory: the process it is stopping is holding it, so a `stop` that wanted the
        # lock could never stop anything.
        "stop",
        # Opens its own runtime in `cli/serving.py` and takes the directory there, which
        # `test_every_long_lived_writer_takes_the_directory_before_it_starts_serving` checks.
        # It is here rather than in WRITERS because it emits no operation for `_cli_ops` to
        # find — it is the runtime, not a call through one.
        "start",
        # The same command under the name the documentation uses. `start` and `serve` are one
        # function registered twice: `serve` says what it does, and `start` is what scripts and
        # habits already type, so neither is taken away.
        "serve",
    }
)
"""Commands that never reach ``_execute`` or ``_dispatch``, and what each one does instead.

Every other command opens a runtime through that one function, which is what makes the
classification a property of the surface rather than of a convention. These four are the
exceptions, and an exception that is not written down is indistinguishable from an oversight —
which is the entire failure mode this file exists to prevent.
"""


def _normalized(name: str) -> str:
    """One spelling for a command path and for an operation name.

    ``document reindex`` and ``document_reindex`` are the same thing said by two surfaces, and
    the comparison below is only meaningful once they look the same.
    """
    return re.sub(r"[ _-]+", "-", name.strip())


def _cli_commands() -> set[str]:
    """Every command the surface offers, from the Typer app rather than from the source."""
    import typer  # noqa: PLC0415 - only this test walks the app

    import manicule.cli.main as cli  # noqa: PLC0415 - only this test needs the module

    def walk(app: typer.Typer, prefix: str = "") -> list[str]:
        found: list[str] = []
        for command in app.registered_commands:
            name = command.name or (command.callback.__name__ if command.callback else "")
            found.append(_normalized(f"{prefix}{name}"))
        for group in app.registered_groups:
            inner = group.typer_instance
            if inner is not None:
                found.extend(walk(inner, f"{group.name} "))
        return found

    return set(walk(cli.app))


def test_every_command_is_accounted_for_by_the_classification_or_named_as_an_exception() -> None:
    """The hole the operation list alone leaves, closed.

    ``_cli_ops`` finds operations, and an operation is what a command *emits*. Three commands
    emit none — ``completion`` and ``config show`` because they open nothing, ``start`` and
    ``stop`` because one opens its own runtime and the other signals somebody else's — so a
    check built only on operations would report a complete enumeration while saying nothing
    about four of the forty-two things an operator can type.

    Comparing command names against operation names needs the mapping to be regular, and it is:
    ``document reindex`` emits ``document_reindex``. Where it is not, the command has to be
    named above with the reason, which is the point.
    """
    commands = _cli_commands()
    assert len(commands) > 30, "the walk must find the real surface, or this proves nothing"

    classified = {_normalized(op) for op in WRITERS | READ_ONLY_OPS}
    # `index` emits `index_path` when given a path and `index_status` when not; `backup` emits
    # `restore` for `--restore`. Each is one command over two operations, so its own name
    # matches neither — and both operations are classified individually above.
    aliases = {"index", "backup", "reembed-execute", "reembed-inspect"}
    exceptions = {_normalized(name) for name in NO_RUNTIME_COMMANDS}
    unaccounted = commands - classified - exceptions - aliases
    assert not unaccounted, (
        f"these commands are neither classified nor named as exceptions: {sorted(unaccounted)}. "
        f"Each one either emits an operation that belongs in WRITERS or READ_ONLY_OPS, or opens "
        f"no runtime and belongs in NO_RUNTIME_COMMANDS with the reason."
    )
