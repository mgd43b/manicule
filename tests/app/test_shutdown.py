"""Stopping a served manicule: the order, the bound, and nothing left running afterwards.

Four things stop, and the sequence is the subject rather than the fact that each one does:
the scheduler, so no new sync starts; the ingest stages of whatever was running, which drain
within ``ingest.shutdown_grace_s``; the control socket, which waits for the write commands in
flight; and last the MCP sessions and the HTTP server, which one lifespan owns together.

**The order is observed, not read.** :class:`Recording` puts each step's name in a list as it
closes, so a test asserts on the sequence that happened. A test that read the source, or that
asserted each part closed without saying when, would pass against an implementation that closed
them in the order that happens to be convenient — which is the order they were closed in before
any of this, and is why a ``SIGTERM`` used to skip three of them entirely.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, override

import pytest

from manicule.api.serve import DRAIN_SECONDS
from manicule.app import control
from manicule.app.daemon import STOP_GRACE_S
from manicule.app.served import ControlHandler, Scheduler, Serving
from manicule.app.service import ApplicationService
from manicule.cli import serving as cli_serving
from manicule.connectors.sessions import SessionVault
from tests.app.fakes import FakeBackend

if TYPE_CHECKING:
    from collections.abc import Iterator

WIDE_ENOUGH = 400
"""A terminal width the refusals below will not wrap at, so a path stays on one line."""


@pytest.fixture
def socket_path(tmp_path: Path) -> Iterator[Path]:
    """A control socket of this test's own, removed however the test ends."""
    path = control.socket_path(tmp_path / "data")
    yield path
    path.unlink(missing_ok=True)


class Recording(Scheduler):
    """A scheduler that says when it was closed, and how long it took.

    A subclass of the real one rather than a stand-in, so the thing being ordered is the thing
    that ships. What is added is a note in a shared list; what is *not* added is any change to
    what ``aclose`` does.
    """

    def __init__(
        self, service: ApplicationService, order: list[str], *, delay: float = 0.0
    ) -> None:
        super().__init__(service, {}, sleep=asyncio.sleep)
        self._order = order
        self._delay = delay

    @override
    async def aclose(self) -> None:
        # Before the sleep, because what is being recorded is when this step *began* — the sleep
        # stands in for a sync draining its ingest stages, which is time spent inside this step
        # rather than after it.
        self._order.append("scheduler")
        if self._delay:
            await asyncio.sleep(self._delay)
        await super().aclose()
        self._order.append("stages drained")


class RecordingSocket(control.ControlServer):
    """The real control server, noting when it was closed."""

    def __init__(self, path: Path, handler: control.Handler, order: list[str]) -> None:
        super().__init__(path, handler)
        self._order = order

    @override
    async def aclose(self) -> None:
        self._order.append("control socket")
        await super().aclose()


def _serving(socket: Path, order: list[str], *, delay: float = 0.0) -> Serving:
    """A ``Serving`` over a fake backend, with both halves recording."""
    service = ApplicationService(FakeBackend())
    return Serving(
        server=RecordingSocket(socket, ControlHandler(service, SessionVault()), order),
        scheduler=Recording(service, order, delay=delay),
    )


# --- the order ----------------------------------------------------------------------------------


async def test_the_scheduler_and_its_stages_close_before_the_control_socket(
    socket_path: Path,
) -> None:
    """Steps one to three, in the sequence they happened rather than in the sequence intended.

    The scheduler first because a loop that is still running would start a sync into a process
    that is stopping. The socket after the stages, because a proxied command still running is
    work an operator typed and is watching — and because closing the socket first would leave it
    with nowhere to answer.
    """
    order: list[str] = []
    serving = _serving(socket_path, order)

    await serving.astart()
    await serving.aclose()

    assert order == ["scheduler", "stages drained", "control socket"], order


async def test_the_transport_is_still_answering_while_the_first_three_close(
    socket_path: Path,
) -> None:
    """The reason the transport is a task rather than an awaited call.

    A proxied command in flight has to be able to finish, and a client mid-request has to be able
    to get its answer, *while* the scheduler and the socket are closing. So the HTTP server is
    stopped last, and this is the assertion that the first three steps do not depend on it having
    stopped first: they complete on their own, against a transport that is still notionally up.

    Expressed as "the close returns" rather than by driving a request, because a request would be
    testing uvicorn's shutdown and this is testing the order around it —
    ``tests/api/test_both_surfaces.py`` drives the transport.
    """
    order: list[str] = []
    serving = _serving(socket_path, order, delay=0.05)

    await serving.astart()
    async with asyncio.timeout(30):
        await serving.aclose()

    assert order[-1] == "control socket"


async def test_nothing_outlives_the_call(socket_path: Path) -> None:
    """The assertion #139 made, remade with the surfaces that have been added since.

    Counted rather than inspected: every task alive before is alive after, and none of the ones
    started in between is. A scheduler loop left running would be a second writer with no lock; a
    control connection left running would be a write command answering into a closed socket.
    """
    order: list[str] = []
    serving = _serving(socket_path, order)

    before = {task.get_name() for task in asyncio.all_tasks()}
    await serving.astart()
    # A connection in flight, so the count is taken with something to leave behind.
    answered = await control.connect(
        socket_path,
        control.Invoke(op="config_set", arguments={"key": "rag.profile", "value": "fast"}),
        on_progress=lambda _: None,
    )
    assert answered["ok"] is True, answered
    await serving.aclose()

    after = {task.get_name() for task in asyncio.all_tasks()}
    assert after - before == set(), f"these tasks outlived the shutdown: {sorted(after - before)}"


# --- the bound ----------------------------------------------------------------------------------


def test_the_http_drain_fits_inside_the_grace_stop_is_willing_to_wait() -> None:
    """Two constants that have to be in a relationship, asserted as one rather than as two numbers.

    ``manicule stop`` waits :data:`~manicule.app.daemon.STOP_GRACE_S` and then reports that it did
    not get a clean exit — deliberately without escalating, because a server killed mid-write is
    how a half-written index happens. So the transport's own drain has to fit inside that, or the
    ordinary case is a ``stop`` that reports a timeout on a server that was shutting down
    correctly, and an operator whose next move is ``kill -9``.
    """
    assert DRAIN_SECONDS < STOP_GRACE_S, (
        f"the HTTP drain is {DRAIN_SECONDS}s and `manicule stop` waits {STOP_GRACE_S}s, so a "
        f"server draining normally is reported as one that would not stop"
    )


def test_the_drain_is_bounded_at_all() -> None:
    """uvicorn's default is to wait for ever, and an MCP client may hold a connection for ever.

    Stated as its own assertion because "it is five" and "it is not unbounded" fail differently:
    the second is the property, and passing ``None`` — which is uvicorn's default and reads as
    "no limit configured" rather than as a change — would satisfy any comparison written the
    other way round.
    """
    assert isinstance(DRAIN_SECONDS, int)
    assert DRAIN_SECONDS > 0


# --- the second interrupt ---------------------------------------------------------------------


async def test_a_second_interrupt_stops_the_wait(socket_path: Path) -> None:
    """The escape hatch, exercised against a drain that would otherwise not end.

    ``_bounded_by`` is what the server wraps its shutdown in: the first signal asks for the
    ordered close, and the second says the operator has stopped waiting. Driven directly here
    rather than by signaling a real process, because what is under test is the decision and not
    the signal plumbing — ``tests/app/test_launchd.py`` covers a real process's exit.
    """
    del socket_path
    started = asyncio.Event()
    impatient = asyncio.Event()

    async def never() -> None:
        started.set()
        await asyncio.Event().wait()  # a drain with nothing to finish it

    waiting = asyncio.create_task(cli_serving.bounded_by(never(), impatient))
    await asyncio.wait_for(started.wait(), timeout=30)
    assert not waiting.done(), "the wait ended before anybody asked it to"

    impatient.set()
    async with asyncio.timeout(30):
        await waiting


async def test_a_shutdown_that_finishes_is_not_cut_short(socket_path: Path) -> None:
    """The control for the test above, without which "it returns" proves nothing.

    A wait that always returned immediately would satisfy the second-interrupt assertion and
    would also abandon every ordinary shutdown. So the same helper is given work that *does*
    finish and nobody interrupts it, and the work is checked to have completed rather than been
    canceled.
    """
    del socket_path
    done = False

    async def finishes() -> None:
        nonlocal done
        await asyncio.sleep(0)
        done = True

    await cli_serving.bounded_by(finishes(), asyncio.Event())

    assert done, "the shutdown was abandoned although nobody interrupted it"


# --- the signal a supervisor actually sends ----------------------------------------------------


MANICULE = Path(sys.executable).with_name("manicule")

SCHEDULER_STOPPING = "stopping the scheduler"
SOCKET_CLOSING = "closing the control socket"
TRANSPORT_CLOSING = "closing the HTTP server"
"""The three lines the shutdown writes to stderr, in the order it writes them.

Matched on a prefix rather than on the whole sentence, so rewording the explanation does not
break the ordering assertion — what is being asserted is the sequence, not the prose.
"""

UVICORN_STOPPING = "Shutting down"
"""What uvicorn says when *it* begins to stop, which is how the inversion is caught.

Its own log line rather than one of manicule's, and that is the point: with uvicorn's signal
handling left in place it shuts the transport down first and manicule's three steps run
afterwards, on the way out. Both orders leave a tidy process and a removed socket, so this line's
*position* is the only thing that tells them apart.
"""


def _stopped(tmp_path: Path) -> str:
    """Start a real server, ``SIGTERM`` it, and return everything it said.

    A real process because signal handling is what is under test and there is no way to install a
    handler in a test's own interpreter without deciding the answer.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    socket = control.socket_path(data_dir)
    process = subprocess.Popen(  # noqa: S603 - this project's own console script
        [str(MANICULE), "serve", "--transport", "http", "--port", str(_free_port()), "--no-web"],
        env=_environment(data_dir),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_socket(socket, process)
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=60) == 0, "the server did not exit cleanly on SIGTERM"
        said = process.stdout.read() if process.stdout is not None else ""
    finally:
        if process.poll() is None:  # pragma: no cover - only a server that ignored the signal
            process.kill()
            process.wait(timeout=30)
        if process.stdout is not None:
            process.stdout.close()

    assert not socket.exists(), (
        f"{socket} is still there after a clean stop, so the control socket's aclose did not run"
    )
    return said


def test_a_sigterm_closes_the_four_things_in_the_stated_order(tmp_path: Path) -> None:
    """The order, observed from outside a process a supervisor stopped the way supervisors do.

    **This is the test the signal arrangement exists for, and getting it wrong is instructive.**
    The first version asserted a clean exit and a removed socket, and passed with
    :meth:`~manicule.api.serve.Server.capture_signals` reverted — because uvicorn restores the
    *previous* handler before re-raising, and the previous handler is manicule's own, so the
    process still unwound. What it unwound in was the wrong order: uvicorn had already drained
    and stopped the transport before the scheduler was touched, which is step four happening
    before step one.

    So the assertion is on the sequence rather than on the tidiness. Each of manicule's three
    steps announces itself, uvicorn announces its own, and the four positions are what a
    reverted override changes.
    """
    said = _stopped(tmp_path)

    for line in (SCHEDULER_STOPPING, SOCKET_CLOSING, TRANSPORT_CLOSING, UVICORN_STOPPING):
        assert line in said, f"the shutdown never said {line!r}. It said:\n{said}"
    positions = [said.index(line) for line in (SCHEDULER_STOPPING, SOCKET_CLOSING)]
    assert positions == sorted(positions), f"the scheduler closed after the socket:\n{said}"
    assert said.index(SOCKET_CLOSING) < said.index(TRANSPORT_CLOSING), (
        f"the transport was closed before the control socket had finished:\n{said}"
    )
    assert said.index(TRANSPORT_CLOSING) < said.index(UVICORN_STOPPING), (
        f"uvicorn began shutting down before manicule asked it to, which means uvicorn is still "
        f"handling the signal and the order above is whatever happened to be left:\n{said}"
    )


def test_the_signal_handlers_are_installed_before_anything_is_started() -> None:
    """The window a supervisor restarting a just-started server lands in.

    ``_writing`` binds the control socket, and the socket appearing is what tells everything
    else — ``manicule stop``, a proxied command, a supervisor — that this process is up. A
    ``SIGTERM`` between that and the handler being installed reaches the *default* handler, which
    kills the process where it stands: nothing is closed, the socket file is left behind, and the
    exit status says the job crashed rather than that it was asked to stop.

    **Asserted structurally, and that is a deliberate choice rather than the easy one.** The
    behavioral form is :func:`test_a_sigterm_closes_the_four_things_in_the_stated_order`, and it
    found this — but only under load, because whether the signal lands inside the window depends
    on the machine. A test that reproduces a race sometimes is a test that reports the fix
    sometimes. This one reads the nesting, which is the thing that closes the window, and is red
    the moment it is inverted.
    """
    import ast  # noqa: PLC0415 - only this assertion parses a function
    import inspect  # noqa: PLC0415
    import textwrap  # noqa: PLC0415

    source = textwrap.dedent(inspect.getsource(cli_serving.serve_over_a_socket))
    body = ast.parse(source).body[0]
    assert isinstance(body, ast.AsyncFunctionDef)

    outermost = next(node for node in body.body if isinstance(node, (ast.With, ast.AsyncWith)))
    assert isinstance(outermost, ast.With), (
        "the outermost context manager is an `async with`, so the signal handlers are not the "
        "first thing installed"
    )
    called = {
        node.func.id
        for item in outermost.items
        if isinstance(node := item.context_expr, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert called == {"_signals"}, (
        f"the outermost context manager is {sorted(called)} rather than the signal handlers. "
        f"Anything started outside them can be killed by a signal the process meant to handle."
    )
    inner = [node for node in ast.walk(outermost) if isinstance(node, ast.AsyncWith)]
    assert inner, "nothing is started inside the signal handlers, so they guard nothing"


def test_a_sigterm_leaves_a_clean_exit_and_no_socket_behind(tmp_path: Path) -> None:
    """The tidiness, separately, because it is a different failure from the order.

    A process that exits non-zero on ``SIGTERM`` tells a supervisor its job crashed, and
    ``KeepAlive`` then restarts something that stopped on purpose. A socket left on disk is a file
    the next start has to recognize as stale — which ``ControlServer.start`` does, so it is
    survivable, and it is still a server that did not finish its own shutdown.
    """
    assert _stopped(tmp_path)


def _free_port() -> int:
    """A port nothing is listening on, found by asking the kernel for one and letting it go.

    ``--port 0`` is not available here and that is deliberate rather than an omission:
    :func:`~manicule.app.bind.resolve_bind` refuses it, because zero means "any free port" and a
    server that chose one would be listening somewhere nobody was told about. So the test does
    the choosing, which is the party that can afterwards say which port it was.

    There is a window between closing this and the server binding. It is the standard one, it is
    small, and the alternative — a fixed port — fails whenever the suite runs twice at once,
    which is a larger window and a more confusing failure.
    """
    import socket as socketlib  # noqa: PLC0415 - only this helper needs it

    with socketlib.socket(socketlib.AF_INET, socketlib.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        chosen: int = probe.getsockname()[1]
    return chosen


def _environment(data_dir: Path) -> dict[str, str]:
    import os  # noqa: PLC0415 - only this helper needs the caller's environment

    return {
        **os.environ,
        "MANICULE_DATA_DIR": str(data_dir),
        "COLUMNS": str(WIDE_ENOUGH),
    }


def _wait_for_socket(
    socket: Path, process: subprocess.Popen[str], *, patience_s: float = 60.0
) -> None:
    """Return once the server is listening on its control socket, or say why it never did.

    The socket rather than the port, because the port is 0 — the kernel picks it and the test
    never needs to know which. What it needs is the moment the process is *serving*, and the
    control socket is bound as part of that.
    """
    deadline = time.monotonic() + patience_s
    while time.monotonic() < deadline:
        if socket.exists():
            return
        if process.poll() is not None:  # pragma: no cover - only a server that failed to start
            output = process.stdout.read() if process.stdout is not None else ""
            pytest.fail(f"the server exited {process.returncode} before listening: {output}")
        time.sleep(0.05)
    with contextlib.suppress(Exception):  # pragma: no cover - only on a server that never started
        process.kill()
    pytest.fail(f"the server never bound {socket} within {patience_s:g}s")
