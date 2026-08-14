"""Starting and stopping the server from the command line.

Three things happen here that do not happen anywhere else, and each is worth keeping in one
place: the bind decision is *printed before it is used*, so an operator sees where a process is
about to listen rather than finding out from a port scan; the running process records itself, so
``stop`` has something to stop; and **this is where the shutdown order lives**.

**The order, once, in one function.** A served manicule is four things running at the same time
and they have to stop in a stated sequence, because each step's work is what the next step must
not interrupt:

1. **The scheduler**, so no new sync starts. Canceling a loop cancels the sync it was inside.
2. **The ingest stages** of whatever was running, which drain within ``ingest.shutdown_grace_s``
   (#138). Not a separate call: it is what a canceled run does, and it is listed because a
   reader has to know that step 1 is not instantaneous and why waiting is correct.
3. **The control socket**, whose ``aclose`` waits for the write commands already in flight
   (#139) — each one an operator typed and is watching.
4. **The MCP sessions and the HTTP server**, together and last, because one lifespan owns both:
   shutting the transport down runs the application's lifespan, which closes the MCP session
   manager. Bounded by :data:`~manicule.api.serve.DRAIN_SECONDS`, because an attached MCP client
   holding a stream open is entitled to keep holding it.

Steps 1 to 3 are :meth:`~manicule.app.served.Serving.aclose`. Step 4 is here, after it, which is
why the transport is a task this function stops rather than a call it awaits.

**A second interrupt cancels the wait.** The first asks for all of the above; the second says the
operator is no longer willing to wait for it, and it takes effect on whichever step is in
progress.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from typing import TYPE_CHECKING, Any, cast

from manicule.app.bind import is_loopback
from manicule.app.control import ControlServer, ProtocolError, socket_path
from manicule.app.daemon import Running, read_pidfile, stop_server, write_pidfile
from manicule.app.dispatch import error_info
from manicule.app.results import Envelope, ServerAddress, failed, succeeded
from manicule.app.runtime import Runtime
from manicule.app.served import ControlHandler, Scheduler, Serving, announce
from manicule.app.service import ApplicationService
from manicule.cli import render
from manicule.config.loader import load_settings
from manicule.connectors.sessions import SESSIONS
from manicule.core.errors import InstanceLockedError, ManiculeError
from manicule.mcp.serve import address_for, serve

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Generator, Mapping

    from manicule.mcp.serve import Transport

EX_TEMPFAIL = 75
"""The exit status for "somebody else has this data directory; try again later".

``EX_TEMPFAIL`` from ``sysexits.h``, whose stated meaning is exactly this one: the failure is
temporary and the caller is invited to retry. It matters because the caller is usually launchd,
which restarts what it supervises and has no way to ask *why* a start failed — so the status is
the only thing that distinguishes "this will never work" from "the previous process has not
finished letting go yet", and the second is an ordinary moment during a restart rather than a
fault.

What makes it retry *sensibly* rather than spin is ``ThrottleInterval`` in the plist
(``docs/deployment.md`` §4), which is where the cadence belongs: a wait built into this process
would be a wait an operator running ``manicule serve`` by hand would also have to sit through,
for a refusal they can read and act on immediately.

Separate from the ``1`` every other refusal exits with, and that is the whole point of it. A bind
this configuration will never permit, a plugin that will not import and a data directory somebody
else holds are not the same event, and restarting the first two thirty seconds later produces the
identical failure thirty seconds later, for ever.
"""


def serve_forever(
    *,
    transport: str,
    host: str | None,
    port: int | None,
    allow_public: bool,
    overrides: Mapping[str, Any],
    json_output: bool,
    mcp_only: bool = False,
    web: bool = True,
) -> int:
    """Run the server until it is stopped. Returns the process's exit status.

    The address is decided and announced **first**. A refusal — a non-loopback bind that was
    not asked for three separate times — happens here, before a socket exists, and is
    reported the same way every other failure is.

    ``--transport http`` serves the **HTTP API, the browser surface and MCP together**, on one
    port; ``--mcp-only`` serves MCP alone over that socket. ``stdio`` is MCP whatever else was
    asked for, because the HTTP API has no stdio form — and it is the default, so the ordinary
    way of running manicule still opens no socket at all.

    Returns:
        ``0`` on a clean stop, :data:`EX_TEMPFAIL` when another process holds the data
        directory, ``2`` for a transport that does not exist, ``130`` for ``Ctrl-C`` on the stdio
        path, and ``1`` for every other refusal.
    """
    if transport not in {"stdio", "http"}:
        out = render.console(stderr=True)
        out.print(f"[red]no such transport {transport!r}. Available: stdio, http[/red]")
        return 2
    try:
        return asyncio.run(
            _serve(
                transport=cast("Transport", transport),
                host=host,
                port=port,
                allow_public=allow_public,
                overrides=overrides,
                json_output=json_output,
                mcp_only=mcp_only,
                web=web,
            )
        )
    except KeyboardInterrupt:  # pragma: no cover - a person pressing ^C
        return 130


async def _serve(
    *,
    transport: Transport,
    host: str | None,
    port: int | None,
    allow_public: bool,
    overrides: Mapping[str, Any],
    json_output: bool,
    mcp_only: bool,
    web: bool = True,
) -> int:
    # A server is a writer for its whole life, which is what `Runtime.open`'s default already
    # says. What this has to get right is the *refusal*: the lock is taken by `__aenter__`, so
    # a second server on the same directory raises from inside the `async with` and would
    # otherwise reach the terminal as a traceback rather than as the one-line refusal the lock
    # was written to produce.
    try:
        runtime = Runtime.open(**overrides)
        runtime.acquire()
    except InstanceLockedError as exc:
        # Named ahead of the general clause and given its own status, because this is the one
        # refusal that fixes itself: the process holding the directory is on its way out, or an
        # operator will stop it, and the next attempt succeeds with nothing changed. See
        # :data:`EX_TEMPFAIL`.
        _report(failed("start", "unknown", error_info(exc)), json_output)
        return EX_TEMPFAIL
    except (ManiculeError, ValueError, OSError) as exc:
        _report(failed("start", "unknown", error_info(exc)), json_output)
        return 1
    async with runtime:
        service = ApplicationService(runtime)
        api = transport == "http" and not mcp_only
        try:
            address = (
                _api_address(service, host=host, port=port, allow_public=allow_public)
                if api
                else address_for(
                    service,
                    transport=transport,
                    host=host,
                    port=port,
                    allow_public=allow_public,
                )
            )
        except ManiculeError as exc:
            _report(failed("start", service.workspace, error_info(exc)), json_output)
            return 1
        # Announced before the socket exists, and to stderr when the transport is stdio —
        # where stdout is the protocol channel and a banner on it is a corrupt message.
        #
        # `web` is the one thing the renderer cannot read off the payload: which surface is
        # serving is in the transport, but `--no-web` is recorded nowhere. Passed only here,
        # where it was decided; `stop` reads a pid file and rightly claims nothing about it.
        _report(
            succeeded("start", service.workspace, address),
            json_output,
            stderr=True,
            web=web if api else None,
        )
        pid = write_pidfile(
            runtime.settings.data_dir,
            transport=address.transport,
            host=address.host,
            port=address.port,
        )
        # The control socket and the scheduler are what make this process *the writer* rather
        # than a process that happens to serve a protocol. They are started after the lock is
        # held and after the address is announced, and closed on every path out — including the
        # one where the transport raises.
        try:
            if api:
                await serve_over_a_socket(
                    service, host=host, port=port, allow_public=allow_public, web=web
                )
            else:
                await _serve_a_protocol(
                    service,
                    transport=transport,
                    host=host,
                    port=port,
                    allow_public=allow_public,
                )
        except ProtocolError as exc:
            # A control socket that will not bind is a refusal rather than a crash: the
            # commonest cause is a second server, and the second commonest is a runtime
            # directory somebody else owns. The first fixes itself and the second does not, and
            # the message says which — but the *status* has to be one of them, so it is the one
            # that retries: a directory somebody else owns is a refusal an operator will read on
            # the first attempt, while a lock held by a process still exiting is a race that a
            # retry resolves and a permanent status would turn into a server that never came up.
            _report(failed("start", service.workspace, error_info(exc)), json_output)
            return EX_TEMPFAIL
        finally:
            pid.unlink(missing_ok=True)
    return 0


async def _serve_a_protocol(
    service: ApplicationService,
    *,
    transport: Transport,
    host: str | None,
    port: int | None,
    allow_public: bool,
) -> None:
    """Serve MCP alone — over stdio, or over a socket with ``--mcp-only``.

    Signals are left exactly as they were, which is the difference from
    :func:`serve_over_a_socket` and is deliberate rather than an omission. Nothing on this path
    captures them: ``Ctrl-C`` raises ``KeyboardInterrupt`` out of the transport, the ``async
    with`` unwinds in order on the way past, and ``manicule serve`` over stdio therefore behaves
    exactly as it did before any of this existed — which matters, because that is the path an
    editor spawns and the one nobody is watching.
    """
    async with _writing(service):
        await serve(service, transport=transport, host=host, port=port, allow_public=allow_public)


async def serve_over_a_socket(
    service: ApplicationService,
    *,
    host: str | None,
    port: int | None,
    allow_public: bool,
    web: bool,
) -> None:
    """Serve the HTTP API, the browser surface and MCP from one process, and stop them in order.

    Public because it *is* the shutdown order this module's docstring describes, and an order
    nothing outside the module can name is an order nothing outside the module can check.

    The transport runs as a *task* rather than as an awaited call, because the shutdown has four
    steps and the transport is the last of them: something has to be able to close the scheduler
    and the control socket while the server is still answering, and then close the server.

    The signal handlers are installed here, over the whole arrangement, for the reason
    :class:`~manicule.api.serve.Server` gives: uvicorn's own handlers end the process at step
    four before steps one to three have happened.
    """
    from manicule.api.serve import server_for  # noqa: PLC0415 - heavy, and only this path serves

    transport = server_for(service, host=host, port=port, allow_public=allow_public, web=web)
    stop = asyncio.Event()
    impatient = asyncio.Event()

    def asked_to_stop() -> None:
        """The first signal asks; the second says the operator has stopped waiting.

        The second sets ``force_exit``, which is uvicorn's own word for "close the connections
        rather than waiting for them", and releases whatever step is currently waiting.
        """
        if stop.is_set():
            transport.force_exit = True
            impatient.set()
            return
        stop.set()

    # **The handlers go on before anything is started**, and the order is load-bearing rather
    # than tidy. `_writing` binds the control socket, and the socket appearing is what tells
    # everything else — `manicule stop`, a proxied command, a supervisor's readiness check — that
    # this process is up. A `SIGTERM` arriving between the socket being bound and the handler
    # being installed reaches the *default* handler, which kills the process where it stands: the
    # scheduler is not stopped, the socket file is left behind, and the exit status says the job
    # crashed rather than that it was asked to stop. Small window, ordinary cause — a supervisor
    # restarting a server it has only just started — and it was found by a test, not by reading.
    with _signals(asked_to_stop):
        async with _writing(service) as writing:
            running = asyncio.create_task(transport.serve(), name="transport")
            await _first_of(stop, running)
            # Steps 1 to 3, and the second interrupt cancels the *wait* rather than the work:
            # what is abandoned is this process's patience, and the ingest stages still get their
            # own bounded drain because that bound is inside them.
            await bounded_by(writing.aclose(), impatient)
            # Step 4. `should_exit` is the field uvicorn's own handler sets, so what happens next
            # is uvicorn's shutdown and not a reimplementation of it.
            announce("closing the HTTP server and any MCP sessions on it")
            transport.should_exit = True
            await running


async def _first_of(stop: asyncio.Event, running: asyncio.Task[None]) -> None:
    """Return when the server is asked to stop, or when the transport ends on its own.

    Both are ordinary. A transport that ends by itself is a port that could not be bound or a
    server that finished; either way the shutdown below still runs, and running it against a
    transport that has already stopped costs one assignment and one already-finished await.
    """
    waiting = asyncio.ensure_future(stop.wait())
    try:
        await asyncio.wait((waiting, running), return_when=asyncio.FIRST_COMPLETED)
    finally:
        waiting.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await waiting


async def bounded_by(work: Awaitable[None], impatient: asyncio.Event) -> None:
    """Await ``work``, unless the operator interrupts a second time.

    Public because it *is* the "a second interrupt cancels the wait" promise, and a promise a
    test can only reach through a private name is one nobody outside this module can check.

    The work is *canceled* on the second interrupt rather than left running, because the thing
    being waited on is a drain and a drain nobody is waiting for is a process that will not exit.
    Cancelation is safe at every point in it: the scheduler's own docstring says so of a sync,
    and a control connection cut off mid-answer is a client that already knows the server is
    going away.
    """
    running = asyncio.ensure_future(work)
    waiting = asyncio.ensure_future(impatient.wait())
    try:
        await asyncio.wait((running, waiting), return_when=asyncio.FIRST_COMPLETED)
        if running.done():
            await running
            return
        running.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await running
    finally:
        waiting.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await waiting


@contextlib.contextmanager
def _signals(handler: Callable[[], None]) -> Generator[None]:
    """Take ``SIGINT`` and ``SIGTERM`` for the length of the block, and give them back after.

    Restoring matters as much as installing: this process may go on to report a failure and exit
    through ordinary paths, and a handler left behind would swallow the ``Ctrl-C`` of whatever
    ran next.

    A platform with no ``add_signal_handler`` — Windows, for the ``SIGTERM`` half — is left
    alone rather than worked around. There is no supervisor there for this to be about, and a
    ``KeyboardInterrupt`` still unwinds everything the ``async with`` above holds.
    """
    loop = asyncio.get_running_loop()
    taken: list[signal.Signals] = []
    for number in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(number, handler)
        except (NotImplementedError, RuntimeError, ValueError):  # pragma: no cover - not POSIX
            continue
        taken.append(number)
    try:
        yield
    finally:
        for number in taken:
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                loop.remove_signal_handler(number)


def _writing(service: ApplicationService) -> Serving:
    """The control socket and the scheduler for this service, not yet started.

    Assembled in one place so that ``what a served manicule adds`` is one expression a reader
    can take in, rather than four statements interleaved with the transport's own setup.
    """
    handler = ControlHandler(service, SESSIONS)
    return Serving(
        server=ControlServer(socket_path(service.settings.data_dir), handler),
        scheduler=Scheduler(service, Scheduler.configure(service)),
    )


def _api_address(
    service: ApplicationService,
    *,
    host: str | None,
    port: int | None,
    allow_public: bool,
) -> ServerAddress:
    """Where the HTTP API will listen, decided before anything is built.

    Imported inside the function so that starting an MCP server — the default — never loads
    FastAPI or uvicorn. The same rule every other optional dependency in this project follows.
    """
    from manicule.api.serve import address_for as api_address_for  # noqa: PLC0415 - heavy

    _, address = api_address_for(service, host=host, port=port, allow_public=allow_public)
    return address


def stop_running(overrides: Mapping[str, Any], *, workspace: str) -> Envelope:
    """Stop the recorded server, and describe what was stopped.

    Configuration is *loaded* rather than a whole runtime opened: all this needs is the data
    directory, and discovering plugins to find a path would make ``stop`` fail on an
    installation whose plugins are the reason somebody is stopping it.
    """
    try:
        settings = load_settings(**overrides)
    except (ManiculeError, ValueError, OSError) as exc:
        return failed("stop", workspace, error_info(exc))
    try:
        running = stop_server(settings.data_dir)
    except (ManiculeError, TimeoutError, OSError) as exc:
        return failed("stop", settings.workspace, error_info(exc))
    # The control socket, on the same principle `stop_server` removes the pid file: the process
    # is confirmed gone, so nothing is behind it.
    #
    # **The server now removes it itself**, on every path including a supervisor's `SIGTERM`:
    # `serve_over_a_socket` owns the signals, so `ControlServer.aclose` runs and unlinks the
    # path. This line therefore removes a file that is usually already gone, and it stays for the
    # cases where it is not — a `SIGKILL`, a power cut, or an older server on the far side of an
    # upgrade. It used to be load-bearing for the ordinary stop as well, because uvicorn's own
    # signal handling ended the process before the `async with` unwound; that is what
    # `manicule.api.serve.Server` exists to prevent, and `tests/app/test_shutdown.py` asserts the
    # socket is gone before this ever runs.
    #
    # A socket left behind by a crash, a `SIGKILL` or a power cut is not a problem either way: it
    # is a file with nothing behind it, and `ControlServer.start` clears one.
    socket_path(settings.data_dir).unlink(missing_ok=True)
    return succeeded(
        "stop",
        settings.workspace,
        _address_of(running),
    )


def running_address(overrides: Mapping[str, Any]) -> ServerAddress | None:
    """What the pid file says is running, if anything usable is recorded."""
    try:
        settings = load_settings(**overrides)
    except (ManiculeError, ValueError, OSError):
        return None
    running = read_pidfile(settings.data_dir)
    if running is None:
        return None
    return _address_of(running)


def _address_of(running: Running) -> ServerAddress:
    """Describe a recorded server, deciding "loopback" the one way this project decides it.

    Through :func:`~manicule.app.bind.is_loopback` rather than a set written out here. A
    second copy of that definition is a second answer to "is this reachable from the
    network", and the two would disagree the first time either changed.
    """
    return ServerAddress(
        transport=running.transport,
        host=running.host,
        port=running.port,
        # A stdio server records no host, and it is loopback in the only sense that matters:
        # it has no socket at all.
        loopback=not running.host or is_loopback(running.host),
    )


def _report(
    envelope: Envelope,
    json_output: bool,
    *,
    stderr: bool = False,
    web: bool | None = None,
) -> None:
    out = render.console(stderr=stderr or not envelope.ok)
    if json_output:
        out.print_json(data=envelope.as_json())
        return
    if envelope.ok and envelope.data is not None:
        render.render_address(out, ServerAddress.model_validate(envelope.data), web=web)
        return
    if envelope.error is not None:
        render.render_error(out, envelope.op, envelope.error)


__all__ = [
    "EX_TEMPFAIL",
    "bounded_by",
    "running_address",
    "serve_forever",
    "serve_over_a_socket",
    "stop_running",
]
