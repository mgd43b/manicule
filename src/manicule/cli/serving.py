"""Starting and stopping the server from the command line.

Two things happen here that do not happen anywhere else, and both are worth keeping in one
place: the bind decision is *printed before it is used*, so an operator sees where a process
is about to listen rather than finding out from a port scan; and the running process records
itself, so ``stop`` has something to stop.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

from manicule.app.bind import is_loopback
from manicule.app.control import ControlServer, ProtocolError, socket_path
from manicule.app.daemon import Running, read_pidfile, stop_server, write_pidfile
from manicule.app.dispatch import error_info
from manicule.app.results import Envelope, ServerAddress, failed, succeeded
from manicule.app.runtime import Runtime
from manicule.app.served import ControlHandler, Scheduler, Serving
from manicule.app.service import ApplicationService
from manicule.cli import render
from manicule.config.loader import load_settings
from manicule.connectors.sessions import SESSIONS
from manicule.core.errors import ManiculeError
from manicule.mcp.serve import address_for, serve

if TYPE_CHECKING:
    from collections.abc import Mapping

    from manicule.mcp.serve import Transport


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

    ``--transport http`` serves the **HTTP API**; ``--mcp-only`` serves the MCP protocol over
    the same transport instead. ``stdio`` is MCP whatever else was asked for, because the HTTP
    API has no stdio form — and it is the default, so the ordinary way of running manicule
    still opens no socket at all.
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
        # one where the transport raises — which is what the context manager is for.
        try:
            async with _writing(service):
                if api:
                    from manicule.api.serve import serve as serve_api  # noqa: PLC0415 - heavy

                    await serve_api(
                        service, host=host, port=port, allow_public=allow_public, web=web
                    )
                else:
                    await serve(
                        service,
                        transport=transport,
                        host=host,
                        port=port,
                        allow_public=allow_public,
                    )
        except ProtocolError as exc:
            # A control socket that will not bind is a refusal rather than a crash: the
            # commonest cause is a second server, and the second commonest is a runtime
            # directory somebody else owns. Both are things an operator fixes.
            _report(failed("start", service.workspace, error_info(exc)), json_output)
            return 1
        finally:
            pid.unlink(missing_ok=True)
    return 0


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
    # It has to happen **here** rather than as the server unwinds, and the reason is uvicorn's:
    # it handles the signal, shuts down cleanly, then restores the default handler and re-raises
    # — so the process dies without unwinding the `async with` that would have tidied up. The
    # stdio transport does unwind and does tidy up, which is exactly the sort of difference
    # nobody would predict from reading either side. A socket left behind by a crash, a
    # `SIGKILL` or a power cut is still possible and is not a problem: it is a file with nothing
    # behind it, and `ControlServer.start` clears one.
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


__all__ = ["running_address", "serve_forever", "stop_running"]
