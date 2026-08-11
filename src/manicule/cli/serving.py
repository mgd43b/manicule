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
from manicule.app.daemon import Running, read_pidfile, stop_server, write_pidfile
from manicule.app.dispatch import error_info
from manicule.app.results import Envelope, ServerAddress, failed, succeeded
from manicule.app.runtime import Runtime
from manicule.app.service import ApplicationService
from manicule.cli import render
from manicule.config.loader import load_settings
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
) -> int:
    try:
        runtime = Runtime.open(**overrides)
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
        _report(succeeded("start", service.workspace, address), json_output, stderr=True)
        pid = write_pidfile(
            runtime.settings.data_dir,
            transport=address.transport,
            host=address.host,
            port=address.port,
        )
        try:
            if api:
                from manicule.api.serve import serve as serve_api  # noqa: PLC0415 - heavy

                await serve_api(service, host=host, port=port, allow_public=allow_public)
            else:
                await serve(
                    service,
                    transport=transport,
                    host=host,
                    port=port,
                    allow_public=allow_public,
                )
        finally:
            pid.unlink(missing_ok=True)
    return 0


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


def _report(envelope: Envelope, json_output: bool, *, stderr: bool = False) -> None:
    out = render.console(stderr=stderr or not envelope.ok)
    if json_output:
        out.print_json(data=envelope.as_json())
        return
    if envelope.ok and envelope.data is not None:
        render.render(out, ServerAddress.model_validate(envelope.data))
        return
    if envelope.error is not None:
        render.render_error(out, envelope.op, envelope.error)


__all__ = ["running_address", "serve_forever", "stop_running"]
