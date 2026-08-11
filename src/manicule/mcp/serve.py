"""Starting the MCP server: the two transports, and the bind decision one of them needs.

``stdio`` is the default and opens no socket at all. That is not a convenience — it is why
the ordinary way of running manicule under an assistant has no address to get wrong.

``http`` exists for a client that cannot spawn a process, and it goes through
:func:`~manicule.app.bind.resolve_bind` like every other server in this project. A
non-loopback bind needs a host somebody wrote down, an explicit opt-in the caller passes, and
authentication switched on. Any one missing is a refusal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from manicule.app.bind import resolve_bind, stdio
from manicule.app.results import ServerAddress
from manicule.core.errors import PolicyError
from manicule.mcp.server import TOOL_NAMES, build_server

if TYPE_CHECKING:
    from manicule.app.service import ApplicationService

type Transport = Literal["stdio", "http"]


def address_for(
    service: ApplicationService,
    *,
    transport: Transport = "stdio",
    host: str | None = None,
    port: int | None = None,
    allow_public: bool = False,
) -> ServerAddress:
    """Decide where the server will listen, before starting anything.

    Separate from :func:`serve` so a caller can print the decision, and so a test can check
    the refusal without opening a socket. A bind that is only decided inside the call that
    performs it is a bind nobody can assert on.

    Raises:
        PolicyError: The requested address is not loopback and something required to widen it
            is missing.
    """
    if transport == "stdio":
        stdio()
        return ServerAddress(transport="stdio", loopback=True, tools=len(TOOL_NAMES))
    bind = resolve_bind(service.settings, host=host, port=port, allow_public=allow_public)
    return ServerAddress(
        transport="http",
        host=bind.host,
        port=bind.port,
        loopback=bind.loopback,
        tools=len(TOOL_NAMES),
    )


async def serve(
    service: ApplicationService,
    *,
    transport: Transport = "stdio",
    host: str | None = None,
    port: int | None = None,
    allow_public: bool = False,
) -> None:
    """Run the server until it is stopped.

    The address is decided **first**, by :func:`address_for`, so a refusal happens before a
    socket exists rather than after one has been listening for a moment.
    """
    address = address_for(
        service, transport=transport, host=host, port=port, allow_public=allow_public
    )
    server = build_server(service)
    if address.transport == "stdio":
        await server.run_stdio_async(show_banner=False)
        return
    if address.port is None:  # pragma: no cover - resolve_bind always decides a port
        msg = "an HTTP transport was resolved without a port"
        raise PolicyError(msg)
    await server.run_http_async(
        show_banner=False,
        # Passed explicitly, never left to the library's own default. A default that happens
        # to be loopback today is a default that can change in a release nobody read — and
        # `port` is refused above rather than defaulted, because the obvious fallback is 0,
        # which means "any free port" and would serve somewhere nobody was told about.
        host=address.host,
        port=address.port,
    )


__all__ = ["Transport", "address_for", "serve"]
