"""Starting the MCP server: the two transports, and what each one is allowed to carry.

``stdio`` is the default and opens no socket at all. That is not a convenience — it is why
the ordinary way of running manicule under an assistant has no address to get wrong, and it is
also why stdio carries the **whole** tool surface: stdin and stdout are a pipe between one
client and one process, so a write tool on it is unreachable from a network by construction.

``http`` exists for a client that cannot spawn a process, and it goes through
:func:`~manicule.app.bind.resolve_bind` like every other server in this project. A
non-loopback bind needs a host somebody wrote down, an explicit opt-in the caller passes, and
authentication switched on. Any one missing is a refusal.

**And it carries the read-only tools only.** Binding a socket removes the property stdio had, so
something has to replace it: :func:`~manicule.mcp.server.build_server` is asked for the read-only
surface, which never registers a tool whose ``readOnlyHint`` is not true. The write tools stay on
stdio, on the command line, and on the control socket of #139 — every one of them a place where a
person is present or a process is the writer. See :data:`NETWORK_SURFACE_IS_READ_ONLY`.

**A socket also gets a front door**, which stdio has no use for and no place to put. ``/`` on
this transport is where ``--mcp-only`` lands somebody who opened the address the process printed,
and :func:`signpost` is the whole of it — see :mod:`manicule.app.frontdoor`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from starlette.responses import PlainTextResponse

from manicule.app import frontdoor
from manicule.app.bind import resolve_bind, stdio
from manicule.app.results import ServerAddress
from manicule.core.errors import PolicyError
from manicule.mcp.server import TOOL_NAMES, Surface, build_surface

if TYPE_CHECKING:
    from fastmcp import FastMCP
    from starlette.requests import Request

    from manicule.app.service import ApplicationService

type Transport = Literal["stdio", "http"]

NETWORK_SURFACE_IS_READ_ONLY = True
"""Whether MCP served over a socket carries only the tools that read. It does, always.

A named constant rather than a parameter, and that is the whole point of it: a parameter is a
way of asking for the other answer, and there is no caller entitled to one. The absence of the
write tools from a network transport is a structural property this project keeps in the same way
it keeps ``tests/api/test_routes.py``'s absences — by not building the thing, rather than by
guarding it — and a setting that could grant an exception would trade that for a configuration
guarantee, which is a weaker one that fails silently.

It exists so the rule can be *read* at the one place a reader would look for it, and so
``tests/mcp/test_transports.py`` can name it. If a write over the network is ever wanted, it is
its own decision with its own threat model rather than a flag flipped here.
"""


def surface(service: ApplicationService, *, transport: Transport) -> Surface:
    """The server this transport serves: everything over stdio, the reads over a socket.

    One function, so "which tools does this transport carry" has one answer and every caller
    reaches it the same way. :func:`serve` uses it, and so does the HTTP API's mount
    (:func:`manicule.api.app.build_app`) — which is the other place MCP reaches a socket, and
    would otherwise be the place the rule was restated slightly differently.
    """
    return build_surface(service, read_only=transport != "stdio" and NETWORK_SURFACE_IS_READ_ONLY)


async def _front_door(request: Request) -> PlainTextResponse:
    """``GET /`` on an ``--mcp-only`` socket: say so, and name the endpoint that is here.

    The address is read off the request rather than off the bind, so what a person is shown is
    the address they actually reached this process on — which means the body carries a value out
    of the ``Host`` header. It is ``text/plain`` and ``nosniff``, so there is nothing for a
    browser to interpret; the header is set **here** rather than inherited, because this
    application is the library's and has none of the middleware
    :data:`manicule.api.app.SECURITY_HEADERS` puts on every other response this project serves.
    """
    return PlainTextResponse(
        frontdoor.mcp_alone(str(request.base_url).rstrip("/")),
        headers={"X-Content-Type-Options": "nosniff"},
    )


def signpost(server: FastMCP) -> None:
    """Give a server that is about to be served **alone** over a socket something at ``/``.

    Only ever called on that path, and the "alone" is load-bearing rather than descriptive. This
    registers a route at the root of whatever application the server is put in, so calling it on
    the surface :func:`manicule.api.app.build_app` mounts would shadow the MCP endpoint itself —
    that mount is made with ``path="/"``, and a custom route at ``/`` wins against it. There the
    front door is the HTTP application's own, and ``/mcp/`` is a path beneath it.

    Registered by call rather than by decorator, for the reason
    :func:`manicule.api.app.build_app` gives about its middleware: a decorated inner function is
    bound to a name nothing reads.
    """
    server.custom_route("/", ["GET"], name="front_door")(_front_door)


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

    ``tools`` counts what *this* transport offers rather than what the module registers, so the
    line an operator reads at startup says how many tools the thing they just started actually
    has. Reporting thirty-seven for a socket that carries twenty-two would be the banner
    disagreeing with ``tools/list`` on the one number somebody would check.

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
        tools=len(surface(service, transport=transport).tools),
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
    server = surface(service, transport=transport).server
    if address.transport == "stdio":
        await server.run_stdio_async(show_banner=False)
        return
    if address.port is None:  # pragma: no cover - resolve_bind always decides a port
        msg = "an HTTP transport was resolved without a port"
        raise PolicyError(msg)
    # Nothing else is on this socket, so `/` is this process's only chance to say what is.
    signpost(server)
    await server.run_http_async(
        show_banner=False,
        # Passed explicitly, never left to the library's own default. A default that happens
        # to be loopback today is a default that can change in a release nobody read — and
        # `port` is refused above rather than defaulted, because the obvious fallback is 0,
        # which means "any free port" and would serve somewhere nobody was told about.
        host=address.host,
        port=address.port,
        # The same rule, and here it is the *advertised* address that depends on it. The banner,
        # the front door, `docs/surfaces.md` §6.1 and the README all name `/mcp/` — so that is
        # the path this has to answer on directly, and it was not.
        #
        # **With the library's default, `--mcp-only` served `/mcp` and redirected `/mcp/` to it**
        # — the opposite direction from the whole server, where the mount makes `/mcp/` the real
        # path and `/mcp` the redirect. So every client configured from the documentation paid a
        # 307 on a `POST`, which the ones that decline to re-send a body do not survive. Two
        # modes disagreeing about which of two addresses is the real one is exactly the kind of
        # thing nobody finds by reading, because both work from a browser.
        path=frontdoor.MCP_ENDPOINT,
    )


__all__ = [
    "NETWORK_SURFACE_IS_READ_ONLY",
    "Transport",
    "address_for",
    "serve",
    "signpost",
    "surface",
]
