"""Starting the HTTP API: the bind decision, then uvicorn.

The address comes from :func:`~manicule.app.bind.resolve_bind` — the same function the MCP
server's HTTP transport uses, and the only one in this project that decides where anything
listens. There is no second bind path here and no literal address anywhere in this module;
``tests/app/test_bind.py`` reads the source tree to keep that true.

The decision is made **before** the server object exists, so a refusal happens with no socket
open, and it is announced before the server starts rather than discovered from a port scan.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from manicule.api.app import ROUTE_GROUPS, build_app
from manicule.app.bind import resolve_bind
from manicule.app.results import ServerAddress

if TYPE_CHECKING:
    from fastapi import FastAPI

    from manicule.app.bind import Bind
    from manicule.app.service import ApplicationService

TRANSPORT = "http-api"
"""What this server records in the pid file, so ``stop`` reports what it stopped.

Distinct from the MCP server's ``http``: the two are different surfaces on the same protocol,
and an operator reading a pid file should not have to guess which one is running.
"""


def address_for(
    service: ApplicationService,
    *,
    host: str | None = None,
    port: int | None = None,
    allow_public: bool = False,
) -> tuple[Bind, ServerAddress]:
    """Decide where the API will listen, before starting anything.

    Separate from :func:`serve` so a caller can print the decision and a test can check the
    refusal without opening a socket. A bind decided only inside the call that performs it is
    a bind nobody can assert on.

    Raises:
        PolicyError: The address is not loopback and something required to widen it is
            missing. The message names which.
    """
    bind = resolve_bind(service.settings, host=host, port=port, allow_public=allow_public)
    return bind, ServerAddress(
        transport=TRANSPORT,
        host=bind.host,
        port=bind.port,
        loopback=bind.loopback,
        # Route groups rather than tools. The field counts what the surface offers, and for
        # this surface that is eleven groups rather than nineteen tools.
        tools=len(ROUTE_GROUPS),
    )


def application(
    service: ApplicationService,
    *,
    host: str | None = None,
    port: int | None = None,
    allow_public: bool = False,
    web: bool = True,
) -> tuple[FastAPI, ServerAddress]:
    """The application and the address it is about to be served on.

    The bind is resolved **first** and handed to :func:`~manicule.api.app.build_app`, so the
    "unauthenticated surface on a routable address" refusal sees the address that was actually
    decided rather than the one configuration happens to hold.

    ``web`` is passed straight through: ``--no-web`` has to reach the mount to mean anything.
    """
    bind, address = address_for(service, host=host, port=port, allow_public=allow_public)
    return build_app(service, bind=bind, web=web), address


async def serve(
    service: ApplicationService,
    *,
    host: str | None = None,
    port: int | None = None,
    allow_public: bool = False,
    web: bool = True,
) -> None:
    """Run the HTTP API until it is stopped."""
    import uvicorn  # noqa: PLC0415 - a server library, imported by the one path that serves

    app, address = application(service, host=host, port=port, allow_public=allow_public, web=web)
    if address.port is None:  # pragma: no cover - resolve_bind always decides a port
        from manicule.core.errors import PolicyError  # noqa: PLC0415

        msg = "an HTTP bind was resolved without a port"
        raise PolicyError(msg)
    config = uvicorn.Config(
        app,
        # Passed explicitly, never left to the library's default. A default that happens to be
        # loopback today is a default that can change in a release nobody read.
        host=address.host,
        port=address.port,
        log_level="info",
        access_log=False,
        # The access log would record every request path. A share token is a path segment, so
        # the default access log writes live credentials to disk on every anonymous read.
        server_header=False,
        date_header=True,
    )
    await uvicorn.Server(config).serve()


__all__ = ["TRANSPORT", "address_for", "application", "serve"]
