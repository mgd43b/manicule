"""Starting the HTTP API: the bind decision, then uvicorn — whose signals this process keeps.

The address comes from :func:`~manicule.app.bind.resolve_bind` — the same function the MCP
server's HTTP transport uses, and the only one in this project that decides where anything
listens. There is no second bind path here and no literal address anywhere in this module;
``tests/app/test_bind.py`` reads the source tree to keep that true.

The decision is made **before** the server object exists, so a refusal happens with no socket
open, and it is announced before the server starts rather than discovered from a port scan.

**And this process handles its own ``SIGTERM``**, which :class:`Server` is the whole of. uvicorn
installs handlers for the length of ``serve()``, shuts down cleanly when one arrives, restores
the previous handler and then **re-raises the signal** — so the process dies at that point, and
every ``finally``, ``async with`` and ``aclose`` above it is skipped. That was survivable while
the only thing above it was a pid file. It is not survivable now: the scheduler, the ingest
stages draining and the control socket's in-flight commands all live above it, in a stated order
(:class:`~manicule.app.served.Serving`), and a supervisor's ``SIGTERM`` is the *ordinary* way this
server stops rather than an edge case.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, override

import uvicorn

from manicule.api.app import ROUTE_GROUPS, build_app
from manicule.app.bind import resolve_bind
from manicule.app.results import ServerAddress

if TYPE_CHECKING:
    from collections.abc import Generator

    from fastapi import FastAPI

    from manicule.app.bind import Bind
    from manicule.app.service import ApplicationService

# uvicorn at module scope, which every other heavy dependency in this project is deliberately
# not. :class:`Server` subclasses ``uvicorn.Server``, and a base class cannot be imported inside
# the function that needs it. The property that mattered is kept where it always was: this
# *module* is imported by the paths that serve HTTP and by nothing else, so `manicule serve`
# over stdio still loads neither this nor FastAPI.

TRANSPORT = "http-api"
"""What this server records in the pid file, so ``stop`` reports what it stopped.

Distinct from the MCP server's ``http``: the two are different surfaces on the same protocol,
and an operator reading a pid file should not have to guess which one is running.
"""

DRAIN_SECONDS = 5
"""How long the HTTP server waits for requests in flight before closing their connections.

**Bounded on purpose, and the reason is MCP.** uvicorn's default is to wait for ever, which is
defensible for request/response traffic because a request either finishes or the client goes
away. An MCP client holds a *stream* open and is entitled to keep holding it, so an unbounded
wait means one attached client can stop the server from ever stopping — and the operator's next
move is ``kill -9`` on a process that is mid-write, which is the outcome
:data:`~manicule.app.daemon.STOP_GRACE_S` refuses to produce on purpose.

Five seconds because it has to fit *inside* that ten-second grace with the rest of the shutdown,
and because nothing here is finishing a write: the writes drained two steps earlier, with the
ingest stages. What is being waited for is a response being serialized.
``tests/app/test_shutdown.py`` asserts the relationship rather than the number.
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
        # this surface that is twelve groups — one of which is the mounted MCP endpoint, whose
        # own tool count is reported by `manicule.mcp.serve.address_for` when MCP is what is
        # being served rather than one group among twelve.
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


class Server(uvicorn.Server):
    """uvicorn's server with its signal handling taken back.

    One method, overridden to do nothing, and it is load-bearing. uvicorn captures ``SIGINT`` and
    ``SIGTERM`` for the length of ``serve()``, and on the way out it restores the previous handler
    and re-raises the signal it caught — which kills the process where it stands. Everything this
    project does *around* the transport is therefore skipped on the commonest way a server is
    stopped: the scheduler is not closed, the ingest stages do not drain, the control socket's
    in-flight commands are cut off and the socket file is left behind.

    So the handlers are installed one level up, by the code that also owns the order
    (:func:`~manicule.cli.serving._serve`), and stopping this server is
    ``should_exit = True`` — the same field uvicorn's own handler sets.

    **Nothing else is changed**, deliberately: this is not a reimplementation of the server, and
    the shutdown it performs when asked is uvicorn's own.
    """

    @override
    @contextlib.contextmanager
    def capture_signals(self) -> Generator[None]:
        """Install nothing and restore nothing.

        A context manager that yields, rather than a no-op ``def``: the base class calls this as
        one, and a version of uvicorn that starts using its return value differently should fail
        at the type checker rather than at three in the morning.
        """
        yield


def server_for(
    service: ApplicationService,
    *,
    host: str | None = None,
    port: int | None = None,
    allow_public: bool = False,
    web: bool = True,
) -> Server:
    """The HTTP server, configured and not yet listening.

    Handed back rather than run, because the caller has to be able to *stop* it: shutting the
    transport down is the last step of an order the caller owns, and a function that both started
    and finished the server would give it no handle to do that with.
    """
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
        timeout_graceful_shutdown=DRAIN_SECONDS,
    )
    return Server(config)


async def serve(
    service: ApplicationService,
    *,
    host: str | None = None,
    port: int | None = None,
    allow_public: bool = False,
    web: bool = True,
) -> None:
    """Run the HTTP API until it is stopped.

    Kept for callers that want the whole thing in one call and have nothing to shut down around
    it — the suites, and anybody embedding manicule. ``manicule serve`` uses
    :func:`server_for` instead, because it does have something to shut down around it.
    """
    await server_for(service, host=host, port=port, allow_public=allow_public, web=web).serve()


__all__ = [
    "DRAIN_SECONDS",
    "TRANSPORT",
    "Server",
    "address_for",
    "application",
    "serve",
    "server_for",
]
