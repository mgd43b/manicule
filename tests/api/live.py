"""A real manicule server, on a real port, for the assertions that need one.

Most of this project's HTTP suites drive the application through Starlette's test client, which
is right for them: they are about routing, envelopes and refusals, and a socket would add a port
to every one of them without adding a fact.

**These are not those.** What is under test here is one *process* serving two surfaces to more
than one client at once, and every part of that is a property of the transport: two clients are
two connections, a wedged client is a connection that is not being read, and a disconnect is a
socket closing. A test client that dispatches into the application in the caller's own task
cannot exhibit any of it, and would pass whatever the answer was.

So this starts uvicorn on port 0, asks the kernel which port it got, and hands back both an
ordinary HTTP client and a real MCP client speaking streamable HTTP to the same address.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any, cast

import httpx2
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from manicule.api.app import MCP_PATH, build_app
from manicule.app.service import ApplicationService

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from fastapi import FastAPI
    from mcp.shared._httpx_utils import McpHttpClientFactory

    from manicule.api.serve import Server
    from tests.app.fakes import FakeBackend


@contextlib.asynccontextmanager
async def mounted(backend: FakeBackend, *, web: bool = True) -> AsyncGenerator[Client[Any]]:
    """An MCP client speaking to the mount, through the application and not through a socket.

    For the assertions that are about *what the surface offers* rather than about connections:
    which tools are published, what a call to an absent one does, what the instructions say. Each
    is a fact about the mounted application, so a port would only make the suite slower and
    flakier.

    The lifespan is entered explicitly, because that is what starts the MCP session manager —
    the same line uvicorn runs, reached the same way. Without it every request is answered with
    "session manager not initialized", which would make an absence assertion pass for the wrong
    reason.
    """
    import httpx  # noqa: PLC0415 - what fastmcp's client is written against

    app = build_app(ApplicationService(backend), web=web)

    def through_the_app(**arguments: Any) -> httpx.AsyncClient:
        """The client fastmcp would have built, pointed at the application instead of a socket.

        ``**arguments`` rather than the three parameters the library's ``McpHttpClientFactory``
        declares, because the two disagree: the declared type names ``headers``, ``timeout`` and
        ``auth``, and the call site also passes ``follow_redirects``. Writing the declared
        signature out therefore type-checks and then fails at run time — which is how this
        comment came to be here. Everything given is passed on; only ``transport`` is dropped,
        because supplying one is the entire point.
        """
        arguments.pop("transport", None)
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), **arguments)

    async with app.router.lifespan_context(app):
        transport = StreamableHttpTransport(
            f"http://testserver{MCP_PATH}/",
            httpx_client_factory=cast("McpHttpClientFactory", through_the_app),
        )
        async with Client(transport) as client:
            yield client


STARTUP_TIMEOUT_S = 20.0
"""How long to wait for uvicorn to report itself started before failing the test.

Generous, because a loaded runner starting an event loop is not the thing under test, and it is
a *timeout* rather than a sleep: a server that starts in eight milliseconds is waited on for
eight milliseconds.
"""


class Live:
    """One running server, and the two ways to talk to it."""

    def __init__(self, app: FastAPI, server: Server, port: int) -> None:
        self.app = app
        self.server = server
        self.port = port

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def mcp_url(self) -> str:
        """The MCP endpoint, with the trailing slash the mount answers on.

        Written out rather than left to a redirect, because a client that followed one would be
        exercising Starlette's redirect and not this surface — and because an operator putting
        this in a client's configuration needs the address that works.
        """
        return f"{self.base_url}{MCP_PATH}/"

    def http(self) -> httpx2.AsyncClient:
        """An ordinary HTTP client for the JSON API and the browser surface."""
        return httpx2.AsyncClient(base_url=self.base_url, timeout=30.0)

    def mcp(self) -> Client[Any]:
        """A fresh MCP client. Each call is a **separate** client with its own session."""
        return Client(StreamableHttpTransport(self.mcp_url))


@contextlib.asynccontextmanager
async def serving(backend: FakeBackend, *, web: bool = True) -> AsyncGenerator[Live]:
    """Run the production application over ``backend`` until the block ends.

    Port 0, so the kernel picks one nobody else is using. A fixed port in a suite is a suite that
    fails when it is run twice at once or when something unrelated happens to be listening, and
    the failure looks like the feature being broken.
    """
    from uvicorn import Config  # noqa: PLC0415

    from manicule.api.serve import Server  # noqa: PLC0415 - pulls in uvicorn

    app = build_app(ApplicationService(backend), web=web)
    server = Server(Config(app, host="127.0.0.1", port=0, log_level="warning", access_log=False))
    running = asyncio.create_task(server.serve(), name="live-server")
    try:
        await _started(server, running)
        yield Live(app, server, _port(server))
    finally:
        server.should_exit = True
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(running, timeout=STARTUP_TIMEOUT_S)
        if not running.done():  # pragma: no cover - only a server that ignored should_exit
            running.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await running


async def _started(server: Server, running: asyncio.Task[None]) -> None:
    """Wait until the server is listening, failing loudly if it stopped instead.

    ``running`` is watched as well as ``started``, because a server that raised while binding
    would otherwise be waited on until the timeout and reported as slow rather than as broken.
    """
    async with asyncio.timeout(STARTUP_TIMEOUT_S):
        while not server.started:
            if running.done():
                await running
                msg = "the server task ended before it started listening"
                raise AssertionError(msg)
            await asyncio.sleep(0.01)


def _port(server: Server) -> int:
    """The port the kernel gave it, read off the socket rather than remembered."""
    sockets = [socket for one in server.servers for socket in one.sockets]
    assert sockets, "the server reported itself started with no socket bound"
    port: int = sockets[0].getsockname()[1]
    return port


__all__ = ["STARTUP_TIMEOUT_S", "Live", "mounted", "serving"]
