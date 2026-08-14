"""Assembling the HTTP application: twelve route groups over one application service.

The service is passed in rather than built here, exactly as :func:`manicule.mcp.server.build_server`
takes one. That is what lets the suites drive the **real** routing, the real dependency
resolution and the real middleware against a fake backend — and a surface that could only be
tested by starting a whole manicule is a surface nobody tests.

Three decisions are made here and nowhere else, because each of them has to hold for every
route or it does not hold at all:

**An unauthenticated application that configuration says is not loopback does not get built.**
:func:`~manicule.app.bind.resolve_bind` already refuses that address before a socket exists,
but an ASGI application can be started by something other than ``manicule start`` — a
production server, a container entry point, somebody's own uvicorn invocation. So the refusal
is repeated where the *application* is made, and it is the same one, stated the same way.

**CORS is explicit or absent.** With no ``security.transport.allowed_origins`` the middleware
is not installed at all, which means same-origin only. There is no wildcard: configuration
refuses ``*`` outright, because a cross-origin wildcard over a document index means every page
a user visits can read it with whatever the browser will send.

**Framing is refused unless somebody named the frames.** Every response carries
``frame-ancestors`` naming ``security.transport.widget_allowed_domains``, and ``'none'`` when
that list is empty. A chat surface inside an invisible frame is a clickjacking target, and the
default answer to a framing attempt is no.

**A cross-site request may not change anything.** :mod:`manicule.api.origins` decides that, and
it is checked here — before routing — because a rule applied per route is a rule missing from
the route somebody added last. The question only arose with the browser surface: the shipped
posture is loopback with no credential at all, which is precisely the ambient authority a page
on another origin can spend on the user's behalf.

**MCP is mounted here too, at** :data:`MCP_PATH`, **and it carries the read-only tools only.**
One process, one port, one bind decision and one address for an operator to remember — a second
port would double what has to be got right and would need its own answer to every question this
module already answers once. Mounting it on *this* application is what makes those answers apply
to it: the middleware above wraps the whole ASGI stack, so the cross-site refusal, the security
headers and the principal resolution reach ``/mcp`` without being restated there.

What does **not** carry over is the tool surface, and that is the point of it being read-only.
``tests/api/test_routes.py`` asserts by name that the destructive operations have no route here;
a mount that offered ``document_delete`` over the same socket would have made that assertion
true and meaningless on the same day. :mod:`manicule.mcp.serve` decides which tools a transport
carries, and this asks it rather than deciding again.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from manicule.api.envelopes import AUTH_ERRORS, malformed, refusal
from manicule.api.origins import FETCH_SITE, ORIGIN, REFUSAL, permitted
from manicule.api.proxy import ProxyPolicy
from manicule.api.routes import (
    admin,
    auth,
    chat,
    conversations,
    documents,
    organization,
    plugins,
    sockets,
    workbench,
)
from manicule.api.routes import health as health_routes
from manicule.api.security import resolve
from manicule.api.widget import router as widget_router
from manicule.app import frontdoor
from manicule.app.bind import is_loopback
from manicule.config.settings import AuthMode
from manicule.core.errors import PolicyError
from manicule.core.version import CORE_VERSION
from manicule.mcp.serve import surface as mcp_surface

NOT_FOUND = 404

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from starlette.responses import Response

    from manicule.app.bind import Bind
    from manicule.app.service import ApplicationService

logger = logging.getLogger("manicule.api")

TITLE = "manicule"

DESCRIPTION = """\
Search and question-answering over a self-hosted document index.

Every response is the same envelope: `ok`, `op`, `workspace`, and then `data` or `error`. Read
`ok` first — a failure is a result with a shape, not a bare status code. Everything is scoped
to one workspace and nothing crosses between them.
"""

ROUTE_GROUPS = (
    "health",
    "documents",
    "chat",
    "conversations",
    "collections",
    "tags",
    "admin",
    "plugins",
    "auth",
    "workbench",
    "websocket-chat",
    "mcp",
)
"""The twelve groups this surface offers, named as data as well as mounted.

Here so that "the API offers exactly these" is a test rather than a count somebody keeps in
their head — the same reason ``manicule.mcp.server`` lists its tool names.

``mcp`` and ``websocket-chat`` are the two that no OpenAPI document describes, so each is
asserted by driving it rather than by reading the schema. A group an OpenAPI-driven check cannot
see is exactly the one it would quietly report as present.
"""

MCP_PATH = frontdoor.MCP
"""Where MCP answers on this application, and it is a path rather than a second port.

One port means one bind decision, one address in a plist, one thing for an operator to
remember and one place a firewall rule applies. A second port would need its own answer to
every question ``manicule.app.bind`` answers here, and the commonest way that goes wrong is
that it gets a *different* answer — usually the lax one, because the second port is the one
added in a hurry for a client that could not spawn a process.

The trailing slash matters to clients: the mount answers at ``/mcp/``, and Starlette redirects
``/mcp`` to it. Both are documented in ``docs/surfaces.md`` §6.1 so nobody has to discover it
from a 307.

Assigned from :data:`manicule.app.frontdoor.MCP` rather than written out, because the front door
and the startup banner both print this path and neither may import this module — the MCP server's
own transport must not load FastAPI. One constant, three readers, no drift.
"""

SECURITY_HEADERS = {
    # The API returns JSON and the widget returns a script. Neither is ever HTML a browser
    # should be guessing about.
    "X-Content-Type-Options": "nosniff",
    # A URL can carry a document id or a share token. Neither belongs in the `Referer` a
    # browser sends to whatever the page links to next.
    "Referrer-Policy": "no-referrer",
    # Nothing here is a document a browser should keep. A shared conversation is a bearer URL,
    # and a cached copy outlives the revocation.
    "Cache-Control": "no-store",
}
"""Headers every response carries, whatever produced it.

In middleware rather than per route, because a header applied per route is a header missing
from the route somebody added last.
"""


async def _open_the_browser_surface(request: Request) -> RedirectResponse:
    """``GET /`` when the browser surface is mounted: go there.

    A redirect rather than the dashboard served at two paths, so the address bar ends up
    somewhere real — bookmarkable, linkable, reloadable — instead of on a URL that only works
    for as long as ``/`` keeps rendering a copy of it.

    Relative, deliberately. An absolute ``Location`` would have to be built from the ``Host``
    header, which is a value the client chose; a relative one names this server's own path and
    cannot be pointed at somebody else's. The status is
    :data:`~manicule.app.frontdoor.TEMPORARY_REDIRECT` and the reason is written there.
    """
    return RedirectResponse(
        url=frontdoor.to_the_browser_surface(request.url.query),
        status_code=frontdoor.TEMPORARY_REDIRECT,
    )


async def _say_what_is_served(request: Request) -> PlainTextResponse:
    """``GET /`` under ``--no-web``: name the surfaces that *are* here.

    Not a redirect, because there is nothing to redirect to. Sending somebody to a browser
    surface an operator switched off would be worse than the 404 this replaces: it would spend
    a second request to arrive at the same answer, having claimed the thing was somewhere.
    """
    return PlainTextResponse(frontdoor.without_the_browser_surface(_origin(request)))


def _origin(request: Request) -> str:
    """The address this request arrived on, with no trailing slash.

    Read off the request rather than off the bind, so what a person is shown is the address they
    are actually using — through a proxy, over a forwarded port, or at a name that resolves to
    this host. The bind knows where the socket is; only the request knows how it was reached.
    """
    return str(request.base_url).rstrip("/")


def frame_policy(origins: tuple[str, ...]) -> str:
    """The ``Content-Security-Policy`` this installation serves.

    ``frame-ancestors 'none'`` when nothing is configured — a refusal, not a permission — and
    the configured origins otherwise. ``default-src 'none'`` alongside it because the API
    serves JSON: a policy that permitted script or connect sources would be describing a page
    that does not exist.
    """
    ancestors = " ".join(origins) if origins else "'none'"
    return f"default-src 'none'; frame-ancestors {ancestors}"


def build_app(
    service: ApplicationService, *, bind: Bind | None = None, web: bool = True
) -> FastAPI:
    """Mount every route group over ``service`` and return the application.

    Args:
        service: The one service this application serves. Its workspace is the tenant every
            route runs in; there is no route that takes a workspace.
        bind: Where this application is about to be served, when the caller has decided.
            Passed so the auth check below can see a *decided* address rather than only the
            configured one — ``manicule start --host`` can name an address configuration does
            not.
        web: Whether to mount the browser surface. ``manicule start --no-web`` sets this
            false. It is a parameter rather than a setting because the flag exists to *reduce*
            what a running process exposes, and a reduction that a configuration file could
            undo is not one.

    Raises:
        PolicyError: The application would serve an unauthenticated surface on something that
            is not loopback.
    """
    # The browser surface, imported here rather than at module scope. Its pages are routes
    # over the same dependencies this package defines, so importing it from the top would make
    # `manicule.api` and `manicule.web` import each other — and which one won would depend on
    # which a caller reached for first.
    from manicule.web.pages import router as web_router  # noqa: PLC0415
    from manicule.web.security import (  # noqa: PLC0415
        PageRefusedError,
        is_page_request,
        not_found_page,
        refused_page,
    )

    settings = service.settings
    _require_auth_for_wide_bind(service, bind)

    # Built before the application, because the application needs its lifespan. FastMCP's ASGI
    # app owns a session manager that has to be started and stopped, and a mount does not run a
    # sub-application's lifespan — so an app assembled without this line serves an MCP endpoint
    # that answers every request with "session manager not initialized". Passing it here is what
    # makes shutting the HTTP server down also close the MCP sessions, which is the last step of
    # the order `manicule.cli.serving` keeps.
    mcp = mcp_surface(service, transport="http").server.http_app(
        path="/",
        # **Nothing about one call outlives it, and both flags say that in a different way.**
        #
        # `stateless_http` builds the protocol session per request and tears it down with the
        # response, so there is no session identifier, no server-side session table and no
        # per-connection state of any kind. That is the honest shape for this surface: it is
        # read-only, it starts nothing long-running, and it sends no server-initiated messages,
        # so a session would be bookkeeping with nothing in it. It is also the answer to "can
        # one client see another's state" — there is no state to see, rather than a table with
        # a lock on it.
        #
        # `json_response` answers a call with one JSON body instead of an event stream carrying
        # one event. Same reason, and it matters twice over at shutdown: an event stream is a
        # connection a client may hold open indefinitely, which is what
        # `manicule.api.serve.DRAIN_SECONDS` exists to bound, and a half-read one is a resource
        # nobody closed. A request/response surface has neither problem.
        stateless_http=True,
        json_response=True,
    )

    app = FastAPI(
        title=TITLE,
        version=CORE_VERSION,
        description=DESCRIPTION,
        # The interactive docs are served from the same origin and reflect only the schema,
        # which is generated from these modules' own type hints.
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
        lifespan=mcp.lifespan,
    )
    app.state.service = service
    app.state.proxy_policy = ProxyPolicy.of(settings)
    app.state.route_groups = ROUTE_GROUPS

    origins = settings.security.transport.allowed_origins
    if origins:
        # Installed only when an operator has listed origins. Absent means same-origin, which
        # is the correct default for a surface over somebody's private index.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(origins),
            allow_credentials=False,
            # Credentials are never cookies here — a key is presented on every request — so
            # `allow_credentials` stays off. With it on, a browser would attach cookies to
            # cross-origin calls, which is the ingredient a CSRF needs.
            allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-API-Key"],
            max_age=600,
        )
        logger.info("cross-origin requests permitted from: %s", ", ".join(origins))

    async def identify(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Refuse a cross-site write, resolve the caller, and dress the response.

        All three here rather than per dependency, so that the address decision — which reads
        a header and a socket peer — happens in exactly one place, every route including the
        ones that need no credential sees the same principal, and the cross-site refusal
        applies to routes that do not exist yet.
        """
        request.state.principal = await resolve(service, app.state.proxy_policy, request)
        response = _refuse_cross_site(service, request) or await call_next(request)
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        response.headers.setdefault(
            "Content-Security-Policy",
            frame_policy(settings.security.transport.widget_allowed_domains),
        )
        return response

    async def refuse(request: Request, exc: Exception) -> JSONResponse:
        """Render an authentication or authorization refusal as the ordinary envelope."""
        return refusal(_op_of(request), service.workspace, exc)

    async def unreadable(request: Request, exc: Exception) -> JSONResponse:
        """Render a request the framework rejected as the ordinary envelope.

        Without this, the single most common failure a client hits — a missing parameter —
        would come back in FastAPI's own ``{"detail": [...]}`` shape, which has no ``ok`` in
        it. One surface, one shape.
        """
        return malformed(_op_of(request), service.workspace, exc)

    # Registered by call rather than by decorator. A decorated inner function is bound to a
    # name nothing reads, which a strict type checker correctly reports as dead — and the
    # honest fix is to reference the function rather than to silence the checker.
    app.middleware("http")(identify)
    for failure in AUTH_ERRORS:
        app.add_exception_handler(failure, refuse)
    app.add_exception_handler(RequestValidationError, unreadable)
    # The browser surface renders its refusals as pages. Registered here rather than inside
    # `manicule.web` because an exception handler belongs to the application, and there is one
    # application.
    app.add_exception_handler(PageRefusedError, refused_page)

    async def missing(request: Request, exc: Exception) -> Response:
        """A 404 under ``/ui`` is a page; everywhere else it stays the framework's envelope.

        Registered only when the browser surface is mounted — with ``--no-web`` there is no
        browser surface, so a ``/ui`` path is as absent as any other and says so in JSON.
        """
        if is_page_request(request):
            return not_found_page(request, exc)
        return await http_exception_handler(request, cast("StarletteHTTPException", exc))

    if web:
        app.add_exception_handler(NOT_FOUND, missing)

    routers = [
        health_routes.router,
        documents.router,
        chat.router,
        conversations.router,
        organization.router,
        admin.router,
        plugins.router,
        auth.router,
        workbench.router,
        sockets.router,
        widget_router,
    ]
    if web:
        # The browser surface (#12). Mounted on the same application because it is the same
        # service, the same principal resolution and the same middleware — and because a
        # second application would be a second place for a security header to be forgotten.
        #
        # Conditional because `--no-web` exists to switch it off. It was mounted
        # unconditionally from #12 until this was noticed: the flag had been accepted and
        # discarded since #8, so an operator who passed `--no-web` was served the whole
        # browser surface anyway. `tests/app/test_serving.py` holds that shut.
        routers.append(web_router)
    for router in routers:
        app.include_router(router)
    # The front door (#145). Registered here rather than in a router because which handler it is
    # depends on `web`, and a router carrying two versions of one path would have to decide
    # between them at request time — the decision is made once, at assembly, like every other
    # decision this function makes.
    app.add_api_route(
        "/",
        _open_the_browser_surface if web else _say_what_is_served,
        methods=["GET"],
        name="front_door",
        summary="The browser surface, or — when it is not served — what this process is serving.",
    )
    # Mounted last, so nothing this application declares can be shadowed by the sub-application's
    # catch-all. A mount matches every path beneath its prefix, and the prefix is its own —
    # `/mcp` names no route group above and never will, because a group that collided with it
    # would be unreachable rather than merely confusing.
    app.mount(MCP_PATH, mcp)
    return app


def _refuse_cross_site(service: ApplicationService, request: Request) -> JSONResponse | None:
    """Refuse an unsafe method a browser says came from another site, or return ``None``.

    The decision itself is :func:`manicule.api.origins.permitted`, which is a pure function of
    four header values so it can be exercised without a server. This carries the headers to it
    and renders the refusal as the ordinary envelope — ``PolicyError``, therefore 403, by the
    same table every other failure uses.
    """
    origin = request.headers.get(ORIGIN)
    if permitted(
        request.method,
        fetch_site=request.headers.get(FETCH_SITE),
        origin=origin,
        host=request.headers.get("host"),
        allowed_origins=service.settings.security.transport.allowed_origins,
    ):
        return None
    logger.warning("refused a cross-site %s from %r", request.method, origin or "an unnamed origin")
    return refusal(
        _op_of(request),
        service.workspace,
        PolicyError(REFUSAL.format(method=request.method, origin=origin or "")),
    )


def _op_of(request: Request) -> str:
    """The operation a refused request was heading for, for the envelope's ``op``.

    Read off the matched route rather than invented, so a 401 names the same operation the
    successful call would have — which is what makes an access log of refusals joinable to one
    of successes.
    """
    route = request.scope.get("route")
    name = getattr(route, "name", "")
    return name or "request"


def _require_auth_for_wide_bind(service: ApplicationService, bind: Bind | None) -> None:
    """Refuse to build an unauthenticated application that is not loopback-only.

    The second of two refusals, and deliberately not the same code as the first.
    :func:`~manicule.app.bind.resolve_bind` decides an *address* and refuses before a socket
    exists; this decides whether an *application* may exist at all, and it fires even when
    somebody else's server is doing the listening. A caller that has already decided an
    address passes it, because a command-line ``--host`` can name one configuration does not.

    Raises:
        PolicyError: Authentication is off and the address is not loopback.
    """
    settings = service.settings
    if settings.security.auth.mode is not AuthMode.NONE:
        return
    host = bind.host if bind is not None else settings.security.transport.bind_host
    if is_loopback(host):
        return
    msg = (
        f"refusing to build an HTTP API bound to {host!r} with security.auth.mode set to "
        f"'none'. Anything that can reach the port could read the whole index. Set "
        f"security.auth.mode to 'api_key' or 'oauth', or leave the bind on 127.0.0.1."
    )
    raise PolicyError(msg)


__all__ = [
    "DESCRIPTION",
    "MCP_PATH",
    "ROUTE_GROUPS",
    "SECURITY_HEADERS",
    "TITLE",
    "build_app",
    "frame_policy",
]
