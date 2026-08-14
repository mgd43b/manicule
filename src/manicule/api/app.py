"""Assembling the HTTP application: eleven route groups over one application service.

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
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
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
from manicule.app.bind import is_loopback
from manicule.config.settings import AuthMode
from manicule.core.errors import PolicyError
from manicule.core.version import CORE_VERSION

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
)
"""The eleven groups this surface offers, named as data as well as mounted.

Here so that "the API offers exactly these" is a test rather than a count somebody keeps in
their head — the same reason ``manicule.mcp.server`` lists its tool names.
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

    app = FastAPI(
        title=TITLE,
        version=CORE_VERSION,
        description=DESCRIPTION,
        # The interactive docs are served from the same origin and reflect only the schema,
        # which is generated from these modules' own type hints.
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
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


__all__ = ["DESCRIPTION", "ROUTE_GROUPS", "SECURITY_HEADERS", "TITLE", "build_app", "frame_policy"]
