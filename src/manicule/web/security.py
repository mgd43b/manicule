"""Who may read a page, and what a refusal looks like when the reader is a browser.

**The authorization decision is not made here.** :func:`manicule.api.security.require` is the
one implementation of "does this principal clear this floor", and this module calls it. A
second implementation for the browser surface is exactly the shape of bug this project has
already found once: a rule that held on one surface and not on the one somebody was actually
using.

What *is* decided here is the rendering. A JSON envelope with ``ok: false`` is the right answer
to a program and the wrong one to a person who typed a URL, so the refusal is re-raised as
:class:`PageRefusedError` and rendered as a page — with the same status code, carrying the same
message the API would have sent.

## What a browser can present, said plainly

It depends on ``security.auth.mode``, and the refusal page says the sentence that is true of
the mode in force rather than one that is true of some other installation:

* ``none`` — the shipped posture, one person on loopback. There is no credential to present and
  nothing is refused for want of one.
* ``api_key`` — a key is presented on every request in a header, and a browser cannot attach a
  header to a top-level navigation. There is no session cookie in this mode, so a page load
  carries no credential and is refused, with a page that says so and says what to use instead.
* ``oauth`` — a person signs in through an identity provider at ``/ui/login`` and the browser
  holds a session cookie from then on (:mod:`manicule.api.cookies`). An unauthenticated page
  load is refused with a link to sign in.

**Refused, not redirected, in that last case.** A ``303`` to the sign-in page would be kinder
to look at and would make every protected page answer a program, a monitor or a cached link
with a success status. The refusal keeps the status the API would have used and puts the way
forward one click away.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, cast

from fastapi import Depends, Request

from manicule.api.security import (
    ForbiddenError,
    Principal,
    UnauthenticatedError,
    anonymous,
    require,
)
from manicule.config.settings import AuthMode, Role
from manicule.core.errors import ManiculeError

NOT_FOUND = 404
OK = 200

LOGIN_PAGE = "/ui/login"
"""Where a person signs in. The one link a refusal offers, and only where it would help."""

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastapi.responses import HTMLResponse
    from pydantic import JsonValue


class PageRefusedError(ManiculeError):
    """A refusal that must be rendered as a page rather than as an envelope.

    It carries the refusal it wraps rather than restating it, so the message a person reads is
    the message the API would have sent — one explanation of what authentication this
    installation wants, not two that can drift.
    """

    def __init__(self, cause: UnauthenticatedError | ForbiddenError) -> None:
        super().__init__(str(cause))
        self.cause = cause


def _page_dependency(floor: Role):  # noqa: ANN202 - the return type is FastAPI's own callable
    async def guard(request: Request) -> Principal:
        principal = anonymous(request)
        try:
            return require(principal, floor)
        except (UnauthenticatedError, ForbiddenError) as exc:
            raise PageRefusedError(exc) from exc

    return guard


Reader = Annotated[Principal, Depends(_page_dependency(Role.VIEWER))]
"""Anyone who may read this workspace. Every page but the two below."""

Operator = Annotated[Principal, Depends(_page_dependency(Role.ADMIN))]
"""The administration and identity areas, matching the API routes they render.

Query logs are the questions somebody asked, the audit trail names who did what and from
where, and the key list and the member list are the workspace's identities. The HTTP routes
behind these pages take an admin, and a page that took less would be a way round them.
"""

Guest = Annotated[Principal, Depends(anonymous)]
"""No credential at all. Two pages use it: the shared conversation, which is a bearer URL, and
the sign-in page, which is how a person without a credential gets one."""


def refused_page(request: Request, exc: Exception) -> HTMLResponse:
    """Render a refusal as a page, at the status the API would have used.

    Registered on the application rather than per route, so a page added later cannot be the
    one that answers a refusal with a JSON body in a browser window.
    """
    from manicule.api.envelopes import FORBIDDEN, UNAUTHORIZED  # noqa: PLC0415 - avoids a cycle
    from manicule.web.rendering import (  # noqa: PLC0415 - avoids a cycle
        ENVIRONMENT,
        STYLESHEET_PATH,
        html_response,
    )

    cause = exc.cause if isinstance(exc, PageRefusedError) else exc
    unauthenticated = isinstance(cause, UnauthenticatedError)
    status = UNAUTHORIZED if unauthenticated else FORBIDDEN
    mode = _mode(request)
    hint = ""
    sign_in = ""
    if unauthenticated and mode == AuthMode.OAUTH.value:
        hint = "This installation lets people sign in. Sign in to continue."
        sign_in = LOGIN_PAGE
    elif unauthenticated:
        hint = (
            "A browser cannot attach a header to a page load, and with API keys there is no "
            "session cookie — a key is presented on every request instead. Use the HTTP API "
            "or the command line; the browser surface serves an installation with no "
            "authentication, or one where people sign in."
        )
    else:
        hint = "Your role does not reach this page; an administrator of this workspace decides it."
    body = ENVIRONMENT.get_template("refused.html").render(
        {
            "title": "Not permitted",
            "message": str(cause),
            "status": status,
            "hint": hint,
            "sign_in": sign_in,
            # The path only. Never the query string: a share token is a path segment on one
            # route and a query value is a place credentials end up, and a refusal page is not
            # where either belongs.
            "path": request.url.path,
            "stylesheet": STYLESHEET_PATH,
        }
    )
    return html_response(body, status=status)


def sign_in_refused(request: Request, message: str, *, status: int) -> HTMLResponse:
    """A sign-in that did not complete, as a page with the way back to the start.

    The message is the one the sign-in route or the service produced, and it has already been
    held to saying nothing that crossed the wire. The only link is the sign-in page, because
    starting again is the only thing a person here can usefully do.
    """
    from manicule.web.rendering import (  # noqa: PLC0415 - avoids a cycle
        ENVIRONMENT,
        STYLESHEET_PATH,
        html_response,
    )

    body = ENVIRONMENT.get_template("refused.html").render(
        {
            "title": "Not signed in",
            "message": message,
            "status": status,
            "hint": "",
            "sign_in": LOGIN_PAGE,
            "path": request.url.path,
            "stylesheet": STYLESHEET_PATH,
        }
    )
    return html_response(body, status=status)


def signed_in_page(request: Request, signed_in: Mapping[str, JsonValue]) -> HTMLResponse:
    """The page that ends a successful sign-in and sends the browser on. See ``signed_in.html``.

    Sent on to ``/`` rather than to a page of this surface: the front door goes to the browser
    surface when it is served and says what *is* served when it is not, so a sign-in completes
    somewhere real under ``--no-web`` too. ``signed_in`` is the service's ``SignedIn`` payload;
    its token is not read here, and is not on the page.
    """
    from manicule.web.rendering import (  # noqa: PLC0415 - avoids a cycle
        ENVIRONMENT,
        STYLESHEET_PATH,
        html_response,
    )

    del request
    raw_user = signed_in.get("user")
    user = cast("dict[str, JsonValue]", raw_user) if isinstance(raw_user, dict) else {}
    body = ENVIRONMENT.get_template("signed_in.html").render(
        {
            "title": "Signed in",
            "person": str(user.get("name") or user.get("email") or user.get("id") or ""),
            "role": str(user.get("role") or ""),
            "workspace": str(user.get("workspace") or ""),
            "next": "/",
            "stylesheet": STYLESHEET_PATH,
        }
    )
    return html_response(body, status=OK)


def _mode(request: Request) -> str:
    """How this installation authenticates, off the principal this request resolved to."""
    principal: Principal | None = getattr(request.state, "principal", None)
    return principal.identity.mode if principal is not None else ""


UI_PREFIX = "/ui"
"""The prefix a request must be under to be answered with a page rather than an envelope."""


def is_page_request(request: Request) -> bool:
    """Whether this 404 is a browser looking at the browser surface.

    Three conditions, and each excludes something that must keep its envelope:

    * **under ``/ui``** — the JSON API's 404 is part of its contract, and a program parsing
      ``{"detail": "Not Found"}`` must keep getting one;
    * **``GET``** — a ``POST`` to a path that does not exist is a program, or an attempt at an
      operation this surface deliberately does not have, and neither wants HTML;
    * **asks for HTML** — ``fetch()`` from this surface's own script sends ``Accept:
      application/json`` and parses what comes back as an envelope.
    """
    return (
        request.url.path.startswith(UI_PREFIX)
        and request.method == "GET"
        and "text/html" in request.headers.get("accept", "")
    )


def not_found_page(request: Request, exc: Exception) -> HTMLResponse:
    """Render a 404 under ``/ui`` as a page, at the status it already had.

    **An exception handler rather than a catch-all route, and the distinction is the point.**
    A ``GET /ui/{rest:path}`` route would make every path under ``/ui`` *exist*, so
    ``POST /ui/index`` would stop answering "there is no such thing" (404) and start answering
    "not that method" (405). ``tests/web/test_boundaries.py`` accepts either, so it would have
    gone on passing — for the new reason that everything under ``/ui`` matches something, which
    is precisely the assertion it exists to make. Routing is left exactly as it was; only the
    rendering of a 404 that was already happening changes.

    The status is **404**, not a 200 carrying an apology. A page that is not there says so to
    the client as well as to the reader.
    """
    from manicule.web.rendering import (  # noqa: PLC0415 - avoids a cycle
        ENVIRONMENT,
        STYLESHEET_PATH,
        html_response,
    )

    del exc  # the framework's own "Not Found"; the page says it better
    body = ENVIRONMENT.get_template("notfound.html").render(
        {
            "title": "Not found",
            "status": NOT_FOUND,
            # The path only, never the query string — the same rule the refusal page follows,
            # for the same reason: a query value is a place credentials end up.
            "path": request.url.path,
            "stylesheet": STYLESHEET_PATH,
        }
    )
    return html_response(body, status=NOT_FOUND)


__all__ = [
    "LOGIN_PAGE",
    "UI_PREFIX",
    "Guest",
    "Operator",
    "PageRefusedError",
    "Reader",
    "is_page_request",
    "not_found_page",
    "refused_page",
    "sign_in_refused",
    "signed_in_page",
]
