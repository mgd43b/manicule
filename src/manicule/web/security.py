"""Who may read a page, and what a refusal looks like when the reader is a browser.

**The authorisation decision is not made here.** :func:`manicule.api.security.require` is the
one implementation of "does this principal clear this floor", and this module calls it. A
second implementation for the browser surface is exactly the shape of bug this project has
already found once: a rule that held on one surface and not on the one somebody was actually
using.

What *is* decided here is the rendering. A JSON envelope with ``ok: false`` is the right answer
to a program and the wrong one to a person who typed a URL, so the refusal is re-raised as
:class:`PageRefusedError` and rendered as a page — with the same status code, carrying the same
message the API would have sent.

## What a browser cannot do, said plainly

There is no session cookie in this build, deliberately (``manicule.api.routes.auth``): a key is
presented on every request, and a signed cookie would be a second credential type with its own
expiry, revocation and CSRF story. A browser cannot attach a header to a top-level navigation,
so on an installation with ``security.auth.mode = api_key`` a page load carries no credential
and this surface refuses it — with a page that says so, and says what to use instead.

That is a real limitation and it is written down rather than worked around. The configuration
this surface is *for* is the one manicule defaults to and ships as: a single person, on
loopback, with ``auth.mode = none``, where the caller is the operator at this machine and holds
the same authority the command line does. An interactive login belongs to team mode
([#13](https://github.com/mgd43b/manicule/issues/13)), which is where the session, its
revocation and its CSRF story can be designed together instead of one of the three appearing
here on its own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, Request

from manicule.api.security import (
    ForbiddenError,
    Principal,
    UnauthenticatedError,
    anonymous,
    require,
)
from manicule.config.settings import Role
from manicule.core.errors import ManiculeError

if TYPE_CHECKING:
    from fastapi.responses import HTMLResponse


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
where, and the key list is the installation's identities. The HTTP routes behind these pages
take an admin, and a page that took less would be a way round them.
"""

Guest = Annotated[Principal, Depends(anonymous)]
"""No credential at all. One page uses it: the shared conversation, which is a bearer URL."""


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
    status = UNAUTHORIZED if isinstance(cause, UnauthenticatedError) else FORBIDDEN
    body = ENVIRONMENT.get_template("refused.html").render(
        {
            "title": "Not permitted",
            "message": str(cause),
            "status": status,
            # The path only. Never the query string: a share token is a path segment on one
            # route and a query value is a place credentials end up, and a refusal page is not
            # where either belongs.
            "path": request.url.path,
            "stylesheet": STYLESHEET_PATH,
        }
    )
    return html_response(body, status=status)


__all__ = ["Guest", "Operator", "PageRefusedError", "Reader", "refused_page"]
