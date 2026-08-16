"""One operation, one envelope, one status code.

Every route in this package returns the **same object** the command line prints under
``--json`` and the MCP tool hands back: built by :func:`~manicule.app.dispatch.run_op`, over a
payload model from :mod:`manicule.app.results`. A consumer that can read one surface has read
all three, and ``tests/api/test_surface_parity.py`` fails the moment that stops being true.

What HTTP adds is a status code, and it is derived from the error's type rather than chosen at
each raise site. A route that picked its own would be a route with an opinion, and the next
one along would have a different one for the same failure.

**The body is the envelope even when the status is not 200.** A client that reads ``ok`` first
never has to parse two shapes, and a proxy that rewrites bodies on error status codes is a
problem an operator can see rather than one that silently empties the response.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse

from manicule.api.security import ForbiddenError, UnauthenticatedError
from manicule.app.dispatch import error_info, run_op
from manicule.app.results import failed

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from manicule.app.results import Envelope, Payload
    from manicule.app.service import ApplicationService

OK = 200
BAD_REQUEST = 400
UNAUTHORIZED = 401
FORBIDDEN = 403
NOT_FOUND = 404
CONFLICT = 409
UNPROCESSABLE = 422
SERVER_ERROR = 500
SERVICE_UNAVAILABLE = 503

STATUS_BY_ERROR: dict[str, int] = {
    # Keyed by class name, so these two must stay spelled exactly as the classes are. A
    # rename that missed this table would fall through to the default and answer 400 for
    # "you did not authenticate", which is the one status a client branches on.
    "UnauthenticatedError": UNAUTHORIZED,
    "ForbiddenError": FORBIDDEN,
    "PolicyError": FORBIDDEN,
    "UnknownEntityError": NOT_FOUND,
    "UnknownComponentError": NOT_FOUND,
    "UnknownConversationError": NOT_FOUND,
    "NameInUseError": CONFLICT,
    "FingerprintMismatchError": CONFLICT,
    "ConfigError": BAD_REQUEST,
    "ValueError": BAD_REQUEST,
    "CapacityRefusedError": SERVICE_UNAVAILABLE,
    "RebuildStorageError": SERVICE_UNAVAILABLE,
    "RebuildDerivationError": UNPROCESSABLE,
    "RebuildValidationError": UNPROCESSABLE,
    "RebuildLeaseError": CONFLICT,
    "RebuildTerminalError": CONFLICT,
    # A cross-workspace refusal is **not** a client error. Nothing the caller sent could have
    # produced it: a store returned another tenant's row and the surface refused it. That is a
    # defect in this installation, and reporting it as 4xx would file it against the caller.
    "CrossWorkspaceError": SERVER_ERROR,
    "OSError": SERVER_ERROR,
}
"""Which status each failure gets, keyed by the exception class name the envelope carries.

By exact name rather than by ``isinstance``, exactly as :mod:`manicule.app.dispatch` keys its
hints: a subclass added later gets a status somebody chose for it rather than inheriting one
chosen for its parent.
"""


def status_for(envelope: Envelope) -> int:
    """The HTTP status this envelope should be sent with."""
    if envelope.ok:
        return OK
    if envelope.data is not None and envelope.data.get("outcome") == "incomplete":
        return SERVICE_UNAVAILABLE
    kind = envelope.error.type if envelope.error else ""
    return STATUS_BY_ERROR.get(kind, BAD_REQUEST)


def as_response(envelope: Envelope) -> JSONResponse:
    """Serialize an envelope, with the status its outcome implies."""
    return JSONResponse(content=envelope.as_json(), status_code=status_for(envelope))


async def respond(
    op: str, service: ApplicationService, call: Callable[[], Awaitable[Payload]]
) -> JSONResponse:
    """Run one service call and render it.

    The whole of a route's body, in the ordinary case. Anything a route does *besides* calling
    this is behavior living in a surface, which is the thing this layer exists not to have.
    """
    return as_response(await run_op(op, service.workspace, call))


def malformed(op: str, workspace: str, exc: Exception) -> JSONResponse:
    """Render a request the framework rejected before a handler ran.

    FastAPI's own shape for this is ``{"detail": [...]}``, which is a **second** response shape
    on a surface whose whole contract is that there is one. A client that reads ``ok`` first
    would find no ``ok`` at all on the single most common failure it will hit, so the
    validation error is re-dressed as the ordinary envelope.

    The status stays 422 rather than becoming 400. The distinction is worth keeping: 400 is
    "the operation refused what you asked for", 422 is "this never reached the operation".
    """
    return JSONResponse(
        content=failed(op, workspace, error_info(exc)).as_json(), status_code=UNPROCESSABLE
    )


def refusal(op: str, workspace: str, exc: Exception) -> JSONResponse:
    """Render a refusal raised before any service call was made.

    Authentication and authorization are the two of these, and they are deliberately shaped
    like every other failure — same envelope, same fields — so a client has one thing to parse
    rather than a special case for the two failures it will see most often.
    """
    return as_response(failed(op, workspace, error_info(exc)))


AUTH_ERRORS = (UnauthenticatedError, ForbiddenError)
"""The refusals this layer raises itself. Everything else comes from the service."""


__all__ = [
    "AUTH_ERRORS",
    "BAD_REQUEST",
    "CONFLICT",
    "FORBIDDEN",
    "NOT_FOUND",
    "OK",
    "SERVER_ERROR",
    "SERVICE_UNAVAILABLE",
    "STATUS_BY_ERROR",
    "UNAUTHORIZED",
    "UNPROCESSABLE",
    "as_response",
    "malformed",
    "refusal",
    "respond",
    "status_for",
]
