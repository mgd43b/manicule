"""Who is calling, what they may do, and what every response says about framing.

Three things live here, and they are together because each one fails the same way if it is
implemented per-route: quietly, on the one route somebody forgot.

**Identity comes from the service.** :meth:`~manicule.app.service.ApplicationService.authenticate`
decides what a valid key is; this module carries the header to it and turns the answer into a
principal. Nothing about "is this key revoked, expired, or another tenant's" is decided here,
because a rule implemented in a surface is a rule the other surfaces do not have.

**Authorisation is a floor, expressed once per route.** ``admin > member > viewer``, and a
route asks for the least it needs. Reads take a viewer; writes take a member; anything that
changes what the installation *is* takes an admin.

**Unauthenticated means loopback.** When ``security.auth.mode`` is ``none`` there is no
credential to check, and the caller is whoever is sitting at this machine — the same authority
the command line has. That is only tolerable because a non-loopback bind with no auth is
refused twice: by :func:`~manicule.app.bind.resolve_bind` before a socket exists, and by
:func:`~manicule.api.app.build_app` before an application exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, Request, WebSocket

from manicule.api.proxy import FORWARDED_FOR
from manicule.app.results import Identity
from manicule.config.settings import AuthMode, Role
from manicule.core.errors import ManiculeError

if TYPE_CHECKING:
    from manicule.api.proxy import ProxyPolicy
    from manicule.app.service import ApplicationService

BEARER = "bearer "
"""The ``Authorization`` scheme manicule accepts, lower-cased for comparison."""

API_KEY_HEADER = "x-api-key"
"""The alternative to ``Authorization: Bearer``, for clients that reserve the former.

Both are headers. Neither is a query parameter, and that is deliberate: a credential in a URL
is a credential in the access log, in the browser history, and in the ``Referer`` of every
link the page loads.
"""

WEBSOCKET_SUBPROTOCOL_PREFIX = "manicule.api-key."
"""How a browser presents a key on a websocket handshake.

A browser cannot set headers on a ``WebSocket``, and the usual workaround — a token in the
query string — writes the credential into the server's access log. The subprotocol header is
the one field a browser *can* set, so the key travels there and the server echoes the chosen
subprotocol back.
"""

_RANK: dict[Role, int] = {Role.VIEWER: 0, Role.MEMBER: 1, Role.ADMIN: 2}
"""Least authority first. A route asks for a floor, not for an exact role."""


class UnauthenticatedError(ManiculeError):
    """No usable credential was presented."""


class ForbiddenError(ManiculeError):
    """A valid credential without the authority this route needs."""


@dataclass(frozen=True, slots=True)
class Principal:
    """The caller, as this request resolved them."""

    identity: Identity
    address: str = ""
    """Where the request came from, decided by :class:`~manicule.api.proxy.ProxyPolicy`.

    Empty when nothing trustworthy could be established. Empty is a real answer: an audit row
    that says "unknown" is honest, and one that repeats a caller-supplied header is not.
    """

    @property
    def role(self) -> Role:
        """What this caller may do.

        An unauthenticated principal on an installation with ``auth.mode = none`` is an
        **admin**, because it is the operator at a loopback socket and they already have the
        command line. An unauthenticated principal on an installation *with* auth configured
        never reaches a route at all — :func:`require` refuses first.
        """
        if not self.identity.authenticated:
            return Role.ADMIN if self.identity.mode == AuthMode.NONE.value else Role.VIEWER
        try:
            return Role(self.identity.role)
        except ValueError:
            # A role the enum does not have. Refuse to guess upward: an unknown role is the
            # least authority, not the most.
            return Role.VIEWER

    @property
    def actor(self) -> str:
        """Who to record in the audit trail. The key's id, or the local operator."""
        return self.identity.key_id or ("local" if not self.identity.authenticated else "")


def token_of(request: Request | WebSocket) -> str:
    """The credential this request presented, from a header and never from a URL."""
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith(BEARER):
        return authorization[len(BEARER) :].strip()
    return request.headers.get(API_KEY_HEADER, "").strip()


def websocket_token(websocket: WebSocket) -> tuple[str, str | None]:
    """The credential on a websocket handshake, and the subprotocol to echo back.

    Returns the token and the exact subprotocol string the client offered, because a server
    that accepts a handshake without echoing the chosen subprotocol makes the browser close
    the connection immediately — a failure that looks like a network problem and is an
    authentication one.
    """
    header = token_of(websocket)
    if header:
        return header, None
    offered = websocket.headers.get("sec-websocket-protocol", "")
    for piece in (part.strip() for part in offered.split(",")):
        if piece.startswith(WEBSOCKET_SUBPROTOCOL_PREFIX):
            return piece[len(WEBSOCKET_SUBPROTOCOL_PREFIX) :], piece
    return "", None


async def resolve(
    service: ApplicationService, policy: ProxyPolicy, request: Request | WebSocket
) -> Principal:
    """Turn a request into a principal. Never raises for a bad credential.

    A missing or unusable key produces an *unauthenticated* principal rather than an error,
    so that the anonymous routes — health, a shared conversation link, the provider list —
    are reachable through the same resolution as everything else. :func:`require` is what
    refuses.
    """
    client = request.client
    return Principal(
        identity=await service.authenticate(token_of(request)),
        address=policy.client_address(
            peer=client.host if client is not None else None,
            # Through the constant, not a literal. The header manicule reads is a decision
            # `manicule.api.proxy` makes once, and a second spelling here is how a rename ends
            # up reading a header nothing sends.
            forwarded_for=request.headers.get(FORWARDED_FOR),
        ),
    )


def require(principal: Principal, floor: Role) -> Principal:
    """Admit a principal that clears ``floor``, and refuse one that does not.

    Raises:
        UnauthenticatedError: Authentication is configured and no usable key was presented.
        ForbiddenError: A valid key without the authority this route needs.
    """
    if principal.identity.mode != AuthMode.NONE.value and not principal.identity.authenticated:
        msg = (
            "this installation requires authentication. Present an API key as "
            "'Authorization: Bearer <key>' or 'X-API-Key: <key>'."
        )
        raise UnauthenticatedError(msg)
    if _RANK[principal.role] < _RANK[floor]:
        msg = (
            f"this operation needs the {floor.value!r} role or higher; this key has "
            f"{principal.role.value!r}."
        )
        raise ForbiddenError(msg)
    return principal


def _dependency(floor: Role):  # noqa: ANN202 - the return type is FastAPI's own callable
    async def guard(request: Request) -> Principal:
        principal: Principal | None = getattr(request.state, "principal", None)
        if principal is None:  # pragma: no cover - the middleware always resolves one
            msg = "the request reached a route without a resolved principal"
            raise ManiculeError(msg)
        return require(principal, floor)

    return guard


ViewerPrincipal = Annotated[Principal, Depends(_dependency(Role.VIEWER))]
"""Any authenticated caller. Reads."""

MemberPrincipal = Annotated[Principal, Depends(_dependency(Role.MEMBER))]
"""Writes to the corpus and to a workspace's own objects."""

AdminPrincipal = Annotated[Principal, Depends(_dependency(Role.ADMIN))]
"""Operations that change what the installation is, or read across everything in it."""


def anonymous(request: Request) -> Principal:
    """The principal for a route that requires no credential at all.

    Explicit rather than absent. A route with no dependency is a route nobody can tell apart
    from one where the dependency was forgotten, and two of the four routes reachable this way
    serve conversation content.
    """
    principal: Principal | None = getattr(request.state, "principal", None)
    if principal is None:  # pragma: no cover - the middleware always resolves one
        msg = "the request reached a route without a resolved principal"
        raise ManiculeError(msg)
    return principal


AnonymousPrincipal = Annotated[Principal, Depends(anonymous)]


__all__ = [
    "API_KEY_HEADER",
    "BEARER",
    "WEBSOCKET_SUBPROTOCOL_PREFIX",
    "AdminPrincipal",
    "AnonymousPrincipal",
    "ForbiddenError",
    "MemberPrincipal",
    "Principal",
    "UnauthenticatedError",
    "ViewerPrincipal",
    "anonymous",
    "require",
    "resolve",
    "token_of",
    "websocket_token",
]
