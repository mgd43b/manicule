"""Who is calling, what they may do, and what every response says about framing.

Three things live here, and they are together because each one fails the same way if it is
implemented per-route: quietly, on the one route somebody forgot.

**Identity comes from the service.** :meth:`~manicule.app.service.ApplicationService.authenticate`
decides what a valid key is; this module carries the header to it and turns the answer into a
principal. Nothing about "is this key revoked, expired, or another tenant's" is decided here,
because a rule implemented in a surface is a rule the other surfaces do not have.

**Authorization is a floor, expressed once per route.** ``admin > member > viewer``, and a
route asks for the least it needs. Reads take a viewer; writes take a member; anything that
changes what the installation *is* takes an admin.

**Unauthenticated means loopback, unless an operator said otherwise at a terminal.** When
``security.auth.mode`` is ``none`` there is no credential to check, and the caller is whoever is
sitting at this machine — the same authority the command line has. That is tolerable because a
non-loopback bind with no auth is refused twice: by
:func:`~manicule.app.bind.resolve_bind` before a socket exists, and by
:func:`~manicule.api.app.build_app` before an application exists.

**A browser that signed in presents a session cookie instead of a header.** Under
``security.auth.mode = 'oauth'`` a request carrying no header credential is resolved from the
``manicule_session`` cookie, if it has one — its signature first, then
:meth:`~manicule.app.service.ApplicationService.authenticate_session`. A header always wins: a
request presenting a key is that key, valid or not, whatever cookie the browser also sent. And
the cookie is **not** honored beneath the MCP mount, which is stateless by design
(``docs/surfaces.md`` §6.1) — a protocol client presents its key on every call, and an assistant
running in a browser tab must not inherit whatever the person in that tab is signed in as.
:mod:`manicule.api.cookies` says why each cookie attribute is what it is, and how a cookie
cannot be spent by another site.

``manicule serve --no-authentication`` satisfies both refusals, and it is the one case where
the assumption above does not hold: the anonymous administrator below is then anything that can
route to the port. **Nothing here bounds that, and the honest thing is to say so rather than to
imply a mitigation.** No route gains a guard for it — a guard is a thing to get wrong on one
route — and the surface does not shrink either: the authority an anonymous administrator holds on
that bind is the whole of it, reads and ``document_create`` over both MCP and
``POST /api/v1/documents``, because authoring over a network is the capability the flag exists to
serve.

What stands in its place is not a check but a person. The flag is argv, so no configuration file
can reach it; it is announced at startup naming the corpus that becomes writable; and
``manicule doctor`` reports it as a failing finding for as long as it holds. The operator is
asserting that the network in front of this process is one they own, and manicule cannot verify
that any more than it can verify ``--allow-public-bind``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, Request, WebSocket

from manicule.api.cookies import SESSION_COOKIE, session_token
from manicule.api.proxy import FORWARDED_FOR
from manicule.app.caller import RANK, Caller
from manicule.app.frontdoor import MCP
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

        ``--no-authentication`` is the case where "at a loopback socket" stops being true, and
        this property is deliberately unchanged by it. A role that read the bind would be a
        second authorization rule, disagreeing with this one on whichever surface forgot to
        consult it; the module docstring says where the bound is kept instead.
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
        """Who to record in the audit trail. See :attr:`manicule.app.caller.Caller.actor`."""
        return self.caller.actor

    @property
    def caller(self) -> Caller:
        """This principal as the service sees it, for :func:`manicule.app.caller.acting_as`.

        The unauthenticated caller of an installation with ``auth.mode = none`` is the local
        operator, for the reason :attr:`role` gives; every other caller carries the role this
        request resolved to, so the service can never hold a network caller to less than the
        surface did.
        """
        identity = self.identity
        if not identity.authenticated and identity.mode == AuthMode.NONE.value:
            return Caller(address=self.address)
        return Caller(
            role=self.role,
            key_id=identity.key_id or None,
            user_id=identity.user_id or None,
            address=self.address,
            rate_limit=identity.rate_limit,
        )


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


def beneath_mcp(path: str) -> bool:
    """Whether ``path`` is the MCP mount or anything under it."""
    return path == MCP or path.startswith(f"{MCP}/")


async def identify(
    service: ApplicationService, request: Request | WebSocket, *, address: str = ""
) -> Identity:
    """Who this request is, from a header credential or — for a browser — a session cookie.

    The order is the rule. A header credential is always the answer when one is presented, so
    a program's key is never overridden by a browser's ambient cookie. Only with no header,
    under ``oauth``, and outside the MCP mount is the cookie consulted, and its signature is
    checked before the service is asked anything: a forged or expired cookie is refused for the
    cost of a hash.

    ``address`` is the client address the proxy policy resolved, handed to
    :meth:`~manicule.app.service.ApplicationService.authenticate` so a key limited to
    ``allowed_ips`` is refused from anywhere else. Empty means none could be established, and
    such a key is then refused rather than trusted.
    """
    if isinstance(request, WebSocket):
        header, _ = websocket_token(request)
    else:
        header = token_of(request)
    settings = service.settings
    if (
        not header
        and settings.security.auth.mode is AuthMode.OAUTH
        and SESSION_COOKIE in request.cookies
        and not beneath_mcp(request.url.path)
    ):
        return await service.authenticate_session(
            session_token(settings, request.cookies.get(SESSION_COOKIE))
        )
    return await service.authenticate(header, address=address)


async def resolve(
    service: ApplicationService, policy: ProxyPolicy, request: Request | WebSocket
) -> Principal:
    """Turn a request into a principal. Never raises for a bad credential.

    A missing or unusable key produces an *unauthenticated* principal rather than an error,
    so that the anonymous routes — health, a shared conversation link, the provider list —
    are reachable through the same resolution as everything else. :func:`require` is what
    refuses. :func:`identify` is what decides which credential a request is presenting.
    """
    client = request.client
    address = policy.client_address(
        peer=client.host if client is not None else None,
        # Through the constant, not a literal. The header manicule reads is a decision
        # `manicule.api.proxy` makes once, and a second spelling here is how a rename ends
        # up reading a header nothing sends.
        forwarded_for=request.headers.get(FORWARDED_FOR),
    )
    return Principal(identity=await identify(service, request, address=address), address=address)


def require(principal: Principal, floor: Role) -> Principal:
    """Admit a principal that clears ``floor``, and refuse one that does not.

    Raises:
        UnauthenticatedError: Authentication is configured and no usable key was presented.
        ForbiddenError: A valid key without the authority this route needs.
    """
    identity = principal.identity
    if identity.mode != AuthMode.NONE.value and not identity.authenticated:
        msg = (
            "this installation requires authentication. Present an API key as "
            "'Authorization: Bearer <key>' or 'X-API-Key: <key>'"
        )
        if identity.mode == AuthMode.OAUTH.value:
            # The login routes rather than the browser surface's page, because those exist
            # whether or not the browser surface is served; `/auth/providers` lists them.
            msg += ", or sign in from a browser — GET /auth/providers lists where"
        raise UnauthenticatedError(f"{msg}.")
    if RANK[principal.role] < RANK[floor]:
        credential = {"key": "this key has", "session": "you have"}.get(
            identity.via, "this caller has"
        )
        msg = (
            f"this operation needs the {floor.value!r} role or higher; {credential} "
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
    "beneath_mcp",
    "identify",
    "require",
    "resolve",
    "token_of",
    "websocket_token",
]
