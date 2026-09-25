"""Signing in through Google and GitHub: the protocol, and nothing that decides anything.

Everything here is a conversation with an identity provider — building the URL a browser is
sent to, exchanging the code it comes back with, and asking the provider who that was. Whether
the person is admitted, what role they get and whether their membership is disabled are the
service's decisions (:meth:`~manicule.app.service.ApplicationService.sign_in`), because a rule
implemented in a surface is a rule the other surfaces do not have.

**authlib, not a hand-rolled client.** Every failure mode of this protocol is a security
failure, and the library has already met them: state, PKCE, client authentication at the token
endpoint and the shape of a provider's error response. What this module adds is the two
providers' endpoints and the reading of their answers.

**PKCE with S256 on every sign-in, and a fresh random ``state``.** The verifier never leaves
this installation except to the token endpoint, so an authorization code intercepted on the way
back — by a browser extension, a proxy log, a history entry — cannot be exchanged by whoever
intercepted it. ``state`` binds the callback to the browser that started the sign-in, and it is
compared in constant time by the route that receives it.

**The network is injectable.** Every request goes through the ``transport`` the caller passes,
which is ``None`` — the real network — in a served process and an ``httpx2.MockTransport``
standing in for the provider under test, so no test of this module can reach the internet.

**httpx2, not httpx, because that is what authlib runs on.** From 1.8 authlib's client is an
``httpx2.AsyncClient`` whenever ``httpx2`` is importable, falling back to ``httpx`` only without
it, and ``manicule[serve]`` declares ``httpx2``. The transport handed in and the errors caught
below have to belong to the same library as the client: an ``httpx`` transport fails an
assertion inside ``httpx2`` on its first response, and ``except httpx.HTTPError`` lets an
unreachable provider through as a 500 rather than a refusal naming the step.

It is a package rather than a module for one reason: authlib ships no type information, and the
type checker's relaxation for an untyped dependency is scoped by directory, exactly as it is for
the parsing and embedding runtimes. Nothing outside this directory touches authlib, and what it
exports is fully typed.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Self, cast

import httpx2
from authlib.common.errors import AuthlibBaseError
from authlib.integrations.httpx_client import AsyncOAuth2Client

from manicule.api.cookies import SignInTransaction
from manicule.app.people import Profile
from manicule.core.errors import ManiculeError

if TYPE_CHECKING:
    from types import TracebackType

    from manicule.config.settings import OAuthProvider


class _Client(Protocol):
    """The part of authlib's ``AsyncOAuth2Client`` this module uses, stated.

    The library declares none of it — its client inherits from ``httpx2.AsyncClient`` through a
    path the type checker cannot follow — so the one construction site casts to this, and every
    call below is checked against a signature somebody wrote down rather than against nothing.
    """

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...

    def create_authorization_url(
        self, url: str, state: str | None = None, code_verifier: str | None = None
    ) -> tuple[str, str]: ...

    async def fetch_token(
        self,
        url: str,
        *,
        grant_type: str,
        code: str,
        code_verifier: str,
        headers: dict[str, str],
    ) -> object: ...

    async def get(self, url: str) -> httpx2.Response: ...


@dataclass(frozen=True, slots=True)
class Endpoints:
    """Where one provider's authorization and token endpoints are, and what is asked of it."""

    authorize: str
    token: str
    scope: str


GOOGLE = Endpoints(
    authorize="https://accounts.google.com/o/oauth2/v2/auth",
    token="https://oauth2.googleapis.com/token",  # noqa: S106 - an endpoint, not a credential
    # `openid` so that `sub` — the stable account id — is what userinfo reports; `email` and
    # `profile` for the address and the name, and nothing that reads anybody's data.
    scope="openid email profile",
)
GOOGLE_USERINFO = "https://openidconnect.googleapis.com/v1/userinfo"

GITHUB = Endpoints(
    authorize="https://github.com/login/oauth/authorize",
    token="https://github.com/login/oauth/access_token",  # noqa: S106 - an endpoint
    # `user:email` because the address on the profile is only the one a person chose to make
    # public; the verified primary address is on a separate endpoint behind this scope.
    scope="read:user user:email",
)
GITHUB_USER = "https://api.github.com/user"
GITHUB_EMAILS = "https://api.github.com/user/emails"

ENDPOINTS: dict[str, Endpoints] = {"google": GOOGLE, "github": GITHUB}
"""Every provider type manicule speaks to, by the type configuration names it with."""

TIMEOUT_S = 10.0
"""The most one request to a provider may take. A sign-in is a person waiting at a browser."""

TOKEN_HEADERS = {
    # GitHub answers the token request as form-encoded text unless asked for JSON.
    "Accept": "application/json",
    "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
}


class SignInFailedError(ManiculeError):
    """The conversation with the provider did not produce a person.

    The message names the provider and what kind of step failed, never a value that crossed the
    wire: an authorization code, a token and a provider's free-text error description are all
    things a refusal page and a log line must not carry.
    """


async def begin(provider: OAuthProvider) -> tuple[str, SignInTransaction]:
    """The URL to send a browser to, and the transaction it must bring back.

    Makes no request: an authorization URL is built, not fetched. The client is still opened
    and closed properly, because it is the same client :func:`complete` uses, configured once —
    in particular with ``S256``, which is what makes it put a challenge on the URL at all.
    """
    endpoints = ENDPOINTS[provider.type]
    state = secrets.token_urlsafe(32)
    # 64 characters of URL-safe alphabet, inside RFC 7636's 43-128.
    verifier = secrets.token_urlsafe(48)
    async with _client(provider, transport=None) as client:
        url, _ = client.create_authorization_url(
            endpoints.authorize, state=state, code_verifier=verifier
        )
    return url, SignInTransaction(provider=provider.type, state=state, verifier=verifier)


async def complete(
    provider: OAuthProvider,
    transaction: SignInTransaction,
    code: str,
    *,
    transport: httpx2.AsyncBaseTransport | None = None,
) -> Profile:
    """Exchange ``code`` for a token, and ask the provider who it belongs to.

    Raises:
        SignInFailedError: Any step failed — the exchange was refused, the provider could not
            be reached, or its answer did not carry an account id.
    """
    endpoints = ENDPOINTS[provider.type]
    client = _client(provider, transport=transport)
    step = "exchanging the authorization code"
    try:
        async with client:
            await client.fetch_token(
                endpoints.token,
                grant_type="authorization_code",
                code=code,
                code_verifier=transaction.verifier,
                headers=TOKEN_HEADERS,
            )
            step = "asking the provider who signed in"
            if provider.type == "google":
                return await _google(client)
            return await _github(client)
    except AuthlibBaseError as exc:
        # The library's own base, which both the provider's refusal (`OAuthError`) and the
        # protocol checks it makes before sending anything derive from. Only the short error
        # code is kept; a provider's description is free text another site chose.
        error = str(getattr(exc, "error", "") or "an error")
        msg = f"the {provider.type} sign-in failed while {step}: the provider answered {error}"
        raise SignInFailedError(msg) from exc
    except (httpx2.HTTPError, ValueError, KeyError, TypeError) as exc:
        msg = f"the {provider.type} sign-in failed while {step}: {type(exc).__name__}"
        raise SignInFailedError(msg) from exc


def _client(provider: OAuthProvider, *, transport: httpx2.AsyncBaseTransport | None) -> _Client:
    """One OAuth client for one sign-in, closed when that sign-in is.

    ``client_secret_post`` at the token endpoint because both providers accept it and GitHub
    documents no other. ``redirect_uri`` is the configured one, never one built from the
    request, for the reason ``OAuthProvider.redirect_uri`` gives.
    """
    client = AsyncOAuth2Client(
        client_id=provider.client_id,
        client_secret=provider.client_secret.get_secret_value(),
        token_endpoint_auth_method="client_secret_post",  # noqa: S106 - a method name, not a secret
        scope=ENDPOINTS[provider.type].scope,
        redirect_uri=provider.redirect_uri,
        code_challenge_method="S256",
        timeout=TIMEOUT_S,
        transport=transport,
        headers={"User-Agent": "manicule"},
    )
    return cast("_Client", client)


async def _json(client: _Client, url: str) -> object:
    response = await client.get(url)
    response.raise_for_status()
    return cast("object", response.json())


async def _google(client: _Client) -> Profile:
    """Google's OpenID Connect userinfo: ``sub`` is the account, ``email_verified`` is its word."""
    body = _mapping(await _json(client, GOOGLE_USERINFO))
    verified = body.get("email_verified")
    return Profile(
        provider="google",
        subject=_subject(body.get("sub")),
        email=str(body.get("email") or ""),
        email_verified=verified is True or str(verified).lower() == "true",
        name=str(body.get("name") or ""),
    )


async def _github(client: _Client) -> Profile:
    """GitHub's user, and the one address it marks primary — verified only if GitHub says so.

    The numeric ``id`` is the account; ``login`` can be changed and later claimed by somebody
    else, so it is a display name here and never an identity.
    """
    user = _mapping(await _json(client, GITHUB_USER))
    listed = await _json(client, GITHUB_EMAILS)
    entries = cast("list[object]", listed) if isinstance(listed, list) else []
    primary = next(
        (entry for entry in map(_mapping, entries) if entry.get("primary") is True),
        cast("dict[str, Any]", {}),
    )
    return Profile(
        provider="github",
        subject=_subject(user.get("id")),
        email=str(primary.get("email") or ""),
        email_verified=primary.get("verified") is True,
        name=str(user.get("name") or user.get("login") or ""),
    )


def _mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        msg = "the provider answered with something other than an object"
        raise TypeError(msg)
    return cast("dict[str, Any]", value)


def _subject(value: object) -> str:
    """The account id, as text. A profile without one names nobody and is refused."""
    if isinstance(value, bool) or value is None or str(value).strip() == "":
        msg = "the provider's answer carries no account id"
        raise KeyError(msg)
    return str(value)


__all__ = [
    "ENDPOINTS",
    "GITHUB",
    "GITHUB_EMAILS",
    "GITHUB_USER",
    "GOOGLE",
    "GOOGLE_USERINFO",
    "Endpoints",
    "SignInFailedError",
    "begin",
    "complete",
]
