"""The two cookies manicule sets, what each carries, and why each attribute is what it is.

Only a browser that signed in through an identity provider ever holds either, and only under
``security.auth.mode = 'oauth'``. Every other caller presents a key in a header on every
request, which is still the only credential a program can use.

**Both are signed, and the signature is checked before anything is looked up.** Each is an
``itsdangerous`` timed serialization keyed by ``security.auth.session_secret``, with a salt of
its own so that a value minted for one cookie is refused as the other. A forged, altered or
over-age cookie therefore costs a hash, never a database query — which is the difference
between a guessing attack against a signature and one against the session table.

``manicule_session`` — **the session**

* carries a random 256-bit token whose SHA-256 the ``auth_sessions`` table holds; the identity
  itself is never in the cookie, so a session is revoked by revoking its row rather than by
  rotating the key that signs every cookie;
* ``HttpOnly``: no script on the page can read it, so an injection that did run could not carry
  the credential away;
* ``SameSite=Strict``: a browser attaches it to no request another site caused — not a form
  post, not an image, not a top-level link. That is the first half of the cross-site request
  defense; :mod:`manicule.api.origins` refusing an unsafe method from another origin is the
  second, and each holds without the other;
* ``Path=/`` and ``Max-Age`` equal to ``security.auth.session_max_age_s``, which is also when
  the row expires and when the signature stops verifying — three clocks that agree;
* ``Secure`` whenever ``security.transport.enforce_https`` is on.

``manicule_signin`` — **one sign-in in progress**

* carries the ``state`` and PKCE ``code_verifier`` generated when the browser left for the
  provider, and which provider it left for — nothing else, and nothing that identifies anybody;
* ``SameSite=Lax``, **deliberately weaker than the session's**: the provider sends the browser
  back with a cross-site top-level ``GET``, and a ``Strict`` cookie is withheld from exactly
  that navigation, so the callback would never see the state it has to compare. ``Lax`` sends
  it there and on no cross-site sub-request or ``POST``;
* ``Path`` narrowed to the callback, ``Max-Age`` ten minutes, ``HttpOnly``, and ``Secure`` on
  the same terms; deleted by the callback whether the sign-in succeeded or not.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any, cast

from itsdangerous import BadData, URLSafeTimedSerializer

from manicule.app.people import CALLBACK_PATH

if TYPE_CHECKING:
    from starlette.responses import Response

    from manicule.config.settings import Settings

SESSION_COOKIE = "manicule_session"
SIGNIN_COOKIE = "manicule_signin"

SIGNIN_MAX_AGE_S = 600
"""How long a browser has to come back from a provider before the sign-in must start again."""

SIGNIN_PATH = CALLBACK_PATH.split("{", 1)[0].rstrip("/")
"""``/auth/callback``: the only path the sign-in cookie is ever sent to."""

_SESSION_SALT = "manicule.session"
_SIGNIN_SALT = "manicule.signin"


@dataclass(frozen=True, slots=True)
class SignInTransaction:
    """What the browser carries to the provider and back, and nothing else."""

    provider: str
    state: str
    verifier: str


@cache
def _serializer(secret: str, salt: str) -> URLSafeTimedSerializer:
    """One serializer per key and purpose, built once.

    Cached on the key rather than held on the application so that every reader — the
    middleware, the websocket handshake, the sign-out route — reaches the same one without a
    second place for it to be configured.
    """
    return URLSafeTimedSerializer(secret, salt=salt)


def _secret(settings: Settings) -> str:
    value = settings.security.auth.session_secret
    return value.get_secret_value() if value is not None else ""


def session_token(settings: Settings, value: str | None) -> str:
    """The session token a ``manicule_session`` cookie carries, or empty if it carries none.

    Empty for an absent cookie, an unconfigured key, a signature that does not verify, and a
    signature older than ``session_max_age_s`` — and for each of them before any lookup.
    """
    secret = _secret(settings)
    if not value or not secret:
        return ""
    try:
        token = _serializer(secret, _SESSION_SALT).loads(
            value, max_age=settings.security.auth.session_max_age_s
        )
    except BadData:
        return ""
    return token if isinstance(token, str) else ""


def set_session(response: Response, settings: Settings, token: str) -> None:
    """Hand a browser its session, with every attribute the module docstring gives."""
    response.set_cookie(
        SESSION_COOKIE,
        _serializer(_secret(settings), _SESSION_SALT).dumps(token),
        max_age=settings.security.auth.session_max_age_s,
        path="/",
        secure=settings.security.transport.enforce_https,
        httponly=True,
        samesite="strict",
    )


def clear_session(response: Response, settings: Settings) -> None:
    """Tell a browser to forget its session. Its row is revoked separately, on the server."""
    response.delete_cookie(
        SESSION_COOKIE,
        path="/",
        secure=settings.security.transport.enforce_https,
        httponly=True,
        samesite="strict",
    )


def set_signin(response: Response, settings: Settings, transaction: SignInTransaction) -> None:
    """Carry one sign-in's state and verifier to the provider and back."""
    response.set_cookie(
        SIGNIN_COOKIE,
        _serializer(_secret(settings), _SIGNIN_SALT).dumps(
            {
                "provider": transaction.provider,
                "state": transaction.state,
                "verifier": transaction.verifier,
            }
        ),
        max_age=SIGNIN_MAX_AGE_S,
        path=SIGNIN_PATH,
        secure=settings.security.transport.enforce_https,
        httponly=True,
        # Lax, not Strict — see this module's docstring. The return from the provider is a
        # cross-site top-level navigation, which is the one request Strict withholds.
        samesite="lax",
    )


def clear_signin(response: Response, settings: Settings) -> None:
    """Forget the sign-in in progress, whatever became of it."""
    response.delete_cookie(
        SIGNIN_COOKIE,
        path=SIGNIN_PATH,
        secure=settings.security.transport.enforce_https,
        httponly=True,
        samesite="lax",
    )


def signin_transaction(settings: Settings, value: str | None) -> SignInTransaction | None:
    """The sign-in a ``manicule_signin`` cookie carries, or ``None`` if it carries no usable one.

    ``None`` for an absent, forged, altered or expired cookie, and for one whose contents are
    not the three strings this module wrote.
    """
    secret = _secret(settings)
    if not value or not secret:
        return None
    try:
        loaded: object = _serializer(secret, _SIGNIN_SALT).loads(value, max_age=SIGNIN_MAX_AGE_S)
    except BadData:
        return None
    if not isinstance(loaded, dict):
        return None
    fields = cast("dict[str, Any]", loaded)
    provider, state, verifier = (fields.get(key) for key in ("provider", "state", "verifier"))
    if not all(isinstance(item, str) and item for item in (provider, state, verifier)):
        return None
    return SignInTransaction(provider=str(provider), state=str(state), verifier=str(verifier))


__all__ = [
    "SESSION_COOKIE",
    "SIGNIN_COOKIE",
    "SIGNIN_MAX_AGE_S",
    "SIGNIN_PATH",
    "SignInTransaction",
    "clear_session",
    "clear_signin",
    "session_token",
    "set_session",
    "set_signin",
    "signin_transaction",
]
