"""What a request authenticates with — asked for per request, not built once.

The client used to build one ``Authorization`` header in ``setup()`` and hand it to the HTTP
client, on the stated principle that a credential is "the credential, and nothing that varies
per request". That held while every credential was a string with no lifetime. A browser session
is not: it expires on the instance's schedule, it is renewed by a person going to a browser, and
between one request and the next it can stop being a credential at all.

So the seam is an object the client **consults**, and the two things it can do are the two
things a credential with a lifetime has to be able to do:

- :meth:`Credential.authorize` returns the headers for *this* request, and may refuse. A
  session past the age this connector will trust raises :class:`SessionExpiredError` here,
  before the request goes out, so a sync that outlives its own session stops rather than
  continuing on a credential manicule no longer believes in.
- :meth:`Credential.renewal` says what a person must do about it, in the terms of *that*
  credential. "Rotate the token" and "sign in again in your browser" are different acts, and a
  message that names the wrong one sends somebody to the wrong place.

**And a credential is asked where the session is, rather than handed one.** A browser session is
not only renewed on a schedule nobody here sets — it is *replaced*, by a person signing in and a
running server being handed the result. The connector consulting the credential was built once
and is cached for the life of the process, so a credential closing over the session it found at
construction goes on offering that one until a restart. :class:`HeldSessionCredential` is the
class that does not: it holds the place sessions live and reads it per request, so a connector
built an hour ago authenticates with the session handed over a minute ago.

**Nothing here imports an HTTP client**, for the same reason :mod:`manicule.connectors.config`
does not: this module is reachable from the plugin factory, which runs before configuration is
read.

**No credential is stored as a plain string.** The secret is held in a
:class:`~pydantic.SecretStr` and the header is assembled at the moment of use, so a repr, a
traceback frame or a logged configuration object carries the wrapper rather than the value.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import logging
import math
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from typing import TYPE_CHECKING, Protocol, cast, override

from pydantic import SecretStr

from manicule.connectors.config import AuthMethod, ConfluenceConfig, GmailConfig
from manicule.connectors.errors import (
    ConnectorError,
    CredentialError,
    GrantRevokedError,
    MailboxMismatchError,
    RateLimitedError,
    RemoteError,
    RequestTimeoutError,
    SessionExpiredError,
    TokenRejectedError,
)
from manicule.core.errors import ConfigError

if TYPE_CHECKING:  # pragma: no cover - the store imports this module back at run time
    import httpx

    from manicule.connectors.sessions import SessionStore

_log = logging.getLogger(__name__)

__all__ = [
    "GMAIL_PROFILE_URL",
    "GOOGLE_TOKEN_URL",
    "Authorization",
    "BrowserSession",
    "BrowserSessionCredential",
    "Credential",
    "HeldSessionCredential",
    "RefreshTokenCredential",
    "TokenCredential",
    "credential_for",
    "gmail_credential",
    "token_credential",
]


@dataclass(frozen=True, slots=True)
class Authorization:
    """The headers one request carries in order to be this account, and nothing else.

    Headers rather than a cookie jar. The client does not follow a redirect without checking
    where it points (``docs/connectors/confluence.md`` §2), so a header cannot travel to another
    host by being carried along a redirect chain — which is the one thing a jar would have
    given, and it is given here by the origin check instead.

    **The account travels with the headers rather than being asked for afterwards**, and that is
    what keeps a credential which can be renewed mid-run coherent. The response check compares
    ``X-AUSERNAME`` against who manicule expected to be; a client that read the credential once
    to build the request and again to check the reply could be handed a fresh session in between
    and would then refuse its own answer for being somebody else's. One reading produces one
    pair, and the check reads the half that was sent.
    """

    headers: Mapping[str, str]

    account: str = ""
    """Who the request is being made as, if that is known, else ``""``.

    Confluence Server and Data Center name the authenticated user on every REST response
    (``X-AUSERNAME``). Knowing who manicule *expects* to be turns that header into a check: a
    response that says ``anonymous``, or says somebody else, is a session that has stopped
    being this account's whether or not the status code admits it.
    """


class Credential(Protocol):
    """Something a request can be made as. Consulted per request; may refuse."""

    def authorize(self) -> Authorization:
        """Headers for the request about to be made, and who they make it as.

        Raises:
            SessionExpiredError: This credential has a lifetime and has outlived it. Raised
                before the request rather than after a response, because a request made with a
                dead session is answered with a sign-in page rather than an error.
            SessionMissingError: This credential is whatever session the process is holding for
                an instance, and it is holding none — nobody has signed in yet, or the one it
                had has been forgotten. Raised here rather than reported once at construction,
                because both of those can happen while a connector is alive.
        """
        ...

    def describe(self) -> str:
        """What this credential is, for a message about it being rejected."""
        ...

    def renewal(self) -> str:
        """What a person does to replace it. One sentence, imperative."""
        ...


@dataclass(frozen=True, slots=True)
class TokenCredential:
    """A Cloud API token or a Server/Data Center personal access token.

    One class for both because they differ only in how the same held secret is spelled into a
    header, and a second class would have been a second place to get the spelling wrong.
    """

    method: AuthMethod
    secret: SecretStr
    subject: str = ""
    """Cloud's ``email``. Empty for a personal access token, which carries its own identity."""

    token_env: str = ""
    """The environment variable this token may have come from, named when it is rejected."""

    def authorize(self) -> Authorization:
        """The header this token is spelled into, as nobody the response check knows.

        ``account`` is left empty deliberately, even for Cloud where :attr:`subject` holds an
        email. The account is compared against ``X-AUSERNAME``, which is the instance's own name
        for the authenticated user — an account id on Cloud, a username on Server. Cloud's Basic
        credential is spelled with an email address, which is a different identifier for the
        same person, and offering it here would make every Cloud response look as though it came
        back as somebody else. Only a session captured *from* an instance knows a name that
        instance will repeat.
        """
        value = self.secret.get_secret_value()
        if self.method is AuthMethod.API_TOKEN:
            pair = f"{self.subject}:{value}".encode()
            return Authorization({"Authorization": f"Basic {base64.b64encode(pair).decode()}"})
        return Authorization({"Authorization": f"Bearer {value}"})

    def describe(self) -> str:
        if self.method is AuthMethod.API_TOKEN:
            return f"an API token for {self.subject or '(no email configured)'}"
        return "a personal access token"

    def renewal(self) -> str:
        where = f" and that ${self.token_env} holds the right one" if self.token_env else ""
        return f"Check the token has not been revoked{where}."


@dataclass(frozen=True, slots=True)
class BrowserSession:
    """The cookies of a signed-in browser session, and what is known about them.

    This is the whole of what manicule keeps. There is no password in it, because manicule never
    sees one: a person signs in to their own browser against their own identity provider, and
    what crosses into manicule is the result of that, never the act.

    ``captured_at`` rather than an expiry, because a session cookie carries no expiry a client
    can read — the instance decides when the session dies and announces it only by answering the
    next request with a sign-in page. An age manicule measures itself is the only thing that can
    turn "this is too old to try" into a refusal *before* a sync starts.

    **There is no serialization on this class, and its absence is the design.** ``to_json`` and
    ``from_json`` existed so that a session could be written into a keychain item and read back;
    nothing persists a session now, so a method that turns one into a string would be a loaded
    gun in a module whose whole claim is that the value never leaves memory. The one place
    cookies are deliberately unwrapped is
    :meth:`~manicule.app.control.Handover.to_line`, which writes them to a ``0600`` socket and
    is named for exactly that.
    """

    base_url: str
    """The instance this belongs to. A session is not portable between instances, and storing
    it under the site it came from is what stops one being offered to another."""

    account: str
    """Who the instance said this was when it was captured, checked against every response."""

    captured_at: datetime
    cookies: Mapping[str, SecretStr] = field(default_factory=dict[str, SecretStr])


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


_RENEW_IN_BROWSER = "Sign in to Confluence in your browser and re-run `manicule connector login`."
"""What a person does about a browser session, in one place.

Both credentials built over a session say it, and a second copy is how two sentences that have to
agree come to differ by a word an operator then searches for and does not find.
"""


def _nothing_held(base_url: str) -> str:
    """Why there is no session for ``base_url``, and what to do about it.

    A function rather than a literal at its raise site, because there are now two: a connector
    being built before anybody has signed in, and a connector *already built* whose session has
    since been forgotten. They are the same fact about the process and an operator should not
    have to notice that manicule described it two ways.
    """
    return (
        f"no Confluence browser session is held for {base_url}. Sessions live in the running "
        f"manicule server's memory and nowhere else — no keychain, no file, no environment "
        f"variable — so there is none until somebody signs in, and there is none again after "
        f"the server restarts. That is the expected path rather than a fault. Start a server "
        f"with `manicule serve` if one is not running, then run `manicule connector login "
        f"<name> --browser`; manicule never asks for the password and never sees it."
    )


@dataclass(frozen=True, slots=True)
class BrowserSessionCredential:
    """One :class:`BrowserSession` used as a credential, with an age this connector will trust.

    The age check is in :meth:`authorize`, which is to say it runs **before every request**
    rather than once at startup. A first sync of a large corpus can outlast the session it
    began under, and the alternative to checking each time is a run that keeps making requests
    with a credential manicule has already decided is too old — and gets sign-in pages back.

    **This one is a snapshot, and that is what it is for.** It speaks for the session it was
    given and no other, which is exactly right for the two callers that have a session in hand:
    :func:`~manicule.connectors.sessions.capture_cookies`, proving a candidate before anything
    stores it, and :class:`HeldSessionCredential`, which builds one per request over whatever
    the process is holding at that moment. A connector is handed the latter, because a
    connector outlives the session it was built with.
    """

    session: BrowserSession
    max_age: timedelta
    now: Callable[[], datetime] = _utcnow

    def expires_at(self) -> datetime:
        return self.session.captured_at + self.max_age

    def authorize(self) -> Authorization:
        """The session's cookies, if manicule still trusts their age.

        Raises:
            SessionExpiredError: The session is older than ``session_max_age_hours``.
        """
        expires = self.expires_at()
        moment = self.now()
        if moment >= expires:
            age = (moment - self.session.captured_at).total_seconds() / 3600
            msg = (
                f"the Confluence browser session for {self.session.base_url} was captured "
                f"{age:.1f} hours ago and this connector will not use one past "
                f"{self.max_age.total_seconds() / 3600:.1f} hours "
                f"(session_max_age_hours). {self.renewal()} Nothing is lost by stopping here: "
                f"a run that does not finish does not advance its watermark, so the next one "
                f"resumes."
            )
            raise SessionExpiredError(msg)
        jar = "; ".join(
            f"{name}={secret.get_secret_value()}" for name, secret in self.session.cookies.items()
        )
        return Authorization({"Cookie": jar}, account=self.session.account)

    def describe(self) -> str:
        signed_in = f" signed in as {self.session.account}" if self.session.account else ""
        return f"a browser session{signed_in}"

    def renewal(self) -> str:
        return _RENEW_IN_BROWSER


@dataclass(frozen=True, slots=True)
class HeldSessionCredential:
    """Whatever session this process is holding for one instance, read again for every request.

    **A connector outlives the session it was built with, and this is where that is
    survivable.** A connector is constructed once and cached for the life of the process
    (:meth:`~manicule.container.container.Container.connector`), because it carries a watermark
    across a run and two objects for one source would each advance their own. The session it
    authenticates with has no such lifetime: it is replaced from outside whenever somebody signs
    in again and ``connector login`` hands the result to the running server.

    A credential closing over the session it found at construction makes those two facts
    contradict each other, and the symptom is the worst shape available — a sign-in that
    *reports success*. The server says it is holding the new session, ``doctor`` agrees, and
    every sync goes on failing against the old one until somebody restarts the server. Restarting
    is precisely what a session held in memory is meant to make unnecessary.

    So this holds no session. It holds the configuration and the store, and every
    :meth:`authorize` is a fresh reading of both — the same thing
    :meth:`BrowserSessionCredential.authorize` already did for the *age* of a session, applied
    one level up to *which* session.

    **One reading per request, not two.** :meth:`authorize` builds a
    :class:`BrowserSessionCredential` over what it found and answers from that, so the cookies a
    request carries and the account its response is checked against come from one snapshot. A
    hand-over landing between the two readings would otherwise make a connector refuse its own
    reply for being somebody else's — a mixed identity check, reported as an expired session,
    at the moment a person had just successfully signed in.

    **Nothing is cached for a forget to miss.** ``connector login --forget`` empties the vault
    and the next :meth:`authorize` finds nothing and raises, rather than a connector carrying on
    with the copy it took at construction.
    """

    config: ConfluenceConfig
    """The connector's own settings: which instance to look up, and how old a session it will
    still use. Read per request rather than resolved once, so ``session_max_age_hours`` is
    enforced against every session this ever hands out and not only against the first."""

    store: SessionStore | None = None
    """Where to look. ``None`` means this process's own vault, resolved at each reading rather
    than at construction — so a suite that substitutes
    :func:`~manicule.connectors.sessions.default_store` substitutes it for credentials built
    before the substitution as well as after."""

    now: Callable[[], datetime] = _utcnow

    @property
    def max_age(self) -> timedelta:
        """How old a session this connector will still use, from its own configuration."""
        return timedelta(hours=self.config.session_max_age_hours)

    def authorize(self) -> Authorization:
        """The cookies of the session held right now, if manicule still trusts their age.

        Raises:
            SessionMissingError: This process is holding no session for the instance — nobody
                has signed in since it started, or the session it had has been forgotten.
            SessionExpiredError: The session it is holding is older than
                ``session_max_age_hours``.
        """
        return self._current().authorize()

    def describe(self) -> str:
        """What this is, for a message about a request being rejected.

        Answers rather than raising when there is no session, because the one caller is
        :meth:`~manicule.connectors.client.ConfluenceClient._credential_message`, assembling the
        sentence that explains a 401. A credential that raised while being described would
        replace a rejection an operator can act on with a traceback about a different problem.
        """
        held = self._held()
        if held is None:
            return "a browser session, which this process is no longer holding"
        return held.describe()

    def renewal(self) -> str:
        return _RENEW_IN_BROWSER

    def _held(self) -> BrowserSessionCredential | None:
        """A snapshot over the session held now, or ``None`` if there is none."""
        from manicule.connectors.sessions import load_session  # noqa: PLC0415 - see above

        session = load_session(self.config, store=self.store)
        if session is None:
            return None
        return BrowserSessionCredential(session=session, max_age=self.max_age, now=self.now)

    def _current(self) -> BrowserSessionCredential:
        held = self._held()
        if held is None:
            from manicule.connectors.errors import SessionMissingError  # noqa: PLC0415

            raise SessionMissingError(_nothing_held(self.config.base_url))
        return held


def token_credential(config: ConfluenceConfig) -> Credential:
    """The credential for a configuration that authenticates with a token.

    Split from :func:`credential_for` so that the client can build its own default without
    reaching the session store: a token is entirely determined by the configuration in hand, and
    a browser session is not.

    Raises:
        ConfigError: This configuration authenticates with a browser session, which cannot be
            derived from configuration, or the token :func:`resolve_credentials` should have
            filled in is missing.
    """
    method = config.auth_method
    if method is AuthMethod.BROWSER_SESSION:
        msg = (
            "this Confluence connector authenticates with a browser session, which is held in "
            "the running server's memory rather than in configuration. Build it with "
            "manicule.connectors.credentials.credential_for(), which the plugin factory calls "
            "before constructing the connector."
        )
        raise ConfigError(msg)
    secret = config.api_token if method is AuthMethod.API_TOKEN else config.personal_access_token
    if secret is None:
        msg = (
            f"no {method.value} is set for {config.base_url}. resolve_credentials() fills it "
            f"from ${config.token_env} and refuses when it is absent; a caller building this "
            f"connector some other way has to call it too."
        )
        raise ConfigError(msg)
    return TokenCredential(
        method=method,
        secret=secret,
        subject=config.email,
        token_env=config.token_env,
    )


def credential_for(
    config: ConfluenceConfig,
    *,
    store: SessionStore | None = None,
    now: Callable[[], datetime] = _utcnow,
) -> Credential:
    """The credential this configuration authenticates with, refusing one that cannot work.

    Called by the plugin factory **before the connector is constructed**, on the same principle
    as :func:`~manicule.connectors.config.resolve_credentials`: a credential problem found at
    the first page of the first sync is a run that reports progress and indexes nothing. For a
    browser session that principle bites harder, because the thing that has to happen next is a
    person opening a browser.

    Args:
        config: The connector's settings, already through
            :func:`~manicule.connectors.config.resolve_credentials`.
        store: Where captured sessions live. ``None`` uses this process's own vault, which for
            a served manicule is the one ``connector login`` hands sessions to.
        now: The clock the age check reads.

    Returns:
        For a token, the token — a credential that is the same string every time. For a browser
        session, a :class:`HeldSessionCredential`, which is deliberately **not** the session
        found here: a connector is cached for the life of the process and the session is not,
        so what it is given is the *place* to look rather than what was there at the time.

    Raises:
        SessionMissingError: No session has been captured for this instance in this process's
            lifetime. The message names the two commands that fix it, because after a restart
            this is the expected state rather than a fault — a session lives in the server's
            memory, so stopping the server is what ends it. It is a
            :class:`~manicule.core.errors.ConfigError`, so a caller catching that still catches
            this; what its own name buys is the scheduler being able to tell "nobody has signed
            in" from "the instance was unreachable".
        SessionExpiredError: A session was captured and is older than this connector will use.
    """
    if config.auth_method is not AuthMethod.BROWSER_SESSION:
        return token_credential(config)

    credential = HeldSessionCredential(config=config, store=store, now=now)
    # Consulted once here so that a session that is absent, or already too old to use, is a
    # startup refusal rather than something the first request discovers — the reason this
    # function is called by the plugin factory rather than by the client. It is a reading and
    # not a capture: authorize() runs again for every request, over whatever the process is
    # holding then, because this one can only speak for the moment it ran.
    credential.authorize()
    return credential


GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"  # noqa: S105 - an endpoint, not a secret
"""Where a refresh token is exchanged. A constant and not a setting, on purpose.

The exchange sends the client secret and the refresh token to this address, so making it
configurable would make one line of configuration an exfiltration channel. Tests reach a fake
through the ``transport`` they inject, which replaces the network rather than the address.
"""

GMAIL_PROFILE_URL = "https://gmail.googleapis.com/gmail/v1/users/me/profile"
"""``users.getProfile``: the one read the mailbox pin makes."""

REFRESH_MARGIN_SECONDS = 60.0
"""How long before ``expires_in`` an access token stops being offered.

A token offered in its last second is rejected by the time the request arrives, and that costs a
401 and a forced refresh where this costs nothing.
"""

FALLBACK_LIFETIME_SECONDS = 300.0
"""The lifetime assumed when Google's answer carries no usable ``expires_in``.

Google always sends one (about an hour). An answer without it is not trusted for long: guessing
short costs a refresh, and guessing long costs a run of 401s.
"""

_ERROR_CODE = re.compile(r"[a-z_]{1,40}")

_QUOTA_REASONS = frozenset(
    {"ratelimitexceeded", "userratelimitexceeded", "dailylimitexceeded", "quotaexceeded"}
)
"""The reasons Google gives a 403 that is throttling and not a refusal. Compared case-folded."""


def _require_httpx() -> None:
    """Raise a ``ConfigError`` saying which extra supplies the HTTP client, if it is missing.

    The client is imported where it is used rather than at module level, so plugin discovery
    loads no HTTP stack. That also means a machine installed without the ``connectors`` extra
    first meets the gap here, and is told what to install instead of getting a
    ``ModuleNotFoundError`` out of a diagnostic.
    """
    if importlib.util.find_spec("httpx") is None:
        msg = (
            "the Gmail connector needs the HTTP client, which this installation lacks. Install "
            "it with `pip install 'manicule[connectors]'`."
        )
        raise ConfigError(msg)


def _forbidden_is_quota(response: httpx.Response) -> bool:
    """Whether a 403 body names a throttling reason rather than a refusal."""
    body = _json_object(response)
    error = body.get("error")
    if not isinstance(error, dict):
        return False
    details = cast("dict[str, object]", error).get("errors")
    if not isinstance(details, list):
        return False
    return any(
        isinstance(item, dict)
        and str(cast("dict[str, object]", item).get("reason", "")).casefold() in _QUOTA_REASONS
        for item in cast("list[object]", details)
    )


class _HasStatus(Protocol):
    @property
    def status_code(self) -> int: ...


def _bearer(token: SecretStr) -> Authorization:
    return Authorization({"Authorization": f"Bearer {token.get_secret_value()}"})


@dataclass(eq=False, repr=False)
class RefreshTokenCredential:
    """A refresh token turned into short-lived access tokens, for one Gmail instance.

    The Gmail counterpart of :class:`TokenCredential`, and not a :class:`Credential`: minting an
    access token is a network call, so its :meth:`authorize` is a coroutine and the synchronous
    protocol cannot carry it. The REST client sends every request through :meth:`send`, which is
    the whole of the policy a 401 gets.

    **Cached until shortly before ``expires_in``**, one refresh at a time. Concurrent callers wait
    on a lock and share the token the first one minted, so a burst of requests after expiry is
    one exchange and not one each.

    **Refresh once on a 401, then fail typed.** The first 401 is an access token that expired
    early or was revoked; the second, on a token minted a moment ago, is something a retry cannot
    change. If another request has already replaced the rejected token by the time this one gets
    the lock, the replacement is used without a second exchange.

    **Nothing secret is reachable from here by accident.** The client secret, the refresh token
    and the access token are all :class:`~pydantic.SecretStr`; the repr names only the instance;
    the exchange is a POST form body and never a query string; no error carries a response body,
    only a short error code Google defines; and ``httpx`` is imported when an exchange is made
    rather than when plugin discovery runs.

    ``invalid_grant`` is the one answer with its own type (:class:`GrantRevokedError`), because
    it is the one a person fixes: the grant was revoked, a password policy revoked it, or it
    aged out, and ``manicule connector login <name>`` mints a replacement.
    """

    connector: str
    """The instance's name: what the renewal text tells a person to log in to."""

    client_id: str
    client_secret: SecretStr
    refresh_token: SecretStr

    client_secret_env: str = ""
    """The variables the secrets may have come from, named when one is rejected."""
    refresh_token_env: str = ""

    transport: httpx.AsyncBaseTransport | None = None
    """Replaces the network. ``None`` uses the real one; a test supplies a ``MockTransport``."""

    clock: Callable[[], float] = time.monotonic
    timeout_seconds: float = 30.0

    _access: SecretStr | None = field(default=None, init=False)
    _expires_at: float = field(default=0.0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @override
    def __repr__(self) -> str:
        return f"RefreshTokenCredential(connector={self.connector!r})"

    def renewal(self) -> str:
        return (
            f"Run `manicule connector login {self.connector}` to mint a new refresh token, and "
            f"put it where the configuration reads it."
        )

    def rejected(self) -> TokenRejectedError:
        """What to raise when a request is rejected again with a freshly minted access token.

        For a client that owns its own retry and calls :meth:`access_token`; :meth:`send` raises
        it itself.
        """
        return TokenRejectedError(
            "Gmail rejected the request with a freshly minted access token. The grant may have "
            "lost the gmail.readonly scope or been revoked. " + self.renewal()
        )

    async def authorize(self) -> Authorization:
        """The bearer header for the next request, minting an access token if none is fresh.

        Raises:
            GrantRevokedError: Google no longer honors the refresh token.
            CredentialError: Google rejected the OAuth client, or answered nonsense.
            RemoteError: Google's token endpoint failed; a later attempt may succeed.
        """
        return _bearer(await self._token())

    async def _token(self) -> SecretStr:
        cached = self._fresh()
        if cached is not None:
            return cached
        async with self._lock:
            # Re-checked under the lock: a caller ahead of this one has usually just minted it.
            return self._fresh() or await self._refresh()

    async def access_token(self, *, refresh: bool = False) -> str:
        """The bare access token, for a client that builds its own header and retry.

        ``refresh=False`` is the cached token, minted if none is fresh. ``refresh=True`` is for a
        caller that has just been answered 401: it mints a new token whatever the cache holds.
        Prefer :meth:`send`, which also coalesces callers that were rejected on the same token
        into one exchange; this exists for a client that already owns its retry, and the token
        it returns is a string only for as long as that client takes to put it in a header.

        Raises:
            GrantRevokedError: Google no longer honors the refresh token.
            CredentialError: Google rejected the OAuth client, or answered nonsense.
            RemoteError: Google's token endpoint failed; a later attempt may succeed.
        """
        if refresh:
            async with self._lock:
                self._access = None
                return (await self._refresh()).get_secret_value()
        return (await self._token()).get_secret_value()

    async def send[R: _HasStatus](self, attempt: Callable[[Authorization], Awaitable[R]]) -> R:
        """Make one request as this credential, and the single retry a 401 earns.

        Args:
            attempt: Sends the request with the headers it is given and returns the response. It
                is called once, or twice when the first answer is 401.

        Raises:
            TokenRejectedError: The request was rejected again with a token minted for the retry.
        """
        authorization = await self.authorize()
        response = await attempt(authorization)
        if response.status_code != HTTPStatus.UNAUTHORIZED:
            return response
        authorization = await self._reauthorize(authorization)
        response = await attempt(authorization)
        if response.status_code == HTTPStatus.UNAUTHORIZED:
            raise self.rejected()
        return response

    async def confirm_mailbox(self, account: str) -> None:
        """Check that this credential reads the mailbox the instance is configured for.

        One refresh if none is cached and one ``users.getProfile``, compared with ``account``.
        Called by ``doctor``, by the connector's setup — which the container awaits before a
        connector is handed to anyone — and at the start of every run, because the token is
        replaced routinely and the mailbox must not change with it (``gmail.md`` §1.4).

        Args:
            account: The configured address, already lowercase (see :class:`GmailConfig`).

        Raises:
            MailboxMismatchError: The token belongs to another mailbox, or to an alias of this
                one. Neither address is in the message.
            CredentialError: The credential cannot be used at all.
            RemoteError: Google failed to answer.
        """
        response = await self.send(self._get_profile)
        if response.status_code == HTTPStatus.FORBIDDEN and _forbidden_is_quota(response):
            msg = "Gmail is throttling users.getProfile"
            raise RateLimitedError(msg)
        if response.status_code == HTTPStatus.FORBIDDEN:
            # Not Google failing to answer: the API is off for the OAuth client's project, or the
            # grant lacks the scope. Typed as a credential problem so ``doctor`` fails it with
            # the right remedy instead of blaming the network.
            msg = (
                "Gmail refused users.getProfile (HTTP 403). Check that the Gmail API is enabled "
                "for the OAuth client's project and that the grant includes gmail.readonly. "
                + self.renewal()
            )
            raise CredentialError(msg)
        if response.status_code != HTTPStatus.OK:
            msg = f"Gmail answered users.getProfile with HTTP {response.status_code}"
            raise RemoteError(msg, response.status_code)
        try:
            reported = response.json().get("emailAddress")
        except (ValueError, AttributeError):
            reported = None
        if not isinstance(reported, str) or not reported.strip():
            msg = (
                "Gmail answered users.getProfile without an address; the mailbox cannot be checked"
            )
            raise CredentialError(msg)
        if reported.strip().casefold() != account.strip().casefold():
            msg = (
                "the credential belongs to a different mailbox than this instance's configured "
                "account (an alias of the account counts as different: configure the primary "
                "address). Indexing another mailbox is a different instance under a new name, "
                "so the run did not begin. " + self.renewal()
            )
            raise MailboxMismatchError(msg)

    # -- the exchange ----------------------------------------------------------------

    def _fresh(self) -> SecretStr | None:
        if self._access is not None and self.clock() < self._expires_at:
            return self._access
        return None

    async def _reauthorize(self, rejected: Authorization) -> Authorization:
        """A new token, unless another request has already replaced the one that was rejected."""
        async with self._lock:
            current = self._fresh()
            if current is not None and _bearer(current).headers != rejected.headers:
                return _bearer(current)
            self._access = None
            return _bearer(await self._refresh())

    async def _refresh(self) -> SecretStr:
        """Exchange the refresh token for an access token. Called with the lock held."""
        _require_httpx()
        import httpx  # noqa: PLC0415 - no HTTP stack at plugin discovery

        form = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token.get_secret_value(),
            "client_id": self.client_id,
            "client_secret": self.client_secret.get_secret_value(),
        }
        try:
            async with httpx.AsyncClient(
                transport=self.transport, timeout=self.timeout_seconds
            ) as http:
                response = await http.post(GOOGLE_TOKEN_URL, data=form)
        except httpx.TimeoutException as exc:
            msg = "Google's token endpoint did not answer in time"
            raise RequestTimeoutError(msg) from exc
        except httpx.HTTPError as exc:
            msg = "Google's token endpoint could not be reached"
            raise ConnectorError(msg) from exc

        body = _json_object(response)
        if response.status_code == HTTPStatus.OK:
            token = body.get("access_token")
            if not isinstance(token, str) or not token:
                msg = "Google's token endpoint answered without an access token"
                raise CredentialError(msg)
            self._access = SecretStr(token)
            self._expires_at = self.clock() + _usable_lifetime(body.get("expires_in"))
            _log.debug("refreshed the access token for Gmail instance %s", self.connector)
            return self._access
        if response.status_code in (HTTPStatus.BAD_REQUEST, HTTPStatus.UNAUTHORIZED):
            raise self._refused(body.get("error"))
        if response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
            msg = "Google's token endpoint is throttling this client"
            raise RateLimitedError(msg)
        msg = f"Google's token endpoint answered HTTP {response.status_code}"
        raise RemoteError(msg, response.status_code)

    def _refused(self, error: object) -> CredentialError:
        """The typed error for a refused exchange, from Google's error code and nothing else.

        ``error_description`` is not repeated: it is free text Google chooses, and the code says
        which of the few things a person can do applies.
        """
        code = error if isinstance(error, str) and _ERROR_CODE.fullmatch(error) else "unknown"
        if code == "invalid_grant":
            msg = (
                "Google no longer honors this instance's refresh token (invalid_grant): the "
                "grant was revoked, the account's password changed under a policy that revokes "
                "tokens, or the token aged out. " + self.renewal()
            )
            return GrantRevokedError(msg)
        if code in {"invalid_client", "unauthorized_client"}:
            where = f" (${self.client_secret_env})" if self.client_secret_env else ""
            msg = (
                f"Google rejected the OAuth client ({code}). Check client_id and the client "
                f"secret{where} belong to the same Desktop-app client."
            )
            return CredentialError(msg)
        return CredentialError(f"Google refused the token exchange ({code}). {self.renewal()}")

    async def _get_profile(self, authorization: Authorization) -> httpx.Response:
        _require_httpx()
        import httpx  # noqa: PLC0415 - no HTTP stack at plugin discovery

        try:
            async with httpx.AsyncClient(
                transport=self.transport, timeout=self.timeout_seconds
            ) as http:
                return await http.get(GMAIL_PROFILE_URL, headers=dict(authorization.headers))
        except httpx.TimeoutException as exc:
            msg = "Gmail did not answer users.getProfile in time"
            raise RequestTimeoutError(msg) from exc
        except httpx.HTTPError as exc:
            msg = "Gmail could not be reached for users.getProfile"
            raise ConnectorError(msg) from exc


def _json_object(response: httpx.Response) -> dict[str, object]:
    try:
        parsed = response.json()
    except ValueError:
        return {}
    return cast("dict[str, object]", parsed) if isinstance(parsed, dict) else {}


def _usable_lifetime(expires_in: object) -> float:
    """Seconds an access token is offered for: its lifetime less the margin."""
    if (
        isinstance(expires_in, bool)
        or not isinstance(expires_in, int | float)
        or not math.isfinite(expires_in)
        or expires_in <= 0
    ):
        return FALLBACK_LIFETIME_SECONDS
    if expires_in > 2 * REFRESH_MARGIN_SECONDS:
        return float(expires_in) - REFRESH_MARGIN_SECONDS
    return float(expires_in) / 2


def gmail_credential(
    config: GmailConfig,
    *,
    instance: str,
    transport: httpx.AsyncBaseTransport | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> RefreshTokenCredential:
    """The credential a Gmail instance mints access tokens with.

    Pure construction, no network: the factory is synchronous and the container calls it from
    inside the event loop, so the refresh and the mailbox check are
    :meth:`RefreshTokenCredential.confirm_mailbox`'s, awaited before the connector is handed
    out.

    Raises:
        ConfigError: The client secret or refresh token is unset, which
            :func:`~manicule.connectors.config.resolve_gmail_credentials` fills in and refuses
            on; a caller building this some other way has to call it first.
    """
    if config.client_secret is None or config.refresh_token is None:
        msg = (
            "the Gmail client secret or refresh token is not set. resolve_gmail_credentials() "
            "fills them from the environment and refuses when they are absent; a caller "
            "building this some other way has to call it too."
        )
        raise ConfigError(msg)
    return RefreshTokenCredential(
        connector=instance,
        client_id=config.client_id,
        client_secret=config.client_secret,
        refresh_token=config.refresh_token,
        client_secret_env=config.client_secret_env,
        refresh_token_env=config.refresh_token_env,
        transport=transport,
        clock=clock,
        timeout_seconds=config.request_timeout_seconds,
    )
