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

import base64
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol

from pydantic import SecretStr

from manicule.connectors.config import AuthMethod, ConfluenceConfig
from manicule.connectors.errors import SessionExpiredError
from manicule.core.errors import ConfigError

if TYPE_CHECKING:  # pragma: no cover - the store imports this module back at run time
    from manicule.connectors.sessions import SessionStore

__all__ = [
    "Authorization",
    "BrowserSession",
    "BrowserSessionCredential",
    "Credential",
    "HeldSessionCredential",
    "TokenCredential",
    "credential_for",
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
