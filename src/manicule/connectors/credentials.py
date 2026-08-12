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

**Nothing here imports an HTTP client**, for the same reason :mod:`manicule.connectors.config`
does not: this module is reachable from the plugin factory, which runs before configuration is
read.

**No credential is stored as a plain string.** The secret is held in a
:class:`~pydantic.SecretStr` and the header is assembled at the moment of use, so a repr, a
traceback frame or a logged configuration object carries the wrapper rather than the value.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol, cast

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
    """

    headers: Mapping[str, str]


class Credential(Protocol):
    """Something a request can be made as. Consulted per request; may refuse."""

    def authorize(self) -> Authorization:
        """Headers for the request about to be made.

        Raises:
            SessionExpiredError: This credential has a lifetime and has outlived it. Raised
                before the request rather than after a response, because a request made with a
                dead session is answered with a sign-in page rather than an error.
        """
        ...

    def describe(self) -> str:
        """What this credential is, for a message about it being rejected."""
        ...

    def renewal(self) -> str:
        """What a person does to replace it. One sentence, imperative."""
        ...

    def account(self) -> str:
        """Who this authenticates as, if that is known, else ``""``.

        Confluence Server and Data Center name the authenticated user on every REST response
        (``X-AUSERNAME``). Knowing who manicule *expects* to be turns that header into a check:
        a response that says ``anonymous``, or says somebody else, is a session that has stopped
        being this account's whether or not the status code admits it.
        """
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

    def account(self) -> str:
        return self.subject


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
    """

    base_url: str
    """The instance this belongs to. A session is not portable between instances, and storing
    it under the site it came from is what stops one being offered to another."""

    account: str
    """Who the instance said this was when it was captured, checked against every response."""

    captured_at: datetime
    cookies: Mapping[str, SecretStr] = field(default_factory=dict[str, SecretStr])

    def to_json(self) -> str:
        """The record as the one string a keychain item holds.

        Serialised by hand rather than through pydantic's JSON mode, which would render every
        secret as asterisks — correct for a log and useless for storage. Writing it out here is
        the one place the values are deliberately unwrapped, which is where a reader looks for
        that.
        """
        return json.dumps(
            {
                "base_url": self.base_url,
                "account": self.account,
                "captured_at": self.captured_at.isoformat(),
                "cookies": {
                    name: secret.get_secret_value() for name, secret in self.cookies.items()
                },
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, raw: str) -> BrowserSession:
        """Rebuild a record written by :meth:`to_json`.

        Raises:
            ValueError: The stored item is not a session record. A keychain outlives the
                versions of the software that wrote to it, so this reads what it is given
                rather than assuming its own format.
        """
        record = _object(json.loads(raw))
        cookies = _object(record.get("cookies"))
        if not cookies:
            msg = "the stored session carries no cookies"
            raise ValueError(msg)
        captured = _text(record.get("captured_at"))
        if not captured:
            msg = "the stored session has no capture time"
            raise ValueError(msg)
        when = datetime.fromisoformat(captured)
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        return cls(
            base_url=_text(record.get("base_url")),
            account=_text(record.get("account")),
            captured_at=when,
            cookies={name: SecretStr(_text(value)) for name, value in cookies.items()},
        )


def _object(value: object) -> Mapping[str, object]:
    """Narrow a decoded JSON value to an object. JSON keys are strings by construction."""
    return cast("Mapping[str, object]", value) if isinstance(value, dict) else {}


def _text(value: object) -> str:
    """A decoded JSON value as a string, treating anything else as absent."""
    return value if isinstance(value, str) else ""


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


@dataclass(frozen=True, slots=True)
class BrowserSessionCredential:
    """A :class:`BrowserSession` used as a credential, with an age this connector will trust.

    The age check is in :meth:`authorize`, which is to say it runs **before every request**
    rather than once at startup. A first sync of a large corpus can outlast the session it
    began under, and the alternative to checking each time is a run that keeps making requests
    with a credential manicule has already decided is too old — and gets sign-in pages back.
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
        return Authorization({"Cookie": jar})

    def describe(self) -> str:
        signed_in = f" signed in as {self.session.account}" if self.session.account else ""
        return f"a browser session{signed_in}"

    def renewal(self) -> str:
        return "Sign in to Confluence in your browser and re-run `manicule connector login`."

    def account(self) -> str:
        return self.session.account


def token_credential(config: ConfluenceConfig) -> Credential:
    """The credential for a configuration that authenticates with a token.

    Split from :func:`credential_for` so that the client can build its own default without
    reaching a keychain: a token is entirely determined by the configuration in hand, and a
    browser session is not.

    Raises:
        ConfigError: This configuration authenticates with a browser session, which cannot be
            derived from configuration, or the token :func:`resolve_credentials` should have
            filled in is missing.
    """
    method = config.auth_method
    if method is AuthMethod.BROWSER_SESSION:
        msg = (
            "this Confluence connector authenticates with a browser session, which is held in "
            "the keychain rather than in configuration. Build it with "
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
    environ: Mapping[str, str] | None = None,
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
        environ: Consulted for ``session_env``. ``None`` reads the process environment.
        store: Where captured sessions live. ``None`` uses the platform's keychain.
        now: The clock the age check reads.

    Raises:
        ConfigError: No session has been captured for this instance.
        SessionExpiredError: A session was captured and is older than this connector will use.
    """
    if config.auth_method is not AuthMethod.BROWSER_SESSION:
        return token_credential(config)

    from manicule.connectors.sessions import load_session  # noqa: PLC0415 - see module docstring

    session = load_session(config, environ=environ, store=store, now=now())
    if session is None:
        msg = (
            f"no Confluence browser session is stored for {config.base_url}. Sign in to it in "
            f"your browser, then run `manicule connector login <name>` and paste the session "
            f"cookies when it asks; manicule never asks for the password and never sees it. On "
            f"a machine with no macOS Keychain, put the same cookies in ${config.session_env}."
        )
        raise ConfigError(msg)
    credential = BrowserSessionCredential(
        session=session,
        max_age=timedelta(hours=config.session_max_age_hours),
        now=now,
    )
    # Consulted once here so that a session already too old to use is a startup refusal rather
    # than something the first request discovers. authorize() is the same check, and it runs
    # again for every request, because this one can only speak for the moment it ran.
    credential.authorize()
    return credential
