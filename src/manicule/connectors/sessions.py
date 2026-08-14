"""Where a Confluence browser session comes from, and where it lives.

Self-hosted Confluence behind an identity provider commonly has personal access tokens disabled
by policy, so the credential its users can actually obtain is the session they already hold in
their browser. Three decisions shape this module, and each is a refusal of an easier option.

**manicule never asks for the password and has nowhere to put one.** No password, no one-time
code and no device approval passes through this process on any path, and that much *is* a fact
about the code rather than a promise: there is no parameter that could carry one and no branch
that would accept one.

**A browser is now driven, and this paragraph used to say the opposite.** It said that Playwright
was the ergonomic answer, that the license (Apache-2.0) was not the objection, and that the
objection was this: a driven browser is a browser manicule controls the DOM of, and the person is
asked to type a corporate password into it, so "manicule never sees the password" would become a
promise about restraint instead of a fact about capability.

That argument was right and it has been overruled, for a reason the argument did not weigh. An
instance behind an identity provider commonly has personal access tokens disabled by policy. For
those installations the manual paste was not the safer of two options — it was the *only* option,
and it asks somebody to open developer tools, find a live session cookie and paste several
kilobytes of it into a terminal. Refusing to drive a browser did not remove the risk; it moved it
onto the person, by hand, every time their session expired.

So the property has genuinely weakened, and it is worth naming precisely rather than glossing:

*Before:* manicule **cannot** see the password, because there is no browser.
*Now, on ``--browser`` only:* manicule **does not** see the password, because
    :mod:`manicule.connectors.browser` reads no page content — a claim a test enforces over that
    module's source, and a reviewer enforces over its diff.

*On this module's own path, unchanged:* manicule cannot see it, because there is still no
    browser. **The paste is not deprecated and is not a fallback.** It is the option that keeps
    the stronger guarantee, and somebody who wants that guarantee should use it.

The two practical warnings from the original argument stand and are documented rather than
solved: a driven Chromium is a new device to a conditional-access policy and may be refused
outright, and the browser is a heavy dependency — which is why it is an extra
(``manicule[browser-auth]``) that nothing else needs.

**The session lives in the running server's memory, and nowhere else.** Not the macOS
Keychain, not a file, not an environment variable. There were three of those and now there is
one, and each of the two that went had a defect that could not be fixed where it was:

*The Keychain prompted, repeatedly.* Every write recreated its item and therefore discarded the
authorization the operator had granted, so a person running syncs was asked for their login
password again and again by a program that is not allowed to know it. Chunking a secret across
numbered items to work around a 128-byte stdin buffer was the machinery that made that worse,
and all of it is gone.

*And there was no store at all on Linux or in a container*, so ``--browser`` there captured a
session it had nowhere to put. The environment variable that stood in for one is a credential
written into a shell's history and visible in every process listing that inherits it.

A session held in a long-lived process's memory has none of those problems: it is never written
anywhere, it prompts for nothing, it works identically on every platform, and it is gone when
the process stops. **That last one is the point rather than the cost.**
``manicule connector login --browser`` has made re-authenticating a few seconds of clicking, so
a credential whose lifetime is the server's is a credential that is re-taken deliberately
instead of persisted indefinitely.

**The consequence is stated rather than left to be discovered: no server, no sync.** A sync run
in a process that is not the server has no session and cannot get one, and the message when a
session is absent says exactly that and names the command that fixes it. ``ConfluenceConfig``
forbids unknown keys, so a ``session_cookie`` written into configuration is still a startup
error rather than a working setting.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol

from pydantic import SecretStr

from manicule.connectors.config import ConfluenceConfig
from manicule.connectors.credentials import BrowserSession
from manicule.core.errors import ConfigError

if TYPE_CHECKING:  # pragma: no cover - import-time only
    import httpx

__all__ = [
    "SESSIONS",
    "SessionStore",
    "SessionVault",
    "capture",
    "capture_cookies",
    "cookies_authenticate",
    "default_store",
    "instance_key",
    "load_session",
    "parse_cookies",
]

_PROBE_PATH = "/rest/api/user/current"

_TIMEOUT_SECONDS = 20.0


class SessionStore(Protocol):
    """Where captured sessions are kept for the life of this process.

    **``load`` is synchronous and the other two are not**, which is an asymmetry worth stating
    rather than tidying away. Loading is consulted by
    :func:`~manicule.connectors.credentials.credential_for` from inside the connector plugin
    factory — ordinary synchronous code that runs before a connector is built — and in every
    implementation it is a dictionary lookup in this process's memory.

    Saving and forgetting are the two that may have to *reach* the process holding the
    sessions. On the command line they cross the control socket to the server, because the
    browser opens where the person is and the syncs run somewhere else. A synchronous ``save``
    would have to do that from inside a running event loop, which is the shape that ends in a
    thread nobody meant to start.
    """

    def load(self, base_url: str) -> BrowserSession | None: ...

    async def save(self, session: BrowserSession) -> None: ...

    async def forget(self, base_url: str) -> bool:
        """Remove the session for ``base_url``. ``True`` if there was one."""
        ...

    def describe(self) -> str:
        """Where this keeps things, for a message that tells somebody what just happened."""
        ...

    def holding(self) -> dict[str, str]:
        """Which instances this holds a session for, keyed by :func:`instance_key`.

        On the protocol rather than on :class:`SessionVault` alone, so that ``doctor``'s session
        check asks *a store* rather than asking whichever object it can prove is the real one.
        The command-line store answers ``{}`` and means it — a short-lived process holds no
        session, which is the property ``tests/app/test_session_handover.py`` already asserts
        about its ``load``.

        Values are accounts and never credentials; see :meth:`SessionVault.holding`.
        """
        ...


class SessionVault:
    """Captured sessions, held in this process's memory and written nowhere.

    The whole credential store. There is no second implementation and no fallback, which is the
    point: three mechanisms meant three ways for a session to be somewhere nobody expected, and
    two of them wrote a live corporate credential to a place that outlived the process.

    **Nothing here has a persistence path to fail to take.** There is no file to chmod, no
    keychain to prompt, no serialization, and no expiry other than
    ``session_max_age_hours`` and the process ending. A reader looking for the write that
    escapes will not find one, because the class is a dictionary.

    Keyed by the instance's base URL, normalized, because a session is not portable between
    instances and filing it under the site it came from is what stops one being offered to
    another.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, BrowserSession] = {}

    def describe(self) -> str:
        return "the running server's memory, which does not survive it"

    def load(self, base_url: str) -> BrowserSession | None:
        return self._sessions.get(instance_key(base_url))

    async def save(self, session: BrowserSession) -> None:
        self._sessions[instance_key(session.base_url)] = session

    async def forget(self, base_url: str) -> bool:
        return self._sessions.pop(instance_key(base_url), None) is not None

    def __len__(self) -> int:
        """How many instances this holds a session for. For a diagnostic that must not read one.

        A count rather than a listing, and deliberately not a way to get at the sessions
        themselves: "is there a session" is a question ``doctor`` and ``connector list`` have a
        use for, and "what is it" is a question nothing outside this module has.
        """
        return len(self._sessions)

    def holding(self) -> dict[str, str]:
        """Which instances this holds a session for, and whose account each one is.

        The listing :meth:`__len__` deliberately is not, added when ``doctor`` grew a check that
        has to name the source an operator must sign in to again — a count answers "is anything
        held" and not "is *handbook* held", and the second is the question somebody has at three
        in the morning.

        **Two fields, and neither is the credential.** The instance is already in configuration
        and the account is already echoed by a hand-off's acknowledgment, so this discloses
        nothing that was not disclosed; the cookies are not reachable from what it returns, which
        is what keeps :meth:`__len__`'s reason intact rather than merely narrowed.

        Keyed by the normalized base URL, because that is what this is keyed by — a listing that
        reported the spelling somebody typed would not match what :meth:`load` looks up.
        """
        return {base_url: session.account for base_url, session in self._sessions.items()}


SESSIONS = SessionVault()
"""The sessions this process is holding.

**Process-level rather than threaded through the container, and the reason is that it is a
fact about the process.** One manicule owns a data directory at a time — that is what
:class:`~manicule.ingest.recovery.InstanceLock` enforces — so "the sessions this process holds"
is unambiguous in a way that a per-runtime vault would only look more careful about. Passing one
through :class:`~manicule.plugins.registry.BuildContext` would also put a credential store into
the plugin API, where every third-party connector would see it.

In a command-line process this is empty and stays empty: capture hands the session to the
server over the control socket rather than keeping a copy. In a served process it is where
``connector login`` puts one and where every sync reads it.
"""


def default_store() -> SessionStore:
    """The store for this process: :data:`SESSIONS`, and there is no other.

    Kept as a function rather than inlining :data:`SESSIONS` at its call sites so that the seam
    tests substitute a vault at is one named thing rather than a module attribute they patch.
    """
    return SESSIONS


def load_session(
    config: ConfluenceConfig,
    *,
    store: SessionStore | None = None,
) -> BrowserSession | None:
    """The session this process is holding for that instance, or ``None``.

    One place is consulted, because there is one place. The environment variable that used to
    stand in for a keychain on Linux is gone with the keychain: it was a live corporate
    credential written into a shell's history and inherited by every child process, offered as
    the answer for the platforms that had no store — and "the platform without a store" is now
    every platform, answered by the server instead.

    ``None`` means nobody has signed in to that instance in this process's lifetime. For a
    server that means since it started, which is the expected path after a restart rather than
    a failure; :func:`~manicule.connectors.credentials.credential_for` is where that is turned
    into a message naming the command that fixes it.
    """
    keeper = store if store is not None else default_store()
    return keeper.load(config.base_url)


def parse_cookies(text: str) -> dict[str, SecretStr]:
    """The cookies in a pasted ``Cookie`` header, in the order they were given.

    Accepts what a browser's developer tools actually hand over: a whole ``Cookie:`` request
    header, the header's value on its own, or several ``name=value`` pairs on separate lines.
    All three are what somebody will paste, and rejecting two of them would only teach people
    to reformat a secret in a text editor.

    Raises:
        ConfigError: There is no ``name=value`` pair in it. Almost always this means a password
            was pasted, so the message says what to do rather than what was wrong with it.
    """
    body = text.strip()
    if body.lower().startswith("cookie:"):
        body = body.split(":", 1)[1]
    cookies: dict[str, SecretStr] = {}
    for line in body.replace("\n", ";").split(";"):
        pair = line.strip()
        if not pair or "=" not in pair:
            continue
        name, _, value = pair.partition("=")
        if name.strip() and value.strip():
            cookies[name.strip()] = SecretStr(value.strip())
    if not cookies:
        msg = (
            "that is not a session cookie: manicule expected something of the form "
            "'JSESSIONID=...; other=...', copied from a browser that is already signed in. "
            "manicule never asks for a password and cannot use one — sign in to Confluence in "
            "your browser first, then copy the Cookie header from its developer tools."
        )
        raise ConfigError(msg)
    return cookies


async def capture(
    config: ConfluenceConfig,
    cookie_text: str,
    *,
    store: SessionStore | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    now: datetime | None = None,
) -> BrowserSession:
    """Prove a pasted session works, then store it.

    The manual path, unchanged. Parsing is the only thing it does that
    :func:`capture_cookies` does not, and everything after the parse is that function — so a
    session captured from a paste, from a driven browser and from an imported state file are
    verified and stored by one piece of code rather than by three that could drift.

    Raises:
        ConfigError: The paste carried no cookies, or the instance does not answer the endpoint
            this asks.
        SessionExpiredError: The instance answered, and answered as somebody signed out.
    """
    return await capture_cookies(
        config,
        parse_cookies(cookie_text),
        store=store,
        transport=transport,
        now=now,
    )


async def capture_cookies(
    config: ConfluenceConfig,
    cookies: Mapping[str, SecretStr],
    *,
    store: SessionStore | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    now: datetime | None = None,
) -> BrowserSession:
    """Prove a set of cookies works, then store it. The only route to the credential store.

    The proof is the point. A cookie that was copied short, taken from the wrong tab, extracted
    from a state file belonging to another instance, or collected from a browser the person had
    not finished signing in to is indistinguishable from a working one until something uses it —
    and "something uses it" would otherwise be the first page of the next sync. So this makes one
    request as the session, reads back who the instance says that is, and stores nothing at all
    if the answer is anybody other than a signed-in user.

    **Verification happens before the store is touched, which is what makes replacement atomic.**
    A failed login leaves whatever was there before exactly as it was: there is no delete-then-
    write window, because the write is the last thing and it only happens on success. That
    matters most for the browser flow, where a person re-authenticating a session that had merely
    aged would otherwise be able to lose a working credential by closing the window.

    Raises:
        ConfigError: No cookies were given, or the instance does not answer the endpoint this
            asks, or it answers without naming a user.
        SessionExpiredError: The instance answered, and answered as somebody signed out — which
            includes the sign-in page served with status 200 that
            :mod:`~manicule.connectors.intercept` exists for.
    """
    from manicule.connectors.client import ConfluenceClient  # noqa: PLC0415 - no HTTP at import
    from manicule.connectors.credentials import BrowserSessionCredential  # noqa: PLC0415
    from manicule.connectors.errors import NotFoundError  # noqa: PLC0415

    if not cookies:
        msg = (
            f"no cookies for {config.base_url} were found, so there is nothing to verify and "
            f"nothing to store. A session with no cookies authenticates as nobody."
        )
        raise ConfigError(msg)

    moment = now if now is not None else datetime.now(tz=UTC)
    candidate = BrowserSession(
        base_url=config.base_url,
        account="",
        captured_at=moment,
        cookies=dict(cookies),
    )
    credential = BrowserSessionCredential(
        session=candidate,
        max_age=timedelta(hours=config.session_max_age_hours),
        now=lambda: moment,
    )
    client = ConfluenceClient(config, credential=credential, transport=transport)
    await client.setup()
    try:
        payload = await client.get_json(client.url(_PROBE_PATH))
    except NotFoundError as exc:
        msg = (
            f"{config.base_url}{_PROBE_PATH} does not exist on this instance, so manicule "
            f"cannot confirm who the session belongs to and will not store it. Check "
            f"base_url names the site root including any context path."
        )
        raise ConfigError(msg) from exc
    finally:
        await client.teardown()

    account = _named(payload)
    if not account:
        msg = (
            f"{config.base_url}{_PROBE_PATH} answered without naming a user, so manicule "
            f"cannot confirm the session is signed in and will not store it."
        )
        raise ConfigError(msg)

    session = BrowserSession(
        base_url=config.base_url,
        account=account,
        captured_at=moment,
        cookies=candidate.cookies,
    )
    keeper = store if store is not None else default_store()
    await keeper.save(session)
    return session


async def cookies_authenticate(
    config: ConfluenceConfig,
    cookies: Mapping[str, SecretStr],
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    now: datetime | None = None,
) -> bool:
    """Whether ``cookies`` are signed in yet — the question a wait loop asks, without raising.

    :func:`capture_cookies` answers the same question by raising, which is right for a command
    that has been asked to store something and cannot. It is wrong for the browser flow's poll:
    a person who has not finished signing in yet is the *expected* state there, several times a
    second, and an exception per poll would turn the ordinary case into a stack of handled
    errors.

    So this is the same probe with the verdict as a boolean. It shares the endpoint and the
    client with :func:`capture_cookies` rather than reimplementing the check, because a poll loop
    that believed something the storing path would then refuse would hang until the timeout while
    the browser sat signed in.

    **A ``True`` here is not a decision to store.** The caller hands the cookies to
    :func:`capture_cookies`, which asks again through the same client and is the only thing that
    writes. The double check costs one request and closes the window between "signed in" and
    "stored" — a session that died in between is refused rather than persisted dead.
    """
    if not cookies:
        return False
    from manicule.connectors.errors import ConnectorError  # noqa: PLC0415

    try:
        await capture_cookies(
            config,
            cookies,
            store=_Discarding(),
            transport=transport,
            now=now,
        )
    except (ConfigError, ConnectorError):
        # Every refusal means "not signed in yet", which is what a poll wants to hear. The
        # distinction between a dead session, a sign-in page and a half-finished login matters
        # to the *final* attempt, and that one goes through `capture_cookies` and keeps its
        # message.
        return False
    return True


class _Discarding:
    """A store that keeps nothing, so the poll can reuse the verifying path without writing.

    The alternative was a ``store=None`` sentinel meaning "do not store", which is a second
    meaning for a parameter that already means "use the default" — and the failure mode of
    getting that wrong is a credential written to the keychain by a loop that was only asking.
    """

    def describe(self) -> str:  # pragma: no cover - never shown to anybody
        return "nowhere; this probe stores nothing"

    def load(self, base_url: str) -> BrowserSession | None:
        del base_url
        return None

    async def save(self, session: BrowserSession) -> None:
        del session

    async def forget(self, base_url: str) -> bool:
        del base_url
        return False

    def holding(self) -> dict[str, str]:
        """Nothing, because :meth:`save` kept nothing. Never consulted; present to be a store."""
        return {}


def _named(payload: Mapping[str, object]) -> str:
    """Who a ``user/current`` response says it is, under whichever key this version uses."""
    for key in ("username", "accountId", "userKey", "email"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def instance_key(base_url: str) -> str:
    """The key a site's session is filed under.

    Normalized so that ``https://wiki.example.com`` and ``https://wiki.example.com/`` are one
    entry rather than two, one of which would be found and the other silently not.

    Public, and named for a URL rather than for an account, because :meth:`SessionVault.holding`
    reports these keys and a caller comparing a configured ``base_url`` against them has to
    normalize it the same way. A second normalizer written at that call site would agree until
    somebody configured a trailing slash — which is precisely the case this exists for.
    """
    return base_url.strip().rstrip("/")
