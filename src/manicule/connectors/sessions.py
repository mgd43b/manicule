"""Where a Confluence browser session comes from, and where it lives.

Self-hosted Confluence behind an identity provider commonly has personal access tokens disabled
by policy, so the credential its users can actually obtain is the session they already hold in
their browser. Three decisions shape this module, and each is a refusal of an easier option.

**manicule never sees the password.** The person signs in to their own browser, against their
own identity provider, with whatever second factor that provider demands, and copies the
resulting session cookies into ``manicule connector login``. No password, no one-time code and
no device approval passes through this process, and that is a property of the design rather
than a promise about the code: there is no code path that could accept one.

**No browser is driven.** Playwright would be the ergonomic answer and its licence (Apache-2.0)
is compatible with GPL-3.0-or-later, so the objection is not licensing. It is that a driven
browser is a browser manicule controls the DOM of, and the person is asked to type a corporate
password into it. "manicule never sees the password" would become a promise about restraint
instead of a fact about capability, and it is the single hardest constraint on this feature.
The practical arguments run the same way: a driven Chromium is a new device to a conditional
access policy and is often refused outright, and a browser download is a heavy dependency for
a paste.

**The session lives in the macOS Keychain, and nowhere else.** Not ``config.toml``, even at
``0600``: a session cookie is the sync account's whole identity at that company rather than a
scoped grant, and a configuration file ends up in version control eventually. Not under
``<data_dir>`` either — which is why none of ``docs/storage.md`` §7.1's permission rules apply
to it, because nothing lands there. ``ConfluenceConfig`` forbids unknown keys, so a
``session_cookie`` written into configuration is a startup error rather than a working setting.

On a machine with no Keychain — Linux, a container — the fallback is an environment variable
(``session_env``), which is a per-run credential and never written down by manicule at all.
"""

from __future__ import annotations

import base64
import shutil
import subprocess
import sys
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
    "KEYCHAIN_SERVICE",
    "KeychainStore",
    "MemoryStore",
    "SessionStore",
    "capture",
    "default_store",
    "load_session",
    "parse_cookies",
]

KEYCHAIN_SERVICE = "manicule: confluence session"
"""The keychain service every stored session shares. The account is the instance's base URL."""

SECURITY = "/usr/bin/security"
"""macOS's keychain command, by absolute path so that ``$PATH`` cannot choose a different one."""

_NOT_FOUND = 44
"""``security``'s exit status for an item that is not in the keychain."""

CHUNK_BYTES = 120
"""How much of the stored record goes into one keychain item.

``security`` reads a secret from stdin through a fixed 128-byte buffer and silently keeps the
first 128 bytes of anything longer — measured on macOS 15, and reported as success either way.
120 leaves room for a version whose buffer is a little smaller, and the read-back comparison in
:meth:`KeychainStore.save` catches one whose buffer is smaller still.
"""

MAX_CHUNKS = 256
"""How many pieces one session may occupy — about 30 KB of cookies, and a bound on the walk."""

_PROBE_PATH = "/rest/api/user/current"

_TIMEOUT_SECONDS = 20.0


class SessionStore(Protocol):
    """Where captured sessions are kept."""

    def load(self, base_url: str) -> BrowserSession | None: ...

    def save(self, session: BrowserSession) -> None: ...

    def forget(self, base_url: str) -> bool:
        """Remove the session for ``base_url``. ``True`` if there was one."""
        ...

    def describe(self) -> str:
        """Where this keeps things, for a message that tells somebody what just happened."""
        ...


class KeychainStore:
    """The macOS Keychain, reached through ``/usr/bin/security``.

    A subprocess rather than a library binding because the alternative binds the Security
    framework through ``ctypes``, and an item created that way trusts *the calling binary* —
    which for manicule is a virtual environment's Python. Recreate the environment or upgrade
    the interpreter and every sync starts raising a Keychain dialog. ``/usr/bin/security`` is a
    path that does not move.

    **The secret never appears in an argument vector.** ``security`` reads it from stdin when
    ``-w`` is given no value, so it is not in ``ps``, not in a process listing and not in
    anything that records command lines. Keeping that property is what forces the next two
    paragraphs, and the property is worth the trouble: a Confluence session is the sync
    account's whole identity at that company.

    **The stdin route truncates at 128 bytes, silently.** Measured, not assumed: 128 bytes are
    stored, 129 are stored as the first 128, and ``security`` reports success either way. A
    session record is several hundred bytes and an instance behind single sign-on often issues
    cookies of its own besides Confluence's, so this is not an edge case — the whole credential
    would be stored broken, and a broken session is one that authenticates as nobody and gets a
    sign-in page back. So the record is written in :data:`CHUNK_BYTES` pieces across numbered
    items, and read back by walking them until one is missing.

    **Every write is read back and compared.** Chunking works around today's limit; the read-back
    is what makes a *different* limit a loud failure rather than a quietly truncated credential.

    Items are created with ``-T /usr/bin/security``, the narrowest grant that still lets an
    unattended sync run: ``security`` may read them without a prompt and any other program
    raises the Keychain's own dialog. ``-A``, which would let anything read them silently, is
    not used.
    """

    def __init__(self, service: str = KEYCHAIN_SERVICE) -> None:
        self._service = service

    @staticmethod
    def available() -> bool:
        """Whether this machine has the Keychain command at all."""
        return sys.platform == "darwin" and shutil.which(SECURITY) is not None

    def describe(self) -> str:
        return f"the macOS Keychain, under the service {self._service!r}"

    def load(self, base_url: str) -> BrowserSession | None:
        payload = self._read(_account(base_url))
        if payload is None:
            return None
        try:
            return BrowserSession.from_json(base64.b64decode(payload).decode())
        except (ValueError, UnicodeDecodeError) as exc:
            msg = (
                f"the keychain item for {base_url} is not a session manicule wrote ({exc}). "
                f"Run `manicule connector login <name>` to replace it."
            )
            raise ConfigError(msg) from exc

    def save(self, session: BrowserSession) -> None:
        payload = base64.b64encode(session.to_json().encode()).decode()
        account = _account(session.base_url)
        # Everything already stored goes first, so a session that shrank cannot leave a tail of
        # older pieces behind for the next read to concatenate onto it.
        self.forget(session.base_url)
        for index in range(0, len(payload), CHUNK_BYTES):
            piece = payload[index : index + CHUNK_BYTES]
            self._run(
                [
                    "add-generic-password",
                    "-a",
                    _chunk(account, index // CHUNK_BYTES),
                    "-s",
                    self._service,
                    "-l",
                    f"manicule — {session.base_url}",
                    "-D",
                    "manicule Confluence session",
                    "-T",
                    SECURITY,
                    "-U",
                    "-w",
                ],
                # `-w` with no value prompts twice and reads both from stdin when stdin is not
                # a terminal. Passing the value as an argument instead would put a live
                # corporate session into this process's command line.
                stdin=f"{piece}\n{piece}\n",
            )
        if self._read(account) != payload:
            self.forget(session.base_url)
            msg = (
                f"the macOS Keychain did not give back the session that was just written for "
                f"{session.base_url}, so nothing has been stored rather than something "
                f"truncated. {SECURITY} keeps only the first 128 bytes of a secret read from "
                f"stdin, which is why the record is written in {CHUNK_BYTES}-byte pieces; a "
                f"version of macOS with a smaller buffer would land here. Put the session in "
                f"the environment variable named by session_env instead."
            )
            raise ConfigError(msg)

    def forget(self, base_url: str) -> bool:
        account = _account(base_url)
        removed = False
        for index in range(MAX_CHUNKS):
            gone = self._run(
                ["delete-generic-password", "-a", _chunk(account, index), "-s", self._service],
                absent_status=_NOT_FOUND,
            )
            if gone is None:
                break
            removed = True
        return removed

    def _read(self, account: str) -> str | None:
        """The stored payload, reassembled, or ``None`` if the first piece is not there."""
        pieces: list[str] = []
        for index in range(MAX_CHUNKS):
            found = self._run(
                ["find-generic-password", "-a", _chunk(account, index), "-s", self._service, "-w"],
                absent_status=_NOT_FOUND,
            )
            if found is None:
                break
            pieces.append(found.strip())
        if not pieces:
            return None
        return "".join(pieces)

    def _run(
        self, arguments: list[str], *, stdin: str = "", absent_status: int | None = None
    ) -> str | None:
        """Run ``security``, returning its stdout, or ``None`` for ``absent_status``.

        Raises:
            ConfigError: The command is unavailable or failed. Its stderr is included and its
                stdout is not, because stdout is where the secret would be.
        """
        if not self.available():
            msg = (
                f"{SECURITY} is not available on this machine, so manicule cannot use the "
                f"Keychain. Put the session cookies in the environment variable named by "
                f"session_env instead."
            )
            raise ConfigError(msg)
        try:
            completed = subprocess.run(  # noqa: S603 - absolute path, fixed argv, no shell
                [SECURITY, *arguments],
                input=stdin,
                capture_output=True,
                text=True,
                check=False,
                timeout=_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            msg = f"could not run {SECURITY}: {exc}"
            raise ConfigError(msg) from exc
        if completed.returncode == 0:
            return completed.stdout
        if absent_status is not None and completed.returncode == absent_status:
            return None
        msg = (
            f"{SECURITY} {arguments[0]} failed with status {completed.returncode}: "
            f"{completed.stderr.strip() or 'no detail given'}"
        )
        raise ConfigError(msg)


class MemoryStore:
    """A store that keeps sessions in this process. For tests, and for nothing else.

    Named and shipped rather than left in the suite because the store is the seam the capture
    flow is tested through, and a fake defined beside the tests would let the real flow drift
    from the one that is exercised.
    """

    def __init__(self) -> None:
        self.sessions: dict[str, BrowserSession] = {}

    def describe(self) -> str:
        return "this process's memory, which does not survive it"

    def load(self, base_url: str) -> BrowserSession | None:
        return self.sessions.get(_account(base_url))

    def save(self, session: BrowserSession) -> None:
        self.sessions[_account(session.base_url)] = session

    def forget(self, base_url: str) -> bool:
        return self.sessions.pop(_account(base_url), None) is not None


def default_store() -> SessionStore:
    """The store for this machine: the Keychain on macOS, and nothing anywhere else.

    There is no file-backed fallback on purpose. A session written to a file inherits every
    question ``docs/storage.md`` §7.1 answers for the data directory and answers none of them
    better than an environment variable does, so the platforms without a keychain get the
    environment variable rather than a second-best file.
    """
    return KeychainStore()


def load_session(
    config: ConfluenceConfig,
    *,
    environ: Mapping[str, str] | None = None,
    store: SessionStore | None = None,
    now: datetime | None = None,
) -> BrowserSession | None:
    """The stored session for this instance, or the one the environment carries, or ``None``.

    The keychain is consulted first. An environment variable that shadowed a session captured a
    moment ago would make ``manicule connector login`` look as though it had not worked, and the
    variable exists for machines that have no keychain rather than as an override.

    A session taken from the environment is recorded as captured **now**, because there is
    nothing else true to say: the variable carries a cookie and no history. The consequence is
    that ``session_max_age_hours`` does not constrain it, which is the honest reading of a
    credential supplied fresh for each run.
    """
    import os  # noqa: PLC0415 - only this function reads the environment

    env = os.environ if environ is None else environ
    keychain = store if store is not None else default_store()
    if isinstance(keychain, KeychainStore) and not KeychainStore.available():
        stored = None
    else:
        stored = keychain.load(config.base_url)
    if stored is not None:
        return stored
    raw = env.get(config.session_env, "").strip()
    if not raw:
        return None
    return BrowserSession(
        base_url=config.base_url,
        account="",
        captured_at=now if now is not None else datetime.now(tz=UTC),
        cookies=parse_cookies(raw),
    )


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

    The proof is the point. A cookie that was copied short, copied from the wrong tab, or copied
    from a session that had already timed out is indistinguishable from a working one until
    something uses it, and "something uses it" would otherwise be the first page of the next
    sync. So this makes one request as the session, reads back who the instance says that is,
    and stores nothing at all if the answer is anybody other than a signed-in user.

    Raises:
        ConfigError: The paste carried no cookies, or the instance does not answer the endpoint
            this asks.
        SessionExpiredError: The instance answered, and answered as somebody signed out.
    """
    from manicule.connectors.client import ConfluenceClient  # noqa: PLC0415 - no HTTP at import
    from manicule.connectors.credentials import BrowserSessionCredential  # noqa: PLC0415
    from manicule.connectors.errors import NotFoundError  # noqa: PLC0415

    moment = now if now is not None else datetime.now(tz=UTC)
    candidate = BrowserSession(
        base_url=config.base_url,
        account="",
        captured_at=moment,
        cookies=parse_cookies(cookie_text),
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
            f"cannot confirm who the pasted session belongs to and will not store it. Check "
            f"base_url names the site root including any context path."
        )
        raise ConfigError(msg) from exc
    finally:
        await client.teardown()

    account = _named(payload)
    if not account:
        msg = (
            f"{config.base_url}{_PROBE_PATH} answered without naming a user, so manicule "
            f"cannot confirm the pasted session is signed in and will not store it."
        )
        raise ConfigError(msg)

    session = BrowserSession(
        base_url=config.base_url,
        account=account,
        captured_at=moment,
        cookies=candidate.cookies,
    )
    keychain = store if store is not None else default_store()
    keychain.save(session)
    return session


def _named(payload: Mapping[str, object]) -> str:
    """Who a ``user/current`` response says it is, under whichever key this version uses."""
    for key in ("username", "accountId", "userKey", "email"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _chunk(account: str, index: int) -> str:
    """The keychain account one piece of a session is filed under."""
    return f"{account}#{index}"


def _account(base_url: str) -> str:
    """The keychain account a site's session is filed under.

    Normalised so that ``https://wiki.example.com`` and ``https://wiki.example.com/`` are one
    entry rather than two, one of which would be found and the other silently not.
    """
    return base_url.strip().rstrip("/")
