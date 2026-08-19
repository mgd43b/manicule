"""Signing in to Confluence through a browser the person drives, and taking only the cookies.

:mod:`.sessions` records why manicule spent a release *not* doing this, and that argument was
sound: a driven browser is one manicule could read the DOM of, and the person is asked to type a
corporate password into it. What changed is not the argument but the alternative. An instance
behind an identity provider commonly has personal access tokens disabled by policy, so the manual
path — open developer tools, find the ``Cookie`` header, paste several kilobytes of secret into a
terminal — was not merely inconvenient. It was the *only* path, and it asks somebody to handle raw
session material by hand in order to avoid a risk they are also capable of running into.

**So the property has genuinely weakened, and this module is where that is written down.** It used
to be "manicule cannot see the password", a fact about capability: there was no browser, so there
was nothing to read a password out of. It is now "manicule does not see the password", a fact
about this code. The difference is real and no amount of prose closes it. What replaces it:

*No DOM is read, ever.* Nothing here calls ``page.content()``, ``page.query_selector``,
    ``page.evaluate`` or any other accessor of what the page contains. The browser is opened at
    the configured base URL and then **observed only through its cookie jar and its own report of
    whether it is still open**. ``tests/connectors/test_browser_login.py`` asserts the absence of
    those calls over this module's source, which is a weaker guarantee than not having a browser
    and a stronger one than a promise.

*Nothing is automated.* No field is filled, no button is clicked, no keystroke is sent. The person
    signs in exactly as they would in their own browser, including whatever second factor,
    passkey or conditional-access step their provider demands.

*The manual path is not deprecated.* It keeps the stronger property for anyone who wants it, and
    it is documented as such rather than as a legacy fallback.

**Only the configured origin's cookies are taken.** Signing in through an identity provider means
visiting that provider, which sets cookies of its own — an account the person may use at other
companies. Those are not manicule's business, are not needed to call Confluence, and are filtered
out by :func:`origin_cookies` before anything is stored. The same function does the same job for
:func:`cookies_from_state`, so an imported state file and a live browser cannot disagree about
which cookies belong.

**Nothing here decides that authentication succeeded.** A browser reaching a plausible URL proves
nothing: SSO flows land on portals, dashboards and interstitials, and a session can be dead at a
URL that looks fine. Success is decided by :func:`~manicule.connectors.sessions.capture_cookies`
making a real request to the instance and reading back who it says is signed in — the same check,
through the same client, that the manual paste path has always used.
"""

from __future__ import annotations

import json
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast
from urllib.parse import urlsplit

from pydantic import SecretStr

from manicule.core.errors import ConfigError

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from collections.abc import Callable, Mapping, Sequence
    from contextlib import AbstractAsyncContextManager as AsyncContextManager

    from playwright.async_api import Browser, BrowserContext, Playwright

    from manicule.connectors.config import ConfluenceConfig

__all__ = [
    "BROWSER_EXTRA_ADVICE",
    "MAX_STATE_BYTES",
    "SUPPORTED_BROWSERS",
    "BrowserSessionProvider",
    "CandidateCookie",
    "InstalledChromiumProvider",
    "PlaywrightProvider",
    "cookies_from_state",
    "installed_browsers",
    "origin_cookies",
    "private_profile",
    "read_state_file",
    "resolve_browser",
]

BROWSER_EXTRA_ADVICE = (
    "browser sign-in needs Playwright, which manicule does not install by default because it "
    "downloads a browser. Install it with:\n"
    "    pip install 'manicule[browser-auth]'\n"
    "    playwright install chromium\n"
    "Or sign in to Confluence in your own browser and use `manicule connector login <name>` "
    "without --browser, which asks for the Cookie header instead and needs nothing installed."
)
"""What to do when the driver is not there.

Both commands, because installing the package is half of it and the half that is easy to miss —
``playwright install chromium`` is a separate download and its absence fails at launch rather
than at import. The manual path is named as well, deliberately: this refusal must not *silently*
become the paste prompt (a person who asked for a browser and got a prompt concludes the feature
is broken), but it should still tell them the thing that works right now with nothing installed.
"""

MAX_STATE_BYTES = 8 * 1024 * 1024
"""Largest ``storage_state`` file that will be read.

A state document is a few hundred cookies at worst. This is roughly a thousand times that, and it
is here because the alternative is that a file the caller names decides how much memory the
command allocates — the same bound, for the same reason, as
:data:`~manicule.connectors.sidecar.MAX_MANIFEST_BYTES`.
"""

_SESSION_COOKIE = -1
"""Playwright's ``expires`` for a cookie with no expiry: it dies with the browser."""


@dataclass(frozen=True, slots=True)
class CandidateCookie:
    """One cookie as a browser reports it, before anything decides it is relevant.

    A record rather than a raw mapping so that :func:`origin_cookies` can be exercised against
    every combination of host-only, secure, path and expiry without constructing a browser or a
    state file. The field names are Playwright's, because this is a wire format read from two
    places — a live context and a ``storage_state`` document — and renaming them here would mean
    two translations to keep in step.
    """

    name: str
    value: str
    domain: str
    path: str = "/"
    expires: float = _SESSION_COOKIE
    secure: bool = False

    @property
    def host_only(self) -> bool:
        """Whether this cookie is for exactly one host rather than a domain and its subdomains.

        The leading dot is the wire format's own marker and browsers still emit it. A cookie set
        without one belongs to the host that set it alone, which is the difference between a
        cookie for ``confluence.example.test`` and one for every host under ``example.test``.
        """
        return not self.domain.startswith(".")


class BrowserSessionProvider(Protocol):
    """Something that can get a person signed in and hand back the resulting cookies.

    One method, and it deliberately returns *candidates* rather than a stored session. Whether
    those cookies actually authenticate is not a question a browser can answer — see the module
    docstring — so the provider's job ends at "the person finished and here is the jar", and
    :func:`~manicule.connectors.sessions.capture_cookies` decides the rest. That split is what
    lets every test of the login flow run without a browser: the seam is where the fake goes.
    """

    async def authenticate(
        self, config: ConfluenceConfig, *, timeout_seconds: float
    ) -> Sequence[CandidateCookie]:
        """Open a browser at ``config.base_url`` and return the cookies once signed in.

        Raises:
            ConfigError: The browser could not be started, the person closed it, or
                ``timeout_seconds`` elapsed. Every one of them is actionable and none of them
                carries a cookie value.
        """
        ...


def origin_cookies(
    candidates: Sequence[CandidateCookie],
    *,
    base_url: str,
    now: datetime | None = None,
) -> dict[str, SecretStr]:
    """The cookies a request to ``base_url`` would actually send, and no others.

    This is the filter that keeps an identity provider's cookies out of manicule. Signing in
    through one means visiting it, and it sets cookies for its own domain — frequently an account
    the person uses at several companies. They are not needed to call Confluence and manicule has
    no business holding them, so relevance is decided here rather than by storing the jar and
    hoping.

    The rules are the browser's own, applied in the order a browser applies them:

    *Domain.* A host-only cookie must match the host exactly. A domain cookie matches that domain
        and any subdomain of it, and the match is on a label boundary — ``.example.test`` covers
        ``confluence.example.test`` and must not cover ``notexample.test``.

    *Path.* The cookie's path must be a prefix of the request's, on a segment boundary. This is
        what makes a context path work: an instance at ``/confluence`` sends a cookie scoped to
        ``/confluence`` and not one scoped to ``/jira``.

    *Secure.* A secure cookie is not sent over plain HTTP, so one is not stored for an ``http://``
        instance — it would be held and never usable.

    *Expiry.* A cookie already past its expiry is dropped rather than stored, because storing it
        produces a credential that fails on first use with nothing to say why. ``expires == -1``
        is Playwright's session cookie, which has no expiry and is the usual case here.

    Returns:
        Name to value, in the order the browser reported them. Empty when nothing applies, which
        the caller must treat as a failure rather than as an empty success — a stored session
        with no cookies authenticates as nobody.
    """
    moment = (now or datetime.now(tz=UTC)).timestamp()
    parts = urlsplit(base_url)
    host = (parts.hostname or "").lower()
    request_path = parts.path or "/"
    https = parts.scheme.lower() == "https"

    kept: dict[str, SecretStr] = {}
    for cookie in candidates:
        if not cookie.name or not cookie.value:
            continue
        if not _domain_matches(cookie, host=host):
            continue
        if not _path_matches(cookie.path, request_path):
            continue
        if cookie.secure and not https:
            continue
        if cookie.expires != _SESSION_COOKIE and cookie.expires <= moment:
            continue
        kept[cookie.name] = SecretStr(cookie.value)
    return kept


def _domain_matches(cookie: CandidateCookie, *, host: str) -> bool:
    """Whether ``cookie`` would be sent to ``host``, by the browser's own domain rule."""
    domain = cookie.domain.lower().lstrip(".")
    if not domain or not host:
        return False
    if cookie.host_only:
        return host == domain
    # A label boundary, not a substring: `.example.test` covers `wiki.example.test` and must not
    # cover `notexample.test`, which `host.endswith(domain)` alone would accept.
    return host == domain or host.endswith(f".{domain}")


def _path_matches(cookie_path: str, request_path: str) -> bool:
    """Whether a cookie scoped to ``cookie_path`` is sent for ``request_path``.

    A segment boundary rather than a string prefix, so a cookie scoped to ``/conf`` is not sent to
    ``/confluence``. That distinction is the whole of the context-path case, and a prefix test
    would import a neighboring application's cookies on any instance that hosts two.
    """
    scope = cookie_path or "/"
    if not scope.startswith("/"):
        scope = f"/{scope}"
    target = request_path or "/"
    if scope == "/":
        return True
    scope = scope.rstrip("/") or "/"
    return target == scope or target.startswith(f"{scope}/")


def read_state_file(path: Path, *, allow_insecure: bool = False) -> str:
    """Read a ``storage_state`` document, refusing one anybody on the machine can read.

    The file holds live session cookies, so its mode is part of whether importing it is safe.
    Group- or world-readable is refused by default and consented to by name, which is the rule
    :meth:`~manicule.app.ports.Maintenance.backup` already applies to a snapshot of the corpus —
    the same shape of secret and the same shape of decision.

    The check is skipped where POSIX modes do not mean what they say. On Windows the bits a stat
    reports are not the access control the platform enforces, so refusing on them would be
    theater, and the residual risk is documented instead.

    **The file is never written to.** It is the caller's, it may be shared with other tooling, and
    a conversion that rewrote it would be a surprise in somebody else's workflow.

    Raises:
        ConfigError: The file is missing, unreadable, too large, or has permissions this refuses.
            No message includes any of the contents.
    """
    import sys  # noqa: PLC0415 - only this check reads the platform

    try:
        info = path.stat()
    except OSError as exc:
        msg = f"cannot read the browser state at {path}: {exc.strerror or exc}"
        raise ConfigError(msg) from exc
    if not stat.S_ISREG(info.st_mode):
        msg = f"{path} is not a regular file, so it is not a browser state document"
        raise ConfigError(msg)
    if info.st_size > MAX_STATE_BYTES:
        msg = (
            f"{path} is {info.st_size} bytes, over the {MAX_STATE_BYTES}-byte limit for a "
            f"browser state document. A state file is a few hundred cookies; this is something "
            f"else."
        )
        raise ConfigError(msg)
    # Read and write bits only. `S_IRWXG | S_IRWXO` would include *execute*, and a state file at
    # mode 0601 is not readable by anybody else — refusing it would be a refusal the message
    # cannot justify, since the message says "readable". Write counts as well as read: a file
    # somebody else can rewrite is one whose contents this could be made to import.
    exposed = info.st_mode & (stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)
    if exposed and not allow_insecure and sys.platform != "win32":
        msg = (
            f"{path} is readable or writable by other users on this machine (mode "
            f"{stat.S_IMODE(info.st_mode):04o}), and it holds live session cookies. Run "
            f"`chmod 600 {path}`, or pass --allow-insecure-state to import it anyway."
        )
        raise ConfigError(msg)
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        msg = f"cannot read the browser state at {path}: {exc}"
        raise ConfigError(msg) from exc


def cookies_from_state(raw: str) -> list[CandidateCookie]:
    """Every cookie a Playwright ``storage_state`` document declares, parsed defensively.

    Defensively because this is somebody else's file: it may be from another tool, another
    Playwright version, or hand-edited. An entry that is not a usable cookie is skipped rather
    than raised over, and the *absence of any usable cookie* is what the caller refuses — one
    malformed row in a good file should not cost the import, and a file of nothing but malformed
    rows is not a state document.

    ``origins`` — local storage — is deliberately ignored. Confluence authenticates with cookies,
    manicule has no use for the rest, and importing it would mean holding page state that has
    nothing to do with being signed in.

    **No part of the document reaches an error message.** A parse failure names the shape that was
    wrong, never the value: the whole file is secret, and a diagnostic that quoted the offending
    entry would put a session cookie in a terminal and a log.

    Raises:
        ConfigError: The document is not JSON, or is not an object with a ``cookies`` array.
    """
    try:
        loaded: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        # The position is safe to report and the text is not, so the message carries the former.
        msg = (
            f"the browser state is not valid JSON ({exc.msg} at line {exc.lineno} column "
            f"{exc.colno}). Pass the file Playwright's `storage_state` wrote."
        )
        raise ConfigError(msg) from exc
    if not isinstance(loaded, dict):
        msg = (
            f"the browser state is a JSON {type(loaded).__name__}, not an object. Pass the file "
            f"Playwright's `storage_state` wrote, which has a `cookies` array in it."
        )
        raise ConfigError(msg)
    entries = cast("Mapping[str, object]", loaded).get("cookies")
    if not isinstance(entries, list):
        msg = (
            "the browser state has no `cookies` array, so there is nothing in it to sign in "
            "with. Pass the file Playwright's `storage_state` wrote."
        )
        raise ConfigError(msg)
    parsed = (_cookie_from(entry) for entry in cast("list[object]", entries))
    return [cookie for cookie in parsed if cookie is not None]


def _cookie_from(entry: object) -> CandidateCookie | None:
    """One state entry as a cookie, or ``None`` when it is not one.

    Every field is read by type rather than trusted, and a missing optional falls back to the
    browser's own default. ``None`` rather than an exception, per the module's rule that one bad
    row does not cost a good file.
    """
    if not isinstance(entry, dict):
        return None
    record = cast("Mapping[str, object]", entry)
    name = record.get("name")
    value = record.get("value")
    domain = record.get("domain")
    if not isinstance(name, str) or not isinstance(value, str) or not isinstance(domain, str):
        return None
    if not name.strip() or not domain.strip():
        return None
    path = record.get("path")
    expires = record.get("expires")
    secure = record.get("secure")
    return CandidateCookie(
        name=name,
        value=value,
        domain=domain,
        path=path if isinstance(path, str) and path else "/",
        # `True`/`False` are `int` subclasses, so a bool would otherwise be read as an expiry of
        # 1 or 0 — the second of which is in the past and would silently drop the cookie.
        expires=float(expires)
        if isinstance(expires, (int, float)) and not isinstance(expires, bool)
        else _SESSION_COOKIE,
        secure=secure if isinstance(secure, bool) else False,
    )


class PlaywrightProvider:
    """A headed browser the person drives, watched only through its cookie jar.

    Satisfies :class:`BrowserSessionProvider`. Everything about the shape of this class is the
    module docstring's constraint made mechanical: it opens a page, and from then on the only
    things it asks are *what cookies do you have* and *are you still open*. There is no call here
    that reads page content, and a test asserts that over this file's source.

    **Polling rather than an event.** Playwright can wait for a URL or a selector, and both are
    the wrong question — a URL proves nothing (the module docstring) and a selector would mean
    knowing what the identity provider's page looks like, which is exactly the DOM coupling this
    refuses. So it polls the cookie jar and asks the *instance* whether those cookies work yet,
    which is the only authority on the matter. The interval is a compromise between a person
    waiting and a instance being asked repeatedly; ``asyncio.sleep`` between polls is what keeps
    Ctrl-C responsive, since the loop is never blocked in a synchronous call for long.
    """

    def __init__(self, *, poll_seconds: float = 2.0, headless: bool = False) -> None:
        """Args:
        poll_seconds: How often to ask whether sign-in has completed.
        headless: Off, and it is a parameter only so a test can prove the launch is headed.
            A headless browser cannot show a sign-in form to a person, so this flow has no
            use for one; it exists so the assertion is about an argument rather than a
            constant nobody reads.
        """
        self._poll = poll_seconds
        self._headless = headless

    async def authenticate(
        self, config: ConfluenceConfig, *, timeout_seconds: float
    ) -> Sequence[CandidateCookie]:
        """Open the browser, wait for the person, and return the jar.

        Raises:
            ConfigError: Playwright is not installed, the browser would not start, the person
                closed it, or the timeout elapsed.
        """
        import asyncio  # noqa: PLC0415 - kept beside its only use

        launcher = _async_playwright()
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        async with launcher() as driver:
            browser = await _launched(driver, headless=self._headless)
            try:
                context = await browser.new_context()
                page = await context.new_page()
                await page.goto(config.base_url)
                return await _wait(
                    context,
                    alive=browser.is_connected,
                    config=config,
                    deadline=deadline,
                    poll_seconds=self._poll,
                )
            finally:
                # The browser is closed on every path, including the timeout and the refusal. A
                # left-open headed Chromium is a window the person did not ask for holding a live
                # session, which is worse than the failure that produced it.
                await _closed(browser)


async def _wait(
    context: BrowserContext,
    *,
    alive: Callable[[], bool],
    config: ConfluenceConfig,
    deadline: float,
    poll_seconds: float,
) -> Sequence[CandidateCookie]:
    """Poll the jar until its cookies authenticate, the browser closes, or time runs out.

    Three ways this ends badly and each says something different, because each has a
    different next action. The one worth the extra state is the last:

    *The window was closed.* Re-run and leave it open.

    *Time ran out with cookies for this instance in the jar.* Sign-in was under way and did
        not finish. A longer ``--timeout`` is the answer.

    *Time ran out having never seen a cookie for this instance at all.* Sign-in never reached
        Confluence, so waiting longer will not help — the browser was refused before it got
        there (a conditional-access policy declining a driven Chromium is the usual cause) or
        ``base_url`` names a host the sign-in never lands on. Telling this person to raise
        the timeout sends them to wait five more minutes for the same nothing.

    Distinguishing the last two costs one boolean: whether :func:`origin_cookies` ever
    returned anything.

    **One loop for every provider, and ``alive`` is why it is a parameter.** A browser this
    process launched is asked ``is_connected``; a persistent profile has no separate browser
    object and is alive while it still has a page open. Those are two questions, and a second
    copy of this loop to ask the second one is how the bundled and installed paths would come
    to disagree about what closing the window means.

    **It is a heuristic and it is worth saying which way it is wrong.** An unauthenticated
    Confluence commonly issues a session cookie on the first visit, before anybody has signed
    in — so a browser that reached the instance and was then sent away still sets the flag,
    and gets the "wait longer" message. The "never reached" case therefore fires for the
    shape where something in front of Confluence intercepts the request before Confluence
    answers at all, which is one conditional-access arrangement among several rather than the
    whole class.

    That is the safer direction to be wrong in. "Wait longer" costs somebody a timeout they
    were going to spend anyway; "give up, this will never work" told to a person whose
    provider merely needed another minute is advice that abandons a working setup. The
    message itself claims no more than it knows — it names the two usual causes rather than
    diagnosing one.
    """
    import asyncio  # noqa: PLC0415 - kept beside its only use

    from manicule.connectors.sessions import cookies_authenticate  # noqa: PLC0415

    loop = asyncio.get_running_loop()
    reached_confluence = False
    while True:
        if not alive():
            msg = (
                "the browser was closed before sign-in finished, so there is no session to "
                "store. Re-run the command and leave the window open until Confluence has "
                "loaded as your signed-in user."
            )
            raise ConfigError(msg)
        candidates = _cookies_of(await context.cookies())
        relevant = origin_cookies(candidates, base_url=config.base_url)
        if relevant:
            reached_confluence = True
            if await cookies_authenticate(config, relevant):
                return candidates
        if loop.time() >= deadline:
            raise ConfigError(_gave_up(config, reached_confluence=reached_confluence))
        await asyncio.sleep(poll_seconds)


SUPPORTED_BROWSERS = ("chrome", "chromium", "edge", "brave")
"""The Chromium-family browsers this can drive, by the name configuration uses.

**A closed list, and the name is `installed_chromium` rather than `system_default` for the same
reason.** Manicule can only claim a browser it can actually launch through Playwright's Chromium
driver and test against; "whatever this operating system considers the default" includes Firefox
and Safari, which this cannot drive at all, and a provider that named them would be a provider
that failed at launch on the machines it most loudly promised to support.

Firefox and Safari are absent rather than pending. Playwright can drive Firefox, but a Confluence
session captured there would need this whole flow re-proved against a different cookie jar
implementation, and Safari cannot be driven with a private profile at all. Both are honest
absences and the refusal says so.
"""

_MAC_CANDIDATES: Mapping[str, tuple[str, ...]] = {
    "chrome": ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",),
    "chromium": ("/Applications/Chromium.app/Contents/MacOS/Chromium",),
    "edge": ("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",),
    "brave": ("/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",),
}

_LINUX_COMMANDS: Mapping[str, tuple[str, ...]] = {
    "chrome": ("google-chrome", "google-chrome-stable"),
    "chromium": ("chromium", "chromium-browser"),
    "edge": ("microsoft-edge", "microsoft-edge-stable"),
    "brave": ("brave-browser", "brave"),
}

_WINDOWS_CANDIDATES: Mapping[str, tuple[str, ...]] = {
    "chrome": (r"C:\Program Files\Google\Chrome\Application\chrome.exe",),
    "chromium": (r"C:\Program Files\Chromium\Application\chrome.exe",),
    "edge": (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",),
    "brave": (r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",),
}


def installed_browsers() -> dict[str, Path]:
    """Every supported browser this machine has, by name.

    **Platform-appropriate rather than one clever mechanism.** macOS puts an application at a
    known path inside a bundle; Linux puts an executable on ``PATH`` under any of several names;
    Windows uses ``Program Files``. A single strategy would be a strategy that is wrong on two
    of the three, and the failure would read as "not installed" on a machine where the browser
    is sitting in the dock.

    Nothing here launches anything or reads a profile — it is a file-existence check, so it is
    safe to call from a refusal path that is trying to say what the alternatives are.
    """
    import shutil  # noqa: PLC0415 - kept beside its only use
    import sys  # noqa: PLC0415

    found: dict[str, Path] = {}
    if sys.platform == "darwin":
        for name, paths in _MAC_CANDIDATES.items():
            for raw in paths:
                if Path(raw).exists():
                    found[name] = Path(raw)
                    break
    elif sys.platform == "win32":
        for name, paths in _WINDOWS_CANDIDATES.items():
            for raw in paths:
                if Path(raw).exists():
                    found[name] = Path(raw)
                    break
    else:
        for name, commands in _LINUX_COMMANDS.items():
            for command in commands:
                located = shutil.which(command)
                if located:
                    found[name] = Path(located)
                    break
    return found


def resolve_browser(requested: str) -> Path:
    """The executable to launch, or a refusal naming what this machine actually has.

    Args:
        requested: A supported name, an absolute path to an executable, or empty to discover.

    Raises:
        ProviderRefusedError: Nothing supported is installed, the request names something that
            is not here, or discovery found several and cannot choose. **Ambiguity refuses
            rather than picking**, because picking would mean signing in through a browser the
            person did not choose — and on a machine with a work Chrome and a personal Brave,
            which one carries the corporate identity is exactly the thing they know and this
            does not.
    """
    from manicule.connectors.errors import ProviderRefusedError  # noqa: PLC0415

    asked = requested.strip()
    available = installed_browsers()
    listing = ", ".join(sorted(available)) or "none"

    if asked and ("/" in asked or "\\" in asked):
        path = Path(asked).expanduser()
        if not path.is_file():
            msg = (
                f"installed_browser names {asked!r}, which is not a file on this machine. Give "
                f"an absolute path to a Chromium-family executable, or one of "
                f"{', '.join(SUPPORTED_BROWSERS)}. Found installed: {listing}."
            )
            raise ProviderRefusedError(msg)
        return path

    if asked:
        if asked not in SUPPORTED_BROWSERS:
            msg = (
                f"installed_browser names {asked!r}, which is not a browser manicule can drive. "
                f"Supported: {', '.join(SUPPORTED_BROWSERS)}. Found installed: {listing}. "
                f"Firefox and Safari are not supported by this provider — use "
                f"`--browser-provider bundled-chromium`, or sign in yourself and use "
                f"`--browser-state` or `--manual-cookie`."
            )
            raise ProviderRefusedError(msg)
        if asked not in available:
            msg = (
                f"installed_browser is set to {asked!r} and it is not installed here. Found "
                f"installed: {listing}. Install it, set authentication.confluence."
                f"installed_browser to one of those, or use `--browser-provider "
                f"bundled-chromium`, `--browser-state <file>` or `--manual-cookie`."
            )
            raise ProviderRefusedError(msg)
        return available[asked]

    if not available:
        msg = (
            f"no supported browser is installed, so there is nothing for installed_chromium to "
            f"drive. Supported: {', '.join(SUPPORTED_BROWSERS)}. Install one, or use "
            f"`--browser-provider bundled-chromium` (which downloads its own), "
            f"`--browser-state <file>`, or `--manual-cookie`."
        )
        raise ProviderRefusedError(msg)
    if len(available) > 1:
        msg = (
            f"several supported browsers are installed ({listing}) and manicule will not choose "
            f"which one holds your work identity. Set authentication.confluence."
            f"installed_browser to one of them."
        )
        raise ProviderRefusedError(msg)
    return next(iter(available.values()))


def private_profile(path: Path) -> Path:
    """Create the authentication profile directory, user-only, refusing an unsafe one.

    The profile holds live session cookies once it has been signed in to, so it is a credential
    at rest and gets the treatment :func:`read_state_file` already gives an imported state file:
    a symlink is refused rather than followed, a directory other users can reach is refused
    rather than tightened, and the refusal names the command that fixes it.

    **Refused rather than repaired**, because a directory that is group-writable may already
    have been written to by somebody else, and quietly chmod-ing it would hide that this had
    been true. The check does not apply on Windows, where the POSIX bits a stat reports are not
    the access control the platform enforces.

    Raises:
        ProviderRefusedError: The path is a symlink, is not a directory, cannot be created, or
            other users on this machine can reach it.
    """
    import sys  # noqa: PLC0415 - kept beside its only use

    from manicule.connectors.errors import ProviderRefusedError  # noqa: PLC0415

    resolved = path.expanduser()
    if resolved.is_symlink():
        msg = (
            f"the browser profile path {resolved} is a symlink. It holds live session cookies "
            f"once it is signed in to, so manicule will not follow one to write a credential "
            f"somewhere it cannot vouch for. Point profile_dir at a real directory."
        )
        raise ProviderRefusedError(msg)
    try:
        resolved.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        msg = f"cannot create the browser profile directory {resolved}: {exc.strerror or exc}"
        raise ProviderRefusedError(msg) from exc
    if not resolved.is_dir():
        msg = f"the browser profile path {resolved} exists and is not a directory"
        raise ProviderRefusedError(msg)
    if sys.platform != "win32":
        mode = resolved.stat().st_mode
        exposed = mode & (
            stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IWOTH | stat.S_IXOTH
        )
        if exposed:
            msg = (
                f"the browser profile directory {resolved} is reachable by other users on this "
                f"machine (mode {stat.S_IMODE(mode):04o}), and it holds live session cookies. "
                f"Run `chmod 700 {resolved}`."
            )
            raise ProviderRefusedError(msg)
    return resolved


class InstalledChromiumProvider:
    """A browser already on this machine, driven through a profile of manicule's own.

    Satisfies :class:`BrowserSessionProvider`, and exists because of one thing the bundled
    Chromium cannot do: an identity provider's conditional-access policy commonly recognizes the
    browser a person actually uses and declines one it has never seen. Driving the installed
    build is the difference between a sign-in that completes and one that sits at a policy screen
    until the timeout.

    **It does not touch the person's own profile, and that is the boundary rather than a
    default.** An ordinary daily-use profile is deliberately not available to an unrelated
    process; the ways to take it anyway are remote debugging on a running browser, copying the
    profile directory, or decrypting the cookie database, and all three are refused by this
    project rather than implemented behind a flag. What is honest is a dedicated profile that
    manicule creates and signs in to, which is what this is. The cost is stated plainly: the
    person signs in there the first time, and the identity provider sees a new profile of a
    familiar browser rather than a familiar profile.

    **Automation is not concealed.** No user-agent is forged, no automation flag is suppressed,
    and no security warning is dismissed. A policy may still refuse this, and when it does the
    refusal says so and names the alternatives rather than retrying with the disguise on.

    Everything the class asks of the browser is the cookie jar and whether a window is still
    open — the same two questions :class:`PlaywrightProvider` asks, through the same loop, and
    the test that asserts no page content is read over this module's source covers both.
    """

    def __init__(
        self,
        *,
        executable: Path,
        profile_dir: Path,
        poll_seconds: float = 2.0,
        headless: bool = False,
    ) -> None:
        """Args:
        executable: The browser to launch, already resolved by :func:`resolve_browser`.
        profile_dir: The dedicated profile, already created by :func:`private_profile`.
        poll_seconds: How often to ask whether sign-in has completed.
        headless: Off. A parameter only so a test can prove the launch is headed, on the same
            principle as :class:`PlaywrightProvider`.
        """
        self._executable = executable
        self._profile = profile_dir
        self._poll = poll_seconds
        self._headless = headless

    async def authenticate(
        self, config: ConfluenceConfig, *, timeout_seconds: float
    ) -> Sequence[CandidateCookie]:
        """Open the installed browser at the instance and return the jar once it authenticates.

        Raises:
            ProviderRefusedError: Playwright is not installed, or the browser would not start.
                Named rather than reported as a generic failure, because this is the path that
                must not become "so we used the bundled one instead".
            ConfigError: The person closed the window, or the timeout elapsed.
        """
        import asyncio  # noqa: PLC0415 - kept beside its only use

        from manicule.connectors.errors import ProviderRefusedError  # noqa: PLC0415

        launcher = _async_playwright()
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        async with launcher() as driver:
            try:
                context = await driver.chromium.launch_persistent_context(
                    str(self._profile),
                    executable_path=str(self._executable),
                    headless=self._headless,
                )
            except Exception as exc:
                msg = (
                    f"the installed browser at {self._executable} would not start "
                    f"({type(exc).__name__}). Nothing has been stored and any previously stored "
                    f"session is untouched. manicule does not fall back to another browser: "
                    f"re-run with `--browser-provider bundled-chromium` to use Playwright's own "
                    f"Chromium, or sign in yourself and use `--browser-state <file>` or "
                    f"`--manual-cookie`."
                )
                raise ProviderRefusedError(msg) from exc
            try:
                page = await context.new_page()
                await page.goto(config.base_url)
                # A persistent context has no separate browser object to ask, so liveness is
                # "has it still got a page open" — closing the last window is how a person
                # cancels, and it has to end the wait rather than spin to the timeout.
                return await _wait(
                    context,
                    alive=lambda: bool(context.pages),
                    config=config,
                    deadline=deadline,
                    poll_seconds=self._poll,
                )
            finally:
                # Closed on every path, as the bundled provider is: a left-open window holding a
                # live session is worse than whatever failure produced it.
                await _closed_context(context)


async def _closed_context(context: BrowserContext) -> None:
    """Close a persistent context, ignoring a failure to.

    Mirrors :func:`_closed`. A close that raises during cleanup would replace the refusal the
    caller is about to see with one about shutting a window, which is never the more useful of
    the two.
    """
    import contextlib  # noqa: PLC0415 - kept beside its only use

    with contextlib.suppress(Exception):
        await context.close()


def _gave_up(config: ConfluenceConfig, *, reached_confluence: bool) -> str:
    """What to say when the timeout expires, which depends on how far sign-in got.

    Two sentences rather than one, because "wait longer" is right for one of these and actively
    wrong for the other — and the person who was refused by a conditional-access policy is the
    one least able to tell which they are in, since both look like a browser that sat there.
    """
    untouched = "Nothing has been stored and any previously stored session is untouched."
    if reached_confluence:
        return (
            f"sign-in did not finish before the timeout, though the browser did reach "
            f"{config.base_url}. {untouched} Re-run with a longer --timeout if the identity "
            f"provider needs more steps than that allowed."
        )
    return (
        f"the browser never received a cookie from {config.base_url} before the timeout, so "
        f"sign-in did not reach Confluence at all. {untouched} A longer --timeout will not help. "
        f"The two usual causes are a conditional-access policy declining a browser it does not "
        f"recognize — in which case sign in with your own browser and use `manicule connector "
        f"login <name>` without --browser — and a base_url that names a different host from the "
        f"one sign-in lands on."
    )


def _async_playwright() -> Callable[[], AsyncContextManager[Playwright]]:
    """Playwright's async entry point, or a refusal naming both installation steps.

    Imported here rather than at module scope so that an installation authenticating with an API
    token never loads a browser driver — the rule every optional dependency in this project
    follows.

    Raises:
        ConfigError: The package is not installed.
    """
    try:
        from playwright.async_api import async_playwright  # noqa: PLC0415 - optional
    except ImportError as exc:
        raise ConfigError(BROWSER_EXTRA_ADVICE) from exc
    return async_playwright


async def _launched(driver: Playwright, *, headless: bool) -> Browser:
    """Start Chromium, or say what to install when the binary is not there.

    The package being importable and the browser being downloaded are two different states, and
    only the second fails here. Reporting them with one message would send somebody to re-run a
    pip install that already worked.

    Raises:
        ConfigError: The browser could not be launched.
    """
    try:
        return await driver.chromium.launch(headless=headless)
    except Exception as exc:
        msg = (
            f"the browser would not start ({type(exc).__name__}). If Playwright is installed but "
            f"its browser is not, run:\n    playwright install chromium"
        )
        raise ConfigError(msg) from exc


async def _closed(browser: Browser) -> None:
    """Close the browser, ignoring a failure to do so.

    Raising here would replace whatever actually went wrong — a timeout, a refusal — with the
    fact that a window would not shut, which is never the more useful message.
    """
    import contextlib  # noqa: PLC0415 - kept beside its only use

    with contextlib.suppress(Exception):
        await browser.close()


def _cookies_of(jar: Sequence[Mapping[str, object]]) -> list[CandidateCookie]:
    """A Playwright cookie list as records this module can filter.

    The same parser the state file goes through, so a live jar and an imported document cannot be
    read two different ways.
    """
    parsed = (_cookie_from(entry) for entry in jar)
    return [cookie for cookie in parsed if cookie is not None]
