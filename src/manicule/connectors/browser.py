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
    "BrowserSessionProvider",
    "CandidateCookie",
    "PlaywrightProvider",
    "cookies_from_state",
    "origin_cookies",
    "read_state_file",
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
    would import a neighbouring application's cookies on any instance that hosts two.
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
    theatre, and the residual risk is documented instead.

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
                return await self._wait(context, browser, config=config, deadline=deadline)
            finally:
                # The browser is closed on every path, including the timeout and the refusal. A
                # left-open headed Chromium is a window the person did not ask for holding a live
                # session, which is worse than the failure that produced it.
                await _closed(browser)

    async def _wait(
        self,
        context: BrowserContext,
        browser: Browser,
        *,
        config: ConfluenceConfig,
        deadline: float,
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

        Distinguishing the last two costs one boolean, and the signal is already in hand: whether
        :func:`origin_cookies` ever returned anything.
        """
        import asyncio  # noqa: PLC0415 - kept beside its only use

        from manicule.connectors.sessions import cookies_authenticate  # noqa: PLC0415

        loop = asyncio.get_running_loop()
        reached_confluence = False
        while True:
            if not browser.is_connected():
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
            await asyncio.sleep(self._poll)


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
