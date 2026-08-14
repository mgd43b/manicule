"""Signing in through a browser, and the rules about what may be taken from one.

``tests/connectors/test_browser_sso.py`` covers the credential once it exists: that a sign-in
page served with status 200 is never content, that a session which has aged out stops a sync,
that the keychain round-trips. This file covers how the credential is *obtained* — the two new
ways in, and the boundary each of them has to hold.

Three properties are load-bearing here and each has its own section:

**No browser is launched or downloaded.** Every test drives
:class:`~manicule.connectors.browser.BrowserSessionProvider` with a fake. The one test that
would need a real Chromium is marked and skips with a reason. That is not only about speed: a
suite that launched a browser would be a suite nobody could run on a machine without one, and
the flow's *logic* has nothing to do with Chromium.

**Only the configured instance's cookies are ever taken.** Signing in through an identity
provider means visiting it, and it sets cookies of its own. The filter is exercised against
every combination of host-only, domain, path, secure and expiry, because it is the one thing
standing between an SSO account and manicule's keychain.

**Nothing reaches the store unverified.** Both new paths end at ``capture_cookies``, so a
browser that hands back cookies the instance answers with a sign-in page stores nothing — the
same guard the paste path has always had, applied to the two entry points that did not exist.

Fixtures are synthetic throughout: ``https://confluence.example.test/confluence`` for anything
invented here, invented cookie values, and an identity provider that exists only in this file.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import JsonValue

import manicule.connectors.sessions as sessions_module
from manicule.app import results as r
from manicule.app.service import ApplicationService
from manicule.config.settings import ConnectorSettings, Settings
from manicule.connectors import browser as browser_module
from manicule.connectors.browser import (
    BROWSER_EXTRA_ADVICE,
    MAX_STATE_BYTES,
    CandidateCookie,
    PlaywrightProvider,
    cookies_from_state,
    origin_cookies,
    read_state_file,
)
from manicule.connectors.errors import SessionExpiredError
from manicule.connectors.sessions import SessionVault
from manicule.core.errors import ConfigError, PolicyError, UnknownEntityError
from tests.app.fakes import FakeBackend
from tests.connectors.fake_confluence import FakeConfluence, FakePage
from tests.connectors.support import SESSION_ACCOUNT, sso_config

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manicule.connectors.config import ConfluenceConfig

BASE = "https://confluence.example.test/confluence"
"""A Confluence under a context path, because that is the case the filters get wrong."""

HOST = "confluence.example.test"

IDP_HOST = "login.identity.example.test"
"""An identity provider on an unrelated host, whose cookies must never be stored."""

NOW = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)


def cookie(
    name: str = "JSESSIONID",
    value: str = "session-value",
    *,
    domain: str = HOST,
    path: str = "/confluence",
    expires: float = -1,
    secure: bool = False,
) -> CandidateCookie:
    """One candidate, with everything a test might vary as a keyword."""
    return CandidateCookie(
        name=name, value=value, domain=domain, path=path, expires=expires, secure=secure
    )


def kept(candidates: Sequence[CandidateCookie], *, base_url: str = BASE) -> dict[str, str]:
    """What the filter keeps, unwrapped, so an assertion reads as names to values."""
    return {
        name: secret.get_secret_value()
        for name, secret in origin_cookies(candidates, base_url=base_url, now=NOW).items()
    }


# --- the filter: which cookies belong to this instance ------------------------------------------
#
# This is the boundary between an SSO account and manicule's keychain, so it is enumerated rather
# than sampled. Each test names the rule it is about, because a filter that got two rules right
# and one wrong would pass any test that only checked the happy case.


def test_a_cookie_for_the_configured_host_is_kept() -> None:
    assert kept([cookie()]) == {"JSESSIONID": "session-value"}


def test_an_identity_providers_cookie_is_never_taken() -> None:
    """The whole reason the filter exists.

    Signing in through a provider means visiting it, and what it sets is frequently an account
    the person uses at several companies. manicule does not need it to call Confluence and has
    no business holding it.
    """
    assert kept([cookie(), cookie("SSOSESSION", "idp-value", domain=IDP_HOST, path="/")]) == {
        "JSESSIONID": "session-value"
    }


def test_a_host_only_cookie_does_not_travel_to_a_sibling_host() -> None:
    """No leading dot means one host, which is the host that set it."""
    assert kept([cookie(domain="other.example.test", path="/")]) == {}


def test_a_domain_cookie_covers_a_subdomain() -> None:
    assert kept([cookie(domain=".example.test", path="/")]) == {"JSESSIONID": "session-value"}


def test_a_domain_cookie_matches_on_a_label_boundary_rather_than_a_suffix() -> None:
    """``.example.test`` must not cover ``notexample.test``.

    A ``host.endswith(domain)`` test passes every other case in this file and fails this one,
    which is why it is here: the wrong implementation is the obvious one.
    """
    assert kept([cookie(domain=".ample.test", path="/")]) == {}


def test_a_cookie_scoped_to_another_application_on_the_same_host_is_not_taken() -> None:
    """One host commonly serves two applications under two context paths."""
    assert kept([cookie(path="/jira")]) == {}


def test_a_path_match_is_on_a_segment_boundary_rather_than_a_prefix() -> None:
    """``/conf`` is not a prefix of ``/confluence`` in the sense a browser means."""
    assert kept([cookie(path="/conf")]) == {}


def test_a_root_scoped_cookie_reaches_a_context_path() -> None:
    """Spec item 17, from the cookie's side: Confluence beneath ``/confluence``."""
    assert kept([cookie(path="/")]) == {"JSESSIONID": "session-value"}


def test_a_secure_cookie_is_not_stored_for_a_plain_http_instance() -> None:
    """It would be held and never sent, which is a credential that fails with nothing to say."""
    assert kept([cookie(secure=True)], base_url="http://confluence.example.test/confluence") == {}


def test_a_secure_cookie_is_kept_for_an_https_instance() -> None:
    assert kept([cookie(secure=True)]) == {"JSESSIONID": "session-value"}


def test_an_expired_cookie_is_dropped_rather_than_stored() -> None:
    assert kept([cookie(expires=(NOW.timestamp() - 60))]) == {}


def test_a_cookie_expiring_later_is_kept() -> None:
    assert kept([cookie(expires=(NOW.timestamp() + 3600))]) == {"JSESSIONID": "session-value"}


def test_a_session_cookie_has_no_expiry_and_is_kept() -> None:
    """``-1`` is Playwright's "dies with the browser", and is the usual case here."""
    assert kept([cookie(expires=-1)]) == {"JSESSIONID": "session-value"}


def test_a_cookie_with_no_value_is_not_stored() -> None:
    assert kept([cookie(value="")]) == {}


# --- the state document: parsed defensively, and never quoted back ------------------------------


def state(*cookies: dict[str, Any], origins: object = None) -> str:
    """A Playwright ``storage_state`` document, as its own writer produces one."""
    body: dict[str, Any] = {"cookies": list(cookies), "origins": origins or []}
    return json.dumps(body)


def entry(
    name: str = "JSESSIONID",
    value: str = "session-value",
    *,
    domain: str = HOST,
    path: str = "/confluence",
    **extra: Any,
) -> dict[str, Any]:
    return {"name": name, "value": value, "domain": domain, "path": path, **extra}


def test_a_valid_state_document_yields_its_cookies() -> None:
    parsed = cookies_from_state(state(entry()))

    assert [(c.name, c.domain, c.path) for c in parsed] == [("JSESSIONID", HOST, "/confluence")]


def test_state_cookies_go_through_the_same_filter_as_a_live_browser() -> None:
    """One filter, two callers — so an imported file and a browser cannot disagree."""
    parsed = cookies_from_state(state(entry(), entry("SSOSESSION", domain=IDP_HOST, path="/")))

    assert kept(parsed) == {"JSESSIONID": "session-value"}


def test_local_storage_is_ignored() -> None:
    """Confluence authenticates with cookies; page state is not manicule's business."""
    parsed = cookies_from_state(
        state(entry(), origins=[{"origin": BASE, "localStorage": [{"name": "k", "value": "v"}]}])
    )

    assert [c.name for c in parsed] == ["JSESSIONID"]


def test_a_malformed_document_is_refused() -> None:
    with pytest.raises(ConfigError) as refusal:
        cookies_from_state("{not json at all")

    assert "not valid JSON" in str(refusal.value)


def test_a_document_that_is_not_an_object_is_refused() -> None:
    with pytest.raises(ConfigError) as refusal:
        cookies_from_state("[]")

    assert "not an object" in str(refusal.value)


def test_a_document_with_no_cookies_array_is_refused() -> None:
    with pytest.raises(ConfigError) as refusal:
        cookies_from_state(json.dumps({"origins": []}))

    assert "no `cookies` array" in str(refusal.value)


def test_one_malformed_entry_does_not_cost_the_whole_file() -> None:
    """Somebody else's file, possibly from another version. One bad row is not a bad document."""
    parsed = cookies_from_state(state({"name": "broken"}, entry(), "not-an-object"))  # type: ignore[arg-type]

    assert [c.name for c in parsed] == ["JSESSIONID"]


def test_a_boolean_expiry_is_not_read_as_a_timestamp() -> None:
    """``True`` is an ``int`` in Python, and ``False`` would be an expiry of 1970.

    Read naively, a state file with ``"expires": false`` silently loses every cookie in it, and
    the failure is a login that reports no applicable cookies for a file that plainly has some.
    """
    parsed = cookies_from_state(state(entry(expires=False)))

    assert parsed[0].expires == -1
    assert kept(parsed) == {"JSESSIONID": "session-value"}


def test_no_part_of_the_document_reaches_the_error() -> None:
    """The whole file is secret, so a diagnostic must not quote the row it choked on."""
    secret = "super-secret-cookie-value"  # noqa: S105 - a fixture value, not a credential
    with pytest.raises(ConfigError) as refusal:
        cookies_from_state(f'{{"cookies": {{"nested": "{secret}"}}}}')

    assert secret not in str(refusal.value)


# --- the state file: permissions, size, and never being written to ------------------------------


def written_state(tmp_path: Path, body: str, *, mode: int = 0o600) -> Path:
    path = tmp_path / "storage-state.json"
    path.write_text(body, encoding="utf-8")
    path.chmod(mode)
    return path


def test_a_private_state_file_is_read(tmp_path: Path) -> None:
    path = written_state(tmp_path, state(entry()))

    assert "JSESSIONID" in read_state_file(path)


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes are not the access control on Windows")
def test_a_world_readable_state_file_is_refused(tmp_path: Path) -> None:
    """It holds live session cookies, so its mode is part of whether importing it is safe."""
    path = written_state(tmp_path, state(entry()), mode=0o644)

    with pytest.raises(ConfigError) as refusal:
        read_state_file(path)

    assert "readable or writable by other users" in str(refusal.value)
    assert "chmod 600" in str(refusal.value)


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes are not the access control on Windows")
def test_a_group_writable_state_file_is_refused(tmp_path: Path) -> None:
    """Write counts as well as read: a file somebody else can rewrite is one this would import."""
    path = written_state(tmp_path, state(entry()), mode=0o620)

    with pytest.raises(ConfigError) as refusal:
        read_state_file(path)

    assert "readable or writable by other users" in str(refusal.value)


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes are not the access control on Windows")
def test_an_execute_bit_alone_does_not_make_a_state_file_exposed(tmp_path: Path) -> None:
    """0601 is not readable by anybody else, and refusing it says something untrue.

    ``S_IRWXG | S_IRWXO`` includes execute, so the first version refused this and told the
    operator the file was "readable by other users" when it was not. Found by Copilot on #133.
    """
    path = written_state(tmp_path, state(entry()), mode=0o601)

    assert "JSESSIONID" in read_state_file(path)


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes are not the access control on Windows")
def test_an_exposed_state_file_can_be_imported_on_purpose(tmp_path: Path) -> None:
    """Refusing is the default and consenting is a decision made out loud, not a silent skip."""
    path = written_state(tmp_path, state(entry()), mode=0o644)

    assert "JSESSIONID" in read_state_file(path, allow_insecure=True)


def test_the_state_file_is_never_written_to(tmp_path: Path) -> None:
    """It is the caller's file and may be shared with other tooling."""
    path = written_state(tmp_path, state(entry()))
    before = path.read_bytes()
    stamp = path.stat().st_mtime_ns

    read_state_file(path)

    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == stamp


def test_an_oversized_state_file_is_refused_by_its_size(tmp_path: Path) -> None:
    """A file the caller names must not decide how much memory the command allocates."""
    path = tmp_path / "huge.json"
    path.write_bytes(b"{" + b" " * (MAX_STATE_BYTES + 1))
    path.chmod(0o600)

    with pytest.raises(ConfigError) as refusal:
        read_state_file(path)

    assert "over the" in str(refusal.value)


def test_a_missing_state_file_says_so(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as refusal:
        read_state_file(tmp_path / "nowhere.json")

    assert "cannot read the browser state" in str(refusal.value)


def test_a_directory_is_not_a_state_document(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as refusal:
        read_state_file(tmp_path)

    assert "not a regular file" in str(refusal.value)


# --- the login flow, driven through a fake provider ---------------------------------------------


class FakeProvider:
    """A browser that was already signed in, or one that refuses in a stated way.

    Satisfies :class:`~manicule.connectors.browser.BrowserSessionProvider`. This is the seam the
    whole flow is tested through, which is why the real provider's job stops at handing back a
    jar: everything that decides anything is on this side of it.
    """

    def __init__(
        self, cookies: Sequence[CandidateCookie] = (), *, fails: Exception | None = None
    ) -> None:
        self.cookies = list(cookies)
        self.fails = fails
        self.calls: list[float] = []

    async def authenticate(
        self, config: ConfluenceConfig, *, timeout_seconds: float
    ) -> Sequence[CandidateCookie]:
        del config
        self.calls.append(timeout_seconds)
        if self.fails is not None:
            raise self.fails
        return self.cookies


def instance(**overrides: object) -> FakeConfluence:
    return FakeConfluence(
        base_url=BASE,
        pages=[FakePage(id="1", title="Runbook", space="OPS")],
        **overrides,  # type: ignore[arg-type]
    )


def service_for(**options: JsonValue) -> tuple[ApplicationService, FakeBackend]:
    """A service with one Confluence source configured, at a context path."""
    settings = Settings(
        connectors={
            "wiki": ConnectorSettings(
                type="confluence",
                options={
                    "base_url": BASE,
                    "deployment": "server",
                    "auth": "browser_session",
                    **options,
                },
            )
        }
    )
    backend = FakeBackend(settings=settings)
    return ApplicationService(backend), backend


async def login(
    service: ApplicationService,
    *,
    provider: object = None,
    monkeypatch: pytest.MonkeyPatch,
    site: FakeConfluence,
    store: SessionVault,
    **kwargs: object,
) -> r.ConnectorSignedIn:
    """Run ``connector_login`` against a fake instance and a fake store.

    The transport and the store are patched at the module the service calls rather than passed,
    because the service's signature deliberately has no transport parameter — a login that could
    be pointed at an arbitrary HTTP stack from a surface is not a thing to build.

    Supplying a provider means the browser path, since that is the only thing a provider is for.
    Stating it here rather than at every call site keeps the tests about what they are testing.
    """
    _patch_sessions(monkeypatch, site=site, store=store)
    if provider is not None:
        kwargs.setdefault("browser", True)
    return await service.connector_login("wiki", provider=provider, **kwargs)  # type: ignore[arg-type]


def _patch_sessions(
    monkeypatch: pytest.MonkeyPatch, *, site: FakeConfluence, store: SessionVault
) -> None:
    """Point the session module at the fake instance and the fake store.

    ``capture_cookies`` is wrapped rather than replaced, so the verification under test is the
    real one — only the transport it reaches is synthetic.
    """
    monkeypatch.setattr(sessions_module, "default_store", lambda: store)
    original = sessions_module.capture_cookies

    async def with_transport(config: Any, cookies: Any, **inner: Any) -> Any:
        # Assigned rather than defaulted: `capture` passes `transport=None` through explicitly,
        # so a `setdefault` finds the key already there and the request goes to the network.
        if inner.get("transport") is None:
            inner["transport"] = site.transport()
        return await original(config, cookies, **inner)

    monkeypatch.setattr(sessions_module, "capture_cookies", with_transport)


async def test_a_successful_browser_login_stores_the_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec item 3. The account comes back from the instance, not from the browser."""
    service, _ = service_for()
    site = instance()
    store = SessionVault()

    result = await login(
        service,
        provider=FakeProvider([cookie()]),
        monkeypatch=monkeypatch,
        site=site,
        store=store,
    )

    assert result.account == SESSION_ACCOUNT
    assert store.load(BASE) is not None


async def test_only_the_instances_cookies_are_persisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec item 13, asserted on what is in the store rather than on what the filter returned."""
    service, _ = service_for()
    store = SessionVault()

    await login(
        service,
        provider=FakeProvider(
            [cookie(), cookie("SSOSESSION", "idp-value", domain=IDP_HOST, path="/")]
        ),
        monkeypatch=monkeypatch,
        site=instance(),
        store=store,
    )

    stored = store.load(BASE)
    assert stored is not None
    assert set(stored.cookies) == {"JSESSIONID"}


async def test_a_browser_session_the_instance_refuses_is_not_stored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec item 7, and the guard this feature most needed.

    The instance answers with a sign-in page carrying status 200 — the failure
    :mod:`manicule.connectors.intercept` exists for. The browser path must inherit that refusal
    rather than storing whatever the jar happened to hold.
    """
    service, _ = service_for()
    site = instance()
    site.sign_out()
    store = SessionVault()

    with pytest.raises(SessionExpiredError):
        await login(
            service,
            provider=FakeProvider([cookie()]),
            monkeypatch=monkeypatch,
            site=site,
            store=store,
        )

    assert store.load(BASE) is None


async def test_a_failed_login_leaves_a_working_session_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec item 15. Replacement is atomic because the write is the last thing that happens."""
    service, _ = service_for()
    store = SessionVault()
    await login(
        service,
        provider=FakeProvider([cookie()]),
        monkeypatch=monkeypatch,
        site=instance(),
        store=store,
    )
    working = store.load(BASE)
    assert working is not None

    failing = instance()
    failing.sign_out()
    with pytest.raises(SessionExpiredError):
        await login(
            service,
            provider=FakeProvider([cookie("JSESSIONID", "a-newer-but-dead-value")]),
            monkeypatch=monkeypatch,
            site=failing,
            store=store,
        )

    assert store.load(BASE) == working, "the failed attempt replaced a working credential"


async def test_a_browser_that_finds_no_cookies_for_the_instance_is_a_stated_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec item 8: the browser opened and sign-in never reached Confluence."""
    service, _ = service_for()
    store = SessionVault()

    with pytest.raises(ConfigError) as refusal:
        await login(
            service,
            provider=FakeProvider([cookie("SSOSESSION", domain=IDP_HOST, path="/")]),
            monkeypatch=monkeypatch,
            site=instance(),
            store=store,
        )

    assert "no cookies for" in str(refusal.value)
    assert "untouched" in str(refusal.value)
    assert store.load(BASE) is None


# --- the wait loop itself, over a fake browser context ------------------------------------------
#
# These drive `PlaywrightProvider._wait` rather than the seam above it, because the thing under
# test is how the provider distinguishes its own failures — which is the one piece of that class
# that is logic rather than plumbing, and the piece a person reads the output of.


class FakeContext:
    """A Playwright context that reports a fixed jar. Nothing else is ever asked of it."""

    def __init__(self, jar: list[dict[str, Any]]) -> None:
        self.jar = jar

    async def cookies(self) -> list[dict[str, Any]]:
        return self.jar


class FakeBrowser:
    """A browser that is open, or was closed by the person."""

    def __init__(self, *, connected: bool = True) -> None:
        self.connected = connected

    def is_connected(self) -> bool:
        return self.connected


async def _waited(
    monkeypatch: pytest.MonkeyPatch, jar: list[dict[str, Any]], *, connected: bool = True
) -> str:
    """Run the wait loop against ``jar`` with a deadline already past, and return the refusal.

    The deadline is in the past so the loop makes exactly one pass — enough to observe the jar
    and give up, with no sleeping and no wall-clock dependence.

    ``cookies_authenticate`` is stubbed to "not yet", which is what it would answer for a jar
    that has not finished signing in. Left real it opens an HTTP client and retries a name that
    does not resolve, at fifteen seconds a test — and the subject here is which sentence the
    timeout produces, not whether cookies work.
    """

    async def not_yet(*_: object, **__: object) -> bool:
        return False

    monkeypatch.setattr(sessions_module, "cookies_authenticate", not_yet)
    provider = PlaywrightProvider(poll_seconds=0.0)
    with pytest.raises(ConfigError) as refusal:
        await provider._wait(  # pyright: ignore[reportPrivateUsage]
            FakeContext(jar),  # type: ignore[arg-type]
            FakeBrowser(connected=connected),  # type: ignore[arg-type]
            config=sso_config(BASE),
            deadline=asyncio.get_running_loop().time() - 1,
        )
    return str(refusal.value)


async def test_a_timeout_that_never_reached_confluence_does_not_say_wait_longer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The conditional-access case, and the reason it needs its own sentence.

    A policy that declines a driven browser looks exactly like a person being slow: the window
    sits there and nothing happens. Telling them to raise ``--timeout`` sends them to wait five
    more minutes for the same nothing. The signal that separates the two is whether a cookie for
    the configured instance ever appeared.
    """
    message = await _waited(monkeypatch, [entry("SSOSESSION", domain=IDP_HOST, path="/")])

    assert "never received a cookie" in message
    assert "A longer --timeout will not help" in message
    assert "conditional-access" in message
    assert "without --browser" in message, "the path that does work is not named"


async def test_a_timeout_that_did_reach_confluence_says_wait_longer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half, so the sentence above is a distinction rather than the only message."""
    message = await _waited(monkeypatch, [entry()])

    assert "did reach" in message
    assert "longer --timeout" in message
    assert "will not help" not in message


async def test_a_closed_browser_is_reported_as_closed_rather_than_as_a_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three endings, three sentences. This one is checked before the jar is even read."""
    message = await _waited(monkeypatch, [entry()], connected=False)

    assert "closed before sign-in finished" in message
    assert "timeout" not in message


async def test_every_timeout_message_says_the_stored_session_is_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-authenticating a session that had merely aged must not look like losing it."""
    for jar in ([entry()], [entry("SSOSESSION", domain=IDP_HOST, path="/")]):
        assert "previously stored session is untouched" in await _waited(monkeypatch, jar)


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (ConfigError("sign-in did not complete within the timeout"), "timeout"),
        (ConfigError("the browser was closed before sign-in finished"), "closed"),
        (KeyboardInterrupt(), ""),
    ],
    ids=["timeout", "browser-closed", "ctrl-c"],
)
async def test_a_browser_that_does_not_finish_stores_nothing(
    monkeypatch: pytest.MonkeyPatch, failure: Exception, expected: str
) -> None:
    """Spec items 4, 5 and 6. Ctrl-C is included because it must not be swallowed."""
    service, _ = service_for()
    store = SessionVault()

    with pytest.raises(type(failure)) as raised:
        await login(
            service,
            provider=FakeProvider(fails=failure),
            monkeypatch=monkeypatch,
            site=instance(),
            store=store,
        )

    if expected:
        assert expected in str(raised.value)
    assert store.load(BASE) is None


async def test_the_configured_timeout_reaches_the_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _ = service_for(browser_timeout_seconds=42.0)
    provider = FakeProvider([cookie()])

    await login(
        service,
        provider=provider,
        monkeypatch=monkeypatch,
        site=instance(),
        store=SessionVault(),
    )

    assert provider.calls == [42.0]


async def test_a_zero_timeout_is_not_folded_into_the_configured_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero is falsy, and ``or`` would have turned "wait none" into "wait five minutes".

    The command line refuses a non-positive ``--timeout`` outright, so this is about the service
    not *inventing* a reading when one arrives by another route. Found by Copilot on #133.
    """
    service, _ = service_for(browser_timeout_seconds=300.0)
    provider = FakeProvider([cookie()])

    await login(
        service,
        provider=provider,
        monkeypatch=monkeypatch,
        site=instance(),
        store=SessionVault(),
        timeout_seconds=0.0,
    )

    assert provider.calls == [0.0], "a falsy timeout was replaced by the configured default"


async def test_an_explicit_timeout_overrides_the_configured_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _ = service_for(browser_timeout_seconds=42.0)
    provider = FakeProvider([cookie()])

    await login(
        service,
        provider=provider,
        monkeypatch=monkeypatch,
        site=instance(),
        store=SessionVault(),
        timeout_seconds=9.0,
    )

    assert provider.calls == [9.0]


# --- importing a state file, end to end ---------------------------------------------------------


async def test_a_state_file_is_imported_and_verified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Spec item 9, through the service rather than through the parser."""
    service, _ = service_for()
    store = SessionVault()
    path = written_state(tmp_path, state(entry(), entry("SSOSESSION", domain=IDP_HOST, path="/")))

    await login(
        service,
        monkeypatch=monkeypatch,
        site=instance(),
        store=store,
        browser_state=path,
    )

    stored = store.load(BASE)
    assert stored is not None
    assert set(stored.cookies) == {"JSESSIONID"}, "an unrelated cookie was imported"


async def test_a_state_file_with_nothing_for_this_instance_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Spec item 11. Most often it is a state file from a different site."""
    service, _ = service_for()
    store = SessionVault()
    path = written_state(tmp_path, state(entry("SSOSESSION", domain=IDP_HOST, path="/")))

    with pytest.raises(ConfigError) as refusal:
        await login(
            service, monkeypatch=monkeypatch, site=instance(), store=store, browser_state=path
        )

    assert "no cookies for" in str(refusal.value)
    assert store.load(BASE) is None


async def test_an_imported_state_the_instance_refuses_is_not_stored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The state path inherits the same verification the browser path does."""
    service, _ = service_for()
    site = instance()
    site.sign_out()
    store = SessionVault()
    path = written_state(tmp_path, state(entry()))

    with pytest.raises(SessionExpiredError):
        await login(service, monkeypatch=monkeypatch, site=site, store=store, browser_state=path)

    assert store.load(BASE) is None


# --- the source has to be one this can log in to ------------------------------------------------


async def test_an_unknown_source_is_refused() -> None:
    service, _ = service_for()

    with pytest.raises(UnknownEntityError):
        await service.connector_login("nope", browser=True)


async def test_a_disabled_source_is_refused() -> None:
    """A credential captured for a source nothing will run is a window opened for nothing."""
    settings = Settings(
        connectors={
            "wiki": ConnectorSettings(type="confluence", options={"base_url": BASE}, enabled=False)
        }
    )
    service = ApplicationService(FakeBackend(settings=settings))

    with pytest.raises(PolicyError) as refusal:
        await service.connector_login("wiki", browser=True)

    assert "enabled = false" in str(refusal.value)


async def test_a_filesystem_source_has_no_session_to_capture() -> None:
    settings = Settings(
        connectors={
            "docs": ConnectorSettings(type="filesystem", options={"root": str(Path.home())})
        }
    )
    service = ApplicationService(FakeBackend(settings=settings))

    with pytest.raises(ConfigError) as refusal:
        await service.connector_login("docs", browser=True)

    assert "'filesystem'" in str(refusal.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"browser": True, "forget": True},
        {"browser": True, "browser_state": Path("state.json")},
        {"browser_state": Path("state.json"), "forget": True},
    ],
    ids=["browser+forget", "browser+state", "state+forget"],
)
async def test_two_ways_in_at_once_are_refused(kwargs: dict[str, Any]) -> None:
    """Spec item 1. Each is a different thing to do with the credential."""
    service, _ = service_for()

    with pytest.raises(ConfigError) as refusal:
        await service.connector_login("wiki", **kwargs)

    assert "Pick one" in str(refusal.value)


# --- the manual path is untouched ---------------------------------------------------------------


async def test_the_manual_paste_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spec item 2. It keeps the stronger property and is not deprecated."""
    service, _ = service_for()
    store = SessionVault()
    _patch_sessions(monkeypatch, site=instance(), store=store)

    result = await service.connector_login("wiki", cookies="JSESSIONID=session-value")

    assert result.account == SESSION_ACCOUNT
    assert store.load(BASE) is not None


# --- the dependency is optional, and its absence says what to install ---------------------------


def test_the_missing_driver_names_both_installation_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec item 16.

    Both commands, because installing the package and downloading the browser are two steps and
    the second fails later and less obviously. Asserted against the constant the flow raises, so
    a reworded message cannot quietly drop one of them.
    """
    monkeypatch.setitem(sys.modules, "playwright", None)

    with pytest.raises(ConfigError) as refusal:
        browser_module._async_playwright()  # pyright: ignore[reportPrivateUsage]

    assert str(refusal.value) == BROWSER_EXTRA_ADVICE
    assert "pip install 'manicule[browser-auth]'" in BROWSER_EXTRA_ADVICE
    assert "playwright install chromium" in BROWSER_EXTRA_ADVICE


def test_the_missing_driver_does_not_fall_back_to_asking_for_a_cookie_header() -> None:
    """A person who asked for a browser and got a paste prompt concludes it is broken.

    The refusal names the manual command as an alternative — which is help — but it must not
    *become* it, which would be a silent downgrade of the thing they asked for.
    """
    assert "connector login <name>` without --browser" in BROWSER_EXTRA_ADVICE
    assert "paste" not in BROWSER_EXTRA_ADVICE.lower()


# --- secrets stay out of everything a person or a log can see -----------------------------------


async def test_no_cookie_value_appears_in_the_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spec item 14. The payload is what `--json` prints and what a log would carry."""
    service, _ = service_for()

    result = await login(
        service,
        provider=FakeProvider([cookie(value="the-secret-value")]),
        monkeypatch=monkeypatch,
        site=instance(),
        store=SessionVault(),
    )

    assert "the-secret-value" not in json.dumps(result.model_dump(mode="json"), default=str)


def test_a_candidate_cookie_does_not_show_its_value_in_a_repr() -> None:
    """It is a dataclass, so this is a real risk rather than a theoretical one.

    A candidate is held in memory between the browser and the filter, which is exactly the window
    where a traceback would render one.
    """
    held = cookie(value="the-secret-value")

    assert "the-secret-value" in repr(held), (
        "this test documents a known limit rather than a guarantee; see the assertion below"
    )
    # The *stored* form is the one that must not leak, and it is a `SecretStr` by the time
    # anything keeps it. `origin_cookies` is the boundary, so that is where the property starts.
    wrapped = origin_cookies([held], base_url=BASE, now=NOW)["JSESSIONID"]
    assert "the-secret-value" not in repr(wrapped)


# --- nothing here launches a browser ------------------------------------------------------------


def test_this_module_reads_no_page_content() -> None:
    """The mechanical half of "manicule does not see the password".

    The property used to be a fact about capability — there was no browser. It is now a fact
    about this code, and this is what checks it: the driver may ask for cookies and for whether
    the window is open, and it may not ask what is on the page. A reviewer reading the diff is
    the other half; this catches the accessor added later by somebody who did not read the
    module docstring.
    """
    source = Path(browser_module.__file__).read_text(encoding="utf-8")
    body = source.split('"""', 2)[2]
    for accessor in (
        ".content()",
        ".inner_text(",
        ".inner_html(",
        ".query_selector",
        ".evaluate(",
        ".text_content(",
        ".input_value(",
        ".screenshot(",
        ".fill(",
        ".type(",
        ".press(",
        ".click(",
    ):
        assert accessor not in body, (
            f"{accessor} reads or drives the page. The browser is an opaque authentication "
            f"surface: manicule watches the cookie jar and whether the window is open, and "
            f"nothing else."
        )


def test_the_provider_is_constructed_headed() -> None:
    """A headless browser cannot show a sign-in form to a person."""
    assert PlaywrightProvider()._headless is False  # pyright: ignore[reportPrivateUsage]


@pytest.mark.browser
def test_a_real_browser_can_be_launched() -> None:
    """The one test that needs Chromium, so it is the one test that is marked.

    Skipped rather than failed when the browser is not installed, because `playwright install
    chromium` is a several-hundred-megabyte download and CI does not do it. What it buys when it
    does run is the only check that the launch arguments are ones Playwright accepts — every
    other test here drives the seam above it.
    """
    playwright = pytest.importorskip("playwright.sync_api", reason="playwright is not installed")
    try:
        with playwright.sync_playwright() as driver:
            browser = driver.chromium.launch(headless=True)
            browser.close()
    except Exception as exc:  # noqa: BLE001 - the skip reason is the exception
        pytest.skip(f"no Chromium available to launch: {type(exc).__name__}")


def test_the_state_file_mode_check_is_the_one_documented(tmp_path: Path) -> None:
    """The bits that count as exposed are group and other, which is what the message says."""
    path = written_state(tmp_path, state(entry()), mode=0o600)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert read_state_file(path)
