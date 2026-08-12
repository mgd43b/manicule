"""Authenticating a Server/Data Center instance as a signed-in browser session.

The failure this suite exists for is not "the credential was rejected". A rejected credential
announces itself with a 401 and stops the run. The failure is the one that **does not** announce
itself: an instance behind an identity provider answers a request it will not serve with a
sign-in page carrying status 200, and a client that took it at its word would index one copy of
that page for every page it tried to read — plausible documents, retrievable, citable, wrong,
and indistinguishable downstream from the corpus they replaced.

So the first tests here are not about sessions at all. They serve a complete, ordinary-looking
sign-in page with a successful status and assert that nothing is indexed, on the JSON path and
on the attachment path — the attachment path being the one where no JSON decoder stands between
that page and the index.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any, Final, cast
from unittest.mock import patch
from uuid import uuid4

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from manicule.connectors import (
    AuthMethod,
    ConfluenceConfig,
    Deployment,
    SessionExpiredError,
    UntrustedLinkError,
    resolve_credentials,
    sessions,
)
from manicule.connectors.client import ConfluenceClient
from manicule.connectors.credentials import (
    BrowserSession,
    BrowserSessionCredential,
    credential_for,
    token_credential,
)
from manicule.connectors.errors import ConnectorError
from manicule.connectors.intercept import Answer, signed_out
from manicule.connectors.sessions import (
    KeychainStore,
    MemoryStore,
    capture,
    load_session,
    parse_cookies,
)
from manicule.core.errors import ConfigError
from tests.connectors.fake_confluence import (
    IDENTITY_PROVIDER,
    SERVER_BASE,
    FakeAttachment,
    FakeConfluence,
    FakePage,
)
from tests.connectors.support import (
    CAPTURED_AT,
    SESSION_ACCOUNT,
    browser_session,
    cloud_config,
    connected,
    drain,
    server_config,
    sso_config,
)

REQUIRE_KEYCHAIN_ENV: Final = "REQUIRE_KEYCHAIN"
"""Set to any non-empty value to turn this suite's Keychain skips into failures. CI sets it.

Read at import, before a fixture has had the chance to touch the environment, and named
outside manicule's own ``MANICULE_`` namespace — which the test environment fixture empties
before each test, and which has already once disarmed a switch of exactly this kind.
"""

KEYCHAIN_REQUIRED: Final = bool(os.environ.get(REQUIRE_KEYCHAIN_ENV, "").strip())

NO_KEYCHAIN: Final = not KeychainStore.available() and not KEYCHAIN_REQUIRED
"""Whether to skip the cases that need a real Keychain.

The Keychain is macOS's, so on Linux these skip — and on the ubuntu matrix that is all of them,
which would leave the tests holding the most dangerous credential manicule stores running on
one developer's machine and nowhere else. The macOS job sets the variable, and then a missing
``/usr/bin/security`` is a failure that says so rather than a skip nobody reads.
"""


def _instance(**overrides: object) -> FakeConfluence:
    """A Server instance with one page and one attachment on it."""
    return FakeConfluence(
        base_url=SERVER_BASE,
        pages=[FakePage(id="1", title="Token Refresh", space="OPS")],
        attachments=[
            FakeAttachment(
                id="att-9",
                title="diagram.pdf",
                space="OPS",
                page_id="1",
                page_title="Token Refresh",
            )
        ],
        **overrides,  # type: ignore[arg-type]
    )


# --- a sign-in page is never content ---------------------------------------------------------


async def test_a_sign_in_page_returned_with_status_200_is_not_discovered_as_content() -> None:
    """The whole ticket, in one assertion.

    No redirect, no 401, no error status: the search endpoint answers 200 with HTML that is a
    complete sign-in page. Everything a client normally checks says this response is fine.
    """
    instance = _instance()
    instance.sign_out()
    config = sso_config(instance.base_url)
    connector = await connected(instance, config, credential=browser_session(config))
    try:
        with pytest.raises(SessionExpiredError, match="sign-in page"):
            await drain(connector.discover(None))
    finally:
        await connector.teardown()


async def test_a_sign_in_page_returned_for_an_attachment_is_not_downloaded_as_the_file() -> None:
    """The path with no backstop, and the reason the check is in the client rather than a parser.

    A sign-in page reaching a JSON endpoint fails to decode, so something catches it either way.
    A sign-in page reaching an attachment download is bytes with a media type: ``text/html``,
    which the parser chain parses perfectly well. Nothing further down can tell it from an
    attached HTML document, so it has to be refused here.
    """
    instance = _instance()
    config = sso_config(instance.base_url)
    connector = await connected(instance, config, credential=browser_session(config))
    try:
        found = await drain(connector.discover(None))
        attachment = next(item for item in found if item.source_id == "att-9")
        instance.sign_out("/download/")
        with pytest.raises(SessionExpiredError, match="sign-in page"):
            await connector.fetch(attachment.ref)
    finally:
        await connector.teardown()


async def test_a_sign_in_page_is_refused_for_a_personal_access_token_too() -> None:
    """A reverse proxy answers a token exactly as it answers a session.

    The check belongs to the response rather than to the credential. One that ran only for
    browser sessions would be a check that had never run on the deployment it was written for.
    """
    instance = _instance()
    instance.sign_out()
    connector = await connected(instance, server_config(instance.base_url))
    try:
        with pytest.raises(SessionExpiredError, match="sign-in page"):
            await drain(connector.discover(None))
    finally:
        await connector.teardown()


async def test_the_refusal_names_what_to_do_about_it() -> None:
    """A stopped sync is only useful if the message says which act resumes it."""
    instance = _instance()
    instance.sign_out()
    config = sso_config(instance.base_url)
    connector = await connected(instance, config, credential=browser_session(config))
    try:
        with pytest.raises(SessionExpiredError, match="manicule connector login"):
            await drain(connector.discover(None))
    finally:
        await connector.teardown()


# --- redirects, which is how the sign-in page usually arrives ---------------------------------


async def test_a_redirect_to_the_identity_provider_is_refused_and_never_followed() -> None:
    """Not following is the point.

    Following would fetch the provider's sign-in page and turn a 302 into a 200 with a body —
    the shape nothing downstream can catch — and it would offer this account's session cookies
    to a host that is not Confluence. Both are avoided by refusing at the redirect.

    The message is asserted, not only the exception type. The pre-existing origin check on every
    outbound URL would also stop this one, a step later, saying "refusing to request" — a true
    sentence about a link nobody wrote down, where what actually happened is that a response
    sent the sync somewhere. Two guards over one hazard is the intent; the second one being the
    only one with a legible account of it is not.
    """
    instance = _instance()
    instance.redirect(f"{IDENTITY_PROVIDER}/sso/saml?RelayState=x", "/rest/api/space")
    config = sso_config(instance.base_url)
    connector = await connected(instance, config, credential=browser_session(config))
    try:
        with pytest.raises(UntrustedLinkError, match=r"redirected the sync to.*idp\.example\.com"):
            await drain(connector.discover(None))
    finally:
        await connector.teardown()

    asked = {request.url.host for request in instance.requests}
    assert asked == {"wiki.example.com"}, "the identity provider must never have been contacted"


async def test_a_redirect_to_the_instances_own_sign_in_is_a_dead_session() -> None:
    """Same origin, so nothing leaves — but ``/login.action`` is still not an answer."""
    instance = _instance()
    instance.redirect("/login.action?os_destination=%2Frest%2Fapi%2Fspace", "/rest/api/space")
    config = sso_config(instance.base_url)
    connector = await connected(instance, config, credential=browser_session(config))
    try:
        with pytest.raises(SessionExpiredError, match=r"/login\.action"):
            await drain(connector.discover(None))
    finally:
        await connector.teardown()


async def test_an_ordinary_same_origin_redirect_is_followed() -> None:
    """Refusing every redirect would be a different bug: an instance may legitimately move a
    path, and a connector that could not follow one on its own origin would fail on a working
    site. Only off-origin and sign-in-bound redirects are refused."""
    instance = _instance()
    moved: dict[str, int] = {"hops": 0}
    handler = instance.handle

    def relocate(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/rest/api/space") and moved["hops"] == 0:
            moved["hops"] += 1
            return httpx.Response(302, headers={"location": f"{SERVER_BASE}/rest/api/space"})
        return handler(request)

    config = sso_config(instance.base_url)
    client = ConfluenceClient(
        config,
        credential=browser_session(config),
        transport=httpx.MockTransport(relocate),
        clock=lambda: 0.0,
    )
    await client.setup()
    try:
        payload = await client.get_json(client.url("/rest/api/space"))
    finally:
        await client.teardown()

    assert moved["hops"] == 1
    assert "results" in payload


async def test_a_redirect_that_never_arrives_anywhere_stops_rather_than_looping() -> None:
    """An instance insisting on a sign-in a cookie-only client cannot complete does this.

    How many requests it takes is asserted as well as the refusal, because "it stops
    eventually" is true of any ceiling at all, and what is being decided is how hard a looping
    instance gets hit before manicule gives up on it: six requests, not six hundred.

    Written as a number rather than read back from ``MAX_REDIRECTS``. A test that imports the
    constant it is checking moves whenever the constant does, and then asserts only that the
    code agrees with itself.
    """
    config = sso_config(SERVER_BASE)
    hops = 0

    def circle(request: httpx.Request) -> httpx.Response:
        nonlocal hops
        hops += 1
        del request
        return httpx.Response(302, headers={"location": f"{SERVER_BASE}/rest/api/space"})

    client = ConfluenceClient(
        config,
        credential=browser_session(config),
        transport=httpx.MockTransport(circle),
        clock=lambda: 0.0,
    )
    await client.setup()
    try:
        with pytest.raises(ConnectorError, match="redirected more than"):
            await client.get_json(client.url("/rest/api/space"))
    finally:
        await client.teardown()

    assert hops == 6


# --- what the instance says about who it thinks you are --------------------------------------


async def test_an_answer_from_an_anonymous_session_is_refused() -> None:
    """Confluence names the authenticated user on every REST response.

    ``anonymous`` with a 200 and a JSON body is the quietest version of this failure: the
    request reached Confluence, was served, and what came back is what a signed-out reader can
    see. Indexing it would be a corpus that silently lost everything the sync account could read.
    """
    instance = _instance()
    instance.headers["X-AUSERNAME"] = "anonymous"
    config = sso_config(instance.base_url)
    connector = await connected(instance, config, credential=browser_session(config))
    try:
        with pytest.raises(SessionExpiredError, match="anonymous"):
            await drain(connector.discover(None))
    finally:
        await connector.teardown()


async def test_an_answer_from_a_different_account_is_refused() -> None:
    """The index holds what the sync account can see, so which account it is, is correctness."""
    instance = _instance()
    instance.headers["X-AUSERNAME"] = "somebody.else"
    config = sso_config(instance.base_url)
    connector = await connected(instance, config, credential=browser_session(config))
    try:
        with pytest.raises(SessionExpiredError, match=r"somebody\.else"):
            await drain(connector.discover(None))
    finally:
        await connector.teardown()


async def test_seraphs_own_verdict_is_believed() -> None:
    """Confluence's authentication filter reports the outcome in a header, whatever the status."""
    instance = _instance()
    instance.headers["X-Seraph-LoginReason"] = "AUTHENTICATED_FAILED"
    config = sso_config(instance.base_url)
    connector = await connected(instance, config, credential=browser_session(config))
    try:
        with pytest.raises(SessionExpiredError, match="AUTHENTICATED_FAILED"):
            await drain(connector.discover(None))
    finally:
        await connector.teardown()


async def test_the_account_the_session_was_captured_as_is_accepted() -> None:
    """The mismatch check must not fire on the account that captured the session."""
    instance = _instance()
    instance.headers["X-AUSERNAME"] = SESSION_ACCOUNT
    config = sso_config(instance.base_url)
    connector = await connected(instance, config, credential=browser_session(config))
    try:
        found = await drain(connector.discover(None))
    finally:
        await connector.teardown()

    assert [item.source_id for item in found] == ["1", "att-9"]


async def test_a_cloud_sync_is_not_refused_for_being_a_different_account() -> None:
    """Cloud's Basic credential is spelled with an email; ``X-AUSERNAME`` is an account id.

    They are two identifiers for one person, and comparing them would make every Cloud response
    look as though it came back as somebody else — a working configuration refused on every
    request. Only a session captured *from* an instance knows a name that instance will repeat,
    so a token supplies no expected account at all.
    """
    instance = FakeConfluence(pages=[FakePage(id="1", title="A", space="ENG")])
    instance.headers["X-AUSERNAME"] = "5b10ac8d82e05b22cc7d4ef5"
    connector = await connected(instance, cloud_config(base_url=instance.base_url))
    try:
        found = await drain(connector.discover(None))
    finally:
        await connector.teardown()

    assert [item.source_id for item in found] == ["1"]


async def test_a_cloud_sync_is_still_refused_when_it_is_anonymous() -> None:
    """Dropping the account comparison for tokens must not drop the check that matters."""
    instance = FakeConfluence(pages=[FakePage(id="1", title="A", space="ENG")])
    instance.headers["X-AUSERNAME"] = "anonymous"
    connector = await connected(instance, cloud_config(base_url=instance.base_url))
    try:
        with pytest.raises(SessionExpiredError, match="anonymous"):
            await drain(connector.discover(None))
    finally:
        await connector.teardown()


async def test_a_sign_in_page_split_across_chunks_is_still_refused() -> None:
    """The marker is looked for in the accumulated opening, not in one chunk.

    A chunk boundary is wherever the network put it. A check that read only the first chunk
    would miss a marker straddling two, and what got through would be a sign-in page indexed as
    an attachment.
    """
    config = sso_config(SERVER_BASE)
    page = (
        b'<html><head><title>Log in</title></head><body><form name="loginform">'
        b'<input name="os_username"></form></body></html>'
    )

    async def dribble() -> AsyncIterator[bytes]:
        for index in range(0, len(page), 7):
            yield page[index : index + 7]

    def serve(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200, content=dribble(), headers={"content-type": "text/html;charset=UTF-8"}
        )

    client = ConfluenceClient(
        config,
        credential=browser_session(config),
        transport=httpx.MockTransport(serve),
        clock=lambda: 0.0,
    )
    await client.setup()
    try:
        with pytest.raises(SessionExpiredError, match="sign-in page"):
            await client.download(f"{SERVER_BASE}/download/attachments/1/x.pdf", max_bytes=99999)
    finally:
        await client.teardown()


def test_a_percent_encoded_username_is_compared_decoded() -> None:
    """Atlassian percent-encodes the header, and a comparison that did not decode it would
    report every account with a space or an accent in its name as somebody else."""
    reason = signed_out(
        Answer("https://wiki.example.com/x", 200, {"x-ausername": "ann%20lee"}),
        expected_account="ann lee",
    )
    assert reason is None


# --- the credential, consulted per request ----------------------------------------------------


async def test_a_browser_session_authenticates_with_its_cookies() -> None:
    """The existing modes send Authorization; this one sends Cookie, and neither sends both."""
    instance = _instance()
    config = sso_config(instance.base_url)
    connector = await connected(instance, config, credential=browser_session(config))
    try:
        await drain(connector.discover(None))
    finally:
        await connector.teardown()

    sent = instance.requests[0].headers
    assert sent["cookie"] == "JSESSIONID=ABC123; seraph.confluence=77"
    assert "authorization" not in sent


async def test_a_session_that_ages_out_mid_sync_stops_the_run() -> None:
    """The reason the credential is asked for per request rather than built once.

    A first sync of a large corpus outlives sessions. A credential baked into the client at
    setup would keep being sent after manicule had decided it was too old — and what comes back
    then is a sign-in page.
    """
    instance = FakeConfluence(
        base_url=SERVER_BASE,
        pages=[FakePage(id=str(index), title=f"P{index}", space="OPS") for index in range(1, 7)],
        page_size=2,
    )
    config = sso_config(instance.base_url, session_max_age_hours=1.0)
    moment = {"now": CAPTURED_AT}
    session = BrowserSession(
        base_url=config.base_url,
        account=SESSION_ACCOUNT,
        captured_at=CAPTURED_AT,
        cookies={"JSESSIONID": SecretStr("ABC123")},
    )
    credential = BrowserSessionCredential(
        session=session, max_age=timedelta(hours=1), now=lambda: moment["now"]
    )
    connector = await connected(instance, config, credential=credential)
    seen: list[str] = []

    async def walk() -> None:
        async for found in connector.discover(None):
            seen.append(found.source_id)
            moment["now"] = CAPTURED_AT + timedelta(hours=2)

    try:
        with pytest.raises(SessionExpiredError, match="session_max_age_hours"):
            await walk()
    finally:
        await connector.teardown()

    assert seen, "the run had made progress before the session aged out"
    assert connector.watermark is None, "an interrupted enumeration must not advance a watermark"


def test_a_session_older_than_the_ceiling_is_refused_before_the_connector_is_built() -> None:
    """A dead session should cost a startup message, not a run that reports progress."""
    config = sso_config(SERVER_BASE, session_max_age_hours=2.0)
    store = MemoryStore()
    store.save(
        BrowserSession(
            base_url=config.base_url,
            account=SESSION_ACCOUNT,
            captured_at=CAPTURED_AT,
            cookies={"JSESSIONID": SecretStr("ABC123")},
        )
    )
    with pytest.raises(SessionExpiredError, match="session_max_age_hours"):
        credential_for(
            config,
            environ={},
            store=store,
            now=lambda: CAPTURED_AT + timedelta(hours=3),
        )


def test_no_stored_session_is_a_startup_refusal_naming_the_command() -> None:
    config = sso_config(SERVER_BASE)
    with pytest.raises(ConfigError, match="manicule connector login"):
        credential_for(config, environ={}, store=MemoryStore())


def test_a_browser_session_configuration_needs_no_token() -> None:
    """``resolve_credentials`` refuses a Server configuration with no personal access token.

    It must not refuse one that authenticates a different way, or browser SSO would be
    unreachable behind the check that exists to catch a missing token.
    """
    config = sso_config(SERVER_BASE)
    assert resolve_credentials(config, {}) == config


# --- the modes that already worked -------------------------------------------------------------


async def test_cloud_still_authenticates_as_email_and_token() -> None:
    """Moved from a baked header to a per-request credential; the wire must not have changed."""
    instance = FakeConfluence(pages=[FakePage(id="1", title="A", space="ENG")])
    connector = await connected(instance, cloud_config(base_url=instance.base_url))
    try:
        await drain(connector.discover(None))
    finally:
        await connector.teardown()

    sent = instance.requests[0].headers
    assert sent["authorization"].startswith("Basic ")
    assert "cookie" not in sent


async def test_server_still_authenticates_with_a_bearer_token() -> None:
    instance = _instance()
    connector = await connected(instance, server_config(instance.base_url))
    try:
        await drain(connector.discover(None))
    finally:
        await connector.teardown()

    sent = instance.requests[0].headers
    assert sent["authorization"] == "Bearer pat"
    assert "cookie" not in sent


def test_the_default_credential_follows_the_deployment() -> None:
    """An existing configuration that never named an auth method keeps the one it had."""
    assert cloud_config().auth_method is AuthMethod.API_TOKEN
    assert server_config(SERVER_BASE).auth_method is AuthMethod.PERSONAL_ACCESS_TOKEN


# --- configuration is not where a session lives ------------------------------------------------


def test_a_session_cookie_written_into_configuration_is_refused() -> None:
    """A session cookie is the sync account's whole identity at that company, and a
    configuration file is a file that reaches version control eventually. There is no field for
    one, which makes writing one there an error rather than a working setting."""
    with pytest.raises(ValidationError, match="session_cookie"):
        ConfluenceConfig.model_validate(
            {
                "base_url": SERVER_BASE,
                "deployment": Deployment.SERVER,
                "auth": AuthMethod.BROWSER_SESSION,
                "session_cookie": "JSESSIONID=ABC123",
            }
        )


def test_there_is_nowhere_to_put_a_password() -> None:
    """manicule never sees the password. Stated as a property of the model rather than as a
    promise about the code: there is no field that could hold one."""
    named = set(ConfluenceConfig.model_fields)
    assert not {field for field in named if "password" in field or "secret" in field}


def test_a_browser_session_on_cloud_is_refused() -> None:
    """Cloud sessions are held by Atlassian's own domains and are not a credential a REST
    client carries. Attempting it would authenticate as nobody and index what an anonymous
    reader can see, which is a sync that succeeds and is wrong."""
    with pytest.raises(ValidationError, match="browser_session"):
        ConfluenceConfig.model_validate(
            {
                "base_url": "https://example.atlassian.net/wiki",
                "deployment": Deployment.CLOUD,
                "auth": AuthMethod.BROWSER_SESSION,
            }
        )


def test_an_api_token_on_server_is_refused() -> None:
    """Server and Data Center have no API tokens; naming one is a configuration that cannot
    work, and saying so beats a 401 on the first request."""
    with pytest.raises(ValidationError, match="no API tokens"):
        ConfluenceConfig.model_validate(
            {
                "base_url": SERVER_BASE,
                "deployment": Deployment.SERVER,
                "auth": AuthMethod.API_TOKEN,
            }
        )


def test_a_personal_access_token_on_cloud_is_refused() -> None:
    """The mirror of the row above. Without it the refusal arrives from the credential builder
    as "no personal_access_token is set", which is true and describes the wrong problem: the
    setting to change is the deployment or the method, not the missing token."""
    with pytest.raises(ValidationError, match="issues API tokens instead"):
        ConfluenceConfig.model_validate(
            {
                "base_url": "https://example.atlassian.net/wiki",
                "deployment": Deployment.CLOUD,
                "auth": AuthMethod.PERSONAL_ACCESS_TOKEN,
            }
        )


def test_a_browser_session_cannot_be_derived_from_configuration() -> None:
    """The client's own default credential covers the token modes and refuses this one, so a
    caller that skipped the factory gets a message rather than a connector that sends nothing."""
    with pytest.raises(ConfigError, match="keychain"):
        token_credential(sso_config(SERVER_BASE))


# --- capturing a session ------------------------------------------------------------------------


def test_the_cookies_a_browser_hands_over_are_all_accepted() -> None:
    """Three spellings, because all three are what somebody will paste."""
    expected = {"JSESSIONID": "ABC", "seraph.confluence": "77"}
    for pasted in (
        "JSESSIONID=ABC; seraph.confluence=77",
        "Cookie: JSESSIONID=ABC; seraph.confluence=77",
        "JSESSIONID=ABC\nseraph.confluence=77",
    ):
        parsed = parse_cookies(pasted)
        assert {name: value.get_secret_value() for name, value in parsed.items()} == expected


def test_something_that_is_not_a_cookie_is_refused_by_saying_what_to_paste() -> None:
    """Overwhelmingly this is a password, and the message has to be about that."""
    with pytest.raises(ConfigError, match="never asks for a password"):
        parse_cookies("hunter2")


async def test_capturing_a_session_proves_it_works_before_storing_it() -> None:
    instance = _instance()
    config = sso_config(instance.base_url)
    store = MemoryStore()
    session = await capture(
        config,
        "JSESSIONID=ABC123",
        store=store,
        transport=instance.transport(),
        now=CAPTURED_AT,
    )

    assert session.account == SESSION_ACCOUNT
    assert store.load(config.base_url) is not None


async def test_a_session_that_does_not_work_is_not_stored() -> None:
    """Storing first and checking later would leave a dead credential behind on every failure,
    and the next sync would find it and use it."""
    instance = _instance()
    instance.sign_out()
    config = sso_config(instance.base_url)
    store = MemoryStore()
    with pytest.raises(SessionExpiredError):
        await capture(
            config,
            "JSESSIONID=ABC123",
            store=store,
            transport=instance.transport(),
            now=CAPTURED_AT,
        )

    assert store.load(config.base_url) is None


def test_a_stored_session_survives_being_written_and_read_back() -> None:
    """The keychain holds one string, so the record has to make the round trip through one."""
    session = BrowserSession(
        base_url=SERVER_BASE,
        account=SESSION_ACCOUNT,
        captured_at=CAPTURED_AT,
        cookies={"JSESSIONID": SecretStr("ABC=123+/"), "other": SecretStr("x;y")},
    )
    restored = BrowserSession.from_json(session.to_json())

    assert restored.base_url == session.base_url
    assert restored.account == session.account
    assert restored.captured_at == session.captured_at
    assert {name: value.get_secret_value() for name, value in restored.cookies.items()} == {
        "JSESSIONID": "ABC=123+/",
        "other": "x;y",
    }


def test_the_environment_carries_a_session_where_there_is_no_keychain() -> None:
    config = sso_config(SERVER_BASE)
    session = load_session(
        config,
        environ={config.session_env: "JSESSIONID=FROM-ENV"},
        store=MemoryStore(),
        now=CAPTURED_AT,
    )

    assert session is not None
    assert session.cookies["JSESSIONID"].get_secret_value() == "FROM-ENV"


def test_a_captured_session_wins_over_a_stale_environment_variable() -> None:
    """Otherwise ``manicule connector login`` would appear not to have worked."""
    config = sso_config(SERVER_BASE)
    store = MemoryStore()
    store.save(
        BrowserSession(
            base_url=config.base_url,
            account=SESSION_ACCOUNT,
            captured_at=CAPTURED_AT,
            cookies={"JSESSIONID": SecretStr("FROM-KEYCHAIN")},
        )
    )
    session = load_session(config, environ={config.session_env: "JSESSIONID=FROM-ENV"}, store=store)

    assert session is not None
    assert session.cookies["JSESSIONID"].get_secret_value() == "FROM-KEYCHAIN"


def test_a_session_never_shows_its_cookies_in_a_repr() -> None:
    """Every message this connector raises names a URL, and several name the credential's
    description. None of them may name the credential."""
    credential = browser_session(sso_config(SERVER_BASE), cookies={"JSESSIONID": "ABC123"})

    assert "ABC123" not in repr(credential)
    assert "ABC123" not in credential.describe()
    assert "ABC123" not in credential.renewal()


def test_a_session_the_keychain_cannot_parse_says_how_to_replace_it() -> None:
    with pytest.raises(ValueError, match="carries no cookies"):
        BrowserSession.from_json('{"base_url": "x", "captured_at": "2026-08-12T09:00:00+00:00"}')


@pytest.mark.skipif(NO_KEYCHAIN, reason="the Keychain is macOS's")
def test_the_keychain_really_holds_a_session_and_gives_it_back() -> None:
    """Against the real ``/usr/bin/security``, because that is the decision being tested.

    A mocked subprocess would prove that this module builds the argument list it was written to
    build. What is actually in question is whether a session survives the Keychain's own
    encoding, whether the item can be read back without a dialog on a machine running a sync
    unattended, and whether cookies containing ``=``, ``+`` and ``;`` come back as they went in.
    None of that is answerable without the Keychain.
    """
    store = KeychainStore(f"manicule test {uuid4()}")
    session = BrowserSession(
        base_url=SERVER_BASE,
        account=SESSION_ACCOUNT,
        captured_at=CAPTURED_AT,
        cookies={"JSESSIONID": SecretStr("A+b/c=="), "seraph.confluence": SecretStr("x;y")},
    )
    try:
        assert store.load(SERVER_BASE) is None
        store.save(session)
        restored = store.load(SERVER_BASE)

        assert restored is not None
        assert restored.account == SESSION_ACCOUNT
        assert restored.captured_at == CAPTURED_AT
        assert {name: value.get_secret_value() for name, value in restored.cookies.items()} == {
            "JSESSIONID": "A+b/c==",
            "seraph.confluence": "x;y",
        }
        assert store.forget(SERVER_BASE) is True
        assert store.load(SERVER_BASE) is None
        assert store.forget(SERVER_BASE) is False
    finally:
        store.forget(SERVER_BASE)


@pytest.mark.skipif(NO_KEYCHAIN, reason="the Keychain is macOS's")
def test_a_session_larger_than_the_keychains_stdin_buffer_survives() -> None:
    """The defect a real Keychain found, kept caught.

    ``/usr/bin/security`` reads a secret from stdin through a 128-byte buffer and keeps the
    first 128 bytes of anything longer, reporting success either way. An ordinary session record
    is already past that, and an instance behind single sign-on issues cookies of its own
    besides Confluence's — this one is about two kilobytes, which is not unusual for SAML. A
    truncated record does not fail to store; it stores, comes back as a broken credential, and
    authenticates as nobody.
    """
    store = KeychainStore(f"manicule test {uuid4()}")
    session = BrowserSession(
        base_url=SERVER_BASE,
        account=SESSION_ACCOUNT,
        captured_at=CAPTURED_AT,
        cookies={
            "JSESSIONID": SecretStr("A" * 32),
            "seraph.confluence": SecretStr("B" * 40),
            "MSISAuth": SecretStr("C" * 1800),
        },
    )
    try:
        store.save(session)
        restored = store.load(SERVER_BASE)

        assert restored is not None
        assert restored.cookies["MSISAuth"].get_secret_value() == "C" * 1800
    finally:
        store.forget(SERVER_BASE)


@pytest.mark.skipif(NO_KEYCHAIN, reason="the Keychain is macOS's")
def test_a_keychain_that_gives_back_something_else_stores_nothing() -> None:
    """The read-back is what turns a *different* buffer limit into a loud failure.

    Chunking works around the limit that exists today. This is what happens when the assumption
    behind the chunk size stops holding: nothing is stored, and the message says so — rather
    than a credential that is silently half of one.
    """
    store = KeychainStore(f"manicule test {uuid4()}")
    session = BrowserSession(
        base_url=SERVER_BASE,
        account=SESSION_ACCOUNT,
        captured_at=CAPTURED_AT,
        cookies={"JSESSIONID": SecretStr("D" * 400)},
    )
    try:
        with (
            patch.object(sessions, "CHUNK_BYTES", 400),
            pytest.raises(ConfigError, match="nothing has been stored"),
        ):
            store.save(session)

        assert store.load(SERVER_BASE) is None
    finally:
        store.forget(SERVER_BASE)


@pytest.mark.skipif(NO_KEYCHAIN, reason="the Keychain is macOS's")
def test_a_session_is_filed_under_the_site_it_came_from() -> None:
    """A session is not portable between instances, and one offered to the wrong site would
    authenticate as nobody there while looking configured."""
    store = KeychainStore(f"manicule test {uuid4()}")
    session = BrowserSession(
        base_url=SERVER_BASE,
        account=SESSION_ACCOUNT,
        captured_at=CAPTURED_AT,
        cookies={"JSESSIONID": SecretStr("ABC")},
    )
    try:
        store.save(session)

        assert store.load("https://other.example.com") is None
        assert store.load(f"{SERVER_BASE}/") is not None, "a trailing slash is the same site"
    finally:
        store.forget(SERVER_BASE)


@pytest.mark.skipif(NO_KEYCHAIN, reason="the Keychain is macOS's")
def test_the_session_never_reaches_the_command_line() -> None:
    """``security`` reads it from stdin. Passing it as an argument would put a live corporate
    session in this process's command line, where anything on the machine can read it."""
    recorded: list[list[str]] = []
    store = KeychainStore(f"manicule test {uuid4()}")
    real = subprocess.run

    def watched(arguments: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        recorded.append(list(arguments))
        return cast(
            "subprocess.CompletedProcess[str]",
            real(arguments, **kwargs),
        )

    session = BrowserSession(
        base_url=SERVER_BASE,
        account=SESSION_ACCOUNT,
        captured_at=CAPTURED_AT,
        cookies={"JSESSIONID": SecretStr("TOP-SECRET-SESSION")},
    )
    try:
        with patch.object(subprocess, "run", watched):
            store.save(session)
    finally:
        store.forget(SERVER_BASE)

    assert recorded, "the keychain command ran"
    flattened = " ".join(" ".join(call) for call in recorded)
    assert "TOP-SECRET-SESSION" not in flattened
    assert "-A" not in flattened, "any application reading it silently is not the grant we want"


def test_a_session_stored_without_a_timezone_is_read_as_utc() -> None:
    """A record written by an older version, or edited by hand, must not compare naively."""
    restored = BrowserSession.from_json(
        '{"captured_at": "2026-08-12T09:00:00", "cookies": {"JSESSIONID": "x"}}'
    )

    assert restored.captured_at == datetime(2026, 8, 12, 9, 0, tzinfo=UTC)
