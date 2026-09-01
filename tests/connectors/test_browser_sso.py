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

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timedelta

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
    SessionVault,
    capture,
    load_session,
    parse_cookies,
)
from manicule.core.errors import ConfigError
from manicule.core.lifecycle import HealthState
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
    store = SessionVault()
    asyncio.run(
        store.save(
            BrowserSession(
                base_url=config.base_url,
                account=SESSION_ACCOUNT,
                captured_at=CAPTURED_AT,
                cookies={"JSESSIONID": SecretStr("ABC123")},
            )
        )
    )
    with pytest.raises(SessionExpiredError, match="session_max_age_hours"):
        credential_for(
            config,
            store=store,
            now=lambda: CAPTURED_AT + timedelta(hours=3),
        )


def test_no_stored_session_is_a_startup_refusal_naming_the_command() -> None:
    config = sso_config(SERVER_BASE)
    with pytest.raises(ConfigError, match="manicule connector login"):
        credential_for(config, store=SessionVault())


def test_a_browser_session_configuration_needs_no_token() -> None:
    """``resolve_credentials`` refuses a Server configuration with no personal access token.

    It must not refuse one that authenticates a different way, or browser SSO would be
    unreachable behind the check that exists to catch a missing token.
    """
    config = sso_config(SERVER_BASE)
    assert resolve_credentials(config, {}) == config


# --- a session replaced while the connector reading it is alive --------------------------------
#
# A connector is built once and cached for the life of the process, because it carries a watermark
# across a run. The session it authenticates with has no such lifetime: it is replaced from
# outside, by somebody signing in again and the running server being handed the result. These are
# the tests that the credential resolves the second fact instead of closing over the first — which
# it used to, so a sign-in reported success and every later sync went on failing against the
# session it had replaced, until the server was restarted.

REPLACEMENT = "DEF456"
"""The cookie of the session a sign-in hands over, distinct from every fixture's ``ABC123``."""


def _held(
    config: ConfluenceConfig,
    value: str,
    *,
    account: str = SESSION_ACCOUNT,
    captured_at: datetime | None = None,
) -> BrowserSession:
    """One captured session for ``config``'s instance, as a hand-over would have left it."""
    return BrowserSession(
        base_url=config.base_url,
        account=account,
        captured_at=captured_at if captured_at is not None else CAPTURED_AT,
        cookies={"JSESSIONID": SecretStr(value)},
    )


async def test_a_session_replaced_after_the_credential_was_built_is_the_one_it_uses() -> None:
    """The defect in one function call.

    `credential_for` runs inside the connector plugin factory, once, before the connector is
    constructed — and the connector is then cached for the life of the process. A credential
    that kept the session it found there is a credential that can never see a renewal.
    """
    config = sso_config(SERVER_BASE)
    store = SessionVault()
    await store.save(_held(config, "ABC123"))
    credential = credential_for(config, store=store, now=lambda: CAPTURED_AT)

    await store.save(_held(config, REPLACEMENT, account="renewed.user"))
    authorization = credential.authorize()

    assert authorization.headers["Cookie"] == f"JSESSIONID={REPLACEMENT}"
    assert authorization.account == "renewed.user", (
        "the cookies were renewed but the account the reply is checked against was not"
    )


async def test_the_age_is_measured_from_the_capture_that_replaced_the_old_one() -> None:
    """Which is the whole point of renewing rather than reporting.

    An aged-out session and a replaced one used to be the same object, so a credential that had
    started refusing went on refusing after the sign-in that fixed it — repeating a message about
    a capture time that was no longer the current one. Here the same credential object goes from
    refusing to working, and nothing about it changed except what the store holds.
    """
    config = sso_config(SERVER_BASE, session_max_age_hours=2.0)
    store = SessionVault()
    await store.save(_held(config, "ABC123"))
    moment = CAPTURED_AT
    credential = credential_for(config, store=store, now=lambda: moment)

    moment = CAPTURED_AT + timedelta(hours=3)
    with pytest.raises(SessionExpiredError, match="session_max_age_hours") as aged:
        credential.authorize()
    await store.save(_held(config, REPLACEMENT, captured_at=moment))

    assert "ABC123" not in str(aged.value), "the refusal quoted the session it refused"
    assert credential.authorize().headers["Cookie"] == f"JSESSIONID={REPLACEMENT}"


async def test_forgetting_a_session_stops_a_credential_that_was_already_using_it() -> None:
    """`--forget` has to mean it. A cached copy would keep authenticating with what was dropped.

    The refusal is the one a connector built before anybody signed in gets, deliberately: the
    fact is the same — this process holds no session for that instance — and an operator should
    not have to work out that manicule described it two ways.
    """
    config = sso_config(SERVER_BASE)
    store = SessionVault()
    await store.save(_held(config, "ABC123"))
    credential = credential_for(config, store=store, now=lambda: CAPTURED_AT)

    assert await store.forget(config.base_url)

    with pytest.raises(ConfigError, match="manicule connector login") as gone:
        credential.authorize()
    assert "ABC123" not in str(gone.value)
    assert "ABC123" not in credential.describe()
    assert "no longer holding" in credential.describe()


async def test_a_hand_over_between_a_request_and_its_reply_is_not_read_as_another_account() -> None:
    """The window reading the store per request opens, and why it is closed by hand.

    `authorize()` builds the request; the account it names is what the reply is checked against.
    Those are two moments with an `await` between them, and a hand-over can land in it — which is
    not a rare race but the expected one, because somebody signs in again *precisely* when a sync
    is failing. A client that asked the credential a second time for the check would compare a
    reply against a session that never sent it, and report the session as expired at the moment
    it had just been renewed.

    The second request is here so the fix cannot be "ignore the store after the first reading":
    renewal has to take effect at the next request boundary, and this asserts it does.
    """
    config = sso_config(SERVER_BASE)
    store = SessionVault()
    await store.save(_held(config, "ABC123"))
    sent: list[str] = []

    async def answer(request: httpx.Request) -> httpx.Response:
        sent.append(request.headers.get("cookie", ""))
        if len(sent) == 1:
            await store.save(_held(config, REPLACEMENT, account="renewed.user"))
            return httpx.Response(
                200, json={"results": []}, headers={"X-AUSERNAME": SESSION_ACCOUNT}
            )
        return httpx.Response(200, json={"results": []}, headers={"X-AUSERNAME": "renewed.user"})

    client = ConfluenceClient(
        config,
        credential=credential_for(config, store=store, now=lambda: CAPTURED_AT),
        transport=httpx.MockTransport(answer),
    )
    await client.setup()
    try:
        await client.get_json(client.url("/rest/api/space"))
        await client.get_json(client.url("/rest/api/space"))
    finally:
        await client.teardown()

    assert sent == ["JSESSIONID=ABC123", f"JSESSIONID={REPLACEMENT}"]


async def test_a_hand_over_arriving_while_a_request_is_open_does_not_disturb_it() -> None:
    """The same window, opened by a task that genuinely overlaps rather than by the transport.

    The test above lands the hand-over between two lines of one coroutine, which is the boundary
    stated exactly. This one runs it as a separate task while a request is really open — the
    client suspended on a socket, another coroutine replacing the session under it — and gates
    the two on events so the ordering is arranged rather than raced. Between them they cover
    both readings of "in flight", and the second is the one a scheduled sync and a
    ``connector login`` running at the same time actually produce.

    The request that was open must be answered as the account it went out as, and the request
    after it must go out as the new one. A fix that took either half alone would pass one of
    these two tests and fail the other.
    """
    config = sso_config(SERVER_BASE)
    store = SessionVault()
    await store.save(_held(config, "ABC123"))
    accounts = {"ABC123": SESSION_ACCOUNT, REPLACEMENT: "renewed.user"}
    opened, handed_over = asyncio.Event(), asyncio.Event()
    sent: list[str] = []

    async def answer(request: httpx.Request) -> httpx.Response:
        cookie = request.headers.get("cookie", "")
        sent.append(cookie)
        if len(sent) == 1:
            opened.set()
            await handed_over.wait()
        return httpx.Response(
            200,
            json={"results": []},
            headers={"X-AUSERNAME": accounts[cookie.removeprefix("JSESSIONID=")]},
        )

    async def sign_in_again() -> None:
        await opened.wait()
        await store.save(_held(config, REPLACEMENT, account="renewed.user"))
        handed_over.set()

    client = ConfluenceClient(
        config,
        credential=credential_for(config, store=store, now=lambda: CAPTURED_AT),
        transport=httpx.MockTransport(answer),
    )
    await client.setup()
    try:
        await asyncio.gather(client.get_json(client.url("/rest/api/space")), sign_in_again())
        await client.get_json(client.url("/rest/api/space"))
    finally:
        await client.teardown()

    assert sent == ["JSESSIONID=ABC123", f"JSESSIONID={REPLACEMENT}"]


async def test_a_forgotten_session_is_a_health_report_rather_than_a_raised_exception() -> None:
    """``doctor`` has to say what happened, and a raised exception is not a diagnosis.

    ``SessionMissingError`` is a :class:`~manicule.core.errors.ConfigError` and not a
    :class:`~manicule.connectors.errors.ConnectorError`, deliberately — the scheduler tells the
    two refusals apart by type. That means ``health()``'s ordinary clause does not catch it, and
    it only became reachable there at all once the credential started reading the vault per
    request: a session forgotten while this connector is alive now stops its next request.

    Left alone, the container's health sweep would report ``health check raised: ...`` with no
    remedy — for the one state whose remedy is a single command. So the connector answers it
    itself, and says the instance was **not contacted**, because "did not answer" would send an
    operator to look at a wiki that is perfectly fine.
    """
    instance = _instance()
    config = sso_config(instance.base_url)
    store = SessionVault()
    await store.save(_held(config, "ABC123"))
    connector = await connected(
        instance, config, credential=credential_for(config, store=store, now=lambda: CAPTURED_AT)
    )
    try:
        assert "was not contacted" not in (await connector.health()).detail
        assert await store.forget(config.base_url)
        report = await connector.health()
    finally:
        await connector.teardown()

    assert report.state is HealthState.DEGRADED
    assert "was not contacted" in report.detail, report.detail
    assert "manicule connector login" in report.remedy, report.remedy
    assert "ABC123" not in report.detail + report.remedy


async def test_a_token_credential_reads_nothing_from_the_vault() -> None:
    """A personal access token is configuration, and no sign-in may reach it.

    Stated as a test because the renewable credential is chosen by `auth_method`, and a mistake
    there would send a token-authenticated connector to a store that has a session in it for the
    same instance — authenticating a sync as whoever last signed in, rather than as the account
    the operator configured.
    """
    config = server_config(SERVER_BASE)
    store = SessionVault()
    await store.save(_held(config, REPLACEMENT, account="somebody.else"))

    credential = credential_for(config, store=store)

    assert credential.authorize().headers == {"Authorization": "Bearer pat"}
    assert credential.authorize().account == ""


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
    with pytest.raises(ConfigError, match="the running server's memory"):
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
    store = SessionVault()
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
    store = SessionVault()
    with pytest.raises(SessionExpiredError):
        await capture(
            config,
            "JSESSIONID=ABC123",
            store=store,
            transport=instance.transport(),
            now=CAPTURED_AT,
        )

    assert store.load(config.base_url) is None


def test_a_session_is_held_in_memory_and_given_back() -> None:
    """The whole credential store, and the whole of what it has to do."""
    vault = SessionVault()
    session = BrowserSession(
        base_url=SERVER_BASE,
        account=SESSION_ACCOUNT,
        captured_at=CAPTURED_AT,
        cookies={"JSESSIONID": SecretStr("ABC=123+/"), "other": SecretStr("x;y")},
    )

    asyncio.run(vault.save(session))
    held = vault.load(SERVER_BASE)

    assert held is not None
    assert {name: value.get_secret_value() for name, value in held.cookies.items()} == {
        "JSESSIONID": "ABC=123+/",
        "other": "x;y",
    }
    assert held.captured_at == CAPTURED_AT, "the capture time is what the age ceiling measures"


def test_a_session_is_filed_under_the_site_it_came_from() -> None:
    """A session is not portable between instances, and this is what stops one being offered
    to another. Trailing slashes are one site rather than two, one of which would be found."""
    vault = SessionVault()
    asyncio.run(
        vault.save(
            BrowserSession(
                base_url="https://wiki.example.test/",
                account=SESSION_ACCOUNT,
                captured_at=CAPTURED_AT,
                cookies={"JSESSIONID": SecretStr("ABC123")},
            )
        )
    )

    assert vault.load("https://wiki.example.test") is not None
    assert vault.load("https://other.example.test") is None


def test_forgetting_a_session_says_whether_there_was_one() -> None:
    vault = SessionVault()
    asyncio.run(
        vault.save(
            BrowserSession(
                base_url=SERVER_BASE,
                account=SESSION_ACCOUNT,
                captured_at=CAPTURED_AT,
                cookies={"JSESSIONID": SecretStr("ABC123")},
            )
        )
    )

    assert asyncio.run(vault.forget(SERVER_BASE)) is True
    assert vault.load(SERVER_BASE) is None
    assert asyncio.run(vault.forget(SERVER_BASE)) is False


def test_the_vault_offers_no_way_to_read_a_session_it_was_not_asked_for() -> None:
    """A diagnostic wants to know *whether* there is a session, never what it is.

    ``len`` and ``holding`` are the whole of the introspection, deliberately, and the second was
    added rather than assumed: this is the only place a live corporate credential exists in the
    running system, so an accessor that handed back the collection would be the one line a future
    report, dump or ``doctor`` check would reach for.

    ``holding`` reports the instance and the account and stops there — the two fields a hand-off
    is already acknowledged with. The assertion below is over its **whole rendering**, not over
    the fields somebody remembered to look at, so an implementation that returned the sessions
    themselves fails here rather than at the surface that eventually printed them.
    """
    vault = SessionVault()
    asyncio.run(
        vault.save(
            BrowserSession(
                base_url=SERVER_BASE,
                account=SESSION_ACCOUNT,
                captured_at=CAPTURED_AT,
                cookies={"JSESSIONID": SecretStr("TOP-SECRET-SESSION")},
            )
        )
    )

    assert len(vault) == 1
    public = {name for name in dir(vault) if not name.startswith("_")}
    assert public == {"describe", "forget", "holding", "load", "save"}, (
        f"the vault grew a public member: {sorted(public)}. Every one of them is a route to a "
        f"live session, so a new one is a decision rather than a convenience."
    )
    assert "TOP-SECRET-SESSION" not in repr(vault)
    assert "TOP-SECRET-SESSION" not in vault.describe()
    assert "TOP-SECRET-SESSION" not in repr(vault.holding())
    assert list(vault.holding().values()) == [SESSION_ACCOUNT], (
        "holding() reports the account and nothing that could be unwrapped into a credential"
    )


def test_a_session_is_gone_when_the_process_that_held_it_is() -> None:
    """The lifetime, stated as a test because it is the design rather than a limitation.

    A fresh vault is what a restarted server has. There is no file to read back, no keychain to
    consult and no environment variable to fall back to — which is why the refusal names
    ``connector login`` rather than reporting a fault.
    """
    first = SessionVault()
    asyncio.run(
        first.save(
            BrowserSession(
                base_url=SERVER_BASE,
                account=SESSION_ACCOUNT,
                captured_at=CAPTURED_AT,
                cookies={"JSESSIONID": SecretStr("ABC123")},
            )
        )
    )

    restarted = SessionVault()

    assert first.load(SERVER_BASE) is not None
    assert restarted.load(SERVER_BASE) is None
    assert load_session(sso_config(SERVER_BASE), store=restarted) is None


def test_there_is_no_second_place_a_session_can_come_from() -> None:
    """One mechanism, not three — asserted over what the module *does*, not what it says.

    There were three: the macOS Keychain, a file, and an environment variable. Each was a place
    a live corporate credential outlived the process that captured it, and the keychain was also
    what prompted the operator for their password on every write.

    **Asserted over the syntax tree rather than over the source text**, because the module's own
    docstring explains at length what was removed and why — so a check that grepped for
    ``keychain`` would fail on the paragraph that documents its absence, and the obvious repair
    would be to stop writing the paragraph. What is actually forbidden is a *call*: reading the
    environment, opening a file, or running a subprocess. Those are nodes.
    """
    import ast  # noqa: PLC0415 - only this assertion reads a syntax tree
    import inspect  # noqa: PLC0415

    from manicule.connectors import sessions as module  # noqa: PLC0415

    tree = ast.parse(inspect.getsource(module))
    called = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    read = {
        ast.unparse(node) for node in ast.walk(tree) if isinstance(node, ast.Attribute | ast.Name)
    }

    forbidden = {name for name in called if name in {"open", "subprocess.run", "Path"}}
    assert forbidden == set(), (
        f"{module.__name__} calls {sorted(forbidden)}. A session is held in memory; a call that "
        f"opens a file or runs a program is a second place for one to be."
    )
    assert "os.environ" not in read, "a session is being read from the environment again"
    assert not any(name.startswith("keyring") or "Keychain" in name for name in called), (
        "a keychain is being consulted again"
    )


def test_a_session_never_shows_its_cookies_in_a_repr() -> None:
    """Every message this connector raises names a URL, and several name the credential's
    description. None of them may name the credential."""
    credential = browser_session(sso_config(SERVER_BASE), cookies={"JSESSIONID": "ABC123"})

    assert "ABC123" not in repr(credential)
    assert "ABC123" not in credential.describe()
    assert "ABC123" not in credential.renewal()
