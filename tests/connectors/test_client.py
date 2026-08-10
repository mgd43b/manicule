"""Rate limits, retries and credentials.

A full first sync of a large space hits the rate limit — that is expected, not exceptional
(``docs/connectors/confluence.md`` §9). What must not happen is a sync that retries straight
back into the limit, or one that goes to sleep for an hour and looks like it is working.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest

from manicule.connectors import ConnectorError, RateLimitedError
from manicule.connectors.client import BACKOFF_BASE_SECONDS, ConfluenceClient
from manicule.connectors.config import Deployment
from tests.connectors.fake_confluence import CLOUD_BASE, SERVER_BASE, FakeConfluence, FakePage
from tests.connectors.support import Waits, client_for, cloud_config, server_config


def _responder(*responses: httpx.Response) -> httpx.MockTransport:
    """A transport that serves a scripted sequence, repeating the last response."""
    queue = list(responses)

    def handle(request: httpx.Request) -> httpx.Response:
        del request
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return httpx.MockTransport(handle)


async def _client(
    transport: httpx.MockTransport, waits: Waits, **overrides: object
) -> ConfluenceClient:
    config = cloud_config(**overrides)
    client = ConfluenceClient(config, transport=transport, sleep=waits, clock=lambda: 0.0)
    await client.setup()
    return client


async def test_a_throttled_request_waits_exactly_as_long_as_it_was_asked_to() -> None:
    """``Retry-After`` is the source telling us when it will answer. Ignoring it is rude and
    useless: the retry lands inside the same window and is throttled again."""
    waits = Waits()
    transport = _responder(
        httpx.Response(429, headers={"Retry-After": "7"}, json={}),
        httpx.Response(200, json={"results": []}),
    )
    client = await _client(transport, waits)
    try:
        await client.get_json(client.url("/rest/api/space"))
    finally:
        await client.teardown()

    assert waits.seconds == [7.0]
    assert client.throttled == 1


async def test_a_retry_after_given_as_a_date_is_honoured_too() -> None:
    """Both forms are sent. A client that reads only the number waits zero seconds for the
    other one, and retries in a hot loop against a source that is already saying stop."""
    waits = Waits()
    when = datetime.now(tz=UTC) + timedelta(seconds=30)
    transport = _responder(
        httpx.Response(429, headers={"Retry-After": format_datetime(when)}, json={}),
        httpx.Response(200, json={"results": []}),
    )
    client = await _client(transport, waits)
    try:
        await client.get_json(client.url("/rest/api/space"))
    finally:
        await client.teardown()

    assert waits.seconds
    assert 25.0 <= waits.seconds[0] <= 30.0


async def test_a_pause_longer_than_the_ceiling_stops_the_run() -> None:
    """A sync asleep for an hour inside one call is indistinguishable from a hung one.

    Stopping costs re-enumeration on the next run and nothing else, because the watermark does
    not advance.
    """
    waits = Waits()
    transport = _responder(httpx.Response(429, headers={"Retry-After": "3600"}, json={}))
    client = await _client(transport, waits, max_retry_after_seconds=60.0)
    try:
        with pytest.raises(RateLimitedError, match="max_retry_after_seconds"):
            await client.get_json(client.url("/rest/api/space"))
    finally:
        await client.teardown()

    assert waits.seconds == []


async def test_endless_throttling_gives_up_and_says_what_to_change() -> None:
    waits = Waits()
    transport = _responder(httpx.Response(429, headers={"Retry-After": "1"}, json={}))
    client = await _client(transport, waits, max_retries=2)
    try:
        with pytest.raises(RateLimitedError, match="concurrency"):
            await client.get_json(client.url("/rest/api/space"))
    finally:
        await client.teardown()

    assert waits.seconds == [1.0, 1.0]


async def test_a_server_error_is_retried_with_a_widening_gap() -> None:
    """Retrying a 500 immediately is a second request into whatever is already struggling."""
    waits = Waits()
    transport = _responder(
        httpx.Response(500, json={}),
        httpx.Response(500, json={}),
        httpx.Response(200, json={"results": []}),
    )
    client = await _client(transport, waits)
    try:
        await client.get_json(client.url("/rest/api/space"))
    finally:
        await client.teardown()

    assert waits.seconds == [BACKOFF_BASE_SECONDS, BACKOFF_BASE_SECONDS * 2]


async def test_a_persistent_server_error_names_the_known_cause() -> None:
    """Batch endpoints have returned 500 with ADF requested, and knowing that saves an hour."""
    waits = Waits()
    client = await _client(_responder(httpx.Response(500, json={})), waits, max_retries=1)
    try:
        with pytest.raises(ConnectorError, match="Atlassian Document Format"):
            await client.get_json(client.url("/rest/api/space"))
    finally:
        await client.teardown()


async def test_a_rejected_credential_says_which_credential() -> None:
    """Cloud and Server authenticate with different things, so "check your token" is not enough."""
    waits = Waits()
    client = await _client(_responder(httpx.Response(401, json={})), waits)
    try:
        with pytest.raises(ConnectorError, match="CONFLUENCE_API_TOKEN"):
            await client.get_json(client.url("/rest/api/space"))
    finally:
        await client.teardown()


async def test_a_permitted_account_denied_a_page_is_told_what_that_means() -> None:
    """403 is not a broken credential. It is the boundary the index does not cross."""
    waits = Waits()
    client = await _client(_responder(httpx.Response(403, json={})), waits)
    try:
        with pytest.raises(ConnectorError, match="restricted spaces stay invisible"):
            await client.get_json(client.url("/rest/api/space"))
    finally:
        await client.teardown()


async def test_a_sign_in_page_is_not_mistaken_for_an_answer() -> None:
    """A proxy in front of Confluence answers 200 with HTML, and every field then reads empty."""
    waits = Waits()
    transport = _responder(
        httpx.Response(200, text="<html>sign in</html>", headers={"content-type": "text/html"})
    )
    client = await _client(transport, waits)
    try:
        with pytest.raises(ConnectorError, match="sign-in page"):
            await client.get_json(client.url("/rest/api/space"))
    finally:
        await client.teardown()


async def test_a_connection_that_never_comes_back_stops_the_run() -> None:
    waits = Waits()

    def handle(request: httpx.Request) -> httpx.Response:
        msg = "connection refused"
        raise httpx.ConnectError(msg, request=request)

    client = await _client(httpx.MockTransport(handle), waits, max_retries=1)
    try:
        with pytest.raises(ConnectorError, match="could not reach"):
            await client.get_json(client.url("/rest/api/space"))
    finally:
        await client.teardown()

    assert waits.seconds == [BACKOFF_BASE_SECONDS]


async def test_cloud_authenticates_as_email_and_token() -> None:
    """Basic over ``email:token``. The token alone authenticates as nobody."""
    instance = FakeConfluence(base_url=CLOUD_BASE, pages=[FakePage(id="1", title="A", space="ENG")])
    _, client = client_for(instance)
    await client.setup()
    try:
        await client.get_json(client.url("/rest/api/space"))
    finally:
        await client.teardown()

    header = instance.requests[0].headers["authorization"]
    assert header.startswith("Basic ")


async def test_server_authenticates_with_a_bearer_token() -> None:
    """Data Center has no API tokens; it has personal access tokens, and they go in a Bearer."""
    instance = FakeConfluence(
        base_url=SERVER_BASE, pages=[FakePage(id="1", title="A", space="OPS")]
    )
    config = server_config(instance.base_url)
    assert config.deployment is Deployment.SERVER
    _, client = client_for(instance, config)
    await client.setup()
    try:
        await client.get_json(client.url("/rest/api/space"))
    finally:
        await client.teardown()

    assert instance.requests[0].headers["authorization"] == "Bearer pat"
