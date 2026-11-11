"""The Graph client: retry, throttling, concurrency, typed failures and what it never prints.

``docs/connectors/outlook.md`` §4.3, §8, §10. The real client runs against the synthetic Graph.
"""

from __future__ import annotations

import asyncio
import logging
from typing import override

import httpx
import pytest

from manicule.connectors.errors import (
    ConnectorError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitedError,
    RemoteError,
    TokenRejectedError,
)
from manicule.connectors.outlook_client import OutlookClient
from tests.connectors.fake_gmail import SimulatedTime
from tests.connectors.fake_graph import (
    FakeCredential,
    FakeGraph,
    FakeGraphSettings,
    mail,
)


def client_for(
    server: FakeGraph,
    time: SimulatedTime,
    *,
    settings: FakeGraphSettings | None = None,
    credential: FakeCredential | None = None,
) -> OutlookClient:
    return OutlookClient(
        settings or FakeGraphSettings(),
        credential or FakeCredential(server),
        transport=server.transport(),
        sleep=time.sleep,
    )


@pytest.fixture
def time() -> SimulatedTime:
    return SimulatedTime()


async def test_a_429_is_waited_out_for_exactly_what_retry_after_says(time: SimulatedTime) -> None:
    server = FakeGraph()
    server.throttle.append((429, "7"))
    client = client_for(server, time)
    await client.setup()
    try:
        profile = await client.get_me()
    finally:
        await client.teardown()

    assert time.waits == [7.0]
    assert profile.object_id
    assert client.throttled == 1


async def test_retry_after_as_an_http_date_is_honored_too(time: SimulatedTime) -> None:
    server = FakeGraph()
    server.throttle.append((429, "Wed, 01 Jan 2020 00:00:00 GMT"))
    client = client_for(server, time)
    await client.setup()
    try:
        await client.get_me()
    finally:
        await client.teardown()

    assert time.waits == [0.0], "a date in the past is no wait, never a hot loop of negative sleeps"


async def test_a_pause_over_the_ceiling_stops_the_run_instead_of_appearing_to_run(
    time: SimulatedTime,
) -> None:
    server = FakeGraph()
    server.throttle.append((429, "9999"))
    client = client_for(server, time)
    await client.setup()
    try:
        with pytest.raises(RateLimitedError, match="ceiling") as stopped:
            await client.get_me()
    finally:
        await client.teardown()

    assert time.waits == []
    assert stopped.value.retry_after == 9999.0


async def test_persistent_throttling_ends_after_the_retry_budget(time: SimulatedTime) -> None:
    server = FakeGraph()
    server.throttle.extend([(429, "1")] * 20)
    client = client_for(server, time, settings=FakeGraphSettings(max_retries=2))
    await client.setup()
    try:
        with pytest.raises(RateLimitedError, match="still limiting"):
            await client.get_me()
    finally:
        await client.teardown()

    assert len(time.waits) == 2


async def test_a_server_error_backs_off_and_then_types_the_failure(time: SimulatedTime) -> None:
    server = FakeGraph()
    server.throttle.extend([(503, None)] * 20)
    client = client_for(server, time, settings=FakeGraphSettings(max_retries=3))
    await client.setup()
    try:
        with pytest.raises(RemoteError, match="HTTP 503"):
            await client.get_me()
    finally:
        await client.teardown()

    assert time.waits == [0.5, 1.0, 2.0], "deterministic exponential backoff"


async def test_a_401_is_refreshed_once_and_the_request_resent(time: SimulatedTime) -> None:
    server = FakeGraph()
    credential = FakeCredential(server)
    server.revoke_access_tokens()
    client = client_for(server, time, credential=credential)
    await client.setup()
    try:
        await client.get_me()
    finally:
        await client.teardown()

    assert credential.refreshes == 1


async def test_a_second_401_is_the_credentials_rejection_not_a_retry_loop(
    time: SimulatedTime,
) -> None:
    server = FakeGraph()

    class Hopeless(FakeCredential):
        @override
        async def access_token(self, *, refresh: bool = False, rejected: str | None = None) -> str:
            del refresh, rejected
            return "never-valid"

        @override
        def rejected(self) -> Exception:
            return TokenRejectedError("rejected twice")

    client = client_for(server, time, credential=Hopeless(server))
    await client.setup()
    try:
        with pytest.raises(TokenRejectedError):
            await client.get_me()
    finally:
        await client.teardown()

    assert len(server.requests) == 2


async def test_a_403_is_a_refusal_that_names_graphs_code_and_nothing_else(
    time: SimulatedTime,
) -> None:
    server = FakeGraph()
    server.forbid_folder_listing = True
    client = client_for(server, time)
    await client.setup()
    try:
        with pytest.raises(PermissionDeniedError) as refused:
            _ = [folder async for folder in client.list_folders()]
    finally:
        await client.teardown()

    assert "ErrorAccessDenied" in str(refused.value)
    assert "sentinel prose" not in str(refused.value), "Graph's free text is never repeated"


async def test_a_missing_item_is_not_found_and_the_error_names_no_id(
    time: SimulatedTime,
) -> None:
    server = FakeGraph()
    client = client_for(server, time)
    await client.setup()
    try:
        with pytest.raises(NotFoundError) as missing:
            await client.get_message_mime("AAMk-secret-id=")
    finally:
        await client.teardown()

    assert "secret-id" not in str(missing.value)


async def test_an_empty_id_is_refused_before_a_request_that_would_list_instead(
    time: SimulatedTime,
) -> None:
    server = FakeGraph()
    client = client_for(server, time)
    await client.setup()
    try:
        with pytest.raises(ConnectorError, match="no id"):
            await client.get_message_mime("")
    finally:
        await client.teardown()

    assert server.requests == []


async def test_the_client_never_has_more_requests_in_flight_than_it_is_configured_for(
    time: SimulatedTime,
) -> None:
    server = FakeGraph()
    graph_id = server.add_message(mail("A"))
    client = client_for(server, time, settings=FakeGraphSettings(max_concurrent_requests=2))
    await client.setup()
    try:
        await asyncio.gather(*(client.get_message_mime(graph_id) for _ in range(12)))
    finally:
        await client.teardown()

    assert server.peak_in_flight == 2


async def test_the_default_concurrency_is_outlooks_cap_of_four(time: SimulatedTime) -> None:
    server = FakeGraph()
    graph_id = server.add_message(mail("A"))
    client = client_for(server, time)
    await client.setup()
    try:
        await asyncio.gather(*(client.get_message_mime(graph_id) for _ in range(12)))
    finally:
        await client.teardown()

    assert server.peak_in_flight == 4


async def test_a_transport_failure_is_retried_then_typed_without_the_url(
    time: SimulatedTime,
) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            "boom: https://graph.microsoft.com/v1.0/me/messages/SECRET", request=request
        )

    client = OutlookClient(
        FakeGraphSettings(max_retries=2),
        FakeCredential(FakeGraph()),
        transport=httpx.MockTransport(refuse),
        sleep=time.sleep,
    )
    await client.setup()
    try:
        with pytest.raises(ConnectorError, match="could not reach Graph") as failed:
            await client.get_me()
    finally:
        await client.teardown()

    assert "SECRET" not in str(failed.value)
    assert failed.value.__cause__ is None, (
        "the library's exception, which carries the URL, is not chained"
    )


async def test_a_request_before_setup_is_a_programming_error(time: SimulatedTime) -> None:
    client = client_for(FakeGraph(), time)
    with pytest.raises(ConnectorError, match="setup"):
        await client.get_me()


async def test_graph_urls_are_redacted_from_the_http_stacks_own_log_records(
    time: SimulatedTime, caplog: pytest.LogCaptureFixture
) -> None:
    server = FakeGraph()
    graph_id = server.add_message(mail("A"))
    client = client_for(server, time)
    await client.setup()
    try:
        with caplog.at_level(logging.INFO, logger="httpx"):
            logging.getLogger("httpx").info(
                'HTTP Request: GET https://graph.microsoft.com/v1.0/me/messages/%s "200 OK"',
                graph_id,
            )
    finally:
        await client.teardown()

    assert graph_id not in caplog.text
    assert "graph.microsoft.com/[redacted]" in caplog.text


async def test_the_folder_tree_is_complete_parents_first_with_slash_paths(
    time: SimulatedTime,
) -> None:
    server = FakeGraph()
    client = client_for(server, time)
    await client.setup()
    try:
        folders = [folder async for folder in client.list_folders()]
    finally:
        await client.teardown()

    paths = {folder.path: folder.parent_id for folder in folders}
    assert paths["Archive/2019"] == "archive"
    assert paths["Sync Issues/Conflicts"] == "syncissues", "hidden folders are walked"
    order = [folder.path for folder in folders]
    assert order.index("Archive") < order.index("Archive/2019")


async def test_a_folder_the_mailbox_lacks_is_none_for_a_well_known_name(
    time: SimulatedTime,
) -> None:
    server = FakeGraph()
    server.delete_folder("Outbox")
    client = client_for(server, time)
    await client.setup()
    try:
        assert await client.well_known_folder_id("outbox") is None
        assert await client.well_known_folder_id("junkemail") == "junk"
    finally:
        await client.teardown()
