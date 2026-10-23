"""The Gmail client against a synthetic Gmail: calls, error typing, pacing, and imports.

The real client runs over :class:`~tests.connectors.fake_gmail.FakeGmail` through an injected
transport, so the real pacer, retry loop and query encoding are what is under test.
"""

from __future__ import annotations

import asyncio
import email
import email.policy
import json
import logging
import subprocess
import sys
import tomllib
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from typing import override

import pytest

from manicule.connectors.errors import (
    ConnectorError,
    CursorExpiredError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitedError,
    RemoteError,
    RequestTimeoutError,
    SessionExpiredError,
)
from manicule.connectors.gmail_client import (
    COST_GET_RAW,
    WINDOW_SECONDS,
    GmailClient,
    QuotaPacer,
)
from manicule.core.errors import ConfigError
from tests.connectors.fake_gmail import (
    ADDRESS,
    FakeCredential,
    FakeGmail,
    FakeGmailSettings,
    SimulatedTime,
    html_only_message,
    message_with_attachment,
    plain_message,
)

ROOT = Path(__file__).resolve().parents[2]


async def _client(
    server: FakeGmail, time: SimulatedTime, **settings: float
) -> tuple[GmailClient, FakeCredential]:
    credential = FakeCredential(server)
    client = GmailClient(
        FakeGmailSettings(**settings),  # pyright: ignore[reportArgumentType]
        credential,
        transport=server.transport(),
        sleep=time.sleep,
        clock=time,
    )
    await client.setup()
    return client, credential


async def _pages[T](pages: AsyncIterator[T]) -> list[T]:
    return [page async for page in pages]


def _server(time: SimulatedTime, **options: int) -> FakeGmail:
    return FakeGmail(clock=time, **options)  # pyright: ignore[reportArgumentType]


# --------------------------------------------------------------------------- the calls


async def test_the_profile_names_the_mailbox_and_its_history_position() -> None:
    time = SimulatedTime()
    server = _server(time)
    client, _ = await _client(server, time)
    try:
        profile = await client.get_profile()
    finally:
        await client.teardown()

    assert profile.email_address == ADDRESS
    assert profile.history_id == "1000"
    assert isinstance(profile.history_id, str)


async def test_labels_keep_their_names_with_spaces_slashes_and_categories() -> None:
    """A name is resolved to an id by the caller, so it must come back exactly as written."""
    time = SimulatedTime()
    server = _server(time)
    client, _ = await _client(server, time)
    try:
        labels = {label.name: label.id for label in await client.list_labels()}
    finally:
        await client.teardown()

    assert labels["Clients/Acme Corp"] == server.label_id("Clients/Acme Corp")
    assert "Receipts 2026" in labels
    assert labels["CATEGORY_PROMOTIONS"] == "CATEGORY_PROMOTIONS"
    assert labels["SPAM"] == "SPAM"


async def test_listing_follows_pagination_tokens_one_page_at_a_time() -> None:
    """Tokens contain ``+`` and ``/``; one that was built into a query by hand would corrupt."""
    time = SimulatedTime()
    server = _server(time, page_size=2)
    made = [server.add_message() for _ in range(5)]
    client, _ = await _client(server, time)
    try:
        pages = await _pages(client.list_messages())
    finally:
        await client.teardown()

    assert [len(page) for page in pages] == [2, 2, 1]
    assert {item for page in pages for item in page} == {message.id for message in made}


async def test_an_empty_mailbox_lists_as_one_empty_page() -> None:
    """Gmail leaves the ``messages`` key out entirely rather than sending an empty list."""
    time = SimulatedTime()
    client, _ = await _client(_server(time), time)
    try:
        pages = await _pages(client.list_messages())
    finally:
        await client.teardown()

    assert pages == [()]


async def test_listing_never_sends_a_search_query_and_leaves_spam_and_trash_out() -> None:
    time = SimulatedTime()
    server = _server(time)
    kept = server.add_message(labels=("INBOX",))
    server.add_message(labels=("SPAM",))
    server.add_message(labels=("INBOX", "TRASH"))
    client, _ = await _client(server, time)
    try:
        pages = await _pages(client.list_messages())
    finally:
        await client.teardown()

    assert pages == [(kept.id,)]
    query = server.queries()[-1]
    assert query["includeSpamTrash"] == ["false"]
    assert query["maxResults"] == ["500"]
    assert all("q" not in each for each in server.queries())


async def test_listing_can_be_narrowed_by_label_ids() -> None:
    time = SimulatedTime()
    server = _server(time)
    server.add_message(labels=("INBOX",))
    wanted = server.add_message(labels=("INBOX", "Clients/Acme Corp"))
    client, _ = await _client(server, time)
    try:
        pages = await _pages(client.list_messages([server.label_id("Clients/Acme Corp")]))
    finally:
        await client.teardown()

    assert pages == [(wanted.id,)]
    assert server.queries()[-1]["labelIds"] == [server.label_id("Clients/Acme Corp")]


async def test_pagination_sent_back_to_a_followed_page_is_refused() -> None:
    time = SimulatedTime()
    server = _server(time, page_size=2)
    for _ in range(6):
        server.add_message()
    server.loop_pagination = True
    client, _ = await _client(server, time)
    try:
        with pytest.raises(ConnectorError, match="already followed"):
            await _pages(client.list_messages())
    finally:
        await client.teardown()


@pytest.mark.parametrize(
    ("raw", "marker"),
    [
        (plain_message("Plain"), "Plain"),
        (html_only_message("Letter"), "Letter"),
        (message_with_attachment("Filed"), "Filed"),
    ],
    ids=["plain", "html-only", "attachment"],
)
async def test_raw_fetch_returns_the_rfc_5322_bytes_unpadded_base64_decoded(
    raw: bytes, marker: str
) -> None:
    time = SimulatedTime()
    server = _server(time)
    message = server.add_message(raw)
    client, _ = await _client(server, time)
    try:
        fetched = await client.get_raw(message.id)
    finally:
        await client.teardown()

    assert fetched == raw
    parsed = email.message_from_bytes(fetched, policy=email.policy.default)
    assert parsed["Subject"] == marker


async def test_raw_fetch_of_a_message_with_an_attachment_keeps_the_attachment() -> None:
    time = SimulatedTime()
    server = _server(time)
    message = server.add_message(message_with_attachment())
    client, _ = await _client(server, time)
    try:
        fetched = await client.get_raw(message.id)
    finally:
        await client.teardown()

    parsed = email.message_from_bytes(fetched, policy=email.policy.default)
    assert [part.get_filename() for part in parsed.iter_attachments()] == ["report.pdf"]


async def test_minimal_fetch_returns_the_current_label_ids() -> None:
    time = SimulatedTime()
    server = _server(time)
    message = server.add_message(labels=("INBOX", "Receipts 2026"))
    unlabeled = server.add_message(labels=())
    client, _ = await _client(server, time)
    try:
        labels = await client.get_minimal(message.id)
        bare = await client.get_minimal(unlabeled.id)
    finally:
        await client.teardown()

    assert labels.label_ids == ("INBOX", server.label_id("Receipts 2026"))
    assert bare.label_ids == ()
    assert server.queries()[-1]["format"] == ["minimal"]


async def test_history_reports_records_with_and_without_label_ids() -> None:
    time = SimulatedTime()
    server = _server(time)
    start = server.history_id
    bare = server.add_message(labels=("INBOX",))
    labeled = server.add_message(labels=("INBOX",), record_labels=True)
    server.add_label(bare, "STARRED")
    server.remove_label(labeled, "INBOX", record_labels=True)
    server.delete_message(bare)
    client, _ = await _client(server, time)
    try:
        pages = await _pages(client.history(str(start)))
    finally:
        await client.teardown()

    changes = [change for page in pages for change in page.changes]
    by_kind = {(change.kind, change.message_id): change for change in changes}
    assert by_kind[("messageAdded", bare.id)].label_ids is None
    assert by_kind[("messageAdded", labeled.id)].label_ids == ("INBOX",)
    assert by_kind[("labelAdded", bare.id)].changed_label_ids == ("STARRED",)
    assert by_kind[("labelRemoved", labeled.id)].changed_label_ids == ("INBOX",)
    assert ("messageDeleted", bare.id) in by_kind
    assert pages[0].history_id == str(server.history_id)
    asked = server.queries()[-1]
    assert asked["startHistoryId"] == [str(start)]
    assert sorted(asked["historyTypes"]) == [
        "labelAdded",
        "labelRemoved",
        "messageAdded",
        "messageDeleted",
    ]


async def test_history_with_nothing_new_is_one_empty_page_with_a_position() -> None:
    time = SimulatedTime()
    server = _server(time)
    client, _ = await _client(server, time)
    try:
        pages = await _pages(client.history(str(server.history_id)))
    finally:
        await client.teardown()

    assert [page.changes for page in pages] == [()]
    assert pages[0].history_id == str(server.history_id)


async def test_history_pages_are_batches_and_each_carries_the_position() -> None:
    time = SimulatedTime()
    server = _server(time, page_size=2)
    start = server.history_id
    for _ in range(5):
        server.add_message()
    client, _ = await _client(server, time)
    try:
        pages = await _pages(client.history(str(start)))
    finally:
        await client.teardown()

    assert [len(page.changes) for page in pages] == [2, 2, 1]
    assert {page.history_id for page in pages} == {str(server.history_id)}


# --------------------------------------------------------------------------- the error table


async def test_429_with_seconds_waits_that_long_and_then_succeeds() -> None:
    time = SimulatedTime()
    server = _server(time)
    server.throttle(1, retry_after="7")
    client, _ = await _client(server, time)
    try:
        profile = await client.get_profile()
    finally:
        await client.teardown()

    assert profile.email_address == ADDRESS
    assert time.waits == [7.0]
    assert client.throttled == 1


async def test_429_with_an_http_date_waits_until_then() -> None:
    time = SimulatedTime()
    server = _server(time)
    server.throttle_until(30)
    client, _ = await _client(server, time)
    try:
        await client.get_profile()
    finally:
        await client.teardown()

    assert len(time.waits) == 1
    assert 25.0 <= time.waits[0] <= 30.0


async def test_429_without_retry_after_backs_off_exponentially() -> None:
    time = SimulatedTime()
    server = _server(time)
    server.throttle(2, retry_after=None)
    client, _ = await _client(server, time)
    try:
        await client.get_profile()
    finally:
        await client.teardown()

    assert time.waits == [0.5, 1.0]


@pytest.mark.parametrize(
    ("reason", "style"),
    [
        ("rateLimitExceeded", "errors"),
        ("userRateLimitExceeded", "errors"),
        ("RATE_LIMIT_EXCEEDED", "details"),
    ],
)
async def test_a_403_whose_reason_is_a_rate_limit_is_retried_like_a_429(
    reason: str, style: str
) -> None:
    time = SimulatedTime()
    server = _server(time)
    server.refuse(reason, retry_after="3", style=style)
    client, _ = await _client(server, time)
    try:
        profile = await client.get_profile()
    finally:
        await client.teardown()

    assert profile.email_address == ADDRESS
    assert time.waits == [3.0]


async def test_a_rate_limit_403_can_be_given_the_date_form_too() -> None:
    time = SimulatedTime()
    server = _server(time)
    server.refuse(
        "userRateLimitExceeded",
        retry_after=format_datetime(datetime.now(tz=UTC) + timedelta(seconds=20), usegmt=True),
    )
    client, _ = await _client(server, time)
    try:
        await client.get_profile()
    finally:
        await client.teardown()

    assert 15.0 <= time.waits[0] <= 20.0


async def test_a_pause_over_the_ceiling_stops_the_run_without_waiting() -> None:
    time = SimulatedTime()
    server = _server(time)
    server.throttle(1, retry_after="3600")
    client, _ = await _client(server, time, max_retry_after_seconds=60.0)
    try:
        with pytest.raises(RateLimitedError, match="max_retry_after_seconds") as caught:
            await client.get_profile()
    finally:
        await client.teardown()

    assert time.waits == []
    assert caught.value.retry_after == 3600.0


async def test_endless_throttling_ends_in_rate_limited_after_the_retry_budget() -> None:
    time = SimulatedTime()
    server = _server(time)
    server.throttle(10, retry_after="1")
    client, _ = await _client(server, time, max_retries=2)
    try:
        with pytest.raises(RateLimitedError):
            await client.get_profile()
    finally:
        await client.teardown()

    assert time.waits == [1.0, 1.0]
    assert client.requests == 3


async def test_a_404_on_history_is_an_expired_cursor() -> None:
    time = SimulatedTime()
    server = _server(time)
    server.add_message()
    server.expire_history_through(server.history_id)
    client, _ = await _client(server, time)
    try:
        with pytest.raises(CursorExpiredError):
            await _pages(client.history("1000"))
    finally:
        await client.teardown()


async def test_a_404_on_get_is_a_message_deleted_between_discovery_and_fetch() -> None:
    time = SimulatedTime()
    server = _server(time)
    message = server.add_message()
    server.delete_message(message, still_listed=True)
    client, _ = await _client(server, time)
    try:
        pages = await _pages(client.list_messages())
        with pytest.raises(NotFoundError):
            await client.get_raw(message.id)
        with pytest.raises(NotFoundError):
            await client.get_minimal(message.id)
    finally:
        await client.teardown()

    assert pages == [(message.id,)]


async def test_a_404_anywhere_else_is_an_ordinary_unusable_answer() -> None:
    """Only the two calls with a meaning for 404 get one; a vanished profile is a fault."""
    time = SimulatedTime()
    server = _server(time)
    server.fail(404)
    client, _ = await _client(server, time)
    try:
        with pytest.raises(RemoteError) as caught:
            await client.get_profile()
    finally:
        await client.teardown()

    assert not isinstance(caught.value, NotFoundError)
    assert caught.value.status_code == 404


async def test_a_401_is_refreshed_once_and_the_request_retried() -> None:
    time = SimulatedTime()
    server = _server(time)
    client, credential = await _client(server, time)
    server.expire_tokens()
    try:
        profile = await client.get_profile()
    finally:
        await client.teardown()

    assert profile.email_address == ADDRESS
    assert credential.refreshes == 1


async def test_a_second_401_is_a_credential_error_naming_the_renewal() -> None:
    time = SimulatedTime()
    server = _server(time)
    server.rejecting_all_tokens = True
    client, credential = await _client(server, time)
    try:
        with pytest.raises(SessionExpiredError, match="manicule connector login"):
            await client.get_profile()
    finally:
        await client.teardown()

    assert credential.refreshes == 1
    assert client.requests == 2


@pytest.mark.parametrize("reason", ["forbidden", "insufficientPermissions", "accessNotConfigured"])
async def test_any_other_403_is_permission_denied_and_not_retried(reason: str) -> None:
    time = SimulatedTime()
    server = _server(time)
    server.refuse(reason)
    client, _ = await _client(server, time)
    try:
        with pytest.raises(PermissionDeniedError, match=reason):
            await client.get_profile()
    finally:
        await client.teardown()

    assert client.requests == 1
    assert time.waits == []


async def test_a_5xx_is_retried_and_then_succeeds() -> None:
    time = SimulatedTime()
    server = _server(time)
    server.fail(503, times=2)
    client, _ = await _client(server, time)
    try:
        await client.get_profile()
    finally:
        await client.teardown()

    assert time.waits == [0.5, 1.0]


async def test_a_5xx_that_persists_is_a_remote_error_with_its_status() -> None:
    time = SimulatedTime()
    server = _server(time)
    server.fail(500, times=10)
    client, _ = await _client(server, time, max_retries=1)
    try:
        with pytest.raises(RemoteError) as caught:
            await client.get_profile()
    finally:
        await client.teardown()

    assert caught.value.status_code == 500
    assert time.waits == [0.5]


async def test_a_timeout_is_retried_and_then_succeeds() -> None:
    time = SimulatedTime()
    server = _server(time)
    server.time_out(2)
    client, _ = await _client(server, time)
    try:
        await client.get_profile()
    finally:
        await client.teardown()

    assert time.waits == [0.5, 1.0]


async def test_a_timeout_that_persists_is_a_request_timeout() -> None:
    time = SimulatedTime()
    server = _server(time)
    server.time_out(10)
    client, _ = await _client(server, time, max_retries=1)
    try:
        with pytest.raises(RequestTimeoutError):
            await client.get_profile()
    finally:
        await client.teardown()


async def test_an_unreachable_server_is_a_connector_error_after_the_budget() -> None:
    time = SimulatedTime()
    server = _server(time)
    server.drop_connection(10)
    client, _ = await _client(server, time, max_retries=1)
    try:
        with pytest.raises(ConnectorError, match="could not reach Gmail"):
            await client.get_profile()
    finally:
        await client.teardown()


@pytest.mark.parametrize("raw", ["!!!!", "$$$$$$$$", "abc*def"])
async def test_a_body_with_characters_outside_base64url_is_not_decoded_to_something(
    raw: str,
) -> None:
    """A lenient decoder discards the characters and returns empty or shortened bytes."""
    time = SimulatedTime()
    server = _server(time)
    message = server.add_message()
    server.raw_override = raw
    client, _ = await _client(server, time)
    try:
        with pytest.raises(ConnectorError, match="not base64url"):
            await client.get_raw(message.id)
    finally:
        await client.teardown()


async def test_a_body_that_is_not_base64url_is_a_connector_error() -> None:
    time = SimulatedTime()
    server = _server(time)
    message = server.add_message()
    server.corrupt_raw = True
    client, _ = await _client(server, time)
    try:
        with pytest.raises(ConnectorError, match="not base64url"):
            await client.get_raw(message.id)
    finally:
        await client.teardown()


async def test_a_message_with_no_id_is_refused_rather_than_sent_as_a_listing() -> None:
    """``/messages/`` is the listing endpoint; an empty id must never reach the wire."""
    time = SimulatedTime()
    server = _server(time)
    client, _ = await _client(server, time)
    try:
        with pytest.raises(ConnectorError, match="no id"):
            await client.get_raw("")
        with pytest.raises(ConnectorError, match="no id"):
            await client.get_minimal("")
    finally:
        await client.teardown()

    assert server.requests == []


async def test_no_error_carries_a_message_id_an_address_or_a_label_name() -> None:
    """``docs/connectors/gmail.md`` §10: a log line must not name what was in the mailbox."""
    time = SimulatedTime()
    server = _server(time)
    message = server.add_message(labels=("INBOX", "Clients/Acme Corp"))
    server.delete_message(message, still_listed=True)
    client, _ = await _client(server, time, max_retries=1)
    raised: list[ConnectorError] = []

    async def failing(
        call: Callable[[], Awaitable[object]], expected: type[ConnectorError]
    ) -> None:
        with pytest.raises(expected) as caught:
            await call()
        raised.append(caught.value)

    try:
        await failing(lambda: client.get_raw(message.id), NotFoundError)
        await failing(lambda: client.get_minimal(message.id), NotFoundError)
        server.expire_history_through(server.history_id)
        await failing(lambda: _pages(client.history("1")), CursorExpiredError)
        # With max_retries=1 a failing call consumes exactly two queued faults.
        server.time_out(2)
        await failing(lambda: client.get_raw(message.id), RequestTimeoutError)
        server.fail(503, times=2)
        await failing(lambda: client.get_minimal(message.id), RemoteError)
        server.refuse("forbidden")
        await failing(lambda: client.get_minimal(message.id), PermissionDeniedError)
        server.throttle(2, retry_after="1")
        await failing(lambda: client.get_raw(message.id), RateLimitedError)
    finally:
        await client.teardown()

    assert server.faults_pending == 0
    assert [type(error) for error in raised] == [
        NotFoundError,
        NotFoundError,
        CursorExpiredError,
        RequestTimeoutError,
        RemoteError,
        PermissionDeniedError,
        RateLimitedError,
    ]
    assert isinstance(raised[4], RemoteError)
    assert raised[4].status_code == 503
    for error in raised:
        text = str(error)
        assert message.id not in text
        assert ADDRESS not in text
        assert "Acme" not in text
        assert "googleapis" not in text


async def test_the_http_stacks_own_request_log_does_not_name_a_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``httpx`` logs each request's full URL at INFO; for ``messages.get`` that is the id."""
    import httpx  # noqa: PLC0415

    time = SimulatedTime()
    server = _server(time)
    message = server.add_message(labels=("INBOX", "Clients/Acme Corp"))
    client, _ = await _client(server, time)
    other = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={}))
    )
    caplog.set_level(logging.INFO, logger="httpx")
    try:
        await client.get_raw(message.id)
        await _pages(client.history("1"))
        await other.get("https://example.org/unrelated/path?page=7")
    finally:
        await client.teardown()
        await other.aclose()

    logged = [record.getMessage() for record in caplog.records if record.name == "httpx"]
    gmail = [line for line in logged if "gmail.googleapis.com" in line]
    assert len(gmail) == 2
    assert all(message.id not in line and "startHistoryId" not in line for line in gmail)
    # Another client's request is logged exactly as before.
    assert any("https://example.org/unrelated/path?page=7" in line for line in logged)


async def test_a_request_before_setup_is_refused() -> None:
    time = SimulatedTime()
    server = _server(time)
    client = GmailClient(
        FakeGmailSettings(),
        FakeCredential(server),
        transport=server.transport(),
        sleep=time.sleep,
        clock=time,
    )
    with pytest.raises(ConnectorError, match="setup"):
        await client.get_profile()


# --------------------------------------------------------------------------- pacing


def _windows(charges: Sequence[tuple[float, int]]) -> int:
    """The most units any rolling ``WINDOW_SECONDS`` span of ``charges`` holds."""
    worst = 0
    for index, (start, _) in enumerate(charges):
        worst = max(
            worst,
            sum(units for moment, units in charges[index:] if moment - start < WINDOW_SECONDS),
        )
    return worst


async def test_a_budget_below_one_raw_fetch_is_refused_at_construction() -> None:
    with pytest.raises(ConfigError, match="quota_units_per_minute"):
        QuotaPacer(COST_GET_RAW - 1)


async def test_the_pacer_never_exceeds_its_budget_over_any_rolling_minute() -> None:
    """A mix of costs, hammered; the account is kept by the pacer's caller, not the pacer."""
    time = SimulatedTime()
    pacer = QuotaPacer(5400, clock=time, sleep=time.sleep)
    charges: list[tuple[float, int]] = []
    pattern = [60, 60, 20, 5, 2, 1, 60, 5, 20, 60]
    for index in range(1500):
        units = pattern[index % len(pattern)]
        await pacer.acquire(units)
        charges.append((time.now, units))

    assert _windows(charges) <= 5400
    assert time.now > 0
    # Fully used rather than merely safe: the pacer must not idle with budget to spare.
    assert _windows(charges) >= 5400 - 60


async def test_a_simulated_minute_of_raw_fetching_stays_inside_the_budget_on_the_server() -> None:
    """The fake counts independently of the client and refuses what overruns Gmail's own limit."""
    time = SimulatedTime()
    server = _server(time)
    messages = [server.add_message() for _ in range(300)]
    client, _ = await _client(server, time)
    try:
        for message in messages:
            await client.get_raw(message.id)
    finally:
        await client.teardown()

    assert server.overruns == 0
    assert server.peak_units <= 5400
    assert server.peak_units == 5400  # 90 raw fetches in a minute, not fewer
    # 300 fetches at 90 a minute: bursts at t = 0, 60, 120 and 180.
    assert time.now == 180.0
    assert client.throttled == 0


async def test_concurrent_fetchers_share_one_budget() -> None:
    time = SimulatedTime()
    server = _server(time)
    messages = [server.add_message() for _ in range(200)]
    client, _ = await _client(server, time)
    try:
        await asyncio.gather(*(client.get_raw(message.id) for message in messages))
    finally:
        await client.teardown()

    assert server.overruns == 0
    assert server.peak_units <= 5400


class _SlowFirstToken(FakeCredential):
    """A credential whose first token costs a second, as a refresh over HTTP would."""

    def __init__(self, server: FakeGmail, time: SimulatedTime) -> None:
        super().__init__(server)
        self._time = time
        self._asked = False

    @override
    async def access_token(self, *, refresh: bool = False) -> str:
        if not self._asked:
            self._asked = True
            await self._time.sleep(1.0)
        return await super().access_token(refresh=refresh)


async def test_a_slow_token_lookup_does_not_shorten_the_gap_between_charges() -> None:
    """The charge is made when the request leaves, not when the token was first asked for."""
    time = SimulatedTime()
    server = _server(time)
    messages = [server.add_message() for _ in range(2)]
    client = GmailClient(
        FakeGmailSettings(quota_units_per_minute=COST_GET_RAW),
        _SlowFirstToken(server, time),
        transport=server.transport(),
        sleep=time.sleep,
        clock=time,
    )
    await client.setup()
    try:
        for message in messages:
            await client.get_raw(message.id)
    finally:
        await client.teardown()

    first, second = server.request_times
    assert second - first >= WINDOW_SECONDS


async def test_a_smaller_budget_is_honored() -> None:
    time = SimulatedTime()
    server = _server(time)
    messages = [server.add_message() for _ in range(20)]
    client, _ = await _client(server, time, quota_units_per_minute=600)
    try:
        for message in messages:
            await client.get_raw(message.id)
    finally:
        await client.teardown()

    assert server.peak_units <= 600
    assert time.now == 60.0


async def test_a_429_still_backs_off_while_the_client_is_pacing() -> None:
    time = SimulatedTime()
    server = _server(time)
    messages = [server.add_message() for _ in range(100)]
    server.throttle(1, retry_after="11")
    client, _ = await _client(server, time)
    try:
        for message in messages:
            await client.get_raw(message.id)
    finally:
        await client.teardown()

    assert 11.0 in time.waits
    assert client.throttled == 1
    assert server.peak_units <= 5400
    assert server.overruns == 0


async def test_every_attempt_is_charged_including_retries() -> None:
    time = SimulatedTime()
    server = _server(time)
    message = server.add_message()
    server.throttle(2, retry_after="1")
    client, _ = await _client(server, time)
    try:
        await client.get_raw(message.id)
    finally:
        await client.teardown()

    assert server.peak_units == 3 * COST_GET_RAW


async def test_each_call_spends_its_documented_units() -> None:
    time = SimulatedTime()
    server = _server(time, page_size=500)
    message = server.add_message()
    client, _ = await _client(server, time)
    try:
        await client.get_profile()
        assert server.peak_units == 1
        await client.list_labels()
        assert server.peak_units == 2
        await _pages(client.list_messages())
        assert server.peak_units == 7
        await _pages(client.history("1"))
        assert server.peak_units == 9
        await client.get_minimal(message.id)
        assert server.peak_units == 29
        await client.get_raw(message.id)
        assert server.peak_units == 89
    finally:
        await client.teardown()


# --------------------------------------------------------------------------- dependencies


def test_no_google_client_library_is_a_dependency() -> None:
    """The REST calls go over the HTTP stack the Confluence client already uses (§12)."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    declared = [
        *project.get("dependencies", []),
        *(dep for group in project.get("optional-dependencies", {}).values() for dep in group),
    ]
    assert not [dep for dep in declared if dep.lower().startswith(("google", "gmail"))]


def test_importing_the_client_loads_no_http_stack_and_setting_it_up_does() -> None:
    """Acceptance: the client is imported by the factory, never by plugin discovery.

    The second half is what keeps the first from being vacuous — the stack does load, but only
    when a client is set up.
    """
    script = (
        "import asyncio, json, sys\n"
        "from manicule.connectors.gmail_client import GmailClient\n"
        "before = sorted(name for name in sys.modules if name.split('.')[0] in "
        "('httpx', 'google', 'googleapiclient'))\n"
        "class Settings:\n"
        "    quota_units_per_minute = 5400\n"
        "    max_retry_after_seconds = 120.0\n"
        "    max_retries = 5\n"
        "    request_timeout_seconds = 30.0\n"
        "class Credential:\n"
        "    async def access_token(self, *, refresh=False): return ''\n"
        "    def renewal(self): return ''\n"
        "client = GmailClient(Settings(), Credential())\n"
        "constructed = 'httpx' in sys.modules\n"
        "asyncio.run(client.setup())\n"
        "print(json.dumps([before, constructed, 'httpx' in sys.modules]))\n"
    )
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-I", "-c", script], capture_output=True, text=True, check=True
    )
    before, constructed, after = json.loads(completed.stdout)

    assert before == []
    assert constructed is False
    assert after is True
