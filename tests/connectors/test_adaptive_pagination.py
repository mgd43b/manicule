"""Adaptive page sizing for the authoritative Data Center inventory.

The production shape: response time grows with the offset, a full configured page at a deep
``start`` exceeds the request timeout while a smaller page at the same offset still answers,
and the walk must converge rather than fail the same way forever. Only the requested limit may
adapt — the offset, the immutable scope, the status filter and the expansion contract travel
unchanged through every retry, and the transport recordings below fail the suite if that stops
being true.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import httpx
import pytest

from manicule.connectors import ConnectorError, RequestTimeoutError, SessionExpiredError
from manicule.connectors.client import ConfluenceClient
from manicule.connectors.config import ConfluenceConfig, FullInventoryAuthority
from manicule.connectors.confluence import ConfluenceConnector
from tests.connectors.fake_confluence import FakeConfluence, FakePage, SlowOffset
from tests.connectors.support import connected, drain, ids, server_config

BASE = "https://wiki.example.test/confluence"


def _direct(**overrides: object) -> ConfluenceConfig:
    settings: dict[str, object] = {
        "spaces": ("DOCS",),
        "include_attachments": False,
        "full_inventory_authority": FullInventoryAuthority.DIRECT_CURRENT_CONTENT,
    }
    settings.update(overrides)
    return server_config(BASE, **settings)


def _pages(count: int) -> list[FakePage]:
    return [
        FakePage(id=str(700000 + number), title=f"Synthetic {number}", space="DOCS")
        for number in range(count)
    ]


def _direct_requests(instance: FakeConfluence) -> list[httpx.Request]:
    return [
        request for request in instance.requests if request.url.path.endswith("/rest/api/content")
    ]


def _shapes(requests: list[httpx.Request]) -> list[tuple[str, str, str, str, str, str]]:
    """Everything each request said, so an assertion can watch what the retry changed."""
    return [
        (
            request.url.params.get("spaceKey", ""),
            request.url.params.get("type", ""),
            request.url.params.get("status", ""),
            request.url.params.get("expand", ""),
            request.url.params.get("start", ""),
            request.url.params.get("limit", ""),
        )
        for request in requests
    ]


async def test_a_large_offset_timeout_shrinks_only_the_limit_to_the_true_end() -> None:
    """Several consecutive reductions at one offset, then convergence to the empty page.

    The source caps pages at 10 below the configured 25 and stops answering anything larger
    than 5 rows from offset 20 on, so the walk must survive both lies at once: the cap decides
    how far each page advances, and the timeout decides what may be asked for at all.
    """
    pages = _pages(30)
    instance = FakeConfluence(base_url=BASE, pages=pages, page_size=10)
    instance.slow_offsets.append(SlowOffset(start=20, max_limit=5))
    connector = await connected(instance, _direct(page_size=25))
    try:
        found = await drain(connector.discover(None))
        watermark = connector.watermark
    finally:
        await connector.teardown()

    # Every current identity exactly once, in source order, to the explicit empty page.
    assert ids(found) == [page.id for page in pages]
    assert watermark is not None

    shapes = _shapes(_direct_requests(instance))
    # One immutable scope for the whole walk.
    assert {shape[:4] for shape in shapes} == {("DOCS", "page", "current", "version,space")}
    # The timed-out offset was re-asked with nothing changed but the limit: 25 → 12 → 6 → 3.
    at_twenty = [shape for shape in shapes if shape[4] == "20"]
    assert [shape[5] for shape in at_twenty] == ["25", "12", "6", "3"]
    # The walk advanced by the rows that actually arrived, and only the empty page ended it.
    assert [(shape[4], shape[5]) for shape in shapes] == [
        ("0", "25"),
        ("10", "25"),
        ("20", "25"),
        ("20", "12"),
        ("20", "6"),
        ("20", "3"),
        ("23", "3"),
        ("26", "3"),
        ("29", "3"),
        ("30", "3"),
    ]

    progress = connector.enumeration_progress()
    assert progress is not None
    assert progress.timeout_retries == 3
    assert progress.page_size_reduced is True
    assert progress.requested_page_size == 3
    assert progress.reached_empty_page is True


async def test_recovery_at_the_minimum_page_size() -> None:
    """The floor is where shrinking stops, not where the walk gives up."""
    pages = _pages(6)
    instance = FakeConfluence(base_url=BASE, pages=pages, page_size=8)
    instance.slow_offsets.append(SlowOffset(start=0, max_limit=2))
    connector = await connected(instance, _direct(page_size=8, adaptive_min_page_size=2))
    try:
        found = await drain(connector.discover(None))
    finally:
        await connector.teardown()

    assert ids(found) == [page.id for page in pages]
    limits = [shape[5] for shape in _shapes(_direct_requests(instance))]
    assert limits == ["8", "4", "2", "2", "2", "2"]


async def test_attempt_exhaustion_preserves_the_original_typed_timeout() -> None:
    """A bounded policy that runs dry re-raises the timeout it was absorbing."""
    instance = FakeConfluence(base_url=BASE, pages=_pages(4), page_size=4)
    instance.slow_offsets.append(SlowOffset(start=4, max_limit=0))
    connector = await connected(instance, _direct(page_size=4, adaptive_max_attempts_per_offset=3))
    try:
        with pytest.raises(RequestTimeoutError):
            await drain(connector.discover(None))
        assert connector.watermark is None, "an exhausted walk must not report a position"
        progress = connector.enumeration_progress()
    finally:
        await connector.teardown()

    slow_requests = [shape for shape in _shapes(_direct_requests(instance)) if shape[4] == "4"]
    assert len(slow_requests) == 3, "the attempt ceiling bounds requests at one offset"
    assert progress is not None
    assert progress.timeout_retries == 3
    assert progress.reached_empty_page is False


async def test_cumulative_time_exhaustion_is_enforced_at_one_offset() -> None:
    """The wall-clock ceiling stops a stall the attempt ceiling would still permit."""
    instance = FakeConfluence(base_url=BASE, pages=_pages(2), page_size=2)
    instance.slow_offsets.append(SlowOffset(start=2, max_limit=0))
    config = _direct(
        page_size=2,
        adaptive_max_attempts_per_offset=50,
        adaptive_max_seconds_per_offset=240.0,
    )
    moments = iter([0.0, 100.0, 250.0, 400.0, 550.0])
    connector = ConfluenceConnector(
        config,
        ConfluenceClient(config, transport=instance.transport(), clock=lambda: 0.0),
        clock=lambda: next(moments),
    )
    await connector.setup()
    try:
        with pytest.raises(RequestTimeoutError):
            await drain(connector.discover(None))
    finally:
        await connector.teardown()

    slow_requests = [shape for shape in _shapes(_direct_requests(instance)) if shape[4] == "2"]
    assert len(slow_requests) == 2, "the second timeout crossed the 240s ceiling"


async def test_a_gateway_timeout_shrinks_exactly_like_a_read_timeout() -> None:
    """504 from whatever fronts Confluence is the same fact as the socket timing out."""
    pages = _pages(6)
    instance = FakeConfluence(base_url=BASE, pages=pages, page_size=4)
    instance.slow_offsets.append(SlowOffset(start=4, max_limit=2, gateway=True))
    connector = await connected(instance, _direct(page_size=4))
    try:
        found = await drain(connector.discover(None))
    finally:
        await connector.teardown()

    assert ids(found) == [page.id for page in pages]
    at_four = [shape for shape in _shapes(_direct_requests(instance)) if shape[4] == "4"]
    assert [shape[5] for shape in at_four] == ["4", "2"]


async def test_growth_regrows_the_page_after_success_only_when_asked() -> None:
    """The growth dial trades recovered throughput for repeated timeout stalls."""
    pages = _pages(10)
    instance = FakeConfluence(base_url=BASE, pages=pages, page_size=8)
    instance.slow_offsets.append(SlowOffset(start=8, max_limit=4))
    connector = await connected(instance, _direct(page_size=8, adaptive_page_size_growth=True))
    try:
        found = await drain(connector.discover(None))
    finally:
        await connector.teardown()

    assert ids(found) == [page.id for page in pages]
    shapes = _shapes(_direct_requests(instance))
    assert [(shape[4], shape[5]) for shape in shapes] == [
        ("0", "8"),
        ("8", "8"),  # times out
        ("8", "4"),  # shrinks, succeeds
        ("10", "8"),  # regrew after the success
        ("10", "4"),  # times out again, shrinks, and the empty page ends the walk
    ]


def _forbidden(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(403, json={"message": "no access"})


def _bad_request(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(400, json={"message": "bad request"})


def _malformed(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"results": "not-a-list"})


def _offsite_base(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={"results": [], "_links": {"base": "https://elsewhere.example.test"}},
    )


@pytest.mark.parametrize(
    ("respond", "raises", "match"),
    [
        (_forbidden, ConnectorError, "not permitted"),
        (_bad_request, ConnectorError, "answered 400"),
        (_malformed, ConnectorError, "malformed results page"),
        (_offsite_base, ConnectorError, "untrusted link base"),
    ],
)
async def test_non_timeout_failures_are_never_resized_or_retried(
    respond: Callable[[httpx.Request], httpx.Response], raises: type[Exception], match: str
) -> None:
    """Authorization, malformed shapes and untrusted links stay themselves, asked exactly once."""
    pages = _pages(4)
    instance = FakeConfluence(base_url=BASE, pages=pages, page_size=4)
    failing_offset = "4"
    original = instance.handle

    def handle(request: httpx.Request) -> httpx.Response:
        if (
            request.url.path.endswith("/rest/api/content")
            and request.url.params.get("start") == failing_offset
        ):
            instance.requests.append(request)
            return respond(request)
        return original(request)

    connector = ConfluenceConnector(
        _direct(page_size=4),
        ConfluenceClient(
            _direct(page_size=4), transport=httpx.MockTransport(handle), clock=lambda: 0.0
        ),
    )
    await connector.setup()
    try:
        with pytest.raises(raises, match=match):
            await drain(connector.discover(None))
    finally:
        await connector.teardown()

    at_offset = [
        request
        for request in _direct_requests(instance)
        if request.url.params.get("start") == failing_offset
    ]
    assert len(at_offset) == 1, "a non-timeout failure must not be re-asked smaller"


async def test_a_session_failure_mid_walk_is_never_resized_or_retried() -> None:
    """An expired session at a deep offset is a credential fact, not a latency fact."""
    pages = _pages(4)
    instance = FakeConfluence(base_url=BASE, pages=pages, page_size=4)
    original = instance.handle

    def handle(request: httpx.Request) -> httpx.Response:
        if (
            request.url.path.endswith("/rest/api/content")
            and request.url.params.get("start") == "4"
        ):
            instance.sign_out()
        return original(request)

    connector = ConfluenceConnector(
        _direct(page_size=4),
        ConfluenceClient(
            _direct(page_size=4), transport=httpx.MockTransport(handle), clock=lambda: 0.0
        ),
    )
    await connector.setup()
    try:
        with pytest.raises(SessionExpiredError):
            await drain(connector.discover(None))
    finally:
        await connector.teardown()

    at_offset = [
        request for request in _direct_requests(instance) if request.url.params.get("start") == "4"
    ]
    assert len(at_offset) == 1


async def test_source_capped_pages_below_the_request_still_converge() -> None:
    """The cap is not a timeout: nothing shrinks, and the walk advances by what arrived."""
    pages = _pages(7)
    instance = FakeConfluence(base_url=BASE, pages=pages, page_size=3)
    connector = await connected(instance, _direct(page_size=250))
    try:
        found = await drain(connector.discover(None))
        progress = connector.enumeration_progress()
    finally:
        await connector.teardown()

    assert ids(found) == [page.id for page in pages]
    shapes = _shapes(_direct_requests(instance))
    assert [(shape[4], shape[5]) for shape in shapes] == [
        ("0", "250"),
        ("3", "250"),
        ("6", "250"),
        ("7", "250"),
    ]
    assert progress is not None
    assert progress.timeout_retries == 0
    assert progress.page_size_reduced is False
    assert progress.requested_page_size == 250
    assert progress.reached_empty_page is True


async def test_incremental_direct_authority_reports_no_empty_page_claim() -> None:
    """A CQL-answered incremental run must not claim an authoritative empty-page end."""
    page = FakePage(id="710000", title="Current", space="DOCS")
    instance = FakeConfluence(base_url=BASE, pages=[page])
    connector = await connected(instance, _direct())
    try:
        await drain(connector.discover(None))
        committed = connector.watermark
        assert committed is not None
        resumed = await connected(instance, _direct())
        try:
            await drain(resumed.discover(committed))
            progress = resumed.enumeration_progress()
        finally:
            await resumed.teardown()
    finally:
        await connector.teardown()

    assert progress is not None
    assert progress.reached_empty_page is None
    assert progress.timeout_retries == 0


async def test_cancellation_during_an_adaptive_retry_is_not_absorbed() -> None:
    """The retry loop must not turn a cancellation into another smaller request.

    A stop arriving while the connector is shrinking its page is the one interruption this
    loop could plausibly swallow — it catches a timeout and immediately asks again — so the
    cancellation is delivered from inside the timing-out request itself.
    """
    pages = _pages(8)
    instance = FakeConfluence(base_url=BASE, pages=pages, page_size=4)
    instance.slow_offsets.append(SlowOffset(start=4, max_limit=0))
    requests = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        if (
            request.url.path.endswith("/rest/api/content")
            and request.url.params.get("start") == "4"
        ):
            requests += 1
            if requests == 2:
                raise asyncio.CancelledError
        return instance.handle(request)

    config = _direct(page_size=4)
    connector = ConfluenceConnector(
        config,
        ConfluenceClient(config, transport=httpx.MockTransport(handle), clock=lambda: 0.0),
    )
    await connector.setup()
    try:
        with pytest.raises(asyncio.CancelledError):
            await drain(connector.discover(None))
    finally:
        await connector.teardown()

    assert requests == 2, "the canceled attempt must not be followed by another retry"
    progress = connector.enumeration_progress()
    assert progress is not None
    # The prefix admitted before the stop is still what the walk had reached, and the run is
    # honestly incomplete rather than claiming an end.
    assert progress.offset == 4
    assert progress.reached_empty_page is False
