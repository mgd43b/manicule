"""Deterministic transport, robots, retry and streaming bounds for the crawler."""

from __future__ import annotations

import asyncio
import gzip
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Self, override

import pytest

from manicule.connectors.crawler_security import ConnectionPlan, CrawlerUrlPolicy
from manicule.connectors.crawler_transport import (
    CRAWLER_USER_AGENT,
    CrawlerHttpClient,
    CrawlerHttpConfig,
    DialResponse,
)
from manicule.connectors.errors import (
    CrawlerChallengeError,
    CrawlerMediaTypeError,
    CrawlerResponseTooLargeError,
    CrawlerRobotsError,
    CrawlerThrottleError,
    CrawlerTimeoutError,
    CrawlerTransportError,
)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.waits: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds


class Stream:
    def __init__(self, chunks: Sequence[bytes], *, gate: asyncio.Event | None = None) -> None:
        self._chunks = deque(chunks)
        self._gate = gate
        self.closed = False
        self.reads = 0

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> bytes:
        if self._gate is not None:
            await self._gate.wait()
        if not self._chunks:
            raise StopAsyncIteration
        self.reads += 1
        return self._chunks.popleft()

    async def aclose(self) -> None:
        self.closed = True


@dataclass(frozen=True, slots=True)
class Answer:
    status: int = 200
    headers: tuple[tuple[str, str], ...] = (("Content-Type", "text/html"),)
    chunks: tuple[bytes, ...] = (b"<main>page</main>",)
    peer: str = "8.8.8.8"
    gate: asyncio.Event | None = None


class Dialer:
    def __init__(self) -> None:
        self.answers: dict[str, deque[Answer | BaseException]] = defaultdict(deque)
        self.requests: list[tuple[str, Mapping[str, str], float]] = []
        self.streams: list[Stream] = []
        self.clock: Clock | None = None

    def add(self, url: str, *answers: Answer | BaseException) -> None:
        self.answers[url].extend(answers)

    async def open(
        self, plan: ConnectionPlan, *, method: str, headers: Mapping[str, str]
    ) -> DialResponse:
        assert method == "GET"
        self.requests.append(
            (plan.target.url, dict(headers), self.clock.now if self.clock is not None else 0.0)
        )
        queued = self.answers[plan.target.url]
        answer = (
            queued.popleft()
            if queued
            else Answer(
                status=404,
                headers=(("Content-Type", "text/plain"),),
                chunks=(),
            )
            if plan.target.path == "/robots.txt"
            else AssertionError("test dialer has no configured answer")
        )
        if isinstance(answer, BaseException):
            raise answer
        stream = Stream(answer.chunks, gate=answer.gate)
        self.streams.append(stream)
        return DialResponse(
            status=answer.status,
            headers=answer.headers,
            peer=answer.peer,
            body=stream,
        )


async def resolver(hostname: str, port: int) -> Sequence[str]:
    del hostname, port
    return ("8.8.8.8",)


def client(
    dialer: Dialer,
    *,
    clock: Clock | None = None,
    config: CrawlerHttpConfig | None = None,
    origins: tuple[str, ...] = ("https://docs.example.test",),
) -> CrawlerHttpClient:
    clock = clock or Clock()
    dialer.clock = clock
    return CrawlerHttpClient(
        CrawlerUrlPolicy(allowed_origins=origins, allowed_path_prefixes=("/docs/",)),
        dialer,
        resolver,
        config=config,
        clock=clock,
        sleep=clock.sleep,
        wall_clock=lambda: datetime(2026, 8, 24, tzinfo=UTC),
    )


def page(path: str = "a") -> str:
    return f"https://docs.example.test/docs/{path}"


def robots() -> str:
    return "https://docs.example.test/robots.txt"


def test_the_user_agent_is_declared_and_stable() -> None:
    assert CRAWLER_USER_AGENT == "ManiculeCrawler/0.1 (+https://github.com/mgd43b/manicule)"


async def test_robots_disallow_prevents_the_page_request() -> None:
    dialer = Dialer()
    dialer.add(
        robots(),
        Answer(
            headers=(("Content-Type", "text/plain"),),
            chunks=(b"User-agent: *\nDisallow: /docs/private\n",),
        ),
    )
    crawler = client(dialer)

    with pytest.raises(CrawlerRobotsError, match="disallowed"):
        await crawler.run().get(page("private/secret"))

    assert [url for url, _, _ in dialer.requests] == [robots()]
    assert dialer.requests[0][1]["User-Agent"] == CRAWLER_USER_AGENT


async def test_robots_allow_precedence_cache_and_crawl_delay() -> None:
    clock = Clock()
    dialer = Dialer()
    dialer.add(
        robots(),
        Answer(
            headers=(("Content-Type", "text/plain"),),
            chunks=(
                b"User-agent: ManiculeCrawler\nDisallow: /docs/\n"
                b"Allow: /docs/public\nCrawl-delay: 3.0\n",
            ),
        ),
    )
    dialer.add(page("public"), Answer(), Answer())
    crawler = client(dialer, clock=clock)
    run = crawler.run()

    await run.get(page("public"))
    await run.get(page("public"))

    assert [url for url, _, _ in dialer.requests] == [
        robots(),
        page("public"),
        page("public"),
    ]
    assert [started for _, _, started in dialer.requests] == [0.0, 3.0, 6.0]
    assert clock.waits == [3.0, 3.0]


async def test_a_missing_robots_file_explicitly_allows_pages() -> None:
    dialer = Dialer()
    dialer.add(
        robots(),
        Answer(status=404, headers=(("Content-Type", "text/plain"),), chunks=()),
    )
    dialer.add(page(), Answer())

    assert (await client(dialer).run().get(page())).status == 200


async def test_robots_redirects_use_the_same_origin_address_and_header_policy() -> None:
    policy_url = "https://docs.example.test/policy.txt"
    dialer = Dialer()
    dialer.add(
        robots(), Answer(status=302, headers=(("Location", policy_url),), chunks=())
    )
    dialer.add(
        policy_url,
        Answer(headers=(("Content-Type", "text/plain"),), chunks=(b"User-agent: *\n",)),
    )
    dialer.add(page(), Answer())

    assert (await client(dialer).run().get(page())).status == 200
    assert [url for url, _, _ in dialer.requests][:2] == [robots(), policy_url]


async def test_every_page_redirect_is_checked_against_robots() -> None:
    dialer = Dialer()
    dialer.add(
        robots(),
        Answer(
            headers=(("Content-Type", "text/plain"),),
            chunks=(b"User-agent: *\nDisallow: /docs/private\n",),
        ),
    )
    dialer.add(
        page(), Answer(status=302, headers=(("Location", "/docs/private"),), chunks=())
    )

    with pytest.raises(CrawlerRobotsError, match="redirect"):
        await client(dialer).run().get(page())
    assert [url for url, _, _ in dialer.requests] == [robots(), page()]


async def test_a_robots_failure_does_not_silently_become_permission() -> None:
    dialer = Dialer()
    dialer.add(robots(), TimeoutError())
    configured = CrawlerHttpConfig(max_retries=0)

    with pytest.raises(CrawlerRobotsError, match="establish"):
        await client(dialer, config=configured).run().get(page())


async def test_robots_cache_is_bounded_by_origin_user_agent_and_ttl() -> None:
    clock = Clock()
    dialer = Dialer()
    allow = Answer(headers=(("Content-Type", "text/plain"),), chunks=(b"User-agent: *\n",))
    dialer.add(robots(), allow, allow)
    dialer.add(page(), Answer(), Answer())
    crawler = client(dialer, clock=clock, config=CrawlerHttpConfig(robots_ttl_s=5))

    await crawler.run().get(page())
    clock.now += 6
    await crawler.run().get(page())

    assert [url for url, _, _ in dialer.requests].count(robots()) == 2


async def test_retry_after_is_bounded_and_only_retryable_statuses_retry() -> None:
    clock = Clock()
    dialer = Dialer()
    dialer.add(
        page(),
        Answer(status=429, headers=(("Retry-After", "2"),), chunks=()),
        Answer(),
    )

    response = await client(dialer, clock=clock).run().get(page())
    assert response.status == 200
    assert clock.waits == [2.0]

    dated_clock = Clock()
    dated = Dialer()
    dated.add(
        page(),
        Answer(
            status=503,
            headers=(("Retry-After", "Mon, 24 Aug 2026 00:00:05 GMT"),),
            chunks=(),
        ),
        Answer(),
    )
    await client(dated, clock=dated_clock).run().get(page())
    assert dated_clock.waits == [5.0]

    malformed = Dialer()
    malformed.add(
        page(), Answer(status=503, headers=(("Retry-After", "later"),), chunks=())
    )
    with pytest.raises(CrawlerThrottleError, match="malformed"):
        await client(malformed).run().get(page())

    non_finite = Dialer()
    non_finite.add(
        page(), Answer(status=429, headers=(("Retry-After", "nan"),), chunks=())
    )
    with pytest.raises(CrawlerThrottleError, match="malformed"):
        await client(non_finite).run().get(page())

    excessive = Dialer()
    excessive.add(page(), Answer(status=429, headers=(("Retry-After", "31"),), chunks=()))
    with pytest.raises(CrawlerThrottleError, match="bound"):
        await client(excessive).run().get(page())


async def test_timeout_retries_are_bounded_and_aggregate_safe() -> None:
    dialer = Dialer()
    dialer.add(page(), TimeoutError("sentinel-private-detail"), TimeoutError("again"))
    with pytest.raises(CrawlerTimeoutError) as raised:
        await client(dialer, config=CrawlerHttpConfig(max_retries=1)).run().get(page())

    assert "sentinel" not in str(raised.value)
    assert [url for url, _, _ in dialer.requests].count(page()) == 2


async def test_wire_and_declared_size_limits_stop_streaming_and_close() -> None:
    dialer = Dialer()
    dialer.add(page(), Answer(chunks=(b"1234", b"5678")))
    configured = CrawlerHttpConfig(max_response_bytes=5)
    with pytest.raises(CrawlerResponseTooLargeError, match="wire"):
        await client(dialer, config=configured).run().get(page())
    assert dialer.streams[-1].reads == 2
    assert dialer.streams[-1].closed

    declared = Dialer()
    declared.add(
        page(),
        Answer(headers=(("Content-Type", "text/html"), ("Content-Length", "100"))),
    )
    with pytest.raises(CrawlerResponseTooLargeError, match="wire"):
        await client(declared, config=configured).run().get(page())
    assert declared.streams[-1].reads == 0
    assert declared.streams[-1].closed


async def test_decompression_bomb_is_stopped_before_parser_allocation() -> None:
    bomb = gzip.compress(b"x" * 10_000)
    dialer = Dialer()
    dialer.add(
        page(),
        Answer(
            headers=(
                ("Content-Type", "text/html"),
                ("Content-Encoding", "gzip"),
            ),
            chunks=(bomb,),
        ),
    )
    configured = CrawlerHttpConfig(
        max_response_bytes=len(bomb) + 1,
        max_decompressed_bytes=100,
    )

    with pytest.raises(CrawlerResponseTooLargeError, match="decoded"):
        await client(dialer, config=configured).run().get(page())
    assert dialer.streams[-1].closed


async def test_truncated_or_concatenated_compression_is_refused() -> None:
    compressed = gzip.compress(b"first")
    for body in (compressed[:-2], compressed + gzip.compress(b"hidden")):
        dialer = Dialer()
        dialer.add(
            page(),
            Answer(
                headers=(
                    ("Content-Type", "text/html"),
                    ("Content-Encoding", "gzip"),
                ),
                chunks=(body,),
            ),
        )
        with pytest.raises(CrawlerTransportError, match=r"truncated|concatenated"):
            await client(dialer).run().get(page())


async def test_total_run_bytes_are_enforced_across_pages() -> None:
    dialer = Dialer()
    dialer.add(page("a"), Answer(chunks=(b"123",)))
    dialer.add(page("b"), Answer(chunks=(b"456",)))
    run = client(dialer, config=CrawlerHttpConfig(max_run_bytes=5)).run()

    await run.get(page("a"))
    with pytest.raises(CrawlerResponseTooLargeError, match="run-wide"):
        await run.get(page("b"))


async def test_non_text_media_is_refused_before_body_streaming() -> None:
    dialer = Dialer()
    dialer.add(
        page(), Answer(headers=(("Content-Type", "application/pdf"),), chunks=(b"pdf",))
    )
    with pytest.raises(CrawlerMediaTypeError):
        await client(dialer).run().get(page())
    assert dialer.streams[-1].reads == 0
    assert dialer.streams[-1].closed


@pytest.mark.parametrize(
    "answer",
    [
        Answer(status=401, chunks=()),
        Answer(chunks=(b"<title>Sign in</title><input type=password>",)),
        Answer(chunks=(b"<title>404 Not Found</title>",)),
        Answer(chunks=(b"<div class='cf-chl-widget'>challenge</div>",)),
    ],
)
async def test_login_challenge_and_error_pages_are_never_content(answer: Answer) -> None:
    dialer = Dialer()
    dialer.add(page(), answer)
    with pytest.raises(CrawlerChallengeError):
        await client(dialer).run().get(page())


async def test_a_login_redirect_is_a_typed_challenge_not_indexable_content() -> None:
    dialer = Dialer()
    dialer.add(
        page(), Answer(status=302, headers=(("Location", "/login"),), chunks=())
    )
    with pytest.raises(CrawlerChallengeError, match="redirected"):
        await client(dialer).run().get(page())


async def test_redirects_revalidate_scope_peer_and_strip_cross_origin_secrets() -> None:
    cdn = "https://cdn.example.test/docs/a"
    dialer = Dialer()
    dialer.add(
        page(),
        Answer(status=302, headers=(("Location", cdn),), chunks=()),
    )
    dialer.add(cdn, Answer())
    crawler = client(
        dialer, origins=("https://docs.example.test", "https://cdn.example.test")
    )

    assert (await crawler.run().get(page())).url.url == cdn
    assert len(dialer.requests) == 4

    rebound = Dialer()
    rebound.add(page(), Answer(peer="127.0.0.1"))
    with pytest.raises(Exception, match="globally routable"):
        await client(rebound).run().get(page())


class BlockingDialer(Dialer):
    def __init__(self) -> None:
        super().__init__()
        self.entered = 0
        self.release = asyncio.Event()
        self.two_entered = asyncio.Event()

    @override
    async def open(
        self, plan: ConnectionPlan, *, method: str, headers: Mapping[str, str]
    ) -> DialResponse:
        self.entered += 1
        if self.entered == 2:
            self.two_entered.set()
        await self.release.wait()
        stream = Stream((b"page",))
        self.streams.append(stream)
        return DialResponse(
            status=200,
            headers=(("Content-Type", "text/html"),),
            peer="8.8.8.8",
            body=stream,
        )


async def test_global_and_per_origin_concurrency_are_both_enforced() -> None:
    same = BlockingDialer()
    same_client = client(
        same, config=CrawlerHttpConfig(concurrency=2, per_origin_concurrency=1)
    )
    tasks = [
        asyncio.create_task(same_client.run().get(page(name)))
        for name in ("a", "b")
    ]
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert same.entered == 1
    same.release.set()
    await asyncio.gather(*tasks)

    different = BlockingDialer()
    different_client = client(
        different,
        config=CrawlerHttpConfig(concurrency=2, per_origin_concurrency=1),
        origins=("https://docs.example.test", "https://other.example.test"),
    )
    tasks = [
        asyncio.create_task(
            different_client.run().get(url)
        )
        for url in (page("a"), "https://other.example.test/docs/b")
    ]
    await asyncio.wait_for(different.two_entered.wait(), timeout=1)
    different.release.set()
    await asyncio.gather(*tasks)


async def test_cancellation_closes_the_stream_and_releases_permits() -> None:
    gate = asyncio.Event()
    dialer = Dialer()
    dialer.add(page("a"), Answer(gate=gate))
    dialer.add(page("b"), Answer())
    crawler = client(dialer, config=CrawlerHttpConfig(concurrency=1, per_origin_concurrency=1))
    running = asyncio.create_task(crawler.run().get(page("a")))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert dialer.streams[0].closed
    assert (await crawler.run().get(page("b"))).status == 200
