"""Robots-aware, rate-limited and bounded transport orchestration for web crawling."""

from __future__ import annotations

import asyncio
import math
import re
import time
import zlib
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Protocol, cast
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

from manicule.connectors.crawler_security import (
    AddressResolver,
    ConnectionPlan,
    CrawlerUrlPolicy,
    NormalizedUrl,
    bounded_response_headers,
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

__all__ = [
    "CRAWLER_USER_AGENT",
    "CrawlerDialer",
    "CrawlerHttpClient",
    "CrawlerHttpConfig",
    "CrawlerResponse",
    "CrawlerRun",
    "DialResponse",
]

CRAWLER_USER_AGENT = "ManiculeCrawler/0.1 (+https://github.com/mgd43b/manicule)"
_REDIRECTS = frozenset({301, 302, 303, 307, 308})
_RETRYABLE = frozenset({429, 503})
_HTTP_OK = 200
_HTTP_SUCCESS_END = 300
_HTTP_NOT_FOUND = 404
_MAX_RETRIES = 10
_PAGE_MEDIA_TYPES = frozenset({"text/html", "application/xhtml+xml"})
_TEXT_MEDIA_TYPES = frozenset(
    {"text/html", "application/xhtml+xml", "application/xml", "text/xml", "text/plain"}
)
_LOGIN = re.compile(
    rb"(?:<input[^>]+type\s*=\s*['\"]?password|"
    rb"<title[^>]*>\s*(?:sign[ -]?in|log[ -]?in|access denied|just a moment)|"
    rb"(?:cf-chl-|captcha-container))",
    re.IGNORECASE,
)
_ERROR_TITLE = re.compile(
    rb"<title[^>]*>\s*(?:404|not found|server error|service unavailable)", re.IGNORECASE
)
_CHALLENGE_PATH_SEGMENTS = frozenset(
    {"auth", "captcha", "challenge", "login", "signin", "sso"}
)


class _BodyStream(Protocol):
    def __aiter__(self) -> AsyncIterator[bytes]: ...

    async def aclose(self) -> None: ...


class _Decoder(Protocol):
    def decompress(self, data: bytes) -> bytes: ...

    def flush(self) -> bytes: ...

    @property
    def eof(self) -> bool: ...

    @property
    def unused_data(self) -> bytes: ...


@dataclass(frozen=True, slots=True)
class DialResponse:
    """One response from a dialer that connected only to ``ConnectionPlan.addresses``."""

    status: int
    headers: tuple[tuple[str, str], ...]
    peer: str
    body: _BodyStream


class CrawlerDialer(Protocol):
    """The socket/TLS adapter; it must dial a numeric plan and verify its TLS server name."""

    async def open(
        self,
        plan: ConnectionPlan,
        *,
        method: str,
        headers: Mapping[str, str],
    ) -> DialResponse: ...


@dataclass(frozen=True, slots=True)
class CrawlerHttpConfig:
    """Operational bounds that do not affect page membership."""

    request_timeout_s: float = 30.0
    max_retries: int = 2
    max_retry_after_s: float = 30.0
    max_header_bytes: int = 64 * 1024
    max_response_bytes: int = 8 * 1024 * 1024
    max_decompressed_bytes: int = 16 * 1024 * 1024
    max_run_bytes: int = 256 * 1024 * 1024
    concurrency: int = 8
    per_origin_concurrency: int = 2
    request_delay_s: float = 0.0
    robots_ttl_s: float = 3600.0

    def __post_init__(self) -> None:
        positive = (
            self.request_timeout_s,
            self.max_retry_after_s,
            self.max_header_bytes,
            self.max_response_bytes,
            self.max_decompressed_bytes,
            self.max_run_bytes,
            self.concurrency,
            self.per_origin_concurrency,
            self.robots_ttl_s,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("crawler transport limits must be positive")
        if self.max_retries < 0 or self.max_retries > _MAX_RETRIES:
            raise ValueError("crawler max_retries must be between 0 and 10")
        if self.request_delay_s < 0:
            raise ValueError("crawler request_delay_s must not be negative")


@dataclass(frozen=True, slots=True)
class CrawlerResponse:
    """One fully bounded and decoded textual response."""

    url: NormalizedUrl
    status: int
    headers: Mapping[str, str]
    media_type: str
    body: bytes


@dataclass(slots=True)
class _OriginGate:
    concurrency: asyncio.Semaphore
    delay_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_started: float | None = None


@dataclass(frozen=True, slots=True)
class _RobotsRules:
    parser: RobotFileParser
    crawl_delay_s: float

    def allows(self, user_agent: str, url: str) -> bool:
        return self.parser.can_fetch(user_agent, url)


@dataclass(frozen=True, slots=True)
class _CachedRobots:
    expires_at: float
    rules: _RobotsRules


class _RunBudget:
    def __init__(self, maximum: int) -> None:
        self._maximum = maximum
        self._used = 0
        self._lock = asyncio.Lock()

    async def consume(self, amount: int) -> None:
        async with self._lock:
            if self._used + amount > self._maximum:
                raise CrawlerResponseTooLargeError(
                    "crawler responses exceed the configured run-wide byte limit"
                )
            self._used += amount


class CrawlerRun:
    """One sync's total-byte budget over a process-wide polite HTTP client."""

    def __init__(self, client: CrawlerHttpClient) -> None:
        self._client = client
        self._budget = _RunBudget(client.config.max_run_bytes)

    async def get(self, url: str) -> CrawlerResponse:
        target = self._client.policy.normalize(url)
        delay = self._client.config.request_delay_s
        rules = await self._client.robots_for_run(target, self._budget)
        if not rules.allows(CRAWLER_USER_AGENT, target.url):
            raise CrawlerRobotsError("crawler page is disallowed by robots policy")
        delay = max(delay, rules.crawl_delay_s)
        response = await self._client.request_bounded(
            target, self._budget, delay_s=delay, check_redirect_robots=True
        )
        return _usable_page(response)


class CrawlerHttpClient:
    """Apply URL, DNS, peer, redirect, robots and resource policy around one dialer."""

    def __init__(
        self,
        policy: CrawlerUrlPolicy,
        dialer: CrawlerDialer,
        resolver: AddressResolver,
        *,
        config: CrawlerHttpConfig | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.policy = policy
        self.config = config or CrawlerHttpConfig()
        self._dialer = dialer
        self._resolver = resolver
        self._sleep = sleep
        self._clock = clock
        self._wall_clock = wall_clock or (lambda: datetime.now(UTC))
        self._global = asyncio.Semaphore(self.config.concurrency)
        self._origins: dict[str, _OriginGate] = {}
        self._robots_cache: dict[tuple[str, str], _CachedRobots] = {}
        self._robots_locks: dict[str, asyncio.Lock] = {}

    def run(self) -> CrawlerRun:
        return CrawlerRun(self)

    def _gate(self, origin: str) -> _OriginGate:
        return self._origins.setdefault(
            origin, _OriginGate(asyncio.Semaphore(self.config.per_origin_concurrency))
        )

    async def robots_for_run(self, page: NormalizedUrl, budget: _RunBudget) -> _RobotsRules:
        key = (page.origin, CRAWLER_USER_AGENT)
        cached = self._robots_cache.get(key)
        if cached is not None and cached.expires_at > self._clock():
            return cached.rules
        lock = self._robots_locks.setdefault(page.origin, asyncio.Lock())
        async with lock:
            cached = self._robots_cache.get(key)
            if cached is not None and cached.expires_at > self._clock():
                return cached.rules
            robots = self.policy.normalize_auxiliary(f"{page.origin}/robots.txt")
            try:
                response = await self.request_bounded(
                    robots,
                    budget,
                    delay_s=self.config.request_delay_s,
                    allowed_media_types=_TEXT_MEDIA_TYPES,
                    enforce_path=False,
                    check_redirect_robots=False,
                )
            except CrawlerTransportError as exc:
                raise CrawlerRobotsError("crawler could not establish robots policy") from exc
            if response.status == _HTTP_NOT_FOUND:
                rules = _allow_all_robots()
            elif response.status != _HTTP_OK:
                raise CrawlerRobotsError("crawler robots endpoint returned an unusable status")
            else:
                rules = _parse_robots(response.body)
            self._robots_cache[key] = _CachedRobots(
                expires_at=self._clock() + self.config.robots_ttl_s,
                rules=rules,
            )
            return rules

    async def request_bounded(
        self,
        initial: NormalizedUrl,
        budget: _RunBudget,
        *,
        delay_s: float,
        allowed_media_types: frozenset[str] = _PAGE_MEDIA_TYPES,
        enforce_path: bool = True,
        check_redirect_robots: bool,
    ) -> CrawlerResponse:
        target = initial
        headers: dict[str, str] = {
            "Accept": ", ".join(sorted(allowed_media_types)),
            "User-Agent": CRAWLER_USER_AGENT,
        }
        for hop in range(self.policy.max_redirects + 1):
            response = await self._attempt(
                target,
                budget,
                headers=headers,
                delay_s=delay_s,
                allowed_media_types=allowed_media_types,
            )
            if response.status not in _REDIRECTS:
                return response
            location = response.headers.get("location")
            if not location:
                raise CrawlerTransportError("crawler redirect has no usable location")
            if _challenge_location(location):
                raise CrawlerChallengeError(
                    "crawler source redirected to a login or challenge endpoint"
                )
            decision = self.policy.redirect(
                target.url, location, hop=hop + 1, enforce_path=enforce_path
            )
            headers = self.policy.redirected_headers(headers, decision)
            target = decision.value
            if check_redirect_robots:
                rules = await self.robots_for_run(target, budget)
                if not rules.allows(CRAWLER_USER_AGENT, target.url):
                    raise CrawlerRobotsError(
                        "crawler redirect target is disallowed by robots policy"
                    )
                delay_s = max(self.config.request_delay_s, rules.crawl_delay_s)
        raise CrawlerTransportError("crawler redirect chain did not terminate")

    async def _attempt(
        self,
        target: NormalizedUrl,
        budget: _RunBudget,
        *,
        headers: Mapping[str, str],
        delay_s: float,
        allowed_media_types: frozenset[str],
    ) -> CrawlerResponse:
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self._attempt_once(
                    target,
                    budget,
                    headers=headers,
                    delay_s=delay_s,
                    allowed_media_types=allowed_media_types,
                )
            except (OSError, TimeoutError) as exc:
                if attempt >= self.config.max_retries:
                    raise CrawlerTimeoutError(
                        "crawler request exhausted its timeout and retry budget"
                    ) from exc
                await self._sleep(min(2.0**attempt, self.config.max_retry_after_s))
                continue
            if response.status not in _RETRYABLE:
                return response
            if attempt >= self.config.max_retries:
                raise CrawlerThrottleError("crawler source remained unavailable after retries")
            delay = _retry_after(response.headers, now=self._wall_clock())
            if delay is None:
                delay = min(2.0**attempt, self.config.max_retry_after_s)
            if delay > self.config.max_retry_after_s:
                raise CrawlerThrottleError("crawler Retry-After exceeds the configured bound")
            await self._sleep(delay)
        raise AssertionError("bounded crawler retry loop did not return")

    async def _attempt_once(
        self,
        target: NormalizedUrl,
        budget: _RunBudget,
        *,
        headers: Mapping[str, str],
        delay_s: float,
        allowed_media_types: frozenset[str],
    ) -> CrawlerResponse:
        gate = self._gate(target.origin)
        async with asyncio.timeout(self.config.request_timeout_s):
            plan = await self.policy.connection_plan(target, self._resolver)
            async with self._global, gate.concurrency:
                await self._wait_turn(gate, delay_s)
                response: DialResponse | None = None
                try:
                    response = await self._dialer.open(plan, method="GET", headers=headers)
                    plan.authorize_peer(response.peer)
                    normalized_headers = bounded_response_headers(
                        response.headers, max_bytes=self.config.max_header_bytes
                    )
                    media_type = (
                        normalized_headers.get("content-type", "")
                        .split(";", 1)[0]
                        .strip()
                        .lower()
                    )
                    if media_type and media_type not in allowed_media_types:
                        raise CrawlerMediaTypeError(
                            "crawler response media type is not allowed"
                        )
                    body = await _read_body(
                        response.body,
                        normalized_headers,
                        budget,
                        max_wire_bytes=self.config.max_response_bytes,
                        max_decoded_bytes=self.config.max_decompressed_bytes,
                    )
                finally:
                    if response is not None:
                        await response.body.aclose()
        if response is None:  # pragma: no cover - dial success assigns before every return
            raise AssertionError("crawler dialer returned no response")
        if body and media_type not in allowed_media_types:
            raise CrawlerMediaTypeError("crawler response media type is not allowed")
        return CrawlerResponse(
            url=target,
            status=response.status,
            headers=normalized_headers,
            media_type=media_type,
            body=body,
        )

    async def _wait_turn(self, gate: _OriginGate, delay_s: float) -> None:
        async with gate.delay_lock:
            now = self._clock()
            if gate.last_started is not None:
                wait = gate.last_started + delay_s - now
                if wait > 0:
                    await self._sleep(wait)
            gate.last_started = self._clock()


async def _read_body(  # noqa: PLR0912 - each branch is one independently enforced bound
    stream: _BodyStream,
    headers: Mapping[str, str],
    budget: _RunBudget,
    *,
    max_wire_bytes: int,
    max_decoded_bytes: int,
) -> bytes:
    declared = headers.get("content-length")
    if declared is not None:
        try:
            length = int(declared)
        except ValueError as exc:
            raise CrawlerTransportError("crawler response has an invalid content length") from exc
        if length < 0 or length > max_wire_bytes:
            raise CrawlerResponseTooLargeError(
                "crawler response exceeds the configured wire-byte limit"
            )
    encoding = headers.get("content-encoding", "identity").lower()
    if encoding in {"", "identity"}:
        decoder: _Decoder | None = None
    elif encoding == "gzip":
        decoder = cast("_Decoder", zlib.decompressobj(16 + zlib.MAX_WBITS))
    elif encoding == "deflate":
        decoder = cast("_Decoder", zlib.decompressobj())
    else:
        raise CrawlerTransportError("crawler response uses an unsupported content encoding")
    wire = 0
    decoded = 0
    parts: list[bytes] = []
    async for chunk in stream:
        wire += len(chunk)
        if wire > max_wire_bytes:
            raise CrawlerResponseTooLargeError(
                "crawler response exceeds the configured wire-byte limit"
            )
        await budget.consume(len(chunk))
        try:
            output = chunk if decoder is None else decoder.decompress(chunk)
        except zlib.error as exc:
            raise CrawlerTransportError("crawler response compression is malformed") from exc
        decoded += len(output)
        if decoded > max_decoded_bytes:
            raise CrawlerResponseTooLargeError(
                "crawler response exceeds the configured decoded-byte limit"
            )
        parts.append(output)
    if decoder is not None:
        try:
            output = decoder.flush()
        except zlib.error as exc:
            raise CrawlerTransportError("crawler response compression is malformed") from exc
        decoded += len(output)
        if decoded > max_decoded_bytes:
            raise CrawlerResponseTooLargeError(
                "crawler response exceeds the configured decoded-byte limit"
            )
        parts.append(output)
        if not decoder.eof or decoder.unused_data:
            raise CrawlerTransportError(
                "crawler response compression is truncated or ambiguously concatenated"
            )
    return b"".join(parts)


def _retry_after(headers: Mapping[str, str], *, now: datetime) -> float | None:
    raw = headers.get("retry-after")
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        raise CrawlerThrottleError("crawler Retry-After is malformed")
    try:
        delay = float(value)
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError) as exc:
            raise CrawlerThrottleError("crawler Retry-After is malformed") from exc
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        delay = (when - now).total_seconds()
    if not math.isfinite(delay):
        raise CrawlerThrottleError("crawler Retry-After is malformed")
    if delay < 0:
        return 0.0
    return delay


def _parse_robots(body: bytes) -> _RobotsRules:
    try:
        text = body.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise CrawlerRobotsError("crawler robots policy is not valid UTF-8") from exc
    parser = RobotFileParser()
    parser.parse(text.splitlines())
    return _RobotsRules(
        parser=parser,
        crawl_delay_s=_robots_crawl_delay(text, CRAWLER_USER_AGENT),
    )


def _allow_all_robots() -> _RobotsRules:
    parser = RobotFileParser()
    parser.parse(("User-agent: *", "Allow: /"))
    return _RobotsRules(parser=parser, crawl_delay_s=0.0)


def _usable_page(response: CrawlerResponse) -> CrawlerResponse:
    if response.status in {401, 403}:
        raise CrawlerChallengeError("crawler source returned an authentication challenge")
    if not _HTTP_OK <= response.status < _HTTP_SUCCESS_END:
        raise CrawlerTransportError("crawler source returned an unusable status")
    if _LOGIN.search(response.body) or _ERROR_TITLE.search(response.body):
        raise CrawlerChallengeError("crawler source returned a login, challenge, or error page")
    return response


def _challenge_location(value: str) -> bool:
    path = urlsplit(value).path.lower()
    return bool(_CHALLENGE_PATH_SEGMENTS.intersection(path.split("/")))


def _robots_crawl_delay(text: str, user_agent: str) -> float:
    """Read a finite non-negative integer or decimal delay from the best matching group."""
    product = user_agent.split("/", 1)[0].lower()
    groups: list[tuple[list[str], list[str]]] = []
    agents: list[str] = []
    delays: list[str] = []
    saw_rule = False
    for raw in (*text.splitlines(), ""):
        line = raw.split("#", 1)[0].strip()
        if not line:
            if agents:
                groups.append((agents, delays))
            agents, delays, saw_rule = [], [], False
            continue
        name, separator, value = line.partition(":")
        if not separator:
            continue
        field_name = name.strip().lower()
        field_value = value.strip()
        if field_name == "user-agent":
            if saw_rule and agents:
                groups.append((agents, delays))
                agents, delays, saw_rule = [], [], False
            agents.append(field_value.lower())
            continue
        saw_rule = True
        if field_name == "crawl-delay":
            delays.append(field_value)
    candidates = [
        (2 if product in group_agents else 1, group_delays)
        for group_agents, group_delays in groups
        if product in group_agents or "*" in group_agents
    ]
    for _, declared in sorted(candidates, reverse=True):
        for value in declared:
            try:
                delay = float(value)
            except ValueError:
                continue
            if math.isfinite(delay) and delay >= 0:
                return delay
    return 0.0
