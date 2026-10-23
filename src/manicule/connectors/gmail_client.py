"""The Gmail REST client: the calls the connector needs, paced against Gmail's quota.

``docs/connectors/gmail.md`` §4-§6 is the design; this is the HTTP half. Six calls — ``getProfile``,
``labels.list``, ``messages.list``, ``messages.get`` (raw and minimal) and ``history.list`` — each
typed on failure, none of them reaching for a Google client library. The HTTP stack is the one the
Confluence client uses, imported inside :meth:`GmailClient.setup` so that registering the
connector, which runs in every process that starts, costs no HTTP stack.

**Paced, not throttled into.** Gmail meters quota in *units* per user per minute, and a raw fetch
costs 60 of the 6,000 — about a hundred messages a minute, which makes pacing the thing a first
sync is bound by rather than a courtesy. :class:`QuotaPacer` spends against a configurable budget
and waits *before* a request that would exceed it. A 429 is still answered (``Retry-After`` in both
its forms, a ceiling on how long one pause may be) but it is the exception: a client that finds its
limit by hitting it spends most of a first sync being told no.

**The credential is an interface, not a class.** :class:`GmailCredential` is the whole of what the
client needs from whatever mints access tokens: a token, a forced refresh after a 401, and a
sentence for a person to act on when the refresh does not help. Minting a token is itself an HTTP
call, so the method is a coroutine and the refresh logic lives with the credential.

**Errors name the operation and nothing else.** The design forbids an address, a label name or a
message id in any log or error (§10), and an HTTP library's own messages carry the URL — which for
``messages.get`` contains the message id. So every message here is built from the operation's name
(``messages.get``), a status code and, for a 403, Google's short ``reason`` token; the library's
exception is deliberately not chained.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import re
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Final, Literal, Protocol, cast, override
from urllib.parse import quote

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
from manicule.connectors.retry import backoff, retry_after
from manicule.core.errors import ConfigError

if TYPE_CHECKING:  # pragma: no cover - import-time only
    import httpx

__all__ = [
    "API_BASE",
    "COST_GET_MINIMAL",
    "COST_GET_RAW",
    "COST_HISTORY",
    "COST_LABELS",
    "COST_LIST",
    "COST_PROFILE",
    "WINDOW_SECONDS",
    "GmailClient",
    "GmailCredential",
    "GmailSettings",
    "HistoryChange",
    "HistoryKind",
    "HistoryPage",
    "Label",
    "MessageLabels",
    "Profile",
    "QuotaPacer",
]

_ORIGIN: Final = "https://gmail.googleapis.com"

API_BASE: Final = f"{_ORIGIN}/gmail/v1/users/me"
"""Every call is made as the authenticated user (``me``); there is no setting for another."""

PAGE_SIZE: Final = 500
"""The most ids a ``messages.list`` or ``history.list`` page returns, and so the cheapest walk."""

WINDOW_SECONDS: Final = 60.0
"""The span Gmail's per-user quota is counted over."""

COST_PROFILE: Final = 1
COST_LABELS: Final = 1
COST_LIST: Final = 5
"""Per request, however many ids (up to :data:`PAGE_SIZE`) it returns."""
COST_HISTORY: Final = 2
COST_GET_MINIMAL: Final = 20
COST_GET_RAW: Final = 60
"""Quota units each call spends (``docs/connectors/gmail.md`` §6)."""

_RATE_LIMIT_REASONS: Final = frozenset({"ratelimitexceeded", "userratelimitexceeded"})
"""403 reasons that mean "slow down" rather than "no". Compared case- and underscore-blind."""

_SAFE_REASON: Final = re.compile(r"[A-Za-z_]{1,64}")
"""What a ``reason`` must look like to be quoted: a token, never free text naming a label."""

_HTTPX_LOGGER: Final = "httpx"


class _RedactGmailUrls(logging.Filter):
    """Keeps a Gmail request's path and query out of the HTTP stack's own log records.

    ``httpx`` logs every request it sends at INFO, URL included, and for ``messages.get`` the URL
    holds the message id; its query holds page tokens, history positions and label ids. An
    application that turns that logger up would then write what §10 forbids, whatever this
    client does with its own exceptions. Only records naming Gmail's host are touched, so no
    other client's logging changes.
    """

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(
                f"{_ORIGIN}/[redacted]" if str(argument).startswith(_ORIGIN) else argument
                for argument in record.args
            )
        return True


_REDACT_GMAIL_URLS: Final = _RedactGmailUrls()

type Params = Sequence[tuple[str, str]]
type Json = Mapping[str, object]
type HistoryKind = Literal["messageAdded", "messageDeleted", "labelAdded", "labelRemoved"]

_HISTORY_KEYS: Final[tuple[tuple[str, HistoryKind], ...]] = (
    ("messagesAdded", "messageAdded"),
    ("messagesDeleted", "messageDeleted"),
    ("labelsAdded", "labelAdded"),
    ("labelsRemoved", "labelRemoved"),
)


class GmailSettings(Protocol):
    """The four settings the client reads. The Gmail configuration model satisfies it as is.

    Properties rather than attributes so a frozen model satisfies it as readily as a mutable one.
    """

    @property
    def quota_units_per_minute(self) -> int:
        """The budget :class:`QuotaPacer` spends against, in quota units."""
        ...

    @property
    def max_retry_after_seconds(self) -> float:
        """The longest single pause a 429 may ask for before the run stops instead."""
        ...

    @property
    def max_retries(self) -> int:
        """How many times one request is retried after a retryable failure."""
        ...

    @property
    def request_timeout_seconds(self) -> float:
        """How long one request may take."""
        ...


class GmailCredential(Protocol):
    """Whatever mints the access tokens. Consulted per request; may refuse."""

    async def access_token(self, *, refresh: bool = False) -> str:
        """The bearer token for the request about to be made.

        ``refresh=False`` is the cached token, minted if there is none or it is about to expire.
        ``refresh=True`` is asked after a 401 and bypasses the cache. A refresh the issuer
        refuses (``invalid_grant``) raises the credential's own error from here.
        """
        ...

    def renewal(self) -> str:
        """What a person does to replace the credential. One sentence, imperative."""
        ...


@dataclass(frozen=True, slots=True)
class Profile:
    """``users.getProfile``: who the token is for, and where the mailbox's history stands."""

    email_address: str
    history_id: str


@dataclass(frozen=True, slots=True)
class Label:
    """One entry of ``users.labels.list``."""

    id: str
    name: str


@dataclass(frozen=True, slots=True)
class MessageLabels:
    """``users.messages.get?format=minimal``: a message's *current* label ids."""

    message_id: str
    label_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HistoryChange:
    """One change a history record reports about one message."""

    kind: HistoryKind
    message_id: str

    label_ids: tuple[str, ...] | None
    """The message's labels if the record carried them, else ``None``.

    Records "typically" carry only message and thread ids, and a ``labelRemoved`` record names
    the labels *removed*, not those that remain — so ``None`` means "ask ``messages.get``", and
    a value is not to be trusted as the complete set without checking (§4.2).
    """

    changed_label_ids: tuple[str, ...]
    """For ``labelAdded`` / ``labelRemoved``, the labels the record says were added or removed."""


@dataclass(frozen=True, slots=True)
class HistoryPage:
    """One page of ``users.history.list``: the unit of a discovery batch (§4.4)."""

    changes: tuple[HistoryChange, ...]

    history_id: str
    """The mailbox's position when this page was answered. The first page's becomes the next
    watermark if the walk completes (§4.2)."""


class QuotaPacer:
    """Spends quota units against a per-minute budget, waiting before it would overspend.

    A rolling window rather than a calendar minute: the units spent in *any* sixty seconds never
    exceed the budget, which is what Gmail counts and what a bucket that resets on the minute
    would break at its seam. Every attempt is charged, retries included, because a request that
    failed was still sent.

    Callers are served one at a time, in order. The lock is held while waiting, so a large
    request is not starved by a stream of small ones behind it.
    """

    def __init__(
        self,
        units_per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if units_per_minute < COST_GET_RAW:
            msg = (
                f"quota_units_per_minute is {units_per_minute}, below the {COST_GET_RAW} units "
                f"one raw message fetch costs; no message could ever be fetched"
            )
            raise ConfigError(msg)
        self._budget = units_per_minute
        self._clock = clock
        self._sleep = sleep
        self._spent: deque[tuple[float, int]] = deque()
        self._total = 0
        self._lock = asyncio.Lock()

    async def acquire(self, units: int) -> None:
        """Return once ``units`` may be spent without exceeding the budget; they are then spent."""
        async with self._lock:
            while True:
                now = self._clock()
                self._expire(now)
                if self._total + units <= self._budget:
                    self._spent.append((now, units))
                    self._total += units
                    return
                # The window frees up when the oldest charge ages out; wait for exactly that.
                await self._sleep(max(self._spent[0][0] + WINDOW_SECONDS - now, 0.0))

    def _expire(self, now: float) -> None:
        while self._spent and now - self._spent[0][0] >= WINDOW_SECONDS:
            self._total -= self._spent.popleft()[1]


class GmailClient:
    """The six calls, paced, retried and typed.

    Constructed with a transport in the suites, so the real client runs against a synthetic
    Gmail (``tests/connectors/fake_gmail.py``): the real pacer, the real retry loop, the real
    query encoding. Call :meth:`setup` before the first request and :meth:`teardown` after.
    """

    def __init__(
        self,
        settings: GmailSettings,
        credential: GmailCredential,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Args:
        settings: Budget, retry ceiling, retry count and timeout.
        credential: Where access tokens come from.
        transport: An ``httpx`` transport; ``None`` uses the network.
        sleep: How to wait, for the pacer and for retries. A test passes one that advances
            ``clock`` so a simulated minute takes no time.
        clock: A monotonic clock, read by the pacer.
        """
        self._settings = settings
        self._credential = credential
        self._transport = transport
        self._sleep = sleep if sleep is not None else asyncio.sleep
        self._pacer = QuotaPacer(settings.quota_units_per_minute, clock=clock, sleep=self._sleep)
        self._client: httpx.AsyncClient | None = None
        self.requests = 0
        """Requests sent, retries included."""
        self.throttled = 0
        """Times Gmail answered "slow down", whichever way it spelled it."""

    async def setup(self) -> None:
        """Build the HTTP client. Idempotent."""
        if self._client is not None:
            return
        # Imported here, not at module level, so that importing this module — which the factory
        # does and plugin discovery does not — is the only thing that pulls in the HTTP stack.
        import httpx  # noqa: PLC0415 - see the module docstring

        # Idempotent: a filter already on the logger is not added twice.
        logging.getLogger(_HTTPX_LOGGER).addFilter(_REDACT_GMAIL_URLS)
        self._client = httpx.AsyncClient(
            transport=self._transport,
            headers={"Accept": "application/json"},
            timeout=self._settings.request_timeout_seconds,
            follow_redirects=False,
        )

    async def teardown(self) -> None:
        """Close the HTTP client."""
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    async def get_profile(self) -> Profile:
        """``users.getProfile``: the mailbox's address and its current ``historyId``."""
        payload = await self._get("users.getProfile", "/profile", (), COST_PROFILE)
        return Profile(
            email_address=_text(payload, "emailAddress", "users.getProfile"),
            history_id=_text(payload, "historyId", "users.getProfile"),
        )

    async def list_labels(self) -> tuple[Label, ...]:
        """``users.labels.list``: every label's id and name."""
        payload = await self._get("users.labels.list", "/labels", (), COST_LABELS)
        return tuple(
            Label(
                id=_text(entry, "id", "users.labels.list"),
                name=_text(entry, "name", "users.labels.list"),
            )
            for entry in _entries(payload, "labels", "users.labels.list")
        )

    async def list_messages(self, label_ids: Sequence[str] = ()) -> AsyncIterator[tuple[str, ...]]:
        """``users.messages.list``, one tuple of ids per page.

        Never sends a search query: scope is never translated into one (§3). ``label_ids``
        narrows to messages carrying all of them. Spam and Trash are always left out.
        """
        params: list[tuple[str, str]] = [
            ("maxResults", str(PAGE_SIZE)),
            ("includeSpamTrash", "false"),
            *(("labelIds", label) for label in label_ids),
        ]
        async for payload in self._pages("users.messages.list", "/messages", params, COST_LIST):
            yield tuple(
                _text(entry, "id", "users.messages.list")
                for entry in _entries(payload, "messages", "users.messages.list")
            )

    async def get_raw(self, message_id: str) -> bytes:
        """``users.messages.get?format=raw``, base64url-decoded to the RFC 5322 message.

        Raises:
            NotFoundError: The message was deleted between discovery and fetch.
        """
        operation = "users.messages.get"
        _require_id(message_id, operation)
        payload = await self._get(
            operation,
            f"/messages/{quote(message_id, safe='')}",
            (("format", "raw"),),
            COST_GET_RAW,
            missing=lambda: NotFoundError(f"Gmail {operation} found no such message"),
        )
        encoded = _text(payload, "raw", operation)
        try:
            # Gmail omits the base64 padding, which the decoder insists on. ``validate`` is what
            # makes a stray character an error: left off, the decoder discards it and returns
            # a message with bytes missing, which would be parsed, cited and answered from.
            return base64.b64decode(
                encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True
            )
        except (binascii.Error, ValueError):
            msg = f"Gmail {operation} returned a body that is not base64url"
            raise ConnectorError(msg) from None

    async def get_minimal(self, message_id: str) -> MessageLabels:
        """``users.messages.get?format=minimal``: the message's current label ids.

        Raises:
            NotFoundError: The message no longer exists.
        """
        operation = "users.messages.get"
        _require_id(message_id, operation)
        payload = await self._get(
            operation,
            f"/messages/{quote(message_id, safe='')}",
            (("format", "minimal"),),
            COST_GET_MINIMAL,
            missing=lambda: NotFoundError(f"Gmail {operation} found no such message"),
        )
        return MessageLabels(
            message_id=_text(payload, "id", operation),
            label_ids=_strings(payload.get("labelIds"), operation),
        )

    async def history(self, start_history_id: str) -> AsyncIterator[HistoryPage]:
        """``users.history.list`` from ``start_history_id``, one page at a time.

        Asks for every record type §4.2 reads; what to do with each is the connector's call.

        Raises:
            CursorExpiredError: Gmail no longer holds history that far back (404).
        """
        operation = "users.history.list"
        params: list[tuple[str, str]] = [
            ("startHistoryId", start_history_id),
            ("maxResults", str(PAGE_SIZE)),
            *(("historyTypes", kind) for _, kind in _HISTORY_KEYS),
        ]
        async for payload in self._pages(
            operation,
            "/history",
            params,
            COST_HISTORY,
            missing=lambda: CursorExpiredError(
                f"Gmail {operation} no longer holds history from the stored position; "
                f"a complete walk is needed"
            ),
        ):
            yield HistoryPage(
                changes=tuple(_changes(payload, operation)),
                history_id=_text(payload, "historyId", operation),
            )

    async def _pages(
        self,
        operation: str,
        path: str,
        params: Params,
        cost: int,
        *,
        missing: Callable[[], ConnectorError] | None = None,
    ) -> AsyncIterator[Json]:
        """Yield each page of a ``nextPageToken``-paginated response.

        A token seen twice is a loop, and a loop over a paginated listing reads as a very large
        mailbox, so it is refused rather than followed.
        """
        # Tokens are short and a walk is hundreds of pages, so a set is enough; the disk-backed
        # guard the Confluence client uses exists for walks whose size the source chooses.
        seen: set[str] = set()
        token: str | None = None
        while True:
            query = [*params, ("pageToken", token)] if token is not None else params
            payload = await self._get(operation, path, query, cost, missing=missing)
            yield payload
            following = payload.get("nextPageToken")
            if not following:
                return
            if not isinstance(following, str):
                msg = f"Gmail {operation} returned a page token that is not text"
                raise ConnectorError(msg)
            if following in seen:
                msg = (
                    f"Gmail {operation} sent pagination back to a page it had already "
                    f"followed; continuing would repeat results forever"
                )
                raise ConnectorError(msg)
            seen.add(following)
            token = following

    async def _get(
        self,
        operation: str,
        path: str,
        params: Params,
        cost: int,
        *,
        missing: Callable[[], ConnectorError] | None = None,
    ) -> Json:
        """One paced, retried GET, returned as a JSON object.

        ``missing`` is what a 404 means *for this call*: a stale history position for
        ``history.list``, a deleted message for ``messages.get``. A 404 anywhere else is an
        ordinary unusable answer.
        """
        client = self._require_client()
        import httpx  # noqa: PLC0415 - see setup()

        attempt = 0
        refresh = False
        refreshed = False
        while True:
            # The token first: asking for one can be an HTTP call of its own (a refresh), and a
            # charge made before it would be a second old by the time the request left, so two
            # requests could land inside one window with the budget spent twice.
            token = await self._credential.access_token(refresh=refresh)
            refresh = False
            await self._pacer.acquire(cost)
            self.requests += 1
            try:
                response = await client.get(
                    API_BASE + path,
                    params=list(params),
                    headers={"Authorization": f"Bearer {token}"},
                )
            except httpx.TransportError as exc:
                await self._sleep(self._transport_wait(operation, exc, attempt))
                attempt += 1
                continue

            if response.status_code == HTTPStatus.UNAUTHORIZED:
                # One refresh, one retry, and the retry does not spend the failure budget: a
                # token that expired mid-run is ordinary, a second rejection is not.
                if refreshed:
                    msg = (
                        f"Gmail rejected the access token for {operation} even after it was "
                        f"refreshed. {self._credential.renewal()}"
                    )
                    raise SessionExpiredError(msg)
                refresh = refreshed = True
                continue
            wait = self._judge(operation, response, attempt, missing)
            if wait is None:
                return _object(response, operation)
            await self._sleep(wait)
            attempt += 1

    def _transport_wait(self, operation: str, exc: Exception, attempt: int) -> float:
        """How long to wait after a transport failure, or the error that ends the run instead."""
        import httpx  # noqa: PLC0415 - see setup()

        if attempt < self._settings.max_retries:
            return backoff(attempt)
        name = type(exc).__name__
        if isinstance(exc, httpx.TimeoutException):
            msg = f"Gmail {operation} timed out ({name})"
            raise RequestTimeoutError(msg) from None
        msg = f"could not reach Gmail for {operation} ({name})"
        raise ConnectorError(msg) from None

    def _judge(
        self,
        operation: str,
        response: httpx.Response,
        attempt: int,
        missing: Callable[[], ConnectorError] | None,
    ) -> float | None:
        """Type a non-401 answer: ``None`` if it is usable, else the pause before retrying.

        Anything that should not be retried raises here, with the error type §6 assigns it.
        """
        status = response.status_code
        if status == HTTPStatus.TOO_MANY_REQUESTS or (
            status == HTTPStatus.FORBIDDEN and _rate_limited(response)
        ):
            return self._throttle_wait(operation, response.headers, attempt)
        if status >= HTTPStatus.INTERNAL_SERVER_ERROR:
            if attempt >= self._settings.max_retries:
                msg = f"Gmail {operation} failed with HTTP {status}"
                raise RemoteError(msg, status_code=status)
            return backoff(attempt)
        if status == HTTPStatus.NOT_FOUND and missing is not None:
            raise missing()
        if status == HTTPStatus.FORBIDDEN:
            reasons = _reasons(response)
            detail = f" ({reasons[0]})" if reasons and _SAFE_REASON.fullmatch(reasons[0]) else ""
            msg = f"Gmail refused {operation} with HTTP 403{detail}"
            raise PermissionDeniedError(msg)
        if not HTTPStatus.OK <= status < HTTPStatus.MULTIPLE_CHOICES:
            msg = f"Gmail answered {operation} with HTTP {status}"
            raise RemoteError(msg, status_code=status)
        return None

    def _throttle_wait(self, operation: str, headers: Mapping[str, str], attempt: int) -> float:
        """How long to wait after "slow down", or the error that ends the run instead."""
        self.throttled += 1
        requested = retry_after(headers)
        wait = requested if requested is not None else backoff(attempt)
        ceiling = self._settings.max_retry_after_seconds
        if wait > ceiling:
            msg = (
                f"Gmail asked for a {wait:.0f}s pause on {operation}, over the "
                f"{ceiling:.0f}s ceiling (max_retry_after_seconds). Stopping rather than "
                f"appearing to run while waiting."
            )
            raise RateLimitedError(msg, retry_after=wait)
        if attempt >= self._settings.max_retries:
            msg = f"Gmail was still limiting {operation} after {attempt + 1} attempts"
            raise RateLimitedError(msg, retry_after=requested)
        return wait

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            msg = "GmailClient.setup() was not called before the first request"
            raise ConnectorError(msg)
        return self._client


def _require_id(message_id: str, operation: str) -> None:
    """Refuse an empty id: ``/messages/`` is the listing endpoint, and would answer as one."""
    if not message_id:
        msg = f"Gmail {operation} was asked for a message with no id"
        raise ConnectorError(msg)


def _mapping(value: object) -> Json | None:
    """``value`` as a JSON object, or ``None`` if it is anything else."""
    return cast("Json", value) if isinstance(value, dict) else None


def _sequence(value: object) -> list[object] | None:
    """``value`` as a JSON array, or ``None`` if it is anything else."""
    return cast("list[object]", value) if isinstance(value, list) else None


def _reasons(response: httpx.Response) -> list[str]:
    """The ``reason`` tokens of an error answer, in the order Google gave them.

    Read from both places Google puts one — ``error.errors[].reason`` (``rateLimitExceeded``)
    and ``error.details[].reason`` (``RATE_LIMIT_EXCEEDED``) — because which a given answer
    uses is one of the things not yet verified against a live mailbox (§11).
    """
    try:
        body: object = response.json()
    except ValueError:
        return []
    outer = _mapping(body)
    error = _mapping(outer.get("error")) if outer is not None else None
    if error is None:
        return []
    found: list[str] = []
    for key in ("errors", "details"):
        for entry in _sequence(error.get(key)) or []:
            reason = _mapping(entry)
            if reason is not None and isinstance(reason.get("reason"), str):
                found.append(str(reason["reason"]))
    return found


def _rate_limited(response: httpx.Response) -> bool:
    """Whether a 403 is Gmail saying "slow down" rather than "no"."""
    return any(
        reason.replace("_", "").lower() in _RATE_LIMIT_REASONS for reason in _reasons(response)
    )


def _object(response: httpx.Response, operation: str) -> Json:
    try:
        body: object = response.json()
    except ValueError:
        msg = f"Gmail {operation} returned a body that is not JSON"
        raise ConnectorError(msg) from None
    payload = _mapping(body)
    if payload is None:
        msg = f"Gmail {operation} returned JSON that is not an object"
        raise ConnectorError(msg)
    return payload


def _text(payload: Json, key: str, operation: str) -> str:
    """A required string field. Numbers are accepted as text: ``historyId`` is one by contract."""
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, str | int) or value == "":
        msg = f"Gmail {operation} returned no usable {key!r}"
        raise ConnectorError(msg)
    return str(value)


def _entries(payload: Json, key: str, operation: str) -> list[Json]:
    """A list-of-objects field, which Gmail omits entirely when it would be empty."""
    value = payload.get(key)
    if value is None:
        return []
    items = _sequence(value)
    entries = [_mapping(item) for item in items] if items is not None else None
    if entries is None or any(entry is None for entry in entries):
        msg = f"Gmail {operation} returned {key!r} in an unexpected shape"
        raise ConnectorError(msg)
    return [entry for entry in entries if entry is not None]


def _strings(value: object, operation: str) -> tuple[str, ...]:
    if value is None:
        return ()
    items = _sequence(value)
    if items is None or not all(isinstance(item, str) for item in items):
        msg = f"Gmail {operation} returned label ids in an unexpected shape"
        raise ConnectorError(msg)
    return tuple(str(item) for item in items)


def _changes(payload: Json, operation: str) -> list[HistoryChange]:
    """Flatten a page's history records into one change per message per record."""
    changes: list[HistoryChange] = []
    for record in _entries(payload, "history", operation):
        for key, kind in _HISTORY_KEYS:
            for entry in _entries(record, key, operation):
                message = _mapping(entry.get("message"))
                if message is None:
                    msg = f"Gmail {operation} returned a history entry without a message"
                    raise ConnectorError(msg)
                carried = message.get("labelIds")
                changes.append(
                    HistoryChange(
                        kind=kind,
                        message_id=_text(message, "id", operation),
                        label_ids=None if carried is None else _strings(carried, operation),
                        changed_label_ids=_strings(entry.get("labelIds"), operation),
                    )
                )
    return changes
