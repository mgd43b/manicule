"""The Microsoft Graph client: the calls the Outlook connector needs, typed on failure.

``docs/connectors/outlook.md`` §4, §5, §7 and §8 are the design; this is the HTTP half. Mail
folders, per-folder message delta, a plain id listing, a message's MIME, and the calendar view and
one event. Each is typed on failure, and none reaches for a Microsoft client library. The HTTP
stack is imported inside :meth:`OutlookClient.setup`, so registering the connector costs no HTTP
stack.

**Every request asks for immutable ids** (``Prefer: IdType="ImmutableId"``), delta requests
included. Without it an item's id changes when it is moved between folders, and a message the
owner files away would be a different document.

**A link Graph hands back is followed only if it is Graph's.** ``@odata.nextLink`` and
``@odata.deltaLink`` are full URLs, and a stored delta link is replayed with a bearer token days
later. Both are checked to be on Graph's origin, under ``/v1.0/me/``, before a request is made
with them, as the Confluence client checks ``_links.next``: a poisoned watermark must not be a way
to send the access token anywhere.

**Concurrency is Outlook's own cap.** Four requests per app and mailbox; a fifth is throttled, not
queued. A semaphore holds the client to the configured number, whose ceiling is four, and a 429 is
still answered (``Retry-After`` in both its forms, a ceiling on how long one pause may be).

**Errors name the operation and nothing else.** No message id, folder name or address is in an
error or a log record: the operation, a status code and, for a refusal, Graph's short ``code``
token. The HTTP library's own messages carry the URL, so its exception is deliberately not chained
and a log filter redacts Graph URLs from its INFO records.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from typing import TYPE_CHECKING, Final, Protocol, cast, override
from urllib.parse import quote, urlencode, urlsplit

from manicule.connectors.errors import (
    ConnectorError,
    CursorExpiredError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitedError,
    RemoteError,
    RequestTimeoutError,
    UntrustedLinkError,
)
from manicule.connectors.outlook_auth import GRAPH_BASE
from manicule.connectors.retry import backoff, retry_after

if TYPE_CHECKING:  # pragma: no cover - import-time only
    import httpx

__all__ = [
    "GRAPH_BASE",
    "PAGE_SIZE",
    "DeltaPage",
    "EventDetail",
    "EventStub",
    "Folder",
    "MessageStub",
    "OutlookClient",
    "OutlookCredentialLike",
    "OutlookSettings",
    "Profile",
]

_ORIGIN: Final = "https://graph.microsoft.com"
_ALLOWED_PATH: Final = "/v1.0/me/"

PAGE_SIZE: Final = 100
"""Items a page asks for: the most a folder listing returns, and a size every list accepts."""

_HTTPX_LOGGER: Final = "httpx"
_SAFE_CODE: Final = re.compile(r"[A-Za-z_]{1,64}")
"""What a Graph ``error.code`` must look like to be quoted: a token, never free text."""

_EXPIRED_CODES: Final = frozenset({"syncstatenotfound", "syncstateinvalid", "resyncrequired"})
"""The ``error.code`` values that say a delta token is gone (compared case-folded)."""

_MESSAGE_FIELDS: Final = "id,conversationId,webLink,receivedDateTime"
_EVENT_LIST_FIELDS: Final = "id,changeKey,webLink"
_EVENT_FIELDS: Final = (
    "id,iCalUId,changeKey,subject,start,end,isAllDay,location,organizer,attendees,body,"
    "webLink,isCancelled,type,seriesMasterId,categories,showAs,sensitivity,createdDateTime"
)


class _RedactGraphUrls(logging.Filter):
    """Keeps a Graph request's path and query out of the HTTP stack's own log records.

    ``httpx`` logs every request at INFO, URL included, and a Graph URL holds message ids, folder
    ids, delta tokens and a date window. Only records naming Graph's host are touched.
    """

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        text = record.getMessage()
        if _ORIGIN in text:
            record.msg = _GRAPH_URL.sub(f"{_ORIGIN}/[redacted]", text)
            record.args = ()
        return True


_GRAPH_URL: Final = re.compile(re.escape(_ORIGIN) + r"\S*")
_REDACT_GRAPH_URLS: Final = _RedactGraphUrls()

type Json = Mapping[str, object]


class OutlookSettings(Protocol):
    """The settings the client reads. The Outlook configuration model satisfies it as is."""

    @property
    def max_concurrent_requests(self) -> int: ...

    @property
    def max_retry_after_seconds(self) -> float: ...

    @property
    def max_retries(self) -> int: ...

    @property
    def request_timeout_seconds(self) -> float: ...


class OutlookCredentialLike(Protocol):
    """Whatever mints the access tokens. Consulted per request; may refuse."""

    async def access_token(self, *, refresh: bool = False) -> str: ...

    def rejected(self) -> Exception: ...


@dataclass(frozen=True, slots=True)
class Profile:
    """``GET /me``: who the token is for."""

    object_id: str
    user_principal_name: str
    mail: str

    @property
    def addresses(self) -> tuple[str, ...]:
        """The values the configured account may match; either may be empty."""
        return tuple(value for value in (self.user_principal_name, self.mail) if value)


@dataclass(frozen=True, slots=True)
class Folder:
    """One mail folder: its immutable id, and its path from the top with slashes."""

    id: str
    path: str
    parent_id: str | None


@dataclass(frozen=True, slots=True)
class MessageStub:
    """What a delta or listing says about a message, enough to ask for it."""

    message_id: str
    conversation_id: str | None
    web_link: str | None
    received: str | None


@dataclass(frozen=True, slots=True)
class DeltaPage:
    """One page of a folder's message delta: the unit of a discovery batch."""

    messages: tuple[MessageStub, ...]
    removed: int
    delta_link: str | None
    """Set on the last page only: where the next round resumes."""


@dataclass(frozen=True, slots=True)
class EventStub:
    """One event of the calendar view, with the change key that is its version."""

    event_id: str
    change_key: str
    web_link: str | None


@dataclass(frozen=True, slots=True)
class EventDetail:
    """One event as Graph describes it (times in UTC, body as text)."""

    event_id: str
    change_key: str
    web_link: str | None
    fields: Json


class OutlookClient:
    """The calls, retried and typed.

    Constructed with a transport in the suites, so the real client and retry loop run against a
    synthetic Graph (``tests/connectors/fake_graph.py``). Call :meth:`setup` before the first
    request and :meth:`teardown` after.
    """

    def __init__(
        self,
        settings: OutlookSettings,
        credential: OutlookCredentialLike,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        """Args:
        settings: Concurrency, retry ceiling, retry count and timeout.
        credential: Where access tokens come from.
        transport: An ``httpx`` transport; ``None`` uses the network.
        sleep: How to wait for retries. A test passes one that advances a simulated clock.
        """
        self._settings = settings
        self._credential = credential
        self._transport = transport
        self._sleep = sleep if sleep is not None else asyncio.sleep
        self._client: httpx.AsyncClient | None = None
        self._slots = asyncio.Semaphore(settings.max_concurrent_requests)
        self.requests = 0
        """Requests sent, retries included."""
        self.throttled = 0
        """Times Graph answered "slow down"."""

    async def setup(self) -> None:
        """Build the HTTP client. Idempotent."""
        if self._client is not None:
            return
        import httpx  # noqa: PLC0415 - see the module docstring

        logging.getLogger(_HTTPX_LOGGER).addFilter(_REDACT_GRAPH_URLS)
        self._client = httpx.AsyncClient(
            transport=self._transport,
            timeout=self._settings.request_timeout_seconds,
            follow_redirects=False,
        )

    async def teardown(self) -> None:
        """Close the HTTP client."""
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    # --- identity ----------------------------------------------------------------------

    async def get_me(self) -> Profile:
        """``GET /me``: the mailbox's object id, sign-in name and primary address."""
        payload = await self._get_json(
            "me.get", _url("/me", {"$select": "id,userPrincipalName,mail"})
        )
        return Profile(
            object_id=_text(payload, "id", "me.get"),
            user_principal_name=_optional_text(payload.get("userPrincipalName")),
            mail=_optional_text(payload.get("mail")),
        )

    # --- mail folders ------------------------------------------------------------------

    async def well_known_folder_id(self, name: str) -> str | None:
        """The id of a well-known folder (``junkemail``), or ``None`` if the mailbox has none."""
        operation = "mailFolders.get"
        try:
            payload = await self._get_json(
                operation, _url(f"/me/mailFolders/{quote(name, safe='')}", {"$select": "id"})
            )
        except NotFoundError:
            return None
        return _text(payload, "id", operation)

    async def list_folders(self) -> AsyncIterator[Folder]:
        """Every mail folder, hidden ones included, parents before children.

        Walks ``/me/mailFolders`` and the ``/childFolders`` of every folder that reports any. A
        folder whose children were not asked about would be a subtree invisible to scope.
        """
        operation = "mailFolders.list"
        paths: dict[str, str] = {}
        queue: list[str] = []
        async for page in self._pages(operation, _folders_url("/me/mailFolders")):
            for entry in _entries(page, "value", operation):
                folder_id, name = _folder(entry, operation)
                paths[folder_id] = name
                yield Folder(id=folder_id, path=name, parent_id=None)
                if _has_children(entry):
                    queue.append(folder_id)
        while queue:
            parent = queue.pop(0)
            children = f"/me/mailFolders/{quote(parent, safe='')}/childFolders"
            async for page in self._pages(operation, _folders_url(children)):
                for entry in _entries(page, "value", operation):
                    folder_id, name = _folder(entry, operation)
                    paths[folder_id] = f"{paths[parent]}/{name}"
                    yield Folder(id=folder_id, path=paths[folder_id], parent_id=parent)
                    if _has_children(entry):
                        queue.append(folder_id)

    # --- mail --------------------------------------------------------------------------

    async def list_message_ids(self, folder_id: str) -> AsyncIterator[tuple[str, ...]]:
        """The ids of one folder's messages, a page at a time (no bodies, no properties)."""
        operation = "messages.list"
        url = _url(
            f"/me/mailFolders/{quote(folder_id, safe='')}/messages",
            {"$select": "id", "$top": str(PAGE_SIZE)},
        )
        async for payload in self._pages(operation, url):
            yield tuple(
                _text(item, "id", operation) for item in _entries(payload, "value", operation)
            )

    async def message_delta(
        self, folder_id: str, delta_link: str | None
    ) -> AsyncIterator[DeltaPage]:
        """One folder's message delta, a page at a time.

        With ``delta_link`` it is an incremental round from the link a previous round ended on;
        without, the first round, which returns every message currently in the folder. Removals
        are counted and not returned: deletion is reconciliation's to act on.

        Raises:
            CursorExpiredError: Graph no longer holds the delta state behind a link of this round
                (a 410, or a 404 or ``syncStateNotFound`` for a stored or paged link), including
                one lost part way through a round from scratch.
            NotFoundError: The first request of a round from scratch found no such folder.
            UntrustedLinkError: A stored or returned link is not Graph's.
        """
        operation = "messages.delta"
        if delta_link is not None:
            url = _trusted(delta_link, operation)
        else:
            url = _url(
                f"/me/mailFolders/{quote(folder_id, safe='')}/messages/delta",
                {"$select": _MESSAGE_FIELDS},
            )
        seen: set[str] = set()
        first = delta_link is None
        while True:
            # A 404 means the delta state is gone for every request but the first of a round from
            # scratch, where there is no state yet and it can only mean the folder is gone.
            payload = await self._get_json(
                operation,
                url,
                prefer="odata.maxpagesize=" + str(PAGE_SIZE),
                expired=not first,
            )
            first = False
            messages: list[MessageStub] = []
            removed = 0
            for entry in _entries(payload, "value", operation):
                if "@removed" in entry:
                    removed += 1
                    continue
                messages.append(_message_stub(entry, operation))
            following = payload.get("@odata.nextLink")
            final = payload.get("@odata.deltaLink")
            if isinstance(following, str) and following:
                yield DeltaPage(tuple(messages), removed, None)
                if following in seen:
                    msg = (
                        f"Graph {operation} sent pagination back to a page it had already "
                        f"followed; continuing would repeat results forever"
                    )
                    raise ConnectorError(msg)
                seen.add(following)
                url = _trusted(following, operation)
                continue
            if isinstance(final, str) and final:
                yield DeltaPage(tuple(messages), removed, _trusted(final, operation))
                return
            msg = f"Graph {operation} ended without a next link or a delta link"
            raise ConnectorError(msg)

    async def get_message_mime(self, message_id: str) -> bytes:
        """``GET /me/messages/{id}/$value``: the whole MIME message, attachments inline.

        Raises:
            NotFoundError: The message was deleted or moved out of the mailbox since discovery.
        """
        operation = "messages.value"
        _require_id(message_id, operation)
        response = await self._request(
            operation,
            _url(f"/me/messages/{quote(message_id, safe='')}/$value", {}),
            accept="*/*",
        )
        return response.content

    async def get_message_stub(self, message_id: str) -> MessageStub:
        """The link, conversation and arrival time of one message, for a fetch that lacks them."""
        operation = "messages.get"
        _require_id(message_id, operation)
        payload = await self._get_json(
            operation,
            _url(f"/me/messages/{quote(message_id, safe='')}", {"$select": _MESSAGE_FIELDS}),
        )
        return _message_stub(payload, operation)

    # --- calendar ----------------------------------------------------------------------

    async def list_events(
        self, start: datetime, end: datetime
    ) -> AsyncIterator[tuple[EventStub, ...]]:
        """The calendar view between two instants, a page at a time, as ids and change keys.

        ``calendarView`` expands recurring events into occurrences, which is the unit that is
        indexed; a series master is not returned by it.
        """
        operation = "calendarView.list"
        url = _url(
            "/me/calendarView",
            {
                "startDateTime": _instant(start),
                "endDateTime": _instant(end),
                "$select": _EVENT_LIST_FIELDS,
                "$top": str(PAGE_SIZE),
            },
        )
        async for payload in self._pages(operation, url):
            yield tuple(
                EventStub(
                    event_id=_text(item, "id", operation),
                    change_key=_text(item, "changeKey", operation),
                    web_link=_optional_text(item.get("webLink")) or None,
                )
                for item in _entries(payload, "value", operation)
            )

    async def get_event(self, event_id: str) -> EventDetail:
        """One event with its times in UTC and its body as text.

        Raises:
            NotFoundError: The event was deleted since discovery.
        """
        operation = "events.get"
        _require_id(event_id, operation)
        payload = await self._get_json(
            operation,
            _url(f"/me/events/{quote(event_id, safe='')}", {"$select": _EVENT_FIELDS}),
            prefer='outlook.timezone="UTC", outlook.body-content-type="text"',
        )
        return EventDetail(
            event_id=_text(payload, "id", operation),
            change_key=_text(payload, "changeKey", operation),
            web_link=_optional_text(payload.get("webLink")) or None,
            fields=payload,
        )

    # --- the request loop --------------------------------------------------------------

    async def _pages(self, operation: str, url: str) -> AsyncIterator[Json]:
        """Yield each page of a ``nextLink`` listing as its payload.

        A link seen twice is a loop, and a loop over a paginated listing reads as a very large
        mailbox, so it is refused rather than followed.
        """
        seen: set[str] = set()
        current = url
        while True:
            payload = await self._get_json(operation, current)
            yield payload
            following = payload.get("@odata.nextLink")
            if not following:
                return
            if not isinstance(following, str):
                msg = f"Graph {operation} returned a next link that is not text"
                raise ConnectorError(msg)
            if following in seen:
                msg = (
                    f"Graph {operation} sent pagination back to a page it had already "
                    f"followed; continuing would repeat results forever"
                )
                raise ConnectorError(msg)
            seen.add(following)
            current = _trusted(following, operation)

    async def _get_json(
        self,
        operation: str,
        url: str,
        *,
        prefer: str = "",
        expired: bool = False,
    ) -> Json:
        response = await self._request(operation, url, prefer=prefer, expired=expired)
        try:
            body: object = response.json()
        except ValueError:
            msg = f"Graph {operation} returned a body that is not JSON"
            raise ConnectorError(msg) from None
        if not isinstance(body, dict):
            msg = f"Graph {operation} returned JSON that is not an object"
            raise ConnectorError(msg)
        return cast("Json", body)

    async def _request(
        self,
        operation: str,
        url: str,
        *,
        prefer: str = "",
        accept: str = "application/json",
        expired: bool = False,
    ) -> httpx.Response:
        """One retried GET. ``expired`` says a 404 or 410 means the delta state is gone."""
        client = self._require_client()
        import httpx  # noqa: PLC0415 - see setup()

        preferences = ", ".join(part for part in ('IdType="ImmutableId"', prefer) if part)
        attempt = 0
        refresh = False
        refreshed = False
        while True:
            token = await self._credential.access_token(refresh=refresh)
            refresh = False
            self.requests += 1
            try:
                async with self._slots:
                    response = await client.get(
                        url,
                        headers={
                            "Authorization": f"Bearer {token}",
                            "Accept": accept,
                            "Prefer": preferences,
                        },
                    )
            except httpx.TransportError as exc:
                await self._sleep(self._transport_wait(operation, exc, attempt))
                attempt += 1
                continue

            if response.status_code == HTTPStatus.UNAUTHORIZED:
                # One refresh, one retry, and the retry does not spend the failure budget.
                if refreshed:
                    raise self._credential.rejected()
                refresh = refreshed = True
                continue
            wait = self._judge(operation, response, attempt, expired=expired)
            if wait is None:
                return response
            await self._sleep(wait)
            attempt += 1

    def _transport_wait(self, operation: str, exc: Exception, attempt: int) -> float:
        import httpx  # noqa: PLC0415 - see setup()

        if attempt < self._settings.max_retries:
            return backoff(attempt)
        name = type(exc).__name__
        if isinstance(exc, httpx.TimeoutException):
            msg = f"Graph {operation} timed out ({name})"
            raise RequestTimeoutError(msg) from None
        msg = f"could not reach Graph for {operation} ({name})"
        raise ConnectorError(msg) from None

    def _judge(
        self, operation: str, response: httpx.Response, attempt: int, *, expired: bool
    ) -> float | None:
        """Type a non-401 answer: ``None`` if it is usable, else the pause before retrying."""
        status = response.status_code
        code = _error_code(response)
        if status == HTTPStatus.TOO_MANY_REQUESTS:
            return self._throttle_wait(operation, response.headers, attempt)
        if status >= HTTPStatus.INTERNAL_SERVER_ERROR:
            if attempt >= self._settings.max_retries:
                msg = f"Graph {operation} failed with HTTP {status}"
                raise RemoteError(msg, status_code=status)
            return backoff(attempt)
        if expired and (
            status in (HTTPStatus.GONE, HTTPStatus.NOT_FOUND)
            or (status == HTTPStatus.BAD_REQUEST and (code or "").casefold() in _EXPIRED_CODES)
        ):
            msg = (
                f"Graph {operation} no longer holds the delta state behind the stored link; "
                f"a complete walk of the folder is needed"
            )
            raise CursorExpiredError(msg)
        if status == HTTPStatus.GONE and (code or "").casefold() in _EXPIRED_CODES:
            msg = f"Graph {operation} reports its delta state is gone"
            raise CursorExpiredError(msg)
        if status == HTTPStatus.NOT_FOUND:
            msg = f"Graph {operation} found no such item"
            raise NotFoundError(msg)
        if status == HTTPStatus.FORBIDDEN:
            detail = f" ({code})" if code is not None else ""
            msg = f"Graph refused {operation} with HTTP 403{detail}"
            raise PermissionDeniedError(msg)
        if not HTTPStatus.OK <= status < HTTPStatus.MULTIPLE_CHOICES:
            detail = f" ({code})" if code is not None else ""
            msg = f"Graph answered {operation} with HTTP {status}{detail}"
            raise RemoteError(msg, status_code=status)
        return None

    def _throttle_wait(self, operation: str, headers: Mapping[str, str], attempt: int) -> float:
        self.throttled += 1
        requested = retry_after(headers)
        wait = requested if requested is not None else backoff(attempt)
        ceiling = self._settings.max_retry_after_seconds
        if wait > ceiling:
            msg = (
                f"Graph asked for a {wait:.0f}s pause on {operation}, over the {ceiling:.0f}s "
                f"ceiling (max_retry_after_seconds). Stopping rather than appearing to run while "
                f"waiting."
            )
            raise RateLimitedError(msg, retry_after=wait)
        if attempt >= self._settings.max_retries:
            msg = f"Graph was still limiting {operation} after {attempt + 1} attempts"
            raise RateLimitedError(msg, retry_after=requested)
        return wait

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            msg = "OutlookClient.setup() was not called before the first request"
            raise ConnectorError(msg)
        return self._client


def _url(path: str, params: Mapping[str, str]) -> str:
    """A Graph URL. ``$`` and ``,`` are left alone: they are OData syntax, not data."""
    query = urlencode(params, quote_via=quote, safe="$,:") if params else ""
    return f"{GRAPH_BASE}{path}" + (f"?{query}" if query else "")


def _trusted(link: str, operation: str) -> str:
    """``link`` if it is on Graph's origin under ``/v1.0/me/``; otherwise refuse it.

    Raises:
        UntrustedLinkError: The link is not Graph's. The link is not in the message: it came from
            a stored watermark or a response, and is exactly what must not be echoed.
    """
    parts = urlsplit(link)
    ok = (
        parts.scheme == "https"
        and parts.hostname == "graph.microsoft.com"
        and parts.port in (None, 443)
        and parts.username is None
        and parts.path.casefold().startswith(_ALLOWED_PATH)
    )
    if not ok:
        msg = (
            f"Graph {operation} was given a link that is not on Graph's origin under "
            f"{_ALLOWED_PATH}; it was not followed, so no credential was sent to it"
        )
        raise UntrustedLinkError(msg)
    return link


def _instant(value: datetime) -> str:
    """An ISO 8601 instant in UTC, as ``calendarView`` takes it."""
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.0000000Z")


def _require_id(item_id: str, operation: str) -> None:
    if not item_id:
        msg = f"Graph {operation} was asked for an item with no id"
        raise ConnectorError(msg)


def _error_code(response: httpx.Response) -> str | None:
    """Graph's ``error.code`` if it is a short token, else ``None``."""
    try:
        body: object = response.json()
    except ValueError:
        return None
    outer = cast("Json", body) if isinstance(body, dict) else None
    error = outer.get("error") if outer is not None else None
    code = cast("Json", error).get("code") if isinstance(error, dict) else None
    return code if isinstance(code, str) and _SAFE_CODE.fullmatch(code) else None


def _text(payload: Json, key: str, operation: str) -> str:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, str | int) or value == "":
        msg = f"Graph {operation} returned no usable {key!r}"
        raise ConnectorError(msg)
    return str(value)


def _optional_text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _entries(payload: Json, key: str, operation: str) -> list[Json]:
    value = payload.get(key)
    if value is None:
        return []
    items = cast("list[object]", value) if isinstance(value, list) else None
    if items is None or not all(isinstance(item, dict) for item in items):
        msg = f"Graph {operation} returned {key!r} in an unexpected shape"
        raise ConnectorError(msg)
    return [cast("Json", item) for item in items]


def _folder(entry: Json, operation: str) -> tuple[str, str]:
    return _text(entry, "id", operation), _text(entry, "displayName", operation)


def _message_stub(entry: Json, operation: str) -> MessageStub:
    return MessageStub(
        message_id=_text(entry, "id", operation),
        conversation_id=_optional_text(entry.get("conversationId")) or None,
        web_link=_optional_text(entry.get("webLink")) or None,
        received=_optional_text(entry.get("receivedDateTime")) or None,
    )


def _has_children(entry: Json) -> bool:
    count = entry.get("childFolderCount")
    return isinstance(count, int) and not isinstance(count, bool) and count > 0


def _folders_url(path: str) -> str:
    return _url(
        path,
        {
            "$select": "id,displayName,childFolderCount",
            "$top": str(PAGE_SIZE),
            "includeHiddenFolders": "true",
        },
    )
