"""A synthetic Gmail: the traps a real mailbox has, behind an ``httpx`` transport.

Every Gmail suite runs the **real** client — the real pacer, retry loop and query encoding —
against this, so none of them needs a Google account or the network. It is deliberately not a
friendly fake. It reproduces, each switchable from a test:

- **Pagination tokens** that are opaque and contain ``+`` and ``/``; a token the server does not
  know is a 400, and a mailbox can be told to send pagination back to a page already followed.
- **A history window that expires**: a ``startHistoryId`` older than the floor is a 404.
- **History records with and without label ids**: some carry the message's labels, most carry
  only message and thread ids.
- **A message deleted between list and get**: still listed, 404 when fetched.
- **429 with both ``Retry-After`` forms**, a 403 whose *reason* says "slow down" (in either of
  the two places Google puts it), and 403s that really mean no.
- **Raw messages** with attachments and with HTML-only bodies, base64url **without padding**.
- **SPAM, TRASH and DRAFT labels**, user labels with spaces and slashes, and ``CATEGORY_*``.
- **A profile whose address can be switched**, as when a token belongs to another mailbox.
- **A token that expires**, and one the server will not accept however often it is refreshed.
- **Quota**: the server meters the units each request costs over a rolling minute, as Gmail
  does, answers 429 when the limit is exceeded, and remembers the most it was ever asked to
  carry (:attr:`FakeGmail.peak_units`) — an account of the client's pacing that does not trust
  the client's own.

Addresses are at ``example.org``. A ``q`` parameter is a 400: scope is never translated into a
search query (``docs/connectors/gmail.md`` §3), and a client that sent one would be silently
narrowing what it indexes.
"""

from __future__ import annotations

import base64
import zlib
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.message import EmailMessage
from email.utils import format_datetime

import httpx

__all__ = [
    "ADDRESS",
    "FakeCredential",
    "FakeGmail",
    "FakeGmailSettings",
    "FakeMessage",
    "SimulatedTime",
    "html_only_message",
    "message_with_attachment",
    "plain_message",
]

ADDRESS = "sync.user@example.org"
"""Whose mailbox the fake starts as."""

HOST = "gmail.googleapis.com"
PREFIX = "/gmail/v1/users/me"

SERVER_LIMIT = 6000
"""Quota units per rolling minute the fake allows, Gmail's per-user figure."""

COSTS = {
    "profile": 1,
    "labels": 1,
    "list": 5,
    "history": 2,
    "minimal": 20,
    "raw": 60,
}

SYSTEM_LABELS = (
    "INBOX",
    "SENT",
    "SPAM",
    "TRASH",
    "DRAFT",
    "UNREAD",
    "STARRED",
    "IMPORTANT",
    "CATEGORY_PROMOTIONS",
    "CATEGORY_SOCIAL",
)
"""Gmail's own labels. Their id is their name, which a user label's never is."""

USER_LABELS = ("Clients/Acme Corp", "Receipts 2026", "Archive/Old Mail/2019")
"""User labels, named the way people name them: spaces, slashes, nesting."""


class SimulatedTime:
    """A clock that sleeping advances, so a simulated minute takes no time.

    ``support.Waits`` records what it was asked to wait for but does not move any clock; paired
    with a pacer that reads one, it would hang. Here sleeping *is* time passing.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.waits: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds


@dataclass(frozen=True, slots=True)
class FakeGmailSettings:
    """What :class:`~manicule.connectors.gmail_client.GmailSettings` reads, at its defaults."""

    quota_units_per_minute: int = 5400
    max_retry_after_seconds: float = 120.0
    max_retries: int = 5
    request_timeout_seconds: float = 30.0


@dataclass(slots=True)
class FakeMessage:
    """One message in the fake mailbox."""

    id: str
    thread_id: str
    raw: bytes
    label_ids: list[str]


class FakeCredential:
    """A credential that mints tokens the fake accepts."""

    def __init__(self, server: FakeGmail) -> None:
        self._server = server
        self._token = server.issue_token()
        self.refreshes = 0
        """Times a refresh was asked for."""

    async def access_token(self, *, refresh: bool = False) -> str:
        if refresh:
            self.refreshes += 1
            self._token = self._server.issue_token()
        return self._token

    def renewal(self) -> str:
        return "Run `manicule connector login mail` to authorize it again."


def plain_message(subject: str = "Quarterly figures", *, sender: str = "pat@example.org") -> bytes:
    """A text/plain message."""
    message = EmailMessage()
    message["From"] = sender
    message["To"] = ADDRESS
    message["Subject"] = subject
    message["Message-ID"] = f"<{zlib.crc32(subject.encode()):08x}@example.org>"
    message.set_content("The figures are attached to nothing; they are in this sentence.\n")
    return message.as_bytes()


def html_only_message(subject: str = "Newsletter") -> bytes:
    """A message whose only body is ``text/html`` — no plain alternative."""
    message = EmailMessage()
    message["From"] = "news@example.org"
    message["To"] = ADDRESS
    message["Subject"] = subject
    message.set_content("<html><body><h1>News</h1><p>Only HTML.</p></body></html>", subtype="html")
    return message.as_bytes()


def message_with_attachment(subject: str = "With a file") -> bytes:
    """A message with a body and a PDF-typed attachment."""
    message = EmailMessage()
    message["From"] = "pat@example.org"
    message["To"] = ADDRESS
    message["Subject"] = subject
    message.set_content("See the attachment.\n")
    message.add_attachment(
        b"%PDF-1.4 synthetic", maintype="application", subtype="pdf", filename="report.pdf"
    )
    return message.as_bytes()


class FakeGmail:
    """The synthetic mailbox and its HTTP face. Hand :meth:`transport` to the client."""

    def __init__(
        self,
        *,
        address: str = ADDRESS,
        page_size: int = 500,
        clock: Callable[[], float] | None = None,
        enforce_quota: bool = True,
    ) -> None:
        self.address = address
        self.page_size = page_size
        """The most ids a page holds, whatever ``maxResults`` asks for. Small, to force paging."""
        self.requests: list[httpx.Request] = []
        self.messages: dict[str, FakeMessage] = {}
        self.listed_but_gone: set[str] = set()
        """Ids still in listings whose ``messages.get`` answers 404."""
        self.history_id = 1000
        self.history_floor = 0
        """History older than this is gone: a ``startHistoryId`` below it is a 404."""
        self.records: list[dict[str, object]] = []
        self.loop_pagination = False
        """Send ``messages.list`` pagination back to the first page's token."""
        self.raw_override: str | None = None
        """Answer ``format=raw`` with this text in place of the encoded message."""
        self.request_times: list[float] = []
        """When each request reached the server, by the fake's clock."""
        self.corrupt_raw = False
        """Answer ``format=raw`` with a stray character in the middle of the base64url."""
        self.rejecting_all_tokens = False
        """Answer 401 to every token, however freshly minted."""
        self.peak_units = 0
        """The most units ever carried in one rolling minute, as the fake counted them."""
        self.overruns = 0
        """Requests refused because the rolling minute was already full."""
        self._clock = clock if clock is not None else (lambda: 0.0)
        self._enforce_quota = enforce_quota
        self._charges: deque[tuple[float, int]] = deque()
        self._valid_tokens: set[str] = set()
        self._token_serial = 0
        self._faults: deque[Callable[[httpx.Request], httpx.Response]] = deque()
        self._labels = {
            **{name: name for name in SYSTEM_LABELS},
            **{name: f"Label_{index + 1}" for index, name in enumerate(USER_LABELS)},
        }
        self._serial = 0

    # ------------------------------------------------------------------ test-side controls

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def issue_token(self) -> str:
        """Mint a token the fake will accept."""
        self._token_serial += 1
        token = f"ya29.synthetic-{self._token_serial}"
        self._valid_tokens.add(token)
        return token

    def expire_tokens(self) -> None:
        """Forget every token minted so far; the next request with one is a 401."""
        self._valid_tokens.clear()

    def label_id(self, name: str) -> str:
        """The id of a label by name (system labels are their own ids)."""
        return self._labels[name]

    def add_message(
        self,
        raw: bytes | None = None,
        *,
        labels: Sequence[str] = ("INBOX",),
        record_labels: bool = False,
    ) -> FakeMessage:
        """Receive a message. ``labels`` are names; ``record_labels`` puts them in the history."""
        self._serial += 1
        message = FakeMessage(
            id=f"18c3f0a1b2{self._serial:06x}",
            thread_id=f"18c3f0a1b2{self._serial:06x}",
            raw=raw if raw is not None else plain_message(f"Message {self._serial}"),
            label_ids=[self._labels[name] for name in labels],
        )
        self.messages[message.id] = message
        self._record("messagesAdded", message, with_labels=record_labels)
        return message

    def add_label(self, message: FakeMessage, name: str, *, record_labels: bool = False) -> None:
        label = self._labels[name]
        message.label_ids.append(label)
        self._record("labelsAdded", message, changed=[label], with_labels=record_labels)

    def remove_label(self, message: FakeMessage, name: str, *, record_labels: bool = False) -> None:
        label = self._labels[name]
        message.label_ids.remove(label)
        self._record("labelsRemoved", message, changed=[label], with_labels=record_labels)

    def delete_message(self, message: FakeMessage, *, still_listed: bool = False) -> None:
        """Delete permanently. ``still_listed`` is the race: gone from the mailbox, not yet from
        a listing already under way."""
        del self.messages[message.id]
        if still_listed:
            self.listed_but_gone.add(message.id)
        self._record("messagesDeleted", message, with_labels=False)

    def expire_history_through(self, history_id: int) -> None:
        """Drop history up to and including ``history_id``."""
        self.history_floor = history_id + 1
        self.records = [record for record in self.records if int(str(record["id"])) > history_id]

    def switch_mailbox(self, address: str) -> None:
        """Make the profile answer as another mailbox."""
        self.address = address

    def throttle(self, times: int = 1, *, retry_after: str | None = "2") -> None:
        """The next ``times`` requests are answered 429, with ``retry_after`` as given."""
        headers = {} if retry_after is None else {"Retry-After": retry_after}
        for _ in range(times):
            self._faults.append(
                lambda _request, headers=headers: httpx.Response(
                    429,
                    headers=headers,
                    json=_error(429, "RESOURCE_EXHAUSTED", "rateLimitExceeded"),
                )
            )

    def throttle_until(self, seconds_from_now: float) -> None:
        """The next request is answered 429 with an HTTP-date ``Retry-After``."""
        when = datetime.now(tz=UTC) + timedelta(seconds=seconds_from_now)
        self.throttle(1, retry_after=format_datetime(when, usegmt=True))

    def refuse(
        self,
        reason: str,
        *,
        times: int = 1,
        retry_after: str | None = None,
        style: str = "errors",
    ) -> None:
        """The next ``times`` requests are answered 403 with ``reason``.

        ``style`` is where Google puts the reason: ``"errors"`` (``error.errors[].reason``, as
        ``rateLimitExceeded``) or ``"details"`` (``error.details[].reason``, as
        ``RATE_LIMIT_EXCEEDED``). The reason is sent exactly as given.
        """
        headers = {} if retry_after is None else {"Retry-After": retry_after}
        for _ in range(times):
            self._faults.append(
                _answering(403, _error(403, "PERMISSION_DENIED", reason, style), headers)
            )

    def fail(self, status: int, *, times: int = 1) -> None:
        """The next ``times`` requests are answered with a bare ``status``."""
        for _ in range(times):
            self._faults.append(_answering(status, _error(status, "UNAVAILABLE", "backendError")))

    def time_out(self, times: int = 1) -> None:
        """The next ``times`` requests never answer."""

        def hang(request: httpx.Request) -> httpx.Response:
            msg = "synthetic read timeout"
            raise httpx.ReadTimeout(msg, request=request)

        self._faults.extend([hang] * times)

    def drop_connection(self, times: int = 1) -> None:
        """The next ``times`` requests cannot connect."""

        def refuse_connection(request: httpx.Request) -> httpx.Response:
            msg = "synthetic connection failure"
            raise httpx.ConnectError(msg, request=request)

        self._faults.extend([refuse_connection] * times)

    @property
    def faults_pending(self) -> int:
        """Injected faults no request has consumed yet."""
        return len(self._faults)

    def queries(self) -> list[dict[str, list[str]]]:
        """Every request's query string, parameter names to all their values."""
        found: list[dict[str, list[str]]] = []
        for request in self.requests:
            query: dict[str, list[str]] = {}
            for key, value in request.url.params.multi_items():
                query.setdefault(key, []).append(value)
            found.append(query)
        return found

    # ------------------------------------------------------------------ the HTTP face

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.request_times.append(self._clock())
        if request.url.host != HOST or not request.url.path.startswith(PREFIX):
            return httpx.Response(404, json=_error(404, "NOT_FOUND", "notFound"))
        path = request.url.path[len(PREFIX) :]
        cost = self._cost(path, request)

        if self._meter(cost) and self._enforce_quota:
            self.overruns += 1
            return httpx.Response(
                429,
                headers={"Retry-After": "5"},
                json=_error(429, "RESOURCE_EXHAUSTED", "rateLimitExceeded"),
            )
        if not self._authorized(request):
            return httpx.Response(401, json=_error(401, "UNAUTHENTICATED", "authError"))
        if self._faults:
            return self._faults.popleft()(request)
        if "q" in request.url.params:
            return httpx.Response(400, json=_error(400, "INVALID_ARGUMENT", "invalidArgument"))
        return self._route(request, path)

    def _authorized(self, request: httpx.Request) -> bool:
        header = request.headers.get("authorization", "")
        return (
            not self.rejecting_all_tokens
            and header.startswith("Bearer ")
            and header.removeprefix("Bearer ") in self._valid_tokens
        )

    def _cost(self, path: str, request: httpx.Request) -> int:
        if path == "/profile":
            return COSTS["profile"]
        if path == "/labels":
            return COSTS["labels"]
        if path == "/history":
            return COSTS["history"]
        if path == "/messages":
            return COSTS["list"]
        if request.url.params.get("format") == "minimal":
            return COSTS["minimal"]
        return COSTS["raw"]

    def _meter(self, cost: int) -> bool:
        """Charge ``cost`` against the rolling minute; ``True`` if that overran the limit."""
        now = self._clock()
        while self._charges and now - self._charges[0][0] >= 60.0:
            self._charges.popleft()
        carried = sum(units for _, units in self._charges)
        if carried + cost > SERVER_LIMIT:
            return True
        self._charges.append((now, cost))
        self.peak_units = max(self.peak_units, carried + cost)
        return False

    def _route(self, request: httpx.Request, path: str) -> httpx.Response:
        if path == "/profile":
            return httpx.Response(
                200,
                json={
                    "emailAddress": self.address,
                    "messagesTotal": len(self.messages),
                    "threadsTotal": len(self.messages),
                    "historyId": str(self.history_id),
                },
            )
        if path == "/labels":
            return httpx.Response(200, json={"labels": self._label_entries()})
        if path == "/messages":
            return self._list(request)
        if path.startswith("/messages/"):
            return self._get(request, path.removeprefix("/messages/"))
        if path == "/history":
            return self._history(request)
        return httpx.Response(404, json=_error(404, "NOT_FOUND", "notFound"))

    def _label_entries(self) -> list[dict[str, str]]:
        return [
            {
                "id": label_id,
                "name": name,
                "type": "system" if name in SYSTEM_LABELS else "user",
            }
            for name, label_id in self._labels.items()
        ]

    def _list(self, request: httpx.Request) -> httpx.Response:
        params = request.url.params
        wanted = params.get_list("labelIds")
        include_spam_trash = params.get("includeSpamTrash", "false") == "true"
        ids = [
            message.id
            for message in self.messages.values()
            if all(label in message.label_ids for label in wanted)
            and (include_spam_trash or not {"SPAM", "TRASH"} & set(message.label_ids))
        ]
        ids = sorted({*ids, *self.listed_but_gone})
        size = min(int(params.get("maxResults", "100")), self.page_size)
        start = _offset(params.get("pageToken"))
        if start is None:
            return httpx.Response(400, json=_error(400, "INVALID_ARGUMENT", "invalidArgument"))
        page = ids[start : start + size]
        body: dict[str, object] = {"resultSizeEstimate": len(ids)}
        if page:
            body["messages"] = [{"id": item, "threadId": item} for item in page]
        if start + size < len(ids):
            body["nextPageToken"] = _token(start + size)
        elif self.loop_pagination and start > 0:
            body["nextPageToken"] = _token(size)
        return httpx.Response(200, json=body)

    def _get(self, request: httpx.Request, message_id: str) -> httpx.Response:
        message = self.messages.get(message_id)
        if message is None:
            return httpx.Response(404, json=_error(404, "NOT_FOUND", "notFound"))
        shape = request.url.params.get("format", "full")
        body: dict[str, object] = {
            "id": message.id,
            "threadId": message.thread_id,
            "historyId": str(self.history_id),
            "sizeEstimate": len(message.raw),
        }
        if message.label_ids:
            body["labelIds"] = list(message.label_ids)
        if shape == "raw":
            # Gmail's base64url omits the padding.
            encoded = base64.urlsafe_b64encode(message.raw).decode().rstrip("=")
            # One stray character inside otherwise valid base64: the kind of damage a lenient
            # decoder discards without a word, returning a message with bytes missing.
            body["raw"] = encoded[:8] + "*" + encoded[8:] if self.corrupt_raw else encoded
            if self.raw_override is not None:
                body["raw"] = self.raw_override
        elif shape != "minimal":
            return httpx.Response(400, json=_error(400, "INVALID_ARGUMENT", "invalidArgument"))
        return httpx.Response(200, json=body)

    def _history(self, request: httpx.Request) -> httpx.Response:
        params = request.url.params
        raw_start = params.get("startHistoryId", "")
        if not raw_start.isdigit():
            return httpx.Response(400, json=_error(400, "INVALID_ARGUMENT", "invalidArgument"))
        start = int(raw_start)
        if start < self.history_floor:
            return httpx.Response(404, json=_error(404, "NOT_FOUND", "notFound"))
        wanted = {
            {
                "messageAdded": "messagesAdded",
                "messageDeleted": "messagesDeleted",
                "labelAdded": "labelsAdded",
                "labelRemoved": "labelsRemoved",
            }[kind]
            for kind in params.get_list("historyTypes")
        }
        matching: list[dict[str, object]] = []
        for record in self.records:
            if int(str(record["id"])) <= start:
                continue
            kept = {
                key: value
                for key, value in record.items()
                if key in ("id", "messages") or not wanted or key in wanted
            }
            if any(key not in ("id", "messages") for key in kept):
                matching.append(kept)
        size = min(int(params.get("maxResults", "100")), self.page_size)
        offset = _offset(params.get("pageToken"))
        if offset is None:
            return httpx.Response(400, json=_error(400, "INVALID_ARGUMENT", "invalidArgument"))
        page = matching[offset : offset + size]
        body: dict[str, object] = {"historyId": str(self.history_id)}
        if page:
            body["history"] = page
        if offset + size < len(matching):
            body["nextPageToken"] = _token(offset + size)
        return httpx.Response(200, json=body)

    def _record(
        self,
        key: str,
        message: FakeMessage,
        *,
        changed: Sequence[str] = (),
        with_labels: bool,
    ) -> None:
        self.history_id += 1
        entry: dict[str, object] = {"message": {"id": message.id, "threadId": message.thread_id}}
        if with_labels:
            entry["message"] = {**_mapping(entry["message"]), "labelIds": list(message.label_ids)}
        if changed:
            entry["labelIds"] = list(changed)
        self.records.append(
            {
                "id": str(self.history_id),
                "messages": [{"id": message.id, "threadId": message.thread_id}],
                key: [entry],
            }
        )


def _answering(
    status: int, body: dict[str, object], headers: Mapping[str, str] | None = None
) -> Callable[[httpx.Request], httpx.Response]:
    """A scripted answer, built fresh for each request that is handed it."""
    return lambda _request: httpx.Response(status, headers=dict(headers or {}), json=body)


def _mapping(value: object) -> Mapping[str, object]:
    assert isinstance(value, dict)
    return value  # pyright: ignore[reportUnknownVariableType]


def _token(offset: int) -> str:
    """An opaque page token, with the characters that corrupt a naively built query."""
    return f"AKv+/{offset}=="


def _offset(token: str | None) -> int | None:
    """The offset a token stands for; ``None`` for one this server never issued."""
    if token is None:
        return 0
    if token.startswith("AKv+/") and token.endswith("=="):
        digits = token.removeprefix("AKv+/").removesuffix("==")
        if digits.isdigit():
            return int(digits)
    return None


def _error(status: int, state: str, reason: str, style: str = "errors") -> dict[str, object]:
    """Google's error envelope, with the reason where ``style`` says."""
    error: dict[str, object] = {"code": status, "message": "synthetic failure", "status": state}
    if style == "details":
        error["details"] = [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": reason}]
    else:
        error["errors"] = [
            {"message": "synthetic failure", "domain": "usageLimits", "reason": reason}
        ]
    return {"error": error}
