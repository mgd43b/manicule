"""A synthetic Microsoft Graph and Entra: the traps of a real tenant, behind an ``httpx`` transport.

Every Outlook suite runs the **real** client, credential and connector against this, so none of
them needs a Microsoft 365 tenant or the network. It is deliberately not a friendly fake. It
reproduces, each switchable from a test:

- **Per-folder delta** with opaque ``$skiptoken`` / ``$deltatoken`` links in Graph's own shape
  (``mailFolders('ID')/messages/delta``), ``@removed`` entries for a deleted or moved-out message,
  and a floor under which a delta token is gone: answered as 410, or as a 404 or a 400
  ``syncStateNotFound``, whichever a test asks for.
- **Immutable ids**: a message keeps its id when it moves between folders, and the fake *refuses*
  (400) any request that did not send ``Prefer: IdType="ImmutableId"``.
- **A folder tree** with hidden folders, children that are only visible through ``childFolders``,
  and the well-known names Graph resolves (``junkemail``, ``deleteditems`` and so on).
- **A message deleted between list and get**: still listed, 404 ``ErrorItemNotFound`` when fetched.
- **429 with ``Retry-After``** in both its forms, and 503.
- **A profile whose identity can be switched**, as when a token belongs to another mailbox.
- **The calendar view**, with a window, change keys that move when an event is edited, occurrences
  of a recurring event, and an all-day event.
- **Entra's token endpoint**: refresh-token rotation (the old token stays valid, as Microsoft's
  does), ``invalid_grant`` and ``invalid_client``, an authorization-code exchange that checks the
  PKCE verifier and the redirect address, and a response ``scope`` a test can widen.
- **Concurrency**: the fake answers asynchronously and records the most requests it was ever asked
  to carry at once (:attr:`FakeGraph.peak_in_flight`), an account of the client's concurrency that
  does not trust the client's own.

Addresses are at ``example.org``.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import re
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from email.message import EmailMessage
from email.policy import SMTP
from urllib.parse import parse_qs, quote, unquote, urlencode, urlsplit

import httpx

from manicule.connectors.errors import SessionExpiredError

__all__ = [
    "ADDRESS",
    "CLIENT_ID",
    "FOLDERS",
    "OBJECT_ID",
    "TENANT",
    "FakeCredential",
    "FakeEvent",
    "FakeGraph",
    "FakeGraphSettings",
    "mail",
]

ADDRESS = "sync.user@example.org"
UPN = "sync.user@contoso.example.org"
"""The sign-in name; the primary address is :data:`ADDRESS`. They differ, as they do in tenants."""
OBJECT_ID = "11111111-2222-3333-4444-555555555555"
TENANT = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
CLIENT_ID = "99999999-8888-7777-6666-555555555555"

HOST = "graph.microsoft.com"
PREFIX = "/v1.0"
ENTRA_HOST = "login.microsoftonline.com"

REFRESH_TOKEN = "refresh-token-sentinel-7730"  # noqa: S105 - a fixture value

FOLDERS = {
    "Inbox": ("inbox", None, False),
    "Sent Items": ("sent", None, False),
    "Archive": ("archive", None, False),
    "2019": ("archive-2019", "archive", False),
    "Old Mail": ("archive-old", "archive", False),
    "Junk Email": ("junk", None, False),
    "Deleted Items": ("deleted", None, False),
    "Trash Subfolder": ("deleted-sub", "deleted", False),
    "Drafts": ("drafts", None, False),
    "Outbox": ("outbox", None, False),
    "Sync Issues": ("syncissues", None, True),
    "Conflicts": ("conflicts", "syncissues", True),
}
"""Display name to ``(id, parent id, hidden)``. The ids are readable so a failure reads."""

WELL_KNOWN = {
    "inbox": "inbox",
    "junkemail": "junk",
    "deleteditems": "deleted",
    "drafts": "drafts",
    "outbox": "outbox",
    "syncissues": "syncissues",
}


@dataclass(frozen=True, slots=True)
class FakeGraphSettings:
    """What :class:`~manicule.connectors.outlook_client.OutlookSettings` reads, at its defaults."""

    max_concurrent_requests: int = 4
    max_retry_after_seconds: float = 120.0
    max_retries: int = 5
    request_timeout_seconds: float = 30.0


@dataclass(slots=True)
class _Message:
    id: str
    folder: str
    raw: bytes
    conversation: str
    received: str
    changed_at: int
    """The sequence number of the last change to this message."""
    deleted: bool = False
    """Still listed, but a fetch is 404: removed between listing and fetching."""


@dataclass(slots=True)
class FakeEvent:
    """One event of the fake calendar."""

    id: str
    subject: str
    start: str
    end: str
    change_key: str = "ck-1"
    all_day: bool = False
    location: str = ""
    body: str = ""
    organizer: str = "Pat Example <pat@example.org>"
    attendees: tuple[str, ...] = ()
    cancelled: bool = False
    series_master: str | None = None
    ical_uid: str = ""


class FakeCredential:
    """A credential that mints tokens the fake accepts."""

    def __init__(self, server: FakeGraph) -> None:
        self._server = server
        self._token = server.issue_token()
        self.refreshes = 0

    async def access_token(self, *, refresh: bool = False) -> str:
        if refresh:
            self.refreshes += 1
            self._token = self._server.issue_token()
        return self._token

    def rejected(self) -> Exception:
        return SessionExpiredError("rejected; run `manicule connector login mail`")


def mail(
    subject: str, body: str = "The figures are in this sentence.", *, with_id: bool = True
) -> bytes:
    """A real message from the standard library's writer."""
    built = EmailMessage()
    built["From"] = "pat@example.org"
    built["To"] = ADDRESS
    built["Subject"] = subject
    built["Date"] = "Tue, 03 Feb 2026 09:14:00 +0000"
    if with_id:
        built["Message-ID"] = f"<{zlib.crc32(subject.encode()):08x}@example.org>"
    built.set_content(body + "\n")
    return built.as_bytes(policy=SMTP)


@dataclass(slots=True)
class FakeGraph:
    """Serves Graph and Entra, records every request, and lets a test break either."""

    page_size: int = 2

    requests: list[httpx.Request] = field(default_factory=list[httpx.Request])
    folders: dict[str, tuple[str, str | None, bool]] = field(
        default_factory=dict[str, tuple[str, str | None, bool]]
    )
    events: dict[str, FakeEvent] = field(default_factory=dict[str, FakeEvent])

    upn: str = UPN
    mail_address: str = ADDRESS
    object_id: str = OBJECT_ID

    # delta
    delta_floor: int = 0
    """Delta tokens recorded before this sequence number are gone."""
    expiry_status: int = 410
    expiry_code: str = "syncStateNotFound"
    loop_pagination: bool = False
    fail_paging: tuple[int, str] | None = None
    """Answer every ``$skiptoken`` request of a delta round with this ``(status, code)``."""
    before_request: Callable[[httpx.Request], None] | None = None
    """Called with each Graph request before it is answered, so a test can change the tenant
    between two calls the way a person editing a mailbox does."""

    # throttling and failure
    throttle: list[tuple[int, str | None]] = field(default_factory=list[tuple[int, str | None]])
    """Responses to give before answering normally: ``(status, retry_after)``."""
    in_flight: int = 0
    peak_in_flight: int = 0
    forbid_folder_listing: bool = False

    # entra
    refresh_failure: tuple[int, str] | None = None
    exchange_failure: tuple[int, str] | None = None
    rotate: bool = False
    granted_scope: str | None = None
    refreshes: int = 0
    exchanges: int = 0
    asked_scopes: list[str] = field(default_factory=list[str])
    authorizations: list[dict[str, str]] = field(default_factory=list[dict[str, str]])

    _messages: dict[str, _Message] = field(default_factory=dict[str, _Message])
    _seq: int = 1
    _removals: list[tuple[int, str, str]] = field(default_factory=list[tuple[int, str, str]])
    """``(sequence, folder, message id)`` for each message that left a folder."""
    _tokens: set[str] = field(default_factory=set[str])
    _refresh_tokens: set[str] = field(default_factory=lambda: {REFRESH_TOKEN})
    _codes: dict[str, tuple[str, str]] = field(default_factory=dict[str, tuple[str, str]])
    _issued: int = 0
    _next_message: int = 0

    def __post_init__(self) -> None:
        self.folders = dict(FOLDERS)

    # --- test controls --------------------------------------------------------------------

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def issue_token(self) -> str:
        self._issued += 1
        token = f"access-token-{self._issued}"
        self._tokens.add(token)
        return token

    def revoke_access_tokens(self) -> None:
        self._tokens.clear()

    def add_message(
        self, raw: bytes, folder: str = "Inbox", *, conversation: str = "conv-1"
    ) -> str:
        """A message in ``folder`` (by display name), returning its immutable id."""
        self._next_message += 1
        self._seq += 1
        message_id = f"AAMk{self._next_message:04d}="
        self._messages[message_id] = _Message(
            id=message_id,
            folder=self.folders[folder][0],
            raw=raw,
            conversation=conversation,
            received="2026-02-03T09:14:00Z",
            changed_at=self._seq,
        )
        return message_id

    def touch(self, message_id: str) -> None:
        """A flag change: the message is named by the next delta, its bytes unchanged."""
        self._seq += 1
        self._messages[message_id].changed_at = self._seq

    def move(self, message_id: str, folder: str) -> None:
        """Move a message: same id, new folder, a removal in the old one."""
        message = self._messages[message_id]
        self._seq += 1
        self._removals.append((self._seq, message.folder, message_id))
        message.folder = self.folders[folder][0]
        message.changed_at = self._seq

    def delete(self, message_id: str) -> None:
        message = self._messages.pop(message_id)
        self._seq += 1
        self._removals.append((self._seq, message.folder, message_id))

    def vanish_on_fetch(self, message_id: str) -> None:
        """Keep a message in every listing, but answer 404 when it is fetched."""
        self._messages[message_id].deleted = True

    def expire_deltas(self, status: int = 410, code: str = "syncStateNotFound") -> None:
        """Every delta token recorded so far stops working."""
        self.delta_floor = self._seq + 1
        self.expiry_status = status
        self.expiry_code = code
        self._seq += 1

    def delete_folder(self, name: str) -> None:
        folder_id = self.folders.pop(name)[0]
        for message_id in [m.id for m in self._messages.values() if m.folder == folder_id]:
            del self._messages[message_id]

    def add_event(self, event: FakeEvent) -> None:
        self.events[event.id] = event

    def edit_event(self, event_id: str, **changes: object) -> None:
        event = self.events[event_id]
        for key, value in changes.items():
            setattr(event, key, value)
        event.change_key = f"ck-{int(event.change_key.split('-')[1]) + 1}"

    def requests_to(self, fragment: str) -> list[httpx.Request]:
        return [r for r in self.requests if fragment in str(r.url)]

    def rotate_refresh_token(self) -> str:
        """A replacement refresh token the fake will accept, as a prior rotation would have."""
        token = f"refresh-token-rotated-{len(self._refresh_tokens)}"
        self._refresh_tokens.add(token)
        return token

    def consent(self, auth_url: str, *, approve: bool = True, state: str | None = None) -> str:
        """What the browser is redirected to after the person has chosen, as an address.

        Validates the request as Entra would for a public client: this tenant's authorization
        endpoint, a code response, an ``S256`` challenge, ``offline_access`` among the scopes and
        a loopback redirect on ``localhost`` or ``127.0.0.1``. ``state`` replaces the one the
        request carried, to play a forged redirect.
        """
        parts = urlsplit(auth_url)
        assert f"{parts.scheme}://{parts.netloc}{parts.path}" == (
            f"https://{ENTRA_HOST}/{TENANT}/oauth2/v2.0/authorize"
        )
        asked = {k: v[0] for k, v in parse_qs(parts.query).items()}
        self.authorizations.append(asked)
        assert asked["response_type"] == "code"
        assert asked["code_challenge_method"] == "S256"
        assert "offline_access" in asked["scope"].split()
        redirect = urlsplit(asked["redirect_uri"])
        assert redirect.scheme == "http"
        assert redirect.hostname in {"localhost", "127.0.0.1"}
        assert redirect.port is not None
        sent_state = asked["state"] if state is None else state
        if not approve:
            return f"{asked['redirect_uri']}/?" + urlencode(
                {"error": "access_denied", "state": sent_state}
            )
        code = self.authorization_code(asked["code_challenge"], asked["redirect_uri"])
        return f"{asked['redirect_uri']}/?" + urlencode({"code": code, "state": sent_state})

    def authorization_code(self, challenge: str, redirect_uri: str) -> str:
        """A code Entra would have issued for a sign-in with this challenge and redirect."""
        code = f"code-{len(self._codes)}"
        self._codes[code] = (challenge, redirect_uri)
        return code

    # --- the wire -------------------------------------------------------------------------

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0)
            if request.url.host == ENTRA_HOST:
                return self._entra(request)
            if request.url.host != HOST:
                return httpx.Response(404)
            return self._graph(request)
        finally:
            self.in_flight -= 1

    # --- entra ----------------------------------------------------------------------------

    def _entra(self, request: httpx.Request) -> httpx.Response:
        match = re.fullmatch(r"/([^/]+)/oauth2/v2\.0/token", request.url.path)
        if match is None or request.method != "POST":
            return httpx.Response(404)
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        if "client_secret" in form:
            return _entra_error(400, "invalid_request")
        self.asked_scopes = form.get("scope", "").split()
        if form.get("grant_type") == "authorization_code":
            return self._exchange(form)
        if self.refresh_failure is not None:
            return _entra_error(*self.refresh_failure)
        if form.get("refresh_token") not in self._refresh_tokens:
            return _entra_error(400, "invalid_grant")
        self.refreshes += 1
        body: dict[str, object] = {
            "token_type": "Bearer",
            "expires_in": 3599,
            "access_token": self.issue_token(),
            "scope": self.granted_scope or " ".join(self.asked_scopes),
        }
        if self.rotate:
            body["refresh_token"] = self.rotate_refresh_token()
        return httpx.Response(200, json=body)

    def _exchange(self, form: dict[str, str]) -> httpx.Response:
        self.exchanges += 1
        if self.exchange_failure is not None:
            return _entra_error(*self.exchange_failure)
        issued = self._codes.get(form.get("code", ""))
        if issued is None:
            return _entra_error(400, "invalid_grant")
        challenge, redirect_uri = issued
        verifier = form.get("code_verifier", "")
        digest = hashlib.sha256(verifier.encode("ascii")).digest()
        expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        if expected != challenge or form.get("redirect_uri") != redirect_uri:
            return _entra_error(400, "invalid_grant")
        token = self.rotate_refresh_token()
        return httpx.Response(
            200,
            json={
                "token_type": "Bearer",
                "expires_in": 3599,
                "access_token": self.issue_token(),
                "refresh_token": token,
                "scope": self.granted_scope or _prefixed(form["scope"]),
            },
        )

    # --- graph ----------------------------------------------------------------------------

    def _graph(self, request: httpx.Request) -> httpx.Response:
        if self.before_request is not None:
            self.before_request(request)
        if self.throttle:
            status, retry_after = self.throttle.pop(0)
            headers = {"Retry-After": retry_after} if retry_after is not None else {}
            return httpx.Response(status, headers=headers, json=_error(status, "TooManyRequests"))
        bearer = request.headers.get("authorization", "").removeprefix("Bearer ")
        if bearer not in self._tokens:
            return httpx.Response(401, json=_error(401, "InvalidAuthenticationToken"))
        path = unquote(request.url.path).removeprefix(PREFIX)
        if path != "/me" and 'IdType="ImmutableId"' not in request.headers.get("prefer", ""):
            return httpx.Response(400, json=_error(400, "ImmutableIdRequired"))
        params = _params(request)
        routes: tuple[tuple[str, Callable[..., httpx.Response]], ...] = (
            (r"/me", self._me),
            (r"/me/mailFolders", self._folders_top),
            (r"/me/mailFolders/([^/]+)/childFolders", self._folders_children),
            (r"/me/mailFolders\('([^']+)'\)/messages/delta", self._delta),
            (r"/me/mailFolders/([^/]+)/messages/delta", self._delta),
            (r"/me/mailFolders/([^/]+)/messages", self._list),
            (r"/me/mailFolders/([^/]+)", self._folder),
            (r"/me/messages/([^/]+)/\$value", self._value),
            (r"/me/messages/([^/]+)", self._message),
            (r"/me/calendarView", self._calendar),
            (r"/me/events/([^/]+)", self._event),
        )
        for pattern, handler in routes:
            match = re.fullmatch(pattern, path, flags=re.IGNORECASE)
            if match is not None:
                return handler(params, *match.groups())
        return httpx.Response(404, json=_error(404, "ErrorInvalidResource"))

    def _me(self, params: dict[str, str]) -> httpx.Response:
        del params
        return httpx.Response(
            200,
            json={"id": self.object_id, "userPrincipalName": self.upn, "mail": self.mail_address},
        )

    def _folder_json(self, name: str) -> dict[str, object]:
        folder_id = self.folders[name][0]
        children = sum(1 for _, parent, _ in self.folders.values() if parent == folder_id)
        return {"id": folder_id, "displayName": name, "childFolderCount": children}

    def _folders_top(self, params: dict[str, str]) -> httpx.Response:
        if self.forbid_folder_listing:
            return httpx.Response(403, json=_error(403, "ErrorAccessDenied"))
        rows = [
            self._folder_json(name)
            for name, (_, parent, hidden) in self.folders.items()
            if parent is None and (not hidden or params.get("includeHiddenFolders") == "true")
        ]
        return self._page(rows, params, "/me/mailFolders")

    def _folders_children(self, params: dict[str, str], parent: str) -> httpx.Response:
        rows = [
            self._folder_json(name)
            for name, (_, owner, hidden) in self.folders.items()
            if owner == parent and (not hidden or params.get("includeHiddenFolders") == "true")
        ]
        return self._page(rows, params, f"/me/mailFolders/{parent}/childFolders")

    def _folder(self, params: dict[str, str], name: str) -> httpx.Response:
        del params
        folder_id = WELL_KNOWN.get(name.casefold(), name)
        if not any(entry[0] == folder_id for entry in self.folders.values()):
            return httpx.Response(404, json=_error(404, "ErrorItemNotFound"))
        return httpx.Response(200, json={"id": folder_id})

    def _in_folder(self, folder_id: str) -> list[_Message]:
        return [m for m in self._messages.values() if m.folder == folder_id]

    def _stub(self, message: _Message) -> dict[str, object]:
        return {
            "id": message.id,
            "conversationId": message.conversation,
            "receivedDateTime": message.received,
            "webLink": f"https://outlook.office365.com/owa/?ItemID={quote(message.id, safe='')}",
        }

    def _list(self, params: dict[str, str], folder_id: str) -> httpx.Response:
        if not any(entry[0] == folder_id for entry in self.folders.values()):
            return httpx.Response(404, json=_error(404, "ErrorItemNotFound"))
        rows: list[dict[str, object]] = [{"id": m.id} for m in self._in_folder(folder_id)]
        return self._page(rows, params, f"/me/mailFolders/{folder_id}/messages")

    def _delta(self, params: dict[str, str], folder_id: str) -> httpx.Response:
        if not any(entry[0] == folder_id for entry in self.folders.values()):
            return httpx.Response(404, json=_error(404, "ErrorItemNotFound"))
        floor_hit = False
        since = 0
        offset = 0
        if "$deltatoken" in params or "$skiptoken" in params:
            raw = params.get("$deltatoken") or params["$skiptoken"]
            parts = raw.split("-")
            if len(parts) < 3 or parts[0] not in {"D", "S"}:
                return httpx.Response(400, json=_error(400, "BadRequest"))
            since = int(parts[1])
            offset = int(parts[2]) if parts[0] == "S" else 0
            floor_hit = since < self.delta_floor and parts[0] == "D"
        if self.fail_paging is not None and "$skiptoken" in params:
            status, code = self.fail_paging
            return httpx.Response(status, json=_error(status, code))
        if floor_hit:
            return httpx.Response(
                self.expiry_status, json=_error(self.expiry_status, self.expiry_code)
            )
        rows: list[dict[str, object]] = [
            self._stub(m) for m in self._in_folder(folder_id) if m.changed_at > since
        ]
        if since > 0:
            removed: list[dict[str, object]] = [
                {"id": mid, "@removed": {"reason": "deleted"}}
                for seq, fid, mid in self._removals
                if fid == folder_id and seq > since
            ]
            rows += removed
        head = self._seq
        size = self.page_size
        page = rows[offset : offset + size]
        base = f"https://{HOST}{PREFIX}/me/mailFolders('{folder_id}')/messages/delta"
        if offset + size < len(rows):
            following = offset if self.loop_pagination else offset + size
            return httpx.Response(
                200,
                json={"value": page, "@odata.nextLink": f"{base}?$skiptoken=S-{since}-{following}"},
            )
        return httpx.Response(
            200, json={"value": page, "@odata.deltaLink": f"{base}?$deltatoken=D-{head}-0"}
        )

    def _value(self, params: dict[str, str], message_id: str) -> httpx.Response:
        del params
        message = self._messages.get(message_id)
        if message is None or message.deleted:
            return httpx.Response(404, json=_error(404, "ErrorItemNotFound"))
        return httpx.Response(200, content=message.raw, headers={"Content-Type": "text/plain"})

    def _message(self, params: dict[str, str], message_id: str) -> httpx.Response:
        del params
        message = self._messages.get(message_id)
        if message is None:
            return httpx.Response(404, json=_error(404, "ErrorItemNotFound"))
        return httpx.Response(200, json=self._stub(message))

    def _calendar(self, params: dict[str, str]) -> httpx.Response:
        start, end = params["startDateTime"], params["endDateTime"]
        rows: list[dict[str, object]] = [
            {
                "id": e.id,
                "changeKey": e.change_key,
                "webLink": f"https://outlook.office365.com/owa/?itemid={quote(e.id, safe='')}",
            }
            for e in self.events.values()
            if e.start < end and e.end > start
        ]
        return self._page(rows, params, "/me/calendarView")

    def _event(self, params: dict[str, str], event_id: str) -> httpx.Response:
        del params
        event = self.events.get(event_id)
        if event is None:
            return httpx.Response(404, json=_error(404, "ErrorItemNotFound"))

        def person(text: str) -> dict[str, object]:
            name, _, address = text.partition(" <")
            return {"emailAddress": {"name": name, "address": address.rstrip(">")}}

        return httpx.Response(
            200,
            json={
                "id": event.id,
                "iCalUId": event.ical_uid or f"uid-{event.id}",
                "changeKey": event.change_key,
                "subject": event.subject,
                "isAllDay": event.all_day,
                "start": {"dateTime": event.start + ".0000000", "timeZone": "UTC"},
                "end": {"dateTime": event.end + ".0000000", "timeZone": "UTC"},
                "location": {"displayName": event.location},
                "organizer": person(event.organizer),
                "attendees": [person(a) for a in event.attendees],
                "body": {"contentType": "text", "content": event.body},
                "isCancelled": event.cancelled,
                "webLink": f"https://outlook.office365.com/owa/?itemid={quote(event.id, safe='')}",
                "createdDateTime": "2026-01-05T10:00:00Z",
            },
        )

    def _page(
        self, rows: list[dict[str, object]], params: dict[str, str], path: str
    ) -> httpx.Response:
        offset = int(params["$skiptoken"].removeprefix("P-")) if "$skiptoken" in params else 0
        size = self.page_size
        body: dict[str, object] = {"value": rows[offset : offset + size]}
        if offset + size < len(rows):
            following = offset if self.loop_pagination else offset + size
            token = f"P-{following}"
            extra = {k: v for k, v in params.items() if k != "$skiptoken"}
            query = "&".join(
                f"{k}={quote(v, safe='')}" for k, v in {**extra, "$skiptoken": token}.items()
            )
            body["@odata.nextLink"] = f"https://{HOST}{PREFIX}{path}?{query}"
        return httpx.Response(200, json=body)


def _params(request: httpx.Request) -> dict[str, str]:
    return {
        k: v[0] for k, v in parse_qs(request.url.query.decode(), keep_blank_values=True).items()
    }


def _prefixed(scope: str) -> str:
    """Graph's scopes as Entra reports them: prefixed, the OpenID ones bare."""
    return " ".join(f"https://graph.microsoft.com/{s}" if "." in s else s for s in scope.split())


def _error(status: int, code: str) -> dict[str, object]:
    return {"error": {"code": code, "message": f"sentinel prose for {status}"}}


def _entra_error(status: int, code: str) -> httpx.Response:
    return httpx.Response(
        status, json={"error": code, "error_description": "AADSTS sentinel-prose user@example.org"}
    )
