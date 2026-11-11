"""A synthetic Trilium Notes server: the traps a real one has, behind an ``httpx`` transport.

Every Trilium suite runs the **real** client, with its real retry loop, headers and query
encoding, against this, so none needs a Trilium server or the network. It reproduces, each
switchable from a test:

- **A tree with clones.** A note can have several parents, and ``root`` and Trilium's own
  ``_hidden`` subtree are in the results like any other note.
- **Protected notes**, whose title is the literal ``[protected]`` and whose content request is
  a 400 ``NOTE_IS_PROTECTED``.
- **A search that is cut off**, silently, at a limit the server chooses.
- **A proxy in front of Trilium** that answers an unauthenticated request by redirecting to its
  sign-in, or answers a content request with a sign-in page that is a 200 of ``text/html``.
- **429 with ``Retry-After``**, and a token the server will not accept.

Every host is ``example.org``. Titles and content are invented.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

__all__ = [
    "BASE_URL",
    "TOKEN",
    "FakeNote",
    "FakeTrilium",
    "Waits",
    "vault",
]

BASE_URL = "https://notes.example.org"
TOKEN = "etapi-test-token"  # noqa: S105 - a fixture, accepted by the fake and nothing else
_STAMP = "2026-01-01 00:00:00.000Z"


@dataclass(slots=True)
class FakeNote:
    """One note as the fake serves it."""

    note_id: str
    title: str
    parents: tuple[str, ...] = ("root",)
    type: str = "text"
    mime: str = "text/html"
    content: str = "<p>body</p>"
    protected: bool = False
    labels: Mapping[str, str | Sequence[str]] = field(default_factory=lambda: {})
    modified: str = _STAMP
    created: str = _STAMP

    @property
    def blob_id(self) -> str:
        return hashlib.sha1(self.content.encode(), usedforsecurity=False).hexdigest()[:20]


class FakeTrilium:
    """A Trilium server with a handful of notes and the faults a test can switch on."""

    def __init__(self, *, base_url: str = BASE_URL, token: str = TOKEN) -> None:
        self.base_url = base_url
        self.token = token
        self.notes: dict[str, FakeNote] = {}
        self.add(FakeNote("root", "root", parents=(), type="book", content=""))
        self.add(FakeNote("_hidden", "Hidden Notes", type="book", content=""))
        self.cap: int | None = None
        """When set, a search silently returns at most this many notes."""
        self.proxy_redirect: str | None = None
        """When set, every request is answered by a redirect to this URL."""
        self.signin_for_content = False
        """When set, a content request is answered by a 200 sign-in page of ``text/html``."""
        self.throttle: list[str | None] = []
        """Each entry is one 429 to send first, with that ``Retry-After`` (``None`` for none)."""
        self.failures: list[int | None] = []
        """Each entry is one failure to send first: a status code, or ``None`` to drop the
        connection."""
        self.extra_results: list[dict[str, object]] = []
        """Entries appended to a search response as they are, for a shape ETAPI would not send."""
        self.requests: list[tuple[str, str]] = []
        """Every request as ``(path, query)``, in order."""
        self.authorizations: list[str] = []
        """The ``Authorization`` header of every request."""

    # --- building the server -------------------------------------------------------------

    def add(self, note: FakeNote) -> FakeNote:
        self.notes[note.note_id] = note
        return note

    def edit(
        self, note_id: str, content: str, *, modified: str = "2026-02-01 00:00:00.000Z"
    ) -> None:
        note = self.notes[note_id]
        note.content = content
        note.modified = modified

    def rename(self, note_id: str, title: str) -> None:
        self.notes[note_id].title = title

    def delete(self, note_id: str) -> None:
        del self.notes[note_id]

    def content_requests(self) -> list[str]:
        """The ids whose content was asked for, in order."""
        found: list[str] = []
        for path, _ in self.requests:
            parts = path.split("/")
            if len(parts) == 5 and parts[2] == "notes" and parts[4] == "content":
                found.append(parts[3])
        return found

    # --- serving -------------------------------------------------------------------------

    def transport(self) -> httpx.MockTransport:
        """The transport. It yields to the event loop before answering, as a network does, so a
        test of concurrent callers actually interleaves them."""

        async def answer(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(0)
            return self.handle(request)

        return httpx.MockTransport(answer)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.url.path, request.url.query.decode()))
        self.authorizations.append(request.headers.get("authorization", ""))
        if self.proxy_redirect is not None:
            return httpx.Response(302, headers={"location": self.proxy_redirect})
        if self.failures:
            failure = self.failures.pop(0)
            if failure is None:
                raise httpx.ConnectError("connection dropped", request=request)
            return httpx.Response(failure)
        if self.throttle:
            wait = self.throttle.pop(0)
            return httpx.Response(429, headers={} if wait is None else {"retry-after": wait})
        if request.headers.get("authorization") != f"Bearer {self.token}":
            return _error(401, "UNAUTHORIZED", "token is invalid")
        return self._route(request)

    def _route(self, request: httpx.Request) -> httpx.Response:
        """The ETAPI routes, served under whatever context path the base URL carries."""
        prefix = urlsplit(self.base_url).path.rstrip("/")
        path = request.url.path.removeprefix(prefix)
        parts = path.split("/")
        if path == "/etapi/notes":
            return self._search(request)
        if len(parts) == 5 and parts[:3] == ["", "etapi", "notes"] and parts[4] == "content":
            return self._content(parts[3])
        return _error(404, "NOT_FOUND", "no such route")

    def _search(self, request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if not params.get("search"):
            return _error(400, "SEARCH_REQUIRED", "search is required")
        limit = int(params.get("limit", "0")) or len(self.notes)
        if self.cap is not None:
            limit = min(limit, self.cap)
        ordered = sorted(self.notes.values(), key=lambda note: note.modified, reverse=True)
        results = [self._json(note) for note in ordered[:limit]]
        return httpx.Response(200, json={"results": [*results, *self.extra_results]})

    def _json(self, note: FakeNote) -> dict[str, object]:
        children = [other.note_id for other in self.notes.values() if note.note_id in other.parents]
        return {
            "noteId": note.note_id,
            "title": "[protected]" if note.protected else note.title,
            "type": note.type,
            "mime": note.mime,
            "isProtected": note.protected,
            "blobId": note.blob_id,
            "attributes": [
                {
                    "attributeId": f"{note.note_id}-{name}-{position}",
                    "noteId": note.note_id,
                    "type": "label",
                    "name": name,
                    "value": value,
                    "position": 10 * (position + 1),
                    "isInheritable": False,
                }
                for name, given in note.labels.items()
                for position, value in enumerate([given] if isinstance(given, str) else given)
            ],
            "parentNoteIds": list(note.parents),
            "childNoteIds": children,
            "parentBranchIds": [],
            "childBranchIds": [],
            "dateCreated": note.created,
            "dateModified": note.modified,
            "utcDateCreated": note.created,
            "utcDateModified": note.modified,
        }

    def _content(self, note_id: str) -> httpx.Response:
        if self.signin_for_content:
            return httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                content=b"<html><title>Sign in</title><form>password</form></html>",
            )
        note = self.notes.get(note_id)
        if note is None:
            return _error(404, "NOTE_NOT_FOUND", "no such note")
        if note.protected:
            return _error(400, "NOTE_IS_PROTECTED", "note is protected")
        return httpx.Response(
            200,
            headers={"content-type": f"{note.mime}; charset=utf-8"},
            content=note.content.encode(),
        )


class Waits:
    """A stand-in for sleeping that records what it was asked to wait for."""

    def __init__(self) -> None:
        self.seconds: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.seconds.append(seconds)


def vault() -> FakeTrilium:
    """A small vault: two top-level folders, a note in each, and a clone between them."""
    server = FakeTrilium()
    server.add(FakeNote("work", "Work", type="book"))
    server.add(FakeNote("home", "Home", type="book"))
    server.add(FakeNote("plan", "Plan", parents=("work",), content="<p>quarterly plan</p>"))
    server.add(FakeNote("garden", "Garden", parents=("home",), content="<p>tomatoes</p>"))
    return server


def _error(status: int, code: str, message: str) -> httpx.Response:
    return httpx.Response(
        status,
        headers={"content-type": "application/json"},
        content=json.dumps({"status": status, "code": code, "message": message}).encode(),
    )
