"""A synthetic Confluence, served over an ``httpx`` transport.

There are no credentials for a real instance in this repository and there will not be, so the
connector is held to the traps in ``docs/connectors/confluence.md`` by a source that reproduces
them. Everything below is shaped by that: it is not a stub that returns whatever the connector
asks for, it is a server that **behaves badly in the specific ways Confluence does**.

- **Cursors contain ``+``.** The one character that breaks pagination is in every cursor this
  fake issues, and an unrecognized cursor is answered with **the first page again** rather than
  an error — which is what makes the failure silent in the first place. A client that corrupts
  a cursor therefore loops over the opening of a space forever instead of raising.
- **Version disagreement.** A page can be told to serve a body older than the version search
  reported, exactly once or forever, so both the retry and the storage-format fallback are
  exercised.
- **Throttling.** A queue of ``429`` responses with ``Retry-After``, consumed by the requests
  that arrive.
- **Deletion.** Pages can be removed between one sync and the next, which is invisible to a
  watermark query by construction.
- **Signing out.** An instance behind an identity provider answers a request it will not serve
  with a **sign-in page carrying status 200**, or with a redirect to the provider. Both are
  available per path prefix, so a test can sign out only the attachment endpoint — where a
  sign-in page has no JSON decoder to fall foul of and would otherwise be indexed as a document.

The transport records every request, so a test can assert on what actually went over the wire —
the raw query string included, which is where the ``%2B`` either is or is not, and the hosts
that were never asked, which is where an identity provider either was or was not contacted.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

CLOUD_BASE = "https://example.atlassian.net/wiki"
SERVER_BASE = "https://wiki.example.com/confluence"

IDENTITY_PROVIDER = "https://idp.example.com"
"""Where an instance behind single sign-on sends a client it no longer recognizes."""

SIGNIN_PAGE = (
    "<!DOCTYPE html><html><head><title>Log in - Confluence</title></head><body>"
    '<form name="loginform" action="/dologin.action" method="post">'
    '<input type="text" name="os_username" id="os_username">'
    '<input type="password" name="os_password" id="os_password">'
    '<input type="submit" name="login" value="Log in">'
    "</form></body></html>"
)
"""What an instance serves instead of an answer once the session is gone.

Deliberately a *complete, plausible* page rather than a stub. It has a title, a form and real
markup; parsed, chunked and embedded it would produce a perfectly ordinary-looking document,
which is the entire reason indexing one is the worst available outcome. It carries no
authentication headers, so a test that uses it exercises the check that reads the **body** —
the header signals are tested separately, and each has to be able to fail on its own.
"""

_SPACE = re.compile(r'space\s*=\s*"((?:[^"\\]|\\.)*)"')
_TITLE = re.compile(r'title\s*=\s*"((?:[^"\\]|\\.)*)"')
_SINCE = re.compile(r'lastmodified\s*>=\s*"([^"]*)"')
_TYPES = re.compile(r"type\s*(?:=\s*(\w+)|in\s*\(([^)]*)\))")


@dataclass(slots=True)
class FakePage:
    """One page. ``adf`` and ``storage`` are the two bodies the same page can be read as."""

    id: str
    title: str
    space: str
    version: int = 1
    when: str = "2026-08-09T14:30:00.000+01:00"
    adf: Mapping[str, object] = field(default_factory=lambda: paragraph("body text"))
    storage: str = "<p>body text</p>"
    ancestors: tuple[str, ...] = ()
    """Ancestor titles, outermost first."""

    served_version: int | None = None
    """Version the *body* endpoints report, when it disagrees with what search reports."""

    stale_once: bool = False
    """Whether the disagreement clears on a second request, as a caching artifact would."""

    adf_available: bool = True
    """Whether the Cloud body endpoint returns an Atlassian Document Format body at all.

    It has been seen to decline the format for a page that exists, which is a different thing
    from the page being empty and wants a different answer.
    """

    storage_version: int | None = None
    """Version the storage-format endpoint reports. ``None`` means it agrees with search.

    Set alongside ``served_version`` to model the case where *no* format is fresh, which is
    what decides whether the connector records the version it asked for or the one it got.
    """


@dataclass(slots=True)
class FakeAttachment:
    id: str
    title: str
    space: str
    page_id: str
    page_title: str
    content: bytes = b"%PDF-1.4 fake"
    media_type: str = "application/pdf"
    version: int = 1
    when: str = "2026-08-09T14:31:00.000+01:00"

    download_link: str = ""
    """Overrides the download link this attachment advertises.

    A response can name any URL it likes, and the client attaches the sync account's
    credential to whatever it is given — so a fixture has to be able to name a hostile one.
    """


def paragraph(text: str) -> dict[str, object]:
    """The smallest complete ADF document."""
    return {
        "version": 1,
        "type": "doc",
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": text}]},
        ],
    }


def with_include(
    text: str, *, title: str, space: str = "", macro: str = "include"
) -> dict[str, object]:
    """An ADF document whose second block is an include macro."""
    params: dict[str, object] = {"": {"value": title}}
    if space:
        params["spaceKey"] = {"value": space}
    return {
        "version": 1,
        "type": "doc",
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": text}]},
            {
                "type": "extension",
                "attrs": {
                    "extensionType": "com.atlassian.confluence.macro.core",
                    "extensionKey": macro,
                    "parameters": {"macroParams": params},
                },
            },
        ],
    }


def storage_include(text: str, *, title: str, space: str = "", macro: str = "include") -> str:
    """A storage-format body whose second block is an include macro."""
    space_param = f'<ac:parameter ac:name="spaceKey">{space}</ac:parameter>' if space else ""
    return (
        f"<p>{text}</p>"
        f'<ac:structured-macro ac:name="{macro}" ac:schema-version="1">'
        f'<ac:parameter ac:name=""><ac:link>'
        f'<ri:page ri:content-title="{title}"{f' ri:space-key="{space}"' if space else ""}/>'
        f"</ac:link></ac:parameter>{space_param}"
        f"</ac:structured-macro>"
    )


class FakeConfluence:
    """A Confluence instance that exists only in this process."""

    def __init__(
        self,
        *,
        base_url: str = CLOUD_BASE,
        pages: Sequence[FakePage] = (),
        attachments: Sequence[FakeAttachment] = (),
        spaces: Mapping[str, str] | None = None,
        page_size: int = 2,
        cursor_seed: str = "cur",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.context = httpx.URL(self.base_url).path.rstrip("/")
        self.pages = {page.id: page for page in pages}
        self.attachments = {item.id: item for item in attachments}
        self.spaces = dict(spaces or {page.space: f"{page.space} space" for page in pages})
        self.page_size = page_size
        self.requests: list[httpx.Request] = []
        self.throttles: list[Mapping[str, str]] = []
        """Queued 429 responses; each request consumes one before being served normally."""

        self.signed_out_paths: list[str] = []
        """Path prefixes answered with :data:`SIGNIN_PAGE` and status 200."""

        self.redirected_paths: list[tuple[str, str]] = []
        """Path prefixes answered with a 302 to the second element."""

        self.user = "sync.user"
        """Who ``/rest/api/user/current`` says this session is."""

        self.headers: dict[str, str] = {}
        """Extra headers on every response — ``X-AUSERNAME`` and Seraph's, in the tests that
        exercise them. Empty by default, so a fixture that signs out has to be caught by
        something other than a header it did not set."""

        self.body_calls: dict[str, int] = {}
        self._cursors: dict[str, tuple[str, int]] = {}
        self._cursor_seed = cursor_seed
        self._issued = 0

    # --- test-side helpers ---------------------------------------------------------------

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def throttle(self, times: int = 1, retry_after: str = "2") -> None:
        self.throttles.extend({"Retry-After": retry_after} for _ in range(times))

    def sign_out(self, path: str = "/") -> None:
        """Answer everything under ``path`` with a sign-in page and status 200.

        The default signs out the whole instance. Passing ``/download/`` signs out only the
        attachment endpoint, which is the case with no JSON decoder standing between a sign-in
        page and the index.
        """
        self.signed_out_paths.append(path)

    def redirect(self, location: str, path: str = "/") -> None:
        """Answer everything under ``path`` with a 302 to ``location``."""
        self.redirected_paths.append((path, location))

    def delete(self, page_id: str) -> None:
        """Remove a page the way a user does: it simply stops appearing in CQL results."""
        self.pages.pop(page_id, None)

    def queries(self) -> list[str]:
        """Every CQL query that was asked, in order."""
        return [
            request.url.params["cql"] for request in self.requests if "cql" in request.url.params
        ]

    def raw_queries(self) -> list[str]:
        """Every request's query string **exactly as it went over the wire**."""
        return [request.url.query.decode() for request in self.requests]

    # --- routing -------------------------------------------------------------------------

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.throttles:
            return httpx.Response(429, headers=dict(self.throttles.pop(0)), json={})

        path = request.url.path
        if self.context and path.startswith(self.context):
            path = path[len(self.context) :]

        for prefix, location in self.redirected_paths:
            if path.startswith(prefix):
                return httpx.Response(302, headers={"location": location, **self.headers})
        if any(path.startswith(prefix) for prefix in self.signed_out_paths):
            # 200, not 401 and not 403. This is what makes the failure worth guarding: every
            # ordinary check a client makes passes, and what it has is a document.
            return httpx.Response(
                200,
                text=SIGNIN_PAGE,
                headers={"content-type": "text/html;charset=UTF-8", **self.headers},
            )

        response = self._route(request, path)
        if self.headers:
            response.headers.update(self.headers)
        return response

    def _route(self, request: httpx.Request, path: str) -> httpx.Response:  # noqa: PLR0911
        if path == "/rest/api/space":
            return self._spaces(request)
        if path == "/rest/api/user/current":
            return httpx.Response(200, json={"type": "known", "username": self.user})
        if path == "/rest/api/content/search":
            return self._search(request)
        if path.startswith("/api/v2/pages/"):
            return self._v2_page(path)
        if path.startswith("/rest/api/content/"):
            return self._v1_page(path)
        if path.startswith("/download/"):
            return self._download(path)
        return httpx.Response(404, json={"message": f"no route for {path}"})

    # --- endpoints -----------------------------------------------------------------------

    def _spaces(self, request: httpx.Request) -> httpx.Response:
        rows: list[dict[str, object]] = [
            {"id": index, "key": key, "name": name, "type": "global"}
            for index, (key, name) in enumerate(sorted(self.spaces.items()))
        ]
        return self._paged(request, rows, "/rest/api/space")

    def _search(self, request: httpx.Request) -> httpx.Response:
        query = request.url.params.get("cql", "")
        expanded = request.url.params.get("expand", "")
        rows = [self._result(item, expanded) for item in self._matching(query)]
        return self._paged(request, rows, "/rest/api/content/search")

    def _matching(self, query: str) -> list[FakePage | FakeAttachment]:
        space = _unescape(_SPACE.search(query))
        title = _unescape(_TITLE.search(query))
        since = _SINCE.search(query)
        kinds = _kinds(query)

        found: list[FakePage | FakeAttachment] = []
        if "page" in kinds:
            found.extend(
                page
                for page in self.pages.values()
                if (not space or page.space == space)
                and (not title or page.title == title)
                and _after(page.when, since.group(1) if since else None)
            )
        if "attachment" in kinds and not title:
            found.extend(
                item
                for item in self.attachments.values()
                if (not space or item.space == space)
                and item.page_id in self.pages
                and _after(item.when, since.group(1) if since else None)
            )
        return sorted(found, key=lambda item: (item.when, item.id))

    def _result(self, item: FakePage | FakeAttachment, expand: str) -> dict[str, object]:
        if isinstance(item, FakePage):
            row: dict[str, object] = {
                "id": item.id,
                "type": "page",
                "status": "current",
                "title": item.title,
                "version": {"number": item.version, "when": item.when},
                "_links": {"webui": f"/spaces/{item.space}/pages/{item.id}/{item.title}"},
            }
            if "space" in expand:
                row["space"] = {"key": item.space, "name": self.spaces.get(item.space, "")}
            if "ancestors" in expand:
                row["ancestors"] = [
                    {"id": f"anc-{index}", "title": title}
                    for index, title in enumerate(item.ancestors)
                ]
            return row
        parent = self.pages.get(item.page_id)
        return {
            "id": item.id,
            "type": "attachment",
            "status": "current",
            "title": item.title,
            "version": {"number": item.version, "when": item.when},
            "metadata": {"mediaType": item.media_type},
            "extensions": {"fileSize": len(item.content)},
            "space": {"key": item.space},
            "container": {
                "id": item.page_id,
                "title": parent.title if parent else item.page_title,
                "_links": {"webui": f"/spaces/{item.space}/pages/{item.page_id}"},
            },
            "_links": {
                "webui": f"/spaces/{item.space}/pages/{item.page_id}/{item.title}",
                "download": item.download_link
                or f"/download/attachments/{item.page_id}/{item.title}",
            },
        }

    def _v2_page(self, path: str) -> httpx.Response:
        rest = path.removeprefix("/api/v2/pages/")
        page_id, _, tail = rest.partition("/")
        page = self.pages.get(page_id)
        if page is None:
            return httpx.Response(404, json={"message": "no such page"})
        if tail == "ancestors":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"id": f"anc-{index}", "type": "page", "title": title}
                        for index, title in enumerate(page.ancestors)
                    ]
                },
            )
        version = self._served_version(page)
        body: dict[str, object] = (
            {"atlas_doc_format": {"value": json.dumps(page.adf)}} if page.adf_available else {}
        )
        return httpx.Response(
            200,
            json={
                "id": page.id,
                "title": page.title,
                "status": "current",
                "version": {"number": version},
                "body": body,
                "_links": {
                    "base": self.base_url,
                    "webui": f"/spaces/{page.space}/pages/{page.id}/{page.title}",
                },
            },
        )

    def _v1_page(self, path: str) -> httpx.Response:
        page_id = path.removeprefix("/rest/api/content/")
        page = self.pages.get(page_id)
        if page is None:
            return httpx.Response(404, json={"message": "no such page"})
        return httpx.Response(
            200,
            json={
                "id": page.id,
                "type": "page",
                "title": page.title,
                "space": {"key": page.space, "name": self.spaces.get(page.space, "")},
                "version": {
                    "number": (
                        page.storage_version if page.storage_version is not None else page.version
                    ),
                    "when": page.when,
                },
                "ancestors": [
                    {"id": f"anc-{index}", "title": title}
                    for index, title in enumerate(page.ancestors)
                ],
                "body": {"storage": {"value": page.storage, "representation": "storage"}},
                "_links": {
                    "base": self.base_url,
                    "webui": f"/spaces/{page.space}/pages/{page.id}/{page.title}",
                },
            },
        )

    def _served_version(self, page: FakePage) -> int:
        """The version the body endpoint reports, honoring a configured disagreement."""
        seen = self.body_calls.get(page.id, 0)
        self.body_calls[page.id] = seen + 1
        if page.served_version is None:
            return page.version
        if page.stale_once and seen >= 1:
            return page.version
        return page.served_version

    def _download(self, path: str) -> httpx.Response:
        name = path.rsplit("/", 1)[-1]
        for item in self.attachments.values():
            if item.title == name:
                return httpx.Response(
                    200, content=item.content, headers={"content-type": item.media_type}
                )
        return httpx.Response(404, json={"message": "no such attachment"})

    # --- pagination ----------------------------------------------------------------------

    def _paged(
        self, request: httpx.Request, rows: Sequence[Mapping[str, object]], path: str
    ) -> httpx.Response:
        """One page of results, with a cursor that contains the character that breaks clients.

        An unrecognized cursor is answered with the **first** page rather than an error. That
        is the shape of the real failure: a client that mangles a cursor does not get told, it
        gets plausible results forever.
        """
        limit = int(request.url.params.get("limit", str(self.page_size)) or self.page_size)
        limit = min(limit, self.page_size)
        cursor = request.url.params.get("cursor")
        start = self._cursors.get(cursor or "", ("", 0))[1] if cursor else 0

        window = list(rows[start : start + limit])
        payload: dict[str, Any] = {
            "results": window,
            "start": start,
            "limit": limit,
            "size": len(window),
            "_links": {"base": self.base_url, "self": f"{self.base_url}{path}"},
        }
        following = start + limit
        if following < len(rows):
            # The link repeats the whole query, as Confluence's does — form-encoded, so its
            # spaces arrive as '+' too. Only the cursor's '+' is data, and telling the two
            # apart is the entire job of manicule.connectors.pagination.split_query.
            kept = httpx.QueryParams(
                [(key, value) for key, value in request.url.params.items() if key != "cursor"]
            )
            # Every issued cursor carries a '+', because that is the character a naive client
            # turns into a space on the way back — see docs/connectors/confluence.md §2. The
            # value is a function of the offset rather than a counter, so a client that keeps
            # asking for the same page keeps being given the same cursor: a corrupted cursor
            # produces a *loop* here, which is a failure a test can catch, rather than an
            # unbounded walk, which is one it cannot.
            issued = f"{self._cursor_seed}+{following}/page=="
            self._cursors[issued] = (path, following)
            # The cursor itself is written unencoded, exactly as Confluence writes it.
            payload["_links"]["next"] = f"{path}?{kept}&cursor={issued}"
        return httpx.Response(200, json=payload)


def _kinds(query: str) -> set[str]:
    match = _TYPES.search(query)
    if match is None:
        return {"page"}
    if match.group(1):
        return {match.group(1).strip()}
    return {part.strip() for part in (match.group(2) or "").split(",") if part.strip()}


def _unescape(match: re.Match[str] | None) -> str:
    if match is None:
        return ""
    return match.group(1).replace('\\"', '"').replace("\\\\", "\\")


def _after(when: str, since: str | None) -> bool:
    """Whether ``when`` is at or after a CQL ``yyyy/MM/dd HH:mm`` timestamp.

    Compared as text on purpose: the fake reproduces CQL's minute granularity, which is the
    reason the connector overlaps its watermark rather than trusting it exactly.
    """
    if since is None:
        return True
    stamp = when[:16].replace("-", "/").replace("T", " ")
    return stamp >= since
