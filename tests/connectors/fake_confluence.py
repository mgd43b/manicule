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
from dataclasses import dataclass, field, replace
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
_ANCESTOR = re.compile(r"ancestor\s*(?:=\s*(\d+)|in\s*\(([^)]*)\))")
_ID = re.compile(r"(?<![\w.])id\s*(?:=\s*(\d+)|in\s*\(([^)]*)\))")

_QUOTED = re.compile(r'"(?:[^"\\]|\\.)*"')
"""One CQL string literal, escaping included. Blanked before the field check below."""

_STATUS_FIELD = re.compile(r"(?<![\w.])status\s*(?:=|!=|<|>|\bin\b|\bnot\b)", re.IGNORECASE)
"""``status`` used as a field in a comparison, which is what Data Center refuses."""


@dataclass(slots=True)
class FakePage:
    """One page. ``adf`` and ``storage`` are the two bodies the same page can be read as."""

    id: str
    title: str
    space: str
    version: int = 1
    when: str = "2026-08-09T14:30:00.000+01:00"
    """When the current version was made — the page's last modification.

    Served as ``version.when`` by Server and Data Center's v1 content endpoint and as
    ``version.createdAt`` by Cloud's v2 page endpoint. One fact, two spellings, and a fixture
    that offered only one of them could not tell a connector that reads only one from a
    connector that reads both.
    """

    created: str = ""
    """When the page first existed, served as Cloud's top-level ``createdAt``.

    Empty by default and absent from the response when empty, because that is the shape Server
    and Data Center have without a ``history`` expansion — and "the API did not say" is the case
    a provenance record has to get right rather than fill in.
    """

    adf: Mapping[str, object] = field(default_factory=lambda: paragraph("body text"))
    storage: str = "<p>body text</p>"
    ancestors: tuple[str, ...] = ()
    """Ancestor titles, outermost first, for a fixture that only cares about breadcrumbs.

    Used when :attr:`parent` is unset. The ids that go with these are invented, because a test
    of a breadcrumb has no page for an ancestor title to be. Anything about *scope* wants
    :attr:`parent` instead, where the hierarchy is real pages and the ids are theirs.
    """

    parent: str = ""
    """Id of this page's parent, which is what makes the hierarchy a real one.

    Set it and ``ancestors`` is computed by walking up — real ids, real titles, in the order
    Confluence reports them. Scope is decided on ancestor **ids**, so a fixture that named only
    titles could never exercise it and would quietly agree with any implementation.
    """

    status: str = "current"
    """``current``, ``archived`` or ``trashed``. Anything else is absent from search results,
    exactly as Confluence's ``status = current`` makes it, while still being fetchable by id —
    which is the difference a root-page check has to be able to see."""

    kind: str = "page"
    """What the source calls this content. ``blogpost`` is the other one that has an id somebody
    can paste into a configuration file, and it is the one that has no descendants."""

    served_version: int | None = None
    """Version the *body* endpoints report, when it disagrees with what search reports."""

    stale_once: bool = False
    """Whether the disagreement clears on a second request, as a caching artifact would."""

    adf_available: bool = True
    """Whether the Cloud body endpoint returns an Atlassian Document Format body at all.

    It has been seen to decline the format for a page that exists, which is a different thing
    from the page being empty and wants a different answer.
    """

    adf_available_calls: int | None = None
    """How many initial ADF requests return a body before the format becomes unavailable.

    ``None`` follows :attr:`adf_available` on every call. One models a stale first response whose
    retry loses the representation, so fallback is exercised at that transition rather than only
    when ADF is absent from the beginning.
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

        self.forbidden_pages: set[str] = set()
        """Page ids answered with 403 rather than served.

        Losing access to a page is not the same event as the page being deleted, and the
        difference is a subtree's worth of documents. A fixture needs to be able to produce the
        first without producing the second.
        """

        self.rejects_status_field = False
        """Whether a CQL query containing ``status`` is answered with an HTTP 400.

        **What the standard Data Center content-search resource does**, and the reason this
        connector's query builders take the decision as a required argument. Off by default so
        that Cloud fixtures behave as Cloud does; a Server or Data Center fixture turns it on
        and every query the connector sends is then checked, not only the ones a test thought
        to look at.

        This is the guard that makes a *forgotten* call site loud. There are eight places a
        content or title query is built, and a test that asserted on the CQL of the two it
        happened to exercise would pass while reconciliation or an include macro failed against
        a real instance weeks later.
        """

        self.ancestor_predicate = "applied"
        """How this instance treats CQL's ``ancestor`` field: the deployment question that
        cannot be settled from this repository.

        Confluence answers an *unsupported* field with a parse error, which is loud and needs
        no guarding. The two shapes worth modeling are the ones that succeed:

        - ``"applied"`` — the descendant predicate works. What is assumed everywhere else.
        - ``"ignored"`` — accepted and not applied, so the answer is the whole space. Every
          request succeeds and every result is a real page, and a connector that trusted the
          query would index a space while its configuration said one page tree.
        - ``"empty"`` — accepted and matching nothing. This is the dangerous one: the subtree
          reads as deleted, and reconciliation deletes what it does not see.
        """

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

    def move(self, page_id: str, parent: str) -> None:
        """Reparent a page, which is what a user does by dragging it in the page tree.

        The page keeps its id, its version history and its attachments. Everything that decides
        whether it is still in scope changes, and nothing that decides what it *is* does — which
        is the whole reason scope must never become part of a document's identity.
        """
        page = self.pages[page_id]
        self.pages[page_id] = replace(page, parent=parent)

    def chain(self, page: FakePage) -> list[FakePage]:
        """``page``'s ancestors, outermost first, walked up through :attr:`FakePage.parent`.

        Stops on a page it has already seen. Confluence does not permit a cycle in its page
        tree, but a fixture can build one, and a fake that hung on it would be testing the test
        rather than the connector.
        """
        found: list[FakePage] = []
        seen = {page.id}
        current = self.pages.get(page.parent) if page.parent else None
        while current is not None and current.id not in seen:
            seen.add(current.id)
            found.append(current)
            current = self.pages.get(current.parent) if current.parent else None
        found.reverse()
        return found

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
        if path.startswith("/rest/api/space/"):
            return self._one_space(path)
        if path == "/rest/api/user/current":
            return httpx.Response(200, json={"type": "known", "username": self.user})
        if path == "/rest/api/content/search":
            return self._search(request)
        if path.startswith("/api/v2/pages/"):
            return self._v2_page(path)
        if path.endswith("/child/page") and path.startswith("/rest/api/content/"):
            return self._child_pages(path)
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

    def _one_space(self, path: str) -> httpx.Response:
        """``/rest/api/space/{key}`` — the direct lookup a configured allowlist uses.

        Matched **case-insensitively and answered with the instance's own spelling**, because
        that is what Confluence does and it is the reason the connector reads the key back out
        of the response rather than reusing the one it asked with. A fake that echoed the
        request would agree with an implementation that assumed configuration was canonical.
        """
        # Not unquoted again: httpx has already decoded `.url.path`, and a second pass
        # would turn a key containing a literal `%` into a different key — which would
        # make an encoding test pass while the encoding was wrong.
        requested = path.removeprefix("/rest/api/space/")
        for key, name in self.spaces.items():
            if key.casefold() == requested.casefold():
                return httpx.Response(
                    200, json={"key": key, "name": name, "type": "global", "id": abs(hash(key))}
                )
        return httpx.Response(404, json={"message": f"no space with key {requested}"})

    def _search(self, request: httpx.Request) -> httpx.Response:
        query = request.url.params.get("cql", "")
        if self.rejects_status_field and _mentions_status_field(query):
            # The shape Data Center actually fails in: a parse error naming the field, before
            # any content is considered. Not a 404 and not an empty result set — this failure
            # announces itself, which is the one mercy in it.
            return httpx.Response(
                400,
                json={
                    "statusCode": 400,
                    "message": (
                        "Could not parse cql : Unsupported CQL field 'status' for this resource"
                    ),
                },
            )
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
                and page.status == "current"
                and self._within(page, query)
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

    def _within(self, page: FakePage, query: str) -> bool:
        """Whether a page satisfies the query's ``ancestor`` and ``id`` predicates.

        Both are absent from a whole-space query, and then every page qualifies. When they are
        present they are combined with OR, because that is the only way the connector writes
        them: descendants of the roots, or the roots themselves.

        :attr:`ancestor_predicate` is what makes this fake able to be a deployment that does not
        support the field while still answering 200.
        """
        ancestors = _listed(_ANCESTOR.search(query))
        ids = _listed(_ID.search(query))
        if not ancestors and not ids:
            return True
        if self.ancestor_predicate == "ignored":
            return True
        if page.id in ids:
            return True
        if self.ancestor_predicate == "empty":
            return False
        return any(parent.id in ancestors for parent in self.chain(page))

    def _child_pages(self, path: str) -> httpx.Response:
        """``/rest/api/content/{id}/child/page`` — the direct children of one page.

        The second opinion the connector asks for when a descendant query comes back empty. It
        deliberately does **not** consult the ``ancestor`` predicate, so a fake that has been
        told to ignore that predicate still answers this one honestly, which is exactly the
        asymmetry the guard depends on.
        """
        page_id = path.removeprefix("/rest/api/content/").removesuffix("/child/page")
        if page_id in self.forbidden_pages:
            return httpx.Response(403, json={"message": "no access"})
        if page_id not in self.pages:
            return httpx.Response(404, json={"message": "no such page"})
        children = [
            {"id": page.id, "type": "page", "status": page.status, "title": page.title}
            for page in sorted(self.pages.values(), key=lambda page: page.id)
            if page.parent == page_id and page.status == "current"
        ]
        return httpx.Response(200, json={"results": children, "size": len(children)})

    def _result(self, item: FakePage | FakeAttachment, expand: str) -> dict[str, object]:
        if isinstance(item, FakePage):
            row: dict[str, object] = {
                "id": item.id,
                "type": item.kind,
                "status": item.status,
                "title": item.title,
                "version": {"number": item.version, "when": item.when},
                "_links": {"webui": f"/spaces/{item.space}/pages/{item.id}/{item.title}"},
            }
            if "space" in expand:
                row["space"] = {"key": item.space, "name": self.spaces.get(item.space, "")}
            if "ancestors" in expand:
                row["ancestors"] = self._ancestor_rows(item)
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

    def _ancestor_rows(self, page: FakePage) -> list[dict[str, object]]:
        """``expand=ancestors``, outermost first.

        Real pages with their real ids when :attr:`FakePage.parent` builds the hierarchy;
        invented ids beside the given titles otherwise, because a fixture that names only
        titles has no pages for them to be.
        """
        if page.parent:
            return [
                {"id": found.id, "type": "page", "title": found.title} for found in self.chain(page)
            ]
        return [
            {"id": f"anc-{index}", "type": "page", "title": title}
            for index, title in enumerate(page.ancestors)
        ]

    def _v2_page(self, path: str) -> httpx.Response:
        rest = path.removeprefix("/api/v2/pages/")
        page_id, _, tail = rest.partition("/")
        page = self.pages.get(page_id)
        if page is None:
            return httpx.Response(404, json={"message": "no such page"})
        if tail == "ancestors":
            return httpx.Response(200, json={"results": self._ancestor_rows(page)})
        version = self._served_version(page)
        within_available_calls = (
            page.adf_available_calls is None or self.body_calls[page.id] <= page.adf_available_calls
        )
        body: dict[str, object] = (
            {"atlas_doc_format": {"value": json.dumps(page.adf)}}
            if page.adf_available and within_available_calls
            else {}
        )
        # Cloud's v2 page carries two timestamps at two levels: the page's own `createdAt`, and
        # the current version's `createdAt`, which is when the page was last edited. Omitted
        # rather than nulled when the fixture has none, because a key present and null and a key
        # absent reach a client as different facts.
        page_created: dict[str, object] = {"createdAt": page.created} if page.created else {}
        return httpx.Response(
            200,
            json={
                "id": page.id,
                "title": page.title,
                "status": "current",
                "version": {"number": version, "createdAt": page.when},
                **page_created,
                "body": body,
                "_links": {
                    "base": self.base_url,
                    "webui": f"/spaces/{page.space}/pages/{page.id}/{page.title}",
                },
            },
        )

    def _v1_page(self, path: str) -> httpx.Response:
        page_id = path.removeprefix("/rest/api/content/")
        if page_id in self.forbidden_pages:
            return httpx.Response(403, json={"message": "no access"})
        page = self.pages.get(page_id)
        if page is None:
            return httpx.Response(404, json={"message": "no such page"})
        return httpx.Response(
            200,
            json={
                "id": page.id,
                "type": page.kind,
                "status": page.status,
                "title": page.title,
                "space": {"key": page.space, "name": self.spaces.get(page.space, "")},
                "version": {
                    "number": (
                        page.storage_version if page.storage_version is not None else page.version
                    ),
                    "when": page.when,
                },
                "ancestors": self._ancestor_rows(page),
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


def _mentions_status_field(query: str) -> bool:
    """Whether ``query`` uses ``status`` as a **field**, rather than merely containing the word.

    Data Center rejects the field, not the seven letters wherever they fall. A page called
    "Build status" reaches CQL as a quoted literal and is ordinary data — a fake that answered
    400 for it would invent a failure the product does not have, and the next person to see it
    would spend an afternoon on a bug that exists only in this file.

    So the literals are blanked first, and what is left has to look like a field in a comparison:
    the bare word followed by an operator. ``_QUOTED`` handles the escaping :func:`cql.quote`
    produces, which is the same reason the connector escapes at all.
    """
    bare = _QUOTED.sub('""', query)
    return _STATUS_FIELD.search(bare) is not None


def _listed(match: re.Match[str] | None) -> set[str]:
    """The ids in ``field = 1`` or ``field in (1, 2)``, or an empty set for neither."""
    if match is None:
        return set()
    if match.group(1):
        return {match.group(1).strip()}
    return {part.strip() for part in (match.group(2) or "").split(",") if part.strip()}


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
