"""The Confluence connector: discovery, fetch, and the deletion pass sync cannot do.

Implements ``docs/connectors/confluence.md``. Three ideas carry the design.

**A sync costs what changed.** Discovery is a CQL query per space with a ``lastmodified``
watermark, not a walk of every page followed by a client-side version comparison. The
watermark is per space because spaces are enumerated separately and a space whose enumeration
failed must not advance past content it never saw.

**Deletion is a separate obligation.** CQL returns what exists, so a deleted page stops
appearing and a watermark sync never learns it is gone. :meth:`ConfluenceConnector.reconcile`
enumerates ids only, and the pipeline diffs them (``docs/ingest.md`` §11). It is part of the
protocol so that no connector can quietly omit it.

**Structure is fetched, not recovered.** Cloud bodies arrive as Atlassian Document Format — a
typed node tree — and go to :class:`~manicule.parsers.adf.ADFParser`. Server and Data Center
have no ADF, so their bodies arrive as storage format and go to the HTML parser, which is a
real parser rather than a pass that strips angle brackets. Neither of those parsers lives here;
this module's job ends at handing over bytes and saying honestly what they are.

**The index is not permission-aware.** Everything is fetched as the configured account, so the
index holds whatever that account can read and anyone who can search this installation can
retrieve it. Confluence's space and page restrictions do not travel with the content. See
``docs/connectors/confluence.md`` §9; it is repeated here because this is the module that does
it.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import cast

from pydantic import JsonValue

from manicule.connectors import cql
from manicule.connectors.client import ConfluenceClient
from manicule.connectors.config import CONNECTOR_NAME, ConfluenceConfig, Deployment
from manicule.connectors.errors import ConnectorError, NotFoundError
from manicule.connectors.macros import (
    IncludedPage,
    Lookup,
    MacroReport,
    MacroTarget,
    find_adf_macros,
    find_storage_macros,
    resolve_adf,
    resolve_storage,
    unresolved_because,
)
from manicule.core.content import Metadata, RawDocument
from manicule.core.lifecycle import HealthReport, Metric
from manicule.core.sources import DiscoveredDoc, DocRef, SourceId, Watermark

__all__ = [
    "ADF_BODY",
    "STORAGE_BODY",
    "STORAGE_MEDIA_TYPE",
    "VERSION_TOKEN",
    "ConfluenceConnector",
]

ADF_BODY = "atlas_doc_format"
STORAGE_BODY = "storage"

STORAGE_MEDIA_TYPE = "text/html"
"""What a storage-format body is routed as.

Storage format is XHTML with Confluence's own ``ac:`` and ``ri:`` elements mixed in. The HTML
parser reads it with a real HTML engine and keeps the structure that survives; calling it
something else would only mean writing a second parser for a dialect of the same thing.
"""

_SEARCH_PATH = "/rest/api/content/search"
_SPACE_PATH = "/rest/api/space"
_CONTENT_PATH = "/rest/api/content"
_V2_PAGE_PATH = "/api/v2/pages"

_SEARCH_EXPAND = "version,ancestors,space,container"
_STORAGE_EXPAND = "body.storage,version,ancestors,space"

PAGE = "page"
ATTACHMENT = "attachment"

_RESOLUTION_OFF = (
    "macro expansion is turned off for this connector (resolve_macros), so the content this "
    "macro renders is not in the index although a reader sees it on the page"
)

KIND = "confluence_kind"
SPACE_KEY = "space_key"
VERSION = "confluence_version"
VERSION_TOKEN = "version_token"  # noqa: S105 - a metadata key name, not a credential
"""Metadata key carrying the version the body **actually** came back with.

Distinct from ``VERSION``, which is what discovery reported. They differ exactly when the
source served a body older than the one it had just described, and the pipeline stores this
one so that the next sync notices."""
DOWNLOAD = "confluence_download"
ANCESTORS = "ancestors"
"""Metadata key the chunker reads to build the breadcrumb prefixed to every chunk.

Owned by ``manicule.chunking.chunker``; named here because this connector is the thing that
has to fill it. The page's own title is *not* part of it — the chunker appends that itself, and
a breadcrumb carrying it twice reaches the embedder as emphasis nobody intended.
"""


@dataclass(frozen=True, slots=True)
class _Body:
    """One page's content as the source returned it."""

    page_id: str
    title: str
    version: int
    body: str
    body_format: str
    space_key: str = ""
    ancestors: tuple[str, ...] = ()
    webui: str = ""
    base: str = ""


class ConfluenceConnector:
    """Discovers, fetches and reconciles Confluence pages and their attachments."""

    def __init__(self, config: ConfluenceConfig, client: ConfluenceClient) -> None:
        self.name = CONNECTOR_NAME
        self._config = config
        self._client = client
        self._observed: dict[str, datetime] = {}
        self._carried: dict[str, str] = {}
        self._enumerated = False

    # --- lifecycle -----------------------------------------------------------------------

    async def setup(self) -> None:
        await self._client.setup()

    async def teardown(self) -> None:
        await self._client.teardown()

    async def health(self) -> HealthReport:
        """Ask the source one cheap question, so a bad credential is a startup fact.

        A connector that reports healthy because nothing has asked it to do anything yet is
        reporting on itself rather than on the source.
        """
        try:
            await self._client.get_json(self._client.url(_SPACE_PATH), [("limit", "1")])
        except ConnectorError as exc:
            return HealthReport.failing(
                f"{self._config.base_url} did not answer: {exc}",
                remedy="Check the base URL and the credential in "
                'plugins.config."connector.confluence".',
            )
        return HealthReport.healthy(f"{self._config.base_url} answered")

    def metrics(self) -> tuple[Metric, ...]:
        return (
            Metric(name="confluence_requests", value=float(self._client.requests)),
            Metric(name="confluence_throttled", value=float(self._client.throttled)),
        )

    # --- watermarks ----------------------------------------------------------------------

    @property
    def watermark(self) -> Watermark | None:
        """What to persist **if this run completed cleanly**, or ``None`` if there is nothing.

        A per-space map rather than one timestamp: spaces are enumerated one at a time, and a
        run that failed in the third space must not advance the first two past content the
        pipeline never got to store. The pipeline decides whether a run was clean
        (``docs/ingest.md`` §13.2) and this is what it stores when it was.

        The :class:`~manicule.core.sources.Watermark` contract is that ``value`` is opaque and
        manicule never interprets it, so the map lives in ``metadata`` and ``value`` carries
        the newest instant in it — which makes a stored watermark legible to a person reading
        the row without making it meaningful to any code.

        ``None`` until :meth:`discover` has run to completion. A consumer that stopped early —
        cancellation, an error, a bounded queue that was never drained — enumerated a prefix,
        and a watermark built from a prefix skips whatever the rest of the walk would have
        returned. Nothing looks for it again, so the guard is here as well as in the pipeline.
        """
        if not self._enumerated:
            return None
        spaces = dict(self._carried)
        for space, observed in self._observed.items():
            spaces[space] = observed.isoformat()
        if not spaces:
            return None
        newest = max(spaces.values())
        return Watermark(
            value=newest,
            observed_at=datetime.now(tz=UTC),
            metadata={"spaces": cast("Metadata", dict(spaces))},
        )

    def _since(self, watermark: Watermark | None, space: str) -> str | None:
        """The CQL timestamp this space's query starts from, or ``None`` for everything."""
        stored = _space_watermarks(watermark).get(space)
        when = cql.parse_when(stored)
        if when is None:
            return None
        return cql.cql_timestamp(when, timedelta(minutes=self._config.watermark_overlap_minutes))

    # --- discovery -----------------------------------------------------------------------

    async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
        """Yield everything created or changed in scope since ``watermark``.

        ``None`` means no previous sync: everything. Otherwise each space is queried from its
        own stored position, minus an overlap, because CQL compares ``lastmodified`` at minute
        granularity and the alternative to overlapping is missing whatever shared the last
        recorded minute.
        """
        self._observed = {}
        self._carried = dict(_space_watermarks(watermark))
        self._enumerated = False

        types = (PAGE, ATTACHMENT) if self._config.include_attachments else (PAGE,)
        for space in await self._spaces():
            query = cql.content_query(space, types=types, since=self._since(watermark, space))
            params = [
                ("cql", query),
                ("limit", str(self._config.page_size)),
                ("expand", _SEARCH_EXPAND),
            ]
            newest: datetime | None = None
            async for payload in self._client.paginate(self._client.url(_SEARCH_PATH), params):
                base = _link_base(payload, self._config.base_url)
                for result in _results(payload):
                    found = self._discovered(result, space=space, base=base)
                    if found is None:
                        continue
                    newest = cql.latest([newest, cql.parse_when(_version_when(result))])
                    yield found
            # Only after the space's enumeration has run to completion: a watermark advanced
            # from a partial walk skips whatever the rest of the walk would have returned, and
            # nothing ever looks for it again.
            if newest is not None:
                self._observed[space] = newest
        self._enumerated = True

    def _discovered(
        self, result: Mapping[str, object], *, space: str, base: str
    ) -> DiscoveredDoc | None:
        kind = _str(result.get("type"))
        source_id = _str(result.get("id"))
        if not source_id or kind not in {PAGE, ATTACHMENT}:
            return None
        links = _obj(result.get("_links"))
        uri = _join(base, _str(links.get("webui")))
        version = _int(_obj(result.get("version")).get("number"))
        title = _str(result.get("title"))
        space_key = _str(_obj(result.get("space")).get("key")) or space

        metadata: Metadata = {
            KIND: kind,
            SPACE_KEY: space_key,
            "title": title,
        }
        if version is not None:
            metadata[VERSION] = version

        if kind == PAGE:
            page_crumbs: list[JsonValue] = [space_key, *_ancestor_titles(result)]
            metadata[ANCESTORS] = page_crumbs
            media_type = self._page_media_type()
            size = None
        else:
            container = _obj(result.get("container"))
            crumbs: list[JsonValue] = [space_key, *_str_list(_str(container.get("title")))]
            metadata[ANCESTORS] = crumbs
            metadata["parent_page_id"] = _str(container.get("id"))
            metadata["parent_page_uri"] = _join(
                base, _str(_obj(container.get("_links")).get("webui"))
            )
            metadata[DOWNLOAD] = _join(base, _str(links.get("download")))
            media_type = _attachment_media_type(result)
            size = _int(_obj(result.get("extensions")).get("fileSize"))
            if media_type is not None:
                metadata["media_type"] = media_type

        return DiscoveredDoc(
            ref=DocRef(source_id=source_id, uri=uri or self._config.base_url, metadata=metadata),
            version_token=str(version) if version is not None else None,
            title=title,
            media_type=media_type,
            size_bytes=size if size is not None and size >= 0 else None,
        )

    def _page_media_type(self) -> str:
        """What a page body will be, before anything has been fetched.

        Imported from the parsers' registration module rather than written out again: routing
        reads that declaration, and a second spelling of the same string would route Confluence
        bodies to nothing the day one of them was edited.
        """
        from manicule.parsers.config import ADF_MEDIA_TYPE  # noqa: PLC0415 - see docstring

        return ADF_MEDIA_TYPE if self._is_cloud else STORAGE_MEDIA_TYPE

    @property
    def _is_cloud(self) -> bool:
        return self._config.deployment is Deployment.CLOUD

    async def _spaces(self) -> list[str]:
        """The spaces to sync: the configured allowlist, checked, or everything visible.

        Enumerated per run rather than cached, so a space created since the last sync is picked
        up without a configuration change — and so that a space the account has *lost* access
        to is reported instead of silently contributing nothing.

        **A configured key that no visible space has is a refusal.** CQL answers a query for a
        space that does not exist with an empty result set, exactly as it answers a query for a
        space with nothing in it, so a typo in an allowlist is a sync that runs, succeeds, and
        indexes nothing — and then reconciliation proposes deleting everything that space ever
        contributed. One extra enumeration per run buys the difference between those two.

        Raises:
            ConnectorError: A configured space is not visible to this account, or the account
                can see no spaces at all.
        """
        visible = await self._visible_spaces()
        if not self._config.spaces:
            if not visible:
                msg = (
                    f"this account can see no spaces at {self._config.base_url}. A sync would "
                    f"index nothing and reconciliation would then propose deleting everything "
                    f"already indexed, so it stops here. Check the credential, and that the "
                    f"account has been granted at least one space."
                )
                raise ConnectorError(msg)
            return list(visible.values())

        chosen: list[str] = []
        missing: list[str] = []
        for key in self._config.spaces:
            found = visible.get(key.strip().casefold())
            if found is None:
                missing.append(key)
            else:
                chosen.append(found)
        if missing:
            available = ", ".join(sorted(visible.values())) or "none"
            msg = (
                f"configured space(s) {', '.join(sorted(missing))} are not visible to this "
                f"account. Visible: {available}. A query for a space that is not there returns "
                f"nothing rather than an error, so this would be a sync that appears to work."
            )
            raise ConnectorError(msg)
        return chosen

    async def _visible_spaces(self) -> dict[str, str]:
        """Every space key this account can see, folded for comparison to its own spelling."""
        keys: dict[str, str] = {}
        params = [("limit", str(self._config.page_size))]
        async for payload in self._client.paginate(self._client.url(_SPACE_PATH), params):
            for space in _results(payload):
                key = _str(space.get("key"))
                if key:
                    keys[key.casefold()] = key
        return keys

    # --- reconciliation ------------------------------------------------------------------

    async def reconcile(self) -> AsyncIterator[SourceId]:
        """Yield the id of everything that still exists, ids only.

        No expansions, no bodies, no versions — which is what makes a full enumeration
        affordable often enough to matter. Any failure propagates rather than being caught and
        smoothed over: the pipeline refuses to diff a partial enumeration, because the ids seen
        so far are a prefix and diffing a prefix soft-deletes everything past it
        (``docs/ingest.md`` §11.1).
        """
        types = (PAGE, ATTACHMENT) if self._config.include_attachments else (PAGE,)
        for space in await self._spaces():
            params = [
                ("cql", cql.content_query(space, types=types, ordered=False)),
                ("limit", str(self._config.page_size)),
            ]
            async for payload in self._client.paginate(self._client.url(_SEARCH_PATH), params):
                for result in _results(payload):
                    source_id = _str(result.get("id"))
                    if source_id:
                        yield source_id

    # --- fetch ---------------------------------------------------------------------------

    async def fetch(self, ref: DocRef) -> RawDocument:
        """Retrieve one page or attachment.

        Raises:
            NotFoundError: It has been deleted since discovery saw it. Ordinary during a long
                sync, and its own failure so the pipeline can tell it from a fault.
        """
        if _str(ref.metadata.get(KIND)) == ATTACHMENT:
            return await self._fetch_attachment(ref)
        return await self._fetch_page(ref)

    async def _fetch_attachment(self, ref: DocRef) -> RawDocument:
        """Download an attachment, to be routed through the ordinary parser chain.

        A PDF attached to a page is a PDF: it gets its own document, its own parser and its own
        anchors. What it keeps is its parent, in metadata and in the breadcrumb, so a citation
        resolves to both the file and the page it hangs off.
        """
        url = _str(ref.metadata.get(DOWNLOAD)) or ref.uri
        downloaded = await self._client.download(url, max_bytes=self._config.max_attachment_bytes)
        declared = _str(ref.metadata.get("media_type"))
        metadata = dict(ref.metadata)
        version = _int(ref.metadata.get(VERSION))
        if version is not None:
            metadata[VERSION_TOKEN] = str(version)
        return RawDocument(
            source_id=ref.source_id,
            uri=ref.uri,
            media_type=declared or downloaded.media_type or _from_name(_str(metadata.get("title"))),
            content=downloaded.content,
            metadata=metadata,
        )

    async def _fetch_page(self, ref: DocRef) -> RawDocument:
        expected = _int(ref.metadata.get(VERSION))
        body = await self._page_body(ref.source_id, expected)
        space_key = body.space_key or _str(ref.metadata.get(SPACE_KEY))
        report = MacroReport()
        content = body.body

        if self._config.resolve_macros:
            lookup = self._lookup_for(body, space_key)
            if body.body_format == ADF_BODY:
                document = _adf_document(content, body.page_id)
                content = json.dumps(
                    await resolve_adf(
                        document,
                        lookup=lookup,
                        depth_limit=self._config.macro_depth,
                        report=report,
                        path=(body.page_id,),
                    )
                )
            else:
                content = await resolve_storage(
                    content,
                    lookup=lookup,
                    depth_limit=self._config.macro_depth,
                    report=report,
                    path=(body.page_id,),
                )
        else:
            report.unresolved.extend(unresolved_because(self._macros_in(body), _RESOLUTION_OFF))

        ancestors: list[JsonValue] = list(body.ancestors) or list(_metadata_ancestors(ref))
        complete = True
        if not ancestors:
            titles, complete = await self._ancestor_titles_of(ref.source_id)
            ancestors = list(_str_list(space_key, *titles))

        metadata: Metadata = {
            KIND: PAGE,
            "title": body.title,
            ANCESTORS: ancestors,
            "breadcrumb_complete": complete,
            SPACE_KEY: space_key,
            VERSION: body.version,
            VERSION_TOKEN: str(body.version),
            "body_format": body.body_format,
            "deployment": self._config.deployment.value,
        }
        metadata.update(report.as_metadata())
        if expected is not None and body.version != expected:
            metadata["version_disagreement"] = {"discovered": expected, "fetched": body.version}

        return RawDocument(
            source_id=ref.source_id,
            uri=_join(body.base, body.webui) or ref.uri,
            media_type=(
                self._page_media_type() if body.body_format == ADF_BODY else STORAGE_MEDIA_TYPE
            ),
            content=content,
            metadata=metadata,
        )

    async def _page_body(self, page_id: str, expected: int | None) -> _Body:
        """A page's body, checked against the version discovery reported.

        Atlassian Document Format has been observed returning stale content
        (``docs/connectors/confluence.md`` §4), and stale content is not a visible failure: the
        page indexes cleanly, reads plausibly, and is wrong. So the version that comes back is
        compared with the one discovery saw, and a disagreement is retried once and then
        answered from storage format, which is a different code path on the source's side.

        If every attempt still disagrees, the body is used **with the version it actually
        carries** rather than the one that was asked for. That is what makes it self-healing:
        the stored token is now behind what the next discovery reports, so the next sync fetches
        the page again instead of concluding it is unchanged. Recording the requested version
        against older bytes would make the document permanently, invisibly stale.
        """
        if not self._is_cloud:
            return await self._storage_body(page_id)

        body = await self._adf_body(page_id)
        if expected is None or body.version >= expected:
            return body
        retried = await self._adf_body(page_id)
        if retried.version >= expected:
            return retried
        fallback = await self._storage_body(page_id)
        return fallback if fallback.version >= expected else retried

    def _macros_in(self, body: _Body) -> list[MacroTarget]:
        """The include macros in a body nobody is going to expand, so they can be recorded."""
        if body.body_format == ADF_BODY:
            return find_adf_macros(_adf_document(body.body, body.page_id))
        return find_storage_macros(body.body)

    async def _ancestor_titles_of(self, page_id: str) -> tuple[tuple[str, ...], bool]:
        """Ancestor titles for a page whose discovery record carried none, and whether they are
        all there.

        Discovery expands ``ancestors`` and puts the titles on the ref, so this runs only for a
        ref built somewhere else — a re-fetch from a stored one, a targeted single-page sync.
        The Atlassian Document Format endpoint does not carry ancestors, hence a second call
        rather than a wider expansion.

        The flag is returned rather than swallowed because a breadcrumb missing a level is not
        visibly wrong: it retrieves slightly worse and says nothing. An ancestor whose title
        the endpoint omitted is skipped rather than filled with its id, which would put a
        number into the text the embedder reads.
        """
        if not self._is_cloud:
            return (), True
        url = f"{self._client.url(_V2_PAGE_PATH)}/{page_id}/ancestors"
        try:
            payload = await self._client.get_json(url, [("limit", "25")])
        except NotFoundError:
            return (), True
        entries = _results(payload)
        titles = tuple(_str(entry.get("title")) for entry in entries)
        return tuple(title for title in titles if title), all(titles)

    async def _adf_body(self, page_id: str) -> _Body:
        url = f"{self._client.url(_V2_PAGE_PATH)}/{page_id}"
        payload = await self._client.get_json(url, [("body-format", ADF_BODY)])
        body = _obj(_obj(payload.get("body")).get(ADF_BODY))
        value = _str(body.get("value"))
        if not value:
            msg = (
                f"page {page_id} came back with no Atlassian Document Format body. The page "
                f"exists, so this is the source declining the format rather than the page "
                f"being empty; storage format is the fallback."
            )
            raise ConnectorError(msg)
        links = _obj(payload.get("_links"))
        return _Body(
            page_id=page_id,
            title=_str(payload.get("title")),
            version=_int(_obj(payload.get("version")).get("number")) or 0,
            body=value,
            body_format=ADF_BODY,
            webui=_str(links.get("webui")),
            base=_str(links.get("base")) or self._config.base_url,
        )

    async def _storage_body(self, page_id: str) -> _Body:
        url = f"{self._client.url(_CONTENT_PATH)}/{page_id}"
        payload = await self._client.get_json(url, [("expand", _STORAGE_EXPAND)])
        body = _obj(_obj(payload.get("body")).get(STORAGE_BODY))
        links = _obj(payload.get("_links"))
        space_key = _str(_obj(payload.get("space")).get("key"))
        return _Body(
            page_id=page_id,
            title=_str(payload.get("title")),
            version=_int(_obj(payload.get("version")).get("number")) or 0,
            body=_str(body.get("value")),
            body_format=STORAGE_BODY,
            space_key=space_key,
            ancestors=(space_key, *_ancestor_titles(payload)) if space_key else (),
            webui=_str(links.get("webui")),
            base=_str(links.get("base")) or self._config.base_url,
        )

    # --- macro resolution ----------------------------------------------------------------

    def _lookup_for(self, body: _Body, space_key: str) -> Lookup:
        """How an include macro finds the page it names, memoised for this fetch.

        Two macros naming the same page — an overview that includes a definition twice — would
        otherwise be two searches and two body fetches for one answer.

        A macro that names no space means *this* one, which is why the space key is threaded in
        rather than read off the body: the Atlassian Document Format endpoint reports a numeric
        space id and not the key CQL compares against.
        """
        cache: dict[tuple[str, str, str], IncludedPage | None] = {}

        async def lookup(target: MacroTarget) -> IncludedPage | None:
            space = target.space or space_key
            key = (target.content_id, target.title, space)
            if key not in cache:
                cache[key] = await self._included(target, space, body.body_format)
            return cache[key]

        return lookup

    async def _included(
        self, target: MacroTarget, space: str, body_format: str
    ) -> IncludedPage | None:
        page_id = target.content_id or await self._page_id_of(target.title, space)
        if not page_id:
            return None
        try:
            found = (
                await self._adf_body(page_id)
                if body_format == ADF_BODY
                else await self._storage_body(page_id)
            )
        except NotFoundError:
            return None
        if body_format == ADF_BODY:
            return IncludedPage(
                page_id=page_id, title=found.title, adf=_adf_document(found.body, page_id)
            )
        return IncludedPage(page_id=page_id, title=found.title, storage=found.body)

    async def _page_id_of(self, title: str, space: str) -> str:
        """The id of the page an include macro names by title, or ``""`` if there is none."""
        if not title or not space:
            return ""
        params = [("cql", cql.title_query(space, title)), ("limit", "1")]
        payload = await self._client.get_json(self._client.url(_SEARCH_PATH), params)
        results = _results(payload)
        return _str(results[0].get("id")) if results else ""


# --- decoding helpers ----------------------------------------------------------------------


def _adf_document(value: str, page_id: str) -> Mapping[str, object]:
    """The ADF body, which arrives as a JSON string inside a JSON response.

    Raises:
        ConnectorError: It is not a JSON object. Declining here rather than passing it on
            means the failure names the page, instead of arriving three stages later as a
            parser complaining about a document nobody can identify.
    """
    try:
        loaded: object = json.loads(value)
    except ValueError as exc:
        msg = f"page {page_id} returned an Atlassian Document Format body that is not JSON: {exc}"
        raise ConnectorError(msg) from exc
    if not isinstance(loaded, dict):
        msg = f"page {page_id} returned a JSON {type(loaded).__name__} where a document was due"
        raise ConnectorError(msg)
    return cast("Mapping[str, object]", loaded)


def _space_watermarks(watermark: Watermark | None) -> Mapping[str, str]:
    """The per-space map inside a stored watermark, tolerating one that predates it."""
    if watermark is None:
        return {}
    spaces = watermark.metadata.get("spaces")
    if not isinstance(spaces, dict):
        return {}
    entries = cast("Mapping[str, object]", spaces)
    return {key: value for key, value in entries.items() if isinstance(value, str)}


def _results(payload: Mapping[str, object]) -> list[Mapping[str, object]]:
    results = payload.get("results")
    if not isinstance(results, list):
        return []
    entries = cast("Sequence[object]", results)
    return [cast("Mapping[str, object]", e) for e in entries if isinstance(e, dict)]


def _ancestor_titles(result: Mapping[str, object]) -> tuple[str, ...]:
    """Ancestor titles, outermost first. The page's own title is not among them."""
    ancestors = result.get("ancestors")
    if not isinstance(ancestors, list):
        return ()
    entries = cast("Sequence[object]", ancestors)
    titles = [_str(_obj(entry).get("title")) for entry in entries]
    return tuple(title for title in titles if title)


def _metadata_ancestors(ref: DocRef) -> tuple[str, ...]:
    value = ref.metadata.get(ANCESTORS)
    if not isinstance(value, list):
        return ()
    entries = cast("Sequence[object]", value)
    return tuple(entry for entry in entries if isinstance(entry, str))


def _attachment_media_type(result: Mapping[str, object]) -> str | None:
    """What the source says an attachment is, or ``None`` when it says nothing.

    ``None`` rather than a guess: the download's own ``Content-Type`` is consulted next, and a
    filename extension after that. Inventing one here would put the guess first.
    """
    declared = _str(_obj(result.get("metadata")).get("mediaType")) or _str(
        _obj(result.get("extensions")).get("mediaType")
    )
    return declared or None


def _from_name(name: str) -> str:
    """The media type a filename implies, shared with the container expansion rules."""
    from manicule.parsers.expansion import media_type_for  # noqa: PLC0415 - light, see below

    return media_type_for(name)


def _link_base(payload: Mapping[str, object], fallback: str) -> str:
    declared = _str(_obj(payload.get("_links")).get("base"))
    return declared or fallback


def _join(base: str, path: str) -> str:
    if not path:
        return ""
    if path.startswith(("http://", "https://")):
        return path
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


def _obj(value: object) -> Mapping[str, object]:
    return cast("Mapping[str, object]", value) if isinstance(value, dict) else {}


def _str(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _str_list(*values: str) -> list[str]:
    return [value for value in values if value]


def _int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _version_when(result: Mapping[str, object]) -> object:
    return _obj(result.get("version")).get("when")
