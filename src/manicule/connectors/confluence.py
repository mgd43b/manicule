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
import logging
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import cast
from urllib.parse import quote

from pydantic import JsonValue

from manicule.connectors import cql, subtree
from manicule.connectors.client import ConfluenceClient
from manicule.connectors.config import CONNECTOR_NAME, ConfluenceConfig, Deployment
from manicule.connectors.errors import BodyUnavailableError, ConnectorError, NotFoundError
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
from manicule.connectors.subtree import CONTENT_PATH, SEARCH_PATH, Subtree
from manicule.core.content import Metadata, RawDocument
from manicule.core.lifecycle import HealthReport, Metric
from manicule.core.provenance import PROVENANCE_KEY, Provenance, SourceMetadata
from manicule.core.sources import DiscoveredDoc, DocRef, SourceId, Watermark
from manicule.parsers.config import CONFLUENCE_MEDIA_TYPE

__all__ = [
    "ADF_BODY",
    "ANCESTOR_IDS",
    "MODIFIED_AT",
    "ROOT_PAGE_IDS",
    "SCOPE",
    "STORAGE_BODY",
    "STORAGE_MEDIA_TYPE",
    "VERSION_TOKEN",
    "ConfluenceConnector",
]

_log = logging.getLogger("manicule.connectors.confluence")

ADF_BODY = "atlas_doc_format"
STORAGE_BODY = "storage"

STORAGE_MEDIA_TYPE = CONFLUENCE_MEDIA_TYPE
"""What a storage-format body is routed as: its own profiled type, read by its own parser.

**This constant was ``text/html`` for as long as this connector existed, and its docstring twice
argued that it should be.** Both arguments are kept below, because each was wrong in a way worth
remembering rather than deleting.

The first said the HTML parser "keeps the structure that survives" and that naming storage format
separately "would only mean writing a second parser for a dialect of the same thing". It is not
the same thing, and the difference destroyed content: storage format wraps the body of every
``code``, ``noformat`` and ``graphviz`` macro in ``<![CDATA[…]]>``, which HTML has no equivalent
of outside foreign content — so a conforming parser reparsed each as a *bogus comment* and deleted
the body. Every code block on every Server or Data Center page was missing from the index, with a
fragment of it indexed as prose, and nothing raised.

The second survived that fix. It conceded the vocabulary was unread but called it "a missing
feature rather than lost content" — and that was wrong too, in the direction nobody checked.
Reading ``ac:parameter`` as generic HTML did not merely lose a code block's language; it *indexed*
it. The language, the diagram engine and a Jira macro's **JQL query** each became a prose block,
went into the vector, and were quotable in a citation as words the page had said. A task's status
arrived as a one-word block reading ``complete``. That is not a missing feature; it is the index
asserting things the document does not contain.

:mod:`manicule.parsers.confluence` reads the vocabulary now, so this declares what the bytes
actually are. Changing it **re-routes every page already ingested under the old type** — which
nothing would otherwise notice, because the bytes are identical and the stored lineage names the
parser that read them last. ``Change.ROUTING`` is what notices, and it exists for this.

The lesson is about the sentences rather than the bugs. Both were written from intent, never
executed, and both read as settled arguments against doing the work. What a parser keeps, and
what it puts in the index, are each checkable in four lines.
"""

_SPACE_PATH = "/rest/api/space"
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

ANCESTOR_IDS = "ancestor_ids"
"""Metadata key carrying the page's ancestors as content ids, outermost first.

Beside :data:`ANCESTORS` rather than instead of it, because they answer different questions and
only one of them survives a rename. Titles are what a reader sees in a breadcrumb; ids are what
says *which* page an ancestor is, which is what a scope check and anybody auditing why a page
was indexed both need."""

ROOT_PAGE_IDS = "root_page_ids"
"""Metadata key naming the configured root page(s) that put this document in scope.

**Derived synchronization metadata, and nothing more.** It records why manicule holds this
document; it is not part of the document's identity, which stays the Confluence page id, and it
is not part of its address, which stays the canonical URL. Absent entirely when no roots are
configured, so its presence means "this source is subtree-scoped" rather than "somebody wrote a
default down"."""

MODIFIED_AT = "source_modified_at"
"""Ref metadata carrying an attachment's modification time, verbatim, from discovery.

**Attachments only, and the asymmetry is deliberate.** A page's body response states when the
retained version was made, so the fetch reads it there and nothing needs carrying. An
attachment's fetch is a byte download that describes nothing, so the search result that found it
is the only source response that ever says — and it is still a source response.

Not set for pages, because a page would then carry two timestamps from two responses that
disagree exactly when the stale-body fallback fires. A second value nobody reads is a value
somebody eventually reads by mistake."""

SCOPE = "scope"
"""Watermark metadata key holding the scope a stored position was recorded within.

A position and the scope it was reached in are one fact. See
:attr:`~manicule.connectors.config.ConfluenceConfig.scope_identity` for why reusing one against
the other is the failure worth a full re-enumeration to avoid."""

WHOLE_SPACE = "whole-space"
"""What a watermark stored before subtree scoping existed is read as.

Not ``""`` and not ``None``: those would make "no scope recorded" a third state that every
comparison would have to handle, when in fact there is only one scope such a watermark can have
had. Written out so that the backward-compatible reading is a value rather than an omission."""


@dataclass(frozen=True, slots=True)
class _Body:
    """One page's content as the source returned it.

    The timestamps belong here rather than being read again somewhere else, and that placement
    is the whole of the stale-body defense as it applies to provenance. ``_page_body`` may
    answer from either of two endpoints, and the one it returns is the one whose bytes are
    kept — so a modification time carried on this object is necessarily the modification time
    of the body that was retained, and cannot come from a response that was discarded.
    """

    page_id: str
    title: str
    version: int
    body: str
    body_format: str
    space_key: str = ""
    ancestors: tuple[str, ...] = ()
    ancestor_ids: tuple[str, ...] = ()
    webui: str = ""
    base: str = ""

    modified_at: datetime | None = None
    """When the source says this version was made. ``None`` when the response did not say, or
    said something without a UTC offset — never a substitute drawn from anywhere else."""

    created_at: datetime | None = None
    """When the source says the page was created, on the deployments whose response carries it.
    Absent is the ordinary answer, and absent is what is recorded."""


class ConfluenceConnector:
    """Discovers, fetches and reconciles Confluence pages and their attachments."""

    def __init__(
        self, config: ConfluenceConfig, client: ConfluenceClient, *, name: str = CONNECTOR_NAME
    ) -> None:
        # The configured source's name, defaulting to the type for a caller outside the
        # container. Two Confluence sites, or two space subsets of one site, are two sources;
        # naming both after the implementation files their documents under one `source`, where
        # a shared page id makes one silently overwrite the other.
        self.name = name
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
        return Watermark(
            value=_newest(spaces.values()),
            observed_at=datetime.now(tz=UTC),
            metadata={
                "spaces": cast("Metadata", dict(spaces)),
                SCOPE: self._config.scope_identity,
            },
        )

    def _resume_from(self, watermark: Watermark | None) -> Mapping[str, str]:
        """The stored per-space positions, or nothing when they belong to a different scope.

        A watermark says "everything up to here has been seen" — but only of the scope it was
        recorded in. Point a source at a different page tree and the same sentence becomes
        false: every page in the new tree that has not changed since is already behind the
        stored position, so an incremental query would never return it, and nothing would ever
        return it again.

        So a scope change discards every position and re-enumerates. That costs one full pass
        and is right in both directions — a root added brings unchanged pages into scope, and a
        root removed leaves indexed documents outside it — where partitioning the old positions
        by root would save that one pass and be wrong the first time a page moved between two
        of them.

        A watermark stored before this connector had scopes at all carries no scope key, and is
        read as :data:`WHOLE_SPACE`: that is the only scope it can have been recorded in, so an
        installation that has not configured roots resumes exactly as before.
        """
        if watermark is None:
            return {}
        stored = watermark.metadata.get(SCOPE)
        recorded = stored if isinstance(stored, str) and stored else WHOLE_SPACE
        current = self._config.scope_identity
        if recorded == current:
            return _space_watermarks(watermark)
        _log.warning(
            "source %r: the configured Confluence scope changed from [%s] to [%s]. Every "
            "stored per-space position has been discarded and this run enumerates the new "
            "scope in full, because a position recorded in one scope says nothing about "
            "another. Documents already indexed from outside the new scope are not touched by "
            "this run; the next reconciliation pass is what proposes removing them, and the "
            "deletion ceiling still applies to that proposal.",
            self.name,
            recorded,
            current,
        )
        return {}

    def _since(self, carried: Mapping[str, str], space: str) -> str | None:
        """The CQL timestamp this space's query starts from, or ``None`` for everything."""
        when = cql.parse_when(carried.get(space))
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

        "In scope" is the whole space when no root pages are configured, and one or more page
        trees inside it when they are (``docs/connectors/confluence.md`` §2.1). The roots are
        validated **before the first query goes out**, so a root that is missing or outside the
        space allowlist stops the run rather than producing a smaller subtree than the one that
        was configured.
        """
        carried = self._resume_from(watermark)
        self._observed = {}
        self._carried = dict(carried)
        self._enumerated = False

        spaces = await self._spaces()
        scope = await self._scope(spaces)

        for space in spaces if scope is None else scope.spaces():
            newest: datetime | None = None
            async for found, when in self._changed_in(space, scope, self._since(carried, space)):
                newest = cql.latest([newest, when])
                yield found
            # Only after the space's enumeration has run to completion: a watermark advanced
            # from a partial walk skips whatever the rest of the walk would have returned, and
            # nothing ever looks for it again.
            if newest is not None:
                self._observed[space] = newest
        self._enumerated = True

    async def _changed_in(
        self, space: str, scope: Subtree | None, since: str | None
    ) -> AsyncIterator[tuple[DiscoveredDoc, datetime | None]]:
        """One space's changed documents, each with the instant its version was saved.

        Whole-space mode is the single query §2 describes, covering pages and attachments
        together. Subtree mode is two queries, and the reason they are two is the difference
        between what Confluence will narrow and what it will not:

        - **Pages are narrowed at the source**, by ``ancestor``. A page that comes back anyway
          is a refusal, because the alternative is a sync that pays for a whole space and
          reports a subtree.
        - **Attachments are not.** There is no descendant predicate for an attachment worth
          relying on, so they are enumerated space-wide and matched against the page holding
          them — which is the authoritative answer in any case, and the one §6 of the
          documentation says to use.
        """
        if scope is None:
            types = (PAGE, ATTACHMENT) if self._config.include_attachments else (PAGE,)
            query = self._content_query(space, types=types, since=since)
            async for result, base in self._search(query, expand=_SEARCH_EXPAND):
                found = self._discovered(result, space=space, base=base)
                if found is not None:
                    yield found, cql.parse_when(_version_when(result))
            return

        pages = self._content_query(space, types=(PAGE,), since=since, subtree=scope.clause(space))
        async for result, base in self._search(pages, expand=_SEARCH_EXPAND):
            page_id = _str(result.get("id"))
            roots = scope.covering_roots(space, page_id, subtree.ancestor_ids(result))
            if not roots:
                raise ConnectorError(scope.out_of_scope(space, page_id))
            found = self._discovered(result, space=space, base=base, roots=roots)
            if found is not None:
                yield found, cql.parse_when(_version_when(result))

        if not self._config.include_attachments:
            return

        # Every in-scope page id, not only the ones this incremental query returned: an
        # attachment can be added to a page that has not itself changed since the watermark,
        # and its container would then be a page this run has never seen.
        members = await scope.members(space)
        attachments = self._content_query(space, types=(ATTACHMENT,), since=since)
        async for result, base in self._search(attachments, expand=_SEARCH_EXPAND):
            container = _str(_obj(result.get("container")).get("id"))
            roots = members.get(container, ())
            if not roots:
                continue
            found = self._discovered(result, space=space, base=base, roots=roots)
            if found is not None:
                yield found, cql.parse_when(_version_when(result))

    def _content_query(
        self,
        space: str,
        *,
        types: Sequence[str] = (PAGE,),
        since: str | None = None,
        ordered: bool = True,
        subtree: str = "",
    ) -> str:
        """Every content query this connector sends, so the deployment is read once.

        :func:`~manicule.connectors.cql.content_query` requires ``current_only`` and has no
        default, which makes forgetting it a type error. This makes *remembering* it a single
        line instead of five: every call site in this class routes through here, so the answer
        to "does this deployment accept ``status``" is looked up in one place and cannot be
        right in four queries and wrong in the fifth.

        :class:`~manicule.connectors.subtree.Subtree` builds the one query that is not here, and
        reads the same :attr:`~manicule.connectors.config.ConfluenceConfig.current_only`
        property to do it.
        """
        return cql.content_query(
            space,
            current_only=self._config.current_only,
            types=types,
            since=since,
            ordered=ordered,
            subtree=subtree,
        )

    async def _search(
        self, query: str, *, expand: str = ""
    ) -> AsyncIterator[tuple[Mapping[str, object], str]]:
        """Every result of one CQL query, with the link base the page it arrived on declared."""
        params = [("cql", query), ("limit", str(self._config.page_size))]
        if expand:
            params.append(("expand", expand))
        async for payload in self._client.paginate(self._client.url(SEARCH_PATH), params):
            base = _link_base(payload, self._config.base_url)
            for result in _results(payload):
                yield result, base

    async def _scope(self, spaces: Sequence[str]) -> Subtree | None:
        """The configured page trees, validated, or ``None`` for whole-space syncing.

        ``None`` rather than an empty subtree, so that "no roots configured" is a different
        object from "roots configured and none of them resolved" — the second is a refusal, and
        a shared empty value would have made it look like the first.
        """
        if not self._config.root_page_ids:
            return None
        return await subtree.resolve(self._client, self._config, spaces, source=self.name)

    def _discovered(
        self,
        result: Mapping[str, object],
        *,
        space: str,
        base: str,
        roots: Sequence[str] = (),
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
        if roots:
            # Why this document is here, kept beside what it is. Derived metadata: the page id
            # is still its identity and the canonical URL is still its address, and neither is
            # computed from this.
            metadata[ROOT_PAGE_IDS] = list(roots)

        if kind == PAGE:
            page_crumbs: list[JsonValue] = [space_key, *_ancestor_titles(result)]
            metadata[ANCESTORS] = page_crumbs
            metadata[ANCESTOR_IDS] = list(subtree.ancestor_ids(result))
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
            # The only response that will ever describe this attachment: the download is bytes.
            when = _str(_version_when(result))
            if when:
                metadata[MODIFIED_AT] = when
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
        """The spaces to sync: the configured allowlist, checked one by one, or everything visible.

        **The two cases ask the source two different questions, and that is the point.** An
        allowlist is a scope boundary, so each configured key is looked up directly and nothing
        else is asked about. Only an unscoped source enumerates the catalog, because only an
        unscoped source has a use for it.

        The previous version enumerated first and then filtered, which meant a connector scoped
        to two spaces still paged through every space the account could see — proportional to
        the account's entitlements rather than to its configuration. Measured on an account
        entitled to 500 spaces and configured for two: six requests carrying 500 space records,
        against two requests carrying two. **On a small catalog the direct lookups cost one
        request more** — two of them against a single catalog page — and that is the right trade
        anyway, because the cost that matters grows with somebody else's wiki rather than with
        this configuration.

        Looked up one at a time rather than concurrently, deliberately: an allowlist is
        typically a handful of keys, and firing them at a rate-limited instance in parallel
        trades a bounded wait for a burst the retry logic would then have to absorb.

        Checked every run rather than cached either way: a space created since the last sync is
        picked up without a configuration change, and a space this account has *lost* is
        reported instead of silently contributing nothing.

        Raises:
            ConnectorError: A configured space is missing or invisible, or an unscoped account
                can see no spaces at all. Both before any content query goes out — CQL answers
                a query for a space that does not exist with an empty result set, exactly as it
                answers one for a space with nothing in it, so a typo would otherwise be a sync
                that runs, succeeds, indexes nothing, and leaves reconciliation proposing the
                deletion of everything that space ever contributed.
        """
        if not self._config.spaces:
            visible = await self._visible_spaces()
            if not visible:
                msg = (
                    f"this account can see no spaces at {self._config.base_url}. A sync would "
                    f"index nothing and reconciliation would then propose deleting everything "
                    f"already indexed, so it stops here. Check the credential, and that the "
                    f"account has been granted at least one space."
                )
                raise ConnectorError(msg)
            return list(visible.values())

        # Deduplicated on what the *source* calls each space rather than on what configuration
        # spelled, because `ENG` and `eng` are one space and would otherwise be enumerated
        # twice — every document in it discovered twice, and a second round trip to find that
        # out. Order is the configured order, which is the order somebody reading a log expects.
        chosen: dict[str, None] = {}
        for key in self._config.spaces:
            chosen[await self._space(key.strip())] = None
        return list(chosen)

    async def _space(self, key: str) -> str:
        """One configured space, confirmed against the source, in the source's own spelling.

        Returns the key **as the instance reports it** rather than as configuration spelled it.
        Confluence space keys are case-insensitive to look up and have one canonical casing, and
        that casing is what goes into every subsequent CQL literal — so echoing the configured
        string back would build queries against a spelling the source does not use.

        Raises:
            ConnectorError: The space is missing, or this account cannot see it — Confluence
                answers both with 404 and there is no way here to tell them apart, so the
                message says both. A credential or permission failure is *not* caught: it
                surfaces as itself, because "your token expired" and "that space is not there"
                need different repairs and collapsing them into the second sends somebody
                looking for a typo in a key that was always right.
        """
        # `quote` rather than interpolation, and the whole key is one path segment: a key
        # containing `/` or `?` would otherwise address a different resource entirely.
        url = f"{self._client.url(_SPACE_PATH)}/{quote(key, safe='')}"
        try:
            payload = await self._client.get_json(url, [])
        except NotFoundError as exc:
            msg = (
                f"configured space {key!r} is not there — either no space has that key, or "
                f"this account cannot see it, and Confluence answers both the same way. A "
                f"query for a space that is not there returns nothing rather than an error, so "
                f"this would otherwise be a sync that appears to work and a reconciliation "
                f"that proposes deleting everything the space ever contributed."
            )
            raise ConnectorError(msg) from exc
        # Deliberately not accompanied by a list of what *is* visible. The allowlist is a scope
        # boundary, and enumerating unrelated spaces to improve an error message is the request
        # this whole path exists to stop making.
        found = _str(payload.get("key"))
        if not found:
            msg = (
                f"the source answered for configured space {key!r} without naming a key, so "
                f"there is no canonical spelling to build a query from."
            )
            raise ConnectorError(msg)
        return found

    async def _visible_spaces(self) -> dict[str, str]:
        """Every space key this account can see, folded for comparison to its own spelling.

        Reached only by an unscoped source. A configured allowlist never comes through here —
        see :meth:`_space`.
        """
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

        No bodies and no versions — which is what makes a full enumeration affordable often
        enough to matter. Any failure propagates rather than being caught and smoothed over:
        the pipeline refuses to diff a partial enumeration, because the ids seen so far are a
        prefix and diffing a prefix soft-deletes everything past it (``docs/ingest.md`` §11.1).

        **This enumerates the same scope discovery does, by asking the same object.** A pass
        that reconciled a whole space against an index built from one page tree would soft-
        delete nothing and hide nothing; the mirror of it — discovery scoped and reconciliation
        wider, or the reverse — deletes the difference. So subtree membership is decided by
        :meth:`~manicule.connectors.subtree.Subtree.covering_roots` in both, and it is the same
        method rather than the same rule written twice.

        Subtree mode expands ``ancestors``, which whole-space mode has no use for. That is the
        one place the two differ in cost, and it buys the check that a page the source returned
        is really in the tree that was asked for.
        """
        spaces = await self._spaces()
        scope = await self._scope(spaces)
        if scope is None:
            types = (PAGE, ATTACHMENT) if self._config.include_attachments else (PAGE,)
            for space in spaces:
                query = self._content_query(space, types=types, ordered=False)
                async for result, _ in self._search(query):
                    source_id = _str(result.get("id"))
                    if source_id:
                        yield source_id
            return

        for space in scope.spaces():
            # The page enumeration and the scope are one thing: `members` is the live query,
            # already checked page by page against the configured roots and already guarded
            # against a subtree that only looks empty.
            members = await scope.members(space)
            for page_id in members:
                yield page_id
            if not self._config.include_attachments:
                continue
            query = self._content_query(space, types=(ATTACHMENT,), ordered=False)
            async for result, _ in self._search(query, expand="container"):
                container = _str(_obj(result.get("container")).get("id"))
                source_id = _str(result.get("id"))
                if source_id and container in members:
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

        **Its provenance is its own, and that is not a detail.** The attachment has a content id
        of its own, a version of its own and an address of its own, and citing the page instead
        would say that a reader following the reference arrives at the diagram — when what they
        would actually arrive at is a page that happens to have a diagram attached somewhere on
        it. The parent survives, in ``parent_page_id`` and in the breadcrumb, as a relationship
        rather than as an identity.

        The timestamp comes from the search result that discovered it, because the download is
        bytes and carries no version metadata at all. That is still a source response, and it is
        the only one that describes this attachment.
        """
        url = _str(ref.metadata.get(DOWNLOAD)) or ref.uri
        downloaded = await self._client.download(url, max_bytes=self._config.max_attachment_bytes)
        declared = _str(ref.metadata.get("media_type"))
        metadata = dict(ref.metadata)
        version = _int(ref.metadata.get(VERSION))
        if version is not None:
            metadata[VERSION_TOKEN] = str(version)
        # Two of these three came from the source and the third did not, and only the first two
        # may reach the record. `metadata.mediaType` is Confluence's own declaration and the
        # download's `Content-Type` is the response's; the filename extension is manicule's
        # inference, sound enough to route bytes by and not something the publisher said.
        stated = declared or downloaded.media_type
        media_type = stated or _from_name(_str(metadata.get("title")))
        metadata[PROVENANCE_KEY] = _record(
            what=f"attachment {ref.source_id}",
            title=_str(metadata.get("title")),
            canonical_uri=ref.uri,
            source_id=ref.source_id,
            version=str(version) if version is not None else "",
            # The search result carries no creation date for an attachment, and there is no
            # second response to ask. Absent rather than borrowed from the page holding it.
            created_at=None,
            modified_at=cql.parse_when(metadata.get(MODIFIED_AT)),
            content_type=stated,
            # Space and the page holding it. The attachment's own filename is left off for the
            # same reason a page's own title is: the chunker appends it.
            section_path=_str_values(ref, ANCESTORS),
        ).as_metadata_value()
        return RawDocument(
            source_id=ref.source_id,
            uri=ref.uri,
            media_type=media_type,
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
        ancestor_ids = body.ancestor_ids or _str_values(ref, ANCESTOR_IDS)
        complete = True
        if not ancestors:
            titles, ancestor_ids, complete = await self._ancestors_of(ref.source_id)
            ancestors = list(_str_list(space_key, *titles))
            # A breadcrumb starts at the space key, and on Cloud there may be no way to learn
            # it here: the body endpoint reports a numeric space id, and a ref rebuilt
            # elsewhere carries nothing. Inventing one is not available and is not wanted —
            # but neither is calling the result complete. Every chunk of this page is prefixed
            # one level short of the rest of the corpus, and this flag is the only thing that
            # says so.
            complete = complete and bool(space_key)

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
        # Carried from the ref rather than recomputed, and carried on **every** fetch rather
        # than only when there is something to say. The pipeline merges the stored metadata
        # under the fetched metadata, so a key this fetch omits keeps whatever the last one
        # wrote — which for a page that has since moved between two configured roots would be
        # the root it used to hang off.
        # ``None`` rather than an omitted key when nothing is scoped, so that a source which
        # stopped being subtree-scoped clears the provenance of the pages it keeps instead of
        # leaving them asserting a root that no longer selects anything.
        metadata[ROOT_PAGE_IDS] = list(_str_values(ref, ROOT_PAGE_IDS)) or None
        metadata[ANCESTOR_IDS] = list(ancestor_ids)
        metadata.update(report.as_metadata())
        if expected is not None and body.version != expected:
            metadata["version_disagreement"] = {"discovered": expected, "fetched": body.version}

        uri = _join(body.base, body.webui) or ref.uri
        media_type = self._page_media_type() if body.body_format == ADF_BODY else STORAGE_MEDIA_TYPE
        metadata[PROVENANCE_KEY] = _record(
            what=f"page {ref.source_id}",
            title=body.title,
            canonical_uri=uri,
            source_id=ref.source_id,
            # The version of the bytes above, which after the stale-body fallback is not always
            # the version discovery expected. The disagreement is recorded beside this and the
            # expected version is never what a citation quotes: a record naming the version
            # manicule asked for would describe a page this index does not hold.
            version=str(body.version),
            created_at=body.created_at,
            modified_at=body.modified_at,
            content_type=media_type,
            # The space and the pages above this one, and not this page's own title — the
            # chunker appends that itself. The same hierarchy the breadcrumb is built from,
            # asked about by a second name.
            section_path=tuple(part for part in ancestors if isinstance(part, str)),
        ).as_metadata_value()

        return RawDocument(
            source_id=ref.source_id,
            uri=uri,
            media_type=media_type,
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

        try:
            body = await self._adf_body(page_id)
        except BodyUnavailableError:
            # The page is there and the format is not. Storage format is a different code path
            # on the source's side, which is exactly the situation it exists for; anything
            # else — a 429, a rejected credential — must not be retried through a second
            # endpoint, which is why only this failure is caught.
            return await self._storage_body(page_id)
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

    async def _ancestors_of(self, page_id: str) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
        """Ancestor titles and ids for a page whose discovery record carried none, and whether
        the titles are all there.

        Discovery expands ``ancestors`` and puts both on the ref, so this runs only for a ref
        built somewhere else — a re-fetch from a stored one, a targeted single-page sync. The
        Atlassian Document Format endpoint does not carry ancestors, hence a second call rather
        than a wider expansion.

        **Ids come back from the same response as the titles**, because otherwise this path
        would record an empty ancestor-id list for a page that has ancestors — and an empty list
        is indistinguishable from a page at the top of its space, which is a different fact.

        The flag is returned rather than swallowed because a breadcrumb missing a level is not
        visibly wrong: it retrieves slightly worse and says nothing. An ancestor whose title
        the endpoint omitted is skipped rather than filled with its id, which would put a
        number into the text the embedder reads.
        """
        if not self._is_cloud:
            return (), (), True
        url = f"{self._client.url(_V2_PAGE_PATH)}/{page_id}/ancestors"
        try:
            payload = await self._client.get_json(url, [("limit", "25")])
        except NotFoundError:
            return (), (), True
        entries = _results(payload)
        titles = tuple(_str(entry.get("title")) for entry in entries)
        found = tuple(_str(entry.get("id")) for entry in entries)
        return (
            tuple(title for title in titles if title),
            tuple(entry for entry in found if entry),
            all(titles),
        )

    async def _adf_body(self, page_id: str) -> _Body:
        url = f"{self._client.url(_V2_PAGE_PATH)}/{page_id}"
        payload = await self._client.get_json(url, [("body-format", ADF_BODY)])
        body = _obj(_obj(payload.get("body")).get(ADF_BODY))
        value = _str(body.get("value"))
        if not value:
            msg = (
                f"page {page_id} came back with no Atlassian Document Format body. The page "
                f"exists, so this is the source declining the format rather than the page "
                f"being empty."
            )
            raise BodyUnavailableError(msg)
        links = _obj(payload.get("_links"))
        version = _obj(payload.get("version"))
        return _Body(
            page_id=page_id,
            title=_str(payload.get("title")),
            version=_int(version.get("number")) or 0,
            body=value,
            body_format=ADF_BODY,
            webui=_str(links.get("webui")),
            base=_str(links.get("base")) or self._config.base_url,
            # Cloud's v2 page: the current version's own `createdAt` is when the page was last
            # edited, and the page's top-level `createdAt` is when it first existed. Two fields
            # with one name at two levels, which is exactly the pair a single careless read
            # would collapse into a page that looks freshly revised because it is old.
            modified_at=cql.parse_when(version.get("createdAt")),
            created_at=cql.parse_when(payload.get("createdAt")),
        )

    async def _storage_body(self, page_id: str) -> _Body:
        url = f"{self._client.url(CONTENT_PATH)}/{page_id}"
        payload = await self._client.get_json(url, [("expand", _STORAGE_EXPAND)])
        body = _obj(_obj(payload.get("body")).get(STORAGE_BODY))
        links = _obj(payload.get("_links"))
        space_key = _str(_obj(payload.get("space")).get("key"))
        version = _obj(payload.get("version"))
        return _Body(
            page_id=page_id,
            title=_str(payload.get("title")),
            version=_int(version.get("number")) or 0,
            body=_str(body.get("value")),
            body_format=STORAGE_BODY,
            space_key=space_key,
            ancestors=(space_key, *_ancestor_titles(payload)) if space_key else (),
            ancestor_ids=subtree.ancestor_ids(payload),
            webui=_str(links.get("webui")),
            base=_str(links.get("base")) or self._config.base_url,
            # Server and Data Center spell the same fact `version.when`. There is no creation
            # date in this response: it lives under `history`, which this request does not
            # expand, so `created_at` stays absent rather than being sourced from anywhere
            # else. Widening the expansion to fill it would cost every page a larger response
            # for a field no citation currently renders.
            modified_at=cql.parse_when(version.get("when")),
        )

    # --- macro resolution ----------------------------------------------------------------

    def _lookup_for(self, body: _Body, space_key: str) -> Lookup:
        """How an include macro finds the page it names, memoized for this fetch.

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
        except (NotFoundError, BodyUnavailableError):
            # An include whose target cannot be read is recorded as unresolved, with the reason,
            # rather than failing the page that includes it: the rest of that page is content
            # somebody is looking for, and the gap is already reported where it happened.
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
        query = cql.title_query(space, title, current_only=self._config.current_only)
        params = [("cql", query), ("limit", "1")]
        payload = await self._client.get_json(self._client.url(SEARCH_PATH), params)
        results = _results(payload)
        return _str(results[0].get("id")) if results else ""


# --- the authoritative record ----------------------------------------------------------------


def _record(
    *,
    what: str,
    title: str,
    canonical_uri: str,
    source_id: str,
    version: str,
    created_at: datetime | None,
    modified_at: datetime | None,
    content_type: str,
    section_path: tuple[str, ...],
) -> Provenance:
    """One document's authoritative source record, or the stated reason it has none.

    **Every argument comes from a source response and none is inferred here.** That is the whole
    contract this function exists to hold: a record assembled partly from what manicule worked
    out is a claim that reads as the publisher's and is not. There is deliberately no clock in
    this function, no filesystem, and no access to the stored document — so the three timestamps
    that must never be confused (``modified_at``, ``retrieved_at``, ``indexed_at``) cannot be,
    because two of them are not reachable from here.

    ``created_at`` and ``modified_at`` arrive already parsed and already offset-aware, or
    ``None``. :func:`manicule.connectors.cql.parse_when` returns ``None`` for a naive timestamp
    rather than guessing a zone, so a response that omitted the offset produces an absent field
    and never a moment that is wrong by the instance's offset.

    Args:
        what: How to name this document if the record is refused. Read in a diagnostic, so it
            is an id and a kind rather than a title, which is content.
        title: The page or attachment title, as the source published it.
        canonical_uri: Where a reader opens it, resolved against the link base the source gave.
        source_id: The Confluence content id. Identity, unchanged by a rename or a move.
        version: The version of the bytes actually retained.
        created_at: When the source says it was created, on deployments that say.
        modified_at: When the source says this version was made.
        content_type: What this connector routed the bytes as.
        section_path: Space key and ancestor titles, coarsest first, own title excluded.

    Returns:
        A record carrying the source metadata, or one carrying only a reason — never nothing.
        A page whose title holds a control character, or whose link base resolved to something
        outside ``http``/``https``, is a page that still gets indexed and still gets a citation;
        what it does not get is a silent absence that looks identical to a connector which was
        never taught to record provenance at all.
    """
    try:
        source = SourceMetadata(
            title=title,
            canonical_uri=canonical_uri,
            source_id=source_id,
            version=version,
            created_at=created_at,
            modified_at=modified_at,
            content_type=content_type,
            section_path=tuple(part for part in section_path if part.strip()),
        )
    except ValueError as exc:
        return Provenance(
            unavailable_reason=f"{what}: the source declared metadata this index will not cite "
            f"({_collapsed(exc)})"
        )
    return Provenance(source=source)


def _collapsed(exc: ValueError) -> str:
    """A pydantic validation error as one line, for a diagnostic somebody reads in a log."""
    return " ".join(str(exc).split())


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


def _newest(stamps: Iterable[str]) -> str:
    """The latest of some ISO-8601 instants, compared as instants.

    Not ``max()`` over the strings. Timestamps keep the offset the instance reported (§2), and
    two spaces on one instance can report different offsets across a daylight-saving boundary —
    at which point the lexicographic maximum is the wrong string. Nothing interprets
    ``Watermark.value``, but a summary that is subtly wrong is worse than none: it is read by
    whoever is working out why a sync did what it did.
    """
    values = list(stamps)
    known = [(when, value) for value in values if (when := cql.parse_when(value)) is not None]
    if not known:
        return max(values)
    return max(known, key=lambda pair: pair[0])[1]


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
    return _str_values(ref, ANCESTORS)


def _str_values(ref: DocRef, key: str) -> tuple[str, ...]:
    """A list-of-strings metadata value from a ref, tolerating one written by anything else."""
    value = ref.metadata.get(key)
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
