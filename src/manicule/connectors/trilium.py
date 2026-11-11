"""The Trilium connector: one document per note, read from a server's ETAPI.

``docs/connectors/trilium.md`` is the design; :mod:`manicule.connectors.trilium_client` is the HTTP
half. This module turns one sweep of every note's metadata into a tree, decides which notes are
documents and where each sits, and answers the connector contract.

**ETAPI has no change feed and no cursor, so every run is a full inventory.** One ``GET /notes``
returns every note's metadata and no content (§6.1). Discovery yields every note that is in scope
with a version token, and the pipeline fetches the ones whose token it does not already hold, so
the sync is incremental in what it *reads* and complete in what it *knows*. That is the filesystem
connector's shape, and it is why every run carries deletion authority
(``enumerates_full_inventory``).

**A response that might be short is refused.** A sweep that returns exactly the limit it asked for
may have been cut off, and ``reconcile`` deletes what it does not see, so such a sweep raises
(§6.1). Raising deletes nothing.

**Nothing here may put a note's text in an error or a log record.** Both are built from counts and
operation names; whatever names a person's thoughts stays in the document it belongs to.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import logging
import re
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, cast

from pydantic import JsonValue

from manicule.connectors.config import (
    TRILIUM_CONNECTOR_NAME,
    TRILIUM_REPRESENTATION_VERSION,
    TriliumConfig,
)
from manicule.connectors.errors import ConnectorError, NotFoundError
from manicule.core.content import Metadata, RawDocument
from manicule.core.provenance import (
    MAX_FIELD_CHARS,
    MAX_SECTION_DEPTH,
    PROVENANCE_KEY,
    Provenance,
    SourceMetadata,
)
from manicule.core.sources import DiscoveredDoc, DocRef, SourceId, Watermark

if TYPE_CHECKING:
    from manicule.connectors.trilium_client import TriliumClient

__all__ = [
    "HTML",
    "INVENTORY_MAX_AGE_SECONDS",
    "MARKDOWN",
    "PLAIN",
    "SWEEP_EXPRESSION",
    "TriliumConnector",
]

_log = logging.getLogger("manicule.connectors.trilium")

SWEEP_EXPRESSION: Final = "note.noteId != ''"
"""The search expression that matches every note (``docs/connectors/trilium.md`` §6.1)."""

HTML: Final = "text/html"
MARKDOWN: Final = "text/markdown"
PLAIN: Final = "text/plain"

_INDEXED_TYPES: Final = frozenset({"text", "code", "mermaid"})
"""The note types whose content is text worth searching (§3). Everything else is a container, a
saved query, a drawing or UI configuration."""

_KNOWN_TYPES: Final = frozenset(
    {
        "text", "code", "render", "file", "image", "search", "relationMap", "book", "noteMap",
        "mermaid", "webView", "shortcut", "doc", "contentWidget", "launcher",
    }
)  # fmt: skip
"""Every ``Note.type`` ETAPI's published description lists. A type outside it is a Trilium newer
than this connector, and it is reported rather than quietly left out (§3)."""

INVENTORY_MAX_AGE_SECONDS: Final = 600.0
"""How long ``fetch`` trusts the last sweep before taking a fresh one (§6.3). Scope, exclusion and
protection are judged on a sweep, so a label added or a note protected since must reach a read
within a bounded time and not whenever the next discovery happens to run."""

_ARCHIVED_LABEL: Final = "archived"
_ROOT: Final = "root"
_SYSTEM_PREFIX: Final = "_"
"""Trilium's own notes (the hidden subtree, share and launcher configuration) have ids that begin
with an underscore. Neither they nor a note that exists only beneath them is a document (§3)."""

_CONTROL: Final = re.compile(r"[\x00-\x1f\x7f]")
_UNTITLED: Final = "untitled"


@dataclass(frozen=True, slots=True)
class _Note:
    """What the sweep says about one admitted note, and what it was judged to be."""

    note_id: str
    title: str
    note_type: str
    mime: str
    media_type: str
    blob_id: str
    created: str
    modified: str
    parents: tuple[str, ...]
    labels: tuple[tuple[str, str], ...]
    breadcrumb: tuple[str, ...]


@dataclass(slots=True)
class _Inventory:
    """One completed sweep: the notes in scope, and counts of what was left out and why."""

    notes: dict[str, _Note] = field(default_factory=lambda: {})
    protected: int = 0
    out_of_scope: int = 0
    unindexed_type: int = 0
    unplaced: int = 0
    unknown_types: set[str] = field(default_factory=set[str])
    newest: str = ""
    swept_at: float = 0.0


class TriliumConnector:
    """Discovers, fetches and reconciles the notes of one Trilium server."""

    enumerates_full_inventory = True
    """Every run sweeps the whole server, so every manifest is a full inventory (see module)."""

    def __init__(
        self,
        config: TriliumConfig,
        client: TriliumClient,
        *,
        name: str = TRILIUM_CONNECTOR_NAME,
        max_note_bytes: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Args:
        config: The validated settings, token resolved.
        client: The ETAPI client. The connector owns its lifecycle: ``setup`` and
            ``teardown`` are forwarded.
        name: The configured instance's name, which files its documents (see
            ``plugin._source_name``).
        max_note_bytes: ``ingest.max_fetch_bytes``. A note longer than this is refused while
            it streams rather than after it has been loaded (§6.3). ``None`` means no limit.
        clock: Monotonic time, for how old the sweep ``fetch`` relies on may be.
        """
        self.name = name
        self._config = config
        self._client = client
        self._max_note_bytes = max_note_bytes
        self._clock = clock
        self._inventory: _Inventory | None = None
        self._sweeping = asyncio.Lock()
        self._reached: Watermark | None = None

    # --- lifecycle -----------------------------------------------------------------------

    async def setup(self) -> None:
        await self._client.setup()

    async def teardown(self) -> None:
        await self._client.teardown()

    @property
    def max_note_bytes(self) -> int | None:
        """The ceiling on one note's content: ``ingest.max_fetch_bytes``, or ``None``."""
        return self._max_note_bytes

    # --- identity and scope --------------------------------------------------------------

    @property
    def source_scope(self) -> str:
        """A non-secret name for this scope: a digest of the server's address."""
        digest = hashlib.blake2b(self._config.base_url.encode(), digest_size=8).hexdigest()
        return f"trilium:{digest}"

    @property
    def scope_fingerprint(self) -> str:
        return self._config.scope_fingerprint(self.source_scope)

    @property
    def reconciliation_scope(self) -> str:
        """Fences a completed inventory, and a held deletion proposal, to this exact scope."""
        return f"trilium:{self.scope_fingerprint}"

    # --- watermark -----------------------------------------------------------------------

    @property
    def watermark(self) -> Watermark | None:
        """The newest ``utcDateModified`` of the last **complete** sweep, once there is one.

        Informational: ETAPI has no "modified since", so discovery does not consult it (§6.2).
        It is set only when a sweep ran to the end, so an interrupted run leaves the previous
        value in place.
        """
        return self._reached

    # --- the sweep -----------------------------------------------------------------------

    async def _sweep(self) -> _Inventory:
        """Every note in scope now, from one request, or an error.

        Raises:
            ConnectorError: The response may be truncated, or an entry is not a note. Either
                would make the inventory a prefix of the truth, and absence is deletion.
        """
        response = await self._client.search(SWEEP_EXPRESSION, limit=self._config.max_notes)
        if len(response) >= self._config.max_notes:
            msg = (
                f"the note search returned {len(response)} notes, which is the most this "
                f"connector asked for (max_notes). ETAPI has no cursor, so a response of exactly "
                f"that size may have been cut off, and an inventory missing notes would delete "
                f"them from the index. Raise max_notes above the number of notes the server has"
            )
            raise ConnectorError(msg)
        inventory = _judge(response, self._config)
        inventory.swept_at = self._clock()
        self._inventory = inventory
        return inventory

    async def _recent(self) -> _Inventory:
        """The last sweep if it is young enough to judge a read by, otherwise a fresh one.

        Under a lock, so concurrent reads that arrive with no usable sweep share one request and
        not one each.
        """
        async with self._sweeping:
            held = self._inventory
            if held is not None and self._clock() - held.swept_at < INVENTORY_MAX_AGE_SECONDS:
                return held
            return await self._sweep()

    async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
        """Yield every note in scope.

        ``watermark`` is accepted and deliberately not used to skip notes. ETAPI cannot filter
        on a date, and a rename or move of an ancestor changes every descendant's breadcrumb
        without touching the descendant's own modification date. The per-note version token is
        what makes the sync incremental, and the pipeline compares it before anything is read.
        """
        del watermark  # see the docstring: the token does the skipping, not the clock
        await self._client.setup()
        inventory = await self._sweep()
        for note in sorted(inventory.notes.values(), key=lambda n: n.note_id):
            yield self._discovered(note)
        _log.info(
            "trilium %s: %d notes in scope; skipped %d protected, %d out of scope, %d of a "
            "type that is not indexed, %d not reachable from the top of the tree",
            self.name,
            len(inventory.notes),
            inventory.protected,
            inventory.out_of_scope,
            inventory.unindexed_type,
            inventory.unplaced,
        )
        if inventory.unknown_types:
            _log.warning(
                "trilium %s: notes of a type this connector does not know were left out: %s. "
                "The server is newer than the connector",
                self.name,
                ", ".join(sorted(inventory.unknown_types)),
            )
        # Only after the enumeration has run to the end. A watermark stored for a partial one is
        # how documents go missing permanently.
        self._reached = Watermark(
            value=inventory.newest or datetime.now(tz=UTC).isoformat(),
            observed_at=datetime.now(tz=UTC),
            metadata={"scope_fingerprint": self.scope_fingerprint},
        )

    def _discovered(self, note: _Note) -> DiscoveredDoc:
        return DiscoveredDoc(
            ref=DocRef(source_id=note.note_id, uri=self._uri(note.note_id)),
            version_token=_version_token(note),
            title=note.title,
            media_type=note.media_type,
        )

    def _uri(self, note_id: str) -> str:
        return f"{self._config.base_url.rstrip('/')}/#root/{note_id}"

    # --- fetch ---------------------------------------------------------------------------

    async def fetch(self, ref: DocRef) -> RawDocument:
        """Read one note's content and wrap it for the parser chain (§3).

        Raises:
            NotFoundError: The note is not in scope: deleted, protected, excluded, or moved
                out of every root since it was swept. The pipeline reads this as the inventory
                having changed.
        """
        await self._client.setup()
        inventory = await self._recent()
        note = inventory.notes.get(ref.source_id)
        if note is None:
            msg = "the note is not one this source serves"
            raise NotFoundError(msg)
        limit = self._max_note_bytes if self._max_note_bytes is not None else 1 << 62
        content = await self._client.content(note.note_id, expected_type=note.mime, max_bytes=limit)
        body = (
            _html_document(note.title, content.content)
            if note.media_type == HTML
            else content.content
        )
        metadata: Metadata = {
            "trilium_note_id": note.note_id,
            "trilium_type": note.note_type,
            "trilium_mime": note.mime,
            "trilium_parent_note_ids": list(note.parents),
            "trilium_labels": _grouped(note.labels),
            "version_token": _version_token(note),
            PROVENANCE_KEY: self._provenance(note),
        }
        return RawDocument(
            source_id=note.note_id,
            uri=self._uri(note.note_id),
            media_type=note.media_type,
            content=body,
            metadata=metadata,
        )

    def _provenance(self, note: _Note) -> JsonValue:
        """What a citation says about the note: its title, where it sits and its own dates.

        A title a record cannot hold (it is over-long or carries a control character) is
        dropped from the record rather than the note from the index: the note is still
        described by its identity and its address.
        """
        for title in (note.title, ""):
            try:
                source = SourceMetadata(
                    title=title,
                    canonical_uri=self._uri(note.note_id),
                    source_id=note.note_id,
                    version=note.blob_id,
                    created_at=_parse_utc(note.created),
                    modified_at=_parse_utc(note.modified),
                    content_type=note.media_type,
                    section_path=note.breadcrumb,
                )
            except ValueError:
                continue
            return Provenance(source=source).as_metadata_value()
        return Provenance(unavailable_reason="the note could not be described").as_metadata_value()

    # --- reconciliation ------------------------------------------------------------------

    async def reconcile(self) -> AsyncIterator[SourceId]:
        """Every note in scope now, from a fresh sweep (§6.4).

        Raises, rather than ending early, on a response that may be truncated or an unreachable
        server: only exhaustion of this iterator is a complete inventory.
        """
        await self._client.setup()
        inventory = await self._sweep()
        for note_id in sorted(inventory.notes):
            yield note_id


# --- judging the sweep ---------------------------------------------------------------------


def _judge(response: list[Mapping[str, object]], config: TriliumConfig) -> _Inventory:
    """Decide which notes are documents, and place each in the tree (§3, §5).

    Plain steps in order: read every entry strictly, build the parent map, walk each note's
    ancestors once to judge scope, then compute each admitted note's breadcrumb.
    """
    entries = {entry.note_id: entry for entry in (_read(item) for item in response)}
    parents = {note_id: entry.parents for note_id, entry in entries.items()}
    depth, best_parent = _shallowest_paths(parents)
    exclude = set(config.exclude_labels)
    if not config.include_archived:
        exclude.add(_ARCHIVED_LABEL)
    roots = set(config.root_note_ids)
    excluded_here = {
        note_id
        for note_id, entry in entries.items()
        if any(name in exclude for name, _ in entry.labels)
    }

    inventory = _Inventory()
    for note_id, entry in sorted(entries.items()):
        if note_id == _ROOT or _is_system(note_id):
            continue
        if note_id not in depth:
            inventory.unplaced += 1
            continue
        family = list(_ancestry(note_id, parents))
        if entry.protected:
            inventory.protected += 1
            continue
        if excluded_here.intersection(family):
            inventory.out_of_scope += 1
            continue
        if roots and not roots.intersection(family):
            inventory.out_of_scope += 1
            continue
        if entry.note_type not in _INDEXED_TYPES:
            inventory.unindexed_type += 1
            if entry.note_type not in _KNOWN_TYPES:
                inventory.unknown_types.add(_clean(entry.note_type))
            continue
        inventory.notes[note_id] = _Note(
            note_id=note_id,
            title=_clean(entry.title),
            note_type=entry.note_type,
            mime=entry.mime,
            media_type=_media_type(entry.note_type, entry.mime),
            blob_id=entry.blob_id,
            created=entry.created,
            modified=entry.modified,
            parents=tuple(sorted(entry.parents)),
            labels=entry.labels,
            breadcrumb=_breadcrumb(note_id, best_parent, entries),
        )
        inventory.newest = max(inventory.newest, entry.modified)
    return inventory


@dataclass(frozen=True, slots=True)
class _Entry:
    note_id: str
    title: str
    note_type: str
    mime: str
    protected: bool
    blob_id: str
    created: str
    modified: str
    parents: tuple[str, ...]
    labels: tuple[tuple[str, str], ...]


def _read(item: Mapping[str, object]) -> _Entry:
    """One search result, read strictly: a shape this does not recognize raises.

    Lenient reading would drop an entry and so hide a note from the inventory, which deletes it.
    """
    note_id = _text(item, "noteId")
    parents = item.get("parentNoteIds")
    attributes = item.get("attributes", [])
    protected = item.get("isProtected")
    if (
        not note_id
        or not isinstance(parents, list)
        or not isinstance(attributes, list)
        or not isinstance(protected, bool)
    ):
        msg = "the note search returned an entry that is not shaped like an ETAPI note"
        raise ConnectorError(msg)
    labels: list[tuple[str, str]] = []
    for attribute in cast("list[object]", attributes):
        if isinstance(attribute, dict):
            fields = cast("dict[str, object]", attribute)
            name, value = fields.get("name"), fields.get("value", "")
            if fields.get("type") == "label" and isinstance(name, str):
                labels.append((name, value if isinstance(value, str) else ""))
    return _Entry(
        note_id=note_id,
        title=_text(item, "title"),
        note_type=_text(item, "type"),
        mime=_text(item, "mime"),
        protected=protected,
        blob_id=_text(item, "blobId"),
        created=_text(item, "utcDateCreated"),
        modified=_text(item, "utcDateModified"),
        parents=tuple(str(parent) for parent in cast("list[object]", parents)),
        labels=tuple(labels),
    )


def _text(item: Mapping[str, object], key: str) -> str:
    value = item.get(key, "")
    return value if isinstance(value, str) else ""


def _is_system(note_id: str) -> bool:
    return note_id.startswith(_SYSTEM_PREFIX)


def _ancestry(note_id: str, parents: Mapping[str, tuple[str, ...]]) -> Iterator[str]:
    """``note_id`` and every note above it on any path, each once.

    A set of seen ids guards the walk, so a tree that is not a tree (it cannot be, in Trilium)
    ends the walk instead of looping.
    """
    seen = {note_id}
    queue = deque([note_id])
    while queue:
        current = queue.popleft()
        yield current
        for parent in parents.get(current, ()):
            if parent not in seen:
                seen.add(parent)
                queue.append(parent)


def _shallowest_paths(
    parents: Mapping[str, tuple[str, ...]],
) -> tuple[dict[str, int], dict[str, str]]:
    """How many steps each note is from the top, and which parent is on its shortest path.

    A note cloned under several parents has several paths; the shortest wins and a tie goes to
    the smaller ``noteId``, so two runs agree (§4). The top is every note with no parent the
    sweep returned, which is ``root`` and whatever Trilium left out.

    **Trilium's own notes are not on any path.** Their ids begin with an underscore and the walk
    neither starts at one nor enters one, so a note that exists only inside them is absent from
    the result and a note *also* cloned into one (a bookmarked note is cloned under the launch
    bar) is placed by its real parents and not by the system's.
    """
    children: dict[str, list[str]] = {}
    for note_id, above in parents.items():
        for parent in above:
            children.setdefault(parent, []).append(note_id)
    depth: dict[str, int] = {}
    queue: deque[str] = deque()
    for note_id, above in sorted(parents.items()):
        if _is_system(note_id):
            continue
        if not any(parent in parents for parent in above):
            depth[note_id] = 0
            queue.append(note_id)
    while queue:
        current = queue.popleft()
        for child in sorted(children.get(current, ())):
            if child not in depth and not _is_system(child):
                depth[child] = depth[current] + 1
                queue.append(child)
    best: dict[str, str] = {}
    for note_id, above in parents.items():
        if note_id not in depth:
            continue
        candidates = [
            parent for parent in above if parent in depth and depth[parent] == depth[note_id] - 1
        ]
        if candidates:
            best[note_id] = min(candidates)
    return depth, best


def _breadcrumb(
    note_id: str, best_parent: Mapping[str, str], entries: Mapping[str, _Entry]
) -> tuple[str, ...]:
    """The titles from the top of the tree down to this note's parent (§4).

    ``root`` is not a level, and the note's own title is not in it: the chunker appends that. A
    protected note is not a level either: its title is the placeholder ETAPI substitutes, and a
    citation reading ``[protected]`` names nothing.
    Capped at the deepest hierarchy a record may carry, keeping the levels nearest the note.
    """
    titles: list[str] = []
    current = best_parent.get(note_id)
    seen: set[str] = {note_id}
    while current is not None and current not in seen:
        seen.add(current)
        above = entries.get(current)
        if current != _ROOT and not (above is not None and above.protected):
            titles.append(_clean(above.title) if above is not None else _UNTITLED)
        current = best_parent.get(current)
    titles.reverse()
    return tuple(titles[-MAX_SECTION_DEPTH:])


def _clean(title: str) -> str:
    """A title safe to print: control characters removed, bounded, never blank."""
    cleaned = _CONTROL.sub(" ", title).strip()[:MAX_FIELD_CHARS]
    return cleaned or _UNTITLED


def _media_type(note_type: str, mime: str) -> str:
    """What a note is delivered as (§3). Markdown is a ``code`` note with a Markdown mime."""
    if note_type == "text":
        return HTML
    if mime.split(";", 1)[0].strip().lower() in {"text/x-markdown", "text/markdown"}:
        return MARKDOWN
    return PLAIN


def _version_token(note: _Note) -> str:
    """A digest of everything that is part of the document's text (§4).

    ``blobId`` alone would miss a rename, which changes the heading and the breadcrumb without
    changing a byte of content, and ``utcDateModified`` rides along so that a server whose
    ``blobId`` is stable across an edit still re-reads it.
    """
    material = json.dumps(
        [
            TRILIUM_REPRESENTATION_VERSION,
            note.blob_id,
            note.modified,
            note.title,
            note.note_type,
            note.mime,
            list(note.breadcrumb),
            [list(label) for label in note.labels],
        ],
        separators=(",", ":"),
    )
    return hashlib.blake2b(material.encode(), digest_size=20).hexdigest()


def _grouped(labels: tuple[tuple[str, str], ...]) -> dict[str, JsonValue]:
    """Label values by name. Trilium allows several labels with one name (``#tag=a #tag=b``), so a
    plain mapping would keep the last and lose the rest."""
    grouped: dict[str, list[str]] = {}
    for name, value in labels:
        grouped.setdefault(name, []).append(value)
    return {name: list(values) for name, values in grouped.items()}


def _html_document(title: str, body: bytes) -> bytes:
    """A note's HTML under its title as ``<h1>``, so the parser makes the title the heading (§3)."""
    escaped = html.escape(title)
    text = body.decode("utf-8", errors="replace")
    return (
        f'<!doctype html><html><head><meta charset="utf-8"><title>{escaped}</title></head>'
        f"<body><h1>{escaped}</h1>\n{text}</body></html>"
    ).encode()


def _parse_utc(value: str) -> datetime | None:
    """Trilium's ``2024-05-01 12:30:00.000Z``, or ``None`` for anything else."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace(" ", "T")).astimezone(UTC)
    except ValueError:
        return None
