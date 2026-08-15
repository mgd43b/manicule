"""A directory of locally mirrored Confluence pages as a source, with no network and no credentials.

The live connector in :mod:`.confluence` needs a base URL, a credential and a reachable instance.
This one needs a directory. It exists because an export is often the only thing available — an
air-gapped install, a wiki nobody has API access to, a snapshot taken once and archived — and
because generic filesystem ingestion of that export throws away everything that makes a citation
useful. A page stored as ``123456.xhtml`` cites as ``123456.xhtml``; the manifest beside it knows
the page's title, its address, its version and where it sat.

**One directory is one page.** A directory containing :data:`MANIFEST_NAME` is a page snapshot:
the manifest, and the raw representation beside it. The walk does not descend into a page
snapshot, so attachments stored alongside cannot be mistaken for pages of their own.

Five decisions carry this module, and each is a trap something else here already learned about.

**Identity is the page id, never the path.** This is the difference from
:class:`~manicule.connectors.filesystem.FilesystemConnector`, where identity *is* the resolved
path, and it is the whole of "an updated page replaces its previous version". A mirroring tool
that renames a directory, or organizes by space this year and by tree next year, has not created
new pages — but a connector keyed on the path would report every document deleted and every
document new. The page id is the one handle Confluence itself promises is stable, so
``DocRef.source_id`` is the page id and the directory travels in ``DocRef.metadata``, which is
what that field is for. :func:`~manicule.parsers.expansion.member_source_id` settled the same
question for archive members: identity comes from a stable key, never from a position.

**The manifest never authorizes a read.** Carried over from :mod:`.sidecar` unchanged, because it
is the same threat: a manifest is a file in the corpus, so anyone who can get a directory indexed
can write one. The raw representation is found by *looking* — it is the file beside the manifest —
and a manifest that declares a filename has that declaration compared against what was found, not
followed. So a ``../../../../etc/passwd`` in a manifest is a name matching nothing in the
directory, refused with a reason, and at no point a path anything opens.

**The change token covers the pair.** A manifest edited to correct a title or record a new version
changes what every citation says while leaving the page's bytes untouched. A token taken from the
raw file alone would not move, the pipeline would skip before fetching, and the correction would
never be read on any later sync — the failure ``docs/ingest.md`` §4 now names. Both files' size
and modification time go into the token, so an edit to either is a change to the pair.

**Nothing is dropped in silence, including a snapshot that cannot be used.** Every directory
holding a manifest becomes a document, even when the manifest is unreadable or the body is missing
or ambiguous. Skipping them would be the quiet failure: an export of ten thousand pages would
ingest nine thousand and report success. So an unusable snapshot is ingested carrying the reason,
and lands in a status ``doctor`` reports. Where no page id can be read there is no identity to key
on, so one is derived from the directory under an explicit :data:`UNIDENTIFIED_PREFIX` — and when
the manifest is later fixed that placeholder stops appearing in
:meth:`~ConfluenceSnapshotConnector.reconcile` and is soft-deleted by the ordinary reconciliation
pass, while the page appears under its real id.

**What this connector cannot yet promise, it says out loud.** Storage format is XHTML with
Confluence's own ``ac:`` and ``ri:`` elements in it, and until a parser understands them it is
read by the generic HTML parser — which is what the live connector does today, so this is no
worse. It is, however, *lossy in a way nothing reported*: an HTML parser has no CDATA in that
context, so every ``ac:plain-text-body`` — the body of every code, noformat and Graphviz macro —
reparses as a bogus comment and its content is dropped entirely. Not degraded, dropped. So every
fetched page is scanned for the macros it contains and the finding is recorded under
:data:`UNINTERPRETED_MACROS`. A partial parse that reports itself is a usable interim; a partial
parse that reports success is a corpus that is quietly wrong, which is the one thing this must not
ship.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from manicule.connectors.confluence import SPACE_KEY, STORAGE_MEDIA_TYPE
from manicule.connectors.errors import NotFoundError
from manicule.connectors.macros import storage_macros
from manicule.core.content import Metadata, RawDocument
from manicule.core.ids import content_hash
from manicule.core.provenance import (
    PROVENANCE_KEY,
    LocalSnapshot,
    Provenance,
    SourceMetadata,
)
from manicule.core.sources import DiscoveredDoc, DocRef, Watermark

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence

    from manicule.core.sources import SourceId

MANIFEST_NAME: Final = "confluence.json"
"""What marks a directory as one page's snapshot.

A fixed name rather than a pattern, because the name is what the walk tests to decide whether a
directory *is* a page — and a pattern would make that answer depend on which file the filesystem
happened to list first.
"""

MAX_MANIFEST_BYTES: Final = 256 * 1024
"""Largest manifest that will be read.

Larger than :data:`~manicule.connectors.sidecar.MAX_MANIFEST_BYTES` because this one legitimately
carries lists — labels, ancestors, attachment references — for a page that may have many. Still a
ceiling, because otherwise a file called ``confluence.json`` decides how much memory a run
allocates.
"""

UNIDENTIFIED_PREFIX: Final = "unidentified:"
"""What a snapshot whose page id could not be read is keyed on instead.

An explicit prefix, so nothing can mistake a derived identity for a Confluence page id — a bare
directory name in that field would be indistinguishable from a page whose id happens to be a word.
The identity is deliberately unstable across a manifest repair: fixing the manifest makes the page
appear under its real id and makes this one stop appearing at all, which reconciliation resolves
by soft-deleting it. A stable-looking placeholder would instead leave two live documents for one
page.
"""

ANCESTOR_IDS: Final = "ancestor_ids"
CONTENT_STATUS: Final = "content_status"
LABELS: Final = "labels"
ATTACHMENTS: Final = "attachments"
"""Metadata keys for the facts that are Confluence's and nobody else's.

**Deliberately not fields on** :class:`~manicule.core.provenance.SourceMetadata`. That model
carries what *every* source has — a title, an address, an identity, a version, a hierarchy — and a
space key or a content status is not that: a filesystem mirror and a documentation export have
nothing to put
there. The rule, which ``docs/storage.md`` §4.2.1 states and this module is the first real test of:
the record carries what every source has, and anything one product means and others do not stays
in the connector's own keys. ``space_key`` is spelled by importing
:data:`~manicule.connectors.confluence.SPACE_KEY` rather than repeated here, because two spellings
of one key are two keys and a citation would read whichever it happened to look for.
"""

UNINTERPRETED_MACROS: Final = "uninterpreted_macros"
"""Metadata key naming the macros on a page that the parser reading it will not understand.

**Names only, and only the ones that are really uninterpreted.** It used to list every macro the
page contained, which was true when nothing read any of them; the storage-format parser reads most
of them now, so listing those would report a loss that does not happen. What remains is the set
this parser has no reader for — the ones it emits an explicit placeholder for — and that set is
asked of the parser rather than restated here, so the two cannot drift apart.

Absent when there is nothing to say. A diagnostic on every page is a diagnostic nobody reads.
"""

UNRECOVERABLE_MACRO_BODY: Final = "unrecoverable_macro_body"
"""Metadata key recording a macro body that could not be recovered as text at all.

The narrow remainder of a warning that used to be broad and wrong. Before #90 every CDATA body was
absent from the index, so the connector said so on every page carrying one; since #90 they are
recovered as escaped text and that sentence became false. What is still true, and much rarer, is an
export truncated inside a CDATA section: the section never closes, recovery leaves it exactly as it
is rather than guessing where the author meant it to end, and its content really does not reach the
index.

Separate from :data:`UNINTERPRETED_MACROS` because they are different claims — "we did not
understand this macro" and "we do not have this macro's text" — and the previous design put the
second into the *list* of the first as a sentence beginning ``!``, which made a list of names into a
list of names-and-one-paragraph that nothing could read programmatically.
"""

_DIRECTORY: Final = "snapshot_directory"
"""``DocRef.metadata`` key carrying the page's directory, for ``fetch`` and nothing else."""


class _Manifest(BaseModel):
    """The on-disk shape of a page manifest, exactly.

    A **wire format**: what somebody else's export tool writes, so its field names are a
    compatibility surface and its spellings are Confluence's rather than manicule's. That is why
    it is a separate model from :class:`~manicule.core.provenance.SourceMetadata` and not a
    convenience over it — the mapping between the two is where a product's vocabulary is
    translated into the core's, and collapsing them would put ``space_key`` inside the model whose
    entire purpose is that it cannot hold one.

    ``page_id`` is the only required field. Everything else is optional because ``bugs/bug1.md``
    says not to require every optional field, and because a tool that knows a page's id and title
    but not its ancestors should say so rather than invent them. ``page_id`` is not optional for a
    reason no default can supply: it is the document's identity, and a snapshot with no stable
    identity cannot be updated or reconciled.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    page_id: str = Field(min_length=1, description="Confluence's own page id. The identity.")
    title: str = ""
    space_key: str = ""
    canonical_url: str = ""
    version: str | int = ""
    created_at: datetime | None = None
    modified_at: datetime | None = None
    ancestors: tuple[str, ...] = Field(
        default=(), description="Ancestor titles, outermost first. Feeds the generic hierarchy."
    )
    ancestor_ids: tuple[str, ...] = ()
    content_status: str = ""
    labels: tuple[str, ...] = ()
    attachments: tuple[str, ...] = Field(
        default=(),
        description="Attachment filenames, recorded and never fetched. Attachment ingestion is a "
        "separate concern, and keeping the references is what lets it be added without a re-crawl.",
    )
    retrieved_at: datetime | None = None
    body_file: str = Field(
        default="",
        description="What the tool believes it wrote the body as. Cross-checked against the file "
        "actually found beside this manifest; never used to locate anything.",
    )
    body_checksum: str = Field(
        default="", description="The tool's digest of the body bytes. Cross-checked, never trusted."
    )

    def source(self) -> SourceMetadata:
        """The canonical half, in manicule's vocabulary.

        The hierarchy is the space key followed by the ancestor titles — the same shape the live
        connector puts in ``ancestors`` and the order the breadcrumb wants, coarsest first. The
        page's own title is **not** appended: the chunker adds that itself, and a breadcrumb
        carrying it twice reaches the embedder as emphasis nobody intended.

        Raises:
            _UnusableManifestError: Any field the interface refuses, named. Translated rather than
            allowed to escape as a :class:`~pydantic.ValidationError`, so one refusal type reaches
            the caller and a metadata typo costs diagnostics rather than the page.
        """
        section = tuple(part for part in (self.space_key, *self.ancestors) if part.strip())
        try:
            return SourceMetadata(
                title=self.title,
                canonical_uri=self.canonical_url,
                source_id=self.page_id,
                version=str(self.version),
                created_at=self.created_at,
                modified_at=self.modified_at,
                content_type=STORAGE_MEDIA_TYPE,
                section_path=section,
            )
        except ValidationError as exc:
            msg = f"declares metadata this index will not cite ({_reasons(exc)})"
            raise _UnusableManifestError(msg) from exc

    def connector_metadata(self) -> Metadata:
        """The Confluence-specific facts, under this connector's own keys.

        Empty values are omitted rather than written as blanks: a stored ``"content_status": ""``
        claims the manifest said something about status, and it did not.
        """
        found: Metadata = {}
        if self.space_key:
            found[SPACE_KEY] = self.space_key
        if self.ancestor_ids:
            found[ANCESTOR_IDS] = list(self.ancestor_ids)
        if self.content_status:
            found[CONTENT_STATUS] = self.content_status
        if self.labels:
            found[LABELS] = list(self.labels)
        if self.attachments:
            found[ATTACHMENTS] = list(self.attachments)
        return found


class _Snapshot(BaseModel):
    """One directory holding a manifest, whether or not it turned out to be usable.

    Three states, and the type holds all three so the walk never has to decide whether a directory
    "counts": a usable page (``parsed`` and ``body`` both set), a page whose manifest parsed but
    whose body could not be settled (``parsed`` set, ``body`` ``None``), and a directory whose
    manifest could not be read at all (``parsed`` ``None``). Every one of them becomes a document,
    because the alternative is an export that silently ingests a subset.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    directory: Path
    manifest: Path
    parsed: _Manifest | None = None
    body: Path | None = None
    refusal: str = ""

    @property
    def usable(self) -> bool:
        return self.parsed is not None and self.body is not None

    def identity(self, *, root: Path) -> SourceId:
        """What reconciliation compares this snapshot by."""
        if self.parsed is not None:
            return self.parsed.page_id
        return f"{UNIDENTIFIED_PREFIX}{_relative(self.directory, root=root)}"


class _UnusableManifestError(Exception):
    """A manifest was found and cannot be used. The message is what an operator reads."""


class ConfluenceSnapshotConnector:
    """A directory of mirrored Confluence pages as a source.

    Satisfies :class:`~manicule.core.protocols.Connector`. Implements none of the optional
    lifecycle protocols: there is no session to open, no credential to check and no instance to
    reach, which is the point.
    """

    def __init__(self, root: Path, *, name: str = "confluence-snapshot") -> None:
        """Point the connector at a root.

        Args:
            root: A directory holding page snapshots, at any depth. Resolved once, so a relative
            path and a symlink to the same tree produce the same reads. name: The source name.
            Part of every document's identity.
        """
        self.name = name
        self._root = root.expanduser().resolve()
        self._reached: datetime | None = None

    @property
    def root(self) -> Path:
        return self._root

    @property
    def source_scope(self) -> str:
        """Resolved mirror root whose deterministic membership forms the snapshot scope."""
        return str(self._root)

    @property
    def watermark(self) -> Watermark | None:
        """When the last **complete** walk finished.

        A directory has no change feed, so this records a time rather than a position and every
        walk is a full one. Set only after ``discover`` has run to the end, so an interrupted walk
        leaves the previous value in place — a watermark advanced by a partial enumeration is a
        position past documents nobody received.
        """
        if self._reached is None:
            return None
        return Watermark(value=self._reached.isoformat(), observed_at=self._reached)

    async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
        """Yield every page snapshot under the root, usable or not.

        ``watermark`` is accepted and deliberately not used to skip. A modification time older
        than the last walk does not mean a page is unchanged — a restored backup and a re-export
        both move content without moving the clock forward — and the per-document token is what
        makes the sync incremental.
        """
        del watermark  # the token does the skipping, not the clock
        started = datetime.now(UTC)
        for snapshot in self._walk():
            yield self._discovered(snapshot)
        # Only after the walk has run to the end. A watermark stored for a partial enumeration is
        # how documents go missing permanently.
        self._reached = started

    async def fetch(self, ref: DocRef) -> RawDocument:
        """Read one page's raw representation, or report why there is none to read.

        A snapshot that cannot be used still returns a document — empty bytes and the reason on
        the record — because the pipeline then stores it in a status ``doctor`` surfaces. Raising
        here instead would make an unreadable manifest look like a fetch failure, which is a
        different fault with a different remedy.

        Raises:
            NotFoundError: The directory named on the ref is outside the root this connector
            serves, or has stopped being a snapshot since discovery. Both are refusals rather than
            reads.
        """
        directory = Path(str(ref.metadata.get(_DIRECTORY, "")))
        if not directory.name or not self._within_root(directory):
            msg = (
                f"{ref.source_id!r} names {str(directory)!r}, which is outside {self._root} — the "
                f"only tree this source serves"
            )
            raise NotFoundError(msg)
        snapshot = await asyncio.to_thread(self._read, directory)
        if snapshot is None:
            msg = f"{ref.source_id!r} no longer has a manifest in {directory}"
            raise NotFoundError(msg)

        body = b""
        if snapshot.body is not None:
            try:
                body = await asyncio.to_thread(snapshot.body.read_bytes)
            except OSError as exc:
                msg = f"cannot read {snapshot.body}: {exc}"
                raise NotFoundError(msg) from exc

        metadata = self._metadata_for(snapshot, body)
        fetched_token = _version_token(snapshot)
        if fetched_token is not None:
            metadata["version_token"] = fetched_token

        return RawDocument(
            source_id=ref.source_id,
            # The body's own location, not the canonical URL. A parse failure must name the file it
            # choked on, and the pipeline reads the canonical address off the record instead.
            uri=ref.uri,
            media_type=STORAGE_MEDIA_TYPE,
            content=body,
            metadata=metadata,
        )

    async def reconcile(self) -> AsyncIterator[SourceId]:
        """Yield the identity of every snapshot that still exists.

        The pass incremental sync cannot do. A page deleted from a later complete export simply
        stops appearing, so without this the index serves it for ever. Shares :meth:`_walk` with
        ``discover``, so the two cannot disagree about what a page is — a snapshot yielded by one
        and not the other would be reported deleted on every sync, or indexed and never
        reconciled.
        """
        for snapshot in self._walk():
            yield snapshot.identity(root=self._root)

    # --- walking ----------------------------------------------------------------------------

    def _walk(self) -> Iterator[_Snapshot]:
        """Every page snapshot under the root, in a stable order.

        Sorted at each level rather than left to the filesystem, so two machines ingesting one
        export take it in the same order — which is what makes a ``--limit`` mean the same thing
        twice.
        """
        yield from self._walk_directory(self._root)

    def _walk_directory(self, directory: Path) -> Iterator[_Snapshot]:
        snapshot = self._read(directory)
        if snapshot is not None:
            # A page snapshot is a leaf. Descending would let a directory of attachments beside a
            # body be mistaken for pages, and a nested manifest override its parent's.
            yield snapshot
            return
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            return
        for entry in entries:
            if entry.name.startswith(".") or entry.is_symlink():
                # Followed nowhere: a symlink out of the tree is the escape `fetch` refuses, and a
                # loop inside it is an infinite walk.
                continue
            if entry.is_dir():
                yield from self._walk_directory(entry)

    def _read(self, directory: Path) -> _Snapshot | None:
        """The snapshot in ``directory``, or ``None`` when the directory is not one.

        ``None`` means only "no manifest here". Every other outcome is a ``_Snapshot`` carrying
        what went wrong, because a directory that *claims* to be a page by holding a manifest and
        then cannot be used is precisely what must not vanish.
        """
        manifest = directory / MANIFEST_NAME
        try:
            if not manifest.is_file():
                return None
        except OSError:  # pragma: no cover - an unstattable directory is not a page
            return None
        try:
            parsed = self._parse(manifest)
        except _UnusableManifestError as refusal:
            return _Snapshot(directory=directory, manifest=manifest, refusal=str(refusal))
        try:
            body = self._body_of(directory, parsed)
        except _UnusableManifestError as refusal:
            return _Snapshot(
                directory=directory, manifest=manifest, parsed=parsed, refusal=str(refusal)
            )
        return _Snapshot(directory=directory, manifest=manifest, parsed=parsed, body=body)

    def _within_root(self, path: Path) -> bool:
        try:
            resolved = path.resolve()
        except OSError:  # pragma: no cover - resolution failure is itself a refusal
            return False
        return resolved == self._root or self._root in resolved.parents

    # --- reading ----------------------------------------------------------------------------

    def _parse(self, manifest: Path) -> _Manifest:
        """Read and validate one manifest, or raise saying why not."""
        try:
            size = manifest.stat().st_size
        except OSError as exc:
            msg = f"cannot be read ({exc.strerror or exc})"
            raise _UnusableManifestError(msg) from exc
        if size > MAX_MANIFEST_BYTES:
            msg = f"is {size} bytes, over the {MAX_MANIFEST_BYTES}-byte manifest limit"
            raise _UnusableManifestError(msg)
        try:
            text = manifest.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            msg = f"is not readable UTF-8 ({exc})"
            raise _UnusableManifestError(msg) from exc
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            msg = f"is not valid JSON ({exc.msg} at line {exc.lineno} column {exc.colno})"
            raise _UnusableManifestError(msg) from exc
        if not isinstance(loaded, dict):
            msg = f"holds a JSON {type(loaded).__name__}, not an object of fields"
            raise _UnusableManifestError(msg)
        try:
            return _Manifest.model_validate(loaded)
        except ValidationError as exc:
            msg = f"is not a usable page manifest ({_reasons(exc)})"
            raise _UnusableManifestError(msg) from exc

    def _body_of(self, directory: Path, parsed: _Manifest) -> Path:
        """The raw representation beside the manifest, found by looking rather than by being told.

        Deterministic, and it refuses rather than guesses: exactly one file that is not the
        manifest is the body. Several, and the manifest must say which — and that declaration is
        *compared* against what is present, never joined to a root or opened. So a declared
        ``../../etc/passwd`` is a name matching nothing here and is refused for the same reason a
        stale filename is.

        Raises:
            _UnusableManifestError: No body; or several with nothing to choose between them; or a
            declaration naming a file that is not here.
        """
        try:
            candidates = sorted(
                entry
                for entry in directory.iterdir()
                if entry.is_file() and entry.name != MANIFEST_NAME and not entry.is_symlink()
            )
        except OSError as exc:
            msg = f"is in a directory that cannot be listed ({exc.strerror or exc})"
            raise _UnusableManifestError(msg) from exc

        declared = parsed.body_file.strip()
        if declared:
            wanted = _normalized(declared)
            found = next((path for path in candidates if _normalized(path.name) == wanted), None)
            if found is None:
                msg = (
                    f"declares body_file {parsed.body_file!r}, which is not a file in this "
                    f"snapshot. Present: {_names(candidates)}. A manifest describes the page "
                    f"beside it and never points anywhere else"
                )
                raise _UnusableManifestError(msg)
            return found

        if not candidates:
            msg = "has no raw page representation beside it, so there is nothing to index"
            raise _UnusableManifestError(msg)
        if len(candidates) > 1:
            msg = (
                f"sits beside {len(candidates)} files and declares no body_file, so which one is "
                f"the page would be a guess: {_names(candidates)}"
            )
            raise _UnusableManifestError(msg)
        return candidates[0]

    # --- what a page reports ----------------------------------------------------------------

    def _discovered(self, snapshot: _Snapshot) -> DiscoveredDoc:
        parsed = snapshot.parsed
        body = snapshot.body
        return DiscoveredDoc(
            ref=DocRef(
                # Identity is the page id where there is one. The directory travels in metadata
                # because that is what `fetch` needs, and nothing more.
                source_id=snapshot.identity(root=self._root),
                uri=(body or snapshot.manifest).as_uri(),
                metadata={_DIRECTORY: str(snapshot.directory)},
            ),
            version_token=_version_token(snapshot),
            title=(parsed.title if parsed else "") or (body.name if body else ""),
            media_type=STORAGE_MEDIA_TYPE,
        )

    def _metadata_for(self, snapshot: _Snapshot, body: bytes) -> Metadata:
        """Everything the pipeline should know about this page beyond its bytes."""
        parsed = snapshot.parsed
        metadata: Metadata = dict(parsed.connector_metadata()) if parsed else {}
        metadata[PROVENANCE_KEY] = self._record(snapshot, body).as_metadata_value()
        uninterpreted = _uninterpreted(body)
        if uninterpreted:
            metadata[UNINTERPRETED_MACROS] = uninterpreted
        unrecoverable = _unrecoverable_body(body)
        if unrecoverable:
            metadata[UNRECOVERABLE_MACRO_BODY] = unrecoverable
        return metadata

    def _record(self, snapshot: _Snapshot, body: bytes) -> Provenance:
        """The source record for this snapshot, or the reason it has none.

        The refusals are checked in the order that decides the outcome, and each returns rather
        than overwriting a record built earlier — building one and then discarding it is how a
        corpus ends up storing the record it decided against.
        """
        location = _snapshot_of(snapshot, root=self._root)
        if snapshot.refusal:
            return Provenance(
                snapshot=location, unavailable_reason=f"{MANIFEST_NAME}: {snapshot.refusal}"
            )
        parsed = snapshot.parsed
        if parsed is None:  # pragma: no cover - a refusal is set whenever `parsed` is None
            return Provenance(
                snapshot=location, unavailable_reason=f"{MANIFEST_NAME}: could not be read"
            )
        mismatch = _checksum_mismatch(parsed, body)
        if mismatch is not None:
            return Provenance(snapshot=location, unavailable_reason=f"{MANIFEST_NAME}: {mismatch}")
        try:
            source = parsed.source()
        except _UnusableManifestError as refusal:
            return Provenance(snapshot=location, unavailable_reason=f"{MANIFEST_NAME}: {refusal}")
        return Provenance(
            source=source,
            snapshot=location.model_copy(update={"retrieved_at": parsed.retrieved_at}),
        )


def _version_token(snapshot: _Snapshot) -> str | None:
    """A change token covering the body *and* its manifest.

    Both, for the reason ``docs/ingest.md`` §4 gives: a manifest corrected to declare a new
    version changes every citation of the page while leaving its bytes identical, and a token
    blind to the manifest would skip the page before fetching and never read the correction.

    ``None`` — "fetch and hash to find out" — where a stat fails, which is honest rather than
    convenient: inventing a token that does not move would skip a changed page for ever.
    """
    stamps: list[str] = []
    for candidate in (snapshot.body, snapshot.manifest):
        if candidate is None:
            continue
        try:
            stat = candidate.stat()
        except OSError:
            return None
        stamps.append(f"{stat.st_size}:{stat.st_mtime_ns}")
    return "+".join(stamps) if stamps else None


def _snapshot_of(snapshot: _Snapshot, *, root: Path) -> LocalSnapshot:
    """Where this copy sits, as observed. Relative, so a citation reproduces on another machine."""
    located = snapshot.body or snapshot.directory
    return LocalSnapshot(path=_relative(located, root=root))


def _checksum_mismatch(parsed: _Manifest, body: bytes) -> str | None:
    """Why a declared body checksum cannot be believed, or ``None`` when it agrees or is absent.

    A manifest and a body from different exports describe different documents, and that is the
    quiet version of the whole bug this connector exists to fix: the citation would carry a title
    and a version belonging to a page whose bytes are not the ones in the index.
    """
    stated = parsed.body_checksum.strip()
    if not stated:
        return None
    digest = content_hash(body)
    if stated == digest:
        return None
    return (
        f"declares body_checksum {stated!r}, but the bytes read hash to {digest!r}. The manifest "
        f"and the page beside it are not from the same export"
    )


def _uninterpreted(body: bytes) -> list[JsonValue]:
    """The macros in ``body`` that the parser reading it will not understand.

    Located with :func:`~manicule.connectors.macros.storage_macros`, which parses rather than
    matches text, so the names are the macros that are really there — then filtered against what
    the storage-format parser declares it reads. Imported from the parser rather than restated,
    for the reason :meth:`ConfluenceConnector._page_media_type` imports its media type: a second
    copy of the answer goes stale the first time a macro is taught to one and not the other, and
    the failure is a diagnostic that quietly describes a version of the code that no longer exists.

    **Semantics only. This says nothing about whether the text arrived**, which is
    :data:`UNRECOVERABLE_MACRO_BODY`'s question and a different one.
    """
    from manicule.parsers.confluence import INTERPRETED_MACROS  # noqa: PLC0415 - see docstring

    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        # Undecodable bytes are the parser chain's to report, not this scan's to guess at.
        return []
    return cast(
        "list[JsonValue]",
        sorted(
            {
                macro.name
                for macro in storage_macros(text)
                if macro.name and macro.name.lower() not in INTERPRETED_MACROS
            }
        ),
    )


def _unrecoverable_body(body: bytes) -> str:
    """Why a macro body in ``body`` cannot reach the index, or ``""`` when they all can.

    One case, and it is the only one left: a CDATA section that never closes.
    :func:`~manicule.parsers.web.recover_cdata` leaves an unterminated section exactly as it found
    it rather than guessing where the author meant it to end, so its content is not recovered and
    genuinely is absent.

    The scan mirrors that function's own loop rather than counting delimiters, because the two have
    to agree about what "unterminated" means — a count would call ``]]>]]>`` balanced and this
    would then contradict the parser about a document neither of them could read.
    """
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return ""
    rest = text
    while True:
        _, opened, after = rest.partition("<![CDATA[")
        if not opened:
            return ""
        _, closed, remainder = after.partition("]]>")
        if not closed:
            return (
                "a CDATA section is never closed, so the macro body it holds is not recovered as "
                "text and its content is absent from this document. The export is truncated or "
                "the section was hand-edited; the retained bytes are the repair"
            )
        rest = remainder


def _reasons(error: ValidationError) -> str:
    """A pydantic failure as one line naming each field and what was wrong with it."""
    parts: list[str] = []
    for detail in error.errors():
        location = ".".join(str(part) for part in detail["loc"]) or "manifest"
        parts.append(f"{location}: {detail['msg']}")
    return "; ".join(parts)


def _names(paths: Sequence[Path]) -> list[str]:
    return sorted(path.name for path in paths)


def _normalized(name: str) -> str:
    """A declared or observed filename, comparable across the two ways of writing one.

    Separators folded only. **No ``..`` resolution**, which matters: collapsing ``a/../b`` to
    ``b`` is exactly the normalization that makes a traversal compare equal to a legitimate name,
    and there is nothing to gain from it when both sides are bare filenames.
    """
    return str(PurePosixPath(name.replace("\\", "/")))


def _relative(path: Path, *, root: Path) -> str:
    """``path`` relative to ``root``, POSIX-separated.

    Relative so a citation reproduces where the corpus lives elsewhere, and does not publish this
    machine's directory layout; POSIX so the stored value does not depend on which platform
    ingested it.
    """
    try:
        return str(PurePosixPath(path.relative_to(root)))
    except ValueError:  # pragma: no cover - the walk cannot leave the root
        return path.name


__all__ = [
    "ANCESTOR_IDS",
    "ATTACHMENTS",
    "CONTENT_STATUS",
    "LABELS",
    "MANIFEST_NAME",
    "MAX_MANIFEST_BYTES",
    "UNIDENTIFIED_PREFIX",
    "UNINTERPRETED_MACROS",
    "UNRECOVERABLE_MACRO_BODY",
    "ConfluenceSnapshotConnector",
]
