"""A local directory as a source.

The connector behind ``manicule index <path>``, and the smallest complete implementation of
:class:`~manicule.core.protocols.Connector`: discover walks the tree, fetch reads the bytes,
reconcile walks it again and says what still exists.

Three decisions are worth stating because each is a trap the design already knows about.

**Identity is the resolved absolute path, unless the document says otherwise.** ``document_id``
is a digest of ``(workspace, source, source_id)``, so a source id that varied with the directory
somebody happened to be standing in would re-index the same file as a new document every time.
The root passed on the command line is a place to start walking, never part of an identity.

Where a :mod:`.sidecar` manifest declares a ``source_id``, **that** is the identity instead, and
the path becomes merely where the copy sits. This is the difference between a mirror and a
directory: a mirrored page reorganized from by-space to by-tree has not become a different page,
but a connector keyed on the path reports every document deleted and every document new — and the
curated collections and tags hanging off the old rows go with them. It is the rule
:class:`~manicule.connectors.confluence_snapshot.ConfluenceSnapshotConnector` was written with,
reached here through a manifest rather than a page directory. Two consequences land here: the
walked path travels in :data:`SNAPSHOT_PATH` because ``fetch`` still needs somewhere to read
from, and two files declaring one identity are **both** returned to the path, because silently
choosing one is how half an export disappears.

**The change token is size and modification time, not a hash.** Discovery has to be decidable
without reading the file, which is the difference between a sync that costs what changed and
one that costs the corpus. Content-hash dedup downstream catches the case where the token
moved and the bytes did not — a ``touch``, or a checkout that rewrites every file.

**Media type comes from a table written down here.** :func:`mimetypes.guess_type` consults
``/etc/mime.types`` and the Windows registry, so the same file would be routed to different
parsers on two machines and therefore chunked two different ways. The platform may change how
fast this runs; it may not change what ends up in the index.

With one exception, and it is stated rather than hidden: **an HTML file's media type is declared
at fetch, not at discovery.** Whether a ``.html`` is an ordinary page or an enriched export
carrying a storage-format body inside it (:mod:`.enriched`) is a fact about its contents, and
discovery does not read contents. So discovery says *nothing* about routing for HTML —
``DiscoveredDoc.media_type`` is ``None``, which
:meth:`~manicule.ingest.pipeline.IngestPipeline._routing_is_current` already reads as "this
connector has not made a claim" rather than as a change. Every other suffix keeps its declaration
and keeps the routing check that goes with it. Guessing instead would be worse in both
directions: declare ``text/html`` and every adapted page re-ingests on every sync because the
stored type disagrees; declare the storage type and every ordinary page does.

**A file may say what it is a copy of.** A mirrored page stored as ``123456.html`` cites as
``123456.html`` unless something tells the connector otherwise, and :mod:`.sidecar` is that
something: an adjacent manifest supplies the document's real title, canonical address, source
identity and version, and the connector attaches it to the fetched bytes. Three consequences
land here rather than there — a manifest is walked over instead of being indexed as a document
of its own, the change token covers the pair so that editing a manifest is a change to the
document it describes, and the record is built at fetch because checking a declared checksum
needs the bytes. Nothing about it is specific to any documentation product, and a directory with
no manifests behaves exactly as it did before it existed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final

from manicule.connectors import sidecar
from manicule.connectors.enriched import (
    ADAPTER_VERSION,
    DEFAULT_PROFILE,
    ENRICHED_KEY,
    MAX_HTML_BYTES,
    Adaptation,
    AdapterOutcome,
    EnrichedProfile,
    UnusablePageError,
    adapt,
)
from manicule.connectors.errors import NotFoundError
from manicule.core.content import Metadata, RawDocument
from manicule.core.ids import content_hash
from manicule.core.provenance import LocalSnapshot, Provenance
from manicule.core.sources import DiscoveredDoc, DocRef, Watermark
from manicule.parsers.grammars import MEDIA_TYPES as GRAMMAR_MEDIA_TYPES

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Mapping, Sequence

    from manicule.core.sources import SourceId

OCTET_STREAM: Final = "application/octet-stream"
"""What an unrecognized file is. Refused later by the parser chain, and visibly so."""

ADAPTABLE_SUFFIXES: Final[frozenset[str]] = frozenset({".htm", ".html"})
"""Suffixes that may turn out to be an enriched export rather than an ordinary page.

The set that decides two things at once, which is why it is one constant: which files have their
media type left undeclared at discovery, and which files the adapter is offered at fetch. Two
sets would drift, and the drift is invisible — a suffix in one and not the other is either a
corpus that re-ingests itself for ever or a page that is never adapted.
"""

SNAPSHOT_PATH: Final = "snapshot_path"
"""``DocRef.metadata`` key carrying the file this document was walked to.

Needed because ``source_id`` stops being a path the moment a manifest declares an identity, and
``fetch`` still has to open something. The same shape
:data:`~manicule.connectors.confluence_snapshot._DIRECTORY` uses, for the same reason: the
location is what a fetch needs and is nothing an identity should carry.
"""

DUPLICATE_IDENTITY: Final = "duplicate_source_identity"
"""``DocRef`` and document-metadata key naming the other files that claimed one identity.

Present exactly when a declared identity was refused for being claimed twice. The refusal is the
point: ``documents`` is UNIQUE on ``(workspace_id, source, source_id)``, so honoring both would
mean the second file's ingest silently overwriting the first's document — two pages in the
corpus, one row, and nothing raised.
"""

_LANGUAGE_BY_SUFFIX: Final[Mapping[str, str]] = {
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".css": "css",
    ".dart": "dart",
    ".ex": "elixir",
    ".exs": "elixir",
    ".go": "go",
    ".h": "c",
    ".hpp": "cpp",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".kt": "kotlin",
    ".lua": "lua",
    ".php": "php",
    ".pl": "perl",
    ".py": "python",
    ".pyi": "python",
    ".rb": "ruby",
    ".rs": "rust",
    ".scala": "scala",
    ".sh": "bash",
    ".sql": "sql",
    ".swift": "swift",
    ".ts": "typescript",
    ".tsx": "tsx",
}
"""Suffix to tree-sitter language.

Kept apart from the media types so the two cannot disagree: the media type of a language is
whatever :data:`manicule.parsers.grammars.MEDIA_TYPES` says, and a language this table names
that the grammar table does not know is simply not routed to the source-code parser.
"""

_MEDIA_TYPE_BY_SUFFIX: Final[Mapping[str, str]] = {
    ".adoc": "text/plain",
    ".csv": "text/csv",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".eml": "message/rfc822",
    ".htm": "text/html",
    ".html": "text/html",
    ".ipynb": "application/x-ipynb+json",
    ".json": "application/json",
    ".log": "text/plain",
    ".md": "text/markdown",
    ".mdx": "text/mdx",
    ".msg": "application/vnd.ms-outlook",
    ".pdf": "application/pdf",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".rst": "text/plain",
    ".text": "text/plain",
    ".toml": "application/toml",
    ".txt": "text/plain",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xml": "application/xml",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".zip": "application/zip",
}

IGNORED_DIRECTORIES: Final[frozenset[str]] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".idea",
        ".DS_Store",
    }
)
"""Directories a walk never descends into.

Tool output and version-control internals are not documents, and a repository's ``.git``
directory is larger than the repository. Indexing it produces thousands of documents nobody
asked about and buries the ones they did.
"""


def media_type_for(path: Path) -> str:
    """The media type a path implies. Never consults the machine's mime database."""
    suffix = path.suffix.lower()
    known = _MEDIA_TYPE_BY_SUFFIX.get(suffix)
    if known is not None:
        return known
    language = _LANGUAGE_BY_SUFFIX.get(suffix)
    if language is not None:
        return GRAMMAR_MEDIA_TYPES.get(language, OCTET_STREAM)
    return OCTET_STREAM


def version_token(path: Path, *, adapter: str = "", location: str = "") -> str | None:
    """A change token from the file's own metadata, or ``None`` if it cannot be read.

    ``None`` is honest rather than convenient: it means "fetch and hash to find out", which is
    what a source with no change signal is entitled to ask for. Inventing a token that does
    not move would skip a changed file forever.

    **A document with a sidecar manifest has two files, and the token covers both.** This is not
    tidiness. A manifest is where the citable facts live, so a manifest edited to correct a
    title or record a new source version changes what a citation says while leaving the
    document's own bytes untouched. A token taken from the document alone would not move,
    ``_unchanged_by_token`` would skip before fetching, and the corrected manifest would never
    be read — a corpus citing a version it was told about and then declined to look at. Folding
    the sibling's size and modification time in makes an edit to either side a change to the
    pair, which is what a "document plus its manifest" actually is.

    Args:
        path: The document.
        adapter: What the derived content is a product of, for a file whose stored bytes are
            *extracted* from it rather than copied from it — the adapter's version and the
            profiles configured to run. Empty for every other file, and that emptiness is
            load-bearing: folding a constant into every token would make bumping the adapter
            re-ingest a corpus of PDFs to produce identical output. The bytes on disk are not the
            whole input to what gets stored, so they are not the whole of the change token
            either; this is ``parse_fp``'s argument (``docs/storage.md`` §6.4) one stage earlier,
            for the stage that runs before a parser is chosen.
        location: Where the file sits, for a document whose identity is *not* its location.
            Empty for every other file, and again the emptiness is the point: a path-keyed
            document cannot move without becoming a different one, so its location is already in
            its identity and repeating it would achieve nothing. A page-keyed document can move,
            and a rename preserves both size and modification time — so without this a mirror
            reorganized into new directories reports every page unchanged, skips before fetching,
            and leaves every stored snapshot path naming a file that is no longer there. The
            document would be correct in every respect a reader sees and wrong in the one an
            audit needs. Found by moving one.
    """
    stamps: list[str] = []
    for candidate in (path, sidecar.manifest_path_for(path)):
        try:
            stat = candidate.stat()
        except OSError:
            # The document itself being unstattable means no token at all; a missing manifest is
            # the ordinary case and contributes nothing.
            if candidate == path:
                return None
            continue
        stamps.append(f"{stat.st_size}:{stat.st_mtime_ns}")
    return "+".join((*stamps, *(part for part in (adapter, location) if part)))


class FilesystemConnector:
    """One directory tree, or one file, as a source.

    Satisfies :class:`~manicule.core.protocols.Connector`.
    """

    def __init__(
        self,
        root: Path,
        *,
        name: str = "local",
        include_hidden: bool = False,
        max_bytes: int | None = None,
        profiles: Sequence[EnrichedProfile] = (DEFAULT_PROFILE,),
        configured: bool = False,
    ) -> None:
        """Point the connector at a root.

        Args:
            root: A file or a directory. Resolved once, so a relative path and a symlink to
                the same tree produce the same document ids.
            name: The source name. Part of every document's identity.
            include_hidden: Whether to walk dot-files and dot-directories.
            max_bytes: Refuse a file larger than this at discovery, before it is read.
            profiles: The enriched-document conventions to recognize, in precedence order. An
                empty sequence turns adaptation off entirely, which is a real and exercised
                configuration rather than a degenerate one: a corpus of ordinary HTML pays
                nothing for a feature it does not use, and turning it off is how an operator
                establishes that an unexpected parse is or is not the adapter's doing.
            configured: Whether :attr:`name` is a configured source instance — the key in
                ``[connectors.<name>]`` — rather than a label for a one-off walk. Set by
                :func:`~manicule.connectors.plugin.build_filesystem` and by nothing else,
                because it decides whether :attr:`sidecar_command` may tell an operator to run
                ``--source <name>``, and a name no configuration holds would be an instruction
                that fails the moment it is followed.
        """
        self.name = name
        self._root = root.expanduser().resolve()
        self._include_hidden = include_hidden
        self._max_bytes = max_bytes
        self._profiles = tuple(profiles)
        self._configured = configured
        # Folded into the change token of every adaptable file, so that a new adapter version or
        # a changed profile set rebuilds the derived body without the source being re-read from
        # anywhere but disk. Computed once because it cannot vary between files in one run.
        self._adapter = (
            "+".join((ADAPTER_VERSION, *(profile.name for profile in self._profiles)))
            if self._profiles
            else ""
        )
        self._reached: datetime | None = None

    @property
    def root(self) -> Path:
        return self._root

    @property
    def profiles(self) -> tuple[EnrichedProfile, ...]:
        """The enriched-document conventions this connector recognizes, in precedence order.

        Public for the same reason :attr:`root` is: it is part of what this source *is* rather
        than an implementation detail, and "which profiles is this connector actually using" is
        the first question when a page is not being adapted. A configured value that reached no
        connector is a defect this repository has had before (#94) and one nothing observes from
        outside unless it can be read back.
        """
        return self._profiles

    @property
    def sidecar_command(self) -> str:
        """The conversion command that will run with **this** connector's profiles.

        The whole of requirement 6, in one place, because remediation text is the thing this
        repository keeps getting wrong: the reason attached to an unapplied identity used to name
        ``manicule connector sidecar <root>`` unconditionally, and for a source with a custom
        profile that command reports ``no_profile`` for every page and writes nothing. An
        instruction that was written and never executed.

        The positional form is correct for the default profile and for nothing else, so that is
        exactly when it is emitted. Where the profiles are this source's own, the command has to
        be the one that resolves them from configuration — and that form needs a name
        configuration actually holds, which is what :attr:`_configured` records. A connector built
        for ``manicule index <path>`` has no configured name to offer and gets the root form,
        which is right for it because it also has the default profiles.

        The fallback is the root form rather than nothing. A connector with custom profiles and
        no configured name is not reachable through the product — ``build_filesystem`` is the only
        thing that supplies custom profiles and it always supplies the name too — but if it ever
        became reachable, naming a command that converts the right *directory* with the wrong
        profiles is a recoverable disappointment, and naming ``--source`` with a name nothing
        resolves is an error the operator cannot act on at all.
        """
        if self._configured and self._profiles != (DEFAULT_PROFILE,):
            return f"manicule connector sidecar --source {self.name}"
        return f"manicule connector sidecar {self._root}"

    @property
    def watermark(self) -> Watermark | None:
        """When the last **complete** walk finished.

        A filesystem has no change feed, so this records a time rather than a position and
        every walk is a full one. It is still worth storing: it is what a later reconciliation
        compares against, and it is set only after ``discover`` has run to the end, so an
        interrupted walk leaves the previous value in place.
        """
        if self._reached is None:
            return None
        return Watermark(value=self._reached.isoformat(), observed_at=self._reached)

    async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
        """Yield every file under the root.

        ``watermark`` is accepted and deliberately not used to skip files. A modification time
        older than the last walk does not mean the file is unchanged — an editor that
        preserves timestamps, a restored backup and a ``git checkout`` all move content
        without moving the clock forward. The per-document version token is what makes the
        sync incremental, and it is compared by the pipeline before anything is read.
        """
        del watermark  # see the docstring: the token does the skipping, not the clock
        started = datetime.now(UTC)
        for path, identity, conflict in self._identities():
            token = version_token(
                path,
                adapter=self._adapter_for(path),
                location="" if identity == str(path) else _relative(path, root=self._root),
            )
            size = path.stat().st_size if token else None
            if self._max_bytes is not None and size is not None and size > self._max_bytes:
                continue
            located: Metadata = {SNAPSHOT_PATH: str(path)}
            yield DiscoveredDoc(
                ref=DocRef(
                    source_id=identity,
                    uri=path.as_uri(),
                    metadata=located if not conflict else {**located, DUPLICATE_IDENTITY: conflict},
                ),
                version_token=token,
                title=path.name,
                # Nothing said about routing for a file that may turn out to be an enriched
                # export; see the module docstring. `None` is a claim this connector is entitled
                # to decline to make, and the pipeline reads it as agreement rather than change.
                media_type=None if _adaptable(path) else media_type_for(path),
                size_bytes=size,
            )
        # Only after the walk has run to the end. A watermark stored for a partial enumeration
        # is how documents go missing permanently.
        self._reached = started

    async def fetch(self, ref: DocRef) -> RawDocument:
        """Read one file, and hand on the storage body if it turns out to hold one.

        **The path comes from the ref's metadata, never from its identity.** They were the same
        string until a manifest could declare an identity, and reading a page id as a path would
        be a stored document addressing whatever ``1002`` resolves to from the working directory.
        The fallback to ``source_id`` is for a caller that built a ``DocRef`` by hand — ``manicule
        index`` on a single file, and the tests — and is bounded by the same containment check.

        Raises:
            NotFoundError: The file is gone, or is outside the root this connector serves.
                Both are refusals rather than reads: a source id that escapes the root would
                let a stored document address any file the process can open.
        """
        path = Path(str(ref.metadata.get(SNAPSHOT_PATH) or ref.source_id))
        if not self._within_root(path):
            msg = (
                f"{str(path)!r} is outside {self._root}, which is the only tree this source serves"
            )
            raise NotFoundError(msg)
        try:
            content = await asyncio.to_thread(path.read_bytes)
        except OSError as exc:
            msg = f"cannot read {path}: {exc}"
            raise NotFoundError(msg) from exc
        # Read here rather than at discovery, because building the record needs the digest of
        # the bytes to check a declared checksum against — and discovery must stay decidable
        # without reading a file. `_within_root` above has already refused anything outside the
        # tree, so the path handed to the reader is one this connector was willing to open.
        provenance = await asyncio.to_thread(
            sidecar.provenance_for, path, root=self._root, checksum=content_hash(content)
        )
        metadata = sidecar.with_provenance({}, provenance)
        conflict = ref.metadata.get(DUPLICATE_IDENTITY)
        if conflict:
            metadata = {**metadata, DUPLICATE_IDENTITY: conflict}

        adapted = await asyncio.to_thread(self._adapt, path, content)
        if not isinstance(adapted, Adaptation):
            return RawDocument(
                source_id=ref.source_id,
                uri=ref.uri,
                media_type=media_type_for(path),
                content=content,
                metadata=metadata
                if adapted is None
                else {**metadata, ENRICHED_KEY: adapted.stated},
            )
        return RawDocument(
            source_id=ref.source_id,
            # The **extracted body**, not the file. The enriched wrapper stays on disk untouched
            # and is the immutable snapshot; what the index is built from is this. So
            # `documents.content_hash` digests the body, and the digest of the whole file is
            # recorded under `ENRICHED_KEY` because nothing else would hold it.
            content=adapted.body.encode("utf-8"),
            media_type=adapted.representation,
            uri=ref.uri,
            metadata=self._enriched_metadata(metadata, adapted, path=path, provenance=provenance),
        )

    async def reconcile(self) -> AsyncIterator[SourceId]:
        """Yield the id of every file that still exists.

        The pass incremental sync cannot do. A deleted file simply stops appearing, so without
        this the index serves it forever and no amount of syncing fixes it.

        Shares :meth:`_identities` with ``discover``, so the two cannot disagree about what a
        document is called — one yielding a path where the other yielded a page id would report
        every mirrored page deleted on every sync, and then re-index it.
        """
        for _, identity, _ in self._identities():
            yield identity

    # --- enriched documents -------------------------------------------------------------------

    def _adapt(self, path: Path, content: bytes) -> Adaptation | _Refused | None:
        """What the adapter made of ``content``: a body, a stated refusal, or nothing to say.

        Three answers rather than two, and the third is the reason this is not a ``| None``.
        ``None`` is "this is an ordinary document" — the wrong suffix, adaptation off, or no
        profile engaging — which is the answer for every HTML page in an ordinary corpus and is
        not worth recording anywhere. :class:`_Refused` is "this looked like an enriched page and
        could not be read as one", which is worth recording on the document, because the
        difference between those two is the whole of whether an operator has a problem.

        **The input is bounded here as well as in the conversion**, and the omission was real
        rather than theoretical. ``write_sidecars`` reads through :data:`~.enriched.MAX_HTML_BYTES`
        and this path did not, so a pathological ``.html`` in a corpus was handed whole to an HTML
        engine — a parse that did not happen before this feature existed, on top of the one the
        web parser was already going to do. ``fetch`` reading the file unbounded is older than
        this branch and is capped downstream by ``ingest.max_fetch_bytes``; what is new is the
        *parse*, so the parse is what this refuses. Over the limit the file falls through to
        ordinary HTML ingestion, exactly as it did before, with the reason recorded.

        A refusal does not fail the document. The enriched wrapper is a perfectly good HTML file
        and is indexed as one, exactly as it was before this existed — with the reason attached
        rather than a partial success nobody can see.
        """
        if not self._profiles or not _adaptable(path):
            return None
        if len(content) > MAX_HTML_BYTES:
            return _Refused(
                outcome=AdapterOutcome.FAILED.value,
                reason=(
                    f"is {len(content)} bytes, over the {MAX_HTML_BYTES}-byte limit for reading a "
                    f"page as an enriched export. It is indexed as ordinary HTML instead; raise "
                    f"the limit or split the export if it is meant to be one page"
                ),
            )
        try:
            return adapt(content.decode("utf-8", errors="replace"), profiles=self._profiles)
        except UnusablePageError as refusal:
            if refusal.outcome is AdapterOutcome.NO_PROFILE:
                return None
            return _Refused(outcome=refusal.outcome.value, reason=str(refusal))

    def _enriched_metadata(
        self,
        metadata: Metadata,
        adapted: Adaptation,
        *,
        path: Path,
        provenance: Provenance | None,
    ) -> Metadata:
        """``metadata`` with the adaptation recorded, and with the page's own identity attached.

        Two things happen here and they are separate. The **record** under :data:`ENRICHED_KEY`
        says what was extracted from what, by which profile, at which adapter version.

        The **provenance** is upgraded from what a manifest said to what the page itself says,
        and only where there was nothing already. A manifest is a deliberate act by whoever ran
        the conversion and outranks a document's own claim about itself; but a page with no
        manifest has been citing its filename while stating its title, address and version inside
        itself the whole time, and there is no reason for a corpus to keep both facts and use the
        weaker one. ``title`` is written into the metadata as well because that is where
        :func:`~manicule.parsers.confluence._title` looks: a storage body is a fragment with no
        ``<title>`` in it, so a parser handed one and told nothing has no title to anchor the
        content above the first heading to.

        **Outranking is not the same as overwriting, and the title is where the difference
        showed.** A manifest outranks the page about what it *states*; a field it leaves empty is
        not a statement, and reading it as one meant a hand-written minimal manifest — a
        ``source_id`` and little else, which is exactly what somebody writes by hand — silently
        cost the document its title. Nothing downstream could recover it: the extracted body is a
        fragment with no ``<title>``, so the storage parser looks at ``metadata["title"]``, finds
        the key absent because an empty string is dropped below, and anchors nothing. The
        fallback is per-field for that reason rather than "use the page when there is no
        manifest at all" — the manifest is still authoritative about every field it fills in.

        **Only ``title``, and the other empty fields are deliberately left empty.**
        :class:`~manicule.core.provenance.SourceMetadata` defaults every field to ``""``, so the
        same reasoning appears to apply to ``canonical_uri`` and ``source_id`` — and applying it
        there would be a different act entirely. ``title`` is what a citation displays;
        ``source_id`` is what the corpus keys rows by, and filling a manifest's empty one in from
        the page would move documents between identities from inside a metadata helper. The
        narrowness is the safety, not a lack of ambition.

        **The record is repaired as well as the metadata key, because they are read by different
        things and only fixing one leaves the defect visible.**
        :meth:`~manicule.ingest.pipeline.IngestPipeline._store_record` takes a document's title
        from the provenance record first and falls back to what discovery reported, which for a
        local file is its *filename*; the parser separately reads ``metadata["title"]`` to anchor
        the content above the first heading. So an empty manifest title cost the citation its name
        *and* the parse its anchor, by two different routes, and writing the fallback into only
        one of them would have fixed the half nobody looks at.
        """
        record = provenance
        if record is None:
            record = Provenance(
                source=adapted.page.source,
                snapshot=LocalSnapshot(
                    path=_relative(path, root=self._root), retrieved_at=adapted.page.retrieved_at
                ),
            )
            metadata = sidecar.with_provenance(metadata, record)
        title = (record.source.title if record.source is not None else "") or (
            adapted.page.source.title
        )
        if record.source is not None and not record.source.title and title:
            record = record.model_copy(
                update={"source": record.source.model_copy(update={"title": title})}
            )
            metadata = sidecar.with_provenance(metadata, record)
        adaptation: Metadata = {
            "outcome": AdapterOutcome.ADAPTED.value,
            "profile": adapted.profile.name,
            "adapter_version": ADAPTER_VERSION,
            "representation": adapted.representation,
            "snapshot_path": _relative(path, root=self._root),
            "snapshot_checksum": adapted.snapshot_checksum,
            "body_checksum": adapted.body_checksum,
        }
        if not sidecar.declared_identity(path):
            # Adapted, cited correctly, and keyed on where it sits — because identity has to be
            # known at discovery and discovery does not read documents. Stated on the document
            # rather than left to be discovered when the page is moved and the corpus has two of
            # it, and it names the step that applies the identity rather than describing the gap.
            adaptation["outcome"] = AdapterOutcome.IDENTITY_NOT_APPLIED.value
            adaptation["reason"] = (
                f"this page declares source_id {adapted.page.source.source_id!r} and is indexed "
                f"under its path, because a document's identity has to be known before it is "
                f"fetched and discovery does not read files. Everything else about it is "
                f"correct — it is parsed as storage format and cited by its own title and "
                f"address — but moving or renaming it will create a second document. Run "
                f"`{self.sidecar_command}` to write the manifest that applies "
                f"the declared identity, then sync. The page is then indexed under its page id "
                f"and this path-keyed copy is superseded; `manicule doctor` names it so it can "
                f"be removed."
            )
        return {
            **metadata,
            **({"title": title} if title else {}),
            ENRICHED_KEY: adaptation,
        }

    def _adapter_for(self, path: Path) -> str:
        """What this file's derived content is a product of, or ``""`` if it has none."""
        return self._adapter if _adaptable(path) else ""

    # --- identity ------------------------------------------------------------------------------

    def _identities(self) -> Iterator[tuple[Path, SourceId, str]]:
        """Every file under the root, with the identity it is stored under and any conflict.

        Two passes over one materialized walk rather than one streaming pass, because a duplicate
        cannot be detected until the second file claiming an identity has been seen — and by then
        a streaming ``discover`` has already yielded the first under the identity it is about to
        lose. Yielding it and correcting later is not available: the pipeline has stored a
        document by then.

        The cost is bounded by the corpus rather than by the number of mirrored pages, and the
        map is not: only files whose manifest declares an identity enter it, so a directory of
        ordinary documents allocates nothing beyond the path list a walk produces anyway.
        """
        declared: list[tuple[Path, str]] = [
            (path, sidecar.declared_identity(path)) for path in self._walk()
        ]
        claims: dict[SourceId, list[Path]] = {}
        for path, identity in declared:
            if identity:
                claims.setdefault(identity, []).append(path)
        for path, identity in declared:
            others = claims.get(identity, ()) if identity else ()
            if not identity or len(others) > 1:
                # Both files fall back, not just the second. Keeping the first would make which
                # page owns the identity depend on walk order, so a directory renamed from `a/`
                # to `z/` would silently move a document's content to a different page's id.
                yield path, str(path), _conflict(identity, others, path=path, root=self._root)
                continue
            yield path, identity, ""

    # --- walking ----------------------------------------------------------------------------

    def _walk(self) -> Iterator[Path]:
        """Every file under the root, in a stable order.

        Sorted at each level rather than left to the filesystem, so two machines indexing the
        same tree ingest it in the same order — which is what makes a ``--limit`` mean the
        same thing twice.
        """
        if self._root.is_file():
            yield self._root
            return
        yield from self._walk_directory(self._root)

    def _walk_directory(self, directory: Path) -> Iterator[Path]:
        try:
            entries: Sequence[Path] = sorted(directory.iterdir())
        except OSError:
            return
        for entry in entries:
            if not self._include_hidden and entry.name.startswith("."):
                continue
            if entry.is_symlink():
                # Followed nowhere. A symlink out of the tree is the same escape `fetch`
                # refuses, and a symlink loop inside it is an infinite walk.
                continue
            if entry.is_dir():
                if entry.name in IGNORED_DIRECTORIES:
                    continue
                yield from self._walk_directory(entry)
            elif entry.is_file():
                if sidecar.is_manifest(entry):
                    # A manifest is metadata about the document beside it, not a document. Walked
                    # over here rather than filtered at ingest so that `discover` and `reconcile`
                    # agree: a manifest yielded by one and not the other would be reported as a
                    # deletion on every sync, or indexed as a contentless entry that cites
                    # nothing and dilutes every result set it appears in.
                    continue
                yield entry

    def _within_root(self, path: Path) -> bool:
        try:
            resolved = path.resolve()
        except OSError:  # pragma: no cover - resolution failure is itself a refusal
            return False
        return resolved == self._root or self._root in resolved.parents


@dataclass(frozen=True, slots=True)
class _Refused:
    """A file that engaged a profile and could not be read as an enriched page."""

    outcome: str
    reason: str

    @property
    def stated(self) -> Metadata:
        """The refusal as it is stored, shaped like the record a success would have written.

        One key holding either outcome, so a query for "what did the adapter do with this
        document" is one lookup rather than two, and a document cannot end up carrying both.
        """
        return {"outcome": self.outcome, "reason": self.reason}


def _adaptable(path: Path) -> bool:
    """Whether this file could turn out to be an enriched export.

    A suffix test and nothing more. Deciding by contents is what ``fetch`` does; deciding it here
    as well would be the file read that discovery must not do.
    """
    return path.suffix.lower() in ADAPTABLE_SUFFIXES


def _conflict(identity: str, others: Sequence[Path], *, path: Path, root: Path) -> str:
    """Why ``path`` kept its path identity, or ``""`` when it simply never declared one.

    An empty string for the ordinary case rather than a sentence saying nothing happened: a
    document with no manifest is not a document with a problem, and a reason attached to every
    file in a corpus is a reason nobody reads.
    """
    if not identity or len(others) <= 1:
        return ""
    named = ", ".join(sorted(_relative(other, root=root) for other in others if other != path))
    return (
        f"declares source_id {identity!r}, which {named} also declares. Two files cannot be one "
        f"document — documents is UNIQUE on (workspace, source, source_id), so honoring both "
        f"would mean whichever synced second silently overwriting the first. Both are indexed "
        f"under their paths until one of the manifests is corrected"
    )


def _relative(path: Path, *, root: Path) -> str:
    """``path`` relative to ``root``, POSIX-separated.

    Relative so that a diagnostic reproduces on a machine whose corpus lives elsewhere, and does
    not publish this one's directory layout.
    """
    try:
        return str(PurePosixPath(path.relative_to(root)))
    except ValueError:  # pragma: no cover - the walk cannot leave the root
        return path.name


__all__ = [
    "ADAPTABLE_SUFFIXES",
    "DUPLICATE_IDENTITY",
    "IGNORED_DIRECTORIES",
    "OCTET_STREAM",
    "SNAPSHOT_PATH",
    "FilesystemConnector",
    "media_type_for",
    "version_token",
]
