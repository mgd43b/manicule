"""A local directory as a source.

The connector behind ``manicule index <path>``, and the smallest complete implementation of
:class:`~manicule.core.protocols.Connector`: discover walks the tree, fetch reads the bytes,
reconcile walks it again and says what still exists.

Three decisions are worth stating because each is a trap the design already knows about.

**Identity is the resolved absolute path.** ``document_id`` is a digest of
``(workspace, source, source_id)``, so a source id that varied with the directory somebody
happened to be standing in would re-index the same file as a new document every time. The
root passed on the command line is a place to start walking, never part of an identity.

**The change token is size and modification time, not a hash.** Discovery has to be decidable
without reading the file, which is the difference between a sync that costs what changed and
one that costs the corpus. Content-hash dedup downstream catches the case where the token
moved and the bytes did not — a ``touch``, or a checkout that rewrites every file.

**Media type comes from a table written down here.** :func:`mimetypes.guess_type` consults
``/etc/mime.types`` and the Windows registry, so the same file would be routed to different
parsers on two machines and therefore chunked two different ways. The platform may change how
fast this runs; it may not change what ends up in the index.

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
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

from manicule.connectors import sidecar
from manicule.connectors.errors import NotFoundError
from manicule.core.content import RawDocument
from manicule.core.ids import content_hash
from manicule.core.sources import DiscoveredDoc, DocRef, Watermark
from manicule.parsers.grammars import MEDIA_TYPES as GRAMMAR_MEDIA_TYPES

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Mapping, Sequence

    from manicule.core.sources import SourceId

OCTET_STREAM: Final = "application/octet-stream"
"""What an unrecognised file is. Refused later by the parser chain, and visibly so."""

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


def version_token(path: Path) -> str | None:
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
    return "+".join(stamps)


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
    ) -> None:
        """Point the connector at a root.

        Args:
            root: A file or a directory. Resolved once, so a relative path and a symlink to
                the same tree produce the same document ids.
            name: The source name. Part of every document's identity.
            include_hidden: Whether to walk dot-files and dot-directories.
            max_bytes: Refuse a file larger than this at discovery, before it is read.
        """
        self.name = name
        self._root = root.expanduser().resolve()
        self._include_hidden = include_hidden
        self._max_bytes = max_bytes
        self._reached: datetime | None = None

    @property
    def root(self) -> Path:
        return self._root

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
        for path in self._walk():
            token = version_token(path)
            size = path.stat().st_size if token else None
            if self._max_bytes is not None and size is not None and size > self._max_bytes:
                continue
            yield DiscoveredDoc(
                ref=DocRef(source_id=str(path), uri=path.as_uri()),
                version_token=token,
                title=path.name,
                media_type=media_type_for(path),
                size_bytes=size,
            )
        # Only after the walk has run to the end. A watermark stored for a partial enumeration
        # is how documents go missing permanently.
        self._reached = started

    async def fetch(self, ref: DocRef) -> RawDocument:
        """Read one file.

        Raises:
            NotFoundError: The file is gone, or is outside the root this connector serves.
                Both are refusals rather than reads: a source id that escapes the root would
                let a stored document address any file the process can open.
        """
        path = Path(ref.source_id)
        if not self._within_root(path):
            msg = (
                f"{ref.source_id!r} is outside {self._root}, which is the only tree this "
                f"source serves"
            )
            raise NotFoundError(msg)
        try:
            content = await asyncio.to_thread(path.read_bytes)
        except OSError as exc:
            msg = f"cannot read {ref.source_id}: {exc}"
            raise NotFoundError(msg) from exc
        # Read here rather than at discovery, because building the record needs the digest of
        # the bytes to check a declared checksum against — and discovery must stay decidable
        # without reading a file. `_within_root` above has already refused anything outside the
        # tree, so the path handed to the reader is one this connector was willing to open.
        provenance = await asyncio.to_thread(
            sidecar.provenance_for, path, root=self._root, checksum=content_hash(content)
        )
        return RawDocument(
            source_id=ref.source_id,
            uri=ref.uri,
            media_type=media_type_for(path),
            content=content,
            metadata=sidecar.with_provenance({}, provenance),
        )

    async def reconcile(self) -> AsyncIterator[SourceId]:
        """Yield the id of every file that still exists.

        The pass incremental sync cannot do. A deleted file simply stops appearing, so without
        this the index serves it forever and no amount of syncing fixes it.
        """
        for path in self._walk():
            yield str(path)

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


__all__ = [
    "IGNORED_DIRECTORIES",
    "OCTET_STREAM",
    "FilesystemConnector",
    "media_type_for",
    "version_token",
]
