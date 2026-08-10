"""Zip archives, whose members are documents rather than chunks.

A member of an archive is a document in its own right: a PDF inside a zip is a PDF, and it
gets its own parser, its own chunks and its own anchors exactly as if it had been fetched
directly (``docs/parsing.md`` §9.1). So **the container emits no blocks at all** — not even a
manifest, because a chunk listing filenames is retrieval noise competing with the real content
inside it. Everything this parser produces arrives through
:meth:`ArchiveParser.expand`.

**One level, never a recursion.** Members are yielded for the pipeline to queue, which is what
makes the traversal breadth-first: a wide archive cannot starve a batch by descending one
branch to the bottom first (§9.2). The whole-tree limits therefore travel with each member in
metadata rather than living in this object, so the answer does not depend on how many archives
this parser has already seen.

**Four limits, because any one of them is bypassable** (§9.3): total uncompressed bytes across
the tree, per-member compression ratio, member count across the tree, and per-member
uncompressed bytes. The important one is how the first is enforced.

    The total-bytes limit is counted while streaming, never read from the header.

``ZipInfo.file_size`` is a field inside the archive, which is to say it is attacker-controlled,
and it is used here only as a cheap pre-filter. It is not even allowed to bound the read:
:mod:`zipfile` truncates a member's decompressed stream at whatever size the header declares,
so a bomb that declares a kilobyte and expands to megabytes comes back either truncated or as
``Bad CRC-32`` — a defence that depends on an implementation detail of one interpreter and
reports an attack as a corrupt file. The reader below sets its **own** ceiling instead and
counts every byte it takes, so the thing that stops a bomb is the counter, and the header
certifies nothing.

**The trap this parser exists to avoid** is that ``.docx``, ``.xlsx``, ``.pptx`` and ``.epub``
are all zip containers (§9.4). A sniffer looking for ``PK\\x03\\x04`` identifies every Office
document as an archive, so this parser declares its media types narrowly *and* refuses any zip
whose directory looks like a document container — which is what covers the realistic version
of the bug, where the extension is wrong and the declared type is
``application/octet-stream``.
"""

from __future__ import annotations

import copy
import io
import zipfile
from collections.abc import AsyncIterator

from manicule.core.anchors import Anchor
from manicule.core.content import DocumentStatus, Metadata, ParsedBlock, RawDocument
from manicule.core.errors import ParseError
from manicule.core.ids import content_hash
from manicule.parsers.base import ParserProfile
from manicule.parsers.config import ARCHIVE_MEDIA_TYPES, ArchiveConfig
from manicule.parsers.expansion import (
    CONTAINER_DEPTH,
    PATH_HASHES,
    TREE_BYTES,
    TREE_MEMBERS,
    ExpandedMember,
    MemberFailure,
    MemberOutcome,
    container_depth_of,
    inner_path,
    media_type_for,
    member_source_id,
    member_uri,
    path_hashes_of,
    tree_bytes_of,
    tree_members_of,
)

__all__ = [
    "ARCHIVE_MEDIA_TYPES",
    "ARCHIVE_SCHEME",
    "NOT_AN_ARCHIVE",
    "ArchiveConfig",
    "ArchiveParser",
]

ARCHIVE_SCHEME = "zip"
"""The scheme a member address carries: ``zip:<container-uri>!/reports/2026-q1.pdf``."""

NOT_AN_ARCHIVE = "OOXML container, not an archive"
"""Why a zip that is really a document is declined. Named because tests assert it and
``doctor`` reports it: this refusal is the difference between citing a spreadsheet and citing
``xl/worksheets/sheet1.xml``."""

_OOXML_DIRECTORY = "[Content_Types].xml"
_DOCUMENT_MIMETYPE = "mimetype"

_READ_CHUNK = 64 * 1024
"""How much of a member is decompressed per call. Small enough that a bomb is stopped after
one chunk past the budget rather than after materialising the whole thing."""

_NO_DECLARED_CEILING = 1 << 62
"""What the reader tells :mod:`zipfile` a member's size is.

Deliberately larger than any real member. :mod:`zipfile` truncates a member's output at the
declared size, so leaving the archive's own number in place would make an attacker-controlled
field the thing that bounds our memory — and would surface an oversized member as a CRC
failure rather than as the limit it actually hit. The ceiling that matters is applied by the
counter below, which is ours.
"""

_SYMLINK_MODE = 0o120000
_UNIX_FILE_TYPE = 0o170000
_ENCRYPTED_FLAG = 0x1


class _RefusalError(Exception):
    """A limit a member hit, carrying whether the rest of the archive can continue.

    An exception rather than a return value because it is raised from inside the read loop,
    several frames below the decision about what to do with it.
    """

    def __init__(self, reason: str, *, fatal: bool = False) -> None:
        """Record the reason, and whether the archive itself is finished.

        Args:
            reason: What was exceeded, in the words a diagnostic will show.
            fatal: ``True`` when a whole-tree budget is exhausted, so every remaining member
                would hit the same wall. Exceeding a limit fails that member, or that
                archive, and never the batch (``docs/parsing.md`` §9.3).
        """
        super().__init__(reason)
        self.reason = reason
        self.fatal = fatal


class ArchiveParser:
    """Expands a zip into the documents inside it, and emits no blocks of its own."""

    media_types = ARCHIVE_MEDIA_TYPES
    profile = ParserProfile(name="archive", max_unlocated_ratio=0.0, max_pagelevel_ratio=None)
    """Present because the registry and the harness both expect a profile, and never fed to
    :func:`~manicule.testing.assert_location_budget`. Both ratios would be zero over zero: this
    parser emits no blocks, so it has no locations to budget, which is exactly why
    ``docs/parsing.md`` §3.4 has no row for it."""

    def __init__(self, config: ArchiveConfig) -> None:
        self._config = config

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        """Yield nothing, having checked that this really is an archive.

        Zero blocks is the whole point: the container's status is ``container``, not
        ``no_extractable_text``, and its content is the documents :meth:`expand` produces.

        Raises:
            ParseError: The bytes are not a zip, or are a document container that merely
                happens to be one (§9.4). Declining lets the parser that should have had it
                take over.
        """
        with self._opened(raw):
            pass
        return
        yield  # pragma: no cover - unreachable, present to make this an async generator

    async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
        """Always ``None``: this parser emits no anchors, so it addresses no text.

        The members' anchors belong to the parsers that read the members, and resolving one
        here would mean this parser claiming a location it did not produce.
        """
        del anchor, raw
        return None

    async def expand(self, raw: RawDocument) -> AsyncIterator[MemberOutcome]:
        """Yield one outcome per member of this archive, in the archive's own order.

        Directory entries are skipped and produce nothing: a directory is not a document, and
        reporting it as a failed one would fill diagnostics with entries nobody can act on.
        Everything that could have been a document produces either a member or a reason.

        The archive is held open by a ``with`` that **encloses every** ``yield``. A consumer
        that stops after one member throws ``GeneratorExit`` in at the suspension point, and
        only a ``finally`` runs after that — so a close placed after the loop would leave a
        ``ZipFile`` and its decompression stream suspended, to be finalised later from a loop
        that may already have gone. That is a crash inside the allocator rather than a warning.

        Raises:
            ParseError: The bytes are not a zip, or are a document container (§9.4).
        """
        depth = container_depth_of(raw) + 1
        hashes = path_hashes_of(raw)
        used_bytes = tree_bytes_of(raw)
        used_members = tree_members_of(raw)

        seen: set[str] = set()

        with self._opened(raw) as archive:
            for ordinal, info in enumerate(archive.infolist()):
                if info.is_dir():
                    continue
                # Counted before the name is judged, because a rejected member still becomes a
                # stored document with a reason. Counting only the readable ones would let an
                # archive of ten thousand hostile names spend no budget at all.
                used_members += 1
                if used_members > self._config.max_members:
                    # Checked here rather than only inside `_member`, which the refusal branch
                    # below never reaches: an archive whose members *all* have hostile names
                    # would otherwise spend the budget the comment above says it spends and
                    # never be stopped by it.
                    yield _refusal(
                        raw,
                        info.filename,
                        depth,
                        f"archive member count exceeded: {used_members} members across this "
                        f"container tree, and the limit is {self._config.max_members}. Split "
                        f"the archive, or raise maxMembers.",
                        DocumentStatus.FAILED,
                        address=_unique(inner_path(info.filename) or "member", ordinal, seen),
                    )
                    return
                path = inner_path(info.filename)
                if path is None:
                    yield _refusal(
                        raw,
                        info.filename,
                        depth,
                        f"archive member name escapes the archive root: "
                        f"{info.filename!r}. Names are normalised and never rewritten into a "
                        f"plausible one, so this member is reported rather than extracted.",
                        DocumentStatus.FAILED,
                        # Unique by position in the central directory. A constant placeholder
                        # would give every escaping member one `source_id`, which storage
                        # reconciles by — so twenty rejected names would collapse into one
                        # document and nineteen refusals would vanish.
                        address=_unique("member", ordinal, seen),
                    )
                    continue
                path = _unique(path, ordinal, seen)
                outcome, stop = self._member(
                    raw,
                    archive,
                    info,
                    path=path,
                    depth=depth,
                    hashes=hashes,
                    used_bytes=used_bytes,
                    used_members=used_members,
                )
                if isinstance(outcome, ExpandedMember):
                    used_bytes += len(outcome.raw.as_bytes())
                yield outcome
                if stop:
                    return

    # --- one member ----------------------------------------------------------------------

    def _member(  # noqa: PLR0911 - one branch per refusal, each with its own reason
        self,
        raw: RawDocument,
        archive: zipfile.ZipFile,
        info: zipfile.ZipInfo,
        *,
        path: str,
        depth: int,
        hashes: tuple[str, ...],
        used_bytes: int,
        used_members: int,
    ) -> tuple[MemberOutcome, bool]:
        """What one member becomes, and whether the archive can continue after it."""
        if used_members > self._config.max_members:
            reason = (
                f"archive member count exceeded: {used_members} members across this container "
                f"tree, and the limit is {self._config.max_members}. Split the archive, or "
                f"raise maxMembers."
            )
            return _refusal(raw, path, depth, reason, DocumentStatus.FAILED), True
        if _is_symlink(info):
            reason = (
                "symlink archive member skipped. Following one would let an archive name a "
                "file outside itself; the target is indexed on its own if it is in the source."
            )
            return _refusal(raw, path, depth, reason, DocumentStatus.FAILED), False
        if info.flag_bits & _ENCRYPTED_FLAG:
            reason = "encrypted archive member. Supply the archive decrypted to index it."
            return _refusal(raw, path, depth, reason, DocumentStatus.FAILED), False
        if depth > self._config.max_depth:
            reason = (
                f"archive nesting depth exceeded — this member is {depth} containers deep and "
                f"the limit is {self._config.max_depth}."
            )
            return (
                _refusal(raw, path, depth, reason, DocumentStatus.UNSUPPORTED_MEDIA_TYPE),
                False,
            )
        if info.file_size > self._config.max_member_bytes:
            # The cheap pre-filter, and the only thing the declared size is trusted for: a
            # member that admits to being oversized need not be read to find out.
            reason = (
                f"archive member declares {info.file_size} uncompressed bytes, above the "
                f"{self._config.max_member_bytes}-byte per-member limit."
            )
            return _refusal(raw, path, depth, reason, DocumentStatus.FAILED), False

        try:
            content = self._read(archive, info, used_bytes)
        except _RefusalError as refusal:
            return (
                _refusal(raw, path, depth, refusal.reason, DocumentStatus.FAILED),
                refusal.fatal,
            )
        except zipfile.BadZipFile as exc:
            reason = (
                f"archive member could not be decompressed ({exc}). Its stream does not match "
                f"what its header declares, so nothing here is safe to index."
            )
            return _refusal(raw, path, depth, reason, DocumentStatus.FAILED), False

        digest = content_hash(content)
        if digest in hashes:
            reason = (
                "container cycle detected — this member is byte-identical to one of the "
                "archives containing it, so descending into it would not terminate."
            )
            return _refusal(raw, path, depth, reason, DocumentStatus.FAILED), False

        source_id = member_source_id(raw.source_id, path, scheme=ARCHIVE_SCHEME)
        uri = member_uri(raw.uri, path, scheme=ARCHIVE_SCHEME)
        metadata: Metadata = {
            CONTAINER_DEPTH: depth,
            PATH_HASHES: [*hashes, digest],
            TREE_BYTES: used_bytes + len(content),
            TREE_MEMBERS: used_members,
            "member_compressed_size": info.compress_size,
            "member_modified": _modified(info),
        }
        return (
            ExpandedMember(
                source_id=source_id,
                uri=uri,
                raw=RawDocument(
                    source_id=source_id,
                    uri=uri,
                    media_type=media_type_for(path),
                    content=content,
                    metadata=metadata,
                ),
                depth=depth,
                metadata=metadata,
            ),
            False,
        )

    def _read(self, archive: zipfile.ZipFile, info: zipfile.ZipInfo, used_bytes: int) -> bytes:
        """Stream a member through a counter that stops it, whatever its header says.

        Raises:
            _RefusalError: A limit was reached. The tree budget is fatal to the archive, because
                every remaining member would hit the same wall; the per-member limits are not.
        """
        member_ceiling = self._config.max_member_bytes
        tree_ceiling = max(0, self._config.max_total_bytes - used_bytes)
        ratio_ceiling = max(1, info.compress_size) * self._config.max_compression_ratio

        probe = copy.copy(info)
        probe.file_size = _NO_DECLARED_CEILING
        read = 0
        parts: list[bytes] = []
        with archive.open(probe) as handle:
            for block in iter(lambda: handle.read(_READ_CHUNK), b""):
                read += len(block)
                if read > member_ceiling:
                    msg = (
                        f"archive member exceeded the {member_ceiling}-byte per-member limit "
                        f"while streaming; it declared {info.file_size} bytes."
                    )
                    raise _RefusalError(msg)
                if read > ratio_ceiling:
                    msg = (
                        f"archive member expanded past {self._config.max_compression_ratio:g}:1 "
                        f"while streaming — {read} bytes out of {info.compress_size} "
                        f"compressed. This is what a zip bomb looks like."
                    )
                    raise _RefusalError(msg)
                if read > tree_ceiling:
                    msg = (
                        f"archive tree exceeded the {self._config.max_total_bytes}-byte total "
                        f"while streaming this member. Members already expanded keep their "
                        f"documents; the rest of this archive is not read."
                    )
                    raise _RefusalError(msg, fatal=True)
                parts.append(block)
        return b"".join(parts)

    def _opened(self, raw: RawDocument) -> zipfile.ZipFile:
        """Open the archive, declining anything that is not one this parser should read.

        Raises:
            ParseError: The bytes are not a zip, or its directory says it is a document
                container rather than an archive.
        """
        try:
            archive = zipfile.ZipFile(io.BytesIO(raw.as_bytes()))
        except (zipfile.BadZipFile, OSError) as exc:
            msg = (
                f"{raw.uri}: declining — these bytes are not a readable zip archive ({exc}). "
                f"Check the media type this document was routed with."
            )
            raise ParseError(msg) from exc
        entries = archive.infolist()
        container = _document_container(entries)
        if container is not None:
            archive.close()
            msg = (
                f"{raw.uri}: declining — {NOT_AN_ARCHIVE}. Its directory contains "
                f"{container!r}, so it is a word processor, spreadsheet, presentation or "
                f"e-book file that happens to be a zip. Route it by its real media type; "
                f"indexing it here would cite its internal XML parts."
            )
            raise ParseError(msg)
        return archive


# --- helpers ---------------------------------------------------------------------------


def _document_container(entries: list[zipfile.ZipInfo]) -> str | None:
    """The directory entry that gives a document container away, if there is one.

    Two signatures, because the formats disagree: OOXML files carry
    ``[Content_Types].xml`` anywhere in the directory, and ODF and EPUB carry an uncompressed
    ``mimetype`` member first, at offset zero.
    """
    for entry in entries:
        if entry.filename == _OOXML_DIRECTORY:
            return entry.filename
    if entries and entries[0].filename == _DOCUMENT_MIMETYPE and entries[0].header_offset == 0:
        return _DOCUMENT_MIMETYPE
    return None


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    """Whether a member is a Unix symlink, recorded in the external attributes.

    :mod:`zipfile` does not follow one today. The defence is against an extraction path being
    added later without remembering that an archive can name a file outside itself.
    """
    return (info.external_attr >> 16) & _UNIX_FILE_TYPE == _SYMLINK_MODE


def _modified(info: zipfile.ZipInfo) -> str:
    """The member's stored timestamp, formatted from the tuple the archive holds.

    No timezone, because a zip does not record one. Rendering it as if it were UTC would
    invent a fact.
    """
    year, month, day, hour, minute, second = info.date_time
    return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}"


def _unique(path: str, ordinal: int, seen: set[str]) -> str:
    """``path``, qualified by its position in the archive if that name is already taken.

    A zip may hold two entries with the same name — appending to an archive produces exactly
    that, and so does a crafted one — and after normalisation ``a/b.txt`` and ``./a/b.txt``
    are also the same name. Storage reconciles members by ``source_id``, which is derived from
    this, so a collision is not an error anybody sees: the second member overwrites the first,
    and the archive quietly contributes fewer documents than it contains.

    The qualifier is the central-directory ordinal, which is positional and therefore stable
    for a given archive — the same disambiguation, and the same reason for it, as two mail
    attachments sharing a filename.
    """
    if path not in seen:
        seen.add(path)
        return path
    qualified = f"{path}#{ordinal}"
    seen.add(qualified)
    return qualified


def _refusal(
    raw: RawDocument,
    path: str,
    depth: int,
    reason: str,
    status: DocumentStatus,
    *,
    address: str | None = None,
) -> MemberFailure:
    """A member that will not become a document, addressed so a person can find it.

    The address is built from the *normalised* path where there is one. A member rejected for
    escaping the archive root has no usable path by definition, so the caller supplies one
    that is unique within the archive and its real name goes in metadata — putting
    ``../../etc/passwd`` into a ``uri`` would reproduce the attack in the index and in every
    diagnostic that prints it, and a shared placeholder would give every rejected member one
    ``source_id``, which storage reconciles by.
    """
    safe = address or inner_path(path) or "member"
    return MemberFailure(
        source_id=member_source_id(raw.source_id, safe, scheme=ARCHIVE_SCHEME),
        uri=member_uri(raw.uri, safe, scheme=ARCHIVE_SCHEME),
        status=status,
        reason=reason,
        depth=depth,
        metadata={"member_name": path},
    )
