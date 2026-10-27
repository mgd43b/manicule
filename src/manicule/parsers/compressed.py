"""Gzip and tar, whose members are documents rather than chunks.

The zip parser (:mod:`manicule.parsers.archive`) is shaped around a zip's central directory.
These two formats are streams, so they get a reader of their own that holds the same promises:
a member is a document in its own right and goes back through the whole parser chain
(``docs/parsing.md`` §9.1), the container emits no chunks, expansion is one level at a time with
the whole-tree limits carried in :attr:`~manicule.core.content.RawDocument.metadata`, and every
limit **counts the bytes it streams** rather than trusting a header (§9.3).

**Gzip is a one-member container.** ``report.xml.gz`` expands to one document named
``report.xml``; ``bundle.tgz`` to ``bundle.tar``, which is then a tar and expands again, one
level deeper. A gzip with no ``.gz`` suffix (a mail attachment is often named for what it was
before compression) is named ``content``.

**A tar refuses everything that is not a regular file.** Symlinks, hard links, devices and
FIFOs name things outside the archive, which a zip can do only with a symlink, and refusing them
is reported rather than silent. The archive is read as an uncompressed stream: ``.tar.gz`` is
two containers, and the gzip layer has already been through the byte and ratio limits.

What a member is, is :func:`~manicule.parsers.mediatype.resolve_member_media_type` over its name
and bytes, as for a mail attachment.
"""

from __future__ import annotations

import gzip
import io
import tarfile
import zlib
from collections.abc import AsyncIterator, Callable, Iterator

from manicule.core.anchors import Anchor
from manicule.core.content import DocumentStatus, Metadata, ParsedBlock, RawDocument
from manicule.core.errors import ParseError
from manicule.core.ids import content_hash
from manicule.parsers.base import ParserProfile
from manicule.parsers.config import COMPRESSED_MEDIA_TYPES, ArchiveConfig
from manicule.parsers.expansion import (
    CONTAINER_DEPTH,
    OCTET_STREAM,
    PATH_HASHES,
    TREE_BYTES,
    TREE_MEMBERS,
    ExpandedMember,
    MemberFailure,
    MemberOutcome,
    container_depth_of,
    inner_path,
    member_source_id,
    member_uri,
    path_hashes_of,
    tree_bytes_of,
    tree_members_of,
)
from manicule.parsers.mediatype import resolve_member_media_type

__all__ = ["COMPRESSED_MEDIA_TYPES", "CompressedParser"]

GZIP_SCHEME = "gzip"
TAR_SCHEME = "tar"

_GZIP_MAGIC = b"\x1f\x8b"
_READ_CHUNK = 64 * 1024
_RATIO_FLOOR = 1 << 20
"""Output below this is never a ratio violation. A tar is padded in 10 KiB blocks, so a few small
files compress by a hundred to one and are not a bomb; anything that matters is stopped by the
byte limits, which this does not relax."""


class _Refused(Exception):  # noqa: N818 - a control-flow signal, named for what it says
    def __init__(self, reason: str, *, fatal: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.fatal = fatal


class CompressedParser:
    """Expands a gzip or tar into the documents inside it, and emits no blocks of its own."""

    media_types = COMPRESSED_MEDIA_TYPES
    profile = ParserProfile(name="compressed", max_unlocated_ratio=0.0, max_pagelevel_ratio=None)

    def __init__(self, config: ArchiveConfig) -> None:
        self._config = config

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        """Yield nothing, having checked that this really is a gzip or a tar.

        Raises:
            ParseError: The bytes are neither, so the next parser in the chain gets a turn.
        """
        _kind(raw)
        return
        yield  # pragma: no cover - unreachable, present to make this an async generator

    async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
        """Always ``None``: this parser emits no anchors."""
        del anchor, raw
        return None

    async def expand(self, raw: RawDocument) -> AsyncIterator[MemberOutcome]:
        """Yield the documents directly inside, one level only.

        Raises:
            ParseError: The bytes are neither a gzip nor a tar.
        """
        kind = _kind(raw)
        walk = self._gzip if kind == GZIP_SCHEME else self._tar
        for outcome in walk(raw):
            yield outcome
            if isinstance(outcome, MemberFailure) and outcome.truncates:
                return

    # --- gzip ----------------------------------------------------------------------------

    def _gzip(self, raw: RawDocument) -> Iterator[MemberOutcome]:
        context = _Context.of(raw, GZIP_SCHEME)
        name = _gunzipped_name(raw.uri)
        context.members += 1
        refusal = self._refusal_before_reading(context, name)
        if refusal is not None:
            yield refusal
            return
        data = raw.as_bytes()
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
                content = self._bounded(
                    stream.read, context.used_bytes, ratio_basis=len(data), what="gzip stream"
                )
        except _Refused as refused:
            yield context.refusal(
                name, refused.reason, DocumentStatus.FAILED, truncates=refused.fatal
            )
            return
        except (OSError, EOFError, zlib.error) as exc:
            reason = (
                f"gzip stream could not be decompressed ({exc}); nothing here is safe to index."
            )
            yield context.refusal(name, reason, DocumentStatus.FAILED)
            return
        yield context.member(self._config, name, content)

    # --- tar -----------------------------------------------------------------------------

    def _tar(self, raw: RawDocument) -> Iterator[MemberOutcome]:
        context = _Context.of(raw, TAR_SCHEME)
        seen: set[str] = set()
        try:
            archive = tarfile.open(fileobj=io.BytesIO(raw.as_bytes()), mode="r:")  # noqa: SIM115
        except (tarfile.TarError, OSError) as exc:
            msg = f"{raw.uri}: declining — not a readable tar archive ({exc})."
            raise ParseError(msg) from exc
        with archive:
            for ordinal, info in enumerate(archive):
                if info.isdir():
                    continue
                context.members += 1
                if context.members > self._config.max_members:
                    yield context.refusal(
                        _unique(inner_path(info.name) or "member", ordinal, seen),
                        f"archive member count exceeded: {context.members} members across this "
                        f"container tree, and the limit is {self._config.max_members}.",
                        DocumentStatus.FAILED,
                        truncates=True,
                    )
                    return
                path = inner_path(info.name)
                if path is None:
                    yield context.refusal(
                        _unique("member", ordinal, seen),
                        f"archive member name escapes the archive root: {info.name!r}. Names "
                        f"are never rewritten into a plausible one, so it is reported instead.",
                        DocumentStatus.FAILED,
                    )
                    continue
                path = _unique(path, ordinal, seen)
                if not info.isfile():
                    yield context.refusal(
                        path,
                        "tar member is a link, device or FIFO and was skipped. Following one "
                        "would let an archive name something outside itself.",
                        DocumentStatus.FAILED,
                    )
                    continue
                refusal = self._refusal_before_reading(context, path, declared=info.size)
                if refusal is not None:
                    yield refusal
                    continue
                handle = archive.extractfile(info)
                if handle is None:  # pragma: no cover - isfile() guarantees a handle
                    continue
                try:
                    content = self._bounded(
                        handle.read, context.used_bytes, ratio_basis=None, what="tar member"
                    )
                except _Refused as refused:
                    yield context.refusal(
                        path, refused.reason, DocumentStatus.FAILED, truncates=refused.fatal
                    )
                    if refused.fatal:
                        return
                    continue
                except (tarfile.TarError, OSError, EOFError) as exc:
                    reason = f"tar member could not be read ({exc})."
                    yield context.refusal(path, reason, DocumentStatus.FAILED)
                    continue
                outcome = context.member(self._config, path, content)
                if isinstance(outcome, ExpandedMember):
                    context.used_bytes += len(content)
                yield outcome

    # --- shared --------------------------------------------------------------------------

    def _refusal_before_reading(
        self, context: _Context, path: str, *, declared: int = 0
    ) -> MemberFailure | None:
        if context.members > self._config.max_members:
            return context.refusal(
                path,
                f"archive member count exceeded: {context.members} members across this "
                f"container tree, and the limit is {self._config.max_members}.",
                DocumentStatus.FAILED,
                truncates=True,
            )
        if context.depth > self._config.max_depth:
            return context.refusal(
                path,
                f"archive nesting depth exceeded — this member is {context.depth} containers "
                f"deep and the limit is {self._config.max_depth}.",
                DocumentStatus.UNSUPPORTED_MEDIA_TYPE,
            )
        if declared > self._config.max_member_bytes:
            return context.refusal(
                path,
                f"archive member declares {declared} uncompressed bytes, above the "
                f"{self._config.max_member_bytes}-byte per-member limit.",
                DocumentStatus.FAILED,
            )
        return None

    def _bounded(
        self,
        read: Callable[[int], bytes],
        used_bytes: int,
        *,
        ratio_basis: int | None,
        what: str,
    ) -> bytes:
        """Stream through a counter that stops at the first limit, whatever a header says."""
        member_ceiling = self._config.max_member_bytes
        tree_ceiling = max(0, self._config.max_total_bytes - used_bytes)
        ratio_ceiling = (
            None
            if ratio_basis is None
            else max(_RATIO_FLOOR, max(1, ratio_basis) * self._config.max_compression_ratio)
        )
        total = 0
        parts: list[bytes] = []
        for block in iter(lambda: read(_READ_CHUNK), b""):
            total += len(block)
            if total > member_ceiling:
                msg = f"{what} exceeded the {member_ceiling}-byte per-member limit while streaming."
                raise _Refused(msg)
            if ratio_ceiling is not None and total > ratio_ceiling:
                msg = (
                    f"{what} expanded past {self._config.max_compression_ratio:g}:1 while "
                    f"streaming. This is what a decompression bomb looks like."
                )
                raise _Refused(msg)
            if total > tree_ceiling:
                msg = (
                    f"archive tree exceeded the {self._config.max_total_bytes}-byte total while "
                    f"streaming this member; the rest of this archive is not read."
                )
                raise _Refused(msg, fatal=True)
            parts.append(block)
        return b"".join(parts)


class _Context:
    """What one container's expansion shares: depth, budgets and the address scheme."""

    def __init__(self, raw: RawDocument, scheme: str) -> None:
        self.raw = raw
        self.scheme = scheme
        self.depth = container_depth_of(raw) + 1
        self.hashes = path_hashes_of(raw)
        self.used_bytes = tree_bytes_of(raw)
        self.members = tree_members_of(raw)

    @classmethod
    def of(cls, raw: RawDocument, scheme: str) -> _Context:
        return cls(raw, scheme)

    def refusal(
        self, path: str, reason: str, status: DocumentStatus, *, truncates: bool = False
    ) -> MemberFailure:
        safe = inner_path(path) or "member"
        return MemberFailure(
            source_id=member_source_id(self.raw.source_id, safe, scheme=self.scheme),
            uri=member_uri(self.raw.uri, safe, scheme=self.scheme),
            status=status,
            reason=reason,
            depth=self.depth,
            truncates=truncates,
            metadata={"member_name": path},
        )

    def member(self, config: ArchiveConfig, path: str, content: bytes) -> MemberOutcome:
        del config
        digest = content_hash(content)
        if digest in self.hashes:
            return self.refusal(
                path,
                "container cycle detected — this member is byte-identical to one of the "
                "containers holding it, so descending into it would not terminate.",
                DocumentStatus.FAILED,
            )
        source_id = member_source_id(self.raw.source_id, path, scheme=self.scheme)
        uri = member_uri(self.raw.uri, path, scheme=self.scheme)
        metadata: Metadata = {
            CONTAINER_DEPTH: self.depth,
            PATH_HASHES: [*self.hashes, digest],
            TREE_BYTES: self.used_bytes + len(content),
            TREE_MEMBERS: self.members,
        }
        return ExpandedMember(
            source_id=source_id,
            uri=uri,
            raw=RawDocument(
                source_id=source_id,
                uri=uri,
                media_type=resolve_member_media_type(OCTET_STREAM, path, content),
                content=content,
                metadata=metadata,
            ),
            depth=self.depth,
            metadata=metadata,
        )


def _kind(raw: RawDocument) -> str:
    """``gzip`` or ``tar`` by the bytes, whatever the media type said.

    Raises:
        ParseError: Neither.
    """
    data = raw.as_bytes()
    if data.startswith(_GZIP_MAGIC):
        return GZIP_SCHEME
    if data[257:262] == b"ustar":
        return TAR_SCHEME
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
            archive.next()
    except (tarfile.TarError, OSError) as exc:
        msg = f"{raw.uri}: declining — these bytes are neither a gzip stream nor a tar archive."
        raise ParseError(msg) from exc
    return TAR_SCHEME


def _gunzipped_name(uri: str) -> str:
    """The member a gzip holds, named by stripping the compression suffix from its own name."""
    base = uri.rsplit("/", 1)[-1]
    lowered = base.lower()
    if lowered.endswith(".tgz"):
        return f"{base[:-4]}.tar"
    if lowered.endswith(".gz") and len(base) > len(".gz"):
        return base[:-3]
    return "content"


def _unique(path: str, ordinal: int, seen: set[str]) -> str:
    if path not in seen:
        seen.add(path)
        return path
    qualified = f"{path}#{ordinal}"
    seen.add(qualified)
    return qualified
