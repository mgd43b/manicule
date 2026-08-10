"""Documents that contain other documents: archives, and email attachments.

A member of an archive is not a chunk of the archive. It is a document in its own right — a
PDF inside a zip is a PDF, and it gets its own parser, its own chunks and its own anchors,
exactly as if it had been fetched directly (``docs/parsing.md`` §9.1). The container emits no
chunks at all: a chunk listing filenames is retrieval noise competing with the real content
inside it.

This module is the small shared vocabulary that makes that work, and nothing else. It imports
only the standard library, pydantic and :mod:`manicule.core`, because both the archive parser
and the mail parser import it and neither should pay for the other's dependencies.

**The parser expands one level and never recurses.** It is handed a container and yields the
documents directly inside it; the ingest pipeline queues those and comes back for the next
level. That is what makes the traversal breadth-first — a wide archive cannot starve a batch
by depth-first descent into one branch (§9.2) — and it is why the whole-tree limits travel in
:attr:`~manicule.core.content.RawDocument.metadata` under the keys named below rather than in
parser state. A parser that accumulated them in itself would give a different answer on the
second document it saw.

**Failure is an outcome, not an omission.** A member that is encrypted, too deep, too large or
named so that it escapes the archive root is reported as a :class:`MemberFailure` carrying the
reason and the status it should be stored under. Dropping it instead would make the member set
of an archive depend silently on what the parser felt able to read, and "the archive had 200
files and we indexed 197" is not a fact anyone would ever discover.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from manicule.core.content import DocumentStatus, Metadata, RawDocument

__all__ = [
    "CONTAINER_DEPTH",
    "CONTAINER_SEPARATOR",
    "MAX_DEPTH",
    "OCTET_STREAM",
    "PATH_HASHES",
    "TREE_BYTES",
    "TREE_MEMBERS",
    "ExpandedMember",
    "MemberFailure",
    "MemberOutcome",
    "SupportsExpansion",
    "container_depth_of",
    "inner_path",
    "media_type_for",
    "member_source_id",
    "member_uri",
    "path_hashes_of",
    "tree_bytes_of",
    "tree_members_of",
]

MAX_DEPTH = 3
"""How far containers may nest, counted from the top-level document.

A zip in a zip in a zip is already unusual and deeper is either a mistake or an attack. A
member past the limit is stored with a reason rather than dropped, so the boundary is visible
to whoever hits it (``docs/parsing.md`` §9.2).
"""

CONTAINER_SEPARATOR = "!/"
"""What separates a container's address from the path inside it.

``zip:file:///corpus/reports.zip!/reports/2026-q1.pdf``. The convention is long-standing, it
is unambiguous against every character a member name may contain, and it survives being
pasted into a bug report.
"""

OCTET_STREAM = "application/octet-stream"
"""The media type for a member whose kind cannot be told from its name.

Not a claim about the content — the absence of one. ``docs/parsing.md`` §6.1 lets content
sniffing replace this and only this, which is exactly why it must not be replaced with a
plausible guess here.
"""

CONTAINER_DEPTH = "container_depth"
"""Metadata key: how far inside a top-level document this one already is. Absent means zero."""

PATH_HASHES = "container_path_hashes"
"""Metadata key: content hashes of every container on the path to this document.

Cycle detection (§9.2). A zip cannot literally contain itself, but self-referential nesting by
identical content is trivial to build and costs nothing to defend against.
"""

TREE_BYTES = "container_tree_bytes"
"""Metadata key: uncompressed bytes already streamed out of this document's container tree."""

TREE_MEMBERS = "container_tree_members"
"""Metadata key: members already produced from this document's container tree."""


_MEDIA_TYPE_BY_SUFFIX: dict[str, str] = {
    ".css": "text/css",
    ".csv": "text/csv",
    ".eml": "message/rfc822",
    ".htm": "text/html",
    ".html": "text/html",
    ".ipynb": "application/x-ipynb+json",
    ".json": "application/json",
    ".md": "text/markdown",
    ".pdf": "application/pdf",
    ".toml": "application/toml",
    ".txt": "text/plain",
    ".xml": "application/xml",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".zip": "application/zip",
}
"""Suffix to media type, spelled out rather than taken from :mod:`mimetypes`.

:func:`mimetypes.guess_type` consults ``/etc/mime.types`` and the Windows registry, so the
same archive would expand to documents of different media types on two machines and therefore
to two different chunkings of one corpus. That is the hazard ``docs/parsing.md`` §8.1 refuses
for grammars, and it is the same hazard here.
"""


def media_type_for(name: str) -> str:
    """The media type a member's name implies, or :data:`OCTET_STREAM` when it implies none.

    A proposal, not a decision: a container knows the name and nothing else, and §6.1 puts the
    filename extension third behind an explicit override and a type the source declared. The
    pipeline is free to overrule it.
    """
    lowered = name.lower()
    dot = lowered.rfind(".")
    slash = lowered.rfind("/")
    if dot <= slash + 1:
        return OCTET_STREAM
    return _MEDIA_TYPE_BY_SUFFIX.get(lowered[dot:], OCTET_STREAM)


_DRIVE_LETTER = re.compile(r"^[A-Za-z]:")


def inner_path(name: str) -> str | None:
    """A member's name as a path inside its container, or ``None`` when it escapes.

    Members are parsed in memory and never written to disk, which removes most of the risk in
    a hostile name — but the name still becomes part of a ``uri`` shown to users and stored in
    the index, so it is normalised, and a name that escapes the container root is **rejected**
    rather than sanitised. Sanitising ``../../etc/passwd`` into ``etc/passwd`` produces a
    citation that looks ordinary and describes a file the archive never contained.

    Backslashes are read as separators because a zip written on Windows uses them, empty and
    ``.`` segments are dropped, and anything else — a leading ``/``, a drive letter, a ``..``
    segment anywhere — is a refusal.
    """
    candidate = name.replace("\\", "/")
    if candidate.startswith("/") or _DRIVE_LETTER.match(candidate):
        return None
    parts: list[str] = []
    for part in candidate.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            return None
        parts.append(part)
    return "/".join(parts) if parts else None


def member_uri(container_uri: str, path: str, *, scheme: str) -> str:
    """The address of a document inside a container, as a person would paste it."""
    return f"{scheme}:{container_uri}{CONTAINER_SEPARATOR}{path}"


def member_source_id(container_source_id: str, path: str, *, scheme: str) -> str:
    """The stable identity of a member, built from its inner path and never from its position.

    Identity is what reconciliation compares, so it has to rest on something the container can
    promise is stable. A member that moves within the archive is the same member, and one
    inserted ahead of another must not inherit its identity — which is exactly what an
    ordinal would do, silently, on the next sync.
    """
    return f"{scheme}:{container_source_id}{CONTAINER_SEPARATOR}{path}"


class ExpandedMember(BaseModel):
    """One document found inside another."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str = Field(min_length=1, description="Built from the inner path, never a position.")
    uri: str = Field(min_length=1, description="``zip:<container-uri>!/reports/2026-q1.pdf``.")
    raw: RawDocument
    depth: int = Field(ge=1, description="1 for a member of a top-level document.")
    metadata: Metadata = Field(default_factory=dict)


class MemberFailure(BaseModel):
    """A member that could not become a document, and why.

    Carries the status as well as the reason so the pipeline stores it mechanically rather
    than re-deriving one from a string. ``docs/parsing.md`` draws the line where §6.4 does:
    depth exceeded is :attr:`~manicule.core.content.DocumentStatus.UNSUPPORTED_MEDIA_TYPE`
    because nothing broke and nothing would break on a retry, while an encrypted or oversized
    member is :attr:`~manicule.core.content.DocumentStatus.FAILED` because there is content
    there that we did not read.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    status: DocumentStatus
    reason: str = Field(min_length=1, description="Actionable, and shown in diagnostics.")
    depth: int = Field(ge=1)
    metadata: Metadata = Field(default_factory=dict)


type MemberOutcome = ExpandedMember | MemberFailure
"""What one member of a container turned into.

A union rather than "yield the good ones": exceeding a limit fails that member, or that
archive, and never the batch (§9.3), and a failure nobody is told about is indistinguishable
from a member that was never there.
"""


@runtime_checkable
class SupportsExpansion(Protocol):
    """A parser whose documents contain other documents: archives, and email attachments."""

    def expand(self, raw: RawDocument) -> AsyncIterator[MemberOutcome]:
        """Yield the documents directly inside ``raw``, one level only.

        Raises:
            ParseError: ``raw`` is not a container of this kind, so the next parser in the
                fallback chain gets a turn.
        """
        ...


def container_depth_of(raw: RawDocument) -> int:
    """How deep inside a top-level document ``raw`` already is. Zero for a top-level one."""
    return _non_negative_int(raw.metadata.get(CONTAINER_DEPTH))


def tree_bytes_of(raw: RawDocument) -> int:
    """Uncompressed bytes already streamed out of this document's container tree."""
    return _non_negative_int(raw.metadata.get(TREE_BYTES))


def tree_members_of(raw: RawDocument) -> int:
    """Members already produced from this document's container tree."""
    return _non_negative_int(raw.metadata.get(TREE_MEMBERS))


def path_hashes_of(raw: RawDocument) -> tuple[str, ...]:
    """Content hashes of every container between the top-level document and this one."""
    value = raw.metadata.get(PATH_HASHES)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    return tuple(entry for entry in value if isinstance(entry, str))


def _non_negative_int(value: object) -> int:
    """A counter carried through metadata: absent, or a non-negative integer.

    Metadata is free-form JSON, so ``True`` is an ``int`` as far as Python is concerned and a
    counter that arrived as a boolean would be read as one. Excluding it here keeps a
    mistyped metadata key from spending a byte of anyone's budget.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)
