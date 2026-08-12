"""The sidecar manifest: a local file declaring what the file beside it is a copy of.

A mirroring tool that writes ``123456.html`` usually knows perfectly well what it just
downloaded — the page's title, its address, its version. Generic filesystem ingestion throws all
of that away and cites the filename, because the filename is the only thing it was given. A
sidecar closes that gap without teaching manicule about any particular documentation product:
next to ``123456.html`` sits ``123456.html.source.json``, and it says what the page was.

**A manifest is untrusted input, and it is treated as such throughout.** It is a file inside the
corpus. Anyone who can put a document where manicule will index it can put a manifest there too,
so every field is validated, and the three that could do real damage are handled by construction
rather than by inspection:

*The manifest cannot cause a read.*
    It declares no path that anything opens. manicule records the location of the artefact it
    actually walked to; :attr:`SNAPSHOT_PATH` is compared against that and refused on
    disagreement. So a ``../../../../etc/passwd`` in a manifest is a refusal with a reason, and
    at no point is it a filename. This is the difference between validating a path and never
    dereferencing one, and only the second is a defence — the first is one refactor away from
    being neither.

*The canonical URI cannot be a scheme a browser executes.*
    :data:`~manicule.core.provenance.CITABLE_SCHEMES` is an allowlist of two. See its docstring
    for why the denylist spelling of this is wrong.

*A title cannot smuggle markup or control codes into a rendered citation.*
    It is escaped on the browser surface like every other corpus string
    (``tests/web/test_escaping.py``), and control characters are refused outright here, because
    the CLI prints these fields to a terminal where ``\\x1b`` is not text.

**An unreadable manifest never fails the document.** The file beside it is a perfectly good
document and is indexed as one, with local identity, exactly as it would have been with no
manifest present. What it gains is a
:attr:`~manicule.core.provenance.Provenance.unavailable_reason` saying what was wrong — the
:class:`~manicule.core.content.Retention` and :class:`~manicule.core.anchors.Unlocated` pattern:
absent with a stated reason, never a silent partial success. A typo'd key would otherwise present
as a citation that quietly names a file, which is the bug this module was written to remove.

**Unknown keys are refused rather than ignored**, which is the one decision here that could
reasonably have gone the other way. The argument for ignoring them is that a mirroring tool may
add a field manicule does not honour. The argument against, which won: the overwhelmingly more
likely unknown key is a misspelling of a known one — ``canonical_url`` for ``canonical_uri`` —
and ignoring it means the manifest silently does nothing, which is indistinguishable from having
no manifest and is the exact failure being fixed. Refusing is loud, costs nothing, and the
refusal is soft.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from manicule.core.provenance import (
    PROVENANCE_KEY,
    LocalSnapshot,
    Provenance,
    SourceMetadata,
)

if TYPE_CHECKING:
    from manicule.core.content import Metadata

MANIFEST_SUFFIX: Final = ".source.json"
"""What a manifest is called, appended to the document's **whole** filename.

``123456.html`` is described by ``123456.html.source.json``, not by ``123456.source.json``.
Appending rather than replacing the suffix means ``page.html`` and ``page.pdf`` in one directory
get one manifest each instead of fighting over a single ``page.source.json``, and the mapping
stays a pure function of the filename in both directions.
"""

MAX_MANIFEST_BYTES: Final = 64 * 1024
"""Largest manifest that will be read.

A manifest describes one document with a dozen short fields, so this is roughly a thousand times
what a real one needs. It is here because the alternative is that a file named ``x.source.json``
decides how much memory an ingest run allocates, and refusing to read it is cheaper than parsing
it to find out it was too big.
"""

SNAPSHOT_PATH: Final = "snapshot_path"
"""The declared-location field, named because two places reason about it.

Accepted so that a tool emitting a complete record does not have its manifest refused for saying
something true, and cross-checked rather than trusted. It is never joined to a root, never
resolved and never opened.
"""


class _Manifest(BaseModel):
    """The on-disk shape of a manifest, exactly.

    Separate from :class:`~manicule.core.provenance.SourceMetadata` on purpose, and not merely as
    a parsing convenience. This model is a **wire format** — it is what somebody else's tool
    writes, so its field names are a compatibility surface and it holds two fields
    (:attr:`snapshot_path`, :attr:`snapshot_checksum`) that are claims to be checked rather than
    facts to be stored. ``SourceMetadata`` is manicule's own vocabulary and has nowhere to put
    either. Collapsing the two would put a declared local path inside the model whose whole
    purpose is that it cannot hold one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str = ""
    canonical_uri: str = ""
    source_id: str = ""
    version: str = ""
    created_at: datetime | None = None
    modified_at: datetime | None = None
    content_type: str = ""
    section_path: tuple[str, ...] = ()
    retrieved_at: datetime | None = None
    snapshot_path: str = Field(
        default="",
        description="Where the tool believes it wrote the snapshot, relative to the ingestion "
        "root. Cross-checked against where manicule found it; never used to locate anything.",
    )
    snapshot_checksum: str = Field(
        default="",
        description="The tool's digest of the snapshot bytes. Cross-checked against the digest "
        "manicule computes over the bytes it actually read.",
    )

    def source(self) -> SourceMetadata:
        """The canonical half, in manicule's vocabulary. Raises on anything invalid."""
        return SourceMetadata(
            title=self.title,
            canonical_uri=self.canonical_uri,
            source_id=self.source_id,
            version=self.version,
            created_at=self.created_at,
            modified_at=self.modified_at,
            content_type=self.content_type,
            section_path=self.section_path,
        )


def manifest_path_for(document: Path) -> Path:
    """Where ``document``'s manifest would be, whether or not one is there.

    One function, so discovery and fetch cannot look in two different places — a discovery that
    folded a manifest's timestamp into a change token while fetch read a manifest somewhere else
    would produce a document whose citation never updated and whose token said it had.
    """
    return document.with_name(document.name + MANIFEST_SUFFIX)


def is_manifest(path: Path) -> bool:
    """Whether ``path`` is a manifest rather than a document.

    A manifest is metadata about the document beside it, so indexing it as a document of its own
    would put a second, contentless entry in the corpus for every mirrored page — retrievable,
    citing nothing, and diluting every result set it appeared in.
    """
    return path.name.endswith(MANIFEST_SUFFIX)


def provenance_for(
    document: Path,
    *,
    root: Path,
    checksum: str,
) -> Provenance | None:
    """The record for ``document``, or ``None`` when it has no manifest.

    Args:
        document: The artefact that was read. Its location is observed, not declared: this is
            the path manicule walked to, and it is the only thing the snapshot half is built
            from.
        root: The configured ingestion root. Used to make the recorded path relative and to
            refuse a document from outside the tree; never combined with anything the manifest
            said.
        checksum: The digest of the bytes actually read, for cross-checking a declared one.

    Returns:
        ``None`` when no manifest exists, which is the ordinary case and leaves the document
        citing exactly as it does today. Otherwise a :class:`Provenance` — carrying a validated
        source record, or carrying the reason there is not one.
    """
    manifest = manifest_path_for(document)
    try:
        if not manifest.is_file():
            return None
    except OSError:  # pragma: no cover - an unstattable sibling is the same as an absent one
        return None

    snapshot = LocalSnapshot(path=_relative(document, root=root))
    try:
        parsed = _parse(manifest)
        _check_declarations(parsed, snapshot=snapshot, checksum=checksum)
        source = parsed.source()
    except _UnusableManifestError as refusal:
        return Provenance(snapshot=snapshot, unavailable_reason=f"{manifest.name}: {refusal}")
    return Provenance(
        source=source,
        snapshot=snapshot.model_copy(update={"retrieved_at": parsed.retrieved_at}),
    )


def with_provenance(metadata: Metadata, provenance: Provenance | None) -> Metadata:
    """``metadata`` carrying ``provenance``, or unchanged when there is none.

    The absent case returns the mapping untouched rather than writing a null under the key. A
    document with no manifest must be byte-identical to what it was before this feature existed,
    or "ordinary local files keep working" is a claim about code nobody exercised.
    """
    if provenance is None:
        return metadata
    return {**metadata, PROVENANCE_KEY: provenance.as_metadata_value()}


class _UnusableManifestError(Exception):
    """A manifest was found and cannot be used. The message is what the operator reads."""


def _parse(manifest: Path) -> _Manifest:
    """Read and validate one manifest, or raise :class:`_UnusableManifestError` saying why not."""
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
        msg = f"is not a usable manifest ({_reasons(exc)})"
        raise _UnusableManifestError(msg) from exc


def _reasons(error: ValidationError) -> str:
    """A pydantic failure as one line naming each field and what was wrong with it.

    Pydantic's own rendering is several lines with a documentation URL per error, which reads
    badly inside a ``status_detail`` and worse inside a citation's diagnostics. Naming the fields
    is the part that tells somebody which line of their manifest to fix.
    """
    parts: list[str] = []
    for detail in error.errors():
        location = ".".join(str(part) for part in detail["loc"]) or "manifest"
        parts.append(f"{location}: {detail['msg']}")
    return "; ".join(parts)


def _check_declarations(parsed: _Manifest, *, snapshot: LocalSnapshot, checksum: str) -> None:
    """Refuse a manifest whose claims about the local file contradict the local file.

    Both checks are comparisons against something already known, which is what makes them safe.
    Neither opens anything, and neither can be made to open anything: the values compared are a
    path manicule walked to and a digest manicule computed.

    A traversal attempt lands here as a plain disagreement — ``../../etc/passwd`` is not the
    relative path of the file that was read — so the guard that catches it is the same one that
    catches a mirroring tool with a stale directory layout. That is deliberate. A separate
    "contains ``..``" check would be a second rule to keep in step with the first, and the
    first already refuses everything the second would.
    """
    declared = parsed.snapshot_path.strip()
    if declared and _normalised(declared) != _normalised(snapshot.path):
        msg = (
            f"declares {SNAPSHOT_PATH} {parsed.snapshot_path!r}, but this snapshot was read from "
            f"{snapshot.path!r} under the ingestion root. A manifest describes the file beside "
            f"it and never points anywhere else"
        )
        raise _UnusableManifestError(msg)

    stated = parsed.snapshot_checksum.strip()
    if stated and stated != checksum:
        msg = (
            f"declares snapshot_checksum {stated!r}, but the bytes read hash to {checksum!r}. "
            f"The manifest and the snapshot beside it are not from the same retrieval"
        )
        raise _UnusableManifestError(msg)


def _normalised(path: str) -> str:
    """A declared or observed relative path, comparable across the two ways of writing one.

    Only separators and redundant ``./`` segments are folded, with :meth:`pathlib.PurePosixPath`
    doing it — a Windows-flavoured manifest saying ``docs\\page.html`` describes the same file as
    ``docs/page.html`` and should not be refused for it. **No ``..`` resolution happens here**,
    which matters: collapsing ``a/../b`` to ``b`` is exactly the normalisation that turns a
    traversal into something that compares equal to a legitimate path, and there is nothing to
    gain from it because both sides of the comparison are already relative and already clean.
    """
    return str(PurePosixPath(path.replace("\\", "/")))


def _relative(document: Path, *, root: Path) -> str:
    """``document`` relative to ``root``, as a POSIX path.

    Relative so that a citation reproduces on a machine whose corpus lives elsewhere, and so
    that a public citation does not publish this machine's directory layout. POSIX-separated so
    that the stored value does not depend on which platform indexed the corpus.

    Falls back to the bare filename when the document is somehow not under the root. That is not
    expected — the connector refuses to read outside its own tree — but a record is not worth
    raising over, and a filename is a true statement where an absolute path would leak one.
    """
    try:
        return str(PurePosixPath(document.relative_to(root)))
    except ValueError:  # pragma: no cover - the connector already refuses this
        return document.name


__all__ = [
    "MANIFEST_SUFFIX",
    "MAX_MANIFEST_BYTES",
    "SNAPSHOT_PATH",
    "is_manifest",
    "manifest_path_for",
    "provenance_for",
    "with_provenance",
]
