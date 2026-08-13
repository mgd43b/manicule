"""Enriched standalone HTML: reading a mirrored page's own account of itself.

Some offline exporters write one HTML file per page, and put the page's real identity inside
it — a machine-identifiable metadata section carrying the page id, its space, its version, when
it was last edited and where it is published — with the original storage-format body in a
``<main>``. Generic filesystem ingestion indexes the text of such a file perfectly well and then
cites ``1002.html`` and a ``file://`` URI, because the filename is the only thing it was handed.
Everything needed to cite the page properly was in the file the whole time.

**This module reads that metadata out. It does not rewrite the file, and that is the design.**

:mod:`.sidecar` already defines what a local file's authoritative metadata looks like on disk —
``1002.html.source.json`` beside ``1002.html`` — and :class:`.filesystem.FilesystemConnector`
already discovers it, folds it into the change token so editing either file re-ingests the page,
and attaches the validated record to the document. So the whole job here is to turn the page's
own metadata section into that manifest. Nothing new is added to the ingestion path, no second
connector walks the same tree, and :class:`~manicule.core.provenance.SourceMetadata` needs no
field it did not already have — every fact one of these pages states is one of its fields spelled
in an exporter's vocabulary, which is exactly what that interface was shaped for.

**The security consequence of not rewriting is the main argument for doing it this way.** A
converter that emitted new files would have to decide where they go, and the page's own metadata
is the obvious thing to name them after — at which point a page id of ``../../../etc/cron.d/x``
is a write primitive, and the defence is a validation somebody has to keep correct forever. Here
the output path is :func:`.sidecar.manifest_path_for` of the file that was walked to, and no
value read out of the document reaches it. Traversal is not refused; it is unrepresentable.

The rest follows the same rule the manifest reader already established:

*Nothing is fetched.* A canonical URL is read out of an ``href`` attribute and written to a
    field. It is metadata about where the page lives, and this module has no network client to
    dereference it with even if it wanted one.

*Nothing is executed.* The page is parsed, never run. Scripts and macro bodies are left in the
    file untouched — the file is not written to at all — and no part of the metadata section is
    interpreted as anything but text.

*Nothing is overwritten by accident.* An existing manifest is left alone unless the caller asks
    for it to be replaced, because the likely reason one is already there is that somebody wrote
    it by hand or with another tool.

*Validation is the real model's, not a copy of it.* The extracted fields are used to construct a
    :class:`~manicule.core.provenance.SourceMetadata`, so a ``javascript:`` canonical URI, a
    control character in a title and a naive timestamp are refused here by exactly the code that
    would refuse them at ingest. A second implementation of those rules would be a second thing
    to keep in step, and the two would disagree the first time either was edited.

**What the body means is deliberately not this module's problem.** The ``<main>`` holds
storage-format XHTML, and giving that real semantics is a parser's job — the existing web parser
indexes it as HTML today, and :func:`~manicule.parsers.web.recover_cdata` already rescues the
macro bodies a conforming HTML parser would silently delete. This module's job is the page's
identity and provenance; nothing here inspects the body except to hash the file it lives in.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

from selectolax.lexbor import LexborHTMLParser, LexborNode

from manicule.connectors import sidecar
from manicule.core.ids import content_hash
from manicule.core.provenance import SourceMetadata

__all__ = [
    "MAX_HTML_BYTES",
    "METADATA_SELECTOR",
    "EnrichedPage",
    "UnusablePageError",
    "extract",
    "manifest_for",
    "write_sidecars",
]

METADATA_SELECTOR: Final = "[data-source-metadata]"
"""How the metadata section identifies itself.

An attribute rather than a class or a heading, because it is the one part of these documents
that is addressed to a machine. A class is styling and an exporter is entitled to change it; a
heading is prose and is translated. Requiring the marker also means this module never guesses:
a page without one is reported as having no metadata rather than having its first table read
hopefully.
"""

MAX_HTML_BYTES: Final = 8 * 1024 * 1024
"""Largest page this will read.

Generous for a wiki page and present for the same reason :data:`.sidecar.MAX_MANIFEST_BYTES` is:
without it, a file in a directory somebody pointed the converter at decides how much memory the
conversion allocates.
"""

_LABELS: Final[Mapping[str, str]] = {
    "page id": "source_id",
    "id": "source_id",
    "title": "title",
    "space": "space",
    "space key": "space",
    "ancestors": "ancestors",
    "parent": "ancestors",
    "parents": "ancestors",
    "version": "version",
    "last modified": "modified_at",
    "modified": "modified_at",
    "updated": "modified_at",
    "created": "created_at",
    "source": "canonical_uri",
    "canonical url": "canonical_uri",
    "canonical": "canonical_uri",
    "url": "canonical_uri",
    "retrieved": "retrieved_at",
    "exported": "retrieved_at",
}
"""Labels this understands, normalised, mapped to the field each fills.

Several spellings per field because these documents are written by exporters that were not
coordinating with each other, and "Last modified", "Modified" and "Updated" are the same fact.
Unknown labels are ignored rather than refused: an exporter adding a row of its own is not an
error, and the failure this module actually has to catch is the *absence* of an identity, which
is checked directly.
"""

_ANCESTOR_SEPARATORS: Final = ("\u203a", "\u00bb", ">", "/", ",")
"""How exporters write a breadcrumb. Tried in order; the first one present splits the value.

The first two are U+203A and U+00BB, written as escapes because they are confusable with an
ASCII ``>`` on sight and that is exactly what makes them worth listing: a breadcrumb rendered
with a typographic separator is a different byte from one rendered with the ASCII character, and
an exporter using the pretty one would otherwise fall through to the ``/`` rule and have its page
path split on the slashes inside a title.
"""


class UnusablePageError(Exception):
    """A page was found and its metadata cannot be used. The message is what an operator reads."""


@dataclass(frozen=True, slots=True)
class EnrichedPage:
    """One page's declared identity, validated.

    Two fields rather than one flat record, mirroring
    :class:`~manicule.core.provenance.Provenance`'s own split: :attr:`source` is what the
    publication says about itself and :attr:`retrieved_at` is a statement about this copy. They
    are kept apart here for the same reason they are kept apart there — the second is not a fact
    about the document, and a model that could hold both would let a caller write one where the
    other belonged.
    """

    source: SourceMetadata
    retrieved_at: datetime | None = None


def extract(html: str) -> EnrichedPage:
    """The page's declared identity, or a refusal naming what was wrong.

    Args:
        html: The document's text. Parsed, never executed; see the module docstring.

    Returns:
        The validated record. :attr:`EnrichedPage.source` is a real
        :class:`~manicule.core.provenance.SourceMetadata`, so anything it would refuse at ingest
        is refused here instead — which is the point of building one rather than a dict.

    Raises:
        UnusablePageError: There is no metadata section, it declares no page id, it declares one
            label twice with two different values, or a value is one the citation interface will
            not carry. Every message names the field, because "invalid page" tells whoever runs
            the exporter nothing about which row to look at.
    """
    tree = LexborHTMLParser(html)
    sections = tree.css(METADATA_SELECTOR)
    if not sections:
        msg = (
            f"no {METADATA_SELECTOR} section, so the file states no identity of its own and "
            f"there is nothing to record that its filename does not already say"
        )
        raise UnusablePageError(msg)
    if len(sections) > 1:
        msg = (
            f"has {len(sections)} {METADATA_SELECTOR} sections. Which one describes the page is "
            f"exactly the ambiguity this cannot guess at"
        )
        raise UnusablePageError(msg)

    fields = _fields(sections[0])
    source_id = fields.get("source_id", "")
    if not source_id:
        msg = (
            "declares no page id. That is the identifier an updated snapshot is recognised by, "
            "and without it a re-export is a second document rather than a new version of one"
        )
        raise UnusablePageError(msg)

    section_path = _section_path(fields)
    try:
        source = SourceMetadata(
            title=fields.get("title") or _title(tree),
            canonical_uri=fields.get("canonical_uri", ""),
            source_id=source_id,
            version=fields.get("version", ""),
            created_at=_timestamp(fields.get("created_at"), field="created"),
            modified_at=_timestamp(fields.get("modified_at"), field="last modified"),
            section_path=section_path,
        )
    except ValueError as exc:
        msg = f"declares metadata this index will not cite ({_reason(exc)})"
        raise UnusablePageError(msg) from exc
    return EnrichedPage(
        source=source, retrieved_at=_timestamp(fields.get("retrieved_at"), field="retrieved")
    )


def manifest_for(page: EnrichedPage, *, html: bytes) -> dict[str, object]:
    """``page`` as a sidecar manifest, ready to write beside the file ``html`` came from.

    ``snapshot_checksum`` is included and ``snapshot_path`` deliberately is not, though
    :mod:`.sidecar` accepts both. They are cross-checked differently and only one of them is safe
    to state from here:

    The checksum is a fact about the bytes and travels with them.
        It is compared at ingest against the digest manicule computes over what it actually read,
        so a page edited after conversion — its version and modification time now stale in the
        manifest beside it — is *refused with a reason* rather than quietly citing metadata that
        describes an older revision. Re-running the conversion is the fix, and it is idempotent.

    The declared path is relative to the **ingestion** root, which this does not know.
        A conversion rooted anywhere other than where the connector is later pointed would emit a
        path that disagrees with the one manicule walked to, and every manifest would be refused
        for saying something true about a different root. Omitted, so the field stays available
        to a tool that genuinely knows both.
    """
    manifest: dict[str, object] = {"source_id": page.source.source_id}
    if page.source.title:
        manifest["title"] = page.source.title
    if page.source.canonical_uri:
        manifest["canonical_uri"] = page.source.canonical_uri
    if page.source.version:
        manifest["version"] = page.source.version
    if page.source.created_at is not None:
        manifest["created_at"] = page.source.created_at.isoformat()
    if page.source.modified_at is not None:
        manifest["modified_at"] = page.source.modified_at.isoformat()
    if page.source.section_path:
        manifest["section_path"] = list(page.source.section_path)
    if page.retrieved_at is not None:
        manifest["retrieved_at"] = page.retrieved_at.isoformat()
    manifest["snapshot_checksum"] = content_hash(html)
    return manifest


@dataclass(frozen=True, slots=True)
class SidecarOutcome:
    """What happened to one page."""

    path: Path
    written: bool = False
    skipped_reason: str = ""


def write_sidecars(root: Path, *, force: bool = False) -> list[SidecarOutcome]:
    """Write a manifest beside every enriched page under ``root``.

    Args:
        root: The directory to walk. Symlinks are not followed, at all — the same rule
            :class:`.filesystem.FilesystemConnector` walks by, and the reason a page cannot
            cause a write outside the tree it was found in.
        force: Replace manifests that already exist. Off by default: a manifest already there
            was most likely written by hand or by another tool, and silently replacing somebody's
            metadata is not a conversion, it is a loss.

    Returns:
        One outcome per HTML file considered, in walk order, whether or not it produced a
        manifest. A file that was skipped carries the reason — a run that reported only its
        successes would present "no metadata section anywhere" as a clean conversion.
    """
    resolved = root.expanduser().resolve()
    return [_convert(page, root=resolved, force=force) for page in _walk(resolved, root=resolved)]


def _convert(page: Path, *, root: Path, force: bool) -> SidecarOutcome:
    manifest_path = sidecar.manifest_path_for(page)
    if manifest_path.exists() and not force:
        return SidecarOutcome(
            page, skipped_reason="a manifest is already there; pass --force to replace it"
        )
    try:
        raw = page.read_bytes()
    except OSError as exc:
        return SidecarOutcome(page, skipped_reason=f"could not be read ({exc.strerror or exc})")
    if len(raw) > MAX_HTML_BYTES:
        return SidecarOutcome(
            page, skipped_reason=f"is {len(raw)} bytes, over the {MAX_HTML_BYTES}-byte limit"
        )
    try:
        extracted = extract(raw.decode("utf-8", errors="replace"))
    except UnusablePageError as refusal:
        return SidecarOutcome(page, skipped_reason=str(refusal))
    if not _within(manifest_path, root=root):  # pragma: no cover - `_walk` already refuses this
        return SidecarOutcome(page, skipped_reason="resolves outside the root")
    manifest_path.write_text(
        json.dumps(manifest_for(extracted, html=raw), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return SidecarOutcome(page, written=True)


def _walk(directory: Path, *, root: Path) -> Iterator[Path]:
    """Every HTML file under ``directory``, in a stable order, following no symlink."""
    try:
        entries: Sequence[Path] = sorted(directory.iterdir())
    except OSError:
        return
    for entry in entries:
        if entry.name.startswith(".") or entry.is_symlink():
            continue
        if entry.is_dir():
            yield from _walk(entry, root=root)
        elif entry.is_file() and entry.suffix.lower() in {".html", ".htm"}:
            if sidecar.is_manifest(entry) or not _within(entry, root=root):
                continue
            yield entry


def _within(path: Path, *, root: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:  # pragma: no cover - resolution failure is itself a refusal
        return False
    return resolved == root or root in resolved.parents


def _fields(section: LexborNode) -> dict[str, str]:
    """The labelled values in ``section``, keyed by the field each fills.

    Raises:
        UnusablePageError: One label appeared twice with two different values. Refused rather
            than resolved by order, because "the first one wins" is a rule nobody writing an
            exporter knows about, and the two values are equally likely to be the right one.
    """
    found: dict[str, str] = {}
    for label, value in _rows(section):
        field = _LABELS.get(label)
        if field is None or not value:
            continue
        previous = found.get(field)
        if previous is not None and previous != value:
            msg = (
                f"declares {label!r} twice, as {previous!r} and {value!r}. Which is the page's "
                f"is not something this can decide"
            )
            raise UnusablePageError(msg)
        found[field] = value
    return found


def _rows(section: LexborNode) -> Iterator[tuple[str, str]]:
    """``(normalised label, value)`` for every labelled row, however the exporter wrote it."""
    for node in section.css("dt"):
        sibling = node.next
        while sibling is not None and sibling.tag == "-text":
            sibling = sibling.next
        if sibling is not None and sibling.tag == "dd":
            yield _label(node.text(deep=True)), _value(sibling)
    for marker in section.css("strong, b, th"):
        parent = marker.parent
        if parent is None:
            continue
        whole = _collapse(parent.text(deep=True))
        marked = _collapse(marker.text(deep=True))
        if not whole.startswith(marked):
            continue
        yield _label(marked), _value(parent, without=marked)


def _value(node: LexborNode, *, without: str = "") -> str:
    """``node``'s value: an anchor's address where it has one, otherwise its text.

    The address is preferred because a canonical link is written ``<a href="…">canonical
    page</a>`` at least as often as it is written out in full, and recording the words "canonical
    page" as the page's address would be a citation pointing at nothing.

    **Read, never dereferenced.** Nothing in this module opens a URL.
    """
    anchors = node.css("a")
    for anchor in anchors:
        href = (anchor.attributes.get("href") or "").strip()
        if href:
            return href
    text = _collapse(node.text(deep=True))
    return text[len(without) :].strip().lstrip(":").strip() if without else text


def _label(raw: str) -> str:
    return _collapse(raw).rstrip(":").strip().lower()


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _section_path(fields: Mapping[str, str]) -> tuple[str, ...]:
    """The page's place in its source's hierarchy, coarsest first.

    The space leads because it is the outermost container, then any declared ancestors. The
    page's own title is **not** appended: the chunker adds that itself, and a path carrying it
    twice reaches the embedder as emphasis nobody intended
    (:attr:`~manicule.core.provenance.SourceMetadata.section_path`).
    """
    path: list[str] = []
    space = fields.get("space", "").strip()
    if space:
        path.append(space)
    path.extend(_ancestors(fields.get("ancestors", "")))
    return tuple(path)


def _ancestors(raw: str) -> Iterator[str]:
    value = raw.strip()
    if not value:
        return
    for separator in _ANCESTOR_SEPARATORS:
        if separator in value:
            for part in value.split(separator):
                if part.strip():
                    yield part.strip()
            return
    yield value


def _timestamp(raw: str | None, *, field: str) -> datetime | None:
    """``raw`` as an aware datetime, or ``None`` when it was not declared.

    Raises:
        UnusablePageError: It is not a timestamp, or it carries no offset. A naive one is
            refused rather than assumed to be UTC: read as UTC it is wrong by the exporting
            host's offset, and it is used to decide which of two versions is newer.
    """
    if raw is None or not raw.strip():
        return None
    text = raw.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        msg = f"declares {field} {text!r}, which is not an ISO-8601 timestamp"
        raise UnusablePageError(msg) from exc
    if parsed.tzinfo is None:
        msg = (
            f"declares {field} {text!r}, which carries no UTC offset and so does not say which "
            f"zone it is in"
        )
        raise UnusablePageError(msg)
    return parsed


def _title(tree: LexborHTMLParser) -> str:
    element = tree.css("title")
    return _collapse(element[0].text(deep=True)) if element else ""


def _reason(error: ValueError) -> str:
    return _collapse(str(error))
