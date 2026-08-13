"""Writing down what enriched HTML pages say about themselves, one manifest per page.

:mod:`.enriched` knows what an enriched page *is*. This is the conversion run over a directory of
them: for every file it considers, it either writes ``<page>.source.json`` beside it or reports
why it did not. Nothing else here — no walking of its own invention, no second idea of what a
page looks like, and no rewriting of anything.

**Why a manifest at all, when the connector adapts pages at fetch anyway.** Adaptation gives a
page its storage-format body and its citation. It cannot give it an *identity*, because identity
has to be known at discovery — before anything is fetched — and discovery does not read
documents. That is not a shortcut: it is what makes a re-sync of an unchanged corpus cost a
``stat`` per file instead of a full read. The manifest is the one thing beside a page small
enough to read on every walk, so it is where the page id has to be written down for the connector
to key on it. A page without one is still adapted, still parsed as storage format and still cited
by its own title and URL; what it does not get is a document identity that survives being moved.

**The security consequence of not rewriting the page is the main argument for this shape.** A
converter that emitted new files would have to decide where they go, and the page's own metadata
is the obvious thing to name them after — at which point a page id of ``../../../etc/cron.d/x``
is a write primitive, and the defence is a validation somebody has to keep correct for ever.
Here the output path is :func:`.sidecar.manifest_path_for` of the file that was walked to, and no
value read out of a document reaches it. Traversal is not refused; it is unrepresentable. It is
also why nothing is materialised on disk for the derived body: there is no derived-artifact
directory to place, name, exclude from discovery or garbage-collect, because the extraction is
cheap, deterministic and done in memory at the moment it is needed.

*Nothing is overwritten by accident.* An existing manifest is left alone unless the caller asks
    for it to be replaced, because the likely reason one is already there is that somebody wrote
    it by hand or with another tool.

*Nothing is fetched or executed.* Inherited from :mod:`.enriched`, which has no network client
    and no renderer, and reinforced by this module never writing to a page.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from manicule.connectors import sidecar
from manicule.connectors.enriched import (
    ADAPTER_VERSION,
    DEFAULT_PROFILE,
    MAX_HTML_BYTES,
    METADATA_SELECTOR,
    STORAGE_SELECTOR,
    Adaptation,
    AdapterOutcome,
    EnrichedPage,
    EnrichedProfile,
    UnusablePageError,
    adapt,
    extract,
)
from manicule.core.ids import content_hash

__all__ = [
    "ADAPTER_VERSION",
    "MAX_HTML_BYTES",
    "METADATA_SELECTOR",
    "STORAGE_SELECTOR",
    "AdapterOutcome",
    "EnrichedPage",
    "SidecarOutcome",
    "UnusablePageError",
    "extract",
    "manifest_for",
    "write_sidecars",
]


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

    **``content_type`` carries the representation and nothing carries the adapter's version.**
    The first is a fact about the page — the body inside it is storage format, and that stays
    true however this code changes — so it belongs in a manifest and reaches
    :attr:`~manicule.core.provenance.SourceMetadata.content_type` on the way through. The second
    is a fact about *this build of manicule*, and a manifest declaring it would go stale the day
    the adapter was improved: either it would be believed, and the corpus would record a version
    that never ran, or it would be checked, and every existing manifest would be refused on
    upgrade. The version is recorded where it is true — on the document, at the fetch that
    produced it — and it travels in the change token so that bumping it rebuilds the derived body
    without a manifest anywhere needing to be rewritten.
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
    if page.source.content_type:
        manifest["content_type"] = page.source.content_type
    if page.source.section_path:
        manifest["section_path"] = list(page.source.section_path)
    if page.retrieved_at is not None:
        manifest["retrieved_at"] = page.retrieved_at.isoformat()
    manifest["snapshot_checksum"] = content_hash(html)
    return manifest


@dataclass(frozen=True, slots=True)
class SidecarOutcome:
    """What happened to one page.

    :attr:`outcome` is the machine-readable half of :attr:`skipped_reason`, on the principle
    :attr:`~manicule.app.results.Check.remedy` follows: a surface that wants to *count* refusals
    by kind should not have to match on prose, and prose is the half that gets rewritten. They are
    set together at every construction, so a report's counters cannot come to describe a different
    failure than its reasons do.
    """

    path: Path
    outcome: AdapterOutcome
    written: bool = False
    skipped_reason: str = ""


def write_sidecars(
    root: Path,
    *,
    force: bool = False,
    profiles: Sequence[EnrichedProfile] = (DEFAULT_PROFILE,),
) -> list[SidecarOutcome]:
    """Write a manifest beside every enriched page under ``root``.

    Args:
        root: The directory to walk. Symlinks are not followed, at all — the same rule
            :class:`.filesystem.FilesystemConnector` walks by, and the reason a page cannot
            cause a write outside the tree it was found in.
        force: Replace manifests that already exist. Off by default: a manifest already there
            was most likely written by hand or by another tool, and silently replacing somebody's
            metadata is not a conversion, it is a loss.
        profiles: The exporter conventions to recognise, in precedence order. The same value the
            connector is configured with, so a page this run adapted is a page that run will
            adapt — a conversion written under one profile set and ingested under another would
            put manifests on disk for pages the connector then declined to read.

    Returns:
        One outcome per HTML file considered, in walk order, whether or not it produced a
        manifest. A file that was skipped carries the reason and the outcome — a run that
        reported only its successes would present "no metadata section anywhere" as a clean
        conversion of nothing.
    """
    resolved = root.expanduser().resolve()
    return [
        _convert(page, root=resolved, force=force, profiles=tuple(profiles))
        for page in _walk(resolved, root=resolved)
    ]


def _convert(
    page: Path, *, root: Path, force: bool, profiles: tuple[EnrichedProfile, ...]
) -> SidecarOutcome:
    manifest_path = sidecar.manifest_path_for(page)
    if manifest_path.exists() and not force:
        return SidecarOutcome(
            page,
            AdapterOutcome.NO_PROFILE,
            skipped_reason="a manifest is already there; pass --force to replace it",
        )
    try:
        raw = _read_bounded(page)
    except OSError as exc:
        return SidecarOutcome(
            page, AdapterOutcome.FAILED, skipped_reason=f"could not be read ({exc.strerror or exc})"
        )
    if raw is None:
        # ``stat`` for the message only, never for the decision — that was already made by a
        # bounded read. Whether the file is one byte over the limit or a thousand times it is
        # what decides between raising the limit and looking at what is in that directory.
        return SidecarOutcome(
            page,
            AdapterOutcome.FAILED,
            skipped_reason=f"is {page.stat().st_size} bytes, over the {MAX_HTML_BYTES}-byte limit",
        )
    try:
        adapted: Adaptation = adapt(raw.decode("utf-8", errors="replace"), profiles=profiles)
    except UnusablePageError as refusal:
        return SidecarOutcome(page, refusal.outcome, skipped_reason=str(refusal))
    if not _within(manifest_path, root=root):  # pragma: no cover - `_walk` already refuses this
        return SidecarOutcome(
            page, AdapterOutcome.FAILED, skipped_reason="resolves outside the root"
        )
    manifest_path.write_text(
        json.dumps(manifest_for(adapted.page, html=raw), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return SidecarOutcome(page, AdapterOutcome.ADAPTED, written=True)


def _read_bounded(page: Path) -> bytes | None:
    """``page``'s bytes, or ``None`` when it is over :data:`MAX_HTML_BYTES`.

    Reads one byte past the limit and stops, rather than reading the file and measuring what came
    back. Measuring afterwards is what the first version of this did, and it made the limit
    decorative: the whole file was already in memory by the time anything decided it was too big,
    which is precisely the allocation the limit exists to prevent.

    Bounding the read rather than trusting ``stat`` also closes the gap between asking a file how
    big it is and reading it, which is not hypothetical for a directory something else is writing.
    """
    with page.open("rb") as handle:
        raw = handle.read(MAX_HTML_BYTES + 1)
    return None if len(raw) > MAX_HTML_BYTES else raw


def _walk(directory: Path, *, root: Path) -> Iterator[Path]:
    """Every HTML file under ``directory``, in a stable order, following no symlink.

    A manifest is not filtered here and does not need to be: it is ``<page>.source.json``, whose
    suffix is ``.json``, so the extension test below has already excluded it. An
    :func:`.sidecar.is_manifest` call as well would read as a second defence and be dead code —
    unreachable, unexercised, and quietly wrong the day the extension test changed.
    """
    try:
        entries: Sequence[Path] = sorted(directory.iterdir())
    except OSError:
        return
    for entry in entries:
        if entry.name.startswith(".") or entry.is_symlink():
            continue
        if entry.is_dir():
            yield from _walk(entry, root=root)
        elif (
            entry.is_file()
            and entry.suffix.lower() in {".html", ".htm"}
            and _within(entry, root=root)
        ):
            yield entry


def _within(path: Path, *, root: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:  # pragma: no cover - resolution failure is itself a refusal
        return False
    return resolved == root or root in resolved.parents
