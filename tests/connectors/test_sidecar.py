"""The sidecar manifest, read from a real directory: what it supplies and what it is refused.

The interface's unit tests are in ``tests/test_provenance.py``. This file is the boundary where a
manifest stops being a model and starts being a file somebody else wrote, so everything here goes
through a real :class:`~manicule.connectors.filesystem.FilesystemConnector` over a real tree.

Every synthetic host is under ``.test``, which RFC 6761 §6.2 reserves, so nothing here resolves.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from manicule.connectors import sidecar
from manicule.connectors.errors import NotFoundError
from manicule.connectors.filesystem import (
    SNAPSHOT_PATH,
    FilesystemConnector,
    version_token,
)
from manicule.core.ids import content_hash
from manicule.core.provenance import PROVENANCE_KEY, Provenance
from manicule.core.sources import DocRef

if TYPE_CHECKING:
    from pathlib import Path

    from manicule.core.sources import DiscoveredDoc

CANONICAL = "https://docs.example.test/pages/123456/retry-policy"

PAGE = "<html><h1>Retry policy</h1><p>The client retries twice.</p></html>"
"""What the mirror wrote. Its filename is the whole problem: nothing about ``123456.html`` says
this is the retry policy, and generic ingestion has only the filename to cite."""

MANIFEST: dict[str, Any] = {
    "title": "Retry policy",
    "canonical_uri": CANONICAL,
    "source_id": "123456",
    "version": "7",
    "created_at": "2026-01-02T03:04:05+00:00",
    "modified_at": "2026-03-04T05:06:07+00:00",
    "content_type": "text/html",
    "section_path": ["Engineering", "Runbooks"],
    "retrieved_at": "2026-06-01T00:00:00+00:00",
}


def mirror(root: Path, *, manifest: object = MANIFEST, name: str = "123456.html") -> Path:
    """A mirrored page under ``root``, with a manifest beside it unless ``manifest`` is ``None``.

    ``manifest`` is written verbatim when it is a string, so a test can put malformed JSON on
    disk rather than something that merely fails validation — those are different failures and
    both have to be survivable.
    """
    page = root / name
    page.write_text(PAGE, encoding="utf-8")
    if manifest is not None:
        body = manifest if isinstance(manifest, str) else json.dumps(manifest)
        sidecar.manifest_path_for(page).write_text(body, encoding="utf-8")
    return page


async def fetch_one(root: Path, page: Path) -> Provenance | None:
    """The record the connector attaches to ``page``'s bytes, through the real fetch path."""
    connector = FilesystemConnector(root)
    found = [doc async for doc in connector.discover(None)]
    # Matched on the **walked path**, which is what identifies a file to this helper, and not on
    # `source_id` — which is the page's own identity whenever a manifest declares one, and is
    # therefore exactly what a test about manifests must not assume the shape of. Matched on the
    # filename rather than on a resolved absolute path: the connector resolves its root once and
    # yields paths beneath it, so the two spellings would only ever differ by a symlink the walk
    # already refuses to follow.
    ref = next(doc.ref for doc in found if str(doc.ref.metadata[SNAPSHOT_PATH]).endswith(page.name))
    raw = await connector.fetch(ref)
    return Provenance.from_metadata(raw.metadata)


async def discovered(root: Path) -> list[DiscoveredDoc]:
    return [doc async for doc in FilesystemConnector(root).discover(None)]


# --- the spec's case list --------------------------------------------------------------------


async def test_a_local_file_with_no_manifest_carries_no_record(tmp_path: Path) -> None:
    """Backward compatibility, asserted at the boundary rather than assumed.

    This is the case that must not change at all. A directory of ordinary files has to produce
    exactly what it produced before this feature existed — no record, no key in the metadata, and
    a citation that is still the filename and the ``file://`` URI, because for a file that really
    is the original that is the truthful citation.
    """
    page = mirror(tmp_path, manifest=None)
    assert await fetch_one(tmp_path, page) is None

    raw = await FilesystemConnector(tmp_path).fetch(
        next(doc.ref for doc in await discovered(tmp_path))
    )
    assert raw.metadata == {}, "a file with no manifest must not gain a metadata key"
    assert raw.uri.startswith("file://")


async def test_a_valid_manifest_supplies_the_canonical_identity(tmp_path: Path) -> None:
    """The whole point: the page's own title and address, not the mirror's filename.

    Both halves are checked in one test on purpose. A record that carried the canonical title and
    lost the snapshot's location would have replaced one incomplete citation with another, and the
    requirement is that both identities survive.
    """
    page = mirror(tmp_path)
    record = await fetch_one(tmp_path, page)

    assert record is not None
    assert record.usable
    assert record.source is not None
    assert record.source.title == "Retry policy"
    assert record.source.canonical_uri == CANONICAL
    assert record.source.source_id == "123456"
    assert record.source.version == "7"
    assert record.source.section_path == ("Engineering", "Runbooks")
    assert record.source.modified_at is not None
    assert record.source.modified_at.tzinfo is not None

    assert record.snapshot is not None
    assert record.snapshot.path == "123456.html", "the snapshot's own location must survive"
    assert record.snapshot.retrieved_at is not None
    assert record.snapshot.retrieved_at != record.source.modified_at, (
        "when the mirror was taken and when the page was edited are different facts"
    )


@pytest.mark.parametrize(
    ("manifest", "expected"),
    [
        ("{ not json at all", "is not valid JSON"),
        ('["a", "list"]', "holds a JSON list"),
        ('"a string"', "holds a JSON str"),
        ("", "is not valid JSON"),
        ('{"canonical_url": "https://docs.example.test/x"}', "canonical_url"),
        ('{"title": 17}', "title"),
        ('{"modified_at": "not a date"}', "modified_at"),
        ('{"section_path": "Engineering"}', "section_path"),
        ("{}", "at least one of title, canonical_uri or source_id"),
    ],
    ids=[
        "broken json",
        "a list",
        "a string",
        "empty",
        "a misspelled key",
        "a title that is a number",
        "an unparseable date",
        "a hierarchy that is not a list",
        "an empty object",
    ],
)
async def test_a_malformed_manifest_is_refused_with_a_reason_and_the_document_survives(
    tmp_path: Path, manifest: str, expected: str
) -> None:
    """A manifest that cannot be used never costs anybody the document beside it.

    Two claims, and the second is the one that matters. The document is still perfectly good
    bytes and is still ingestible — refusing it would turn a metadata typo into a missing
    document. And the reason is *recorded*, because the alternative is a manifest that is
    silently ignored, which presents as a citation naming a file and is indistinguishable from
    having written no manifest at all.

    The misspelled-key case is why unknown keys are refused rather than ignored:
    ``canonical_url`` for ``canonical_uri`` is the single most likely mistake, and ignoring it
    would leave the document citing its filename with nothing anywhere saying why.
    """
    page = mirror(tmp_path, manifest=manifest)
    record = await fetch_one(tmp_path, page)

    assert record is not None
    assert not record.usable
    assert record.source is None
    assert expected in record.unavailable_reason, record.unavailable_reason
    assert record.unavailable_reason.startswith("123456.html.source.json:"), (
        "the reason must name the file to look at"
    )
    assert record.snapshot is not None
    assert record.snapshot.path == "123456.html", (
        "a refused manifest must not cost the snapshot's own identity, which manicule observed "
        "and the manifest had no part in"
    )


async def test_the_manifest_title_wins_over_the_filename(tmp_path: Path) -> None:
    """Conflicting filename and manifest title: the manifest is the authority.

    ``123456.html`` is a fact about a mirror and describes nothing. That is the entire defect, so
    the resolution is not a compromise — the filename is not a fallback that sometimes wins, and
    the connector still reports it as the discovered title so nothing is lost.
    """
    page = mirror(tmp_path)
    record = await fetch_one(tmp_path, page)

    assert record is not None
    assert record.source is not None
    assert record.source.title == "Retry policy"
    assert record.source.title != page.name

    found = next(
        doc for doc in await discovered(tmp_path) if doc.ref.metadata[SNAPSHOT_PATH] == str(page)
    )
    assert found.ref.source_id == "123456", (
        "the manifest's declared identity is the document's, so moving the file does not create "
        "a second one; the path is kept beside it because fetch still has to open something"
    )
    assert found.title == "123456.html", (
        "discovery still reports what it saw; preferring the canonical title is the pipeline's "
        "decision, made in one place, and not something the connector hides here"
    )


@pytest.mark.parametrize(
    "hostile",
    [
        "javascript:window.__owned=1",
        "data:text/html;base64,PHNjcmlwdD4=",
        "file:///etc/passwd",
        "https:///no-host",
    ],
    ids=["javascript", "data", "a local path", "no host"],
)
async def test_an_invalid_canonical_uri_is_refused_and_never_reaches_a_citation(
    tmp_path: Path, hostile: str
) -> None:
    """An unusable address takes the whole record down rather than being dropped from it.

    Dropping just the URI and keeping the title would leave a document that looks as though it
    has authoritative metadata and cites nowhere — and the operator would have no signal that the
    address they wrote was thrown away. Refusing the record says so.
    """
    page = mirror(tmp_path, manifest={"title": "Retry policy", "canonical_uri": hostile})
    record = await fetch_one(tmp_path, page)

    assert record is not None
    assert not record.usable
    assert hostile not in (record.source.canonical_uri if record.source else "")
    assert "canonical_uri" in record.unavailable_reason


@pytest.mark.parametrize(
    "declared",
    [
        "../../../../etc/passwd",
        "/etc/passwd",
        "../sibling/123456.html",
        "mirror/../../../root/.ssh/id_rsa",
        "subdir/123456.html",
    ],
    ids=["deep traversal", "absolute", "one level up", "traversal mid-path", "wrong subdirectory"],
)
async def test_a_manifest_cannot_point_anywhere_but_the_file_beside_it(
    tmp_path: Path, declared: str
) -> None:
    """Attempted path traversal. The guard is a comparison, and that is the point.

    Nothing here is opened, resolved or joined to a root — the declared string is compared
    against the relative path manicule *walked to*, and anything that is not that path is a
    refusal. So the traversal attempts and the last case, an honest mirroring tool with a stale
    directory layout, are caught by one rule instead of by a "contains ``..``" check that would
    have to be kept in step with it.

    The recorded snapshot path is asserted too, because the failure worth fearing is not a
    refusal that lets the read happen — it is a record that stores the attacker's string as
    though it were where the file is.
    """
    page = mirror(tmp_path, manifest={**MANIFEST, "snapshot_path": declared})
    record = await fetch_one(tmp_path, page)

    assert record is not None
    assert not record.usable
    assert sidecar.SNAPSHOT_PATH in record.unavailable_reason
    assert record.snapshot is not None
    assert record.snapshot.path == "123456.html", (
        "the recorded location is the one manicule observed, never the one it was handed"
    )
    assert declared not in record.snapshot.path


async def test_a_manifest_may_declare_the_path_it_actually_wrote(tmp_path: Path) -> None:
    """The positive control for the traversal guard.

    Without this, a ``provenance_for`` that refused *every* declared ``snapshot_path`` would pass
    every assertion above while making the field useless — and the refusals would be testing a
    blanket rejection rather than a comparison.
    """
    nested = tmp_path / "mirror"
    nested.mkdir()
    page = mirror(nested, manifest={**MANIFEST, "snapshot_path": "mirror/123456.html"})
    record = await fetch_one(tmp_path, page)

    assert record is not None
    assert record.usable, record.unavailable_reason
    assert record.snapshot is not None
    assert record.snapshot.path == "mirror/123456.html"


async def test_a_windows_flavored_declaration_describes_the_same_file(tmp_path: Path) -> None:
    """Separators are folded, so a manifest written on Windows is not refused for saying so."""
    nested = tmp_path / "mirror"
    nested.mkdir()
    page = mirror(nested, manifest={**MANIFEST, "snapshot_path": "mirror\\123456.html"})
    record = await fetch_one(tmp_path, page)

    assert record is not None
    assert record.usable, record.unavailable_reason


async def test_a_declared_checksum_that_disagrees_is_refused(tmp_path: Path) -> None:
    """A manifest and a snapshot from different retrievals describe different documents.

    Left unchecked, this is the quiet version of the whole bug: the citation would name a version
    and a title belonging to a page whose bytes are not the ones in the index.
    """
    page = mirror(tmp_path, manifest={**MANIFEST, "snapshot_checksum": "sha256:not-these-bytes"})
    record = await fetch_one(tmp_path, page)

    assert record is not None
    assert not record.usable
    assert "snapshot_checksum" in record.unavailable_reason


async def test_a_declared_checksum_that_agrees_is_accepted(tmp_path: Path) -> None:
    """The positive control for the checksum guard, for the reason the path one has one."""
    page = mirror(
        tmp_path,
        manifest={**MANIFEST, "snapshot_checksum": content_hash(PAGE.encode("utf-8"))},
    )
    record = await fetch_one(tmp_path, page)

    assert record is not None
    assert record.usable, record.unavailable_reason


async def test_an_oversized_manifest_is_refused_before_it_is_parsed(tmp_path: Path) -> None:
    """A file named ``x.source.json`` does not get to decide how much memory ingest allocates."""
    page = mirror(tmp_path, manifest=json.dumps({**MANIFEST, "version": "7"}))
    sidecar.manifest_path_for(page).write_text(
        " " * (sidecar.MAX_MANIFEST_BYTES + 1), encoding="utf-8"
    )
    record = await fetch_one(tmp_path, page)

    assert record is not None
    assert not record.usable
    assert "over the" in record.unavailable_reason


async def test_a_manifest_that_is_not_utf8_is_refused(tmp_path: Path) -> None:
    """Undecodable bytes are a refusal with a reason, not an exception out of a walk."""
    page = mirror(tmp_path)
    sidecar.manifest_path_for(page).write_bytes(b'{"title": "\xff\xfe"}')
    record = await fetch_one(tmp_path, page)

    assert record is not None
    assert not record.usable
    assert "UTF-8" in record.unavailable_reason


# --- the connector's own obligations ---------------------------------------------------------


async def test_a_manifest_is_not_indexed_as_a_document_of_its_own(tmp_path: Path) -> None:
    """A manifest describes the document beside it; it is not a second document.

    Indexed, it would be a retrievable, contentless entry per mirrored page, citing nothing and
    diluting every result set it appeared in — and ``.json`` is in the connector's media-type
    table, so it would be parsed quite happily.
    """
    mirror(tmp_path)
    found = [doc.ref.source_id for doc in await discovered(tmp_path)]

    assert len(found) == 1
    assert not any(path.endswith(sidecar.MANIFEST_SUFFIX) for path in found)


async def test_reconciliation_and_discovery_agree_about_manifests(tmp_path: Path) -> None:
    """Both walks skip them, or a manifest is reported deleted on every single sync.

    Two passes over one tree that disagree about what a document is produce a corpus that
    oscillates: discovery indexes what reconciliation then reports gone, for ever.
    """
    mirror(tmp_path)
    connector = FilesystemConnector(tmp_path)
    still_there = [source_id async for source_id in connector.reconcile()]

    assert still_there == [doc.ref.source_id for doc in await discovered(tmp_path)]


async def test_editing_only_the_manifest_moves_the_documents_change_token(tmp_path: Path) -> None:
    """The skip that would otherwise never re-read a corrected manifest.

    A manifest holds the citable facts, so editing one to fix a title or record a new source
    version changes what a citation says while leaving the page's own bytes untouched. If the
    token came from the document alone it would not move, the pipeline would skip *before
    fetching*, and the correction would never be read — a corpus citing a version it was told
    about and then declined to look at. Nothing downstream could detect that.
    """
    page = mirror(tmp_path)
    before = version_token(page)

    manifest_file = sidecar.manifest_path_for(page)
    manifest_file.write_text(json.dumps({**MANIFEST, "version": "8"}), encoding="utf-8")
    # A rewrite of the same length would leave size unchanged and, on a coarse clock, mtime too.
    # The assertion below is about the token covering the sibling at all, so make the sibling
    # unambiguously different rather than relying on timer resolution.
    manifest_file.write_text(
        json.dumps({**MANIFEST, "version": "8", "title": "Retry policy, revised"}),
        encoding="utf-8",
    )

    assert version_token(page) != before, (
        "the document's bytes did not change and its citable facts did; a token that cannot see "
        "the manifest skips the document before fetching and never reads the correction"
    )


async def test_a_document_with_no_manifest_still_has_a_token(tmp_path: Path) -> None:
    """The ordinary case keeps working: one file, one stamp, still a usable change signal."""
    page = mirror(tmp_path, manifest=None)
    token = version_token(page)

    assert token is not None
    assert "+" not in token, "with no manifest there is only one stamp to fold in"


async def test_an_unreadable_document_still_has_no_token(tmp_path: Path) -> None:
    """``None`` means "fetch and hash to find out", and that must survive the manifest change.

    Inventing a token for a file that cannot be statted would skip a changed file for ever, and
    the sibling lookup must not have turned an honest ``None`` into a manifest-only stamp.
    """
    assert version_token(tmp_path / "does-not-exist.html") is None


async def test_the_manifest_path_is_derived_from_the_whole_filename(tmp_path: Path) -> None:
    """``page.html`` and ``page.pdf`` in one directory get a manifest each, not one between them.

    Replacing the suffix instead of appending to it would make the mapping lossy, and two
    documents would silently share — and then fight over — one description.
    """
    html = tmp_path / "page.html"
    pdf = tmp_path / "page.pdf"

    assert sidecar.manifest_path_for(html) != sidecar.manifest_path_for(pdf)
    assert sidecar.manifest_path_for(html).name == "page.html.source.json"
    assert sidecar.is_manifest(sidecar.manifest_path_for(html))
    assert not sidecar.is_manifest(html)


async def test_a_manifest_outside_the_root_is_never_reached(tmp_path: Path) -> None:
    """The connector's own root refusal still runs, and the manifest changes nothing about it.

    The record is built from the path ``fetch`` has already proved is inside the tree, so this is
    the ordering that matters: a source id that escaped the root is refused *before* anything
    looks for a manifest beside it.
    """
    root = tmp_path / "corpus"
    root.mkdir()
    outside = mirror(tmp_path, name="outside.html")

    connector = FilesystemConnector(root)
    with pytest.raises(NotFoundError, match="outside"):
        await connector.fetch(DocRef(source_id=str(outside), uri=outside.as_uri()))


async def test_the_record_reaches_the_documents_metadata_under_the_reserved_key(
    tmp_path: Path,
) -> None:
    """The record has to arrive where the pipeline and every reader look for it.

    Asserted on the raw fetched metadata rather than through ``Provenance.from_metadata``,
    because that helper would find the record under whatever key it also writes — so a test that
    only used the helper would pass with the connector and the readers agreeing on the wrong
    name, and disagree with the stored corpus.
    """
    page = mirror(tmp_path)
    connector = FilesystemConnector(tmp_path)
    ref = next(
        doc.ref
        for doc in await discovered(tmp_path)
        if doc.ref.metadata[SNAPSHOT_PATH] == str(page)
    )
    raw = await connector.fetch(ref)

    assert PROVENANCE_KEY in raw.metadata
    assert PROVENANCE_KEY == "source_provenance"
