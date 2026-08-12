"""Mirrored Confluence pages read from disk: what they cite, and what the connector refuses.

Everything here runs over a real directory. Every fixture is synthetic and invented in this file:
no company names, no real URLs, no copied page content. Hosts are under ``.test``, which RFC 6761
§6.2 reserves, so nothing here can resolve.

Two properties get the most attention, because they are the two that would be silent if wrong.

**Identity is the page id.** A mirroring tool that reorganises its directories has not created new
pages, and a connector keyed on the path would report every document deleted and every document
new on the next sync. Several tests move a page's directory and assert its identity did not move.

**Nothing is dropped in silence.** An unusable manifest, a missing body, an ambiguous directory —
each still becomes a document carrying the reason. The failure this prevents is an export of ten
thousand pages that ingests nine thousand and reports a clean run.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from manicule.connectors.confluence import SPACE_KEY
from manicule.connectors.confluence_snapshot import (
    ANCESTOR_IDS,
    ATTACHMENTS,
    BODY_CONTENT_DROPPED,
    CONTENT_STATUS,
    LABELS,
    MANIFEST_NAME,
    UNIDENTIFIED_PREFIX,
    UNINTERPRETED_MACROS,
    ConfluenceSnapshotConnector,
)
from manicule.connectors.errors import NotFoundError
from manicule.core.lifecycle import SupportsTeardown
from manicule.core.protocols import Connector
from manicule.core.provenance import Provenance
from manicule.core.sources import DocRef
from manicule.testing import assert_connector_contract

if TYPE_CHECKING:
    from pathlib import Path

    from manicule.core.sources import DiscoveredDoc

CANONICAL = "https://docs.example.test/wiki/spaces/ENG/pages/123456/Retry-policy"

BODY = (
    "<h2>Retry policy</h2><p>The client retries twice, with backoff.</p>"
    "<p>A non-idempotent write is never retried.</p>"
)
"""A storage-format body with no macros, so the macro-warning tests are the ones that see one."""

MANIFEST: dict[str, Any] = {
    "page_id": "123456",
    "title": "Retry policy",
    "space_key": "ENG",
    "canonical_url": CANONICAL,
    "version": 7,
    "created_at": "2026-01-02T03:04:05+00:00",
    "modified_at": "2026-03-04T05:06:07+00:00",
    "ancestors": ["Runbooks"],
    "ancestor_ids": ["999"],
    "content_status": "current",
    "labels": ["runbook", "on-call"],
    "attachments": ["diagram.png"],
    "retrieved_at": "2026-06-01T00:00:00+00:00",
}


def snapshot(
    root: Path,
    *,
    manifest: object = MANIFEST,
    body: str | None = BODY,
    at: str = "ENG/123456",
    body_name: str = "body.xhtml",
) -> Path:
    """One page snapshot under ``root``.

    ``manifest`` is written verbatim when it is a string, so a test can put malformed JSON on disk
    rather than something that merely fails validation — those are different failures and both have
    to be survivable. ``body=None`` writes no body at all.
    """
    directory = root / at
    directory.mkdir(parents=True, exist_ok=True)
    if body is not None:
        (directory / body_name).write_text(body, encoding="utf-8")
    if manifest is not None:
        text = manifest if isinstance(manifest, str) else json.dumps(manifest)
        (directory / MANIFEST_NAME).write_text(text, encoding="utf-8")
    return directory


async def discovered(root: Path) -> list[DiscoveredDoc]:
    connector = ConfluenceSnapshotConnector(root)
    return [doc async for doc in connector.discover(None)]


async def record_of(root: Path, index: int = 0) -> Provenance | None:
    """The source record the connector attaches to a page's bytes, through the real fetch path."""
    connector = ConfluenceSnapshotConnector(root)
    found = await discovered(root)
    raw = await connector.fetch(found[index].ref)
    return Provenance.from_metadata(raw.metadata)


async def metadata_of(root: Path, index: int = 0) -> dict[str, Any]:
    connector = ConfluenceSnapshotConnector(root)
    found = await discovered(root)
    raw = await connector.fetch(found[index].ref)
    return dict(raw.metadata)


# --- the contract -----------------------------------------------------------------------------


@pytest.mark.contract
async def test_the_snapshot_connector_satisfies_the_connector_contract(tmp_path: Path) -> None:
    """The suite every connector passes, run against this one.

    It checks what is the same for every source: that discovery is decidable, that a watermark
    reflects a *complete* enumeration, that ``fetch`` returns the id it was asked for, and that
    reconciliation reports everything discovery just did — the last being the one that drives
    deletion, so anything it omits is deleted from the index.
    """
    snapshot(tmp_path)
    snapshot(tmp_path, manifest={**MANIFEST, "page_id": "222"}, at="ENG/222")
    connector = ConfluenceSnapshotConnector(tmp_path)
    assert isinstance(connector, Connector)
    await assert_connector_contract(connector)
    # No teardown: this connector implements none of the optional lifecycle protocols, because
    # there is no session to close, no credential to forget and no instance to disconnect from.
    assert not isinstance(connector, SupportsTeardown)


# --- identity ---------------------------------------------------------------------------------


async def test_identity_is_the_page_id_rather_than_the_directory(tmp_path: Path) -> None:
    """The property that makes an updated export replace pages instead of duplicating them."""
    snapshot(tmp_path)
    found = await discovered(tmp_path)

    assert [doc.ref.source_id for doc in found] == ["123456"]
    assert "ENG" not in found[0].ref.source_id


async def test_moving_a_page_between_directories_does_not_change_its_identity(
    tmp_path: Path,
) -> None:
    """The trap this connector exists to avoid, and the reason it does not key on the path.

    A mirroring tool that organises by space this year and by page tree next year has not created
    new pages. Keyed on the path, every document would be reported deleted and every document new
    — a full re-index, and every citation into the old corpus dangling.
    """
    original = snapshot(tmp_path, at="by-space/ENG/123456")
    before = [doc.ref.source_id for doc in await discovered(tmp_path)]
    assert before == ["123456"]

    # The old tree is really removed, which is what a re-export does. An earlier version of this
    # test left it in place and then asserted over a *set* of ids — so two snapshots of one page
    # deduplicated to one entry and it passed without anything having moved. The count below is
    # what makes the move load-bearing.
    for path in sorted(original.iterdir()):
        path.unlink()
    original.rmdir()
    (tmp_path / "by-space" / "ENG").rmdir()
    (tmp_path / "by-space").rmdir()
    snapshot(tmp_path, at="by-tree/Runbooks/Retry-policy")

    found = await discovered(tmp_path)
    assert [doc.ref.source_id for doc in found] == ["123456"], (
        "the same page under a different path is the same document, and there is exactly one of it"
    )
    record = await record_of(tmp_path)
    assert record is not None
    assert record.snapshot is not None
    assert record.snapshot.path.startswith("by-tree/"), (
        "the snapshot location did follow the move, which is the half that is supposed to change"
    )


async def test_a_page_whose_id_cannot_be_read_is_still_ingested_under_a_marked_identity(
    tmp_path: Path,
) -> None:
    """A snapshot with an unreadable manifest is a page, and dropping it would be the quiet failure.

    There is no page id to key on, so one is derived from the directory behind an explicit prefix —
    explicit, because a bare directory name in that field would be indistinguishable from a page
    whose id happens to be a word.
    """
    snapshot(tmp_path, manifest="{ not json at all")
    found = await discovered(tmp_path)

    assert len(found) == 1, "an unusable snapshot must not vanish from discovery"
    assert found[0].ref.source_id.startswith(UNIDENTIFIED_PREFIX)
    assert "ENG/123456" in found[0].ref.source_id

    record = await record_of(tmp_path)
    assert record is not None
    assert not record.usable
    assert "is not valid JSON" in record.unavailable_reason


async def test_repairing_a_manifest_moves_the_page_to_its_real_id(tmp_path: Path) -> None:
    """And the placeholder stops being reported, which is what lets reconciliation clear it.

    This is the other half of the derived identity being deliberately unstable: the fixed page
    appears under its real id, the placeholder disappears from ``reconcile``, and the ordinary
    deletion pass soft-deletes it. A stable-looking placeholder would leave two live documents for
    one page instead.
    """
    snapshot(tmp_path, manifest="{ not json at all")
    connector = ConfluenceSnapshotConnector(tmp_path)
    before = {source_id async for source_id in connector.reconcile()}
    assert any(source_id.startswith(UNIDENTIFIED_PREFIX) for source_id in before)

    snapshot(tmp_path)  # the same directory, with a manifest that parses
    after = {source_id async for source_id in ConfluenceSnapshotConnector(tmp_path).reconcile()}

    assert after == {"123456"}
    assert not any(source_id.startswith(UNIDENTIFIED_PREFIX) for source_id in after), (
        "the placeholder must stop being reported, or reconciliation can never clear it"
    )


# --- what a citation gets ---------------------------------------------------------------------


async def test_a_manifest_supplies_the_canonical_identity(tmp_path: Path) -> None:
    """The headline: the page's own title and address rather than the mirror's filename."""
    snapshot(tmp_path)
    record = await record_of(tmp_path)

    assert record is not None
    assert record.usable, record.unavailable_reason
    assert record.source is not None
    assert record.source.title == "Retry policy"
    assert record.source.canonical_uri == CANONICAL
    assert record.source.source_id == "123456"
    assert record.source.version == "7"
    assert record.source.modified_at is not None
    assert record.source.created_at is not None


async def test_the_hierarchy_is_the_space_then_the_ancestors(tmp_path: Path) -> None:
    """Coarsest first, and without the page's own title.

    The chunker appends the title itself, so a hierarchy carrying it reaches the embedder as
    emphasis nobody intended — the stutter ``breadcrumb.elements`` exists to collapse.
    """
    snapshot(tmp_path)
    record = await record_of(tmp_path)

    assert record is not None
    assert record.source is not None
    assert record.source.section_path == ("ENG", "Runbooks")
    assert "Retry policy" not in record.source.section_path


async def test_the_local_snapshot_is_recorded_relative_to_the_root(tmp_path: Path) -> None:
    """Both identities survive, and the local one does not publish this machine's layout."""
    snapshot(tmp_path)
    record = await record_of(tmp_path)

    assert record is not None
    assert record.snapshot is not None
    assert record.snapshot.path == "ENG/123456/body.xhtml"
    assert not record.snapshot.path.startswith("/")
    assert str(tmp_path) not in record.snapshot.path
    assert record.snapshot.retrieved_at is not None


async def test_confluence_specific_facts_stay_in_this_connectors_own_keys(tmp_path: Path) -> None:
    """The rule that keeps the general record product-neutral without making it useless.

    A space key, a content status, labels and attachment references are Confluence's and nobody
    else's — a filesystem mirror has nothing to put there. So they live here, under this
    connector's keys, and :class:`~manicule.core.provenance.SourceMetadata` gained no field for
    any of them. That the interface needed none is the test its design was set to pass.
    """
    snapshot(tmp_path)
    metadata = await metadata_of(tmp_path)

    assert metadata[SPACE_KEY] == "ENG"
    assert metadata[ANCESTOR_IDS] == ["999"]
    assert metadata[CONTENT_STATUS] == "current"
    assert metadata[LABELS] == ["runbook", "on-call"]

    record = Provenance.from_metadata(metadata)
    assert record is not None
    assert record.source is not None
    for absent in ("space_key", "content_status", "labels", "attachments", "ancestor_ids"):
        assert not hasattr(record.source, absent), (
            f"{absent} reached the general record, which means the general layer was shaped by "
            f"its first consumer after all"
        )


async def test_attachment_references_are_recorded_and_never_fetched(tmp_path: Path) -> None:
    """Keeping the references is what lets attachment ingestion be added without a re-crawl.

    The negative half matters as much: no attachment file exists in this fixture, and a connector
    that tried to read one would fail rather than record it.
    """
    snapshot(tmp_path)
    metadata = await metadata_of(tmp_path)

    assert metadata[ATTACHMENTS] == ["diagram.png"]
    assert not (tmp_path / "ENG" / "123456" / "diagram.png").exists()


async def test_a_manifest_may_omit_every_optional_field(tmp_path: Path) -> None:
    """A tool that knows a page's id and nothing else should say so rather than invent the rest."""
    snapshot(tmp_path, manifest={"page_id": "123456"})
    metadata = await metadata_of(tmp_path)
    record = Provenance.from_metadata(metadata)

    assert record is not None
    assert record.usable, record.unavailable_reason
    assert record.source is not None
    assert record.source.source_id == "123456"
    assert record.source.title == ""
    assert record.source.section_path == ()
    for key in (SPACE_KEY, ANCESTOR_IDS, CONTENT_STATUS, LABELS, ATTACHMENTS):
        assert key not in metadata, (
            f"{key} was written for a manifest that said nothing about it, which claims the "
            f"manifest made a statement it did not make"
        )


# --- refusals ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("manifest", "expected"),
    [
        ("{ not json at all", "is not valid JSON"),
        ('["a", "list"]', "holds a JSON list"),
        ("{}", "page_id"),
        ('{"page_id": ""}', "page_id"),
        ('{"page_id": "1", "canonical_url": "javascript:window.__owned=1"}', "canonical_uri"),
        ('{"page_id": "1", "canonical_url": "file:///etc/passwd"}', "canonical_uri"),
        ('{"page_id": "1", "title": "Retry\\u001bpolicy"}', "control character"),
        ('{"page_id": "1", "modified_at": "not a date"}', "modified_at"),
        ('{"page_id": "1", "page_title": "typo"}', "page_title"),
    ],
    ids=[
        "broken json",
        "a list",
        "no page id",
        "an empty page id",
        "a javascript canonical url",
        "a local path as the canonical url",
        "a control character in the title",
        "an unparseable date",
        "a misspelled key",
    ],
)
async def test_an_unusable_manifest_is_refused_with_a_reason_and_the_page_survives(
    tmp_path: Path, manifest: str, expected: str
) -> None:
    """A manifest that cannot be used never costs anybody the page beside it.

    Two claims, and the second is the one that matters. The body is real bytes and is still
    ingestible — refusing it would turn a metadata typo into a missing page. And the reason is
    *recorded*, because a silently ignored manifest presents as a citation naming a filename,
    which is indistinguishable from having written none.

    The misspelled-key case is why unknown keys are refused rather than ignored: ``page_title``
    for ``title`` is the likeliest mistake, and ignoring it would leave the page citing its
    filename with nothing anywhere saying why.
    """
    snapshot(tmp_path, manifest=manifest)
    record = await record_of(tmp_path)

    assert record is not None
    assert not record.usable
    assert record.source is None
    assert expected in record.unavailable_reason, record.unavailable_reason
    assert record.unavailable_reason.startswith(f"{MANIFEST_NAME}:")
    assert record.snapshot is not None, (
        "a refused manifest must not cost the snapshot's own location, which manicule observed and "
        "the manifest had no part in"
    )


@pytest.mark.parametrize(
    "declared",
    [
        "../../../../etc/passwd",
        "/etc/passwd",
        "../sibling/body.xhtml",
        "subdir/body.xhtml",
        "not-here.xhtml",
    ],
    ids=["deep traversal", "absolute", "one level up", "a subdirectory", "simply absent"],
)
async def test_a_manifest_cannot_point_at_a_body_outside_its_own_directory(
    tmp_path: Path, declared: str
) -> None:
    """Attempted path traversal. The guard is a comparison against what was found.

    Nothing here is opened, resolved or joined to a root: the declared name is compared against the
    files actually beside the manifest, and anything that is not one of them is refused. So the
    traversal attempts and the last case — a stale filename in an otherwise honest manifest — are
    caught by one rule rather than by a "contains ``..``" check that would have to be kept in step
    with it.
    """
    snapshot(tmp_path, manifest={**MANIFEST, "body_file": declared})
    record = await record_of(tmp_path)

    assert record is not None
    assert not record.usable
    assert "body_file" in record.unavailable_reason
    assert record.snapshot is not None
    assert declared not in record.snapshot.path


async def test_a_manifest_may_name_the_body_it_actually_wrote(tmp_path: Path) -> None:
    """The positive control for the traversal guard.

    Without it, a connector that refused *every* declared ``body_file`` would pass every assertion
    above while making the field useless — and the refusals would be testing a blanket rejection
    rather than a comparison.
    """
    snapshot(tmp_path, manifest={**MANIFEST, "body_file": "body.xhtml"})
    record = await record_of(tmp_path)

    assert record is not None
    assert record.usable, record.unavailable_reason


async def test_a_directory_holding_several_files_and_no_declaration_is_refused(
    tmp_path: Path,
) -> None:
    """Which file is the page would be a guess, and a guess here is a citation for other bytes."""
    directory = snapshot(tmp_path)
    (directory / "second.xhtml").write_text("<p>another</p>", encoding="utf-8")
    record = await record_of(tmp_path)

    assert record is not None
    assert not record.usable
    assert "which one is the page would be a guess" in record.unavailable_reason
    assert "body.xhtml" in record.unavailable_reason, "the reason must name the candidates"


async def test_a_manifest_with_no_body_beside_it_is_refused_rather_than_skipped(
    tmp_path: Path,
) -> None:
    """There is nothing to index, and that has to be visible rather than a page quietly missing."""
    snapshot(tmp_path, body=None)
    found = await discovered(tmp_path)
    assert len(found) == 1

    record = await record_of(tmp_path)
    assert record is not None
    assert not record.usable
    assert "no raw page representation" in record.unavailable_reason


async def test_a_declared_checksum_that_disagrees_is_refused(tmp_path: Path) -> None:
    """A manifest and a body from different exports describe different documents.

    Left unchecked this is the quiet version of the whole bug: the citation carries a title and a
    version belonging to a page whose bytes are not the ones in the index.
    """
    snapshot(tmp_path, manifest={**MANIFEST, "body_checksum": "sha256:not-these-bytes"})
    record = await record_of(tmp_path)

    assert record is not None
    assert not record.usable
    assert "body_checksum" in record.unavailable_reason


async def test_fetching_a_directory_outside_the_root_is_refused(tmp_path: Path) -> None:
    """The root bounds every read, and a ref naming somewhere else is a refusal not a read."""
    root = tmp_path / "corpus"
    root.mkdir()
    outside = snapshot(tmp_path, at="elsewhere/123456")

    connector = ConfluenceSnapshotConnector(root)
    with pytest.raises(NotFoundError, match="outside"):
        await connector.fetch(
            DocRef(
                source_id="123456",
                uri=(outside / "body.xhtml").as_uri(),
                metadata={"snapshot_directory": str(outside)},
            )
        )


# --- the walk ---------------------------------------------------------------------------------


async def test_a_page_snapshot_is_a_leaf_and_its_contents_are_not_pages(tmp_path: Path) -> None:
    """A directory of attachments beside a body must not be mistaken for pages of its own."""
    directory = snapshot(tmp_path)
    nested = directory / "attachments"
    nested.mkdir()
    (nested / MANIFEST_NAME).write_text(json.dumps({"page_id": "should-not-appear"}), "utf-8")
    (nested / "body.xhtml").write_text("<p>no</p>", encoding="utf-8")

    found = [doc.ref.source_id for doc in await discovered(tmp_path)]

    assert found == ["123456"]
    assert "should-not-appear" not in found


async def test_reconciliation_and_discovery_agree(tmp_path: Path) -> None:
    """Two walks that disagree make a corpus oscillate: indexed by one, reported gone by
    the other."""
    snapshot(tmp_path)
    snapshot(tmp_path, manifest={**MANIFEST, "page_id": "222"}, at="ENG/222")
    connector = ConfluenceSnapshotConnector(tmp_path)

    still_there = {source_id async for source_id in connector.reconcile()}

    assert still_there == {doc.ref.source_id for doc in await discovered(tmp_path)}


async def test_a_page_removed_from_a_later_export_stops_being_reported(tmp_path: Path) -> None:
    """What tombstoning is built on: reconciliation reports what exists, and nothing else.

    The connector's half of deletion. It does not delete — ``ingest.reconcile`` diffs this against
    what is stored, and refuses to act at all on a partial enumeration — but if this over-reported,
    a deleted page would be served for ever, and if it under-reported, live pages would be deleted.
    """
    snapshot(tmp_path)
    snapshot(tmp_path, manifest={**MANIFEST, "page_id": "222"}, at="ENG/222")
    connector = ConfluenceSnapshotConnector(tmp_path)
    assert {source_id async for source_id in connector.reconcile()} == {"123456", "222"}

    for path in sorted((tmp_path / "ENG" / "222").iterdir()):
        path.unlink()
    (tmp_path / "ENG" / "222").rmdir()

    assert {source_id async for source_id in connector.reconcile()} == {"123456"}


async def test_a_symlinked_directory_is_not_followed(tmp_path: Path) -> None:
    """A symlink out of the tree is the escape ``fetch`` refuses; one inside it is an endless
    walk."""
    snapshot(tmp_path)
    outside = tmp_path.parent / "outside-the-root"
    outside.mkdir(exist_ok=True)
    snapshot(outside, manifest={**MANIFEST, "page_id": "elsewhere"}, at="ENG/elsewhere")
    (tmp_path / "link").symlink_to(outside, target_is_directory=True)

    found = [doc.ref.source_id for doc in await discovered(tmp_path)]

    assert found == ["123456"]
    assert "elsewhere" not in found


# --- change detection -------------------------------------------------------------------------


async def test_editing_only_the_manifest_moves_the_change_token(tmp_path: Path) -> None:
    """The skip that would otherwise never re-read a corrected manifest.

    A manifest holds the citable facts, so correcting a version or a title changes what every
    citation says while leaving the page's bytes untouched. A token blind to the manifest would not
    move, the pipeline would skip *before fetching*, and the correction would never be read on any
    later sync — a corpus citing a version it was told about and then declined to look at.
    """
    directory = snapshot(tmp_path)
    before = (await discovered(tmp_path))[0].version_token

    # **Only the manifest**, which is the whole point and which an earlier version of this test got
    # wrong: the fixture helper rewrites both files by default, so the body's modification time
    # moved too and the token moved with it. The test passed with the manifest excluded from the
    # token entirely — found by disabling that guard and watching this stay green.
    snapshot(
        tmp_path,
        manifest={**MANIFEST, "version": 8, "title": "Retry policy, revised"},
        body=None,
    )

    after = (await discovered(tmp_path))[0].version_token
    assert (directory / "body.xhtml").read_text(encoding="utf-8") == BODY, (
        "the body must be untouched, or this is not a metadata-only change"
    )
    assert before is not None
    assert after != before, (
        "the page's bytes did not change and its citable facts did; a token that cannot see the "
        "manifest skips the page before fetching and never reads the correction"
    )


async def test_editing_only_the_body_moves_the_change_token(tmp_path: Path) -> None:
    """The other half, so the token is not merely a function of the manifest.

    Written with the same care as the case above: the manifest is deliberately *not* rewritten, so
    a token derived only from the manifest would leave this red.
    """
    directory = snapshot(tmp_path)
    before = (await discovered(tmp_path))[0].version_token
    manifest_text = (directory / MANIFEST_NAME).read_text(encoding="utf-8")

    snapshot(tmp_path, body=BODY + "<p>And a third paragraph.</p>", manifest=None)

    after = (await discovered(tmp_path))[0].version_token
    assert (directory / MANIFEST_NAME).read_text(encoding="utf-8") == manifest_text, (
        "the manifest must be untouched, or this is not a body-only change"
    )
    assert after != before


async def test_an_unchanged_snapshot_keeps_its_token(tmp_path: Path) -> None:
    """The cost control. A token that moved on every walk would re-ingest the corpus every sync."""
    snapshot(tmp_path)
    first = (await discovered(tmp_path))[0].version_token
    second = (await discovered(tmp_path))[0].version_token

    assert first is not None
    assert first == second


# --- the interim honesty this connector ships with --------------------------------------------


async def test_a_page_with_macros_says_the_parser_will_not_understand_them(
    tmp_path: Path,
) -> None:
    """The visible-degradation guard, and the reason this connector can ship before the parser.

    Storage format goes to the generic HTML parser for now, which is what the live connector does
    today — so this is no worse than shipped behaviour. It is, however, lossy, and the loss is
    recorded rather than left for somebody to discover in a search result that should have matched.
    """
    snapshot(
        tmp_path,
        body=BODY + '<ac:structured-macro ac:name="warning"><ac:rich-text-body><p>Careful.</p>'
        "</ac:rich-text-body></ac:structured-macro>",
    )
    metadata = await metadata_of(tmp_path)

    assert metadata[UNINTERPRETED_MACROS] == ["warning"]


async def test_a_macro_body_held_in_cdata_is_reported_as_content_lost(tmp_path: Path) -> None:
    """Not flattened — **gone**, and the difference is the whole point of the warning.

    An HTML parser has no CDATA in that context, so ``<![CDATA[...]]>`` reparses as a bogus comment
    and every code, noformat and Graphviz macro body is absent from the document rather than merely
    stripped of its semantics. An operator has to be told the difference between "this macro's
    formatting was lost" and "this macro's text is not in the index".
    """
    snapshot(
        tmp_path,
        body=BODY + '<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">python'
        "</ac:parameter><ac:plain-text-body><![CDATA[def retry(): ...]]>"
        "</ac:plain-text-body></ac:structured-macro>",
    )
    metadata = await metadata_of(tmp_path)
    reported = metadata[UNINTERPRETED_MACROS]

    assert isinstance(reported, list)
    assert "code" in reported
    assert BODY_CONTENT_DROPPED in reported


async def test_a_page_with_no_macros_carries_no_warning(tmp_path: Path) -> None:
    """The positive control. A warning on every page is a warning nobody reads.

    Without this, a connector that unconditionally wrote the key would pass every assertion above
    while making the signal meaningless — and an operator triaging an export would have no way to
    tell which pages actually lost content.
    """
    snapshot(tmp_path)
    metadata = await metadata_of(tmp_path)

    assert UNINTERPRETED_MACROS not in metadata
