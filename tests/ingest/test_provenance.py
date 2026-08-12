"""Source metadata through the pipeline: what a citation ends up saying, and when it updates.

**Driven through** :class:`~tests.ingest.fakes.DictConnector`, **not through the connector that
reads sidecar manifests.** A dictionary is about as far from a filesystem as a source gets, so if
these pass then the pipeline honours a record from *any* connector — which is the requirement the
interface exists to meet, and the thing a test over a real directory could not establish. The
sidecar reader has its own suite in ``tests/connectors/test_sidecar.py``.

Synthetic hosts only, under the reserved ``.test`` name.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from manicule.core.content import DocumentStatus
from manicule.core.provenance import PROVENANCE_KEY, LocalSnapshot, Provenance, SourceMetadata
from tests.ingest import fakes
from tests.ingest.test_pipeline import build

if TYPE_CHECKING:
    from manicule.core.content import Document, Metadata

MIRRORED = "123456.html"
CANONICAL = "https://docs.example.test/pages/123456/retry-policy"


def a_record(
    *,
    title: str = "Retry policy",
    version: str = "7",
    section_path: tuple[str, ...] = ("Engineering", "Runbooks"),
    canonical_uri: str = CANONICAL,
) -> Metadata:
    """A connector-supplied record, in the shape the pipeline reads it out of."""
    record = Provenance(
        source=SourceMetadata(
            title=title,
            canonical_uri=canonical_uri,
            source_id="123456",
            version=version,
            modified_at=datetime(2026, 3, 4, 5, 6, 7, tzinfo=UTC),
            section_path=section_path,
        ),
        snapshot=LocalSnapshot(
            path=f"mirror/{MIRRORED}", retrieved_at=datetime(2026, 6, 1, tzinfo=UTC)
        ),
    )
    return {PROVENANCE_KEY: record.as_metadata_value()}


async def ingest_once(
    connector: fakes.DictConnector,
    store: fakes.MemoryIngestStore | None = None,
) -> tuple[fakes.MemoryIngestStore, Document]:
    """One run, returning the store and the single document it wrote."""
    pipeline, store, _ = build(store=store)
    await pipeline.run(connector)
    documents = list(store.documents.values())
    assert len(documents) == 1, f"expected one document, got {len(documents)}"
    return store, documents[0]


def a_connector(*, metadata: Metadata | None = None, body: str = "The client retries twice.\n"):  # noqa: ANN201 - the fake's own type, inferred
    connector = fakes.DictConnector({MIRRORED: body})
    if metadata is not None:
        connector.metadata[MIRRORED] = metadata
    return connector


# --- what a citation says --------------------------------------------------------------------


async def test_a_document_with_no_record_cites_exactly_what_it_cited_before() -> None:
    """The backward-compatible path through the pipeline itself.

    A source that supplies no record must produce a byte-identical document to the one it
    produced before this feature existed — same title, same URI, and **no key** in the stored
    metadata. A null under the key would be a stored value where there was none, and every
    reader would then have two absent cases to handle instead of one.
    """
    _, document = await ingest_once(a_connector())

    assert document.uri == f"memory://{MIRRORED}"
    assert PROVENANCE_KEY not in document.metadata
    assert document.provenance is None


async def test_a_record_replaces_the_filename_and_the_local_uri_in_the_citation() -> None:
    """The defect, fixed: the page's own title and address rather than the mirror's filename.

    Written into ``documents.title`` and ``documents.uri`` deliberately, rather than left for
    each renderer to prefer. Every citation surface already reads those two columns, so this is
    the one change that makes the command line, the MCP tool, the HTTP payload, the browser page
    *and the slot label the model itself is shown* all correct at once — including a surface added
    next year that nobody remembers to teach.
    """
    _, document = await ingest_once(a_connector(metadata=a_record()))

    assert document.title == "Retry policy"
    assert document.uri == CANONICAL


async def test_the_local_snapshot_identity_survives_being_superseded() -> None:
    """Preferring the canonical identity must not cost the local one. Both, or neither is trusted.

    Each of these is the answer to a different question, and an audit needs all of them: which
    file was read (``source_id``), what its bytes were (``content_hash``), where it sits under the
    ingestion root (the record's snapshot half), and what the document actually is (the canonical
    half). Losing any one of them makes a result irreproducible in a way nothing reports.
    """
    _, document = await ingest_once(a_connector(metadata=a_record()))
    record = document.provenance

    assert document.source_id == MIRRORED, "the local artefact this was fetched by"
    assert document.content_hash, "the digest of the bytes actually read"
    assert record is not None
    assert record.snapshot is not None
    assert record.snapshot.path == f"mirror/{MIRRORED}"
    assert record.source is not None
    assert record.source.canonical_uri == CANONICAL
    assert record.snapshot.path != record.source.canonical_uri


async def test_a_record_that_supplies_no_title_leaves_the_discovered_one_alone() -> None:
    """Preference, not replacement. An absent canonical title is not an instruction to blank one.

    A manifest may legitimately know a page's URL and not its title. Overwriting with the empty
    string would leave a document with no title at all, which is worse than the filename it
    started with.
    """
    _, document = await ingest_once(a_connector(metadata=a_record(title="")))

    assert document.title == "", (
        "the fake connector reports no discovered title, so there is nothing to fall back to — "
        "what matters is that the record did not overwrite with a canonical blank"
    )
    assert document.uri == CANONICAL, "the address it did supply still wins"


# --- an update, which is where the interesting bug was ---------------------------------------


async def test_a_higher_version_under_the_same_source_id_replaces_the_stored_record() -> None:
    """The same document, re-ingested after its manifest declared a new version.

    **This is the test that found a real bug.** ``_store_record`` merged the layers as
    ``{**raw.metadata, **existing.metadata, **result.metadata}``, so a freshly fetched record lost
    to the one already stored. The document would be re-fetched, the new record read and
    validated, and then discarded in favour of the version it superseded — citing version 7 for
    ever while the source was on 8, with nothing anywhere looking wrong. The fix assigns the
    record after the merge, because it is this run's conclusion rather than accumulated state.

    One document throughout, which is the other half of the claim: an update under an unchanged
    source id is the same document, not a second one.
    """
    connector = a_connector(metadata=a_record(version="7"))
    store, first = await ingest_once(connector)
    assert first.provenance is not None
    assert first.provenance.source is not None
    assert first.provenance.source.version == "7"

    # The page's own bytes are untouched; only its declared version and title moved. The token is
    # moved explicitly because that is what a real source does when metadata alone changes, and
    # leaving it would make the pipeline skip before fetching and test nothing.
    connector.metadata[MIRRORED] = a_record(version="8", title="Retry policy, revised")
    connector.tokens[MIRRORED] = "v8"
    store, second = await ingest_once(connector, store)

    assert second.content_hash == first.content_hash, (
        "the point of this case is that the *bytes did not move* — if they had, the ordinary "
        "content-hash path would carry the new record and neither guard would be exercised"
    )

    assert second.id == first.id, "an update under one source id is one document"
    assert second.provenance is not None
    assert second.provenance.source is not None
    assert second.provenance.source.version == "8", (
        "the stored record must be the one just fetched, not the one it superseded"
    )
    assert second.provenance.source.source_id == "123456", "the immutable identity is unchanged"
    assert second.title == "Retry policy, revised"


async def test_re_ingesting_an_unchanged_document_with_an_unchanged_record_still_skips() -> None:
    """The cost control on the guard above, and the reason it compares rather than invalidates.

    Making "has a record" mean "always re-parse" would be a correct-looking change that quietly
    turns every sync of a mirrored corpus into a full re-parse and re-embed. The comparison has to
    find records *equal* across two runs — which also means the record has to serialise and
    deserialise stably, because a field that round-tripped to a different value would compare
    unequal every time and produce exactly that runaway.
    """
    connector = a_connector(metadata=a_record())
    store, _ = await ingest_once(connector)
    fetches = len(connector.fetches)

    _, again = await ingest_once(connector, store)

    assert again.provenance is not None
    assert len(connector.fetches) == fetches, (
        "the version token has not moved, so this document should not even be fetched"
    )


async def test_a_connector_that_stops_supplying_a_record_does_not_erase_the_stored_one() -> None:
    """The other side of the precedence fix, and the reason it is an assignment and not a merge.

    Clearing the record whenever a fetch happened to arrive without one would make an unrelated
    re-ingest — a re-parse from retained bytes, a connector that supplies metadata only on some
    paths — silently demote a document's citation back to its filename. The rule is "a fresh
    record wins", not "the absence of one wins".
    """
    connector = a_connector(metadata=a_record())
    store, first = await ingest_once(connector)
    assert first.provenance is not None

    connector.metadata.pop(MIRRORED)
    connector.documents[MIRRORED] = "The client retries three times.\n"
    store, second = await ingest_once(connector, store)

    assert second.provenance is not None, "a fetch without a record must not clear one"
    assert second.provenance.source is not None
    assert second.provenance.source.canonical_uri == CANONICAL


async def test_a_refused_record_is_stored_so_the_refusal_is_visible() -> None:
    """A manifest that could not be used has to reach the corpus as a stated reason.

    Otherwise the only symptom is a citation naming a file, which is exactly what a document with
    no manifest looks like — so the operator cannot tell "I wrote no manifest" from "the one I
    wrote is broken". ``document list`` is where they will look, and this is what puts it there.
    """
    refused = Provenance(
        snapshot=LocalSnapshot(path=f"mirror/{MIRRORED}"),
        unavailable_reason=f"{MIRRORED}.source.json: is not valid JSON",
    )
    _, document = await ingest_once(
        a_connector(metadata={PROVENANCE_KEY: refused.as_metadata_value()})
    )

    assert document.provenance is not None
    assert not document.provenance.usable
    assert "is not valid JSON" in document.provenance.unavailable_reason
    assert document.uri == f"memory://{MIRRORED}", (
        "a refused record must leave the local citation in place rather than blanking it"
    )
    assert document.status is DocumentStatus.INDEXED, (
        "an unusable manifest never costs anybody the document beside it"
    )


async def test_a_hostile_stored_record_cannot_take_over_the_citation() -> None:
    """A connector that supplies an unvalidatable record changes nothing about the citation.

    The write path is not the only way bytes get under that key — a plugin middleware, a restored
    backup, a hand-edited row. Since the pipeline reads the record back through the validating
    accessor before preferring anything from it, a ``javascript:`` canonical URI cannot become
    ``documents.uri``, and the document keeps the local address it can defend.
    """
    _, document = await ingest_once(
        a_connector(
            metadata={PROVENANCE_KEY: {"source": {"canonical_uri": "javascript:window.__owned=1"}}}
        )
    )

    assert document.uri == f"memory://{MIRRORED}"
    assert "javascript:" not in document.uri
    assert document.provenance is None, "an unvalidatable record reads as no record"
