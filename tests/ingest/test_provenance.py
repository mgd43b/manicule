"""Source metadata through the pipeline: what a citation ends up saying, and when it updates.

**Driven through** :class:`~tests.ingest.fakes.DictConnector`, **not through the connector that
reads sidecar manifests.** A dictionary is about as far from a filesystem as a source gets, so if
these pass then the pipeline honors a record from *any* connector — which is the requirement the
interface exists to meet, and the thing a test over a real directory could not establish. The
sidecar reader has its own suite in ``tests/connectors/test_sidecar.py``.

Synthetic hosts only, under the reserved ``.test`` name.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from manicule.core.content import DocumentStatus, RawDocument
from manicule.core.provenance import PROVENANCE_KEY, LocalSnapshot, Provenance, SourceMetadata
from manicule.ingest.pipeline import Change
from manicule.ingest.reindex import re_parse
from tests.fakes import MEDIA_TYPE
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

    assert document.source_id == MIRRORED, "the local artifact this was fetched by"
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
    validated, and then discarded in favor of the version it superseded — citing version 7 for
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
    find records *equal* across two runs — which also means the record has to serialize and
    deserialize stably, because a field that round-tripped to a different value would compare
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


async def test_a_connectors_own_metadata_keys_survive_beside_the_record() -> None:
    """The generic record does not displace a connector's own vocabulary, and must not.

    **This is the contract that keeps the interface product-neutral without making it useless.**
    A record carries what *every* source has — a title, an address, an identity, a version, a
    hierarchy. Anything a particular product means and others do not has no business in it: a
    wiki's space key, a page's content status, its labels, its attachment references. Those stay
    in the connector's own metadata keys, which is where they already live
    (``connectors/confluence.py`` sets ``space_key`` today).

    So the two have to coexist, and the assignment in ``_store_record`` writes one key rather
    than replacing the mapping. If it ever replaced it, every connector-specific fact in the
    corpus would vanish the moment its connector started supplying a record — and the symptom
    would be a citation that looked complete while having quietly lost the field a reader of that
    particular product actually navigates by.
    """
    connector = a_connector(metadata=a_record())
    connector.metadata[MIRRORED] = {
        **a_record(),
        "space_key": "ENG",
        "content_status": "current",
        "labels": ["runbook", "on-call"],
        "attachments": [{"id": "att-1", "filename": "diagram.png"}],
    }
    _, document = await ingest_once(connector)

    assert document.provenance is not None
    assert document.provenance.source is not None
    assert document.provenance.source.title == "Retry policy"
    # The connector's own keys, untouched and beside the record rather than inside it.
    assert document.metadata["space_key"] == "ENG"
    assert document.metadata["content_status"] == "current"
    assert document.metadata["labels"] == ["runbook", "on-call"]
    assert document.metadata["attachments"] == [{"id": "att-1", "filename": "diagram.png"}]
    # And none of them leaked into the generic vocabulary, which has nowhere to put them.
    assert not hasattr(document.provenance.source, "space_key")


async def test_a_re_parse_from_retained_bytes_keeps_the_record_and_the_citation() -> None:
    """The raw snapshot is the authority; text and vectors are replaceable derivatives.

    Re-parsing is rung 3 of ``docs/storage.md`` §1's blast-radius ladder — the retained bytes are
    read back and run through the current chain, with no network and no re-fetch. The record has
    to survive that, because it is a fact about the *document* rather than about any particular
    derivation of it: a parser upgrade must not be able to demote a citation back to its filename.

    That relationship is what makes "the raw snapshot remains the source of authority" more than
    a slogan, and it is stronger than a stored checksum: ``original_ref`` names the bytes,
    ``parse_fp`` names the process that produced this text from them, and the citation's identity
    is independent of both.

    **The record reaches a re-parse by two independent routes, and this test pins neither of them
    on its own.** Found by disabling them: ``reindex.re_parse`` copies ``document.metadata`` onto
    the rebuilt ``RawDocument``, *and* ``_store_record``'s merge carries ``existing.metadata`` —
    so blanking either one leaves this green, and only blanking both turns it red. Written down
    because the consequence is a trap: somebody removing one route would see this test pass and
    reasonably conclude the route they deleted was dead. It was not; it was redundant. The
    property is what is asserted here, deliberately, and the redundancy is why it holds.
    """
    blobs = fakes.MemoryBlobs()
    pipeline, store, _ = build(blobs=blobs)
    await pipeline.run(a_connector(metadata=a_record()))
    document = await store.find_document("memory", MIRRORED)
    assert document is not None
    assert document.original_ref, "the bytes must have been retained for a re-parse to be possible"

    report = await re_parse([document], pipeline=pipeline, blobs=blobs)

    assert report.documents == 1
    after = await store.find_document("memory", MIRRORED)
    assert after is not None
    assert after.provenance is not None, "a re-parse must not discard the source record"
    assert after.provenance.source is not None
    assert after.provenance.source.version == "7"
    assert after.title == "Retry policy", "nor demote the citation back to a local name"
    assert after.uri == CANONICAL


async def test_content_and_metadata_changes_are_detected_independently() -> None:
    """Two axes, asked separately, because they cost different things to repair.

    A body edited with its manifest untouched needs a re-parse; a manifest corrected over an
    unchanged body needs only the record rewritten. Collapsed into one boolean, "should this be
    re-ingested" is answerable and "why" is not — and *why* is the question somebody watching an
    unexpected re-ingest of a whole corpus actually has.

    The classifier is the one the skip decision is expressed in terms of, so it cannot rot into a
    description of a decision made elsewhere: an empty set **is** the skip condition.
    """
    connector = a_connector(metadata=a_record(version="7"))
    pipeline, store, _ = build()
    await pipeline.run(connector)
    stored = next(iter(store.documents.values()))
    digest = stored.content_hash

    # Nothing moved.
    assert pipeline.changes_since(stored, digest, _raw(a_record(version="7"))) == frozenset()

    # The manifest declares a new version; the bytes are identical.
    assert pipeline.changes_since(stored, digest, _raw(a_record(version="8"))) == {Change.METADATA}

    # The bytes moved; the record did not.
    assert pipeline.changes_since(stored, "another-digest", _raw(a_record(version="7"))) == {
        Change.CONTENT
    }

    # Both, and reported as both rather than as one.
    assert pipeline.changes_since(stored, "another-digest", _raw(a_record(version="8"))) == {
        Change.CONTENT,
        Change.METADATA,
    }


def _raw(metadata: Metadata, *, media_type: str = MEDIA_TYPE) -> RawDocument:
    """A fetched document carrying ``metadata``, for asking the classifier a question."""
    return RawDocument(
        source_id=MIRRORED,
        uri=f"memory://{MIRRORED}",
        media_type=media_type,
        content="The client retries twice.\n",
        metadata=metadata,
    )


# --- re-routing ------------------------------------------------------------------------------

RE_ROUTED = "text/x-fake-storage"
"""A second media type for the same bytes, standing in for a source that has learned to
declare what it was serving all along."""


async def test_a_source_that_declares_a_new_media_type_reports_it_as_its_own_axis() -> None:
    """A re-route is not a lineage change, and reporting it as one would be a lie.

    Lineage asks whether the parser that ran has since changed *version*, and it answers by
    looking up ``parser_used`` — so on a re-route it compares the old parser against itself,
    finds it unchanged, and says the document is current. Nothing else notices either: the bytes
    are identical and the source record has not moved.
    """
    connector = a_connector(metadata=a_record(version="7"))
    pipeline, store, _ = build()
    await pipeline.run(connector)
    stored = next(iter(store.documents.values()))
    digest = stored.content_hash

    # The same bytes, the same record, declared as something else.
    re_routed = _raw(a_record(version="7"), media_type=RE_ROUTED)
    assert pipeline.changes_since(stored, digest, re_routed) == {Change.ROUTING}

    # And it is reported *beside* the others rather than instead of them.
    assert pipeline.changes_since(
        stored, "another-digest", _raw(a_record(version="8"), media_type=RE_ROUTED)
    ) == {Change.CONTENT, Change.METADATA, Change.ROUTING}


async def test_a_fetch_that_declares_nothing_is_not_treated_as_a_re_route() -> None:
    """Silence is agreement, not ignorance.

    A caller with no fetch in hand has said nothing about routing. Treating that as a change
    would re-ingest every such corpus on every sync to learn nothing — the same rule the source
    record follows one field along.
    """
    connector = a_connector(metadata=a_record(version="7"))
    pipeline, store, _ = build()
    await pipeline.run(connector)
    stored = next(iter(store.documents.values()))

    assert pipeline.changes_since(stored, stored.content_hash, None) == frozenset()


async def test_a_re_routed_document_is_not_skipped_on_an_unchanged_version_token() -> None:
    """The level-1 half, which is the half that would actually have bitten.

    A page nobody has edited reports the version token it reported last time, so level 1 answers
    and **the fetch never happens** — meaning a routing check placed only in ``changes_since``
    never runs at all on precisely the connectors that are best behaved. It is the same trap
    ``_parse_lineage_is_current`` exists for, one axis along, and the consequence is a corpus
    holding text from a parser nothing routes to any more, for ever, with nothing reported.

    Asserted on the fetch rather than on the classifier: whether the pipeline *went and looked*
    is the observable behavior, and a test of the private predicate would pass while the
    document was still being skipped.
    """
    connector = a_connector(metadata=a_record(version="7"))
    connector.tokens[MIRRORED] = "unmoved"
    pipeline, store, _ = build(
        parsers={"lines": fakes.LineParser(), "storage": fakes.LineParser()},
        routes={MEDIA_TYPE: ("lines",), RE_ROUTED: ("storage",)},
    )
    await pipeline.run(connector)
    assert connector.fetches == [MIRRORED]
    first = next(iter(store.documents.values()))
    assert first.metadata.get("parser_used") == "lines"

    # Nothing about the page has moved, so a second run must skip it.
    await pipeline.run(connector)
    assert connector.fetches == [MIRRORED], "an unchanged page must not be re-fetched"

    # Now the source declares the type it was serving all along. The token is still 'unmoved'.
    connector.media_types[MIRRORED] = RE_ROUTED
    await pipeline.run(connector)

    assert connector.fetches == [MIRRORED, MIRRORED], (
        "a re-routed document was skipped without being fetched, so its text stays whatever the "
        "old parser made of it for ever"
    )
    after = next(iter(store.documents.values()))
    assert after.media_type == RE_ROUTED
    assert after.metadata.get("parser_used") == "storage", (
        "the point of noticing is that the new parser actually gets to read it"
    )


async def test_a_failed_parse_keeps_the_indexed_revision_retryable_on_the_next_sync() -> None:
    """A failed attempt learned only its error, not that the new source revision was indexed.

    Adopting the fetched token while retaining the old chunks makes the next healthy sync skip at
    level 1 forever. The same rule applies to every source fact beside the token: failed bytes do
    not get to change the retained-byte pointer, routing, provenance or connector metadata of the
    indexed revision they failed to replace.
    """
    blobs = fakes.MemoryBlobs()
    connector = a_connector(metadata={**a_record(version="7"), "content_status": "current"})
    healthy, store, vectors = build(blobs=blobs)
    await healthy.run(connector)
    first = await store.find_document("memory", MIRRORED)
    assert first is not None

    connector.documents[MIRRORED] = "The client retries three times.\n"
    connector.tokens[MIRRORED] = "v8"
    connector.media_types[MIRRORED] = RE_ROUTED
    connector.metadata[MIRRORED] = {
        **a_record(version="8", title="Retry policy, revised"),
        "content_status": "archived",
    }
    broken, _, _ = build(
        store=store,
        vectors=vectors,
        blobs=blobs,
        parsers={"lines": fakes.ExplodingParser()},
    )

    await broken.run(connector)

    failed = await store.find_document("memory", MIRRORED)
    assert failed is not None
    assert failed.revision == first.revision, (
        "the failed bytes' token, hash, retained-byte pointer or provenance was adopted"
    )
    assert (failed.uri, failed.title, failed.media_type) == (
        first.uri,
        first.title,
        first.media_type,
    ), "the failed bytes changed a reader-visible source fact"
    assert failed.metadata["content_status"] == "current"
    error = failed.metadata["last_ingest_error"]
    assert isinstance(error, dict)
    assert error["stage"] == "parse"

    connector.fetches.clear()
    report = await healthy.run(connector)

    assert connector.fetches == [MIRRORED], (
        "the healthy retry trusted the failed bytes' version token and skipped the fetch"
    )
    assert report.indexed == 1
    repaired = await store.find_document("memory", MIRRORED)
    assert repaired is not None
    assert repaired.content_hash != first.content_hash
    assert repaired.original_ref != first.original_ref
    assert repaired.media_type == RE_ROUTED
    assert repaired.provenance == Provenance.from_metadata(connector.metadata[MIRRORED])
    assert [chunk.text for chunk in store.chunks[repaired.id]] == [
        "The client retries three times."
    ]


async def test_a_connectors_mutable_facts_refresh_rather_than_freezing() -> None:
    """Labels and a content status move at the source; the stored copy must not win.

    **The second instance of one bug, and the reason the general rule changed rather than gaining
    another exception.** The merge order used to put ``existing.metadata`` over ``raw.metadata``, so
    anything a connector re-derives on every fetch froze at whatever was stored first. bug4 fixed
    that narrowly, for the source record alone, by assigning it after the merge — and the narrow fix
    is what made the remaining breakage *invisible*, because the record's version refreshed while
    everything beside it did not.

    That asymmetry is the failure, not the staleness. A page archived and deprecated at the source
    reads as ``current`` for ever **while its version number says the sync is up to date**, so an
    operator filtering ``content_status == "archived"`` to exclude retired runbooks gets one back,
    ranked and plausible, carrying evidence of its own freshness.
    """
    connector = a_connector(metadata=a_record(version="7"))
    connector.metadata[MIRRORED] = {
        **a_record(version="7"),
        "labels": ["runbook"],
        "content_status": "current",
    }
    store, first = await ingest_once(connector)
    assert first.metadata["labels"] == ["runbook"]
    assert first.metadata["content_status"] == "current"

    # The page is re-labeled and archived and its version bumps. Its bytes do not move, which is
    # the ordinary shape of a metadata edit.
    connector.metadata[MIRRORED] = {
        **a_record(version="8"),
        "labels": ["runbook", "deprecated"],
        "content_status": "archived",
    }
    connector.tokens[MIRRORED] = "v8"
    _, second = await ingest_once(connector, store)

    assert second.provenance is not None
    assert second.provenance.source is not None
    assert second.provenance.source.version == "8", "the record refreshed, as it already did"
    assert second.metadata["labels"] == ["runbook", "deprecated"], (
        "labels froze at the stored copy while the version refreshed beside them — a document that "
        "looks freshly synced and is not"
    )
    assert second.metadata["content_status"] == "archived", (
        "an archived page still reads as current, so a filter meant to exclude it returns it"
    )


async def test_accumulated_state_no_connector_supplies_survives_a_re_ingest() -> None:
    """The property the old order was protecting, preserved without a special case.

    A key absent from ``raw.metadata`` overrides nothing, so per-document state written by
    ``IngestStore.annotate`` — ``last_ingest_error``, ``last_after_store_error``, which no connector
    supplies — is untouched by the reorder. This is the assertion that would have sunk the change if
    it failed, so it is here rather than in the reasoning.
    """
    connector = a_connector(metadata=a_record())
    store, first = await ingest_once(connector)
    await store.annotate(first.id, {"last_ingest_error": {"stage": "parse", "detail": "earlier"}})

    connector.documents[MIRRORED] = "The client retries three times.\n"
    _, second = await ingest_once(connector, store)

    assert second.metadata["last_ingest_error"] == {"stage": "parse", "detail": "earlier"}, (
        "accumulated state a connector never supplies was erased by the reorder"
    )


async def test_the_parse_stage_still_beats_the_connector() -> None:
    """``result.metadata`` is highest precedence, and the reorder must not have disturbed that.

    What the parse stage concluded about the bytes outranks what the source guessed about them: a
    connector's declared ``media_type`` or title is a hint, and the chain that actually read the
    document is the authority. The reorder moved the *lower* two layers past each other and left
    this one on top.
    """
    connector = a_connector(metadata={**a_record(), "parser_used": "a-connector-lie"})
    _, document = await ingest_once(connector)

    assert document.metadata["parser_used"] == "lines", (
        "a connector overrode what the parse stage concluded; result.metadata is no longer highest"
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
