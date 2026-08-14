"""A Confluence page's own account of itself, all the way out to a protocol response.

The connector building a record is half the claim. The half that matters to somebody writing a
durable research note is that the record survives storage, the search path and the serializer,
and arrives at an unattended client as the *same* page identity and the *same* version the
document holds — not a title and a URI reassembled from ordinary document fields, which is what
a citation had before and which says nothing a filename does not.

So these run the real connector against a synthetic instance, take the metadata it produced,
and assert on what three surfaces do with it — the application service, the MCP tool a client
actually calls, and the two of them agreeing.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from manicule.app import results as r
from manicule.app.results import SourceReference
from manicule.app.service import ApplicationService, source_reference
from manicule.config.settings import ConnectorSettings
from manicule.core.content import Document, DocumentStatus
from manicule.core.ids import content_hash, document_id
from manicule.core.provenance import PROVENANCE_KEY, Provenance
from manicule.core.retrieval import Candidate
from manicule.mcp.server import build_server
from manicule.parsers.config import ADF_MEDIA_TYPE
from tests.app.fakes import FakeBackend, make_chunk, make_document
from tests.connectors.fake_confluence import FakeConfluence, FakePage
from tests.connectors.support import cloud_config, connected, drain

WORKSPACE = "default"
CLOUD = "https://docs.example.test/wiki"
PAGE_ID = "4100"
MODIFIED = "2026-08-09T14:30:00.000+01:00"
TITLE = "Retry Policy"


async def _fetched_metadata() -> dict[str, Any]:
    """Everything the live connector would hand the pipeline for one page.

    The real connector over a synthetic instance, rather than a hand-written record: a fixture
    that wrote the provenance itself would assert that this test can build one, which is not the
    claim.
    """
    instance = FakeConfluence(
        base_url=CLOUD,
        pages=[
            FakePage(
                id=PAGE_ID,
                title=TITLE,
                space="ENG",
                version=3,
                when=MODIFIED,
                ancestors=("Platform", "Auth Service"),
            )
        ],
        spaces={"ENG": "Engineering"},
        page_size=10,
    )
    connector = await connected(instance, cloud_config(base_url=CLOUD, page_size=10))
    try:
        found = await drain(connector.discover(None))
        raw = await connector.fetch(found[0].ref)
    finally:
        await connector.teardown()
    return dict(raw.metadata)


def _document(metadata: dict[str, Any]) -> Document:
    """The stored document, keyed exactly as the pipeline keys one from this connector."""
    return Document(
        id=document_id(WORKSPACE, "wiki", PAGE_ID),
        source="wiki",
        source_id=PAGE_ID,
        uri=f"{CLOUD}/spaces/ENG/pages/{PAGE_ID}/{TITLE}",
        title=TITLE,
        content_hash=content_hash(f"{WORKSPACE}/{PAGE_ID}"),
        media_type=ADF_MEDIA_TYPE,
        status=DocumentStatus.INDEXED,
        metadata=metadata,
    )


@pytest.fixture
def stored() -> Document:
    return _document(asyncio.run(_fetched_metadata()))


def _seeded(document: Document) -> FakeBackend:
    """A backend holding the document and a retriever that will return its chunk.

    The fake retriever answers with exactly the candidates it was given, which is what makes a
    search test about the *reporting* rather than about the ranking.
    """
    made = FakeBackend()
    chunk = make_chunk(document)
    made.store.add(document, chunk)
    made.retriever_.candidates = [Candidate(chunk=chunk, score=1.0)]
    return made


@pytest.fixture
def backend(stored: Document) -> FakeBackend:
    return _seeded(stored)


def _check(diagnosis: r.Diagnosis, name: str) -> r.Check:
    """The named check, or a failure listing what was emitted instead of a bare StopIteration."""
    for check in diagnosis.checks:
        if check.name == name:
            return check
    names = ", ".join(sorted(check.name for check in diagnosis.checks))
    pytest.fail(f"doctor emitted no {name!r} check; it emitted: {names}")


def _reference(document: Document) -> SourceReference:
    reference = source_reference(document)
    assert reference is not None, "a live Confluence document must carry a source reference"
    return reference


# --- the record survives the round trip ----------------------------------------------------------


def test_a_live_page_becomes_a_document_whose_provenance_is_not_null(stored: Document) -> None:
    """The whole of the problem statement, in one assertion.

    Before this, a Confluence document had a title, a URI and a version as ordinary fields and
    ``Document.provenance`` was ``None`` — so every surface fell back to exactly what it would
    have shown for an untitled local file.
    """
    assert PROVENANCE_KEY in stored.metadata
    record = stored.provenance
    assert record is not None
    assert record.source is not None
    assert record.unavailable_reason == ""


def test_a_citation_reports_the_page_identity_the_document_holds(stored: Document) -> None:
    """Same id, same version, from the record rather than from the row beside it.

    The two agreeing is not a coincidence to be relied on — it is the property that makes a
    citation checkable against the source, and it is asserted rather than assumed.
    """
    reference = _reference(stored)

    assert reference.source_id == stored.source_id == PAGE_ID
    assert reference.version == "3"
    assert reference.title == TITLE
    assert reference.canonical_uri.startswith(CLOUD)
    assert reference.modified_at == "2026-08-09T14:30:00+01:00"
    assert reference.content_type == ADF_MEDIA_TYPE
    assert reference.section_path == ("ENG", "Platform", "Auth Service")


def test_the_three_timestamps_arrive_under_three_names(stored: Document) -> None:
    """A network fetch has no local snapshot, so two of the three are absent and say so.

    What must never happen is the absent ones being filled from the present one, or the source's
    being filled from this installation's. ``indexed_at`` is the one with a clock behind it.
    """
    reference = _reference(stored)

    assert reference.modified_at == "2026-08-09T14:30:00+01:00"
    assert reference.retrieved_at is None
    assert reference.snapshot_path == ""


def test_a_document_with_no_record_reports_no_reference() -> None:
    """The ordinary local file, unchanged. Nothing new runs for it and nothing is invented."""
    plain = Document(
        id=document_id(WORKSPACE, "local", "notes.md"),
        source="local",
        source_id="notes.md",
        uri="file:///corpus/notes.md",
        title="Notes",
        content_hash=content_hash("x"),
        media_type="text/markdown",
        status=DocumentStatus.INDEXED,
    )

    assert source_reference(plain) is None


# --- through the protocol ------------------------------------------------------------------------


def _tool(service: ApplicationService, name: str, arguments: dict[str, Any]) -> Any:
    """One MCP tool call, from a synchronous test, exactly as a client makes it."""

    async def call() -> Any:
        server = build_server(service)
        result = await server.call_tool(name, arguments)
        return result.structured_content

    return asyncio.run(call())


def _hits(payload: Any) -> list[dict[str, Any]]:
    data = cast("dict[str, Any]", payload["data"])
    found: object = data["hits"] if "hits" in data else data["results"]
    assert isinstance(found, list)
    return cast("list[dict[str, Any]]", found)


def test_the_search_tool_reports_structured_provenance_over_the_protocol(
    backend: FakeBackend, stored: Document
) -> None:
    """The protocol-level claim: a client that never sees this database gets the page identity.

    ``search`` is what an assistant calls, and its answer is the only thing that assistant can
    cite from. A null ``provenance`` there means the record exists and nothing can reach it.
    """
    service = ApplicationService(backend)
    payload = _tool(service, "search", {"query": "body", "limit": 5})

    hits = _hits(payload)
    assert hits, "the fixture document must be searchable"
    provenance = hits[0]["provenance"]
    assert provenance is not None, "structured provenance must not be null over the protocol"
    assert provenance["source_id"] == stored.source_id == PAGE_ID
    assert provenance["version"] == "3"
    assert provenance["content_type"] == ADF_MEDIA_TYPE
    assert provenance["modified_at"] == "2026-08-09T14:30:00+01:00"
    assert provenance["canonical_uri"].startswith(CLOUD)


def test_the_protocol_and_the_service_report_the_same_values(
    backend: FakeBackend, stored: Document
) -> None:
    """One builder, so a client and a local caller cannot be told different things.

    The failure this catches is a surface that assembles its own citation — which is how two
    consumers of one index come to disagree about what a document is.
    """
    service = ApplicationService(backend)
    over_protocol = _hits(_tool(service, "search", {"query": "body", "limit": 5}))[0]["provenance"]
    directly = _reference(stored)

    assert over_protocol["source_id"] == directly.source_id
    assert over_protocol["version"] == directly.version
    assert over_protocol["canonical_uri"] == directly.canonical_uri
    assert over_protocol["modified_at"] == directly.modified_at
    assert over_protocol["content_type"] == directly.content_type
    assert list(over_protocol["section_path"]) == list(directly.section_path)


def test_a_refused_record_reaches_the_protocol_as_a_reason_rather_than_as_silence() -> None:
    """Absent and refused are different facts, and an operator can only act on the second."""
    refused = Provenance(
        unavailable_reason="page 4100: the source declared metadata this index "
        "will not cite (control character)"
    )
    document = _document({PROVENANCE_KEY: refused.as_metadata_value()})
    service = ApplicationService(_seeded(document))

    provenance = _hits(_tool(service, "search", {"query": "body", "limit": 5}))[0]["provenance"]
    assert provenance is not None
    assert provenance["source_id"] == ""
    assert "will not cite" in provenance["unavailable_reason"]


# --- detecting the corpus indexed before the record existed ------------------------------------


def _configured(backend: FakeBackend, **connectors: str) -> FakeBackend:
    """Point the fake's settings at some configured sources, by type."""
    backend.settings = backend.settings.model_copy(
        update={
            "connectors": {name: ConnectorSettings(type=kind) for name, kind in connectors.items()}
        }
    )
    return backend


OLDER_PAGE_ID = "4200"
"""A second page, deliberately not the one the connector fixture produced.

Sharing an id would make the two documents one row, and the recorded one would silently replace
the unrecorded one — a fixture that reported a clean corpus by overwriting the evidence."""


def _wiki_document(backend: FakeBackend, *, record: Provenance | None = None) -> Document:
    return make_document(
        backend.settings.workspace,
        source="wiki",
        source_id=OLDER_PAGE_ID,
        provenance=record,
    )


async def test_doctor_counts_the_documents_that_predate_the_record(
    stored: Document,
) -> None:
    """The number an operator needs before deciding, and the sources it is spread across.

    Reported before anything changes, because the decision is whether to re-fetch a wiki — and
    "some documents" is not a quantity anybody can weigh against that.
    """
    backend = _configured(FakeBackend(), wiki="confluence")
    backend.store.add(_wiki_document(backend))
    backend.store.add(stored)

    check = _check(await ApplicationService(backend).doctor(), "wiki-provenance")

    assert check.state == "degraded"
    assert check.facts["missing_provenance"] == 1
    assert check.facts["sources"] == ["wiki"]
    listed = check.facts["documents"]
    assert isinstance(listed, list)
    assert listed[0] == {"source": "wiki", "source_id": OLDER_PAGE_ID}


async def test_doctor_says_a_routine_sync_will_not_perform_the_migration() -> None:
    """The sentence that stops somebody running a sync, seeing it succeed, and stopping.

    An unchanged page is neither re-enumerated nor re-fetched, so the incremental operation an
    operator would reach for first is exactly the one that does nothing here.
    """
    backend = _configured(FakeBackend(), wiki="confluence")
    backend.store.add(_wiki_document(backend))

    check = _check(await ApplicationService(backend).doctor(), "wiki-provenance")

    assert "routine incremental sync will not" in check.detail
    assert "must not be invented" in check.detail
    assert "resync" in check.remedy


async def test_doctor_leaves_a_stated_refusal_alone() -> None:
    """A refused record is a different finding, and re-fetching would refuse identically.

    Counting it as missing would send an operator to re-crawl a wiki in order to reproduce a
    diagnostic they already have.
    """
    backend = _configured(FakeBackend(), wiki="confluence")
    backend.store.add(
        _wiki_document(backend, record=Provenance(unavailable_reason="page 4100: refused"))
    )

    check = _check(await ApplicationService(backend).doctor(), "wiki-provenance")

    assert check.state == "ok"


async def test_doctor_ignores_documents_from_every_other_kind_of_source(
    stored: Document,
) -> None:
    """A local file has no canonical record to be missing, and a snapshot already writes one.

    Counting either would report a migration that has nothing to migrate, on installations that
    have never pointed at a live wiki at all.
    """
    backend = _configured(FakeBackend(), mirror="confluence-snapshot", notes="filesystem")
    backend.store.add(make_document(backend.settings.workspace, source="notes", source_id="a.md"))
    backend.store.add(make_document(backend.settings.workspace, source="mirror", source_id="7"))

    check = _check(await ApplicationService(backend).doctor(), "wiki-provenance")

    assert check.state == "ok"
    assert "no live Confluence source is configured" in check.detail


async def test_doctor_reports_a_fully_recorded_corpus_as_healthy(stored: Document) -> None:
    """And says so about the source rather than falling silent, so the check is legible."""
    backend = _configured(FakeBackend(), wiki="confluence")
    backend.store.add(stored)

    check = _check(await ApplicationService(backend).doctor(), "wiki-provenance")

    assert check.state == "ok"
    assert check.facts["sources"] == ["wiki"]


async def test_doctor_changes_nothing_when_asked_to_fix() -> None:
    """``--fix`` seeds grammars and vocabularies. It does not arm a re-crawl of a remote wiki.

    That is a deliberate limit rather than an omission: the repair here is minutes of network
    against somebody else's server, and a health command is what a person runs to un-break a
    machine.
    """
    backend = _configured(FakeBackend(), wiki="confluence")
    document = _wiki_document(backend)
    backend.store.add(document)

    check = _check(await ApplicationService(backend).doctor(fix=True), "wiki-provenance")

    assert check.state == "degraded"
    held = backend.store.documents[document.id]
    assert held.provenance is None, "the corpus is untouched by a health command"
