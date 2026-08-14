"""What a live Confluence document says about itself, and where every word of it came from.

A provenance record is a claim made in the publisher's voice. Everything in it is rendered into
a citation that a reader takes as the source's own account — so the failure that matters here is
not an absent field but a **present** one that manicule worked out rather than read. A record
assembled half from responses and half from inference is indistinguishable, at the point it is
read, from one that is entirely authoritative.

So these tests are mostly about provenance of the provenance: which response each field came out
of, that the two deployments are parsed as the two different shapes they actually are, and that
a field the API did not supply is absent rather than filled from the nearest plausible clock.

**Three timestamps, and two of them are not reachable from the connector at all.**
``modified_at`` is the source's. ``retrieved_at`` is a local snapshot's, which a network fetch
does not have. ``indexed_at`` is this installation's. The one this module guards is the first,
because it is the one with a plausible-looking substitute one import away.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from manicule.connectors.confluence import MODIFIED_AT, VERSION_TOKEN
from manicule.core.provenance import PROVENANCE_KEY, Provenance, SourceMetadata
from manicule.core.sources import DocRef
from manicule.parsers.config import ADF_MEDIA_TYPE, CONFLUENCE_MEDIA_TYPE
from tests.connectors.fake_confluence import FakeAttachment, FakeConfluence, FakePage
from tests.connectors.support import cloud_config, connected, drain, server_config

CLOUD = "https://docs.example.test/wiki"
"""Cloud fixtures. Synthetic, and the only base URL a Cloud test here uses."""

SERVER = "https://docs.example.test/confluence"
"""Server and Data Center, served from a context path, because that is the shape whose links
resolve differently and the provenance record carries a resolved link."""

PAGE_ID = "4100"
MODIFIED = "2026-08-09T14:30:00.000+01:00"
CREATED = "2026-01-05T09:00:00.000+01:00"


def _page(**overrides: object) -> FakePage:
    settings: dict[str, object] = {
        "id": PAGE_ID,
        "title": "Retry Policy",
        "space": "ENG",
        "version": 3,
        "when": MODIFIED,
        "ancestors": ("Platform", "Auth Service"),
    }
    settings.update(overrides)
    return FakePage(**settings)  # type: ignore[arg-type]


def _cloud(**overrides: object) -> FakeConfluence:
    settings: dict[str, object] = {
        "base_url": CLOUD,
        "pages": [_page(created=CREATED)],
        "spaces": {"ENG": "Engineering"},
        "page_size": 10,
    }
    settings.update(overrides)
    return FakeConfluence(**settings)  # type: ignore[arg-type]


def _server(**overrides: object) -> FakeConfluence:
    settings: dict[str, object] = {
        "base_url": SERVER,
        "pages": [_page()],
        "spaces": {"ENG": "Engineering"},
        "page_size": 10,
    }
    settings.update(overrides)
    return FakeConfluence(**settings)  # type: ignore[arg-type]


async def _record(
    instance: FakeConfluence, *, server: bool = False, source_id: str = PAGE_ID, **overrides: object
) -> Provenance | None:
    """Discover, fetch, and read back the record exactly as the pipeline would store it."""
    config = (
        server_config(instance.base_url, page_size=10, **overrides)
        if server
        else cloud_config(base_url=instance.base_url, page_size=10, **overrides)
    )
    connector = await connected(instance, config)
    try:
        found = await drain(connector.discover(None))
        ref = next(document.ref for document in found if document.source_id == source_id)
        raw = await connector.fetch(ref)
    finally:
        await connector.teardown()
    return Provenance.from_metadata(raw.metadata)


def _source(record: Provenance | None) -> SourceMetadata:
    assert record is not None, "a fetched page must carry a provenance record"
    assert record.source is not None, f"expected a source record, got {record.unavailable_reason!r}"
    return record.source


# --- the record both deployments produce -------------------------------------------------------


async def test_a_cloud_page_records_every_field_the_v2_response_supplies() -> None:
    """Each field named beside the response key it came out of, which is the whole contract."""
    record = _source(await _record(_cloud()))

    assert record.title == "Retry Policy"
    assert record.canonical_uri == f"{CLOUD}/spaces/ENG/pages/{PAGE_ID}/Retry Policy"
    assert record.source_id == PAGE_ID
    assert record.version == "3"
    # `version.createdAt` is when the current version was made: the page's last edit.
    assert record.modified_at == datetime.fromisoformat(MODIFIED)
    # The page's own top-level `createdAt`, which is a different field at a different level.
    assert record.created_at == datetime.fromisoformat(CREATED)
    assert record.content_type == ADF_MEDIA_TYPE
    assert record.section_path == ("ENG", "Platform", "Auth Service")


async def test_a_server_page_records_the_fields_its_own_response_shape_supplies() -> None:
    """Server and Data Center spell the modification time ``version.when``, and say no more.

    The creation date lives under a ``history`` expansion this connector does not request, so
    it is absent — which is the correct record rather than an incomplete one. Filling it from
    the modification time would assert that the page was created when it was last edited.
    """
    record = _source(await _record(_server(), server=True))

    assert record.source_id == PAGE_ID
    assert record.version == "3"
    assert record.modified_at == datetime.fromisoformat(MODIFIED)
    assert record.created_at is None
    assert record.content_type == CONFLUENCE_MEDIA_TYPE
    assert record.canonical_uri == f"{SERVER}/spaces/ENG/pages/{PAGE_ID}/Retry Policy"


async def test_the_two_deployments_are_parsed_as_the_two_shapes_they_are() -> None:
    """One assertion for the thing a shared parser would get wrong in exactly one direction.

    Reading ``version.when`` on Cloud, or ``version.createdAt`` on Server, finds nothing and
    produces a record with no modification time — a silent hole rather than an error, on the
    deployment nobody happened to test.
    """
    assert _source(await _record(_cloud())).modified_at is not None
    assert _source(await _record(_server(), server=True)).modified_at is not None


async def test_the_hierarchy_stops_short_of_the_page_own_title() -> None:
    """The chunker appends the title itself, and a path carrying it twice is emphasis."""
    record = _source(await _record(_cloud()))

    assert "Retry Policy" not in record.section_path
    assert record.section_path[0] == "ENG"


# --- the timestamp that must never be substituted ----------------------------------------------


async def test_a_response_with_no_timestamp_leaves_the_field_absent() -> None:
    """Absent, and specifically not this run's clock wearing the publisher's voice.

    A citation reading "last modified today" for a page untouched for two years is wrong in the
    direction that makes somebody act on it.
    """
    instance = _cloud(pages=[_page(when="", created="")])
    record = _source(await _record(instance))

    assert record.modified_at is None
    assert record.created_at is None


async def test_a_timestamp_with_no_offset_is_refused_rather_than_read_as_utc() -> None:
    """A naive timestamp is not a moment. Read as UTC it is wrong by the instance's offset.

    The failure is quiet in the worst way: the record validates, the citation renders, and the
    time is wrong by hours in a value used to decide which of two versions is newer.
    """
    instance = _cloud(pages=[_page(when="2026-08-09T14:30:00", created="2026-01-05T09:00:00")])
    record = _source(await _record(instance))

    assert record.modified_at is None
    assert record.created_at is None


async def test_the_record_carries_no_local_clock_at_all() -> None:
    """A network fetch has no local snapshot, so there is nothing to confuse the source with.

    ``LocalSnapshot`` has nowhere to put a canonical URI and ``SourceMetadata`` has nowhere to
    put a path; this asserts the live connector uses only the second, so ``retrieved_at`` cannot
    arrive wearing ``modified_at``'s name.
    """
    before = datetime.now(tz=UTC)
    record = await _record(_cloud())
    assert record is not None

    assert record.snapshot is None
    modified = _source(record).modified_at
    assert modified is not None
    assert modified < before, "the source's own time, not a clock this run read"


# --- the stale-body defense --------------------------------------------------------------------


async def test_a_stale_body_is_cited_at_the_version_of_the_bytes_that_were_kept() -> None:
    """Provenance describes what this index holds, never what it asked for.

    Recording the expected version against older bytes would publish a claim to be holding a
    revision that was never retrieved — and the version is the field a reader would use to check.
    """
    instance = _cloud(
        pages=[_page(version=5, served_version=4, storage_version=4, created=CREATED)]
    )
    config = cloud_config(base_url=instance.base_url, page_size=10)
    connector = await connected(instance, config)
    try:
        found = await drain(connector.discover(None))
        raw = await connector.fetch(found[0].ref)
    finally:
        await connector.teardown()

    record = _source(Provenance.from_metadata(raw.metadata))
    assert record.version == "4", "the version of the bytes that were retained"
    assert raw.metadata[VERSION_TOKEN] == "4"
    # The diagnostic survives beside the record rather than being smoothed over by it.
    assert raw.metadata["version_disagreement"] == {"discovered": 5, "fetched": 4}


async def test_the_fallback_body_supplies_the_timestamp_that_goes_with_its_own_bytes() -> None:
    """The record's modification time comes from the response the bytes came from.

    Reading it from the Cloud response after falling back to storage format would date the
    retained bytes by a response that was discarded — the same confusion as the version, one
    field along, and with nothing that would ever report it.
    """
    storage_when = "2026-07-01T08:00:00.000+01:00"
    # Atlassian Document Format stays stale and storage format is current, so storage is the
    # body that is kept. With *both* stale the connector keeps the Atlassian one — which is the
    # neighbouring case, and is covered above by the version it cites.
    instance = _cloud(pages=[_page(version=5, served_version=4, when=storage_when)])
    record = _source(await _record(instance))

    assert record.version == "5"
    assert record.content_type == CONFLUENCE_MEDIA_TYPE, "the storage body's own type"
    assert record.modified_at == datetime.fromisoformat(storage_when)


# --- identity across a rename and a move -------------------------------------------------------


async def test_a_renamed_and_moved_page_keeps_its_id_and_updates_everything_else() -> None:
    """The page id is what makes the second fetch an update rather than a second document.

    Everything derived from where the page sits and what it is called moves with it, and the one
    field that does not move is the one identity is keyed on.
    """
    instance = _cloud()
    first = _source(await _record(instance))

    instance.pages[PAGE_ID] = _page(
        title="Retry and Backoff",
        version=4,
        when="2026-08-20T10:00:00.000+01:00",
        ancestors=("Platform", "Resilience"),
        created=CREATED,
    )
    second = _source(await _record(instance))

    assert second.source_id == first.source_id == PAGE_ID
    assert second.title == "Retry and Backoff"
    assert second.section_path == ("ENG", "Platform", "Resilience")
    assert second.version == "4"
    assert second.modified_at is not None
    assert first.modified_at is not None
    assert second.modified_at > first.modified_at
    assert second.canonical_uri != first.canonical_uri


# --- attachments have identities of their own ---------------------------------------------------


def _with_attachment() -> FakeConfluence:
    return _cloud(
        attachments=[
            FakeAttachment(
                id="att-7",
                title="sequence.pdf",
                space="ENG",
                page_id=PAGE_ID,
                page_title="Retry Policy",
                version=2,
                when="2026-08-10T11:00:00.000+01:00",
            )
        ]
    )


async def test_an_attachment_is_cited_as_itself_and_not_as_the_page_holding_it() -> None:
    """A reference to the page would send a reader to a page that has a diagram *somewhere*.

    So the attachment's id, version, address and media type are its own. The page survives as a
    relationship — in ``parent_page_id`` and in the hierarchy — which is what it is.
    """
    instance = _with_attachment()
    record = _source(await _record(instance, source_id="att-7", include_attachments=True))

    assert record.source_id == "att-7"
    assert record.source_id != PAGE_ID
    assert record.title == "sequence.pdf"
    assert record.version == "2"
    assert record.content_type == "application/pdf"
    assert record.canonical_uri.endswith("sequence.pdf")
    assert record.section_path == ("ENG", "Retry Policy")


async def test_an_attachment_takes_its_modification_time_from_the_response_that_found_it() -> None:
    """The download is bytes and describes nothing, so discovery's result is the only source.

    It is still a source response — which is the distinction that matters. What would not be
    acceptable is the page's timestamp, or this run's.
    """
    instance = _with_attachment()
    connector = await connected(
        instance, cloud_config(base_url=instance.base_url, page_size=10, include_attachments=True)
    )
    try:
        found = await drain(connector.discover(None))
        ref = next(document.ref for document in found if document.source_id == "att-7")
        raw = await connector.fetch(ref)
    finally:
        await connector.teardown()

    assert ref.metadata[MODIFIED_AT] == "2026-08-10T11:00:00.000+01:00"
    record = _source(Provenance.from_metadata(raw.metadata))
    assert record.modified_at == datetime.fromisoformat("2026-08-10T11:00:00.000+01:00")
    assert record.modified_at != datetime.fromisoformat(MODIFIED), "not the page's"


async def test_a_page_carries_no_discovery_timestamp_to_be_confused_with_its_body_one() -> None:
    """Only attachments carry one, and a page having two would be two answers to one question.

    They disagree exactly when the stale-body fallback fires, which is the moment the wrong one
    would be picked.
    """
    instance = _cloud()
    connector = await connected(instance, cloud_config(base_url=instance.base_url, page_size=10))
    try:
        found = await drain(connector.discover(None))
    finally:
        await connector.teardown()

    assert MODIFIED_AT not in found[0].ref.metadata


# --- refusals ------------------------------------------------------------------------------------


async def test_a_title_a_citation_cannot_render_is_a_stated_refusal_not_a_lost_page() -> None:
    """The page is still indexed and still searchable; what it loses is the canonical record.

    Refused **with a reason**, because a silently absent record is indistinguishable from a
    connector that was never taught to write one — and an operator would have nothing to act on.
    """
    instance = _cloud(pages=[_page(title="Retry\x1bPolicy", created=CREATED)])
    record = await _record(instance)

    assert record is not None
    assert record.source is None
    assert PAGE_ID in record.unavailable_reason
    assert "will not cite" in record.unavailable_reason


async def test_a_refusal_names_the_page_and_quotes_no_content() -> None:
    """A diagnostic is read in a log, and a page title is content.

    The offending code point is named by the core validator, which is what somebody needs to fix
    it; the title itself is not reproduced by this connector's half of the message.
    """
    instance = _cloud(pages=[_page(title="Retry\x1bPolicy", created=CREATED)])
    record = await _record(instance)

    assert record is not None
    assert record.unavailable_reason.startswith(f"page {PAGE_ID}:")
    assert "\x1b" not in record.unavailable_reason


async def test_a_refused_record_is_still_written_rather_than_omitted() -> None:
    """Absent and refused are different facts, and the surface renders them differently."""
    instance = _cloud(pages=[_page(title="Retry\x1bPolicy", created=CREATED)])
    config = cloud_config(base_url=instance.base_url, page_size=10)
    connector = await connected(instance, config)
    try:
        found = await drain(connector.discover(None))
        raw = await connector.fetch(found[0].ref)
    finally:
        await connector.teardown()

    assert PROVENANCE_KEY in raw.metadata


# --- what provenance is not --------------------------------------------------------------------


async def test_scoping_metadata_stays_out_of_the_publication_record() -> None:
    """Why a document was selected is manicule's reason, not the publisher's account of itself.

    ``root_page_ids`` would read, inside a source record, as something Confluence said about the
    page. It says something about this installation's configuration.
    """
    instance = _cloud(
        pages=[
            _page(created=CREATED),
            FakePage(id="4200", title="Child", space="ENG", parent=PAGE_ID),
        ]
    )
    record = _source(
        await _record(instance, spaces=("ENG",), root_page_ids=(PAGE_ID,), source_id="4200")
    )

    assert record.source_id == "4200"
    assert PAGE_ID not in record.section_path
    dumped = record.model_dump()
    assert "root_page_ids" not in dumped
    assert "ancestor_ids" not in dumped


async def test_the_record_does_not_change_the_documents_identity_or_its_bytes() -> None:
    """Rule 7, asserted rather than assumed: provenance is added beside everything, not instead.

    The stored ``source_id`` and the record's ``source_id`` agree for this connector, which is
    also what keeps the page-keyed identity migration from moving these rows — it moves exactly
    the rows where the two differ.
    """
    instance = _server()
    config = server_config(instance.base_url, page_size=10)
    connector = await connected(instance, config)
    try:
        found = await drain(connector.discover(None))
        raw = await connector.fetch(found[0].ref)
    finally:
        await connector.teardown()

    record = _source(Provenance.from_metadata(raw.metadata))
    assert raw.source_id == PAGE_ID
    assert record.source_id == raw.source_id
    assert raw.media_type == CONFLUENCE_MEDIA_TYPE
    assert raw.as_text() == "<p>body text</p>", "the stored bytes are untouched"


@pytest.mark.parametrize("server", [False, True], ids=["cloud", "server"])
async def test_the_recorded_content_type_is_the_one_the_bytes_were_routed_by(server: bool) -> None:
    """Which is also a media type the core pattern has to admit.

    Both Confluence formats are identified by a profile parameter, and a record that could not
    state the type this system routed a document by would have to lie or say nothing.
    """
    instance = _server() if server else _cloud()
    config = (
        server_config(instance.base_url, page_size=10)
        if server
        else cloud_config(base_url=instance.base_url, page_size=10)
    )
    connector = await connected(instance, config)
    try:
        found = await drain(connector.discover(None))
        raw = await connector.fetch(found[0].ref)
    finally:
        await connector.teardown()

    record = _source(Provenance.from_metadata(raw.metadata))
    assert record.content_type == raw.media_type
    assert ";profile=" in record.content_type


async def test_a_ref_rebuilt_elsewhere_still_produces_a_usable_record() -> None:
    """A re-fetch carries no discovery metadata, and the record is built from the response.

    It is thinner — the hierarchy is what the ancestors endpoint could supply — and it is still
    a record, because everything in it came from the source either way.
    """
    instance = _cloud()
    connector = await connected(instance, cloud_config(base_url=instance.base_url, page_size=10))
    try:
        raw = await connector.fetch(DocRef(source_id=PAGE_ID, uri=f"{CLOUD}/pages/{PAGE_ID}"))
    finally:
        await connector.teardown()

    record = _source(Provenance.from_metadata(raw.metadata))
    assert record.source_id == PAGE_ID
    assert record.version == "3"
    assert record.modified_at == datetime.fromisoformat(MODIFIED)
