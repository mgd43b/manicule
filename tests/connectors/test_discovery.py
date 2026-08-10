"""Discovery: what a sync asks for, and what it remembers afterwards.

The point of the watermark (``docs/connectors/confluence.md`` §2) is that a sync costs what
changed. That is either true of the query or it is not, so these assert on the CQL that went
out as well as on the documents that came back.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from manicule.connectors import ConnectorError
from manicule.connectors.confluence import ANCESTORS, KIND, SPACE_KEY
from manicule.core.sources import Watermark
from manicule.testing import closing
from tests.connectors.fake_confluence import FakeAttachment, FakeConfluence, FakePage
from tests.connectors.support import cloud_config, connected, drain, ids


def _instance() -> FakeConfluence:
    return FakeConfluence(
        pages=[
            FakePage(
                id="1",
                title="Token Refresh",
                space="ENG",
                version=3,
                when="2026-08-09T14:30:00.000+01:00",
                ancestors=("Platform", "Auth Service"),
            ),
            FakePage(
                id="2",
                title="Runbook",
                space="OPS",
                version=1,
                when="2026-08-08T09:00:00.000+01:00",
            ),
        ],
        spaces={"ENG": "Engineering", "OPS": "Operations"},
        page_size=10,
    )


async def test_a_first_sync_asks_for_everything_in_scope() -> None:
    """No watermark means no date clause: the whole space, once."""
    instance = _instance()
    connector = await connected(instance)
    try:
        found = await drain(connector.discover(None))
    finally:
        await connector.teardown()

    assert sorted(ids(found)) == ["1", "2"]
    assert all("lastmodified >=" not in query for query in instance.queries())


async def test_an_incremental_sync_asks_only_for_what_changed() -> None:
    """The whole point: the query carries the watermark, so the source does the filtering.

    Re-enumerating every page and comparing versions client-side gets the same answer and pays
    for the entire corpus every time.
    """
    instance = _instance()
    connector = await connected(instance)
    watermark = Watermark(
        value="2026-08-09T14:30:00+01:00",
        observed_at=datetime.now(tz=UTC),
        metadata={"spaces": {"ENG": "2026-08-09T14:30:00+01:00"}},
    )
    try:
        await drain(connector.discover(watermark))
    finally:
        await connector.teardown()

    eng = [query for query in instance.queries() if '"ENG"' in query]
    ops = [query for query in instance.queries() if '"OPS"' in query]
    assert any('lastmodified >= "2026/08/09 14:25"' in query for query in eng), eng
    assert all("lastmodified >=" not in query for query in ops), ops


async def test_the_watermark_reaches_back_far_enough_to_cover_a_shared_minute() -> None:
    """CQL compares ``lastmodified`` to the minute, so an exact resume can skip a page.

    Two pages saved in the same minute, one indexed and one not, are indistinguishable to the
    query. Overlapping costs a version comparison the pipeline was going to make anyway; not
    overlapping costs a page that is wrong in the index until something unrelated touches it.
    """
    instance = _instance()
    connector = await connected(
        instance, cloud_config(base_url=instance.base_url, watermark_overlap_minutes=0)
    )
    watermark = Watermark(
        value="2026-08-09T14:30:00+01:00",
        observed_at=datetime.now(tz=UTC),
        metadata={"spaces": {"ENG": "2026-08-09T14:30:59+01:00"}},
    )
    try:
        await drain(connector.discover(watermark))
    finally:
        await connector.teardown()

    exact = [query for query in instance.queries() if '"ENG"' in query]
    assert any('lastmodified >= "2026/08/09 14:30"' in query for query in exact), exact


async def test_each_space_carries_its_own_watermark() -> None:
    """Spaces are enumerated separately, so one failing must not advance the others."""
    instance = _instance()
    connector = await connected(instance)
    try:
        await drain(connector.discover(None))
        watermark = connector.watermark
    finally:
        await connector.teardown()

    assert watermark is not None
    spaces = watermark.metadata["spaces"]
    assert isinstance(spaces, dict)
    assert set(spaces) == {"ENG", "OPS"}
    eng = spaces["ENG"]
    assert isinstance(eng, str)
    assert eng.startswith("2026-08-09T14:30")


async def test_a_watermark_is_not_offered_for_an_abandoned_enumeration() -> None:
    """A consumer that stopped early saw a prefix, and a watermark from a prefix skips the rest.

    Nothing ever looks for what was skipped again, so this is the failure that makes a sync
    quietly incomplete forever. The pipeline has the same rule; the connector does not rely on
    it remembering.
    """
    instance = _instance()
    connector = await connected(instance)
    try:
        async with closing(connector.discover(None)) as stream:
            await anext(stream)
        assert connector.watermark is None
    finally:
        await connector.teardown()


async def test_a_space_that_yielded_nothing_keeps_the_position_it_had() -> None:
    """An empty incremental run must not erase where the last one got to."""
    instance = _instance()
    connector = await connected(instance)
    stored = Watermark(
        value="2026-09-01T00:00:00+01:00",
        observed_at=datetime.now(tz=UTC),
        metadata={"spaces": {"ENG": "2026-09-01T00:00:00+01:00"}},
    )
    try:
        found = await drain(connector.discover(stored))
        watermark = connector.watermark
    finally:
        await connector.teardown()

    assert "1" not in ids(found)
    assert watermark is not None
    spaces = watermark.metadata["spaces"]
    assert isinstance(spaces, dict)
    assert spaces["ENG"] == "2026-09-01T00:00:00+01:00"


async def test_a_page_carries_the_breadcrumb_its_chunks_will_be_prefixed_with() -> None:
    """A section called "Configuration" is unretrievable without knowing what it configures.

    The chunker reads ``ancestors`` from the document and appends the title itself, so the
    connector supplies the space and the hierarchy above the page and stops there.
    """
    instance = _instance()
    connector = await connected(instance)
    try:
        found = await drain(connector.discover(None))
    finally:
        await connector.teardown()

    page = next(document for document in found if document.source_id == "1")
    assert page.ref.metadata[ANCESTORS] == ["ENG", "Platform", "Auth Service"]
    assert page.ref.metadata[SPACE_KEY] == "ENG"
    assert page.title == "Token Refresh"
    assert page.version_token == "3"  # noqa: S105 - a version number, not a credential


async def test_an_attachment_is_discovered_as_a_document_of_its_own() -> None:
    """A PDF attached to a page is a PDF, and it keeps a link to the page it hangs off."""
    instance = FakeConfluence(
        pages=[FakePage(id="1", title="Token Refresh", space="ENG")],
        attachments=[
            FakeAttachment(
                id="att-9",
                title="diagram.pdf",
                space="ENG",
                page_id="1",
                page_title="Token Refresh",
            )
        ],
    )
    connector = await connected(instance)
    try:
        found = await drain(connector.discover(None))
    finally:
        await connector.teardown()

    attachment = next(document for document in found if document.source_id == "att-9")
    assert attachment.media_type == "application/pdf"
    assert attachment.ref.metadata[KIND] == "attachment"
    assert attachment.ref.metadata["parent_page_id"] == "1"
    assert attachment.ref.metadata[ANCESTORS] == ["ENG", "Token Refresh"]
    assert attachment.size_bytes == len(b"%PDF-1.4 fake")


async def test_attachments_can_be_left_out_entirely() -> None:
    """And when they are, the query says so rather than the results being filtered afterwards."""
    instance = FakeConfluence(
        pages=[FakePage(id="1", title="Token Refresh", space="ENG")],
        attachments=[
            FakeAttachment(
                id="att-9",
                title="diagram.pdf",
                space="ENG",
                page_id="1",
                page_title="Token Refresh",
            )
        ],
    )
    connector = await connected(
        instance, cloud_config(base_url=instance.base_url, include_attachments=False)
    )
    try:
        found = await drain(connector.discover(None))
    finally:
        await connector.teardown()

    assert ids(found) == ["1"]
    assert all("attachment" not in query for query in instance.queries())


async def test_a_space_key_that_does_not_exist_is_a_refusal() -> None:
    """CQL answers a query for a space that is not there with an empty result set.

    So a typo in the allowlist is a sync that runs, succeeds and indexes nothing — and then
    reconciliation proposes deleting everything that space ever contributed. One extra
    enumeration per run is what stands between those two outcomes.
    """
    instance = _instance()
    connector = await connected(
        instance, cloud_config(base_url=instance.base_url, spaces=("ENG", "ENGG"))
    )
    try:
        with pytest.raises(ConnectorError, match="ENGG"):
            await drain(connector.discover(None))
    finally:
        await connector.teardown()


async def test_an_account_that_can_see_no_spaces_is_a_refusal() -> None:
    """Indexing nothing quietly is worse than failing: reconciliation would then empty the index."""
    instance = FakeConfluence(pages=[], spaces={})
    connector = await connected(instance)
    try:
        with pytest.raises(ConnectorError, match="no spaces"):
            await drain(connector.discover(None))
    finally:
        await connector.teardown()


async def test_only_current_content_is_enumerated() -> None:
    """Reconciliation depends on a trashed page *not* being returned.

    A query that included trashed content would report every deleted page as still present, and
    deletion detection would run, succeed, and find nothing, forever.
    """
    instance = _instance()
    connector = await connected(instance)
    try:
        await drain(connector.discover(None))
    finally:
        await connector.teardown()

    assert all("status = current" in query for query in instance.queries())


async def test_a_space_key_cannot_end_the_string_literal_it_is_quoted_in() -> None:
    """A key containing a quote would otherwise continue as query syntax — a wider search.

    The failure is not an error; it is results, from a query nobody wrote.
    """
    instance = FakeConfluence(
        pages=[FakePage(id="1", title="Page", space='A" OR type=blogpost OR space="B')],
        spaces={'A" OR type=blogpost OR space="B': "Odd"},
    )
    connector = await connected(instance)
    try:
        await drain(connector.discover(None))
    finally:
        await connector.teardown()

    assert any('space = "A\\" OR type=blogpost OR space=\\"B"' in q for q in instance.queries())
