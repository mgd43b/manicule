"""Fetching a body, and refusing to trust it blindly.

Atlassian Document Format has been observed returning stale content
(``docs/connectors/confluence.md`` §4). Stale content is not a visible failure: the page indexes
cleanly, reads plausibly and is wrong. The version discovery reported is the only thing that can
contradict it, so it is compared, and a disagreement is retried and then answered from a
different code path on the source's side.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from manicule.connectors import (
    AttachmentTooLargeError,
    BodyUnavailableError,
    NotFoundError,
    RateLimitedError,
    UntrustedLinkError,
)
from manicule.connectors.confluence import (
    ANCESTOR_IDS,
    ANCESTORS,
    STORAGE_MEDIA_TYPE,
    VERSION_TOKEN,
)
from manicule.core.content import DocumentStatus
from manicule.core.sources import DocRef
from manicule.parsers.config import ADF_MEDIA_TYPE
from tests.connectors.fake_confluence import (
    SERVER_BASE,
    FakeAttachment,
    FakeConfluence,
    FakePage,
    paragraph,
)
from tests.connectors.support import cloud_config, connected, drain, server_config
from tests.ingest.test_pipeline import build


async def _fetched(instance: FakeConfluence, source_id: str, **overrides: object):  # noqa: ANN202
    """Discover, then fetch one document, so the ref is the one discovery actually built."""
    config = cloud_config(base_url=instance.base_url, **overrides)
    connector = await connected(instance, config)
    try:
        found = await drain(connector.discover(None))
        ref = next(document.ref for document in found if document.source_id == source_id)
        return await connector.fetch(ref)
    finally:
        await connector.teardown()


async def test_a_cloud_page_arrives_as_a_typed_document_tree() -> None:
    """ADF is the reason Cloud is worth a separate path: a codeBlock says it is code.

    The connector hands over the tree and says what it is; recovering structure from markup is
    what the format exists to avoid.
    """
    instance = FakeConfluence(
        pages=[
            FakePage(
                id="1",
                title="Token Refresh",
                space="ENG",
                version=4,
                ancestors=("Platform", "Auth Service"),
                adf=paragraph("rotate the refresh token"),
            )
        ]
    )
    raw = await _fetched(instance, "1")

    assert raw.media_type == ADF_MEDIA_TYPE
    assert json.loads(raw.as_text())["type"] == "doc"
    assert raw.metadata["title"] == "Token Refresh"
    assert raw.metadata[ANCESTORS] == ["ENG", "Platform", "Auth Service"]
    assert raw.metadata[VERSION_TOKEN] == "4"
    assert raw.uri.endswith("/spaces/ENG/pages/1/Token Refresh")


async def test_a_server_page_arrives_as_storage_format() -> None:
    """Data Center has no ADF, so the body is XHTML and goes to a real HTML parser.

    Not a pass that strips angle brackets: every table, code block and heading in a page
    survives that only if something parses it.
    """
    instance = FakeConfluence(
        base_url=SERVER_BASE,
        pages=[
            FakePage(
                id="1",
                title="Runbook",
                space="OPS",
                ancestors=("Platform",),
                storage="<h2>Restart</h2><p>drain first</p>",
            )
        ],
    )
    connector = await connected(instance, server_config(instance.base_url))
    try:
        found = await drain(connector.discover(None))
        raw = await connector.fetch(found[0].ref)
    finally:
        await connector.teardown()

    assert raw.media_type == STORAGE_MEDIA_TYPE
    assert raw.as_text() == "<h2>Restart</h2><p>drain first</p>"
    assert raw.metadata[ANCESTORS] == ["OPS", "Platform"]
    assert raw.metadata["body_format"] == "storage"


async def test_a_body_older_than_discovery_reported_is_fetched_again() -> None:
    """The known staleness bug, and the cheapest answer to it.

    Nothing about a stale body looks wrong, so the only signal is the version search reported
    a moment earlier. One retry clears a caching artifact.
    """
    instance = FakeConfluence(
        pages=[
            FakePage(
                id="1", title="Page", space="ENG", version=5, served_version=4, stale_once=True
            )
        ]
    )
    raw = await _fetched(instance, "1")

    assert instance.body_calls["1"] == 2
    assert raw.metadata[VERSION_TOKEN] == "5"
    assert "version_disagreement" not in raw.metadata


async def test_a_body_that_stays_stale_falls_back_to_storage_format() -> None:
    """A different body format is a different code path on the source's side.

    Which is the whole reason it is worth trying: the failure being worked around is in one
    format's rendering, not in the page.
    """
    instance = FakeConfluence(
        pages=[FakePage(id="1", title="Page", space="ENG", version=5, served_version=4)]
    )
    raw = await _fetched(instance, "1")

    assert raw.media_type == STORAGE_MEDIA_TYPE
    assert raw.metadata["body_format"] == "storage"
    assert raw.metadata[VERSION_TOKEN] == "5"


async def test_a_body_stale_in_every_format_is_refused() -> None:
    """No response can certify bytes older than discovery's version.

    The pipeline persists discovery's token, so returning version-four bytes here would store
    them as version five and make the next sync skip the page as current forever.
    """
    instance = FakeConfluence(
        pages=[
            FakePage(
                id="1", title="Page", space="ENG", version=5, served_version=4, storage_version=4
            )
        ]
    )
    with pytest.raises(
        BodyUnavailableError,
        match=r"discovered at version 5.*Format returned versions 4, 4.*storage format "
        r"returned version 4",
    ):
        await _fetched(instance, "1")


async def test_stale_cloud_bytes_cannot_cross_the_connector_pipeline_boundary() -> None:
    """The end-to-end guard for the token owner mismatch that caused permanent staleness.

    Connector metadata can describe the fetched revision, but ingest deliberately persists the
    discovery token. A connector-only assertion on metadata therefore cannot prove stale bytes
    are safe; the real pipeline must see a failed fetch and store no document under version five.
    """
    instance = FakeConfluence(
        pages=[
            FakePage(
                id="1", title="Page", space="ENG", version=5, served_version=4, storage_version=4
            )
        ]
    )
    connector = await connected(instance)
    pipeline, store, _ = build()
    try:
        refused = await pipeline.run(connector)

        assert refused.indexed == 0
        assert refused.by_status[DocumentStatus.FAILED.value] == 1
        assert refused.unrecorded == 1, "the watermark must not advance past the refused page"
        assert connector.name not in store.watermarks
        assert await store.find_document(connector.name, "1") is None, (
            "no stored token may certify bytes the connector refused"
        )

        instance.pages["1"] = replace(
            instance.pages["1"],
            served_version=5,
            storage_version=5,
            adf=paragraph("current version five"),
        )
        recovered = await pipeline.run(connector)
    finally:
        await connector.teardown()

    document = await store.find_document(connector.name, "1")
    assert recovered.indexed == 1
    assert document is not None
    assert document.version_token == "5"  # noqa: S105 - a change token, not a credential
    assert any("current version five" in chunk.text for chunk in store.chunks[document.id])
    reached = connector.watermark
    assert reached is not None
    stored = store.watermarks[connector.name]
    assert (stored.value, stored.metadata) == (reached.value, reached.metadata), (
        "the first complete run, and only that run, advances the source position"
    )


async def test_a_stale_refresh_preserves_the_last_indexed_revision() -> None:
    """Failing closed must not make an accurate older revision disappear from retrieval."""
    instance = FakeConfluence(
        pages=[
            FakePage(id="1", title="Page", space="ENG", version=4, adf=paragraph("version four"))
        ]
    )
    connector = await connected(instance)
    pipeline, store, _ = build()
    try:
        await pipeline.run(connector)
        before = await store.find_document(connector.name, "1")
        assert before is not None
        chunks_before = list(store.chunks[before.id])

        instance.pages["1"] = replace(
            instance.pages["1"], version=5, served_version=4, storage_version=4
        )
        refreshed = await pipeline.run(connector)
    finally:
        await connector.teardown()

    after = await store.find_document(connector.name, "1")
    assert refreshed.indexed == 1, "the preserved revision remains servable"
    assert after is not None
    assert after.version_token == "4", (  # noqa: S105 - a change token, not a credential
        "the refused revision five was never certified"
    )
    assert after.content_hash == before.content_hash
    assert store.chunks[after.id] == chunks_before
    assert "BodyUnavailableError" in str(after.metadata["last_ingest_error"])


async def test_an_attachment_is_downloaded_and_keeps_its_page() -> None:
    """A citation should resolve to the file and to the page it hangs off, so both are kept."""
    instance = FakeConfluence(
        pages=[FakePage(id="1", title="Token Refresh", space="ENG")],
        attachments=[
            FakeAttachment(
                id="att-9",
                title="diagram.pdf",
                space="ENG",
                page_id="1",
                page_title="Token Refresh",
                content=b"%PDF-1.4 diagram",
            )
        ],
    )
    raw = await _fetched(instance, "att-9")

    assert raw.media_type == "application/pdf"
    assert raw.as_bytes() == b"%PDF-1.4 diagram"
    assert raw.metadata["parent_page_id"] == "1"
    assert raw.metadata[ANCESTORS] == ["ENG", "Token Refresh"]


async def test_an_attachment_over_the_ceiling_is_refused_by_name() -> None:
    """Counted while streaming, never taken from a declared length.

    The point of a ceiling is to survive a source whose claim about the response turns out to
    be wrong, and the refusal names the setting so the document records why it has no content.
    """
    instance = FakeConfluence(
        pages=[FakePage(id="1", title="Page", space="ENG")],
        attachments=[
            FakeAttachment(
                id="att-9",
                title="big.pdf",
                space="ENG",
                page_id="1",
                page_title="Page",
                content=b"x" * 64,
            )
        ],
    )
    with pytest.raises(AttachmentTooLargeError, match="max_attachment_bytes"):
        await _fetched(instance, "att-9", max_attachment_bytes=16)


async def test_a_page_deleted_between_discovery_and_fetch_is_reported_as_gone() -> None:
    """Ordinary during a sync that takes hours, and not the same thing as a fault."""
    instance = FakeConfluence(pages=[FakePage(id="1", title="Page", space="ENG")])
    connector = await connected(instance)
    try:
        found = await drain(connector.discover(None))
        instance.delete("1")
        with pytest.raises(NotFoundError):
            await connector.fetch(found[0].ref)
    finally:
        await connector.teardown()


async def test_a_ref_without_ancestors_still_gets_a_breadcrumb() -> None:
    """A ref rebuilt elsewhere — a re-fetch, a single-page sync — has no discovery record.

    A breadcrumb missing a level is not visibly wrong: it retrieves slightly worse and says
    nothing, so the ancestors are fetched rather than skipped.
    """
    instance = FakeConfluence(
        pages=[FakePage(id="1", title="Page", space="ENG", ancestors=("Platform", "Auth Service"))]
    )
    connector = await connected(instance)
    try:
        raw = await connector.fetch(
            DocRef(source_id="1", uri=f"{instance.base_url}/pages/1", metadata={"space_key": "ENG"})
        )
    finally:
        await connector.teardown()

    assert raw.metadata[ANCESTORS] == ["ENG", "Platform", "Auth Service"]
    assert raw.metadata["breadcrumb_complete"] is True


async def test_a_breadcrumb_without_its_space_key_does_not_claim_to_be_complete() -> None:
    """The Cloud body endpoint reports a numeric space id, not the key a breadcrumb starts with.

    So a ref that carries neither ancestors nor a space key — a re-fetch from a stored record
    written before this connector kept one — produces a breadcrumb one level short: every chunk
    of that page is prefixed ``Platform > Auth Service`` where the rest of the corpus reads
    ``ENG > Platform > Auth Service``, and it retrieves worse against exactly the queries a
    space key disambiguates.

    Inventing the key is not available and is not wanted; saying so is. ``breadcrumb_complete``
    exists for a breadcrumb that is short, and it has to be false when this one is, or a
    survey of incomplete breadcrumbs reports clean while the gap is in the index.
    """
    instance = FakeConfluence(
        pages=[FakePage(id="1", title="Page", space="ENG", ancestors=("Platform", "Auth Service"))]
    )
    connector = await connected(instance)
    try:
        raw = await connector.fetch(DocRef(source_id="1", uri=f"{instance.base_url}/pages/1"))
    finally:
        await connector.teardown()

    assert raw.metadata[ANCESTORS] == ["Platform", "Auth Service"]
    assert raw.metadata["breadcrumb_complete"] is False


async def test_recovered_ancestors_bring_their_ids_and_not_only_their_titles() -> None:
    """The Cloud ancestors endpoint reports both, so recording one of them is a choice.

    An empty ancestor-id list is indistinguishable from a page at the top of its space, and
    that is a different fact about a different page. Whichever fetch happened to run last would
    otherwise decide which of the two the document asserts.
    """
    instance = FakeConfluence(
        pages=[
            FakePage(id="1", title="Page", space="ENG", parent="7"),
            FakePage(id="7", title="Platform", space="ENG"),
        ]
    )
    connector = await connected(instance)
    try:
        raw = await connector.fetch(DocRef(source_id="1", uri=f"{instance.base_url}/pages/1"))
    finally:
        await connector.teardown()

    assert raw.metadata[ANCESTORS] == ["Platform"]
    assert raw.metadata[ANCESTOR_IDS] == ["7"]


async def test_a_server_breadcrumb_comes_from_the_fetch_own_expansion() -> None:
    """Storage format expands ``ancestors`` alongside the body, so no ref and no second call.

    Worth pinning because ``docs/connectors/confluence.md`` §4 read as though discovery were
    the only source of a breadcrumb; on Server and Data Center the fetch has its own, it is
    complete including the space key, and it is preferred over anything the ref carries.
    """
    instance = FakeConfluence(
        base_url=SERVER_BASE,
        pages=[FakePage(id="1", title="Page", space="ENG", ancestors=("Platform", "Auth Service"))],
    )
    connector = await connected(instance, server_config(instance.base_url))
    try:
        raw = await connector.fetch(DocRef(source_id="1", uri=f"{SERVER_BASE}/pages/1"))
    finally:
        await connector.teardown()

    assert raw.metadata[ANCESTORS] == ["ENG", "Platform", "Auth Service"]
    assert raw.metadata["breadcrumb_complete"] is True
    assert [request.url.path for request in instance.requests] == ["/confluence/rest/api/content/1"]


async def test_a_page_whose_adf_body_is_declined_falls_back_to_storage() -> None:
    """The page exists and the format does not, which is not the same as an empty page.

    Falling back is the whole reason two formats are read; the alternative is a document that
    fails for a reason the source did not actually give.
    """
    instance = FakeConfluence(
        pages=[FakePage(id="1", title="Page", space="ENG", version=5, adf_available=False)]
    )
    raw = await _fetched(instance, "1")

    assert raw.media_type == STORAGE_MEDIA_TYPE
    assert raw.metadata["body_format"] == "storage"
    assert raw.metadata[VERSION_TOKEN] == "5"


async def test_an_initially_unavailable_adf_cannot_bypass_a_stale_storage_refusal() -> None:
    """The fallback is another candidate body, not an escape from the version boundary."""
    instance = FakeConfluence(
        pages=[
            FakePage(
                id="1",
                title="Page",
                space="ENG",
                version=5,
                adf_available=False,
                storage_version=4,
            )
        ]
    )

    with pytest.raises(
        BodyUnavailableError,
        match=r"Format was unavailable.*storage format returned version 4",
    ):
        await _fetched(instance, "1")


async def test_an_unavailable_adf_retry_falls_back_to_current_storage() -> None:
    """Losing ADF on retry still permits a separately current representation."""
    instance = FakeConfluence(
        pages=[
            FakePage(
                id="1",
                title="Page",
                space="ENG",
                version=5,
                served_version=4,
                adf_available_calls=1,
            )
        ]
    )

    raw = await _fetched(instance, "1")

    assert instance.body_calls["1"] == 2
    assert raw.media_type == STORAGE_MEDIA_TYPE
    assert raw.metadata[VERSION_TOKEN] == "5"


async def test_an_unavailable_adf_retry_cannot_bypass_a_stale_storage_refusal() -> None:
    """Every bounded path ends at the same fail-closed version comparison."""
    instance = FakeConfluence(
        pages=[
            FakePage(
                id="1",
                title="Page",
                space="ENG",
                version=5,
                served_version=4,
                adf_available_calls=1,
                storage_version=4,
            )
        ]
    )

    with pytest.raises(
        BodyUnavailableError,
        match=r"Format returned version 4 before becoming unavailable.*storage format "
        r"returned version 4",
    ):
        await _fetched(instance, "1")


async def test_a_throttled_page_is_not_retried_through_a_second_endpoint() -> None:
    """The fallback is for one failure, and catching any error would widen it into a hazard.

    A 429 answered by trying a different endpoint doubles the load on a source that has just
    said stop, and reports the wrong problem when it fails again. So the retry budget is spent
    on the endpoint that was asked, and the throttling surfaces as throttling.
    """
    instance = FakeConfluence(pages=[FakePage(id="1", title="Page", space="ENG")])
    connector = await connected(instance, cloud_config(base_url=instance.base_url, max_retries=1))
    try:
        found = await drain(connector.discover(None))
        instance.throttle(times=4, retry_after="1")
        with pytest.raises(RateLimitedError):
            await connector.fetch(found[0].ref)
    finally:
        await connector.teardown()

    v1 = [request for request in instance.requests if "/rest/api/content/1" in request.url.path]
    assert v1 == [], "a throttled page must not be retried through the storage endpoint"


async def test_a_download_link_naming_another_host_is_refused() -> None:
    """An attachment's download link comes from the response, and the request carries the
    sync account's credential. Using it unchecked lets a response choose who receives that
    credential."""
    instance = FakeConfluence(
        pages=[FakePage(id="1", title="Page", space="ENG")],
        attachments=[
            FakeAttachment(
                id="att-9",
                title="d.pdf",
                space="ENG",
                page_id="1",
                page_title="Page",
                download_link="https://collector.example/steal",
            )
        ],
    )
    with pytest.raises(UntrustedLinkError, match=r"collector\.example"):
        await _fetched(instance, "att-9")
