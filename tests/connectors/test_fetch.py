"""Fetching a body, and refusing to trust it blindly.

Atlassian Document Format has been observed returning stale content
(``docs/connectors/confluence.md`` §4). Stale content is not a visible failure: the page indexes
cleanly, reads plausibly and is wrong. The version discovery reported is the only thing that can
contradict it, so it is compared, and a disagreement is retried and then answered from a
different code path on the source's side.
"""

from __future__ import annotations

import json

import pytest

from manicule.connectors import (
    AttachmentTooLargeError,
    NotFoundError,
    RateLimitedError,
    UntrustedLinkError,
)
from manicule.connectors.confluence import ANCESTORS, STORAGE_MEDIA_TYPE, VERSION_TOKEN
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


async def test_a_body_stale_in_every_format_is_stored_under_the_version_it_actually_has() -> None:
    """This is what makes it self-healing, and why the honest token matters.

    Recording the version that was *asked for* against older bytes would make the document
    permanently stale: the next sync would compare the two, find them equal, and skip the page
    forever. Recording what came back leaves the stored token behind, so the next sync fetches
    it again.
    """
    instance = FakeConfluence(
        pages=[
            FakePage(
                id="1", title="Page", space="ENG", version=5, served_version=4, storage_version=4
            )
        ]
    )
    raw = await _fetched(instance, "1")

    assert raw.metadata[VERSION_TOKEN] == "4"
    assert raw.metadata["version_disagreement"] == {"discovered": 5, "fetched": 4}


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
        pages=[FakePage(id="1", title="Page", space="ENG", adf_available=False)]
    )
    raw = await _fetched(instance, "1")

    assert raw.media_type == STORAGE_MEDIA_TYPE
    assert raw.metadata["body_format"] == "storage"


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
