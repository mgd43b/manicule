"""Authoritative Data Center membership, without weakening CQL incremental discovery."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing
from datetime import UTC, datetime
from typing import cast

import httpx
import pytest
from pydantic import SecretStr

from manicule.connectors import ConnectorError
from manicule.connectors.client import ConfluenceClient
from manicule.connectors.config import (
    ConfluenceConfig,
    Deployment,
    FullInventoryAuthority,
)
from manicule.connectors.confluence import ConfluenceConnector
from manicule.core.sources import DiscoveredDoc, Watermark
from tests.connectors.fake_confluence import FakeAttachment, FakeConfluence, FakePage
from tests.connectors.support import connected, drain, ids, server_config

BASE = "https://wiki.example.test/confluence"


def _server(**overrides: object) -> ConfluenceConfig:
    return server_config(BASE, spaces=("DOCS",), **overrides)


def _direct(**overrides: object) -> ConfluenceConfig:
    return _server(
        full_inventory_authority=FullInventoryAuthority.DIRECT_CURRENT_CONTENT,
        **overrides,
    )


def _connector(config: ConfluenceConfig) -> ConfluenceConnector:
    return ConfluenceConnector(config, ConfluenceClient(config, clock=lambda: 0.0))


def test_effective_authority_changes_only_data_center_whole_space_cursor_identity() -> None:
    search = _connector(_server())
    direct = _connector(_direct())
    cloud_config = ConfluenceConfig(
        base_url=BASE,
        deployment=Deployment.CLOUD,
        email="sync@example.test",
        api_token=SecretStr("synthetic-token"),
        spaces=("DOCS",),
        full_inventory_authority=FullInventoryAuthority.DIRECT_CURRENT_CONTENT,
    )
    cloud = _connector(cloud_config)
    subtree_config = _direct(root_page_ids=("100100",))
    subtree = _connector(subtree_config)

    assert search.source_scope == direct.source_scope == "spaces=DOCS;whole-space"
    assert search.scope_fingerprint == "d2e33cdaca77eb3c557a9fe49d091693134e44cb"
    assert direct.scope_fingerprint != search.scope_fingerprint
    assert cloud.full_inventory_authority == "search"
    assert subtree.full_inventory_authority == "search"
    assert cloud.scope_fingerprint == cloud_config.scope_fingerprint(cloud.source_scope)
    assert subtree.scope_fingerprint == subtree_config.scope_fingerprint(subtree.source_scope)
    assert search.reconciliation_scope == "whole-connector:confluence"
    assert direct.reconciliation_scope != search.reconciliation_scope


async def test_direct_full_inventory_reaches_true_end_when_search_diverges() -> None:
    current = [
        FakePage(id=str(200000 + number), title=f"Current {number}", space="DOCS")
        for number in range(102)
    ]
    stale = FakePage(id="299999", title="Stale search row", space="DOCS")
    instance = FakeConfluence(
        base_url=BASE,
        pages=current,
        search_pages=[*current[:99], stale],
        page_size=7,
    )
    search = await connected(instance, _server(page_size=250))
    try:
        search_found = await drain(search.discover(None))
    finally:
        await search.teardown()
    assert len(search_found) == 100
    assert stale.id in ids(search_found)

    instance.requests.clear()
    direct = await connected(instance, _direct(page_size=250))
    try:
        batches = [tuple(batch) async for batch in direct.discover_batches(None)]
        reconciled = await drain(direct.reconcile())
    finally:
        await direct.teardown()

    found = [document for batch in batches for document in batch]
    assert len(found) == 102
    assert set(ids(found)) == {page.id for page in current}
    assert stale.id not in ids(found)
    assert all(len(batch) <= 7 for batch in batches)
    assert direct.watermark is not None
    assert direct.watermark.metadata["full_inventory_authority"] == "direct_current_content"
    assert direct.watermark.metadata["spaces"] == {"DOCS": "2026-08-09T14:30:00+01:00"}
    assert set(reconciled) == {page.id for page in current}
    direct_requests = [
        request for request in instance.requests if request.url.path.endswith("/rest/api/content")
    ]
    assert len(direct_requests) > 14
    assert all(request.url.params["spaceKey"] == "DOCS" for request in direct_requests)
    assert all(request.url.params["type"] in {"page", "attachment"} for request in direct_requests)
    assert sum(request.url.params["type"] == "page" for request in direct_requests) > 14
    assert all(request.url.params["status"] == "current" for request in direct_requests)
    assert all(
        request.url.params["expand"] == "version,ancestors,space,container"
        for request in direct_requests
    )


@pytest.mark.parametrize("spaces", [("DOCS",), ("DOCS", "OPS")])
async def test_empty_direct_inventory_still_commits_each_space_resume_position(
    spaces: tuple[str, ...],
) -> None:
    config = server_config(
        BASE,
        spaces=spaces,
        include_attachments=False,
        full_inventory_authority=FullInventoryAuthority.DIRECT_CURRENT_CONTENT,
    )
    instance = FakeConfluence(
        base_url=BASE,
        spaces={space: f"Synthetic {space}" for space in spaces},
    )
    first = await connected(instance, config)
    try:
        assert await drain(first.discover(None)) == []
        committed = first.watermark
    finally:
        await first.teardown()

    assert committed is not None
    assert committed.metadata["full_inventory_authority"] == "direct_current_content"
    assert committed.metadata["spaces"] == dict.fromkeys(spaces, "direct-current:true-end-empty")

    created = FakePage(
        id="250100",
        title="Created after empty inventory",
        space=spaces[0],
        when="2001-01-01T00:01:00+00:00",
    )
    instance.pages[created.id] = created

    instance.requests.clear()
    resumed = await connected(instance, config)
    try:
        found = await drain(resumed.discover(committed))
    finally:
        await resumed.teardown()
    assert ids(found) == [created.id]
    content_requests = [
        request for request in instance.requests if request.url.path.endswith("/rest/api/content")
    ]
    search_requests = [
        request
        for request in instance.requests
        if request.url.path.endswith("/rest/api/content/search")
    ]
    assert content_requests == []
    assert len(search_requests) == len(spaces)


async def test_malformed_direct_resume_position_forces_authoritative_reenumeration() -> None:
    page = FakePage(id="260100", title="Current", space="DOCS")
    instance = FakeConfluence(base_url=BASE, pages=[page])
    corrupted = Watermark(
        value="corrupted-position",
        observed_at=datetime(2026, 8, 17, tzinfo=UTC),
        metadata={
            "spaces": {"DOCS": "not-a-source-time-or-empty-marker"},
            "scope": "whole-space",
            "full_inventory_authority": "direct_current_content",
        },
    )
    connector = await connected(instance, _direct(include_attachments=False))
    try:
        found = await drain(connector.discover(corrupted))
    finally:
        await connector.teardown()

    assert ids(found) == [page.id]
    assert any(request.url.path.endswith("/rest/api/content") for request in instance.requests)
    assert not any(
        request.url.path.endswith("/rest/api/content/search") for request in instance.requests
    )


async def test_direct_page_and_attachment_discovery_matches_search_metadata() -> None:
    page = FakePage(
        id="300100",
        title="Runbook",
        space="DOCS",
        ancestors=("Operations",),
    )
    attachment = FakeAttachment(
        id="300200",
        title="diagram.pdf",
        space="DOCS",
        page_id=page.id,
        page_title=page.title,
    )
    instance = FakeConfluence(
        base_url=BASE,
        pages=[page],
        attachments=[attachment],
        page_size=1,
    )
    search = await connected(instance, _server())
    direct = await connected(instance, _direct())
    try:
        searched = {item.source_id: item for item in await drain(search.discover(None))}
        found = {item.source_id: item for item in await drain(direct.discover(None))}
    finally:
        await search.teardown()
        await direct.teardown()
    assert found == searched


async def test_direct_native_page_is_consumed_before_its_next_request() -> None:
    pages = [
        FakePage(id=str(350100 + number), title=f"Current {number}", space="DOCS")
        for number in range(5)
    ]
    instance = FakeConfluence(base_url=BASE, pages=pages, page_size=2)
    interrupted = await connected(instance, _direct(include_attachments=False, page_size=250))
    try:
        native_batches = cast(
            "AsyncGenerator[Sequence[DiscoveredDoc], None]",
            interrupted.discover_batches(None),
        )
        async with aclosing(native_batches) as batches:
            first = await anext(batches)
            direct_requests = [
                request
                for request in instance.requests
                if request.url.path.endswith("/rest/api/content")
            ]
            assert ids(first) == [pages[0].id, pages[1].id]
            assert len(direct_requests) == 1, "the next native page was requested before hand-off"
    finally:
        await interrupted.teardown()

    instance.requests.clear()
    resumed = await connected(instance, _direct(include_attachments=False, page_size=250))
    try:
        replayed = [tuple(batch) async for batch in resumed.discover_batches(None)]
    finally:
        await resumed.teardown()
    assert [ids(batch) for batch in replayed] == [
        [pages[0].id, pages[1].id],
        [pages[2].id, pages[3].id],
        [pages[4].id],
    ]
    starts = [
        request.url.params.get("start", "0")
        for request in instance.requests
        if request.url.path.endswith("/rest/api/content")
    ]
    assert starts == ["0", "2", "4"]


def _row() -> dict[str, object]:
    return {
        "id": "400100",
        "type": "page",
        "status": "current",
        "title": "Synthetic page",
        "version": {"number": 1, "when": "2026-08-17T12:00:00+00:00"},
        "space": {"key": "DOCS"},
        "ancestors": [],
        "_links": {"webui": "/spaces/DOCS/pages/400100"},
    }


@pytest.mark.parametrize(
    "next_link",
    [
        "/rest/api/content?start=1&type=attachment",
        "/rest/api/content?start=1&ancestor=400000",
        "/rest/api/content?start=one",
        "/rest/api/content?start=1&start=2",
        "/rest/api/content?start=1&cursor=opaque",
        "/rest/api/content?limit=2",
    ],
)
async def test_direct_next_links_cannot_mutate_or_narrow_scope(next_link: str) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/rest/api/space/DOCS"):
            return httpx.Response(200, json={"key": "DOCS"})
        if request.url.path.endswith("/rest/api/content"):
            return httpx.Response(
                200,
                json={
                    "results": [_row()],
                    "_links": {"base": BASE, "next": next_link},
                },
            )
        raise AssertionError("unexpected synthetic request")

    config = _direct(include_attachments=False)
    connector = ConfluenceConnector(
        config,
        ConfluenceClient(config, transport=httpx.MockTransport(handle), clock=lambda: 0.0),
    )
    await connector.setup()
    try:
        with pytest.raises(ConnectorError, match="authoritative current-content pagination"):
            await drain(connector.discover(None))
    finally:
        await connector.teardown()


@pytest.mark.parametrize(
    ("next_link", "failure"),
    [
        (
            "https://wiki.example.test:444/rest/api/content?start=1",
            "another host",
        ),
        (
            "/rest/api/content?start=0",
            "pagination was sent back to a page it has already followed",
        ),
    ],
)
async def test_direct_next_links_fail_closed_on_cross_origin_and_loops(
    next_link: str, failure: str
) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/rest/api/space/DOCS"):
            return httpx.Response(200, json={"key": "DOCS"})
        if request.url.path.endswith("/rest/api/content"):
            return httpx.Response(
                200,
                json={
                    "results": [_row()],
                    "_links": {"base": BASE, "next": next_link},
                },
            )
        raise AssertionError("unexpected synthetic request")

    config = _direct(include_attachments=False)
    connector = ConfluenceConnector(
        config,
        ConfluenceClient(config, transport=httpx.MockTransport(handle), clock=lambda: 0.0),
    )
    await connector.setup()
    try:
        with pytest.raises(ConnectorError, match=failure):
            await drain(connector.discover(None))
    finally:
        await connector.teardown()


@pytest.mark.parametrize("results", [None, {}, ["not-an-object"]])
async def test_malformed_direct_results_never_become_an_empty_authoritative_page(
    results: object,
) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/rest/api/space/DOCS"):
            return httpx.Response(200, json={"key": "DOCS"})
        return httpx.Response(200, json={"results": results, "_links": {"base": BASE}})

    config = _direct(include_attachments=False)
    connector = ConfluenceConnector(
        config,
        ConfluenceClient(config, transport=httpx.MockTransport(handle), clock=lambda: 0.0),
    )
    await connector.setup()
    try:
        with pytest.raises(ConnectorError, match="malformed results page"):
            await drain(connector.discover(None))
    finally:
        await connector.teardown()


@pytest.mark.parametrize("malformed", ["status", "space", "version"])
async def test_direct_rows_fail_closed_when_required_membership_evidence_is_wrong(
    malformed: str,
) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/rest/api/space/DOCS"):
            return httpx.Response(200, json={"key": "DOCS"})
        row = _row()
        if malformed == "status":
            row["status"] = "archived"
        elif malformed == "space":
            row["space"] = {"key": "OPS"}
        else:
            row["version"] = {"when": "2026-08-17T12:00:00+00:00"}
        return httpx.Response(200, json={"results": [row], "_links": {"base": BASE}})

    config = _direct(include_attachments=False)
    connector = ConfluenceConnector(
        config,
        ConfluenceClient(config, transport=httpx.MockTransport(handle), clock=lambda: 0.0),
    )
    await connector.setup()
    try:
        with pytest.raises(ConnectorError, match="incomplete or out-of-scope metadata"):
            await drain(connector.discover(None))
    finally:
        await connector.teardown()


async def test_after_direct_full_completion_incremental_discovery_remains_cql_backed() -> None:
    page = FakePage(id="500100", title="Current", space="DOCS")
    instance = FakeConfluence(base_url=BASE, pages=[page])
    first = await connected(instance, _direct(include_attachments=False))
    try:
        await drain(first.discover(None))
        watermark = first.watermark
    finally:
        await first.teardown()
    assert watermark is not None

    instance.requests.clear()
    incremental = await connected(instance, _direct(include_attachments=False))
    try:
        await drain(incremental.discover(watermark))
    finally:
        await incremental.teardown()
    paths = [request.url.path for request in instance.requests]
    assert any(path.endswith("/rest/api/content/search") for path in paths)
    assert not any(path.endswith("/rest/api/content") for path in paths)
