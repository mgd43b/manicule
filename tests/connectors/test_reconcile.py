"""Deletion detection: the part a watermark sync cannot do.

CQL returns what exists. A page deleted since the last sync simply stops appearing, so an
incremental sync never learns it is gone and the index serves it forever
(``docs/connectors/confluence.md`` §3). ``reconcile`` is the periodic id-only diff that fixes
that, and it is part of the protocol so no connector can quietly omit it.

The dangerous failure is the opposite one: a partial enumeration diffed against the stored set
marks everything not yet reached as deleted (``docs/ingest.md`` §11.1). So the interesting
assertions here are about what happens when reconciliation *fails*.
"""

from __future__ import annotations

import httpx
import pytest

from manicule.connectors import ConnectorError
from manicule.connectors.client import ConfluenceClient
from manicule.connectors.confluence import ConfluenceConnector
from tests.connectors.fake_confluence import FakeAttachment, FakeConfluence, FakePage
from tests.connectors.support import Waits, cloud_config, connected, drain


def _instance() -> FakeConfluence:
    return FakeConfluence(
        pages=[
            FakePage(id="1", title="A", space="ENG"),
            FakePage(id="2", title="B", space="ENG"),
            FakePage(id="3", title="C", space="ENG"),
        ],
        attachments=[
            FakeAttachment(id="att-9", title="d.pdf", space="ENG", page_id="1", page_title="A")
        ],
        page_size=2,
    )


async def test_reconciliation_reports_everything_that_still_exists() -> None:
    """Attachments included: a document the pass omits is a document the pipeline deletes."""
    instance = _instance()
    connector = await connected(instance)
    try:
        existing = await drain(connector.reconcile())
    finally:
        await connector.teardown()

    assert sorted(existing) == ["1", "2", "3", "att-9"]


async def test_reconciliation_asks_for_ids_and_nothing_else() -> None:
    """Ids only — no bodies, no versions — is what makes a full enumeration affordable weekly."""
    instance = _instance()
    connector = await connected(instance)
    try:
        await drain(connector.reconcile())
    finally:
        await connector.teardown()

    searches = [r for r in instance.requests if r.url.path.endswith("/content/search")]
    assert searches
    assert all("expand" not in request.url.params for request in searches)


async def test_a_deleted_page_is_absent_from_reconciliation_and_present_in_no_sync() -> None:
    """The whole reason the pass exists, demonstrated end to end.

    An incremental sync after the deletion returns nothing at all — which is exactly what an
    incremental sync returns when nothing has changed. Only the diff can tell them apart.
    """
    instance = _instance()
    connector = await connected(instance)
    try:
        before = await drain(connector.reconcile())
        instance.delete("2")
        changed = await drain(connector.discover(connector.watermark))
        after = await drain(connector.reconcile())
    finally:
        await connector.teardown()

    assert "2" in before
    assert "2" not in [document.source_id for document in changed]
    assert "2" not in after
    assert set(before) - set(after) == {"2"}


async def test_a_failure_part_way_through_raises_rather_than_returning_a_prefix() -> None:
    """The ids seen so far are a prefix, not the truth.

    Diffing a prefix against the stored set marks everything not yet enumerated as deleted: one
    transient error, and the corpus is soft-deleted. The connector's part of the guard is to
    fail loudly rather than to return what it has.
    """
    instance = _instance()
    connector = await connected(instance)
    seen: list[str] = []

    async def walk() -> None:
        async for source_id in connector.reconcile():
            seen.append(source_id)
            if len(seen) == 2:
                instance.throttles.append({"Retry-After": "999"})

    try:
        with pytest.raises(ConnectorError):
            await walk()
    finally:
        await connector.teardown()

    assert len(seen) < 4, "the enumeration must not have completed"


async def test_an_empty_source_is_still_a_complete_enumeration() -> None:
    """A space with nothing in it reports nothing, and that is a fact rather than a failure.

    The pipeline's deletion ceiling is what protects an index whose source really did empty;
    inventing a refusal here would make an ordinary empty space unsyncable.
    """
    instance = FakeConfluence(pages=[], spaces={"ENG": "Engineering"})
    connector = await connected(instance)
    try:
        assert await drain(connector.reconcile()) == []
    finally:
        await connector.teardown()


async def test_a_source_that_stops_answering_does_not_look_like_an_empty_source() -> None:
    """The difference between "everything is gone" and "nothing answered" is the whole index."""

    def handle(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(503, json={})

    config = cloud_config()
    client = ConfluenceClient(
        config, transport=httpx.MockTransport(handle), sleep=Waits(), clock=lambda: 0.0
    )
    connector = ConfluenceConnector(config, client)
    await connector.setup()
    try:
        with pytest.raises(ConnectorError):
            await drain(connector.reconcile())
    finally:
        await connector.teardown()
