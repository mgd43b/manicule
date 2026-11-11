"""The Outlook connector through the real ingest pipeline and the real stores.

A parser's own suite once passed while every ``.eml`` was indexed with zero chunks through the
runner, so every criterion that depends on what the pipeline does with a message or an event is
driven here through :class:`~manicule.ingest.pipeline.IngestPipeline` with durable acquisition on,
over the real document store, blob store and vector store. ``docs/connectors/outlook.md`` §4-§8.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio

from manicule.connectors.outlook import EVENT_PREFIX, MESSAGE_PREFIX, OutlookConnector
from manicule.core.content import DocumentStatus
from manicule.core.mail_ids import MAIL_MESSAGE_ID_KEY, MAIL_THREAD_ID_KEY
from manicule.ingest.pipeline import IngestPipeline, RunReport
from manicule.ingest.reconcile import confirm_proposed_deletion, connector_scope, reconcile
from manicule.storage.docstore import SqliteDocStore
from manicule.storage.vectors import LanceVectorStore
from tests.connectors.fake_gmail import SimulatedTime
from tests.connectors.fake_graph import FakeEvent, FakeGraph, mail
from tests.connectors.outlook_support import connector_over, pipeline_for, tenant
from tests.fakes import HashEmbedder

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

SUBJECT = "Confidential merger plan"


@dataclass
class Harness:
    pipeline: IngestPipeline
    store: SqliteDocStore
    server: FakeGraph
    time: SimulatedTime

    async def connector(self, **options: Any) -> OutlookConnector:
        connector = connector_over(self.server, self.time, **options)
        await connector.setup()
        return connector

    async def sync(self, connector: OutlookConnector, *, retain: bool | None = None) -> RunReport:
        return await self.pipeline.run(connector, retain_source_bytes=retain)

    def fetches(self, fragment: str) -> int:
        return sum(1 for r in self.server.requests if fragment in r.url.path)

    async def live(self) -> dict[str, DocumentStatus]:
        documents = await self.store.list_documents(limit=1_000)
        return {d.source_id: d.status for d in documents if "!/" not in d.source_id}


@pytest_asyncio.fixture
async def vectors(data_dir: Path) -> AsyncIterator[LanceVectorStore]:
    store = LanceVectorStore(data_dir / "vectors")
    await store.ensure_ready(HashEmbedder().fingerprint)
    yield store
    await store.teardown()


@pytest_asyncio.fixture
async def harness(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path, vectors: LanceVectorStore
) -> Harness:
    server, time = tenant()
    return Harness(pipeline_for(store, engine, data_dir, vectors), store, server, time)


def planning() -> FakeEvent:
    return FakeEvent(
        "e1",
        "Quarterly planning",
        "2026-03-02T14:00:00",
        "2026-03-02T15:00:00",
        location="Room 4",
        body="Agenda for the quarter.",
        attendees=("Lee Example <lee@example.org>",),
    )


# --- through the runner (§5, §7) -------------------------------------------------------------


@pytest.mark.parametrize("retain", [True, False])
async def test_a_message_is_indexed_with_chunks_through_the_runner(
    harness: Harness, retain: bool
) -> None:
    graph_id = harness.server.add_message(mail(SUBJECT, "Retry budgets are netted down twice."))
    connector = await harness.connector(calendar=False)
    try:
        report = await harness.sync(connector, retain=retain)
    finally:
        await connector.teardown()

    document = await harness.store.find_document("work", MESSAGE_PREFIX + graph_id)
    assert document is not None
    assert document.status is DocumentStatus.INDEXED
    assert report.error == ""
    chunks = await harness.store.document_chunks(document.id)
    text = "\n".join(chunk.text for chunk in chunks)
    assert chunks, "a message that parses must produce chunks"
    assert "Retry budgets" in text
    assert SUBJECT in text, "the headers"
    assert (document.original_ref is not None) is retain


async def test_the_document_is_titled_by_the_subject_linked_by_graph_and_carries_the_two_ids(
    harness: Harness,
) -> None:
    graph_id = harness.server.add_message(mail(SUBJECT), conversation="conv-42")
    connector = await harness.connector(calendar=False)
    try:
        await harness.sync(connector)
    finally:
        await connector.teardown()

    document = await harness.store.find_document("work", MESSAGE_PREFIX + graph_id)
    assert document is not None
    assert document.title == SUBJECT
    assert document.uri.startswith("https://outlook.office365.com/owa/")
    assert document.metadata[MAIL_THREAD_ID_KEY] == "conv-42"
    assert str(document.metadata[MAIL_MESSAGE_ID_KEY]).endswith("@example.org")


async def test_an_event_is_indexed_through_the_calendar_parser(harness: Harness) -> None:
    harness.server.add_event(planning())
    connector = await harness.connector(mail=False)
    try:
        report = await harness.sync(connector)
    finally:
        await connector.teardown()

    document = await harness.store.find_document("work", EVENT_PREFIX + "e1")
    assert document is not None
    assert document.status is DocumentStatus.INDEXED
    assert report.error == ""
    assert document.title == "Quarterly planning"
    text = "\n".join(c.text for c in await harness.store.document_chunks(document.id))
    assert "Event: Quarterly planning" in text
    assert "When: 2026-03-02 14:00 UTC to 2026-03-02 15:00 UTC" in text
    assert "Where: Room 4" in text
    assert "Lee Example <lee@example.org>" in text
    assert "Agenda for the quarter." in text


# --- change detection (§4) --------------------------------------------------------------------


async def test_a_second_sync_fetches_nothing_that_is_already_stored(harness: Harness) -> None:
    server = harness.server
    first = server.add_message(mail("First"))
    server.add_message(mail("Second"), "Archive")
    server.add_event(planning())
    connector = await harness.connector()
    try:
        initial = await harness.sync(connector)
        message_fetches = harness.fetches("/$value")
        event_fetches = harness.fetches("/me/events/")
        server.touch(first)
        again = await harness.sync(connector)
    finally:
        await connector.teardown()

    assert initial.discovered == 3
    assert initial.indexed == 3
    assert again.error == ""
    assert again.indexed == 0
    assert harness.fetches("/$value") == message_fetches, "a flag change is not a refetch"
    assert harness.fetches("/me/events/") == event_fetches, "an unchanged event is not either"


async def test_an_edited_event_is_fetched_again_and_nothing_else_is(harness: Harness) -> None:
    server = harness.server
    server.add_message(mail("Mail"))
    server.add_event(planning())
    connector = await harness.connector()
    try:
        await harness.sync(connector)
        messages = harness.fetches("/$value")
        server.edit_event("e1", subject="Moved to Thursday", location="Room 9")
        report = await harness.sync(connector)
    finally:
        await connector.teardown()

    document = await harness.store.find_document("work", EVENT_PREFIX + "e1")
    assert document is not None
    assert report.indexed == 1
    assert harness.fetches("/$value") == messages
    assert document.title == "Moved to Thursday"
    text = "\n".join(c.text for c in await harness.store.document_chunks(document.id))
    assert "Room 9" in text
    assert "Room 4" not in text


async def test_a_new_message_in_a_second_sync_is_the_only_one_fetched(harness: Harness) -> None:
    server = harness.server
    server.add_message(mail("Old"))
    connector = await harness.connector(calendar=False)
    try:
        await harness.sync(connector)
        server.add_message(mail("New"))
        report = await harness.sync(connector)
    finally:
        await connector.teardown()

    assert report.indexed == 1
    assert harness.fetches("/$value") == 2


async def test_a_message_deleted_between_discovery_and_fetch_does_not_stop_the_run(
    harness: Harness,
) -> None:
    server = harness.server
    kept = server.add_message(mail("Kept"))
    vanished = server.add_message(mail("Vanishes"))
    server.vanish_on_fetch(vanished)
    connector = await harness.connector(calendar=False)
    try:
        report = await harness.sync(connector)
    finally:
        await connector.teardown()

    assert MESSAGE_PREFIX + kept in await harness.live()
    assert MESSAGE_PREFIX + vanished not in await harness.live()
    assert report.error == ""


# --- deletion is reconciliation's (§6) -----------------------------------------------------------


async def test_a_deleted_message_leaves_the_index_only_at_reconciliation(harness: Harness) -> None:
    server = harness.server
    kept = server.add_message(mail("Kept"))
    gone = server.add_message(mail("Deleted"))
    moved_to_trash = server.add_message(mail("Trashed"))
    connector = await harness.connector(calendar=False)
    try:
        await harness.sync(connector)
        server.delete(gone)
        server.move(moved_to_trash, "Deleted Items")
        await harness.sync(connector)
        assert len(await harness.live()) == 3, "an incremental sync cannot see a deletion"

        result = await reconcile(connector, harness.store, max_delete_fraction=1.0)
    finally:
        await connector.teardown()

    assert result.clean
    assert result.applied_count == 2, "deleted, and moved into an always-excluded folder"
    assert set(await harness.live()) == {MESSAGE_PREFIX + kept}


async def test_an_event_that_left_the_window_leaves_the_index_at_reconciliation(
    harness: Harness,
) -> None:
    server = harness.server
    server.add_message(mail("Mail"))
    server.add_event(planning())
    connector = await harness.connector()
    try:
        await harness.sync(connector)
        server.events["e1"].start = "2024-01-01T09:00:00"
        server.events["e1"].end = "2024-01-01T10:00:00"
        result = await reconcile(connector, harness.store, max_delete_fraction=1.0)
    finally:
        await connector.teardown()

    assert result.applied_count == 1
    assert EVENT_PREFIX + "e1" not in await harness.live()


async def test_a_partial_inventory_deletes_nothing(harness: Harness) -> None:
    for number in range(6):
        harness.server.add_message(mail(f"Message {number}"))
    connector = connector_over(harness.server, harness.time, max_retries=0, calendar=False)
    await connector.setup()
    try:
        await harness.sync(connector)
        harness.server.throttle.extend([(503, None)] * 50)
        result = await reconcile(connector, harness.store, max_delete_fraction=1.0)
    finally:
        await connector.teardown()

    assert result.error
    assert result.applied_count == 0
    assert len(await harness.live()) == 6


async def test_the_deletion_ceiling_holds_a_large_removal_for_confirmation(
    harness: Harness,
) -> None:
    messages = [harness.server.add_message(mail(f"Message {n}")) for n in range(10)]
    connector = await harness.connector(calendar=False)
    try:
        await harness.sync(connector)
        for message in messages[:5]:
            harness.server.delete(message)

        refused = await reconcile(connector, harness.store)  # the default 10% ceiling
        assert refused.refused
        assert len(await harness.live()) == 10
        applied = await confirm_proposed_deletion(
            connector.name, harness.store, scope=connector_scope(connector)
        )
    finally:
        await connector.teardown()

    assert applied.applied_count == 5
    assert len(await harness.live()) == 5


async def test_narrowing_the_scope_removes_nothing_until_reconciliation(harness: Harness) -> None:
    server = harness.server
    keep = server.add_message(mail("Keep"))
    drop = server.add_message(mail("Drop"), "Archive")
    wide = await harness.connector(calendar=False)
    try:
        await harness.sync(wide)
    finally:
        await wide.teardown()

    narrow = await harness.connector(calendar=False, exclude_folders=["Archive"])
    try:
        assert narrow.scope_fingerprint != wide.scope_fingerprint
        report = await harness.sync(narrow)
        assert report.error == ""
        assert set(await harness.live()) == {MESSAGE_PREFIX + keep, MESSAGE_PREFIX + drop}

        result = await reconcile(narrow, harness.store, max_delete_fraction=1.0)
    finally:
        await narrow.teardown()

    assert result.applied_count == 1
    assert set(await harness.live()) == {MESSAGE_PREFIX + keep}


async def test_expired_delta_state_costs_a_complete_walk_and_loses_nothing(
    harness: Harness,
) -> None:
    server = harness.server
    server.add_message(mail("First"))
    server.add_message(mail("Second"), "Archive")
    connector = await harness.connector(calendar=False)
    try:
        await harness.sync(connector)
        fetched = harness.fetches("/$value")
        server.expire_deltas()
        report = await harness.sync(connector)
        live = await harness.live()
    finally:
        await connector.teardown()

    assert report.error == ""
    assert report.indexed == 0
    assert len(live) == 2
    assert harness.fetches("/$value") == fetched, "a stored message is never fetched again"
