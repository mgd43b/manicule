"""The Gmail connector through the real ingest pipeline and the real stores.

A parser's own suite once passed while every ``.eml`` was indexed with zero chunks through the
runner, so every criterion that depends on what the pipeline does with a message is driven here
through :class:`~manicule.ingest.pipeline.IngestPipeline` with durable acquisition on, over the
real document store, blob store and vector store. ``docs/connectors/gmail.md`` §4-§9.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import pytest_asyncio

from manicule.connectors import config as gmail_settings
from manicule.connectors.gmail import GmailConnector
from manicule.core.acquisition import SnapshotPromotionPolicy
from manicule.core.content import DocumentStatus
from manicule.ingest.middleware import MiddlewareRunner
from manicule.ingest.pipeline import IngestPipeline, RunReport
from manicule.ingest.reconcile import confirm_proposed_deletion, connector_scope, reconcile
from manicule.ingest.workers import InProcessRunner
from manicule.parsers.config import MailConfig
from manicule.parsers.mail import MailParser
from manicule.storage.blobs import BlobStore
from manicule.storage.docstore import SqliteDocStore
from manicule.storage.vectors import LanceVectorStore
from tests.connectors.fake_gmail import ADDRESS, FakeGmail, SimulatedTime
from tests.connectors.gmail_support import connector_over, mail, mailbox
from tests.fakes import HashEmbedder
from tests.ingest import fakes

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

SUBJECT = "Confidential merger plan"
LABEL = "Clients/Acme Corp"


@dataclass
class Harness:
    pipeline: IngestPipeline
    store: SqliteDocStore
    server: FakeGmail
    time: SimulatedTime

    def connector(self, **options: Any) -> GmailConnector:
        return connector_over(self.server, self.time, **options)

    async def sync(
        self, connector: GmailConnector, *, retain: bool | None = None, limit: int | None = None
    ) -> RunReport:
        return await self.pipeline.run(connector, retain_source_bytes=retain, limit=limit)

    def raw_fetches(self) -> int:
        return sum(1 for r in self.server.requests if r.url.params.get("format") == "raw")

    async def live(self) -> dict[str, DocumentStatus]:
        """Source ids of the live documents, messages only, with their status."""
        documents = await self.store.list_documents(limit=1_000)
        return {d.source_id: d.status for d in documents if "!/" not in d.source_id}


def mail_pipeline(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    vectors: LanceVectorStore,
    **options: Any,
) -> IngestPipeline:
    """The real pipeline, with the mail parser, durable acquisition and retained bytes."""
    chunker = fakes.BlockChunker()
    return IngestPipeline(
        store=store,
        acquisitions=store,
        chunker=chunker,
        embedder=HashEmbedder(),
        vectors=vectors,
        runner=InProcessRunner({"email": MailParser(MailConfig()), "lines": fakes.LineParser()}),
        resolve_chain=lambda _: ["email", "lines"],
        middleware=MiddlewareRunner([]),
        chunk_fingerprint=chunker.fingerprint,
        blobs=BlobStore(engine, data_dir),
        **options,
    )


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
    server, time = mailbox()
    return Harness(mail_pipeline(store, engine, data_dir, vectors), store, server, time)


async def opened(harness: Harness, **options: Any) -> GmailConnector:
    connector = harness.connector(**options)
    await connector.setup()
    return connector


# --- a message through the runner (§2, §5, §8) ---------------------------------------------------


@pytest.mark.parametrize("retain", [True, False])
async def test_a_message_with_an_attachment_is_indexed_through_the_runner(
    harness: Harness, retain: bool
) -> None:
    """Headers, body and attachment produce chunks — through the runner, not only ``parse()``."""
    message = harness.server.add_message(
        mail(SUBJECT, "Retry budgets are netted down twice.", attachment=True)
    )
    connector = await opened(harness)
    try:
        report = await harness.sync(connector, retain=retain)
    finally:
        await connector.teardown()

    document = await harness.store.find_document("mail", message.id)
    assert document is not None
    assert document.status is DocumentStatus.INDEXED
    assert report.error == ""
    chunks = await harness.store.document_chunks(document.id)
    text = "\n".join(chunk.text for chunk in chunks)
    assert chunks, "a message that parses must produce chunks"
    assert "Retry budgets" in text, "the body"
    assert SUBJECT in text, "the headers"
    attachment = await harness.store.find_document("mail", f"mail:{message.id}!/notes.txt")
    assert attachment is not None
    assert attachment.container_id == document.id
    assert (document.original_ref is not None) is retain


async def test_the_document_is_titled_by_the_subject_and_linked_by_the_message_id(
    harness: Harness,
) -> None:
    message = harness.server.add_message(mail(SUBJECT, message_id="<link.1@example.org>"))
    connector = await opened(harness)
    try:
        await harness.sync(connector)
    finally:
        await connector.teardown()

    document = await harness.store.find_document("mail", message.id)
    assert document is not None
    assert document.title == SUBJECT
    assert document.uri.endswith("#search/rfc822msgid:link.1%40example.org")
    assert document.metadata["gmail_thread_id"] == message.thread_id


async def test_a_message_deleted_between_discovery_and_fetch_does_not_stop_the_run(
    harness: Harness,
) -> None:
    server = harness.server
    kept = server.add_message(mail("Kept"))
    gone = server.add_message(mail("Gone"))
    server.delete_message(gone, still_listed=True)
    connector = await opened(harness)
    try:
        report = await harness.sync(connector)
    finally:
        await connector.teardown()

    assert kept.id in await harness.live()
    assert gone.id not in await harness.live()
    assert gone.id not in json.dumps(report.as_metadata(), default=str)


# --- change detection (§4) -----------------------------------------------------------------------


async def test_repeated_complete_runs_are_idempotent(harness: Harness) -> None:
    """Complete walks, not incremental no-ops: the history is expired before every run."""
    server = harness.server
    for number in range(5):
        server.add_message(mail(f"Message {number}", attachment=number == 0))
    connector = await opened(harness)

    async def snapshot() -> list[tuple[object, ...]]:
        found: list[tuple[object, ...]] = []
        for d in await harness.store.list_documents(limit=1_000):
            chunks = await harness.store.document_chunks(d.id)
            found.append(
                (
                    d.source_id,
                    d.status,
                    d.version_token,
                    d.content_hash,
                    tuple(c.id for c in chunks),
                )
            )
        return sorted(found, key=str)

    try:
        first = await harness.sync(connector)
        fetched = harness.raw_fetches()
        before = await snapshot()
        for _ in range(2):
            server.expire_history_through(server.history_id)
            listings = sum(1 for r in server.requests if r.url.path.endswith("/messages"))
            report = await harness.sync(connector)
            assert report.error == ""
            assert report.indexed == 0
            assert sum(1 for r in server.requests if r.url.path.endswith("/messages")) > listings
        assert await snapshot() == before
        first_pass = await reconciled(harness, connector)
        second_pass = await reconciled(harness, connector)
    finally:
        await connector.teardown()

    assert first.discovered == 5
    assert first.indexed == 6, "five messages and the attachment one of them carries"
    assert harness.raw_fetches() == fetched, "a stored message is never fetched again"
    assert first_pass.applied_count == second_pass.applied_count == 0
    assert await snapshot() == before


async def test_new_mail_arrives_incrementally_and_only_it_is_fetched(harness: Harness) -> None:
    harness.server.add_message(mail("Old"))
    connector = await opened(harness)
    try:
        await harness.sync(connector)
        fresh = harness.server.add_message(mail("New"))
        before = harness.raw_fetches()
        report = await harness.sync(connector)
    finally:
        await connector.teardown()

    assert report.indexed == 1
    assert harness.raw_fetches() - before == 1
    assert fresh.id in await harness.live()


@pytest.mark.parametrize("retain", [True, False])
async def test_an_expired_history_falls_back_to_a_walk_that_fetches_only_what_is_not_stored(
    harness: Harness, retain: bool
) -> None:
    for number in range(4):
        harness.server.add_message(mail(f"Message {number}"))
    connector = await opened(harness)
    try:
        await harness.sync(connector, retain=retain)
        fresh = harness.server.add_message(mail("While away"))
        harness.server.expire_history_through(harness.server.history_id)
        before = harness.raw_fetches()
        report = await harness.sync(connector, retain=retain)
    finally:
        await connector.teardown()

    assert report.error == ""
    assert report.indexed == 1
    assert harness.raw_fetches() - before == 1, "the walk listed ids and fetched the new one only"
    assert fresh.id in await harness.live()


async def test_history_that_expires_part_way_through_a_run_still_indexes_every_message_once(
    harness: Harness,
) -> None:
    """The fallback walk re-yields ids the half-finished incremental walk already yielded."""
    server = harness.server
    for number in range(3):
        server.add_message(mail(f"Message {number}"))
    connector = await opened(harness)
    try:
        await harness.sync(connector)
        for number in range(3, 8):
            server.add_message(mail(f"Message {number}"))

        def expires_on_the_second_page(request: httpx.Request) -> None:
            if request.url.path.endswith("/history") and "pageToken" in request.url.params:
                server.history_floor = server.history_id + 1

        server.before_request = expires_on_the_second_page
        report = await harness.sync(connector)
    finally:
        await connector.teardown()

    assert report.error == ""
    assert not report.retry_required
    assert len(await harness.live()) == 8
    assert harness.raw_fetches() == 8, "each message was fetched exactly once"


async def test_a_representation_bump_mid_run_starts_a_fresh_complete_walk(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A history position was stored and a run under it was interrupted; the bump discards both."""
    server = harness.server
    for number in range(3):
        server.add_message(mail(f"Message {number}"))
    old = await opened(harness)
    try:
        await harness.sync(old)
        for number in range(3, 6):
            server.add_message(mail(f"Message {number}"))
        interrupted = await harness.sync(old, limit=1)
    finally:
        await old.teardown()
    assert interrupted.limited
    fetched = harness.raw_fetches()

    monkeypatch.setattr(gmail_settings, "REPRESENTATION_VERSION", "rfc822/2")
    new = await opened(harness)
    try:
        report = await harness.sync(new)
        stored = await harness.store.list_documents(limit=1_000)
    finally:
        await new.teardown()

    assert report.error == ""
    assert not report.retry_required
    assert len(await harness.live()) == 6
    assert harness.raw_fetches() - fetched == 6, "every in-scope message, not just the new ones"
    assert {d.version_token for d in stored if "!/" not in d.source_id} == {"rfc822/2"}


async def test_a_representation_bump_refetches_every_stored_message(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    for number in range(3):
        harness.server.add_message(mail(f"Message {number}"))
    old = await opened(harness)
    try:
        await harness.sync(old)
    finally:
        await old.teardown()
    fetched = harness.raw_fetches()

    monkeypatch.setattr(gmail_settings, "REPRESENTATION_VERSION", "rfc822/2")
    new = await opened(harness)
    try:
        await harness.sync(new)
    finally:
        await new.teardown()

    assert harness.raw_fetches() - fetched == 3


# --- deletions (§7) ------------------------------------------------------------------------------


async def reconciled(harness: Harness, connector: GmailConnector, *, fraction: float = 1.0):  # noqa: ANN201
    return await reconcile(connector, harness.store, max_delete_fraction=fraction)


async def test_a_message_that_left_the_mailbox_leaves_the_index_at_the_next_reconciliation(
    harness: Harness,
) -> None:
    server = harness.server
    kept = server.add_message(mail("Kept"))
    deleted = server.add_message(mail("Deleted"))
    trashed = server.add_message(mail("Trashed"))
    spam = server.add_message(mail("Spam"))
    labeled = server.add_message(mail("Labeled"))
    connector = await opened(harness, exclude_labels=[LABEL])
    try:
        await harness.sync(connector)
        assert len(await harness.live()) == 5
        server.delete_message(deleted)
        server.trash(trashed)
        server.add_label(spam, "SPAM")
        server.add_label(labeled, LABEL)
        await harness.sync(connector)
        assert len(await harness.live()) == 5, "an incremental sync cannot see a deletion"

        result = await reconciled(harness, connector)
    finally:
        await connector.teardown()

    assert result.clean
    assert result.applied_count == 4
    assert set(await harness.live()) == {kept.id}


async def test_a_partial_inventory_deletes_nothing(harness: Harness) -> None:
    for number in range(6):
        harness.server.add_message(mail(f"Message {number}"))
    connector = connector_over(harness.server, harness.time, max_retries=0)
    await connector.setup()
    try:
        await harness.sync(connector)
        harness.server.before_request = None
        calls = 0

        def fails_after_the_first_page(request: object) -> None:
            nonlocal calls
            calls += 1
            if calls == 4:  # profile, then listing: DRAFT, page one, page two
                harness.server.fail(503, times=50)

        harness.server.before_request = fails_after_the_first_page
        result = await reconciled(harness, connector)
    finally:
        await connector.teardown()

    assert result.error
    assert result.applied_count == 0
    assert len(await harness.live()) == 6


async def test_the_deletion_ceiling_holds_a_large_removal_for_confirmation(
    harness: Harness,
) -> None:
    messages = [harness.server.add_message(mail(f"Message {n}")) for n in range(10)]
    connector = await opened(harness)
    try:
        await harness.sync(connector)
        for message in messages[:5]:
            harness.server.trash(message)

        refused = await reconcile(connector, harness.store)  # the default 10% ceiling
        assert refused.refused
        assert refused.applied_count == 0
        assert len(await harness.live()) == 10
        applied = await confirm_proposed_deletion(
            connector.name, harness.store, scope=connector_scope(connector)
        )
    finally:
        await connector.teardown()

    assert applied.applied_count == 5
    assert len(await harness.live()) == 5


# --- restore from Trash (§7) ---------------------------------------------------------------------


async def test_a_message_trashed_reconciled_away_and_restored_is_indexed_again(
    harness: Harness,
) -> None:
    server = harness.server
    message = server.add_message(mail(SUBJECT))
    server.add_message(mail("Bystander"))
    connector = await opened(harness)
    try:
        await harness.sync(connector)
        server.trash(message)
        await harness.sync(connector)
        gone = await reconciled(harness, connector)
        assert gone.applied_count == 1
        assert message.id not in await harness.live()

        server.restore(message)
        report = await harness.sync(connector)
    finally:
        await connector.teardown()

    assert report.error == ""
    assert report.indexed == 1, "restored, not skipped as unchanged"
    live = await harness.live()
    assert live[message.id] is DocumentStatus.INDEXED
    document = await harness.store.find_document("mail", message.id)
    assert document is not None
    assert await harness.store.document_chunks(document.id)


async def test_a_message_restored_from_trash_with_another_excluded_label_stays_out(
    harness: Harness,
) -> None:
    server = harness.server
    message = server.add_message(mail(SUBJECT), labels=("INBOX", LABEL))
    server.add_message(mail("Bystander"))
    connector = await opened(harness, exclude_labels=[LABEL])
    try:
        await harness.sync(connector)
        server.trash(message)
        await harness.sync(connector)
        server.restore(message)
        report = await harness.sync(connector)
    finally:
        await connector.teardown()

    assert report.indexed == 0
    assert message.id not in await harness.live()


# --- narrowing scope (§3) ------------------------------------------------------------------------


async def test_adding_an_excluded_label_forces_a_complete_walk_and_reconciliation_removes_them(
    harness: Harness,
) -> None:
    server = harness.server
    keep = server.add_message(mail("Keep"))
    drop = server.add_message(mail("Drop"), labels=("INBOX", LABEL))
    wide = await opened(harness)
    try:
        await harness.sync(wide)
        assert set(await harness.live()) == {keep.id, drop.id}
    finally:
        await wide.teardown()

    narrow = await opened(harness, exclude_labels=[LABEL])
    try:
        assert narrow.scope_fingerprint != wide.scope_fingerprint
        before = len(server.requests)
        report = await harness.sync(narrow)
        walked = [r.url.path.removeprefix("/gmail/v1/users/me") for r in server.requests[before:]]
        assert report.error == ""
        assert "/history" not in walked, "the stored position was discarded"
        assert "/messages" in walked, "a complete walk under the new scope"
        assert set(await harness.live()) == {keep.id, drop.id}, "a walk removes nothing"

        result = await reconciled(harness, narrow)
    finally:
        await narrow.teardown()

    assert result.applied_count == 1
    assert set(await harness.live()) == {keep.id}


async def test_narrowing_past_the_deletion_ceiling_waits_for_confirmation(
    harness: Harness,
) -> None:
    server = harness.server
    for number in range(6):
        server.add_message(mail(f"Keep {number}"))
    for number in range(4):
        server.add_message(mail(f"Drop {number}"), labels=("INBOX", LABEL))
    wide = await opened(harness)
    try:
        await harness.sync(wide)
    finally:
        await wide.teardown()
    narrow = await opened(harness, exclude_labels=[LABEL])
    try:
        await harness.sync(narrow)
        refused = await reconcile(narrow, harness.store)
        assert refused.refused
        assert len(await harness.live()) == 10, "nothing leaves until a person confirms"
        applied = await confirm_proposed_deletion(
            narrow.name, harness.store, scope=connector_scope(narrow)
        )
    finally:
        await narrow.teardown()

    assert applied.applied_count == 4
    assert len(await harness.live()) == 6


# --- diagnostics are aggregate (§10) -------------------------------------------------------------


async def test_no_surface_carries_an_address_a_subject_a_label_or_a_message_id(
    harness: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    server = harness.server
    message = server.add_message(mail(SUBJECT, message_id="<secret.id@example.org>"))
    other = server.add_message(mail("Another subject"))
    excluded = server.add_message(mail("Excluded subject"), labels=("INBOX", LABEL))
    connector = await opened(harness, exclude_labels=[LABEL])
    surfaces: list[str] = []
    try:
        surfaces.append(json.dumps((await harness.sync(connector)).as_metadata(), default=str))
        server.trash(other)
        server.expire_history_through(server.history_id)
        server.add_message(mail("Later"))
        surfaces.append(json.dumps((await harness.sync(connector)).as_metadata(), default=str))
        result = await reconciled(harness, connector)
        surfaces.append(json.dumps(asdict(result), default=str))
        surfaces.append((await connector.health()).detail)
        surfaces.append(json.dumps([m.name for m in connector.metrics()]))
        server.switch_mailbox("someone.else@example.org")
        refused = await harness.sync(connector)
        assert refused.error_type == "MailboxMismatchError"
        surfaces.append(json.dumps(refused.as_metadata(), default=str))
        surfaces.append(refused.error + refused.error_message)
    finally:
        await connector.teardown()

    # The database driver's own DEBUG records print every row it writes, the private document
    # store's included; they are the store's business, not a diagnostic this connector emits.
    ours = ("manicule", "httpx", "httpcore")
    surfaces.extend(
        record.getMessage() for record in caplog.records if record.name.startswith(ours)
    )
    assert any(record.name == "httpx" for record in caplog.records), "the HTTP log was exercised"
    everything = "\n".join(surfaces).casefold()
    for secret in (
        "example.org",
        ADDRESS,
        SUBJECT,
        "another subject",
        "secret.id",
        LABEL,
        "excluded subject",
        server.label_id(LABEL),
        message.id,
        other.id,
        excluded.id,
    ):
        assert secret.casefold() not in everything, secret


@pytest.mark.parametrize(
    "policy", [SnapshotPromotionPolicy.REQUIRE_COMPLETE, SnapshotPromotionPolicy.ALLOW_OMISSIONS]
)
async def test_a_message_over_the_fetch_cap_is_refused_and_counted_never_truncated(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    vectors: LanceVectorStore,
    policy: SnapshotPromotionPolicy,
) -> None:
    """The pipeline's cap, not a Gmail setting: the oversize message is an omission, in counts.

    Under the default policy a run with an omission is not complete and advances nothing, which is
    the pipeline's rule for any body it could not acquire; ``allow_omissions`` is how an
    installation that knows it has one lets the rest of the mailbox move on.
    """
    pipeline = mail_pipeline(
        store, engine, data_dir, vectors, max_fetch_bytes=2_000, snapshot_policy=policy
    )
    server, time = mailbox()
    small = server.add_message(mail("Small"))
    huge = server.add_message(mail("Huge", "x" * 10_000))
    connector = connector_over(server, time)
    await connector.setup()
    try:
        report = await pipeline.run(connector)
    finally:
        await connector.teardown()

    live = {d.source_id for d in await store.list_documents(limit=100)}
    assert small.id in live
    assert huge.id not in live, "never truncated, never half-indexed"
    assert report.snapshot_omissions == 1
    assert report.retry_required, "the omission is still owed a retry (the cap may be raised)"
    assert report.watermark_advanced is (policy is SnapshotPromotionPolicy.ALLOW_OMISSIONS)
    assert SUBJECT not in json.dumps(report.as_metadata(), default=str)
    assert huge.id not in json.dumps(report.as_metadata(), default=str)
