"""The Gmail connector against the synthetic mailbox: discovery, scope, fetch, reconciliation.

``docs/connectors/gmail.md`` §2-§7. The pipeline-level criteria — a message driven through the
runner, restore from Trash, narrowing, durable acquisition — are in ``test_gmail_pipeline.py``.
"""

from __future__ import annotations

import httpx
import pytest

from manicule.connectors import config as gmail_settings
from manicule.connectors.errors import ConnectorError, MailboxMismatchError, RemoteError
from manicule.core.content import RawDocument
from manicule.core.errors import ConfigError
from manicule.core.provenance import PROVENANCE_KEY, Provenance
from manicule.core.sources import DocRef
from manicule.testing import closing
from tests.connectors.fake_gmail import ADDRESS, FakeGmail, SimulatedTime
from tests.connectors.gmail_support import (
    connector_over,
    drain,
    ids,
    mail,
    mailbox,
    opened,
)

SCOPED_LABELS = (
    "Clients/Acme Corp",
    "Receipts 2026",
    "Archive/Old Mail/2019",
    "CATEGORY_PROMOTIONS",
)
"""Names with spaces and slashes, a nested label, and a category label (search spells that one
``category:promotions``, which is why no path translates a label into a query)."""


def _canonical(fetched: RawDocument) -> str:
    """The address a citation opens: the provenance record's, which the pipeline stores."""
    record = Provenance.from_metadata(fetched.metadata)
    assert record is not None
    assert record.source is not None
    return record.source.canonical_uri


def paths(server: FakeGmail) -> list[str]:
    return [request.url.path.removeprefix("/gmail/v1/users/me") for request in server.requests]


# --- identity and fetch (§2, §5) ---------------------------------------------------------------


async def test_a_discovered_message_is_identified_by_its_gmail_id_with_a_constant_token() -> None:
    server, time = mailbox()
    first = server.add_message()
    second = server.add_message()

    async with opened(server, time) as connector:
        found = await drain(connector)

    assert sorted(ids(found)) == sorted([first.id, second.id])
    assert {doc.version_token for doc in found} == {gmail_settings.REPRESENTATION_VERSION}
    assert {doc.media_type for doc in found} == {"message/rfc822"}
    assert all("#all/" in doc.ref.uri for doc in found), "no Message-ID is known before the fetch"


async def test_a_fetch_returns_the_raw_message_with_the_rfc822msgid_link_and_its_metadata() -> None:
    server, time = mailbox()
    raw = mail("Quarterly figures", message_id="<abc.123@example.org>")
    message = server.add_message(raw, thread_id="thread-7", internal_date_ms=1_760_000_123_000)

    async with opened(server, time) as connector:
        fetched = await connector.fetch(DocRef(source_id=message.id, uri="https://unused.example"))

    assert fetched.media_type == "message/rfc822"
    assert fetched.as_bytes() == raw
    assert fetched.source_id == message.id
    assert fetched.uri == "https://unused.example", "the address discovery gave"
    assert _canonical(fetched) == (
        "https://mail.google.com/mail/?authuser=sync.user%40example.org"
        "#search/rfc822msgid:abc.123%40example.org"
    )
    assert fetched.metadata["gmail_thread_id"] == "thread-7"
    assert fetched.metadata["internal_date"] == "2025-10-09T08:55:23+00:00"
    assert fetched.metadata["version_token"] == gmail_settings.REPRESENTATION_VERSION


@pytest.mark.parametrize(
    ("header", "encoded"),
    [
        ("<a+b@example.org>", "a%2Bb%40example.org"),
        ("<100%25#1@example.org>", "100%2525%231%40example.org"),
    ],
)
async def test_a_message_id_is_percent_encoded_because_plus_percent_and_hash_are_not_safe_raw(
    header: str, encoded: str
) -> None:
    """A ``#`` ends the search term and a ``+`` reads as a space; neither may go in raw."""
    server, time = mailbox()
    message = server.add_message(mail("Encoded", message_id=header))

    async with opened(server, time) as connector:
        fetched = await connector.fetch(DocRef(source_id=message.id, uri="https://unused.example"))

    assert _canonical(fetched).endswith(f"#search/rfc822msgid:{encoded}")


async def test_a_message_without_a_message_id_links_by_its_gmail_id() -> None:
    server, time = mailbox()
    message = server.add_message(mail("Anonymous", with_id=False))

    async with opened(server, time) as connector:
        fetched = await connector.fetch(DocRef(source_id=message.id, uri="https://unused.example"))

    assert _canonical(fetched).endswith(f"#all/{message.id}")


async def test_a_message_deleted_between_discovery_and_fetch_is_not_found() -> None:
    server, time = mailbox()
    message = server.add_message()

    async with opened(server, time) as connector:
        from manicule.connectors.errors import NotFoundError  # noqa: PLC0415

        server.delete_message(message)
        with pytest.raises(NotFoundError):
            await connector.fetch(DocRef(source_id=message.id, uri="https://unused.example"))


# --- the full walk (§4.1) ----------------------------------------------------------------------


async def test_the_full_walk_takes_its_watermark_from_before_it_lists() -> None:
    """A message that arrives mid-walk may or may not be in the listing; the next sync sees it."""
    server, time = mailbox()
    for _ in range(4):
        server.add_message()
    position_before = server.history_id
    arrived: list[str] = []

    def arrives_when_listing_starts(request: httpx.Request) -> None:
        if request.url.path.endswith("/messages") and not arrived:
            arrived.append(server.add_message().id)

    async with opened(server, time) as connector:
        server.before_request = arrives_when_listing_starts
        await drain(connector)
        watermark = connector.watermark
        later = await drain(connector, watermark)

    assert watermark is not None
    assert int(watermark.value) == position_before, "captured before the first listing"
    assert arrived[0] in ids(later), "the next incremental sync sees what arrived mid-walk"


async def test_each_listing_page_is_one_batch() -> None:
    server, time = mailbox()
    for _ in range(5):
        server.add_message()

    async with opened(server, time) as connector:
        batches = [len(batch) async for batch in connector.discover_batches(None)]

    assert batches == [2, 2, 1]


async def test_the_watermark_exists_only_after_a_complete_walk() -> None:
    server, time = mailbox()
    for _ in range(5):
        server.add_message()

    async with opened(server, time) as connector:
        assert connector.watermark is None
        async with closing(connector.discover_batches(None)) as batches:
            await anext(batches)
            assert connector.watermark is None, "a walk abandoned part-way advances nothing"
        await drain(connector)
        assert connector.watermark is not None


# --- scope (§3) --------------------------------------------------------------------------------


def populated_mailbox() -> tuple[FakeGmail, SimulatedTime, dict[str, str]]:
    """One message per kind: kept, and one carrying each thing that takes a message out."""
    server, time = mailbox()
    kept = server.add_message(labels=("INBOX",))
    labeled_in_scope = server.add_message(labels=("INBOX", "STARRED"))
    kinds = {
        "kept": kept.id,
        "starred": labeled_in_scope.id,
        "draft": server.add_message(labels=("DRAFT",)).id,
        "spam": server.add_message(labels=("SPAM",)).id,
        "trash": server.add_message(labels=("TRASH",)).id,
    }
    for label in SCOPED_LABELS:
        kinds[label] = server.add_message(labels=("INBOX", label)).id
    return server, time, kinds


async def test_drafts_spam_and_trash_are_never_in_scope() -> None:
    server, time, kinds = populated_mailbox()

    async with opened(server, time) as connector:
        found = set(ids(await drain(connector)))

    assert {kinds["kept"], kinds["starred"]} <= found
    assert found.isdisjoint({kinds["draft"], kinds["spam"], kinds["trash"]})


@pytest.mark.parametrize("path", ["full", "reconcile", "incremental"])
async def test_an_excluded_label_is_excluded_identically_on_every_path(path: str) -> None:
    """Names with spaces or slashes, a nested label and a category label, by label id."""
    server, time, kinds = populated_mailbox()

    async with opened(server, time, exclude_labels=list(SCOPED_LABELS)) as connector:
        if path == "full":
            found = set(ids(await drain(connector)))
        elif path == "reconcile":
            found = {source_id async for source_id in connector.reconcile()}
        else:
            await drain(connector)
            watermark = connector.watermark
            late = {
                label: server.add_message(labels=("INBOX", label)).id for label in SCOPED_LABELS
            }
            late["kept"] = server.add_message(labels=("INBOX",)).id
            found = set(ids(await drain(connector, watermark)))
            kinds = late

    assert kinds["kept"] in found
    for label in SCOPED_LABELS:
        assert kinds[label] not in found, label


async def test_a_name_that_is_not_a_label_is_refused_at_setup_without_listing_labels() -> None:
    server, time, _ = populated_mailbox()
    connector = connector_over(server, time, exclude_labels=["Typo/Of A Label"])

    with pytest.raises(ConfigError) as caught:
        await connector.setup()

    assert "Typo/Of A Label" in str(caught.value)
    assert "Receipts 2026" not in str(caught.value), "the mailbox's own labels are not echoed"
    assert "/messages" not in paths(server)


async def test_with_no_excluded_labels_no_label_listing_is_made() -> None:
    server, time = mailbox()
    server.add_message()

    async with opened(server, time) as connector:
        await drain(connector)

    assert "/labels" not in paths(server)


# --- incremental discovery (§4.2) ---------------------------------------------------------------


async def test_a_new_message_is_a_candidate_and_is_yielded_when_in_scope() -> None:
    server, time = mailbox()
    server.add_message()

    async with opened(server, time) as connector:
        await drain(connector)
        watermark = connector.watermark
        fresh = server.add_message()
        found = await drain(connector, watermark)

    assert ids(found) == [fresh.id]


async def test_a_message_that_arrived_in_trash_spam_or_as_a_draft_is_not_yielded() -> None:
    server, time = mailbox()
    server.add_message()

    async with opened(server, time) as connector:
        await drain(connector)
        watermark = connector.watermark
        server.add_message(labels=("SPAM",))
        server.add_message(labels=("DRAFT",))
        server.add_message(labels=("TRASH",))
        found = await drain(connector, watermark)

    assert found == []


async def test_a_message_restored_from_trash_is_yielded_again() -> None:
    server, time = mailbox()
    message = server.add_message()

    async with opened(server, time) as connector:
        await drain(connector)
        server.trash(message)
        await drain(connector, connector.watermark)
        watermark = connector.watermark
        server.restore(message)
        found = await drain(connector, watermark)

    assert ids(found) == [message.id]


async def test_a_message_restored_from_trash_that_keeps_an_excluded_label_is_not_yielded() -> None:
    """A ``labelRemoved`` record names what came off, not what is left: the lookup decides."""
    server, time = mailbox()
    message = server.add_message(labels=("INBOX", "Receipts 2026"))

    async with opened(server, time, exclude_labels=["Receipts 2026"]) as connector:
        await drain(connector)
        server.trash(message, record_labels=False)
        await drain(connector, connector.watermark)
        watermark = connector.watermark
        server.restore(message)
        found = await drain(connector, watermark)

    assert found == []


async def test_label_added_and_message_deleted_records_cost_no_lookup() -> None:
    """They can only take a message out of scope, which reconciliation acts on."""
    server, time = mailbox()
    labeled = server.add_message()
    doomed = server.add_message()

    async with opened(server, time) as connector:
        await drain(connector)
        watermark = connector.watermark
        server.add_label(labeled, "STARRED")
        server.delete_message(doomed)
        before = len(server.requests)
        found = await drain(connector, watermark)

    assert found == []
    assert not [
        request for request in server.requests[before:] if request.url.params.get("format")
    ], "no message was read"


async def test_a_candidate_named_by_many_records_is_looked_up_once_per_walk() -> None:
    server, time = mailbox()
    message = server.add_message()
    server.trash(message)  # off the walk's baseline below

    async with opened(server, time) as connector:
        await drain(connector)
        watermark = connector.watermark
        for _ in range(3):
            server.restore(message)
            server.trash(message)
        server.restore(message)
        before = len(server.requests)
        found = await drain(connector, watermark)

    lookups = [
        request
        for request in server.requests[before:]
        if request.url.params.get("format") == "minimal"
    ]
    assert ids(found) == [message.id]
    assert len(lookups) == 1


async def test_a_candidate_that_no_longer_exists_is_skipped() -> None:
    server, time = mailbox()
    server.add_message()

    async with opened(server, time) as connector:
        await drain(connector)
        watermark = connector.watermark
        gone = server.add_message()
        server.delete_message(gone)
        found = await drain(connector, watermark)

    assert found == []


async def test_the_first_history_page_position_is_the_next_watermark() -> None:
    server, time = mailbox()
    server.add_message()

    async with opened(server, time) as connector:
        await drain(connector)
        watermark = connector.watermark
        server.add_message()
        await drain(connector, watermark)
        advanced = connector.watermark

    assert watermark is not None
    assert advanced is not None
    assert int(advanced.value) == server.history_id
    assert int(advanced.value) > int(watermark.value)


async def test_an_expired_history_position_falls_back_to_a_complete_walk() -> None:
    server, time = mailbox()
    first = server.add_message()
    second = server.add_message()

    async with opened(server, time) as connector:
        await drain(connector)
        watermark = connector.watermark
        server.expire_history_through(server.history_id)
        found = await drain(connector, watermark)

    assert sorted(ids(found)) == sorted([first.id, second.id])
    assert connector.watermark is not None


async def test_a_watermark_recorded_under_another_scope_is_not_resumed() -> None:
    """Changing ``exclude_labels`` forces a complete walk, whatever the pipeline hands back."""
    server, time = mailbox()
    receipts = server.add_message(labels=("INBOX", "Receipts 2026"))
    other = server.add_message()

    async with opened(server, time) as connector:
        await drain(connector)
        watermark = connector.watermark

    async with opened(server, time, exclude_labels=["Receipts 2026"]) as narrower:
        found = await drain(narrower, watermark)

    assert ids(found) == [other.id], "a full walk under the new scope, not a resumed delta"
    assert receipts.id not in ids(found)


async def test_a_watermark_recorded_for_another_representation_is_not_resumed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, time = mailbox()
    first = server.add_message()
    second = server.add_message()

    async with opened(server, time) as connector:
        await drain(connector)
        watermark = connector.watermark

    monkeypatch.setattr(gmail_settings, "REPRESENTATION_VERSION", "rfc822/2")
    async with opened(server, time) as bumped:
        found = await drain(bumped, watermark)

    assert sorted(ids(found)) == sorted([first.id, second.id])
    assert {doc.version_token for doc in found} == {"rfc822/2"}


async def test_the_representation_version_and_the_excluded_labels_are_in_the_scope_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, time, _ = populated_mailbox()
    async with (
        opened(server, time) as plain,
        opened(server, time, exclude_labels=["Receipts 2026"]) as narrowed,
        opened(server, time, exclude_labels=["Receipts 2026", "Clients/Acme Corp"]) as narrower,
    ):
        fingerprints = {
            plain.scope_fingerprint,
            narrowed.scope_fingerprint,
            narrower.scope_fingerprint,
        }
        before = plain.scope_fingerprint
        monkeypatch.setattr(gmail_settings, "REPRESENTATION_VERSION", "rfc822/2")

        assert len(fingerprints) == 3
        assert plain.scope_fingerprint != before


async def test_a_watermark_from_another_mailbox_identity_is_not_resumed() -> None:
    server, time = mailbox()
    first = server.add_message()

    async with opened(server, time) as connector:
        await drain(connector)
        watermark = connector.watermark
        assert watermark is not None
        forged = watermark.model_copy(
            update={"metadata": {**watermark.metadata, "pinned_identity": "another-digest"}}
        )
        server.add_message()
        found = await drain(connector, forged)

    assert first.id in ids(found), "a complete walk, not a delta"


async def test_history_that_expires_part_way_through_a_walk_restarts_as_a_complete_walk() -> None:
    server, time = mailbox()
    everything = [server.add_message().id for _ in range(3)]

    async with opened(server, time) as connector:
        await drain(connector)
        watermark = connector.watermark
        everything += [server.add_message().id for _ in range(4)]

        def expires_on_the_second_page(request: httpx.Request) -> None:
            if request.url.path.endswith("/history") and "pageToken" in request.url.params:
                server.history_floor = server.history_id + 1

        server.before_request = expires_on_the_second_page
        found = await drain(connector, watermark)

    assert set(everything) <= set(ids(found)), "the fallback walk lists the whole mailbox"
    assert connector.watermark is not None


# --- reconciliation (§7) ------------------------------------------------------------------------


async def test_reconcile_yields_the_complete_in_scope_inventory() -> None:
    server, time, kinds = populated_mailbox()

    async with opened(server, time, exclude_labels=["Receipts 2026"]) as connector:
        inventory = {source_id async for source_id in connector.reconcile()}
        discovered = set(ids(await drain(connector)))

    assert inventory == discovered
    assert kinds["Receipts 2026"] not in inventory
    assert kinds["trash"] not in inventory


async def test_reconcile_raises_rather_than_ending_on_a_prefix() -> None:
    server, time = mailbox()
    for _ in range(6):
        server.add_message()
    connector = connector_over(server, time, max_retries=1)
    await connector.setup()
    try:
        stream = connector.reconcile_batches()
        first = await anext(stream)
        server.fail(503, times=20)

        with pytest.raises(RemoteError):
            async for _ in stream:
                pass
    finally:
        await connector.teardown()

    assert len(first) == 2


# --- the mailbox pin (§1.4) ---------------------------------------------------------------------


async def test_a_token_for_another_mailbox_is_refused_at_setup_before_any_message_is_read() -> None:
    server, time = mailbox()
    server.add_message()
    server.switch_mailbox("someone.else@example.org")
    connector = connector_over(server, time)

    with pytest.raises(MailboxMismatchError) as caught:
        await connector.setup()

    assert paths(server) == ["/profile"]
    assert "example.org" not in str(caught.value)
    assert ADDRESS not in str(caught.value)


async def test_a_token_that_changes_mailbox_after_setup_is_refused_at_the_start_of_a_run() -> None:
    server, time = mailbox()
    server.add_message()

    async with opened(server, time) as connector:
        server.switch_mailbox("someone.else@example.org")
        before = len(server.requests)

        with pytest.raises(MailboxMismatchError) as caught:
            await drain(connector)

        with pytest.raises(MailboxMismatchError) as reconciling:
            async for _ in connector.reconcile():
                pass

    assert paths(server)[before:] == ["/profile", "/profile"], "no message and no label was read"
    assert "example.org" not in str(caught.value)
    assert "example.org" not in str(reconciling.value)


async def test_an_alias_is_a_mismatch_because_the_profile_reports_the_primary_address() -> None:
    server, time = mailbox(address="primary@example.org")
    connector = connector_over(server, time)

    with pytest.raises(MailboxMismatchError):
        await connector.setup()


async def test_the_pinned_identity_is_a_digest_not_the_address() -> None:
    server, time = mailbox()
    connector = connector_over(server, time)

    assert connector.pinned_identity
    assert "example" not in connector.pinned_identity
    assert "example" not in connector.source_scope


async def test_the_scope_cannot_be_read_before_setup_has_resolved_the_excluded_labels() -> None:
    server, time = mailbox()
    connector = connector_over(server, time, exclude_labels=["Receipts 2026"])

    with pytest.raises(ConnectorError):
        _ = connector.scope_fingerprint


async def test_health_and_metrics_are_aggregate() -> None:
    server, time = mailbox()
    server.add_message()

    async with opened(server, time) as connector:
        await drain(connector)
        health = await connector.health()
        metrics = {metric.name: metric.value for metric in connector.metrics()}

    assert health.ok
    assert set(metrics) == {"gmail_requests", "gmail_throttled", "gmail_messages_fetched"}
    assert metrics["gmail_requests"] > 0


async def test_a_message_that_becomes_a_draft_leaves_at_reconciliation() -> None:
    server, time = mailbox()
    message = server.add_message()
    kept = server.add_message()

    async with opened(server, time) as connector:
        server.add_label(message, "DRAFT")
        inventory = {source_id async for source_id in connector.reconcile()}

    assert inventory == {kept.id}


async def test_a_failure_while_listing_an_excluded_label_raises_and_yields_nothing() -> None:
    """The exclusion listing is part of the inventory: a failed one must not become a bigger one."""
    server, time = mailbox()
    for _ in range(4):
        server.add_message(labels=("INBOX", "Receipts 2026"))
    connector = connector_over(server, time, max_retries=0, exclude_labels=["Receipts 2026"])
    await connector.setup()
    try:

        def fails_when_an_excluded_label_is_listed(request: httpx.Request) -> None:
            if request.url.path.endswith("/messages") and "Label_2" in request.url.params.get_list(
                "labelIds"
            ):
                server.fail(503, times=20)

        server.before_request = fails_when_an_excluded_label_is_listed
        yielded: list[str] = []

        async def reconcile() -> None:
            async for batch in connector.reconcile_batches():
                yielded.extend(batch)

        with pytest.raises(RemoteError):
            await reconcile()
    finally:
        await connector.teardown()

    assert yielded == []


# --- health, metrics and what a fetch records ----------------------------------------------------


async def test_health_fails_for_a_token_that_reads_another_mailbox_without_naming_either() -> None:
    server, time = mailbox()

    async with opened(server, time) as connector:
        server.switch_mailbox("someone.else@example.org")
        health = await connector.health()

    assert not health.ok
    assert health.remedy
    assert "example.org" not in health.detail


async def test_metrics_count_requests_throttles_and_fetches() -> None:
    server, time = mailbox()
    message = server.add_message()

    async with opened(server, time) as connector:
        server.throttle(1, retry_after="1")
        await connector.fetch(DocRef(source_id=message.id, uri="https://unused.example"))
        metrics = {metric.name: metric.value for metric in connector.metrics()}

    assert metrics["gmail_messages_fetched"] == 1
    assert metrics["gmail_throttled"] == 1
    assert metrics["gmail_requests"] == len(server.requests)


async def test_a_subject_that_cannot_be_a_title_leaves_the_message_indexable_by_its_link() -> None:
    """A control character is refused as a title; the message is still fetched and cited."""
    server, time = mailbox()
    raw = (
        b"From: pat@example.org\r\nSubject: bad\x07subject\r\n"
        b"Message-ID: <c1@example.org>\r\n\r\nBody.\r\n"
    )
    message = server.add_message(raw)

    async with opened(server, time) as connector:
        fetched = await connector.fetch(DocRef(source_id=message.id, uri="https://unused.example"))

    assert "bad" not in str(fetched.metadata[PROVENANCE_KEY])
    assert _canonical(fetched).endswith("#search/rfc822msgid:c1%40example.org")
    record = Provenance.from_metadata(fetched.metadata)
    assert record is not None
    assert record.source is not None
    assert record.source.title == "", "no title, but the link survives"


async def test_a_message_with_no_subject_is_fetched_with_an_empty_title() -> None:
    server, time = mailbox()
    message = server.add_message(b"From: pat@example.org\r\n\r\nJust a body.\r\n")

    async with opened(server, time) as connector:
        fetched = await connector.fetch(DocRef(source_id=message.id, uri="https://unused.example"))

    assert fetched.as_bytes().endswith(b"Just a body.\r\n")
    assert _canonical(fetched).endswith(f"#all/{message.id}")


async def test_a_fetch_reconfirms_the_mailbox_once_the_last_check_has_gone_stale() -> None:
    """A run resumed at acquisition never passes through discovery; the fetch is its check."""
    server, time = mailbox()
    first = server.add_message()
    second = server.add_message()
    ref = DocRef(source_id=first.id, uri="https://unused.example")

    async with opened(server, time) as connector:
        await connector.fetch(ref)
        assert paths(server).count("/profile") == 1, "setup's check is still fresh"
        server.switch_mailbox("someone.else@example.org")
        await connector.fetch(DocRef(source_id=second.id, uri="https://unused.example"))
        time.now += 120.0
        raw_before = sum(1 for r in server.requests if r.url.params.get("format") == "raw")

        with pytest.raises(MailboxMismatchError) as caught:
            await connector.fetch(ref)

    assert sum(1 for r in server.requests if r.url.params.get("format") == "raw") == raw_before
    assert "example.org" not in str(caught.value)


async def test_the_reconciliation_scope_follows_the_excluded_labels() -> None:
    server, time = mailbox()

    async with (
        opened(server, time) as wide,
        opened(server, time, exclude_labels=["Receipts 2026"]) as narrow,
    ):
        assert wide.reconciliation_scope != narrow.reconciliation_scope
        assert "example" not in narrow.reconciliation_scope


async def test_a_complete_walk_says_so_and_an_incremental_one_does_not() -> None:
    server, time = mailbox()
    server.add_message()

    async with opened(server, time) as connector:
        assert not connector.walked_full_inventory
        await drain(connector)
        assert connector.walked_full_inventory
        watermark = connector.watermark
        server.add_message()
        await drain(connector, watermark)
        assert not connector.walked_full_inventory
        server.expire_history_through(server.history_id)
        await drain(connector, connector.watermark)
        assert connector.walked_full_inventory, "the fallback walked everything"
