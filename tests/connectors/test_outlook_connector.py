"""The Outlook connector over a synthetic Graph: scope, delta, fetch, calendar, reconciliation.

The real client, retry loop and connector run against
:class:`~tests.connectors.fake_graph.FakeGraph` through an ``httpx`` transport.
``docs/connectors/outlook.md`` §2-§8.
"""

from __future__ import annotations

import json
import logging
import zlib

import pytest

from manicule.connectors.errors import (
    ConnectorError,
    MailboxMismatchError,
    NotFoundError,
    UntrustedLinkError,
)
from manicule.connectors.outlook import EVENT_PREFIX, MESSAGE_PREFIX
from manicule.connectors.outlook_config import (
    EVENT_REPRESENTATION_VERSION,
    MAIL_REPRESENTATION_VERSION,
)
from manicule.core.errors import ConfigError
from manicule.core.mail_ids import MAIL_MESSAGE_ID_KEY, MAIL_THREAD_ID_KEY
from manicule.core.provenance import PROVENANCE_KEY
from manicule.core.sources import DocRef, Watermark
from manicule.testing.contracts import closing
from tests.connectors.fake_graph import ADDRESS, UPN, FakeEvent, mail
from tests.connectors.outlook_support import (
    NOW,
    connector_over,
    drain,
    ids,
    opened,
    outlook_config,
    tenant,
)


def msg(graph_id: str) -> str:
    return MESSAGE_PREFIX + graph_id


# --- scope (§3) ----------------------------------------------------------------------------


async def test_a_full_walk_covers_every_folder_that_is_in_scope() -> None:
    server, time = tenant()
    inbox = server.add_message(mail("In the inbox"))
    archived = server.add_message(mail("Archived"), "2019")
    sent = server.add_message(mail("Sent"), "Sent Items")
    async with opened(server, time, calendar=False) as connector:
        found = await drain(connector)

    assert sorted(ids(found)) == sorted([msg(inbox), msg(archived), msg(sent)])


async def test_the_five_always_excluded_folders_are_dropped_with_their_subfolders() -> None:
    server, time = tenant()
    kept = server.add_message(mail("Kept"))
    for folder in (
        "Junk Email",
        "Deleted Items",
        "Trash Subfolder",
        "Drafts",
        "Outbox",
        "Sync Issues",
        "Conflicts",
    ):
        server.add_message(mail(f"In {folder}"), folder)
    async with opened(server, time, calendar=False) as connector:
        found = await drain(connector)
        reconciled = [source async for source in connector.reconcile()]

    assert ids(found) == [msg(kept)]
    assert reconciled == [msg(kept)]


async def test_exclude_folders_drops_a_folder_and_everything_below_it() -> None:
    server, time = tenant()
    kept = server.add_message(mail("Kept"))
    server.add_message(mail("Old"), "Old Mail")
    server.add_message(mail("Nineteen"), "2019")
    async with opened(server, time, calendar=False, exclude_folders=["Archive"]) as connector:
        found = await drain(connector)

    assert ids(found) == [msg(kept)]


async def test_a_nested_folder_is_excluded_by_its_path_alone() -> None:
    server, time = tenant()
    server.add_message(mail("Nineteen"), "2019")
    old = server.add_message(mail("Old"), "Old Mail")
    async with opened(server, time, calendar=False, exclude_folders=["archive/2019"]) as connector:
        found = await drain(connector)

    assert ids(found) == [msg(old)]


async def test_a_path_the_mailbox_does_not_have_is_refused_without_listing_its_folders() -> None:
    server, time = tenant()
    server.add_message(mail("A"))
    connector = connector_over(server, time, exclude_folders=["Archive/Nonesuch"])
    with pytest.raises(ConfigError, match="not a folder") as refused:
        await connector.setup()

    assert "Nonesuch" in str(refused.value), "the operator's own text may be quoted"
    assert "Inbox" not in str(refused.value), "the mailbox's folders never are"


async def test_every_request_asks_for_immutable_ids() -> None:
    server, time = tenant()
    server.add_message(mail("A"))
    server.add_event(FakeEvent("e1", "Standup", "2026-03-02T09:00:00", "2026-03-02T09:30:00"))
    async with opened(server, time) as connector:
        found = await drain(connector)
        for document in found:
            await connector.fetch(document.ref)
        _ = [source async for source in connector.reconcile()]

    graph = [r for r in server.requests if r.url.host == "graph.microsoft.com"]
    assert graph
    assert all('IdType="ImmutableId"' in r.headers["prefer"] for r in graph)


# --- discovery and the watermark (§4) --------------------------------------------------------


async def test_a_second_sync_costs_what_changed() -> None:
    server, time = tenant()
    first = server.add_message(mail("First"))
    second = server.add_message(mail("Second"))
    async with opened(server, time, calendar=False) as connector:
        await drain(connector)
        watermark = connector.watermark
        assert watermark is not None
        third = server.add_message(mail("Third"))
        server.touch(first)
        found = await drain(connector, watermark)

    assert sorted(ids(found)) == sorted([msg(first), msg(third)])
    assert msg(second) not in ids(found)


async def test_every_message_carries_one_constant_version_token() -> None:
    server, time = tenant()
    first = server.add_message(mail("First"))
    async with opened(server, time, calendar=False) as connector:
        before = await drain(connector)
        watermark = connector.watermark
        server.touch(first)
        after = await drain(connector, watermark)

    assert {d.version_token for d in before + after} == {MAIL_REPRESENTATION_VERSION}


async def test_the_watermark_holds_one_link_per_folder_and_records_its_scope() -> None:
    server, time = tenant()
    server.add_message(mail("A"))
    server.add_message(mail("B"), "2019")
    async with opened(server, time, calendar=False) as connector:
        assert connector.watermark is None, "nothing has been walked yet"
        await drain(connector)
        watermark = connector.watermark

    assert watermark is not None
    links = json.loads(watermark.value)["mail"]
    assert set(links) == {"inbox", "sent", "archive", "archive-2019", "archive-old"}
    assert all(link.startswith("https://graph.microsoft.com/v1.0/me/") for link in links.values())
    assert watermark.metadata["pinned_identity"] == outlook_config().pinned_identity


async def test_an_abandoned_walk_does_not_move_the_watermark() -> None:
    server, time = tenant()
    for number in range(5):
        server.add_message(mail(f"Message {number}"))
    async with opened(server, time, calendar=False) as connector:
        await drain(connector)
        before = connector.watermark
        async with closing(connector.discover(None)) as stream:
            await anext(stream)

    assert connector.watermark is None or connector.watermark == before
    assert connector.walked_full_inventory is False


async def test_a_message_moved_between_folders_keeps_its_id_and_is_not_fetched_again() -> None:
    server, time = tenant()
    moved = server.add_message(mail("Filed away"))
    async with opened(server, time, calendar=False) as connector:
        await drain(connector)
        watermark = connector.watermark
        server.move(moved, "Old Mail")
        found = await drain(connector, watermark)

    assert ids(found) == [msg(moved)], "named again by the new folder, at the id it always had"
    assert found[0].version_token == MAIL_REPRESENTATION_VERSION, "so the version check skips it"


async def test_an_expired_delta_walks_that_folder_again_and_the_others_resume(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server, time = tenant()
    server.add_message(mail("Inbox one"))
    archived = server.add_message(mail("Archived"), "Archive")
    async with opened(server, time, calendar=False) as connector:
        await drain(connector)
        watermark = connector.watermark
        assert watermark is not None
        # Only Archive's state is lost: Inbox's link is replaced by a fresh one first.
        server.expire_deltas()
        links = json.loads(watermark.value)["mail"]
        links["inbox"] = links["inbox"].replace("D-", "D-9999")
        patched = watermark.model_copy(
            update={"value": json.dumps({"mail": links}, sort_keys=True, separators=(",", ":"))}
        )
        with caplog.at_level(logging.WARNING, logger="manicule.connectors.outlook"):
            found = await drain(connector, patched)

    assert msg(archived) in ids(found), "the folder whose state was gone is walked from scratch"
    assert "no longer holds a folder's delta state" in caplog.text
    assert connector.walked_full_inventory is False, (
        "some folders resumed, so it is not a full walk"
    )


@pytest.mark.parametrize(
    ("status", "code"),
    [(410, "syncStateNotFound"), (404, "ErrorItemNotFound"), (400, "syncStateNotFound")],
)
async def test_every_way_graph_says_the_state_is_gone_means_a_complete_walk(
    status: int, code: str
) -> None:
    server, time = tenant()
    first = server.add_message(mail("First"))
    async with opened(server, time, calendar=False) as connector:
        await drain(connector)
        watermark = connector.watermark
        server.expire_deltas(status, code)
        found = await drain(connector, watermark)

    assert msg(first) in ids(found)


async def test_a_new_folder_is_walked_from_scratch_and_a_vanished_one_leaves_the_watermark() -> (
    None
):
    server, time = tenant()
    server.add_message(mail("Inbox"))
    async with opened(server, time, calendar=False) as connector:
        await drain(connector)
        watermark = connector.watermark
        assert watermark is not None
        assert "archive-old" in json.loads(watermark.value)["mail"]
        server.delete_folder("Old Mail")
        found = await drain(connector, watermark)
        after = connector.watermark

    assert found == []
    assert after is not None
    assert "archive-old" not in json.loads(after.value)["mail"]


async def test_a_watermark_recorded_under_another_scope_is_not_resumed() -> None:
    server, time = tenant()
    first = server.add_message(mail("First"))
    async with opened(server, time, calendar=False) as connector:
        await drain(connector)
        watermark = connector.watermark
    async with opened(server, time, calendar=False, exclude_folders=["Archive"]) as narrowed:
        found = await drain(narrowed, watermark)

    assert ids(found) == [msg(first)], "a changed scope discards the positions: a complete walk"


async def test_a_watermark_recorded_for_another_mailbox_is_not_resumed() -> None:
    server, time = tenant()
    first = server.add_message(mail("First"))
    async with opened(server, time, calendar=False) as connector:
        await drain(connector)
        watermark = connector.watermark
        assert watermark is not None
    stranger = watermark.model_copy(
        update={"metadata": {**watermark.metadata, "pinned_identity": "someone-else"}}
    )
    async with opened(server, time, calendar=False) as connector:
        found = await drain(connector, stranger)

    assert ids(found) == [msg(first)]


async def test_a_stored_link_that_is_not_graphs_is_never_followed() -> None:
    server, time = tenant()
    server.add_message(mail("First"))
    async with opened(server, time, calendar=False) as connector:
        await drain(connector)
        watermark = connector.watermark
        assert watermark is not None
        links = json.loads(watermark.value)["mail"]
        links["inbox"] = "https://evil.example.net/v1.0/me/mailFolders('inbox')/messages/delta"
        poisoned = watermark.model_copy(
            update={"value": json.dumps({"mail": links}), "metadata": watermark.metadata}
        )
        before = len(server.requests)
        with pytest.raises(UntrustedLinkError) as refused:
            await drain(connector, poisoned)

    assert all(r.url.host != "evil.example.net" for r in server.requests[before:])
    assert "evil.example.net" not in str(refused.value), "the link is never echoed"


async def test_pagination_that_loops_is_refused() -> None:
    server, time = tenant()
    for number in range(6):
        server.add_message(mail(f"Message {number}"))
    async with opened(server, time, calendar=False) as connector:
        server.loop_pagination = True
        with pytest.raises(ConnectorError, match="already followed"):
            await drain(connector)


async def test_a_garbled_watermark_is_no_watermark() -> None:
    server, time = tenant()
    first = server.add_message(mail("First"))
    async with opened(server, time, calendar=False) as connector:
        found = await drain(
            connector,
            Watermark(
                value="not json",
                observed_at=NOW,
                metadata={
                    "pinned_identity": connector.pinned_identity,
                    "scope_fingerprint": connector.scope_fingerprint,
                },
            ),
        )

    assert ids(found) == [msg(first)]


# --- identity (§1.4) -----------------------------------------------------------------------


@pytest.mark.parametrize("who", [ADDRESS, UPN, ADDRESS.upper()])
async def test_the_account_may_be_the_primary_address_or_the_sign_in_name(who: str) -> None:
    server, time = tenant()
    async with opened(server, time, account=who, calendar=False) as connector:
        assert (await connector.health()).ok


async def test_a_token_for_another_mailbox_is_refused_before_anything_is_read() -> None:
    server, time = tenant()
    server.upn = "someone.else@contoso.example.org"
    server.mail_address = "someone.else@example.org"
    connector = connector_over(server, time, calendar=False)
    with pytest.raises(MailboxMismatchError) as refused:
        await connector.setup()

    text = str(refused.value)
    assert ADDRESS not in text
    assert "someone.else" not in text, "neither address is in the message"
    assert [r for r in server.requests if "mailFolders" in str(r.url)] == []


async def test_the_mailbox_is_rechecked_before_a_fetch_once_the_confirmation_is_stale() -> None:
    server, time = tenant()
    graph_id = server.add_message(mail("A"))
    async with opened(server, time, calendar=False) as connector:
        found = await drain(connector)
        server.upn = server.mail_address = "someone.else@example.org"
        await connector.fetch(found[0].ref)  # still fresh: no new check
        time.now += 120
        with pytest.raises(MailboxMismatchError):
            await connector.fetch(found[0].ref)

    assert graph_id


# --- fetch (§5) ----------------------------------------------------------------------------


async def test_a_message_is_fetched_as_its_whole_mime_with_the_facts_only_graph_has() -> None:
    server, time = tenant()
    raw = mail("Quarterly plan", "Retry budgets are netted down twice.")
    graph_id = server.add_message(raw, conversation="conv-77")
    async with opened(server, time, calendar=False) as connector:
        found = await drain(connector)
        fetched = await connector.fetch(found[0].ref)

    assert fetched.source_id == msg(graph_id)
    assert fetched.media_type == "message/rfc822"
    assert fetched.as_bytes() == raw
    assert fetched.metadata[MAIL_THREAD_ID_KEY] == "conv-77"
    assert (
        fetched.metadata[MAIL_MESSAGE_ID_KEY] == f"{zlib.crc32(b'Quarterly plan'):08x}@example.org"
    )
    provenance = fetched.metadata[PROVENANCE_KEY]
    assert isinstance(provenance, dict)
    assert provenance["source"]["title"] == "Quarterly plan"  # type: ignore[index]
    assert "outlook.office365.com" in provenance["source"]["canonical_uri"]  # type: ignore[index]
    assert fetched.metadata["version_token"] == MAIL_REPRESENTATION_VERSION


async def test_a_fetch_needs_one_request_when_discovery_carried_the_facts() -> None:
    server, time = tenant()
    server.add_message(mail("A"))
    async with opened(server, time, calendar=False) as connector:
        found = await drain(connector)
        server.requests.clear()
        time.now += 120  # the mailbox confirmation is stale, so the one /me check is expected too
        await connector.fetch(found[0].ref)

    paths = [r.url.path for r in server.requests]
    assert [p for p in paths if p.endswith("/$value")], "the message itself"
    assert not [p for p in paths if "/messages/" in p and not p.endswith("/$value")]


async def test_a_fetch_without_discovery_metadata_asks_graph_for_it_once() -> None:
    server, time = tenant()
    graph_id = server.add_message(mail("A"), conversation="conv-9")
    async with opened(server, time, calendar=False) as connector:
        fetched = await connector.fetch(DocRef(source_id=msg(graph_id), uri="https://x.example/"))

    assert fetched.metadata[MAIL_THREAD_ID_KEY] == "conv-9"


async def test_a_message_without_a_message_id_header_is_still_indexed() -> None:
    server, time = tenant()
    server.add_message(mail("No id", with_id=False))
    async with opened(server, time, calendar=False) as connector:
        found = await drain(connector)
        fetched = await connector.fetch(found[0].ref)

    assert MAIL_MESSAGE_ID_KEY not in fetched.metadata


async def test_a_message_deleted_between_discovery_and_fetch_is_not_found() -> None:
    server, time = tenant()
    graph_id = server.add_message(mail("A"))
    async with opened(server, time, calendar=False) as connector:
        found = await drain(connector)
        server.vanish_on_fetch(graph_id)
        with pytest.raises(NotFoundError) as refused:
            await connector.fetch(found[0].ref)

    assert graph_id not in str(refused.value), "no id in an error"


async def test_a_source_id_the_connector_did_not_issue_is_not_found() -> None:
    server, time = tenant()
    async with opened(server, time, calendar=False) as connector:
        with pytest.raises(NotFoundError):
            await connector.fetch(DocRef(source_id="gmail:1", uri="https://x.example/"))


# --- reconciliation (§6) -------------------------------------------------------------------


async def test_reconcile_lists_exactly_what_a_full_walk_would_and_drops_what_left() -> None:
    server, time = tenant()
    stays = server.add_message(mail("Stays"))
    leaves = server.add_message(mail("Leaves"))
    async with opened(server, time, calendar=False) as connector:
        walked = ids(await drain(connector))
        server.delete(leaves)
        reconciled = [source async for source in connector.reconcile()]

    assert sorted(walked) == sorted([msg(stays), msg(leaves)])
    assert reconciled == [msg(stays)]


async def test_a_failure_part_way_through_reconciliation_raises_rather_than_ending_early() -> None:
    server, time = tenant()
    for number in range(6):
        server.add_message(mail(f"Message {number}"))
    async with opened(server, time, calendar=False) as connector:
        stream = connector.reconcile()
        await anext(stream)
        server.throttle.extend([(503, None)] * 20)
        with pytest.raises(ConnectorError):
            async for _ in stream:
                pass


async def test_reconcile_starts_with_the_mailbox_check() -> None:
    server, time = tenant()
    server.add_message(mail("A"))
    async with opened(server, time, calendar=False) as connector:
        server.upn = server.mail_address = "someone.else@example.org"
        with pytest.raises(MailboxMismatchError):
            _ = [source async for source in connector.reconcile()]


# --- calendar (§7) -------------------------------------------------------------------------


def standup(event_id: str = "e1") -> FakeEvent:
    return FakeEvent(
        event_id,
        "Quarterly planning",
        "2026-03-02T14:00:00",
        "2026-03-02T15:00:00",
        location="Room 4",
        body="Agenda for the quarter.",
        attendees=("Lee Example <lee@example.org>",),
    )


async def test_an_event_is_discovered_at_its_change_key_and_rendered_as_icalendar() -> None:
    server, time = tenant()
    server.add_event(standup())
    async with opened(server, time, mail=False) as connector:
        found = await drain(connector)
        fetched = await connector.fetch(found[0].ref)

    assert ids(found) == [EVENT_PREFIX + "e1"]
    assert found[0].version_token == f"ck-1|{EVENT_REPRESENTATION_VERSION}"
    assert found[0].media_type == "text/calendar"
    text = fetched.as_bytes().decode()
    assert "BEGIN:VCALENDAR" in text
    assert "SUMMARY:Quarterly planning" in text
    assert "DTSTART:20260302T140000Z" in text
    assert "LOCATION:Room 4" in text
    assert 'ATTENDEE;CN="Lee Example":mailto:lee@example.org' in text
    assert fetched.media_type == "text/calendar"


async def test_an_edited_event_is_a_new_version() -> None:
    server, time = tenant()
    server.add_event(standup())
    async with opened(server, time, mail=False) as connector:
        before = await drain(connector)
        server.edit_event("e1", subject="Moved to Thursday")
        after = await drain(connector)

    assert before[0].version_token != after[0].version_token


async def test_the_calendar_window_decides_what_is_listed() -> None:
    server, time = tenant()
    server.add_event(standup("inside"))
    server.add_event(FakeEvent("old", "Long ago", "2025-01-02T09:00:00", "2025-01-02T10:00:00"))
    server.add_event(FakeEvent("far", "Far ahead", "2027-12-02T09:00:00", "2027-12-02T10:00:00"))
    async with opened(server, time, mail=False) as connector:
        found = await drain(connector)

    assert ids(found) == [EVENT_PREFIX + "inside"]
    assert NOW.year == 2026


async def test_an_all_day_event_keeps_no_time() -> None:
    server, time = tenant()
    server.add_event(
        FakeEvent("day", "Offsite", "2026-03-03T00:00:00", "2026-03-04T00:00:00", all_day=True)
    )
    async with opened(server, time, mail=False) as connector:
        found = await drain(connector)
        fetched = await connector.fetch(found[0].ref)

    assert "DTSTART;VALUE=DATE:20260303" in fetched.as_bytes().decode()


async def test_a_calendar_only_instance_never_touches_mail_and_lists_everything_each_time() -> None:
    server, time = tenant()
    server.add_message(mail("Mail"))
    server.add_event(standup())
    async with opened(server, time, mail=False) as connector:
        assert connector.enumerates_full_inventory is True
        await drain(connector)
        assert connector.walked_full_inventory is True

    assert not [r for r in server.requests if "mailFolders" in str(r.url)]


async def test_a_mail_only_instance_never_touches_the_calendar() -> None:
    server, time = tenant()
    server.add_message(mail("Mail"))
    server.add_event(standup())
    async with opened(server, time, calendar=False) as connector:
        assert connector.enumerates_full_inventory is False
        await drain(connector)

    assert not [r for r in server.requests if "calendar" in str(r.url).casefold()]


async def test_reconciliation_includes_events_in_the_window_and_drops_those_that_left_it() -> None:
    server, time = tenant()
    server.add_message(mail("Mail"))
    server.add_event(standup("inside"))
    async with opened(server, time) as connector:
        first = {source async for source in connector.reconcile()}
        server.events["inside"].start = "2024-01-01T09:00:00"
        server.events["inside"].end = "2024-01-01T10:00:00"
        second = {source async for source in connector.reconcile()}

    assert EVENT_PREFIX + "inside" in first
    assert EVENT_PREFIX + "inside" not in second


async def test_a_message_and_an_event_can_never_share_a_source_id() -> None:
    server, time = tenant()
    server.add_message(mail("Mail"))
    server.add_event(standup())
    async with opened(server, time) as connector:
        found = ids(await drain(connector))

    assert len(set(found)) == len(found)
    assert {i.split(":")[0] for i in found} == {"msg", "evt"}


# --- observability --------------------------------------------------------------------------


async def test_metrics_count_requests_and_fetches_and_never_name_anything() -> None:
    server, time = tenant()
    server.add_message(mail("A"))
    async with opened(server, time, calendar=False) as connector:
        found = await drain(connector)
        await connector.fetch(found[0].ref)
        metrics = {m.name: m.value for m in connector.metrics()}

    assert metrics["outlook_items_fetched"] == 1.0
    assert metrics["outlook_requests"] > 0


async def test_the_scope_cannot_be_read_before_setup_when_folders_are_excluded() -> None:
    server, time = tenant()
    connector = connector_over(server, time, exclude_folders=["Archive"])
    with pytest.raises(ConnectorError, match="setup"):
        _ = connector.scope_fingerprint


def test_the_representation_versions_are_what_the_design_says() -> None:
    assert MAIL_REPRESENTATION_VERSION == "mime/1"
    assert EVENT_REPRESENTATION_VERSION == "ics/1"
