"""Regression tests for the ten findings of the first code review of the Outlook pull request.

Each section names the finding it pins. ``docs/connectors/outlook.md`` says what each decision is.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from manicule.config.settings import Settings
from manicule.connectors.errors import (
    ConnectorError,
    MailboxMismatchError,
    UntrustedLinkError,
)
from manicule.connectors.outlook import OutlookConnector
from manicule.connectors.outlook_auth import OutlookCredential
from manicule.connectors.outlook_client import (
    LIST_PAGE_SIZE,
    OutlookClient,
    _trusted,  # pyright: ignore[reportPrivateUsage]
)
from manicule.connectors.outlook_config import OutlookConfig, resolve_outlook_credentials
from manicule.connectors.outlook_ics import render_event
from manicule.connectors.plugin import build_outlook
from manicule.plugins.registry import BuildContext
from tests.connectors.fake_gmail import SimulatedTime
from tests.connectors.fake_graph import (
    ADDRESS,
    CLIENT_ID,
    OBJECT_ID,
    REFRESH_TOKEN,
    TENANT,
    FakeGraph,
    FakeGraphSettings,
    mail,
)
from tests.connectors.outlook_support import (
    connector_over,
    drain,
    ids,
    opened,
    outlook_config,
    tenant,
)

G = "https://graph.microsoft.com/v1.0"
OTHER = "99999999-0000-0000-0000-000000000000"
OBJECT_ELSEWHERE = "22222222-3333-4444-5555-666666666666"


# --- 1. the links a response may hand back ----------------------------------------------------


@pytest.mark.parametrize(
    "link",
    [
        f"{G}/me/mailFolders('inbox')/messages/delta?$deltatoken=D-1-0",
        f"{G}/ME/mailFolders/inbox/messages/delta?$skiptoken=S-1-2",
        f"{G}/users('{OBJECT_ID}')/mailFolders('inbox')/messages/delta",
        f"{G}/users/{OBJECT_ID}/mailFolders/inbox/messages/delta",
        f"{G}/USERS('{OBJECT_ID.upper()}')/mailFolders/inbox",
    ],
)
def test_the_signed_in_users_own_paths_are_trusted_in_every_spelling_graph_uses(link: str) -> None:
    assert _trusted(link, "op", OBJECT_ID) == link


@pytest.mark.parametrize(
    "link",
    [
        f"{G}/users('{OTHER}')/mailFolders",
        f"{G}/users/{OTHER}/mailFolders",
        f"{G}/users/{ADDRESS}/mailFolders",
        "http://graph.microsoft.com/v1.0/me/mailFolders",
        "https://graph.microsoft.com.evil.example/v1.0/me/mailFolders",
        "https://graph.microsoft.com:8443/v1.0/me/mailFolders",
        "https://user:pw@graph.microsoft.com/v1.0/me/mailFolders",
        "https://graph.microsoft.com/beta/me/mailFolders",
        "https://graph.microsoft.com/v1.0/groups/abc/messages",
        f"{G}/users('{OBJECT_ID}x')/mailFolders",
        f"{G}/users('{OBJECT_ID}')extra/mailFolders",
    ],
)
def test_anything_else_is_refused_including_another_users_mailbox(link: str) -> None:
    with pytest.raises(UntrustedLinkError) as refused:
        _trusted(link, "op", OBJECT_ID)

    assert "evil" not in str(refused.value)
    assert "99999999" not in str(refused.value), "the link is never echoed"


def test_a_user_path_is_refused_until_the_users_own_id_is_known() -> None:
    with pytest.raises(UntrustedLinkError):
        _trusted(f"{G}/users('{OBJECT_ID}')/mailFolders", "op", None)


@pytest.mark.parametrize("shape", ["users-paren", "users-slash"])
async def test_a_walk_follows_links_that_name_the_user_by_object_id(shape: str) -> None:
    server, time = tenant()
    for number in range(5):
        server.add_message(mail(f"Message {number}"))
    async with opened(server, time, calendar=False) as connector:
        server.link_shape = shape
        first = await drain(connector)
        watermark = connector.watermark
        assert watermark is not None
        assert OBJECT_ID in watermark.value, "the stored links are the user-named ones"
        server.add_message(mail("New"))
        second = await drain(connector, watermark)

    assert len(first) == 5
    assert len(second) == 1


async def test_a_link_into_another_users_mailbox_is_never_followed() -> None:
    server, time = tenant()
    for number in range(5):
        server.add_message(mail(f"Message {number}"))
    async with opened(server, time, calendar=False) as connector:
        server.link_shape = "other"
        before = len(server.requests)
        with pytest.raises(UntrustedLinkError):
            await drain(connector)

    followed = [r for r in server.requests[before:] if "99999999" in str(r.url)]
    assert followed == [], "no request, and so no bearer token, went to the other user's path"


# --- 2. a rotation that is not stored ---------------------------------------------------------


def credential(server: FakeGraph, persist: object = None) -> OutlookCredential:
    return OutlookCredential(
        connector="work",
        tenant_id=TENANT,
        client_id=CLIENT_ID,
        refresh_token=SecretStr(REFRESH_TOKEN),
        scopes=("offline_access", "User.Read", "Mail.Read"),
        persist=persist,  # pyright: ignore[reportArgumentType]
        transport=server.transport(),
    )


async def test_a_rotation_with_nowhere_to_go_is_counted_and_warned_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = FakeGraph()
    server.rotate = True
    cred = credential(server)
    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            await cred.access_token(refresh=True)

    assert cred.rotations == 5
    assert cred.unstored_rotations == 5
    assert caplog.text.count("rotated its refresh token") == 1, "once per process, not hourly"
    assert "no refresh_token_file is configured" in caplog.text
    assert cred.refresh_token.get_secret_value() not in caplog.text


async def test_a_rotation_that_cannot_be_written_is_counted_warned_once_and_names_no_path(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = FakeGraph()
    server.rotate = True

    def read_only(token: SecretStr) -> None:
        raise OSError(30, "Read-only file system", f"/run/secrets/{token.get_secret_value()}")

    cred = credential(server, persist=read_only)
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            await cred.access_token(refresh=True)

    assert cred.unstored_rotations == 3
    assert caplog.text.count("rotated its refresh token") == 1
    assert "OSError" in caplog.text
    assert "/run/secrets" not in caplog.text
    assert "Read-only" not in caplog.text


async def test_a_rotation_that_is_stored_is_not_counted() -> None:
    server = FakeGraph()
    server.rotate = True
    kept: list[SecretStr] = []
    cred = credential(server, persist=kept.append)
    await cred.access_token()

    assert cred.rotations == 1
    assert cred.unstored_rotations == 0


async def test_the_connector_reports_the_count_as_a_metric() -> None:
    server, time = tenant()
    count = [3]
    client = OutlookClient(
        FakeGraphSettings(),
        credential(server),
        transport=server.transport(),
        sleep=time.sleep,
    )
    reporting = OutlookConnector(outlook_config(), client, unstored_rotations=lambda: count[0])

    metrics = {m.name: m.value for m in reporting.metrics()}

    assert metrics["outlook_token_rotations_unstored"] == 3.0


# --- 3. concurrent 401s are one exchange ------------------------------------------------------


async def test_many_requests_answered_401_on_the_same_token_share_one_exchange() -> None:
    server = FakeGraph()
    client = OutlookClient(
        FakeGraphSettings(max_concurrent_requests=4),
        credential(server),
        transport=server.transport(),
        sleep=SimulatedTime().sleep,
    )
    await client.setup()
    try:
        await client.get_me()
        assert server.refreshes == 1
        server.revoke_access_tokens()
        results = await asyncio.gather(*(client.get_me() for _ in range(12)))
    finally:
        await client.teardown()

    assert len(results) == 12
    assert server.refreshes == 2, "twelve 401s, one exchange"


async def test_a_forced_refresh_that_names_no_rejected_token_always_mints() -> None:
    server = FakeGraph()
    cred = credential(server)
    first = await cred.access_token()
    second = await cred.access_token(refresh=True)
    third = await cred.access_token(refresh=True)

    assert len({first, second, third}) == 3
    assert server.refreshes == 3


async def test_a_forced_refresh_returns_a_newer_token_another_request_already_minted() -> None:
    server = FakeGraph()
    cred = credential(server)
    stale = await cred.access_token()
    newer = await cred.access_token(refresh=True)

    again = await cred.access_token(refresh=True, rejected=stale)

    assert again == newer
    assert server.refreshes == 2, "no third exchange for a token that was already replaced"


# --- 4. a message named twice in one walk -----------------------------------------------------


async def test_a_message_moved_mid_walk_is_named_twice_and_the_journal_absorbs_it() -> None:
    """The connector does not hold the inventory to prevent it (§4.3); the pipeline test below
    shows the journal making it one document."""
    server, time = tenant()
    moved = server.add_message(mail("Moves"), "Inbox")

    def move_before_old_mail_is_walked(request: httpx.Request) -> None:
        if request.url.path.endswith("/messages/delta") and "/archive-old/" in request.url.path:
            server.before_request = None
            server.move(moved, "Old Mail")

    async with opened(server, time, calendar=False) as connector:
        server.before_request = move_before_old_mail_is_walked
        found = await drain(connector)

    named = [d for d in found if d.ref.source_id == "msg:" + moved]
    assert len(named) == 2
    assert {d.version_token for d in named} == {"mime/1"}


# --- 5. the folder tree is not read twice in a row --------------------------------------------


def tree_reads(server: FakeGraph) -> int:
    return sum(
        1
        for r in server.requests
        if r.url.path.endswith("/me/mailFolders") and "$skiptoken" not in str(r.url)
    )


async def test_the_tree_read_at_setup_is_reused_by_the_first_run_and_not_the_second() -> None:
    server, time = tenant()
    server.add_message(mail("A"))
    async with opened(server, time, calendar=False) as connector:
        after_setup = tree_reads(server)
        await drain(connector)
        after_first = tree_reads(server)
        await drain(connector)
        after_second = tree_reads(server)

    assert after_setup == 1
    assert after_first == 1, "the first run reused setup's read"
    assert after_second == 2, "the second run read it afresh"


async def test_a_run_after_the_tree_has_gone_stale_reads_it_again() -> None:
    server, time = tenant()
    server.add_message(mail("A"))
    async with opened(server, time, calendar=False) as connector:
        time.now += 120
        await drain(connector)

    assert tree_reads(server) == 2


async def test_the_five_well_known_lookups_run_concurrently_inside_the_cap() -> None:
    server, time = tenant()
    connector = connector_over(server, time, calendar=False)
    await connector.setup()
    await connector.teardown()

    names = {"junkemail", "deleteditems", "drafts", "outbox", "syncissues"}
    lookups = [r for r in server.requests if r.url.path.split("/")[-1] in names]
    assert len(lookups) == 5
    assert 2 <= server.peak_in_flight <= 4


# --- 6. page sizes ----------------------------------------------------------------------------


async def test_reconciliation_lists_ids_at_the_largest_page_and_delta_keeps_its_own() -> None:
    server, time = tenant()
    server.add_message(mail("A"))
    async with opened(server, time, calendar=False) as connector:
        await drain(connector)
        _ = [source async for source in connector.reconcile()]

    listings = [r for r in server.requests if r.url.path.endswith("/inbox/messages")]
    deltas = [r for r in server.requests if r.url.path.endswith("/messages/delta")]
    assert listings
    assert all(f"$top={LIST_PAGE_SIZE}" in str(r.url) for r in listings)
    assert LIST_PAGE_SIZE == 1_000
    assert "$select=id" in str(listings[0].url)
    assert deltas
    assert all("odata.maxpagesize=100" in r.headers["prefer"] for r in deltas)


# --- 7. a bounded download --------------------------------------------------------------------


async def capped(server: FakeGraph, time: SimulatedTime, cap: int) -> OutlookConnector:
    connector = connector_over(server, time, calendar=False)
    connector._max_message_bytes = cap  # pyright: ignore[reportPrivateUsage]
    await connector.setup()
    return connector


async def test_a_message_over_the_cap_is_refused_by_its_declared_length() -> None:
    server, time = tenant()
    graph_id = server.add_message(mail("Big", "x" * 5_000))
    connector = await capped(server, time, 1_000)
    try:
        found = await drain(connector)
        with pytest.raises(ConnectorError, match="fetch cap") as refused:
            await connector.fetch(found[0].ref)
    finally:
        await connector.teardown()

    assert graph_id not in str(refused.value)
    assert "1000-byte" in str(refused.value)


async def test_a_streamed_message_is_abandoned_as_soon_as_it_passes_the_cap() -> None:
    """No ``Content-Length`` to go by: the count stops the read, and the rest is never pulled."""
    server, time = tenant()
    server.add_message(mail("Big", "x" * 50_000))
    server.value_chunk = 1_000
    connector = await capped(server, time, 4_000)
    try:
        found = await drain(connector)
        with pytest.raises(ConnectorError, match="fetch cap"):
            await connector.fetch(found[0].ref)
    finally:
        await connector.teardown()

    assert 0 < server.streamed < 50_000, "stopped part way, not read whole"


async def test_a_message_under_the_cap_is_whole_whether_or_not_it_is_streamed() -> None:
    server, time = tenant()
    raw = mail("Small", "y" * 3_000)
    server.add_message(raw)
    server.value_chunk = 500
    connector = await capped(server, time, 100_000)
    try:
        found = await drain(connector)
        fetched = await connector.fetch(found[0].ref)
    finally:
        await connector.teardown()

    assert fetched.as_bytes() == raw


def test_the_factory_passes_the_pipelines_fetch_cap_to_the_connector() -> None:
    settings = Settings()
    context = BuildContext(
        settings=settings,
        config=OutlookConfig.model_validate(
            {
                "account": ADDRESS,
                "tenant_id": TENANT,
                "client_id": CLIENT_ID,
                "refresh_token": "x",
            }
        ),
        data_dir=Path(),
        cache_dir=Path(),
        components=None,  # pyright: ignore[reportArgumentType]
        instance="work",
    )

    built = build_outlook(context)

    assert isinstance(built, OutlookConnector)
    assert built._max_message_bytes == settings.ingest.max_fetch_bytes  # pyright: ignore[reportPrivateUsage]


# --- 8. control characters in the rendered calendar -------------------------------------------


def event(**changes: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "e1",
        "iCalUId": "uid-1",
        "subject": "Planning",
        "start": {"dateTime": "2026-03-02T14:00:00.0000000"},
        "end": {"dateTime": "2026-03-02T15:00:00.0000000"},
    }
    return {**base, **changes}


def lines_of(raw: bytes) -> list[str]:
    """Content lines as any RFC 5545 reader sees them: unfolded, then split at CRLF."""
    return raw.decode().replace("\r\n ", "").split("\r\n")


def test_an_attendee_address_cannot_forge_another_property() -> None:
    forged = "lee@example.org\r\nATTENDEE;CN=Eve:mailto:eve@evil.example\r\nSTATUS:CANCELLED"
    raw = render_event(event(attendees=[{"emailAddress": {"name": "Lee", "address": forged}}]))

    properties = [line.split(":")[0].split(";")[0] for line in lines_of(raw) if line]
    assert properties.count("ATTENDEE") == 1
    assert properties.count("STATUS") == 1
    assert "eve@evil.example" not in raw.decode()
    assert 'ATTENDEE;CN="Lee":' in raw.decode(), "a non-address is dropped and the name kept"


@pytest.mark.parametrize(
    "bad", ["a\r\nb", "a\nb", "a\rb", "a\x00b", "a\x1bb", "a\u2028b", "a\u2029b"]
)
def test_no_control_or_line_separator_survives_in_a_single_line_value(bad: str) -> None:
    raw = render_event(
        event(
            iCalUId=f"uid{bad}x",
            subject=f"s{bad}s",
            location={"displayName": f"l{bad}l"},
            categories=[f"c{bad}c"],
            organizer={"emailAddress": {"name": f"n{bad}n", "address": "o@example.org"}},
        )
    )

    text = raw.decode()
    for forbidden in ("\x00", "\x1b", chr(0x2028), chr(0x2029)):
        assert forbidden not in text
    for line in lines_of(raw):
        assert line == "" or ":" in line, f"a stray line would be a forged property: {line!r}"
    assert "SUMMARY:sa bs" in text
    assert "LOCATION:la bl" in text


def test_a_description_keeps_its_line_breaks_and_loses_other_controls() -> None:
    raw = render_event(event(body={"content": "one\r\ntwo\x00\x07\nthree\tfour"}))

    assert "DESCRIPTION:one\\ntwo\\nthree\tfour" in raw.decode()


@pytest.mark.parametrize(
    "bad", ["a@b.example;x=y", 'a@b.example"', "a b@example.org", "a@b,c.example", "<a@b>"]
)
def test_an_address_with_a_character_that_ends_a_value_is_not_an_address(bad: str) -> None:
    raw = render_event(
        event(attendees=[{"emailAddress": {"name": "Pat", "address": bad}}])
    ).decode()

    assert "mailto:a" not in raw
    assert 'ATTENDEE;CN="Pat":' in raw


# --- 9. one token source, not two -------------------------------------------------------------


def test_setting_both_an_inline_token_and_a_token_file_is_refused_with_the_reason() -> None:
    with pytest.raises(ValidationError, match="both set") as refused:
        OutlookConfig.model_validate(
            {
                "account": ADDRESS,
                "tenant_id": TENANT,
                "client_id": CLIENT_ID,
                "refresh_token": "the-inline-secret",
                "refresh_token_file": "/var/lib/outlook/token",
            }
        )

    assert "rotates" in str(refused.value)
    assert "the-inline-secret" not in str(refused.value)


def test_the_file_alone_resolves_ahead_of_the_environment(tmp_path: Path) -> None:
    token = tmp_path / "token"
    token.write_text("from-the-file\n")
    config = OutlookConfig.model_validate(
        {
            "account": ADDRESS,
            "tenant_id": TENANT,
            "client_id": CLIENT_ID,
            "refresh_token_file": str(token),
        }
    )

    resolved = resolve_outlook_credentials(config, {"OUTLOOK_REFRESH_TOKEN": "env"})

    assert resolved.refresh_token is not None
    assert resolved.refresh_token.get_secret_value() == "from-the-file"


# --- 10. the directory object is pinned -------------------------------------------------------


async def test_the_watermark_records_a_digest_of_the_object_id_and_never_the_id() -> None:
    server, time = tenant()
    server.add_message(mail("A"))
    async with opened(server, time, calendar=False) as connector:
        await drain(connector)
        watermark = connector.watermark

    assert watermark is not None
    recorded = watermark.metadata["object_identity"]
    assert isinstance(recorded, str)
    assert OBJECT_ID not in recorded
    assert OBJECT_ID not in json.dumps(watermark.model_dump(mode="json"))


async def test_an_address_reassigned_to_another_account_is_refused_against_the_watermark() -> None:
    server, time = tenant()
    server.add_message(mail("A"))
    async with opened(server, time, calendar=False) as connector:
        await drain(connector)
        watermark = connector.watermark
    server.object_id = OBJECT_ELSEWHERE  # the same address, another directory object
    async with opened(server, time, calendar=False) as later:
        before = len(server.requests)
        with pytest.raises(MailboxMismatchError) as refused:
            await drain(later, watermark)

    text = str(refused.value)
    assert OBJECT_ID not in text
    assert "22222222" not in text
    assert ADDRESS not in text
    assert not [r for r in server.requests[before:] if "/messages/delta" in str(r.url)]


async def test_the_same_object_resumes_and_a_first_sync_needs_no_recorded_object() -> None:
    server, time = tenant()
    first = server.add_message(mail("First"))
    async with opened(server, time, calendar=False) as connector:
        found = await drain(connector)
        watermark = connector.watermark
        server.add_message(mail("Second"))
        again = await drain(connector, watermark)

    assert ids(found) == ["msg:" + first]
    assert len(again) == 1


async def test_a_watermark_from_before_the_object_was_recorded_is_accepted() -> None:
    server, time = tenant()
    server.add_message(mail("A"))
    async with opened(server, time, calendar=False) as connector:
        await drain(connector)
        watermark = connector.watermark
        assert watermark is not None
        legacy = watermark.model_copy(
            update={
                "metadata": {k: v for k, v in watermark.metadata.items() if k != "object_identity"}
            }
        )
        found = await drain(connector, legacy)

    assert found == []


async def test_a_different_object_seen_later_in_the_same_process_is_refused() -> None:
    server, time = tenant()
    server.add_message(mail("A"))
    async with opened(server, time, calendar=False) as connector:
        found = await drain(connector)
        server.object_id = OBJECT_ELSEWHERE
        time.now += 120
        with pytest.raises(MailboxMismatchError):
            await connector.fetch(found[0].ref)
