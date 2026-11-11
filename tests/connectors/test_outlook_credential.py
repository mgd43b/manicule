"""The Outlook credential: refresh, rotation, typed refusals, and the mailbox check.

``docs/connectors/outlook.md`` §1. The real credential runs against the synthetic Entra and Graph.
"""

from __future__ import annotations

import asyncio
import logging
import stat
from pathlib import Path

import pytest
from pydantic import SecretStr

from manicule.connectors.errors import (
    CredentialError,
    GrantRevokedError,
    MailboxMismatchError,
    RateLimitedError,
    RemoteError,
)
from manicule.connectors.outlook_auth import OutlookCredential, outlook_credential, token_url
from manicule.connectors.outlook_config import resolve_outlook_credentials
from manicule.core.errors import ConfigError
from tests.connectors.fake_graph import (
    ADDRESS,
    REFRESH_TOKEN,
    TENANT,
    UPN,
    FakeGraph,
)
from tests.connectors.outlook_support import outlook_config

SCOPES = ("offline_access", "User.Read", "Mail.Read", "Calendars.Read")


def credential(
    server: FakeGraph, *, persist: object = None, clock: object = None
) -> OutlookCredential:
    return OutlookCredential(
        connector="work",
        tenant_id=TENANT,
        client_id="99999999-8888-7777-6666-555555555555",
        refresh_token=SecretStr(REFRESH_TOKEN),
        scopes=SCOPES,
        persist=persist,  # type: ignore[arg-type]
        transport=server.transport(),
        **({"clock": clock} if clock is not None else {}),  # type: ignore[arg-type]
    )


async def test_a_refresh_sends_the_public_client_form_to_this_tenants_endpoint() -> None:
    server = FakeGraph()
    token = await credential(server).access_token()

    assert token.startswith("access-token-")
    request = server.requests[0]
    assert str(request.url) == token_url(TENANT)
    form = request.content.decode()
    assert "client_secret" not in form, "a public client has no secret to send"
    assert "grant_type=refresh_token" in form
    assert server.asked_scopes == list(SCOPES)


async def test_the_token_is_cached_until_shortly_before_it_expires() -> None:
    server = FakeGraph()
    now = [0.0]
    cred = credential(server, clock=lambda: now[0])
    first = await cred.access_token()
    assert await cred.access_token() == first
    now[0] = 3600.0
    assert await cred.access_token() != first
    assert server.refreshes == 2


async def test_concurrent_callers_after_expiry_share_one_exchange() -> None:
    server = FakeGraph()
    cred = credential(server)
    tokens = await asyncio.gather(*(cred.access_token() for _ in range(8)))

    assert len(set(tokens)) == 1
    assert server.refreshes == 1


async def test_a_forced_refresh_replaces_the_cache() -> None:
    server = FakeGraph()
    cred = credential(server)
    first = await cred.access_token()
    second = await cred.access_token(refresh=True)

    assert second != first
    assert await cred.access_token() == second


# --- rotation --------------------------------------------------------------------------------


async def test_a_rotated_refresh_token_is_adopted_and_handed_to_where_it_is_kept() -> None:
    server = FakeGraph()
    server.rotate = True
    kept: list[str] = []

    def keep(token: SecretStr) -> None:
        kept.append(token.get_secret_value())

    cred = credential(server, persist=keep)
    await cred.access_token()
    await cred.access_token(refresh=True)

    assert cred.rotations == 2
    assert len(kept) == 2
    assert kept[0] != REFRESH_TOKEN
    assert kept[-1] == cred.refresh_token.get_secret_value()


async def test_the_next_refresh_uses_the_token_the_last_one_returned() -> None:
    server = FakeGraph()
    server.rotate = True
    cred = credential(server)
    await cred.access_token()
    after_first = cred.refresh_token.get_secret_value()
    await cred.access_token(refresh=True)

    assert after_first != REFRESH_TOKEN
    assert f"refresh_token={REFRESH_TOKEN}&" in server.requests[0].content.decode()
    assert f"refresh_token={after_first}&" in server.requests[1].content.decode()


async def test_an_unchanged_refresh_token_is_not_a_rotation() -> None:
    server = FakeGraph()
    kept: list[SecretStr] = []
    cred = credential(server, persist=kept.append)
    await cred.access_token()

    assert cred.rotations == 0
    assert kept == []


async def test_a_token_that_cannot_be_stored_is_logged_without_it_and_the_run_goes_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = FakeGraph()
    server.rotate = True

    def broken(token: SecretStr) -> None:
        raise PermissionError(f"/secret/path/{token.get_secret_value()}")

    cred = credential(server, persist=broken)
    with caplog.at_level(logging.WARNING):
        token = await cred.access_token()

    assert token
    assert "PermissionError" in caplog.text
    assert "/secret/path" not in caplog.text
    assert cred.refresh_token.get_secret_value() not in caplog.text


async def test_the_old_refresh_token_keeps_working_after_a_rotation_as_microsofts_does() -> None:
    server = FakeGraph()
    server.rotate = True
    await credential(server).access_token()

    again = await credential(server).access_token()
    assert again


async def test_a_file_instance_writes_the_rotated_token_back_readable_only_by_its_owner(
    tmp_path: Path,
) -> None:
    server = FakeGraph()
    server.rotate = True
    folder = tmp_path / "tokens"
    folder.mkdir()
    target = folder / "outlook-token"
    target.write_text(REFRESH_TOKEN + "\n")
    config = resolve_outlook_credentials(outlook_config(refresh_token_file=str(target)), {})
    cred = outlook_credential(config, instance="work", transport=server.transport())
    await cred.access_token()

    written = target.read_text().strip()
    assert written == cred.refresh_token.get_secret_value() != REFRESH_TOKEN
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert sorted(path.name for path in folder.glob("*")) == ["outlook-token"], "no staging file"


def test_the_factory_refuses_a_config_whose_token_was_never_resolved() -> None:
    config = outlook_config().model_copy(update={"refresh_token": None})

    with pytest.raises(ConfigError, match="resolve_outlook_credentials"):
        outlook_credential(config, instance="work")


# --- typed refusals --------------------------------------------------------------------------


@pytest.mark.parametrize("code", ["invalid_grant", "interaction_required", "consent_required"])
async def test_a_grant_that_needs_a_person_is_typed_and_names_the_fix(code: str) -> None:
    server = FakeGraph()
    server.refresh_failure = (400, code)
    with pytest.raises(GrantRevokedError) as refused:
        await credential(server).access_token()

    text = str(refused.value)
    assert "manicule connector login work" in text
    assert "sentinel-prose" not in text
    assert "user@example.org" not in text, "Microsoft's free text is never repeated"


async def test_a_rejected_app_registration_points_at_public_client_flows() -> None:
    server = FakeGraph()
    server.refresh_failure = (401, "invalid_client")
    with pytest.raises(CredentialError, match="public client flows"):
        await credential(server).access_token()


async def test_an_unknown_error_code_is_reported_as_unknown_not_quoted() -> None:
    server = FakeGraph()
    server.refresh_failure = (400, "Weird Free Text user@example.org")
    with pytest.raises(CredentialError, match=r"\(unknown\)") as refused:
        await credential(server).access_token()

    assert "user@example.org" not in str(refused.value)


async def test_throttling_and_outages_at_the_token_endpoint_are_their_own_types() -> None:
    server = FakeGraph()
    server.refresh_failure = (429, "temporarily_unavailable")
    with pytest.raises(RateLimitedError):
        await credential(server).access_token()
    server.refresh_failure = (503, "temporarily_unavailable")
    with pytest.raises(RemoteError):
        await credential(server).access_token()


def test_the_repr_names_the_instance_and_never_a_token() -> None:
    server = FakeGraph()

    assert repr(credential(server)) == "OutlookCredential(connector='work')"
    assert REFRESH_TOKEN not in repr(credential(server))


# --- the mailbox check -----------------------------------------------------------------------


@pytest.mark.parametrize("account", [ADDRESS, UPN])
async def test_the_account_may_be_either_identity_graph_reports(account: str) -> None:
    server = FakeGraph()

    await credential(server).confirm_mailbox(account)


async def test_a_token_for_another_mailbox_is_refused_and_neither_address_is_printed() -> None:
    server = FakeGraph()
    with pytest.raises(MailboxMismatchError) as refused:
        await credential(server).confirm_mailbox("someone.else@example.org")

    text = str(refused.value)
    assert ADDRESS not in text
    assert UPN not in text
    assert "someone.else" not in text
    assert "manicule connector login work" in text


async def test_a_profile_with_no_address_cannot_confirm_anything() -> None:
    server = FakeGraph()
    server.upn = server.mail_address = ""
    with pytest.raises(CredentialError, match="without an address"):
        await credential(server).confirm_mailbox(ADDRESS)
