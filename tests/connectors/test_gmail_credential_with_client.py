"""The refresh-token credential driving the Gmail REST client, over both synthetic servers.

Each side has its own fake — Google's token endpoint here, the mailbox in ``fake_gmail`` — and the
token endpoint mints tokens the mailbox accepts, so what is exercised is the seam between the two
pieces of work: the client's ``GmailCredential`` protocol and ``RefreshTokenCredential``.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import pytest
from pydantic import SecretStr

from manicule.connectors.credentials import RefreshTokenCredential
from manicule.connectors.errors import GrantRevokedError, TokenRejectedError
from manicule.connectors.gmail_client import GmailClient, GmailCredential
from tests.connectors.fake_gmail import ADDRESS, FakeGmail, FakeGmailSettings
from tests.connectors.fake_google_oauth import CLIENT_SECRET, REFRESH_TOKEN, FakeGoogleOAuth


@asynccontextmanager
async def running(client: GmailClient) -> AsyncGenerator[GmailClient]:
    await client.setup()
    try:
        yield client
    finally:
        await client.teardown()


def wired() -> tuple[FakeGmail, FakeGoogleOAuth, RefreshTokenCredential, GmailClient]:
    mailbox = FakeGmail()
    google = FakeGoogleOAuth(address=ADDRESS, issue=mailbox.issue_token)
    credential = RefreshTokenCredential(
        connector="mail",
        client_id="client-id.example.org",
        client_secret=SecretStr(CLIENT_SECRET),
        refresh_token=SecretStr(REFRESH_TOKEN),
        transport=google.transport,
    )
    shaped: GmailCredential = credential  # the protocol, checked by the type checker
    client = GmailClient(FakeGmailSettings(), shaped, transport=mailbox.transport())
    return mailbox, google, credential, client


async def test_the_client_reads_the_mailbox_with_tokens_the_credential_minted() -> None:
    _, google, _, client = wired()

    async with running(client):
        profile = await client.get_profile()

    assert profile.email_address == ADDRESS
    assert google.refreshes == 1


async def test_an_expired_token_is_refreshed_once_and_the_request_succeeds() -> None:
    mailbox, google, _, client = wired()

    async with running(client):
        await client.get_profile()
        mailbox.expire_tokens()
        profile = await client.get_profile()

    assert profile.email_address == ADDRESS
    assert google.refreshes == 2


async def test_a_token_rejected_even_after_a_refresh_is_the_credentials_typed_error() -> None:
    mailbox, _, _, client = wired()
    mailbox.rejecting_all_tokens = True

    async with running(client):
        with pytest.raises(TokenRejectedError, match="manicule connector login mail"):
            await client.get_profile()


async def test_a_revoked_grant_surfaces_through_the_client_as_the_credentials_error() -> None:
    _, google, _, client = wired()
    google.fail_refresh(400, "invalid_grant")

    async with running(client):
        with pytest.raises(GrantRevokedError, match="manicule connector login mail"):
            await client.get_profile()


async def test_the_mailbox_pin_and_the_clients_profile_agree_on_the_address() -> None:
    _, _, credential, client = wired()

    async with running(client):
        await credential.confirm_mailbox(ADDRESS)
        assert (await client.get_profile()).email_address == ADDRESS
