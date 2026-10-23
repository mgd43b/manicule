"""The refresh-token credential, against a local stand-in for Google's token endpoint.

Everything here goes through ``httpx.MockTransport``, so the real request building runs — the
form body, the headers, the status handling — and nothing reaches a network. Secrets are
sentinel strings so a leak is a substring search rather than a judgment.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from manicule.connectors.config import GmailConfig, resolve_gmail_credentials
from manicule.connectors.credentials import (
    GOOGLE_TOKEN_URL,
    Authorization,
    RefreshTokenCredential,
    gmail_credential,
)
from manicule.connectors.errors import (
    ConnectorError,
    CredentialError,
    GrantRevokedError,
    MailboxMismatchError,
    RemoteError,
    TokenRejectedError,
)
from manicule.core.errors import ConfigError
from tests.connectors.fake_google_oauth import CLIENT_SECRET, REFRESH_TOKEN, FakeGoogleOAuth

ACCOUNT = "mail@example.org"


class Clock:
    """A clock a test moves by hand."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def credential(
    fake: FakeGoogleOAuth, clock: Clock | None = None, **overrides: object
) -> RefreshTokenCredential:
    arguments: dict[str, Any] = {
        "connector": "mail",
        "client_id": "client-id.example.org",
        "client_secret": SecretStr(CLIENT_SECRET),
        "refresh_token": SecretStr(REFRESH_TOKEN),
        "client_secret_env": "GMAIL_CLIENT_SECRET",
        "refresh_token_env": "GMAIL_REFRESH_TOKEN",
        "transport": fake.transport,
        "clock": clock or Clock(),
    }
    arguments.update(overrides)
    return RefreshTokenCredential(**arguments)


async def profile(fake: FakeGoogleOAuth, credential: RefreshTokenCredential) -> httpx.Response:
    async def attempt(authorization: Authorization) -> httpx.Response:
        headers = dict(authorization.headers)
        async with httpx.AsyncClient(transport=fake.transport) as http:
            return await http.get(
                "https://gmail.googleapis.com/gmail/v1/users/me/profile", headers=headers
            )

    return await credential.send(attempt)


# --- refresh and caching ------------------------------------------------------------------------


async def test_the_first_request_exchanges_the_refresh_token_for_a_bearer_header() -> None:
    fake = FakeGoogleOAuth()

    authorization = await credential(fake).authorize()

    assert authorization.headers == {"Authorization": "Bearer access-token-sentinel-1"}
    assert fake.refreshes == 1


async def test_the_exchange_is_a_post_form_and_never_a_query_string() -> None:
    fake = FakeGoogleOAuth()
    await credential(fake).authorize()

    (request,) = fake.requests
    assert request.method == "POST"
    assert str(request.url) == GOOGLE_TOKEN_URL, "no query string on the token endpoint"
    assert request.headers["content-type"] == "application/x-www-form-urlencoded"
    body = request.content.decode()
    assert "grant_type=refresh_token" in body
    assert REFRESH_TOKEN in body
    assert CLIENT_SECRET in body


async def test_the_token_is_cached_until_shortly_before_it_expires() -> None:
    fake, clock = FakeGoogleOAuth(expires_in=3600), Clock()
    cred = credential(fake, clock)

    await cred.authorize()
    clock.now += 3600 - 61  # one second before the 60-second margin
    await cred.authorize()
    assert fake.refreshes == 1

    clock.now += 2  # inside the margin
    renewed = await cred.authorize()
    assert fake.refreshes == 2
    assert renewed.headers["Authorization"].endswith("-2")


@pytest.mark.parametrize("expires_in", [None, "soon", 0, -5, True])
async def test_a_missing_or_malformed_lifetime_is_trusted_for_five_minutes_only(
    expires_in: object,
) -> None:
    fake, clock = FakeGoogleOAuth(expires_in=expires_in), Clock()
    cred = credential(fake, clock)

    await cred.authorize()
    clock.now += 299
    await cred.authorize()
    assert fake.refreshes == 1
    clock.now += 2
    await cred.authorize()
    assert fake.refreshes == 2


async def test_a_very_short_lifetime_is_still_used_for_part_of_it() -> None:
    fake, clock = FakeGoogleOAuth(expires_in=90), Clock()
    cred = credential(fake, clock)

    await cred.authorize()
    clock.now += 40
    await cred.authorize()
    assert fake.refreshes == 1
    clock.now += 10
    await cred.authorize()
    assert fake.refreshes == 2


async def test_concurrent_callers_share_one_exchange() -> None:
    fake = FakeGoogleOAuth()
    cred = credential(fake)

    results = await asyncio.gather(*(cred.authorize() for _ in range(8)))

    assert fake.refreshes == 1
    assert {r.headers["Authorization"] for r in results} == {"Bearer access-token-sentinel-1"}


async def test_access_token_serves_the_cache_and_refresh_true_mints_a_new_one() -> None:
    fake = FakeGoogleOAuth()
    cred = credential(fake)

    first = await cred.access_token()
    assert await cred.access_token() == first
    assert fake.refreshes == 1

    renewed = await cred.access_token(refresh=True)
    assert renewed != first
    assert fake.refreshes == 2
    assert await cred.access_token() == renewed, "the replacement is what the cache now holds"


async def test_access_token_surfaces_a_revoked_grant_as_the_typed_error() -> None:
    fake = FakeGoogleOAuth()
    fake.fail_refresh(400, "invalid_grant")

    with pytest.raises(GrantRevokedError, match="manicule connector login mail"):
        await credential(fake).access_token(refresh=True)


def test_the_request_timeout_setting_reaches_the_credential() -> None:
    config = GmailConfig.model_validate(
        {"account": ACCOUNT, "client_id": "id.example.org", "request_timeout_seconds": 7}
    )
    resolved = resolve_gmail_credentials(
        config, environ={"GMAIL_CLIENT_SECRET": "s", "GMAIL_REFRESH_TOKEN": "r"}
    )

    assert gmail_credential(resolved, instance="mail").timeout_seconds == 7


# --- one refresh and one retry on 401 -----------------------------------------------------------


async def test_a_401_refreshes_once_and_resends() -> None:
    fake = FakeGoogleOAuth()
    cred = credential(fake)
    await cred.authorize()
    fake.revoke_access_tokens()

    response = await profile(fake, cred)

    assert response.status_code == 200
    assert fake.refreshes == 2
    sent = [r for r in fake.requests if r.url.path.endswith("/profile")]
    assert [r.headers["Authorization"] for r in sent] == [
        "Bearer access-token-sentinel-1",
        "Bearer access-token-sentinel-2",
    ]


async def test_a_second_401_is_a_typed_credential_error_and_not_a_loop() -> None:
    fake = FakeGoogleOAuth()
    cred = credential(fake)

    async def always_401(_: Authorization) -> httpx.Response:
        return httpx.Response(401)

    with pytest.raises(TokenRejectedError, match=r"manicule connector login mail"):
        await cred.send(always_401)
    assert fake.refreshes == 2, "the first token, and the one minted for the retry — no more"


async def test_a_request_that_is_not_a_401_is_never_retried() -> None:
    fake = FakeGoogleOAuth()
    cred = credential(fake)
    calls = 0

    async def forbidden(_: Authorization) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(403)

    assert (await cred.send(forbidden)).status_code == 403
    assert calls == 1
    assert fake.refreshes == 1


async def test_a_token_already_replaced_by_another_request_is_not_refreshed_again() -> None:
    """Two requests fail on the same token; the second to take the lock finds a new one."""
    fake = FakeGoogleOAuth()
    cred = credential(fake)
    stale = await cred.authorize()
    fake.revoke_access_tokens()

    first, second = await asyncio.gather(cred._reauthorize(stale), cred._reauthorize(stale))  # pyright: ignore[reportPrivateUsage]

    assert fake.refreshes == 2, "one initial exchange and one replacement, shared"
    assert first == second


# --- failures are typed -------------------------------------------------------------------------


async def test_invalid_grant_is_a_typed_error_whose_renewal_names_the_login_command() -> None:
    fake = FakeGoogleOAuth()
    fake.fail_refresh(400, "invalid_grant")

    with pytest.raises(GrantRevokedError) as caught:
        await credential(fake).authorize()

    assert "manicule connector login mail" in str(caught.value)
    assert "sentinel-prose" not in str(caught.value), "Google's free text is not repeated"


async def test_a_rejected_oauth_client_names_the_variable_to_check() -> None:
    fake = FakeGoogleOAuth()
    fake.fail_refresh(401, "invalid_client")

    with pytest.raises(CredentialError, match=r"invalid_client.*\$GMAIL_CLIENT_SECRET") as caught:
        await credential(fake).authorize()

    assert not isinstance(caught.value, GrantRevokedError)


async def test_an_unrecognized_error_code_is_not_echoed_verbatim() -> None:
    fake = FakeGoogleOAuth()
    fake.fail_refresh(400, "CLIENT SECRET sentinel-leak!")

    with pytest.raises(CredentialError, match=r"\(unknown\)") as caught:
        await credential(fake).authorize()

    assert "sentinel-leak" not in str(caught.value)


async def test_a_server_error_from_the_token_endpoint_is_retryable_not_a_credential_verdict() -> (
    None
):
    fake = FakeGoogleOAuth()
    fake.token_failure = (503, {})

    with pytest.raises(RemoteError) as caught:
        await credential(fake).authorize()

    assert caught.value.status_code == 503
    assert not isinstance(caught.value, CredentialError)


async def test_an_unreachable_token_endpoint_is_a_connector_error() -> None:
    def refuse(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    cred = credential(FakeGoogleOAuth(), transport=httpx.MockTransport(refuse))

    with pytest.raises(ConnectorError, match="could not be reached"):
        await cred.authorize()


async def test_a_success_without_an_access_token_is_refused() -> None:
    cred = credential(
        FakeGoogleOAuth(),
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"expires_in": 3599})),
    )

    with pytest.raises(CredentialError, match="without an access token"):
        await cred.authorize()


# --- the mailbox pin ----------------------------------------------------------------------------


async def test_the_mailbox_matching_the_account_is_accepted_whatever_its_case() -> None:
    fake = FakeGoogleOAuth(address="Mail@Example.org")

    await credential(fake).confirm_mailbox(ACCOUNT)


async def test_a_token_for_another_mailbox_is_refused_and_no_address_is_printed() -> None:
    fake = FakeGoogleOAuth(address="someone.else@example.org")

    with pytest.raises(MailboxMismatchError) as caught:
        await credential(fake).confirm_mailbox(ACCOUNT)

    text = str(caught.value)
    assert "different mailbox" in text
    assert "someone.else" not in text
    assert ACCOUNT not in text
    assert "example.org" not in text
    assert "manicule connector login mail" in text


async def test_an_alias_of_the_account_counts_as_a_different_mailbox() -> None:
    """getProfile reports the primary address; configuring an alias is refused, not followed."""
    fake = FakeGoogleOAuth(address="primary@example.org")

    with pytest.raises(MailboxMismatchError, match="alias"):
        await credential(fake).confirm_mailbox("alias@example.org")


async def test_the_pin_check_survives_one_expired_access_token() -> None:
    fake = FakeGoogleOAuth()
    cred = credential(fake)
    await cred.authorize()
    fake.revoke_access_tokens()

    await cred.confirm_mailbox(ACCOUNT)

    assert fake.refreshes == 2


async def test_a_403_from_the_profile_read_says_what_to_check_and_prints_nothing_else() -> None:
    fake = FakeGoogleOAuth()
    fake.profile_status = 403

    with pytest.raises(RemoteError, match="Gmail API is enabled") as caught:
        await credential(fake).confirm_mailbox(ACCOUNT)

    assert caught.value.status_code == 403


async def test_a_profile_without_an_address_cannot_confirm_the_mailbox() -> None:
    fake = FakeGoogleOAuth(address="")

    with pytest.raises(CredentialError, match="without an address"):
        await credential(fake).confirm_mailbox(ACCOUNT)


# --- secrets never leave ------------------------------------------------------------------------


def secrets_in(text: str) -> list[str]:
    return [s for s in (CLIENT_SECRET, REFRESH_TOKEN, "access-token-sentinel") if s in text]


def test_the_credentials_repr_names_the_instance_and_nothing_else() -> None:
    fake = FakeGoogleOAuth()
    cred = credential(fake)

    assert repr(cred) == "RefreshTokenCredential(connector='mail')"
    assert secrets_in(repr(cred) + str(cred)) == []


async def test_no_log_record_carries_a_secret_or_an_address(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The request lines httpx and httpcore write at DEBUG, and anything this module logs."""
    fake = FakeGoogleOAuth()
    cred = credential(fake)
    caplog.set_level(logging.DEBUG)

    await cred.confirm_mailbox(ACCOUNT)
    with pytest.raises(MailboxMismatchError):
        await credential(FakeGoogleOAuth(address="other@example.org")).confirm_mailbox(ACCOUNT)

    logged = "\n".join(f"{r.name} {r.getMessage()}" for r in caplog.records)
    assert logged, "the assertion below must be able to fail"
    assert secrets_in(logged) == []
    assert "example.org" not in logged


async def test_no_error_message_carries_a_secret() -> None:
    messages: list[str] = []
    for status, code in [(400, "invalid_grant"), (401, "invalid_client"), (400, "weird")]:
        fake = FakeGoogleOAuth()
        fake.fail_refresh(status, code)
        with pytest.raises(CredentialError) as caught:
            await credential(fake).authorize()
        messages.append(str(caught.value))

    assert secrets_in("\n".join(messages)) == []


async def test_the_tracebacks_locals_never_hold_a_plain_secret_on_the_credential() -> None:
    cred = credential(FakeGoogleOAuth())
    await cred.authorize()

    for value in (cred.client_secret, cred.refresh_token, cred._access):  # pyright: ignore[reportPrivateUsage]
        assert isinstance(value, SecretStr)
        assert "sentinel" not in repr(value)


# --- construction and discovery -----------------------------------------------------------------


def test_the_factory_helper_refuses_a_config_whose_secrets_were_never_resolved() -> None:
    config = GmailConfig.model_validate({"account": ACCOUNT, "client_id": "id.example.org"})

    with pytest.raises(ConfigError, match="resolve_gmail_credentials"):
        gmail_credential(config, instance="mail")


def test_a_resolved_config_builds_a_credential_that_remembers_the_variable_names() -> None:
    config = GmailConfig.model_validate({"account": ACCOUNT, "client_id": "id.example.org"})
    resolved = resolve_gmail_credentials(
        config, environ={"GMAIL_CLIENT_SECRET": "s", "GMAIL_REFRESH_TOKEN": "r"}
    )

    cred = gmail_credential(resolved, instance="mail")

    assert cred.client_secret_env == "GMAIL_CLIENT_SECRET"  # noqa: S105 - a variable name
    assert cred.connector == "mail"


def test_importing_the_credential_module_loads_no_http_client() -> None:
    """Plugin discovery imports configuration; the refresh imports httpx when it runs."""
    script = (
        "import sys\n"
        "import manicule.connectors.credentials, manicule.connectors.config\n"
        "print(sorted(m for m in sys.modules if m == 'httpx' or m.startswith('httpx.')))\n"
    )
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-I", "-c", script], capture_output=True, text=True, check=True
    )

    assert completed.stdout.strip() == "[]"
