"""``doctor``'s ``gmail-credential`` check: the one place a credential is asked about by name.

Check names are a contract — monitors select on them — so the name is pinned here, along with
the closed set of words the facts may carry. Google is a ``MockTransport``; addresses are at
``example.org``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from manicule.app import results as r
from manicule.app.service import ApplicationService
from manicule.config.settings import ConnectorSettings, Settings
from manicule.connectors import credentials
from manicule.connectors.credentials import RefreshTokenCredential
from tests.app.fakes import FakeBackend
from tests.connectors.fake_google_oauth import CLIENT_SECRET, REFRESH_TOKEN, FakeGoogleOAuth

if TYPE_CHECKING:
    from manicule.connectors.config import GmailConfig

CHECK = "gmail-credential"
OPTIONS = {"account": "mail@example.org", "client_id": "id.example.org"}


def _without_httpx(name: str) -> object | None:
    """``importlib.util.find_spec`` on an installation lacking the HTTP client."""
    return None if name == "httpx" else object()


@pytest.fixture
def google(monkeypatch: pytest.MonkeyPatch) -> FakeGoogleOAuth:
    """Google, reachable only through the fake, with both secrets in the environment."""
    fake = FakeGoogleOAuth()
    real = credentials.gmail_credential

    def through_the_fake(config: GmailConfig, **kwargs: Any) -> RefreshTokenCredential:
        return real(config, transport=fake.transport, **kwargs)

    monkeypatch.setattr(credentials, "gmail_credential", through_the_fake)
    monkeypatch.setenv("GMAIL_CLIENT_SECRET", CLIENT_SECRET)
    monkeypatch.setenv("GMAIL_REFRESH_TOKEN", REFRESH_TOKEN)
    return fake


def service_with(**connectors: dict[str, Any]) -> ApplicationService:
    backend = FakeBackend()
    backend.settings = Settings().model_copy(
        update={
            "connectors": {
                name: ConnectorSettings.model_validate(spec) for name, spec in connectors.items()
            }
        }
    )
    return ApplicationService(backend)


async def gmail_check(**connectors: dict[str, Any]) -> r.Check:
    diagnosis = await service_with(**connectors).doctor()
    for check in diagnosis.checks:
        if check.name == CHECK:
            return check
    offered = ", ".join(c.name for c in diagnosis.checks)
    message = f"no check named {CHECK!r}; the diagnosis carried: {offered}"
    raise AssertionError(message)


def mail(**options: object) -> dict[str, Any]:
    return {"type": "gmail", "options": {**OPTIONS, **options}}


async def test_a_corpus_without_gmail_sees_the_check_pass(google: FakeGoogleOAuth) -> None:
    check = await gmail_check(notes={"type": "filesystem", "options": {"root": "/x"}})

    assert check.state == "ok"
    assert check.facts == {"sources": {}}
    assert google.requests == [], "nothing is asked of Google when no source needs it"


async def test_a_disabled_source_is_not_asked_about(google: FakeGoogleOAuth) -> None:
    check = await gmail_check(mail={**mail(), "enabled": False})

    assert check.state == "ok"
    assert google.requests == []


async def test_a_working_credential_for_the_right_mailbox_is_ok(google: FakeGoogleOAuth) -> None:
    check = await gmail_check(mail=mail())

    assert check.state == "ok"
    assert check.facts == {"sources": {"mail": "ok"}}
    assert google.refreshes == 1


async def test_a_missing_secret_fails_naming_the_source_and_asks_nothing_of_google(
    google: FakeGoogleOAuth, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GMAIL_REFRESH_TOKEN")

    check = await gmail_check(mail=mail())

    assert check.state == "failing"
    assert check.facts == {"sources": {"mail": "missing_secret"}}
    assert "'mail'" in check.detail
    assert "client_secret" in check.remedy
    assert google.requests == []


async def test_a_revoked_grant_fails_with_the_login_command_as_the_remedy(
    google: FakeGoogleOAuth,
) -> None:
    google.fail_refresh(400, "invalid_grant")

    check = await gmail_check(mail=mail())

    assert check.state == "failing"
    assert check.facts == {"sources": {"mail": "grant_revoked"}}
    assert check.remedy == "manicule connector login mail"


async def test_a_refused_oauth_client_fails(google: FakeGoogleOAuth) -> None:
    google.fail_refresh(401, "invalid_client")

    check = await gmail_check(mail=mail())

    assert check.state == "failing"
    assert check.facts == {"sources": {"mail": "credential_rejected"}}


async def test_a_token_for_another_mailbox_fails_and_prints_no_address(
    google: FakeGoogleOAuth,
) -> None:
    google.address = "someone.else@example.org"

    check = await gmail_check(mail=mail())

    assert check.state == "failing"
    assert check.facts == {"sources": {"mail": "different_mailbox"}}
    payload = check.model_dump_json()
    assert "example.org" not in payload
    assert "someone.else" not in payload


async def test_google_not_answering_degrades_rather_than_fails(google: FakeGoogleOAuth) -> None:
    google.token_failure = (503, {})

    check = await gmail_check(mail=mail())

    assert check.state == "degraded"
    assert check.facts == {"sources": {"mail": "unreachable"}}
    assert "reach Google" in check.remedy


async def test_each_source_is_reported_and_the_first_broken_one_names_the_remedy(
    google: FakeGoogleOAuth,
) -> None:
    check = await gmail_check(
        alpha=mail(),
        beta=mail(account="other@example.org"),
    )

    assert check.state == "failing"
    assert check.facts == {"sources": {"alpha": "ok", "beta": "different_mailbox"}}
    assert check.remedy == "manicule connector login beta"


async def test_options_that_do_not_validate_fail_the_check_instead_of_vanishing(
    google: FakeGoogleOAuth,
) -> None:
    """Nothing else reports them, and a dropped source reads as a corpus with no Gmail in it."""
    check = await gmail_check(mail={"type": "gmail", "options": {"client_id": "x"}})

    assert check.state == "failing"
    assert check.facts == {"sources": {"mail": "invalid_options"}}
    assert "'mail'" in check.detail
    assert "account and client_id" in check.remedy
    assert google.requests == []


async def test_a_rejected_value_is_not_echoed_when_options_do_not_validate(
    google: FakeGoogleOAuth,
) -> None:
    check = await gmail_check(mail=mail(account="not-an-address-sentinel"))

    assert check.facts == {"sources": {"mail": "invalid_options"}}
    assert "sentinel" not in check.model_dump_json()


async def test_nothing_a_secret_or_a_token_reaches_the_check(google: FakeGoogleOAuth) -> None:
    google.fail_refresh(400, "invalid_grant")

    check = await gmail_check(mail=mail())

    payload = json.dumps(check.model_dump())
    for secret in (CLIENT_SECRET, REFRESH_TOKEN, "access-token-sentinel", "sentinel-prose"):
        assert secret not in payload


async def test_the_check_follows_sessions_in_the_order_doctor_emits_them() -> None:
    diagnosis = await service_with().doctor()
    names = [check.name for check in diagnosis.checks]

    assert names.index(CHECK) == names.index("sessions") + 1


async def test_the_gmail_api_being_off_fails_with_the_login_remedy_and_not_the_network(
    google: FakeGoogleOAuth,
) -> None:
    google.profile_status = 403

    check = await gmail_check(mail=mail())

    assert check.state == "failing"
    assert check.facts == {"sources": {"mail": "credential_rejected"}}
    assert "reach Google" not in check.remedy


async def test_a_probe_that_outlasts_the_deadline_is_unreachable_and_does_not_hold_doctor(
    google: FakeGoogleOAuth, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio  # noqa: PLC0415

    from manicule.app import service as service_module  # noqa: PLC0415

    async def never(self: object, account: str) -> None:
        del self, account
        await asyncio.sleep(3600)

    monkeypatch.setattr(RefreshTokenCredential, "confirm_mailbox", never)
    monkeypatch.setattr(service_module, "GMAIL_PROBE_DEADLINE_SECONDS", 0.05)

    check = await gmail_check(alpha=mail(), beta=mail(account="other@example.org"))

    assert check.state == "degraded"
    assert check.facts == {"sources": {"alpha": "unreachable", "beta": "unreachable"}}


async def test_a_missing_http_client_fails_with_the_extra_to_install(
    google: FakeGoogleOAuth, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A machine installed without the connectors extra gets a diagnosis, not a traceback."""
    monkeypatch.setattr("importlib.util.find_spec", _without_httpx)

    check = await gmail_check(mail=mail())

    assert check.state == "failing"
    assert check.facts == {"sources": {"mail": "dependency_missing"}}
    assert check.remedy == "pip install 'manicule[connectors]'"


async def test_options_set_in_the_types_global_defaults_are_in_force_for_the_check(
    google: FakeGoogleOAuth,
) -> None:
    """The container merges ``plugins.config."connector.gmail"`` under each instance's options."""
    backend = FakeBackend()
    backend.settings = Settings.model_validate(
        {
            "plugins": {"config": {"connector.gmail": OPTIONS}},
            "connectors": {"mail": {"type": "gmail"}},
        }
    )

    diagnosis = await ApplicationService(backend).doctor()
    check = next(c for c in diagnosis.checks if c.name == CHECK)

    assert check.facts == {"sources": {"mail": "ok"}}
