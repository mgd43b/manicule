"""Gmail through the container: what a misconfigured source says at startup, and what a good one
builds.

The container validates a source's options against the registered model before it calls the
factory, and the factory resolves the credential before it would construct anything — so every
gap is a startup refusal that names the setting, rather than a run that indexes nothing. A
complete configuration builds the connector, and its ``setup()`` — awaited before the container
hands it out — refuses a token that reads some other mailbox.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from manicule.config.settings import ConnectorSettings, Settings
from manicule.connectors import credentials, gmail_client
from manicule.connectors.config import GmailConfig
from manicule.connectors.credentials import RefreshTokenCredential
from manicule.connectors.errors import MailboxMismatchError
from manicule.connectors.gmail import GmailConnector
from manicule.connectors.plugin import PLUGIN, build_gmail
from manicule.container import Container
from manicule.core.errors import ConfigError
from manicule.plugins.registry import BuildContext, ComponentRegistry
from tests.connectors.fake_gmail import FakeGmail
from tests.connectors.fake_google_oauth import CLIENT_SECRET, REFRESH_TOKEN, FakeGoogleOAuth

CLIENT_ID = "1234567890-example.apps.googleusercontent.com"


def container(settings: Settings, **options: str) -> Container:
    registry = ComponentRegistry()
    PLUGIN.register(registry.bind("connectors"))
    source = ConnectorSettings(type="gmail", options={**options})
    return Container(settings.model_copy(update={"connectors": {"mail": source}}), registry)


@pytest.fixture(autouse=True)
def _no_ambient_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("GMAIL_CLIENT_SECRET", "GMAIL_REFRESH_TOKEN"):
        monkeypatch.delenv(name, raising=False)


async def test_a_missing_account_is_a_startup_error_naming_the_setting_and_the_source(
    settings: Settings,
) -> None:
    with pytest.raises(ConfigError, match=r"account") as caught:
        await container(settings, client_id=CLIENT_ID).connector("mail")

    assert "mail" in str(caught.value)


async def test_a_missing_client_id_is_a_startup_error_naming_the_setting(
    settings: Settings,
) -> None:
    with pytest.raises(ConfigError, match="client_id"):
        await container(settings, account="mail@example.org").connector("mail")


async def test_a_missing_client_secret_names_the_setting_and_the_environment_variable(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GMAIL_REFRESH_TOKEN", "refresh-sentinel")

    with pytest.raises(ConfigError, match=r"client_secret.*\$GMAIL_CLIENT_SECRET.*mail"):
        await container(settings, account="mail@example.org", client_id=CLIENT_ID).connector("mail")


async def test_a_missing_refresh_token_names_the_setting_and_the_environment_variable(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GMAIL_CLIENT_SECRET", "secret-sentinel")

    with pytest.raises(ConfigError, match=r"refresh_token.*\$GMAIL_REFRESH_TOKEN"):
        await container(settings, account="mail@example.org", client_id=CLIENT_ID).connector("mail")


async def test_an_unrecognized_option_is_refused_rather_than_ignored(settings: Settings) -> None:
    with pytest.raises(ConfigError, match="query"):
        await container(
            settings, account="mail@example.org", client_id=CLIENT_ID, query="from:boss"
        ).connector("mail")


@pytest.fixture
def mailbox_through_the_fakes(monkeypatch: pytest.MonkeyPatch) -> FakeGmail:
    """Google and Gmail, reachable only through their synthetic selves, with both secrets set.

    The container awaits ``setup()`` before it hands a connector out, and that reads the profile;
    these route the token endpoint and the API through the fakes the way ``doctor``'s suite does.
    """
    server = FakeGmail(address="mail@example.org")
    google = FakeGoogleOAuth(address="mail@example.org", issue=server.issue_token)
    real_credential = credentials.gmail_credential
    real_client = gmail_client.GmailClient

    def credential_through_the_fake(config: GmailConfig, **kwargs: Any) -> RefreshTokenCredential:
        return real_credential(config, transport=google.transport, **kwargs)

    class ClientThroughTheFake(real_client):
        def __init__(self, settings: Any, credential: Any, **kwargs: Any) -> None:
            super().__init__(settings, credential, transport=server.transport(), **kwargs)

    monkeypatch.setattr(credentials, "gmail_credential", credential_through_the_fake)
    monkeypatch.setattr(gmail_client, "GmailClient", ClientThroughTheFake)
    monkeypatch.setenv("GMAIL_CLIENT_SECRET", CLIENT_SECRET)
    monkeypatch.setenv("GMAIL_REFRESH_TOKEN", REFRESH_TOKEN)
    return server


async def test_a_complete_configuration_builds_a_connector_that_is_ready_to_read_mail(
    settings: Settings, mailbox_through_the_fakes: FakeGmail
) -> None:
    mailbox_through_the_fakes.add_message()

    built = await container(settings, account="mail@example.org", client_id=CLIENT_ID).connector(
        "mail"
    )

    assert isinstance(built, GmailConnector)
    assert built.name == "mail"
    found = [doc async for doc in built.discover(None)]
    assert len(found) == 1
    await built.teardown()


async def test_a_token_for_another_mailbox_is_refused_before_the_container_hands_it_out(
    settings: Settings, mailbox_through_the_fakes: FakeGmail
) -> None:
    mailbox_through_the_fakes.switch_mailbox("someone.else@example.org")

    with pytest.raises(MailboxMismatchError) as caught:
        await container(settings, account="mail@example.org", client_id=CLIENT_ID).connector("mail")

    paths = [
        r.url.path.removeprefix("/gmail/v1/users/me") for r in mailbox_through_the_fakes.requests
    ]
    assert paths == ["/profile"], "nothing was read before the refusal"
    assert "example.org" not in str(caught.value)


def test_the_factory_makes_no_network_call(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It runs inside the container's event loop; a refresh there would block every request."""
    import httpx  # noqa: PLC0415

    def forbidden(*_: object, **__: object) -> None:
        msg = "the factory touched the network"
        raise AssertionError(msg)

    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden)
    monkeypatch.setenv("GMAIL_CLIENT_SECRET", "s")
    monkeypatch.setenv("GMAIL_REFRESH_TOKEN", "r")
    context = BuildContext(
        settings=settings,
        config=GmailConfig(account="mail@example.org", client_id=CLIENT_ID),
        data_dir=Path(),
        cache_dir=Path(),
        components=None,  # pyright: ignore[reportArgumentType] - unused by this factory
        instance="mail",
    )

    built = build_gmail(context)

    assert isinstance(built, GmailConnector)
