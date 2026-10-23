"""Gmail through the container: what a misconfigured source says at startup.

The container validates a source's options against the registered model before it calls the
factory, and the factory resolves the credential before it would construct anything — so every
gap is a startup refusal that names the setting, rather than a run that indexes nothing.
"""

from __future__ import annotations

import pytest

from manicule.config.settings import ConnectorSettings, Settings
from manicule.connectors.plugin import PLUGIN
from manicule.container import Container
from manicule.core.errors import ConfigError
from manicule.plugins.registry import ComponentRegistry

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


async def test_a_complete_configuration_resolves_and_then_says_the_connector_is_not_built(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Everything this build has of Gmail ran; the object that would read mail does not exist."""
    monkeypatch.setenv("GMAIL_CLIENT_SECRET", "secret-sentinel")
    monkeypatch.setenv("GMAIL_REFRESH_TOKEN", "refresh-sentinel")

    with pytest.raises(ConfigError, match="not implemented yet") as caught:
        await container(settings, account="mail@example.org", client_id=CLIENT_ID).connector("mail")

    text = str(caught.value)
    assert "'mail'" in text
    assert "secret-sentinel" not in text
    assert "refresh-sentinel" not in text
    assert "example.org" not in text


async def test_the_factory_makes_no_network_call(
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

    with pytest.raises(ConfigError, match="not implemented yet"):
        await container(settings, account="mail@example.org", client_id=CLIENT_ID).connector("mail")
