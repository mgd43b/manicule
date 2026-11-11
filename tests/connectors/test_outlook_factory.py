"""Outlook through the container: what a misconfigured source says at startup, and what a good one
builds.

The container validates a source's options against the registered model before it calls the
factory, and the factory resolves the token before it would construct anything, so every gap is a
startup refusal that names the setting rather than a run that indexes nothing. A complete
configuration builds the connector, and its ``setup()``, awaited before the container hands it
out, refuses a token that reads some other mailbox.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from manicule.config.settings import ConnectorSettings, Settings
from manicule.connectors import outlook_auth, outlook_client
from manicule.connectors.errors import MailboxMismatchError
from manicule.connectors.outlook import OutlookConnector
from manicule.connectors.outlook_auth import OutlookCredential
from manicule.connectors.outlook_config import OutlookConfig
from manicule.connectors.plugin import PLUGIN, build_outlook
from manicule.container import Container
from manicule.core.errors import ConfigError
from manicule.plugins.registry import BuildContext, ComponentRegistry
from tests.connectors.fake_graph import ADDRESS, CLIENT_ID, REFRESH_TOKEN, TENANT, FakeGraph, mail

OPTIONS = {"account": ADDRESS, "tenant_id": TENANT, "client_id": CLIENT_ID}


def container(settings: Settings, **options: Any) -> Container:
    registry = ComponentRegistry()
    PLUGIN.register(registry.bind("connectors"))
    source = ConnectorSettings(type="outlook", options={**options})
    return Container(settings.model_copy(update={"connectors": {"work": source}}), registry)


@pytest.fixture(autouse=True)
def _no_ambient_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OUTLOOK_REFRESH_TOKEN", raising=False)


@pytest.mark.parametrize("missing", ["account", "tenant_id", "client_id"])
async def test_a_missing_required_option_is_a_startup_error_naming_it_and_the_source(
    settings: Settings, missing: str
) -> None:
    options = {k: v for k, v in OPTIONS.items() if k != missing}
    with pytest.raises(ConfigError, match=missing) as caught:
        await container(settings, **options).connector("work")

    assert "work" in str(caught.value)


async def test_a_missing_token_names_every_place_it_looked(settings: Settings) -> None:
    with pytest.raises(ConfigError, match=r"refresh_token.*\$OUTLOOK_REFRESH_TOKEN.*work"):
        await container(settings, **OPTIONS).connector("work")


async def test_an_unrecognized_option_is_refused_rather_than_ignored(settings: Settings) -> None:
    with pytest.raises(ConfigError, match="client_secret"):
        await container(settings, **{**OPTIONS, "client_secret": "nope"}).connector("work")


async def test_a_tenant_alias_is_a_startup_error(settings: Settings) -> None:
    with pytest.raises(ConfigError, match="tenant_id"):
        await container(settings, **{**OPTIONS, "tenant_id": "common"}).connector("work")


@pytest.fixture
def tenant_through_the_fakes(monkeypatch: pytest.MonkeyPatch) -> FakeGraph:
    """Entra and Graph, reachable only through their synthetic selves, with a token set.

    The container awaits ``setup()`` before it hands a connector out, and that reads ``/me``; these
    route the token endpoint and the API through the fake the way ``doctor``'s suite does.
    """
    server = FakeGraph()
    real_credential = outlook_auth.outlook_credential
    real_client = outlook_client.OutlookClient

    def credential_through_the_fake(config: OutlookConfig, **kwargs: Any) -> OutlookCredential:
        return real_credential(config, transport=server.transport(), **kwargs)

    class ClientThroughTheFake(real_client):
        def __init__(self, settings: Any, credential: Any, **kwargs: Any) -> None:
            super().__init__(settings, credential, transport=server.transport(), **kwargs)

    monkeypatch.setattr(outlook_auth, "outlook_credential", credential_through_the_fake)
    monkeypatch.setattr(outlook_client, "OutlookClient", ClientThroughTheFake)
    monkeypatch.setenv("OUTLOOK_REFRESH_TOKEN", REFRESH_TOKEN)
    return server


async def test_a_complete_configuration_builds_a_connector_that_is_ready_to_read_mail(
    settings: Settings, tenant_through_the_fakes: FakeGraph
) -> None:
    tenant_through_the_fakes.add_message(mail("A"))

    built = await container(settings, **OPTIONS, calendar=False).connector("work")

    assert isinstance(built, OutlookConnector)
    assert built.name == "work"
    assert len([doc async for doc in built.discover(None)]) == 1
    await built.teardown()


async def test_a_token_for_another_mailbox_is_refused_before_the_container_hands_it_out(
    settings: Settings, tenant_through_the_fakes: FakeGraph
) -> None:
    tenant_through_the_fakes.upn = tenant_through_the_fakes.mail_address = "else@example.org"

    with pytest.raises(MailboxMismatchError) as caught:
        await container(settings, **OPTIONS).connector("work")

    paths = [
        r.url.path for r in tenant_through_the_fakes.requests if r.url.host == "graph.microsoft.com"
    ]
    assert paths == ["/v1.0/me"], "nothing was read before the refusal"
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
    monkeypatch.setenv("OUTLOOK_REFRESH_TOKEN", "r")
    context = BuildContext(
        settings=settings,
        config=OutlookConfig.model_validate(OPTIONS),
        data_dir=Path(),
        cache_dir=Path(),
        components=None,  # pyright: ignore[reportArgumentType] - unused by this factory
        instance="work",
    )

    built = build_outlook(context)

    assert isinstance(built, OutlookConnector)


def test_the_token_file_is_what_the_factory_reads_first_after_the_config(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    token = tmp_path / "token"
    token.write_text("from-the-file\n")
    monkeypatch.setenv("OUTLOOK_REFRESH_TOKEN", "from-the-environment")
    context = BuildContext(
        settings=settings,
        config=OutlookConfig.model_validate({**OPTIONS, "refresh_token_file": str(token)}),
        data_dir=Path(),
        cache_dir=Path(),
        components=None,  # pyright: ignore[reportArgumentType] - unused by this factory
        instance="work",
    )

    built = build_outlook(context)

    assert isinstance(built, OutlookConnector)
    assert "from-the-file" not in repr(built), "no secret is reachable from the connector's repr"
