"""The obligations every connector carries, and how this one is wired in.

``assert_connector_contract`` is the shipped suite: three methods, and the one that is easy to
leave out. Everything else here is about the seams — registration through the public entry
point, configuration that is validated rather than ignored, and a credential checked before
anything is constructed.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from pydantic import BaseModel, SecretStr, ValidationError

from manicule.connectors import (
    ConfluenceConfig,
    ConnectorError,
    Deployment,
    resolve_credentials,
)
from manicule.connectors.plugin import PLUGIN, build_confluence
from manicule.container import keys
from manicule.core.errors import ConfigError
from manicule.core.protocols import Connector
from manicule.plugins import ComponentRegistry, discover
from manicule.plugins.manifest import ComponentKind
from manicule.plugins.registry import BuildContext
from manicule.testing import assert_connector_contract, assert_protocol_signatures
from tests.connectors.fake_confluence import FakeAttachment, FakeConfluence, FakePage
from tests.connectors.support import Waits, cloud_config, connected


@pytest.mark.contract
async def test_the_confluence_connector_satisfies_the_connector_contract() -> None:
    """Including that ``reconcile`` reports what ``discover`` just returned.

    A connector that reports nothing from reconciliation is claiming its source is empty, and
    the pipeline would act on that claim.
    """
    instance = FakeConfluence(
        pages=[FakePage(id="1", title="A", space="ENG"), FakePage(id="2", title="B", space="ENG")],
        attachments=[
            FakeAttachment(id="att-9", title="d.pdf", space="ENG", page_id="1", page_title="A")
        ],
    )
    connector = await connected(instance)
    try:
        await assert_connector_contract(connector)
    finally:
        await connector.teardown()


@pytest.mark.contract
async def test_the_connector_matches_the_protocol_signature_by_signature() -> None:
    """``@runtime_checkable`` checks that attributes exist, never what they accept."""
    instance = FakeConfluence(pages=[FakePage(id="1", title="A", space="ENG")])
    connector = await connected(instance)
    try:
        assert_protocol_signatures(connector, Connector)
    finally:
        await connector.teardown()


@pytest.mark.contract
def test_the_connector_registers_through_the_public_entry_point() -> None:
    """The same route a third-party connector takes. A shorter internal one could rot unseen."""
    from manicule.plugins import ENTRY_POINT_GROUP, installed_entry_points  # noqa: PLC0415
    from tests.test_import_boundary import STALE_INSTALL  # noqa: PLC0415 - one diagnosis

    found = {point.name: point.value for point in installed_entry_points(ENTRY_POINT_GROUP)}

    assert found.get("connectors") == "manicule.connectors.plugin:PLUGIN", STALE_INSTALL


@pytest.mark.contract
def test_discovery_finds_the_connector_with_its_configuration_model() -> None:
    """Without a declared model, settings written for a component are rejected rather than
    silently ignored — and a connector's settings include where it points and as whom."""
    registry = discover().registry
    record = registry.record(keys.CONNECTOR.named("confluence"))

    assert record.config_model is ConfluenceConfig
    assert "confluence" in registry.names(ComponentKind.CONNECTOR)


@pytest.mark.contract
def test_registering_the_connector_loads_no_http_client() -> None:
    """Discovery runs before configuration is read, in every process that starts.

    Registration needs the configuration model and nothing else, so ``manicule doctor`` on a
    machine that never syncs anything does not pay for an HTTP stack.
    """
    script = (
        "import json, sys\n"
        "before = set(sys.modules)\n"
        "from manicule.plugins import discover\n"
        "discover()\n"
        "print(json.dumps(sorted(set(sys.modules) - before)))\n"
    )
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-I", "-c", script], capture_output=True, text=True, check=True
    )
    loaded: list[str] = json.loads(completed.stdout)

    assert not [name for name in loaded if name == "httpx" or name.startswith("httpx.")]


def test_a_factory_handed_the_wrong_configuration_refuses_it() -> None:
    """Substituting defaults would build a connector whose settings appear to be in force."""

    class Other(BaseModel):
        pass

    context = BuildContext(
        settings=None,  # pyright: ignore[reportArgumentType] - unused on this path
        config=Other(),
        data_dir=None,  # pyright: ignore[reportArgumentType] - unused on this path
        cache_dir=None,  # pyright: ignore[reportArgumentType] - unused on this path
        components=None,  # pyright: ignore[reportArgumentType] - unused on this path
    )
    with pytest.raises(ConfigError, match="ConfluenceConfig"):
        build_confluence(context)


def test_the_plugin_registers_the_sources_this_build_ships() -> None:
    """Confluence live, Confluence from disk, and the local filesystem ``manicule index`` walks.

    Asserted as an exact list rather than a membership check: a connector registered and
    never mentioned anywhere is a source nobody knows they can configure.

    ``confluence-snapshot`` is a third entry rather than a mode of ``confluence`` because the two
    share no configuration — it has no base URL, no credential and no deployment. Folding them
    together would mean a config model where over half the fields are refused depending on another
    field's value, and a connector that reaches no network could then be misconfigured into trying.
    """
    registry = ComponentRegistry()
    PLUGIN.register(registry.bind("connectors"))

    assert registry.names(ComponentKind.CONNECTOR) == [
        "confluence",
        "confluence-snapshot",
        "filesystem",
    ]


def test_a_cloud_token_without_an_email_is_refused_before_anything_is_built() -> None:
    """Basic auth is ``email:token``. The token alone authenticates as nobody, and the 401 that
    follows reads exactly like a revoked token."""
    config = cloud_config(email="")
    with pytest.raises(ConfigError, match="email:token"):
        resolve_credentials(config, environ={})


def test_a_missing_token_names_the_variable_it_would_have_come_from() -> None:
    config = ConfluenceConfig(base_url="https://example.atlassian.net/wiki", email="a@b.c")
    with pytest.raises(ConfigError, match="CONFLUENCE_API_TOKEN"):
        resolve_credentials(config, environ={})


def test_a_token_in_the_environment_is_used_when_none_is_configured() -> None:
    """A credential in a configuration file is a credential in version control eventually."""
    config = ConfluenceConfig(base_url="https://example.atlassian.net/wiki", email="a@b.c")
    resolved = resolve_credentials(config, environ={"CONFLUENCE_API_TOKEN": "from-env"})

    assert resolved.api_token is not None
    assert resolved.api_token.get_secret_value() == "from-env"


def test_an_explicit_token_is_never_overwritten_by_the_environment() -> None:
    config = cloud_config(api_token=SecretStr("explicit"))
    resolved = resolve_credentials(config, environ={"CONFLUENCE_API_TOKEN": "from-env"})

    assert resolved.api_token is not None
    assert resolved.api_token.get_secret_value() == "explicit"


def test_a_blank_environment_variable_counts_as_absent() -> None:
    """An empty variable is what a shell reports for one that was never set in the file it
    sourced, and authenticating with an empty password produces a 401 that reads like a wrong
    password."""
    config = ConfluenceConfig(base_url="https://example.atlassian.net/wiki", email="a@b.c")
    with pytest.raises(ConfigError):
        resolve_credentials(config, environ={"CONFLUENCE_API_TOKEN": ""})


def test_a_server_deployment_needs_a_personal_access_token() -> None:
    config = ConfluenceConfig(base_url="https://wiki.example.com", deployment=Deployment.SERVER)
    with pytest.raises(ConfigError, match="personal access token"):
        resolve_credentials(config, environ={})


def test_a_base_url_that_is_not_a_url_is_refused_at_validation() -> None:
    """Misconfiguration fails before construction, with the shape it wanted spelled out."""
    with pytest.raises(ValidationError, match="absolute http"):
        ConfluenceConfig(base_url="example.atlassian.net")


@pytest.mark.parametrize(
    ("base_url", "reason"),
    [
        ("https://", "hostname"),
        ("https://sync.user:sentinel@wiki.example.test/wiki", "credentials"),
        ("https://wiki.example.test/wiki?token=sentinel", "query string"),
        ("https://wiki.example.test/wiki#sentinel", "fragment"),
    ],
    ids=["hostless", "userinfo", "query", "fragment"],
)
def test_a_confluence_base_url_is_a_secret_free_site_root(base_url: str, reason: str) -> None:
    """These components are not a site root, and none may be reflected into configuration errors."""
    with pytest.raises(ValidationError, match=reason) as raised:
        ConfluenceConfig(base_url=base_url)

    assert "sentinel" not in str(raised.value)


@pytest.mark.parametrize(
    "base_url",
    [
        "https://wiki.example.test/wiki/",
        "https://[2001:db8::10]:8443/wiki/",
        "https://bücher.example/wiki/",
        "https://xn--bcher-kva.example/wiki/",
    ],
)
def test_a_confluence_base_url_keeps_valid_context_paths_in_one_spelling(base_url: str) -> None:
    """IPv6, ports, and host spellings remain valid while a trailing slash cannot fork scope."""
    config = ConfluenceConfig(base_url=base_url)

    assert not config.base_url.endswith("/")


def test_a_setting_nobody_declared_is_rejected_rather_than_ignored() -> None:
    """A setting that appears to be in force and silently is not is worse than one that fails."""
    with pytest.raises(ValidationError):
        ConfluenceConfig.model_validate(
            {"base_url": "https://example.atlassian.net/wiki", "spaes": ["ENG"]}
        )


async def test_health_asks_the_source_rather_than_reporting_on_itself() -> None:
    instance = FakeConfluence(pages=[FakePage(id="1", title="A", space="ENG")])
    connector = await connected(instance)
    try:
        report = await connector.health()
    finally:
        await connector.teardown()

    assert report.ok


async def test_health_says_what_to_change_when_the_source_refuses() -> None:
    """A rejected credential is a startup fact, and a startup fact should name its remedy."""
    import httpx  # noqa: PLC0415

    from manicule.connectors.client import ConfluenceClient  # noqa: PLC0415
    from manicule.connectors.confluence import ConfluenceConnector  # noqa: PLC0415

    def handle(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(401, json={})

    config = cloud_config()
    client = ConfluenceClient(config, transport=httpx.MockTransport(handle), sleep=Waits())
    connector = ConfluenceConnector(config, client)
    await connector.setup()
    try:
        report = await connector.health()
    finally:
        await connector.teardown()

    assert not report.ok
    assert "credential" in report.remedy


async def test_using_the_client_before_setup_says_so() -> None:
    """Rather than an AttributeError from somewhere three frames down."""
    from manicule.connectors.client import ConfluenceClient  # noqa: PLC0415

    client = ConfluenceClient(cloud_config())
    with pytest.raises(ConnectorError, match="before setup"):
        await client.get_json("https://example.atlassian.net/wiki/rest/api/space")
