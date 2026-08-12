"""The built-in connectors plugin.

Registered through the public ``manicule.plugins`` entry point, exactly as a third-party
plugin is. There is no shorter internal route, so the extension mechanism is exercised by
every installation rather than only by the people extending it.

**Nothing here imports an HTTP client.** Registration needs one thing eagerly — the
configuration model, so settings written for the connector are validated rather than silently
ignored — and it lives in :mod:`manicule.connectors.config`, which imports nothing heavier
than pydantic. Plugin discovery runs before configuration is read, in every process that
starts, including ``manicule doctor`` on a machine that is never going to sync anything.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from manicule.connectors.config import (
    CONNECTOR_NAME,
    FILESYSTEM_CONNECTOR_NAME,
    ConfluenceConfig,
    FilesystemConfig,
    resolve_credentials,
)
from manicule.container import keys
from manicule.core.errors import ConfigError
from manicule.plugins import BuildContext, ComponentRegistry, Plugin, PluginManifest

if TYPE_CHECKING:
    from manicule.core.protocols import Connector

__all__ = ["PLUGIN", "ConnectorsPlugin", "build_confluence", "build_filesystem"]


def build_confluence(context: BuildContext) -> Connector:
    """Construct the Confluence connector from validated configuration.

    The credential is resolved and checked **here**, before construction, because a connector
    that discovers its missing token at the first page of the first sync produces a run that
    reports progress and indexes nothing. It happens in two steps for two kinds of credential:
    :func:`~manicule.connectors.config.resolve_credentials` fills a token in from the
    environment, and :func:`~manicule.connectors.credentials.credential_for` builds the object
    each request will be made with — which for a browser session means reading the keychain and
    refusing a session already too old to use.

    Raises:
        ConfigError: The context carries configuration of some other type, or the credential
            this deployment needs is absent. The container validates against the registered
            model before calling a factory, so the first case is reachable only by a caller
            building this some other way — and substituting defaults there would build a
            connector whose settings appear to be in force and are not.
        SessionExpiredError: A browser session was captured for this instance and manicule will
            not use one that old. A startup refusal on purpose: the answer is a person going to
            a browser, and hearing that at the first page of a sync wastes the run.
    """
    from manicule.connectors.client import ConfluenceClient  # noqa: PLC0415 - see module docstring
    from manicule.connectors.confluence import ConfluenceConnector  # noqa: PLC0415
    from manicule.connectors.credentials import credential_for  # noqa: PLC0415

    settings = context.config
    if not isinstance(settings, ConfluenceConfig):
        msg = (
            f"connector {CONNECTOR_NAME!r} was built with {type(settings).__name__} where it "
            f"declares ConfluenceConfig. Configuration reaching a factory is validated against "
            f"the model the component registered; a factory called outside the container has "
            f"to supply that model itself."
        )
        raise ConfigError(msg)
    resolved = resolve_credentials(settings)
    credential = credential_for(resolved)
    return ConfluenceConnector(resolved, ConfluenceClient(resolved, credential=credential))


def build_filesystem(context: BuildContext) -> Connector:
    """Construct the local-directory connector from validated configuration.

    Raises:
        ConfigError: The context carries configuration of some other type, or names no root.
            A connector with no root would discover nothing and report a clean run, which is
            the shape of a sync that looks like it worked.
    """
    from pathlib import Path  # noqa: PLC0415 - kept beside its only use

    from manicule.connectors.filesystem import FilesystemConnector  # noqa: PLC0415

    settings = context.config
    if not isinstance(settings, FilesystemConfig):
        msg = (
            f"connector {FILESYSTEM_CONNECTOR_NAME!r} was built with "
            f"{type(settings).__name__} where it declares FilesystemConfig."
        )
        raise ConfigError(msg)
    if not settings.root:
        msg = (
            f"connector {FILESYSTEM_CONNECTOR_NAME!r} has no root. Set "
            f'plugins.config."connector.filesystem".root to the directory to index, or use '
            f"`manicule index <path>` for a one-off."
        )
        raise ConfigError(msg)
    return FilesystemConnector(
        Path(settings.root),
        name=FILESYSTEM_CONNECTOR_NAME,
        include_hidden=settings.include_hidden,
        max_bytes=settings.max_bytes,
    )


class ConnectorsPlugin:
    """The plugin object the ``connectors`` entry point resolves to."""

    manifest = PluginManifest(
        name="connectors",
        version="0.1.0",
        core_version=">=0.1,<0.2",
        summary="Sources manicule ingests from. A local directory, and Confluence for v1.",
    )

    def register(self, registry: ComponentRegistry) -> None:
        registry.add(
            keys.CONNECTOR.named(CONNECTOR_NAME),
            build_confluence,
            config_model=ConfluenceConfig,
            summary="CQL watermark sync, cursor pagination, macro resolution, ID reconciliation.",
        )
        registry.add(
            keys.CONNECTOR.named(FILESYSTEM_CONNECTOR_NAME),
            build_filesystem,
            config_model=FilesystemConfig,
            summary="A local directory tree, walked in a stable order, reconciled by path.",
        )


PLUGIN = ConnectorsPlugin()

# Checked when this file is type-checked, so the plugin cannot drift out of conformance with
# the protocol every installation loads it through.
_plugin: Plugin = PLUGIN
