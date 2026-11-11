"""``git-docs`` through the plugin: what a misconfigured source says at startup, and what builds.

The factory reads the token before it constructs anything, so a source that names a variable
nobody set is a startup refusal naming the setting and the variable, rather than a ``401`` from the
first fetch of a sync nobody is watching.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from pydantic import BaseModel

from manicule.connectors.config import GitDocsConfig
from manicule.connectors.git_docs import GitDocsConnector
from manicule.connectors.plugin import PLUGIN, build_git_docs
from manicule.core.errors import ConfigError
from manicule.plugins.registry import BuildContext, ComponentKind, ComponentRegistry

REPOSITORY = "https://git.example.org/team/docs"


def context(config: BaseModel, tmp_path: Path, instance: str = "planning") -> BuildContext:
    return BuildContext(
        settings=None,  # pyright: ignore[reportArgumentType] - unused by this factory
        config=config,
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        components=None,  # pyright: ignore[reportArgumentType] - unused by this factory
        instance=instance,
    )


def test_a_public_repository_builds_without_any_credential(tmp_path: Path) -> None:
    config = GitDocsConfig.model_validate({"repository": REPOSITORY})

    connector = build_git_docs(context(config, tmp_path))

    assert isinstance(connector, GitDocsConnector)
    assert connector.name == "planning"


def test_a_token_is_read_from_the_variable_the_source_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PLANNING_TOKEN", "  synthetic-token  ")
    config = GitDocsConfig.model_validate({"repository": REPOSITORY, "token_env": "PLANNING_TOKEN"})

    connector = build_git_docs(context(config, tmp_path))

    assert isinstance(connector, GitDocsConnector)
    assert connector.mirror is not None
    environment = connector.mirror.environment()
    entries = {
        environment[f"GIT_CONFIG_KEY_{i}"]: environment[f"GIT_CONFIG_VALUE_{i}"]
        for i in range(int(environment["GIT_CONFIG_COUNT"]))
    }
    sent = base64.b64decode(
        entries["http.https://git.example.org/.extraHeader"].removeprefix("Authorization: Basic ")
    )
    assert sent == b"x-access-token:synthetic-token", "read, stripped, and sent as Basic"
    assert connector.mirror.directory.parent == tmp_path / "cache" / "git-mirrors"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_a_variable_that_is_unset_or_blank_is_refused_naming_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    """A deployment that mounts an empty secret has not provided one."""
    if value is None:
        monkeypatch.delenv("PLANNING_TOKEN", raising=False)
    else:
        monkeypatch.setenv("PLANNING_TOKEN", value)
    config = GitDocsConfig.model_validate({"repository": REPOSITORY, "token_env": "PLANNING_TOKEN"})

    with pytest.raises(ConfigError) as caught:
        build_git_docs(context(config, tmp_path))

    assert "[connectors.planning.options]" in str(caught.value)
    assert "$PLANNING_TOKEN" in str(caught.value)


def test_the_connector_is_registered_with_its_configuration_model() -> None:
    registry = ComponentRegistry()
    PLUGIN.register(registry.bind("connectors"))

    assert "git-docs" in registry.names(ComponentKind.CONNECTOR)


def test_a_factory_handed_the_wrong_configuration_refuses_it(tmp_path: Path) -> None:
    class Other(BaseModel):
        pass

    with pytest.raises(ConfigError, match="GitDocsConfig"):
        build_git_docs(context(Other(), tmp_path))
