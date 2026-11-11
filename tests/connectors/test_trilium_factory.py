"""The Trilium source through registration and configuration: what it needs and what it refuses.

``docs/connectors/trilium.md`` §1, §2, §5, §7.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from manicule.config.settings import Settings
from manicule.connectors.config import TriliumConfig, resolve_trilium_token
from manicule.connectors.plugin import PLUGIN, build_trilium
from manicule.connectors.trilium import TriliumConnector
from manicule.core.errors import ConfigError
from manicule.plugins.manifest import ComponentKind
from manicule.plugins.registry import BuildContext, ComponentRegistry

BASE = "https://notes.example.org"


class _Components:
    def get(self, key: Any) -> object:
        return object()


def context(
    settings: Settings, tmp_path: Path, config: object, instance: str = "notes"
) -> BuildContext:
    return BuildContext(
        settings=settings,
        config=config,  # type: ignore[arg-type]
        data_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        components=_Components(),  # type: ignore[arg-type]
        instance=instance,
    )


def test_the_type_is_registered_and_sensitive() -> None:
    registry = ComponentRegistry()
    PLUGIN.register(registry.bind("connectors"))

    assert "trilium" in registry.names(ComponentKind.CONNECTOR)
    assert "trilium" in registry.sensitive_names(ComponentKind.CONNECTOR)


def test_the_factory_builds_a_connector_named_for_its_instance(
    settings: Settings, tmp_path: Path
) -> None:
    config = TriliumConfig(base_url=BASE, token=SecretStr("t"))

    connector = build_trilium(context(settings, tmp_path, config))

    assert isinstance(connector, TriliumConnector)
    assert connector.name == "notes"


def test_the_fetch_cap_is_the_installation_s_not_a_setting_of_its_own(
    settings: Settings, tmp_path: Path
) -> None:
    config = TriliumConfig(base_url=BASE, token=SecretStr("t"))

    connector = build_trilium(context(settings, tmp_path, config))

    assert isinstance(connector, TriliumConnector)
    assert connector.max_note_bytes == settings.ingest.max_fetch_bytes
    assert "max_note_bytes" not in TriliumConfig.model_fields


def test_a_missing_token_is_refused_naming_where_to_set_it(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TRILIUM_ETAPI_TOKEN", raising=False)

    with pytest.raises(ConfigError, match=r"connectors\.notes\.options.*TRILIUM_ETAPI_TOKEN"):
        build_trilium(context(settings, tmp_path, TriliumConfig(base_url=BASE)))


def test_a_token_in_the_environment_is_used_when_none_is_configured() -> None:
    config = TriliumConfig(base_url=BASE, token_env="MY_TRILIUM")  # noqa: S106 - a variable name

    resolved = resolve_trilium_token(config, {"MY_TRILIUM": "from-env"})

    assert resolved.token is not None
    assert resolved.token.get_secret_value() == "from-env"


def test_a_configured_token_is_never_overwritten_by_the_environment() -> None:
    config = TriliumConfig(base_url=BASE, token=SecretStr("explicit"))

    resolved = resolve_trilium_token(config, {"TRILIUM_ETAPI_TOKEN": "env"})

    assert resolved.token is not None
    assert resolved.token.get_secret_value() == "explicit"


def test_a_blank_environment_variable_counts_as_unset() -> None:
    with pytest.raises(ConfigError):
        resolve_trilium_token(TriliumConfig(base_url=BASE), {"TRILIUM_ETAPI_TOKEN": ""})


def test_an_empty_token_is_refused_at_load() -> None:
    with pytest.raises(ValidationError, match="empty string"):
        TriliumConfig(base_url=BASE, token=SecretStr(""))


def test_the_token_is_not_printed_with_the_configuration() -> None:
    config = TriliumConfig(base_url=BASE, token=SecretStr("hunter2"))

    assert "hunter2" not in repr(config)
    assert "hunter2" not in config.model_dump_json()


@pytest.mark.parametrize("option", ["since", "types", "labels", "max_note_bytes", "include_files"])
def test_the_settings_that_are_deliberately_absent_are_refused(option: str) -> None:
    with pytest.raises(ValidationError):
        TriliumConfig.model_validate({"base_url": BASE, option: True})


@pytest.mark.parametrize(
    "url",
    [
        "notes.example.org",
        "ftp://example.org",
        "https://u:p@example.org",
        "https://example.org/?q=1",
    ],
)
def test_a_base_url_that_is_not_a_plain_http_origin_is_refused(url: str) -> None:
    with pytest.raises(ValidationError):
        TriliumConfig(base_url=url)


def test_labels_are_written_without_the_hash_and_blank_names_are_refused() -> None:
    assert TriliumConfig(base_url=BASE, exclude_labels=("#private", "x")).exclude_labels == (
        "private",
        "x",
    )
    with pytest.raises(ValidationError):
        TriliumConfig(base_url=BASE, exclude_labels=("#",))
    with pytest.raises(ValidationError):
        TriliumConfig(base_url=BASE, root_note_ids=(" ",))


def test_every_option_that_changes_what_is_yielded_changes_the_scope() -> None:
    def fingerprint(**options: object) -> str:
        return TriliumConfig.model_validate({"base_url": BASE, **options}).scope_fingerprint("s")

    base = fingerprint()
    assert fingerprint() == base
    assert fingerprint(root_note_ids=["abc"]) != base
    assert fingerprint(exclude_labels=["other"]) != base
    assert fingerprint(include_archived=True) != base
    assert fingerprint(token=SecretStr("rotated")) == base, "rotating a token is not a new source"


def test_the_scope_names_no_address_in_the_clear(settings: Settings, tmp_path: Path) -> None:
    config = TriliumConfig(base_url=BASE, token=SecretStr("t"))
    connector = build_trilium(context(settings, tmp_path, config))

    assert isinstance(connector, TriliumConnector)
    assert "example.org" not in connector.source_scope
    assert "example.org" not in connector.reconciliation_scope


def test_the_wrong_configuration_type_is_refused(settings: Settings, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="TriliumConfig"):
        build_trilium(context(settings, tmp_path, object()))
