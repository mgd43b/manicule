"""The mbox source through registration: what it needs, what it refuses, what it builds.

``docs/connectors/mbox.md`` §1, §6, §7.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from manicule.config.settings import ConnectorSettings, Settings
from manicule.connectors.config import MboxConfig
from manicule.connectors.mbox import MboxConnector
from manicule.connectors.plugin import PLUGIN, build_mbox
from manicule.core.errors import ConfigError
from manicule.core.mail_ids import message_id_from_link, message_id_of, normalize_message_id
from manicule.plugins.manifest import ComponentKind
from manicule.plugins.registry import BuildContext, ComponentRegistry


class _Components:
    def __init__(self) -> None:
        self.asked: list[object] = []

    def get(self, key: Any) -> object:
        self.asked.append(key)
        return object()


def context(
    settings: Settings, tmp_path: Path, config: MboxConfig, instance: str = "old"
) -> BuildContext:
    return BuildContext(
        settings=settings,
        config=config,
        data_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        components=_Components(),  # type: ignore[arg-type]
        instance=instance,
    )


def test_the_type_is_registered_and_sensitive() -> None:
    registry = ComponentRegistry()
    PLUGIN.register(registry.bind("connectors"))
    assert "mbox" in registry.sensitive_names(ComponentKind.CONNECTOR)


def test_a_source_with_no_root_is_refused_naming_where_to_set_it(
    settings: Settings, tmp_path: Path
) -> None:
    with pytest.raises(ConfigError, match=r"connectors\.old\.options"):
        build_mbox(context(settings, tmp_path, MboxConfig()))


def test_any_other_option_is_refused() -> None:
    for option in ("since", "folders", "message_id_identity"):
        with pytest.raises(ValidationError):
            MboxConfig.model_validate({"root": "/srv/mail", option: True})


def test_an_unsafe_exclude_pattern_is_refused_at_load() -> None:
    with pytest.raises(ValidationError):
        MboxConfig(root="/srv/mail", exclude=("/etc/passwd",))
    with pytest.raises(ValidationError):
        MboxConfig(root="/srv/mail", exclude=("../elsewhere",))


def test_the_other_mail_sources_are_the_other_gmail_and_mbox_instances(
    settings: Settings, tmp_path: Path
) -> None:
    configured = settings.model_copy(
        update={
            "connectors": {
                "old": ConnectorSettings(type="mbox"),
                "other-archive": ConnectorSettings(type="mbox", enabled=False),
                "mail": ConnectorSettings(type="gmail"),
                "wiki": ConnectorSettings(type="confluence"),
                "files": ConnectorSettings(type="filesystem"),
            }
        }
    )
    connector = build_mbox(context(configured, tmp_path, MboxConfig(root=str(tmp_path))))
    assert isinstance(connector, MboxConnector)
    assert connector.name == "old"
    assert connector.other_sources == ("mail", "other-archive")


def test_every_option_that_changes_what_is_yielded_changes_the_scope(tmp_path: Path) -> None:
    def fingerprint(**options: object) -> str:
        return MboxConnector(
            MboxConfig.model_validate({"root": str(tmp_path), **options})
        ).scope_fingerprint

    base = fingerprint()
    assert fingerprint() == base
    assert fingerprint(include_hidden=True) != base
    assert fingerprint(exclude=["Trash"]) != base
    assert fingerprint(skip_known_message_ids=False) != base
    elsewhere = tmp_path / "other"
    elsewhere.mkdir()
    assert MboxConnector(MboxConfig(root=str(elsewhere))).scope_fingerprint != base


def test_the_scope_names_no_path(tmp_path: Path) -> None:
    connector = MboxConnector(MboxConfig(root=str(tmp_path)))
    assert str(tmp_path) not in connector.source_scope
    assert str(tmp_path) not in connector.reconciliation_scope


# --- the Message-ID a mail document records ----------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("<a.b@example.org>", "a.b@example.org"),
        ("  <a.b@example.org>  ", "a.b@example.org"),
        ("a.b@example.org", "a.b@example.org"),
        ("junk <a.b@example.org> trailing", "a.b@example.org"),
        ("", None),
        ("<>", None),
        ("<a\x00b@example.org>", None),
        ("<" + "x" * 600 + ">", None),
    ],
)
def test_a_message_id_is_normalized(value: str, expected: str | None) -> None:
    assert normalize_message_id(value) == expected


def test_a_gmail_link_carries_the_id_and_a_fallback_link_carries_none() -> None:
    link = "https://mail.google.com/mail/?authuser=a%40example.org#search/rfc822msgid:x%2By%40example.org"
    assert message_id_from_link(link) == "x+y@example.org"
    assert message_id_from_link("https://mail.google.com/mail/?authuser=a#all/18c") is None
    assert message_id_of({}, link) == "x+y@example.org"
    assert message_id_of({"mail_message_id": "<m@example.org>"}, link) == "m@example.org"
