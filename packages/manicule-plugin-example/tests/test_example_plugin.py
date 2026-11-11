"""The example plugin is held to the same standard as anything else.

This file is also the answer to "how do I test my plugin": import the conformance suite
from :mod:`manicule.testing` and run it against your components.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from manicule_plugin_example import (
    CONNECTOR_NAME,
    MEDIA_TYPE,
    PLUGIN,
    DirectoryConnector,
    DirectoryConnectorConfig,
    DirectoryConnectorError,
    LineParser,
    LineParserConfig,
    PassthroughStage,
)

from manicule.config.settings import ConnectorSettings, Settings
from manicule.container import Container, keys
from manicule.core.anchors import LineAnchor, Unlocated
from manicule.core.content import RawDocument
from manicule.core.errors import ConfigError
from manicule.core.protocols import Connector, Middleware, Parser, RetrievalStage
from manicule.core.retrieval import Filter, Query
from manicule.core.sources import DocRef
from manicule.core.version import CORE_VERSION
from manicule.plugins import ComponentRegistry, check_core_version, discover
from manicule.testing import (
    assert_connector_contract,
    assert_parser_contract,
    assert_retrieval_stage_contract,
)


def raw(text: str = "alpha\n\nbeta\ngamma") -> RawDocument:
    return RawDocument(source_id="1", uri="example://1", media_type=MEDIA_TYPE, content=text)


def test_the_plugin_is_installed_and_discoverable() -> None:
    assert "example" in discover().names


def test_it_declares_a_core_version_this_core_satisfies() -> None:
    assert check_core_version(PLUGIN.manifest, CORE_VERSION) is None


def test_it_registers_what_it_says_it_does() -> None:
    registry = ComponentRegistry().bind("example")
    PLUGIN.register(registry)
    assert len(registry) == 4
    assert registry.record(keys.CONNECTOR.named(CONNECTOR_NAME)).config_model is (
        DirectoryConnectorConfig
    )
    assert registry.record(keys.PARSER.named("example")).config_model is LineParserConfig


async def test_the_parser_passes_the_conformance_suite() -> None:
    """Including the round-trip check: every anchor resolves to the text its block claims."""
    parser = LineParser(LineParserConfig())
    blocks = await assert_parser_contract(parser, raw())
    assert [block.text for block in blocks] == ["alpha", "beta", "gamma"]


async def test_anchors_point_at_the_lines_the_blocks_came_from() -> None:
    document = raw()
    parser = LineParser(LineParserConfig())
    blocks = [block async for block in parser.parse(document)]

    third = blocks[1]
    assert isinstance(third.anchor, LineAnchor)
    assert third.anchor.start == 3
    assert await parser.resolve(third.anchor, document) == "beta"


async def test_an_anchor_that_addresses_nowhere_resolves_to_nothing() -> None:
    parser = LineParser(LineParserConfig())
    assert await parser.resolve(Unlocated(reason="no location"), raw()) is None
    assert await parser.resolve(LineAnchor(start=999, end=999), raw()) is None


async def test_configuration_reaches_the_component() -> None:
    parser = LineParser(LineParserConfig(skip_blank=False))
    blocks = [block async for block in parser.parse(raw())]
    assert [block.text for block in blocks] == ["alpha", "", "beta", "gamma"]


async def test_the_control_stage_changes_nothing() -> None:
    query = Query(text="q", filter=Filter(workspace_ids=frozenset({"default"})))
    await assert_retrieval_stage_contract(PassthroughStage(), query, [])


def test_each_component_satisfies_its_protocol(settings: Settings) -> None:
    """Built through the container, exactly as an installation would build them."""
    registry = ComponentRegistry().bind("example")
    PLUGIN.register(registry)
    container = Container(settings, registry)

    assert isinstance(container.get(keys.PARSER.named("example")), Parser)
    assert isinstance(container.get(keys.MIDDLEWARE.named("trim")), Middleware)
    assert isinstance(container.get(keys.RETRIEVAL_STAGE.named("passthrough")), RetrievalStage)


# --- the connector ---------------------------------------------------------------------------


@pytest.fixture
def notes(tmp_path: Path) -> Path:
    """A directory of three notes, one in a subdirectory, and one file that is not a note."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta\n", encoding="utf-8")
    (tmp_path / "sub" / "c.txt").write_text("gamma\n", encoding="utf-8")
    (tmp_path / "ignored.md").write_text("not a note\n", encoding="utf-8")
    # Distinct, ordered times rather than whatever the clock gave three writes in one tick.
    for seconds, name in enumerate(["a.txt", "b.txt", "sub/c.txt"], start=1):
        stamp = seconds * 1_000_000_000
        os.utime(tmp_path / name, ns=(stamp, stamp))
    return tmp_path


def connector(root: Path) -> DirectoryConnector:
    return DirectoryConnector("notes", DirectoryConnectorConfig(root=root))


async def discovered_ids(source: DirectoryConnector, watermark: object = None) -> list[str]:
    return [doc.ref.source_id async for doc in source.discover(watermark)]  # pyright: ignore[reportArgumentType]


async def test_the_connector_passes_the_conformance_suite(notes: Path) -> None:
    """Including the two obligations that are easy to leave out: ``reconcile``, and a watermark
    that does not move for an enumeration that never finished."""
    await assert_connector_contract(connector(notes))


async def test_it_finds_notes_by_suffix_and_names_them_by_relative_path(notes: Path) -> None:
    assert await discovered_ids(connector(notes)) == ["a.txt", "b.txt", "sub/c.txt"]


async def test_a_sync_after_a_change_yields_what_changed(notes: Path) -> None:
    source = connector(notes)
    assert await discovered_ids(source) == ["a.txt", "b.txt", "sub/c.txt"]
    first = source.watermark
    assert first is not None

    newer = notes / "b.txt"
    stat = newer.stat()
    newer.write_text("beta, revised\n", encoding="utf-8")
    os.utime(newer, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000_000))

    # ``a.txt`` is older than the position and is not enumerated again. ``sub/c.txt`` is the
    # newest document of the first run, so it sits exactly on the position and is read again:
    # the deliberate overlap, which change detection discards by its unchanged version token.
    assert await discovered_ids(source, first) == ["b.txt", "sub/c.txt"]


async def test_a_changed_note_has_a_new_version_token(notes: Path) -> None:
    source = connector(notes)
    before = {doc.ref.source_id: doc.version_token async for doc in source.discover(None)}

    path = notes / "a.txt"
    stat = path.stat()
    path.write_text("alpha, revised\n", encoding="utf-8")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000_000))

    after = {doc.ref.source_id: doc.version_token async for doc in source.discover(None)}
    assert after["a.txt"] != before["a.txt"]
    assert after["b.txt"] == before["b.txt"]


async def test_a_deleted_note_is_reported_missing_by_reconcile_and_not_by_discover(
    notes: Path,
) -> None:
    """The reason ``reconcile`` is part of the protocol: discovery cannot see a deletion."""
    source = connector(notes)
    (notes / "b.txt").unlink()

    assert "b.txt" not in await discovered_ids(source)
    assert [source_id async for source_id in source.reconcile()] == ["a.txt", "sub/c.txt"]


async def test_the_watermark_is_not_offered_for_an_empty_source(tmp_path: Path) -> None:
    source = connector(tmp_path)
    assert await discovered_ids(source) == []
    assert source.watermark is None


async def test_fetch_returns_the_bytes_under_the_documents_media_type(notes: Path) -> None:
    source = connector(notes)
    (doc, *_rest) = [found async for found in source.discover(None)]

    raw = await source.fetch(doc.ref)

    assert raw.source_id == "a.txt"
    assert raw.media_type == MEDIA_TYPE
    assert raw.as_text() == "alpha\n"


async def test_fetch_refuses_an_id_that_leaves_the_directory(notes: Path, tmp_path: Path) -> None:
    secret = tmp_path.parent / "secret.txt"
    secret.write_text("outside", encoding="utf-8")
    source = connector(notes)

    with pytest.raises(DirectoryConnectorError, match="outside"):
        await source.fetch(DocRef(source_id=f"../{secret.name}", uri="file:///x"))


async def test_the_container_builds_a_configured_instance_under_the_name_the_user_gave(
    settings: Settings, notes: Path
) -> None:
    """The source is ``handbook``; ``example-directory`` is only the implementation it asked for."""
    registry = ComponentRegistry().bind("example")
    PLUGIN.register(registry)
    configured = settings.model_copy(
        update={
            "connectors": {
                "handbook": ConnectorSettings(type=CONNECTOR_NAME, options={"root": str(notes)})
            }
        }
    )
    container = Container(configured, registry)

    built = await container.connector("handbook")

    assert isinstance(built, Connector)
    assert built.name == "handbook"
    assert await assert_connector_contract(built) is None


async def test_a_misspelled_option_is_a_startup_error_and_not_a_setting_that_does_nothing(
    settings: Settings, notes: Path
) -> None:
    registry = ComponentRegistry().bind("example")
    PLUGIN.register(registry)
    configured = settings.model_copy(
        update={
            "connectors": {
                "handbook": ConnectorSettings(
                    type=CONNECTOR_NAME, options={"root": str(notes), "sufix": ".md"}
                )
            }
        }
    )

    with pytest.raises(ConfigError, match="sufix"):
        await Container(configured, registry).connector("handbook")
