"""The example plugin is held to the same standard as anything else.

This file is also the answer to "how do I test my plugin": import the conformance suite
from :mod:`manicule.testing` and run it against your components.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import override

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
    root = tmp_path / "notes"
    (root / "sub").mkdir(parents=True)
    (root / "a.txt").write_text("alpha\n", encoding="utf-8")
    (root / "b.txt").write_text("beta\n", encoding="utf-8")
    (root / "sub" / "c.txt").write_text("gamma\n", encoding="utf-8")
    (root / "ignored.md").write_text("not a note\n", encoding="utf-8")
    # Distinct, ordered times rather than whatever the clock gave three writes in one tick.
    for seconds, name in enumerate(["a.txt", "b.txt", "sub/c.txt"], start=1):
        stamp = seconds * 1_000_000_000
        os.utime(root / name, ns=(stamp, stamp))
    return root


def connector(root: Path, **options: str) -> DirectoryConnector:
    return DirectoryConnector("notes", DirectoryConnectorConfig(root=root, **options))


async def discovered(source: DirectoryConnector) -> dict[str, str | None]:
    """Source id to version token, for one complete walk."""
    return {doc.ref.source_id: doc.version_token async for doc in source.discover(None)}


def rewrite(path: Path, text: str) -> None:
    """Change a file and give it a modification time that cannot collide with the old one."""
    stat = path.stat()
    path.write_text(text, encoding="utf-8")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000_000))


async def test_the_connector_passes_the_conformance_suite(notes: Path) -> None:
    """Including the two obligations that are easy to leave out: ``reconcile``, and a watermark
    that does not move for an enumeration that never finished."""
    await assert_connector_contract(connector(notes))


async def test_it_finds_notes_by_suffix_and_names_them_by_relative_path(notes: Path) -> None:
    assert list(await discovered(connector(notes))) == ["a.txt", "b.txt", "sub/c.txt"]


async def test_the_suffix_is_matched_literally_and_not_as_a_glob(notes: Path) -> None:
    (notes / "odd[1].txt").write_text("x", encoding="utf-8")

    assert list(await discovered(connector(notes, suffix="[1].txt"))) == ["odd[1].txt"]


async def test_a_changed_note_has_a_new_version_token_and_the_others_do_not(notes: Path) -> None:
    source = connector(notes)
    before = await discovered(source)

    rewrite(notes / "a.txt", "alpha, revised\n")

    after = await discovered(source)
    assert after["a.txt"] != before["a.txt"]
    assert after["b.txt"] == before["b.txt"]
    assert after["sub/c.txt"] == before["sub/c.txt"]


async def test_a_note_that_arrives_with_an_old_modification_time_is_still_found(
    notes: Path,
) -> None:
    """The reason there is no cursor: ``cp -p`` and a restore bring old timestamps with them."""
    source = connector(notes)
    await discovered(source)

    restored = notes / "restored.txt"
    restored.write_text("from a backup\n", encoding="utf-8")
    os.utime(restored, ns=(1, 1))

    assert "restored.txt" in await discovered(source)


async def test_the_same_file_under_two_roots_is_two_different_revisions(
    notes: Path, tmp_path: Path
) -> None:
    """Without the root in the token, repointing a source keeps the old directory's content.

    Same relative name, same size, same modification time — and different bytes.
    """
    other = tmp_path / "other"
    other.mkdir()
    (other / "a.txt").write_text("ALPHA\n", encoding="utf-8")
    os.utime(other / "a.txt", ns=(1_000_000_000, 1_000_000_000))

    assert (await discovered(connector(notes)))["a.txt"] != (await discovered(connector(other)))[
        "a.txt"
    ]


async def test_the_watermark_is_set_when_a_walk_completes(notes: Path) -> None:
    source = connector(notes)
    assert source.watermark is None

    await discovered(source)

    assert source.watermark is not None


async def test_fetch_returns_the_bytes_and_the_revision_it_read(notes: Path) -> None:
    """Durable acquisition refuses a body without proof of its revision."""
    source = connector(notes)
    docs = [doc async for doc in source.discover(None)]

    for doc in docs:
        raw = await source.fetch(doc.ref)
        assert raw.media_type == MEDIA_TYPE
        assert raw.metadata["version_token"] == doc.version_token

    assert (await source.fetch(docs[0].ref)).as_text() == "alpha\n"


async def test_fetch_refuses_a_file_that_changed_while_it_was_being_read(notes: Path) -> None:
    """Otherwise the new bytes would be stored under the old revision's token."""

    class Racing(DirectoryConnector):
        @override
        async def _read(self, path: Path) -> bytes:
            content = await super()._read(path)
            rewrite(path, "changed underneath the read\n")
            return content

    source = Racing("notes", DirectoryConnectorConfig(root=notes))
    (doc, *_rest) = [found async for found in source.discover(None)]

    with pytest.raises(DirectoryConnectorError, match="changed while"):
        await source.fetch(doc.ref)


async def test_fetch_refuses_an_id_that_leaves_the_directory(notes: Path) -> None:
    secret = notes.parent / "secret.txt"
    secret.write_text("outside", encoding="utf-8")

    with pytest.raises(DirectoryConnectorError, match="outside"):
        await connector(notes).fetch(DocRef(source_id=f"../{secret.name}", uri="file:///x"))


async def test_a_deleted_note_is_reported_missing_by_reconcile_and_not_by_discover(
    notes: Path,
) -> None:
    """The reason ``reconcile`` is part of the protocol: discovery cannot see a deletion."""
    source = connector(notes)
    (notes / "b.txt").unlink()

    assert "b.txt" not in await discovered(source)
    assert [source_id async for source_id in source.reconcile()] == ["a.txt", "sub/c.txt"]


async def test_a_symlink_out_of_the_directory_is_not_a_document(notes: Path) -> None:
    """Listed and unfetchable would fail one document on every sync, forever."""
    outside = notes.parent / "outside.txt"
    outside.write_text("elsewhere", encoding="utf-8")
    (notes / "link.txt").symlink_to(outside)
    source = connector(notes)

    assert "link.txt" not in await discovered(source)
    assert "link.txt" not in [source_id async for source_id in source.reconcile()]
    await assert_connector_contract(source)


async def test_a_dangling_symlink_is_a_vanished_file_and_not_a_failure(notes: Path) -> None:
    (notes / "gone.txt").symlink_to(notes / "nowhere.txt")

    assert "gone.txt" not in await discovered(connector(notes))


async def reconciled(source: DirectoryConnector) -> list[str]:
    return [source_id async for source_id in source.reconcile()]


@pytest.mark.parametrize("enumerate_", [discovered, reconciled])
async def test_a_missing_root_is_an_error_and_not_an_empty_source(
    tmp_path: Path, enumerate_: Callable[[DirectoryConnector], Awaitable[object]]
) -> None:
    """An empty inventory is what a source whose files were all deleted looks like."""
    with pytest.raises(DirectoryConnectorError, match="not a directory"):
        await enumerate_(connector(tmp_path / "unmounted"))


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read a directory with no permissions")
async def test_a_directory_that_cannot_be_listed_is_an_error_and_not_a_deletion(
    notes: Path,
) -> None:
    """``rglob`` would skip it, and ``reconcile`` would then report its notes deleted."""
    (notes / "sub").chmod(0o000)
    try:
        with pytest.raises(DirectoryConnectorError, match="cannot list"):
            await reconciled(connector(notes))
    finally:
        (notes / "sub").chmod(0o755)


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
    await assert_connector_contract(built)


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
