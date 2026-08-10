"""The example plugin is held to the same standard as anything else.

This file is also the answer to "how do I test my plugin": import the conformance suite
from :mod:`manicule.testing` and run it against your components.
"""

from __future__ import annotations

from manicule_plugin_example import (
    MEDIA_TYPE,
    PLUGIN,
    LineParser,
    LineParserConfig,
    PassthroughStage,
)

from manicule.config.settings import Settings
from manicule.container import Container, keys
from manicule.core.anchors import LineAnchor, Unlocated
from manicule.core.content import RawDocument
from manicule.core.protocols import Middleware, Parser, RetrievalStage
from manicule.core.retrieval import Query
from manicule.core.version import CORE_VERSION
from manicule.plugins import ComponentRegistry, check_core_version, discover
from manicule.testing import assert_parser_contract, assert_retrieval_stage_contract


def raw(text: str = "alpha\n\nbeta\ngamma") -> RawDocument:
    return RawDocument(source_id="1", uri="example://1", media_type=MEDIA_TYPE, content=text)


def test_the_plugin_is_installed_and_discoverable() -> None:
    assert "example" in discover().names


def test_it_declares_a_core_version_this_core_satisfies() -> None:
    assert check_core_version(PLUGIN.manifest, CORE_VERSION) is None


def test_it_registers_what_it_says_it_does() -> None:
    registry = ComponentRegistry().bind("example")
    PLUGIN.register(registry)
    assert len(registry) == 3
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
    await assert_retrieval_stage_contract(PassthroughStage(), Query(text="q"), [])


def test_each_component_satisfies_its_protocol(settings: Settings) -> None:
    """Built through the container, exactly as an installation would build them."""
    registry = ComponentRegistry().bind("example")
    PLUGIN.register(registry)
    container = Container(settings, registry)

    assert isinstance(container.get(keys.PARSER.named("example")), Parser)
    assert isinstance(container.get(keys.MIDDLEWARE.named("trim")), Middleware)
    assert isinstance(container.get(keys.RETRIEVAL_STAGE.named("passthrough")), RetrievalStage)
