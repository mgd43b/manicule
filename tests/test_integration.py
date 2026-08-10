"""End to end through the public path: discovery, wiring, lifecycle, use.

Everything here goes through the same route a real installation takes — entry points found
in the environment, configuration validated against them, components built by the container.
Nothing is stubbed, and no private attribute is touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from manicule_plugin_example import MEDIA_TYPE

from manicule.config.settings import Settings
from manicule.container import build_container, keys
from manicule.core.errors import PolicyError
from manicule.core.protocols import Parser
from manicule.plugins import ComponentKind, ComponentRegistry, discover
from manicule.testing import assert_parser_contract
from tests.fakes import BlockChunker, HashEmbedder, make_raw


def _install_the_rest(registry: ComponentRegistry) -> None:
    """Stand in for the components the other core tickets will provide as plugins."""
    registry.add(keys.EMBEDDER.named("mlx"), lambda _: HashEmbedder())
    registry.add(keys.CHUNKER.named("structural"), lambda _: BlockChunker())
    registry.add(keys.GENERATOR.named("ollama"), lambda _: object())
    registry.add(keys.VECTOR_STORE.named("lancedb"), lambda _: object())
    registry.add(keys.DOC_STORE.named("sqlite"), lambda _: object())


def test_a_configuration_naming_nothing_installed_refuses_to_start() -> None:
    """The whole failure at once, before a single component is constructed."""
    with pytest.raises(PolicyError) as caught:
        build_container(Settings())
    message = str(caught.value)
    assert "embedding.provider" in message
    assert "storage.db" in message


async def test_the_example_plugin_works_through_the_container(
    manicule_environment: Path,
) -> None:
    """Discovered from installed metadata, configured, built, set up, used, torn down."""
    del manicule_environment

    found = discover()
    _install_the_rest(found.registry.bind("test-harness"))

    settings = Settings(
        rag={"pipeline": ("passthrough",), "chunker": "structural"},  # pyright: ignore[reportArgumentType]
        plugins={"middleware": ("trim",)},  # pyright: ignore[reportArgumentType]
    )
    container = build_container(settings, discovery=found)

    async with container:
        parsers = await container.parser_chain(MEDIA_TYPE)
        assert [type(p).__name__ for p in parsers] == ["LineParser"]

        blocks = await assert_parser_contract(parsers[0], make_raw("alpha\nbeta"))
        assert [block.text for block in blocks] == ["alpha", "beta"]

        assert [m.name for m in await container.middleware()] == ["trim"]
        assert [s.name for s in await container.retrieval_pipeline()] == ["passthrough"]

        report = await container.health()
        assert report.ok, report.detail
        assert any(metric.name == "documents_parsed" for metric in container.metrics())


async def test_a_component_is_set_up_before_it_is_used(manicule_environment: Path) -> None:
    del manicule_environment
    found = discover()
    _install_the_rest(found.registry.bind("test-harness"))

    settings = Settings(rag={"pipeline": ("passthrough",)})  # pyright: ignore[reportArgumentType]
    container = build_container(settings, discovery=found)

    async with container:
        parser = await container.aget(keys.PARSER.named("example"))
        assert isinstance(parser, Parser)
    assert container.describe() == []


def test_per_component_configuration_reaches_the_component(
    manicule_environment: Path,
) -> None:
    """Through the config file, exactly as a user would set it."""
    (manicule_environment / "manicule.toml").write_text(
        '[plugins.config."parser.example"]\nskip_blank = false\n'
    )
    settings = Settings()
    assert discover().registry.has(ComponentKind.PARSER, "example")
    assert settings.component_config("parser", "example") == {"skip_blank": False}
