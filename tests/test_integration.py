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
from manicule.core.content import Document, DocumentStatus, RawDocument
from manicule.core.errors import ConfigError, PolicyError, UnknownComponentError
from manicule.core.ids import content_hash, document_id
from manicule.core.protocols import Chunker, Parser
from manicule.plugins import (
    BuildContext,
    ComponentKey,
    ComponentKind,
    ComponentRegistry,
    discover,
)
from manicule.testing import assert_parser_contract, closing
from tests.fakes import HashEmbedder, make_raw

EMBEDDER_NAME = "local"
"""The stand-in embedder these tests select.

Registered under a name of its own rather than shadowing ``mlx``, because the built-in
embedding plugin now provides ``mlx`` for real — and a stand-in registered over it would both
clash with it and hide it, exactly as a stand-in chunker would hide ``structural``. Selecting
this one instead keeps the integration tests hermetic: building the real embedder would
download a model, which is not something a test suite should do.

``local`` rather than an invented name, because provider names are also what the credential
policy reads: anything outside ``LOCAL_PROVIDERS`` is required to carry an API key, and a
stand-in that tripped that check would be testing the wrong thing.
"""


def _install_the_rest(registry: ComponentRegistry) -> None:
    """Stand in for the components the other core tickets will provide as plugins.

    The chunker and the embedder are deliberately absent: the built-in parsing and embedding
    plugins register ``structural``, ``mlx`` and ``onnx`` through the same entry point every
    other plugin uses. What is left is the components no plugin provides yet.
    """
    registry.add(keys.EMBEDDER.named(EMBEDDER_NAME), lambda _: HashEmbedder())
    registry.add(keys.GENERATOR.named("ollama"), lambda _: object())
    registry.add(keys.VECTOR_STORE.named("lancedb"), lambda _: object())
    registry.add(keys.DOC_STORE.named("sqlite"), lambda _: object())


def test_a_configuration_naming_nothing_installed_refuses_to_start() -> None:
    """The whole failure at once, before a single component is constructed.

    The embedder is no longer among the missing, and that is the point of the second
    assertion: ``embedding.provider`` defaults to ``mlx``, the built-in embedding plugin
    registers it through the public entry point, and so a default installation validates it
    rather than reporting it absent. If that assertion ever starts failing, the plugin has
    stopped being discovered — which nothing else in the suite would notice, because every
    other test registers its own embedder.
    """
    with pytest.raises(PolicyError) as caught:
        build_container(Settings())
    message = str(caught.value)
    assert "embedding.provider" not in message
    assert "storage.db" in message
    assert "llm.provider" in message


async def test_the_example_plugin_works_through_the_container(
    manicule_environment: Path,
) -> None:
    """Discovered from installed metadata, configured, built, set up, used, torn down."""
    del manicule_environment

    found = discover()
    _install_the_rest(found.registry.bind("test-harness"))

    settings = Settings(
        embedding={"provider": EMBEDDER_NAME},  # pyright: ignore[reportArgumentType]
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

    settings = Settings(
        embedding={"provider": EMBEDDER_NAME},  # pyright: ignore[reportArgumentType]
        rag={"pipeline": ("passthrough",)},  # pyright: ignore[reportArgumentType]
    )
    container = build_container(settings, discovery=found)

    async with container:
        parser = await container.aget(keys.PARSER.named("example"))
        assert isinstance(parser, Parser)
    assert container.describe() == []


MARKDOWN = """\
# Release notes

The scheduler now retries a failed job three times before giving up.

## Known issues

Large exports still time out on the reporting endpoint.
"""


async def test_a_document_parses_and_chunks_through_the_container(
    manicule_environment: Path,
) -> None:
    """A built-in parser and the built-in chunker, reached only through discovery.

    The point of the route rather than the result: nothing here imports a parser module, names
    a class, or constructs a chunker. A document is routed by media type to whatever the
    registry says owns it, and chunked by whatever ``rag.chunker`` names. If the built-in
    parsers ever stopped registering through the public entry point — an internal shortcut, a
    missing entry-point declaration in ``pyproject.toml`` — every one of these lookups would
    fail, and nothing else in the suite would notice, because every other test constructs its
    parser directly.
    """
    del manicule_environment

    found = discover()
    _install_the_rest(found.registry.bind("test-harness"))
    container = build_container(
        Settings(
            embedding={"provider": EMBEDDER_NAME},  # pyright: ignore[reportArgumentType]
            rag={"pipeline": ("passthrough",), "chunker": "structural"},  # pyright: ignore[reportArgumentType]
        ),
        discovery=found,
    )

    async with container:
        parsers = await container.parser_chain("text/markdown")
        assert [type(parser).__name__ for parser in parsers] == ["MarkdownParser"]
        parser = parsers[0]

        raw = RawDocument(
            source_id="release-notes.md",
            uri="release-notes.md",
            media_type="text/markdown",
            content=MARKDOWN,
            metadata={"title": "Release notes"},
        )
        blocks = await assert_parser_contract(parser, raw)
        assert blocks[0].text == "# Release notes"

        chunker = await container.aget(keys.CHUNKER.named("structural"))
        assert isinstance(chunker, Chunker)
        chunks = chunker.chunk(_document_for(raw), blocks)

    assert chunks, "a document with two sections produced no chunks"
    assert [chunk.position for chunk in chunks] == list(range(len(chunks)))
    assert any("scheduler" in chunk.text for chunk in chunks)


async def test_an_anchor_from_the_container_still_resolves_to_its_own_text(
    manicule_environment: Path,
) -> None:
    """The obligation that survives the whole path, not just the parser in isolation.

    A citation is produced by a parser reached through discovery and resolved by the same
    instance the container handed out. Checking it here as well as in the parser's own suite
    is what catches a container that hands back a *differently configured* parser — one built
    from settings that were validated against a config model nobody registered, whose anchors
    are measured against a document the resolver never sees the same way.
    """
    del manicule_environment

    found = discover()
    _install_the_rest(found.registry.bind("test-harness"))
    settings = Settings(
        embedding={"provider": EMBEDDER_NAME},  # pyright: ignore[reportArgumentType]
        rag={"pipeline": ("passthrough",)},  # pyright: ignore[reportArgumentType]
    )
    container = build_container(settings, discovery=found)

    async with container:
        parser = (await container.parser_chain("text/markdown"))[0]
        raw = RawDocument(
            source_id="release-notes.md",
            uri="release-notes.md",
            media_type="text/markdown",
            content=MARKDOWN,
        )
        async with closing(parser.parse(raw)) as stream:
            blocks = [block async for block in stream]

        located = [block for block in blocks if block.anchor.kind != "unlocated"]
        assert located, "every block gave up its location, so the check below proves nothing"
        for block in located:
            resolved = await parser.resolve(block.anchor, raw)
            assert resolved is not None
            assert block.text in resolved


def test_building_a_parser_with_the_wrong_configuration_model_is_refused() -> None:
    """The guard on a factory called outside the container.

    The container validates configuration against the model a component registered before it
    calls the factory, so a mistyped config cannot arrive by that route. A factory called any
    other way could hand a parser someone else's settings — and substituting defaults instead,
    which is what this used to do, builds a parser whose configuration appears to be in force
    and is not. That is the failure validation exists to prevent, so it is an error.
    """
    from manicule.parsers.config import PdfConfig  # noqa: PLC0415 - a parsing extra

    # Through the registered factory, which is the object the container calls — so this
    # exercises the real route rather than a private function that happens to be behind it.
    factory = discover().registry.record(keys.PARSER.named("markdown")).factory
    context = BuildContext(
        settings=Settings(),
        config=PdfConfig(),
        data_dir=Path(),
        cache_dir=Path(),
        components=_NoComponents(),
    )

    with pytest.raises(ConfigError) as caught:
        factory(context)

    assert "MarkdownConfig" in str(caught.value)
    assert "PdfConfig" in str(caught.value)


class _NoComponents:
    """A resolver that provides nothing, for a factory that asks it for nothing."""

    def get[T](self, key: ComponentKey[T]) -> T:
        msg = f"nothing provides {key}"
        raise UnknownComponentError(msg)


def _document_for(raw: RawDocument) -> Document:
    """The stored record the chunker writes chunk ids against."""
    return Document(
        id=document_id("integration", "fixtures", raw.source_id),
        source="fixtures",
        source_id=raw.source_id,
        uri=raw.uri,
        title="Release notes",
        content_hash=content_hash(raw.as_bytes()),
        media_type=raw.media_type,
        status=DocumentStatus.PARSED,
    )


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
