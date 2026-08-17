"""The container: resolution, configuration, lifecycle and the startup gate."""

from __future__ import annotations

from collections.abc import Callable
from typing import override

import pytest
from pydantic import BaseModel, ConfigDict, Field

from manicule.config.settings import Settings
from manicule.container import Container, build_container, check_wiring, keys
from manicule.core.errors import (
    CircularDependencyError,
    ConfigError,
    PluginError,
    PolicyError,
    UnknownComponentError,
)
from manicule.core.lifecycle import HealthReport, HealthState, Metric
from manicule.core.protocols import Chunker, Embedder, Parser
from manicule.plugins.registry import BuildContext, ComponentRegistry, Discovery
from tests.fakes import MEDIA_TYPE, BlockChunker, HashEmbedder, LineParser


class ParserConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    greeting: str = Field(default="hello")


class Recorder:
    """A component that records what the container did to it, and when."""

    def __init__(self, name: str, log: list[str], *, fail_setup: bool = False) -> None:
        self.name = name
        self._log = log
        self._fail_setup = fail_setup

    async def setup(self) -> None:
        if self._fail_setup:
            msg = f"{self.name} refuses to start"
            raise RuntimeError(msg)
        self._log.append(f"setup:{self.name}")

    async def teardown(self) -> None:
        self._log.append(f"teardown:{self.name}")

    async def health(self) -> HealthReport:
        return HealthReport.healthy()

    def metrics(self) -> tuple[Metric, ...]:
        return (Metric(name="ready", value=1.0),)


@pytest.fixture
def wired(settings: Settings) -> tuple[Container, list[str]]:
    """A container over a small hand-built registry."""
    log: list[str] = []
    registry = ComponentRegistry().bind("test")
    registry.add(
        keys.PARSER.named("lines"),
        lambda _: LineParser(),
        config_model=ParserConfig,
        media_types={MEDIA_TYPE},
    )
    registry.add(keys.EMBEDDER.named("onnx"), lambda _: HashEmbedder())
    registry.add(keys.CHUNKER.named("structural"), lambda _: BlockChunker())
    registry.add(keys.MIDDLEWARE.named("first"), lambda _: Recorder("first", log))
    registry.add(keys.MIDDLEWARE.named("second"), lambda _: Recorder("second", log))
    return Container(settings, registry), log


# --- resolution ---------------------------------------------------------------------------


def test_resolution_is_typed_and_memoized(wired: tuple[Container, list[str]]) -> None:
    container, _ = wired
    parser = container.get(keys.PARSER.named("lines"))
    assert isinstance(parser, Parser)
    assert container.get(keys.PARSER.named("lines")) is parser


def test_an_unnamed_key_resolves_to_whatever_configuration_selected(
    wired: tuple[Container, list[str]],
) -> None:
    container, _ = wired
    embedder = container.get(keys.EMBEDDER)
    assert isinstance(embedder, Embedder)
    assert isinstance(container.get(keys.CHUNKER), Chunker)


def test_a_kind_with_no_default_must_be_asked_for_by_name(
    wired: tuple[Container, list[str]],
) -> None:
    """Configuration lists several parsers and several connectors; there is no "the" one."""
    container, _ = wired
    with pytest.raises(UnknownComponentError, match="must be requested by name"):
        container.get(keys.PARSER)


def test_a_dependency_cycle_names_the_cycle(settings: Settings) -> None:
    registry = ComponentRegistry().bind("test")
    registry.add(keys.CHUNKER.named("structural"), lambda ctx: ctx.components.get(keys.CHUNKER))
    container = Container(settings, registry)
    with pytest.raises(CircularDependencyError, match="cycle"):
        container.get(keys.CHUNKER)


def test_metadata_dependencies_never_construct_executable_components(settings: Settings) -> None:
    built: list[str] = []
    described: list[str] = []
    registry = ComponentRegistry().bind("test")

    def forbidden_embedder(_context: object) -> HashEmbedder:
        built.append("executable")
        return HashEmbedder()

    def forbidden_chunker(_context: object) -> BlockChunker:
        built.append("executable")
        return BlockChunker()

    registry.add(
        keys.EMBEDDER.named("onnx"),
        forbidden_embedder,
        metadata_factory=lambda _: described.append("embed") or "embedding-id",
    )
    registry.add(
        keys.CHUNKER.named("structural"),
        forbidden_chunker,
        metadata_factory=lambda context: (
            described.append("chunk") or f"chunk-for-{context.components.get(keys.EMBEDDER)}"
        ),
    )
    container = Container(settings, registry)

    assert container.metadata(keys.CHUNKER) == "chunk-for-embedding-id"
    assert container.metadata(keys.CHUNKER) == "chunk-for-embedding-id"
    assert built == []
    assert described == ["chunk", "embed"]


def test_missing_metadata_refuses_instead_of_falling_back_to_a_factory(settings: Settings) -> None:
    registry = ComponentRegistry().bind("test")
    registry.add(keys.EMBEDDER.named("onnx"), lambda _: HashEmbedder())

    with pytest.raises(ConfigError, match="no metadata-only identity"):
        Container(settings, registry).metadata(keys.EMBEDDER)


def test_a_factory_receives_its_validated_configuration(settings: Settings) -> None:
    seen: list[BuildContext] = []
    registry = ComponentRegistry().bind("test")
    registry.add(
        keys.PARSER.named("lines"),
        lambda ctx: seen.append(ctx) or LineParser(),
        config_model=ParserConfig,
        media_types={MEDIA_TYPE},
    )
    configured = settings.model_copy(
        update={
            "plugins": settings.plugins.model_copy(
                update={"config": {"parser.lines": {"greeting": "bonjour"}}}
            )
        }
    )
    Container(configured, registry).get(keys.PARSER.named("lines"))

    assert isinstance(seen[0].config, ParserConfig)
    assert seen[0].config.greeting == "bonjour"


def test_configuration_a_component_cannot_accept_is_rejected(settings: Settings) -> None:
    registry = ComponentRegistry().bind("test")
    registry.add(
        keys.PARSER.named("lines"),
        lambda _: LineParser(),
        config_model=ParserConfig,
        media_types={MEDIA_TYPE},
    )
    configured = settings.model_copy(
        update={
            "plugins": settings.plugins.model_copy(
                update={"config": {"parser.lines": {"greetign": "typo"}}}
            )
        }
    )
    with pytest.raises(ConfigError, match=r"parser\.lines"):
        Container(configured, registry).get(keys.PARSER.named("lines"))


def test_a_setting_of_the_wrong_type_names_the_field(settings: Settings) -> None:
    registry = ComponentRegistry().bind("test")
    registry.add(
        keys.PARSER.named("lines"),
        lambda _: LineParser(),
        config_model=ParserConfig,
        media_types={MEDIA_TYPE},
    )
    configured = settings.model_copy(
        update={
            "plugins": settings.plugins.model_copy(
                update={"config": {"parser.lines": {"greeting": 42}}}
            )
        }
    )
    with pytest.raises(ConfigError, match="greeting"):
        Container(configured, registry).get(keys.PARSER.named("lines"))


def test_configuring_a_component_that_declares_no_model_is_rejected(settings: Settings) -> None:
    """Silently ignoring it would leave the setting looking like it was in force."""
    registry = ComponentRegistry().bind("test")
    registry.add(keys.EMBEDDER.named("onnx"), lambda _: HashEmbedder())
    configured = settings.model_copy(
        update={
            "plugins": settings.plugins.model_copy(
                update={"config": {"embedder.onnx": {"anything": 1}}}
            )
        }
    )
    with pytest.raises(ConfigError, match="declares no configuration model"):
        Container(configured, registry).get(keys.EMBEDDER)


# --- lifecycle ------------------------------------------------------------------------------


async def test_setup_runs_in_dependency_order_and_teardown_in_reverse(
    wired: tuple[Container, list[str]],
) -> None:
    container, log = wired
    container.get(keys.MIDDLEWARE.named("first"))
    container.get(keys.MIDDLEWARE.named("second"))

    async with container:
        assert log == ["setup:first", "setup:second"]
    assert log == ["setup:first", "setup:second", "teardown:second", "teardown:first"]


async def test_a_component_resolved_after_startup_is_still_set_up(
    wired: tuple[Container, list[str]],
) -> None:
    container, log = wired
    async with container:
        await container.aget(keys.MIDDLEWARE.named("first"))
        assert log == ["setup:first"]


async def test_a_failed_startup_leaves_nothing_running(settings: Settings) -> None:
    log: list[str] = []
    registry = ComponentRegistry().bind("test")
    registry.add(keys.MIDDLEWARE.named("good"), lambda _: Recorder("good", log))
    registry.add(keys.MIDDLEWARE.named("bad"), lambda _: Recorder("bad", log, fail_setup=True))
    container = Container(settings, registry)
    container.get(keys.MIDDLEWARE.named("good"))
    container.get(keys.MIDDLEWARE.named("bad"))

    with pytest.raises(RuntimeError, match="refuses to start"):
        await container.start()

    # "bad" is torn down too. Its setup raised part-way, which is precisely when a component
    # has acquired something and not finished with it.
    assert log == ["setup:good", "teardown:bad", "teardown:good"]


async def test_one_component_failing_to_stop_does_not_strand_the_others(
    settings: Settings,
) -> None:
    """A cache that will not close must not leave a database handle open."""
    log: list[str] = []

    class Stubborn(Recorder):
        @override
        async def teardown(self) -> None:
            msg = "will not close"
            raise RuntimeError(msg)

    registry = ComponentRegistry().bind("test")
    registry.add(keys.MIDDLEWARE.named("first"), lambda _: Recorder("first", log))
    registry.add(keys.MIDDLEWARE.named("second"), lambda _: Stubborn("second", log))

    container = Container(settings, registry)
    container.get(keys.MIDDLEWARE.named("first"))
    container.get(keys.MIDDLEWARE.named("second"))
    await container.start()

    with pytest.raises(ExceptionGroup):
        await container.aclose()
    assert "teardown:first" in log


async def test_a_shutdown_error_never_hides_the_error_that_caused_the_shutdown(
    settings: Settings,
) -> None:
    """The failure being handled is more interesting than a failure while handling it."""

    class Stubborn(Recorder):
        @override
        async def teardown(self) -> None:
            msg = "will not close"
            raise RuntimeError(msg)

    registry = ComponentRegistry().bind("test")
    registry.add(keys.MIDDLEWARE.named("first"), lambda _: Stubborn("first", []))
    container = Container(settings, registry)
    container.get(keys.MIDDLEWARE.named("first"))
    await container.start()

    async def fail_inside_the_context() -> None:
        async with container:
            msg = "the original problem"
            raise ValueError(msg)

    with pytest.raises(ValueError, match="the original problem") as caught:
        await fail_inside_the_context()

    assert any("shutting down also failed" in note for note in caught.value.__notes__)


async def test_health_says_which_component_is_unwell(settings: Settings) -> None:
    class Unwell(Recorder):
        @override
        async def health(self) -> HealthReport:
            return HealthReport.degraded(
                "running on the fallback runtime", remedy="install manicule-mlx"
            )

    registry = ComponentRegistry().bind("test")
    registry.add(keys.MIDDLEWARE.named("fine"), lambda _: Recorder("fine", []))
    registry.add(keys.MIDDLEWARE.named("poorly"), lambda _: Unwell("poorly", []))

    container = Container(settings, registry)
    container.get(keys.MIDDLEWARE.named("fine"))
    container.get(keys.MIDDLEWARE.named("poorly"))
    async with container:
        report = await container.health()

    assert report.state is HealthState.DEGRADED
    assert "middleware:poorly" in report.detail
    assert {check.name for check in report.checks} == {"middleware:fine", "middleware:poorly"}


async def test_a_health_check_that_raises_is_itself_the_diagnosis(settings: Settings) -> None:
    class Exploding(Recorder):
        @override
        async def health(self) -> HealthReport:
            msg = "connection reset"
            raise RuntimeError(msg)

    registry = ComponentRegistry().bind("test")
    registry.add(keys.MIDDLEWARE.named("boom"), lambda _: Exploding("boom", []))
    container = Container(settings, registry)
    container.get(keys.MIDDLEWARE.named("boom"))
    async with container:
        report = await container.health()

    assert report.state is HealthState.FAILING
    assert "connection reset" in report.checks[0].detail


async def test_metrics_are_labeled_with_where_they_came_from(
    wired: tuple[Container, list[str]],
) -> None:
    container, _ = wired
    container.get(keys.MIDDLEWARE.named("first"))
    async with container:
        metrics = container.metrics()
    assert metrics[0].labels["component"] == "middleware:first"


# --- subsystem views --------------------------------------------------------------------------


async def test_middleware_runs_in_the_order_configuration_lists_it(settings: Settings) -> None:
    """Declared where a reader can see it, not emerging from priority numbers."""
    log: list[str] = []
    registry = ComponentRegistry().bind("test")
    registry.add(keys.MIDDLEWARE.named("first"), lambda _: Recorder("first", log))
    registry.add(keys.MIDDLEWARE.named("second"), lambda _: Recorder("second", log))

    configured = settings.model_copy(
        update={"plugins": settings.plugins.model_copy(update={"middleware": ("second", "first")})}
    )
    chain = await Container(configured, registry).middleware()
    assert [m.name for m in chain] == ["second", "first"]


async def test_the_parser_chain_puts_the_configured_order_first(settings: Settings) -> None:
    registry = ComponentRegistry().bind("test")
    registry.add(keys.PARSER.named("lines"), lambda _: LineParser(), media_types={MEDIA_TYPE})
    registry.add(keys.PARSER.named("other"), lambda _: LineParser(), media_types={MEDIA_TYPE})

    configured = settings.model_copy(
        update={"parser_fallbacks": {MEDIA_TYPE: ("other",), "*": ("lines",)}}
    )
    chain = await Container(configured, registry).parser_chain(MEDIA_TYPE)
    assert len(chain) == 2


async def test_routing_does_not_construct_the_parsers_it_did_not_choose(settings: Settings) -> None:
    """A parser's factory is where its heavy imports live, so routing must not run them all."""
    built: list[str] = []

    def factory(name: str) -> Callable[[BuildContext], LineParser]:
        def build(_: BuildContext) -> LineParser:
            built.append(name)
            return LineParser()

        return build

    registry = ComponentRegistry().bind("test")
    registry.add(keys.PARSER.named("wanted"), factory("wanted"), media_types={MEDIA_TYPE})
    registry.add(keys.PARSER.named("other"), factory("other"), media_types={"application/pdf"})

    chain = await Container(settings, registry).parser_chain(MEDIA_TYPE)
    assert len(chain) == 1
    assert built == ["wanted"]


async def test_a_parser_disagreeing_with_its_own_declaration_is_caught(settings: Settings) -> None:
    """Routing reads the declaration, so a parser that handles something else is unreachable."""
    registry = ComponentRegistry().bind("test")
    registry.add(
        keys.PARSER.named("mislabeled"), lambda _: LineParser(), media_types={"application/pdf"}
    )
    container = Container(settings, registry)
    with pytest.raises(PluginError, match="must agree"):
        await container.parser_chain("application/pdf")


async def test_a_connector_is_named_by_the_user_and_typed_by_configuration(
    settings: Settings,
) -> None:
    """The source is ``docs``; ``memory`` is only the implementation it asked for.

    This assertion used to read ``.name == "memory"`` — the *type* — under this same name,
    so the test asserted the defect while its title described the fix. That is worse than no
    test: it reported a property nobody had checked, and it is why the collapse of "connector
    type" into "configured source" survived as long as it did.
    """
    from tests.fakes import MemoryConnector  # noqa: PLC0415

    registry = ComponentRegistry().bind("test")
    registry.add(
        keys.CONNECTOR.named("memory"), lambda context: MemoryConnector(name=context.instance)
    )
    configured = settings.model_copy(update={"connectors": {"docs": _connector_settings("memory")}})
    container = Container(configured, registry)

    assert (await container.connector("docs")).name == "docs"
    with pytest.raises(UnknownComponentError, match="no connector named 'absent'"):
        await container.connector("absent")


def _connector_settings(type_: str) -> object:
    from manicule.config.settings import ConnectorSettings  # noqa: PLC0415

    return ConnectorSettings(type=type_)


# --- the startup gate ---------------------------------------------------------------------------


def test_configuration_naming_something_not_installed_is_caught_at_startup(
    settings: Settings,
) -> None:
    """Not at the first document that needs it, by which point a corpus already differs."""
    registry = ComponentRegistry()
    problems = check_wiring(settings, registry)
    assert any("embedding.provider" in problem for problem in problems)
    assert any("rag.pipeline" in problem for problem in problems)


def test_a_missing_parser_in_a_fallback_chain_is_caught(settings: Settings) -> None:
    """A chain that varies by machine chunks the same document differently on each one."""
    configured = settings.model_copy(
        update={"parser_fallbacks": {"application/pdf": ("marker", "pypdfium2")}}
    )
    problems = check_wiring(configured, ComponentRegistry())
    assert any("marker" in problem for problem in problems)


def test_the_gate_reports_everything_wrong_at_once(settings: Settings) -> None:
    with pytest.raises(PolicyError) as caught:
        build_container(settings, discovery=Discovery(registry=ComponentRegistry()))
    message = str(caught.value)
    assert message.count("\n  - ") >= 4


def test_a_complete_configuration_builds(settings: Settings) -> None:
    registry = ComponentRegistry().bind("test")
    registry.add(keys.EMBEDDER.named("onnx"), lambda _: HashEmbedder())
    registry.add(keys.CHUNKER.named("structural"), lambda _: BlockChunker())
    # By the name `llm.generator` selects, which is the registered *component* rather
    # than the vendor `llm.provider` names — one implementation reaches every vendor.
    registry.add(keys.GENERATOR.named("litellm"), lambda _: object())
    registry.add(keys.VECTOR_STORE.named("lancedb"), lambda _: object())
    registry.add(keys.DOC_STORE.named("sqlite"), lambda _: object())
    for stage in settings.rag.pipeline:
        registry.add(keys.RETRIEVAL_STAGE.named(stage), lambda _: object())

    container = build_container(settings, discovery=Discovery(registry=registry))
    assert isinstance(container.get(keys.EMBEDDER), Embedder)
    assert container.describe() == ["embedder:onnx"]
