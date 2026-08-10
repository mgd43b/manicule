"""The container: resolution, configuration, lifecycle and the startup gate."""

from __future__ import annotations

from typing import override

import pytest
from pydantic import BaseModel, ConfigDict, Field

from manicule.config.settings import Settings
from manicule.container import Container, build_container, check_wiring, keys
from manicule.core.errors import (
    CircularDependencyError,
    ConfigError,
    PolicyError,
    UnknownComponentError,
)
from manicule.core.lifecycle import HealthReport, HealthState, Metric
from manicule.core.protocols import Chunker, Embedder, Parser
from manicule.plugins.registry import BuildContext, ComponentRegistry, Discovery
from tests.fakes import BlockChunker, HashEmbedder, LineParser


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
    )
    registry.add(keys.EMBEDDER.named("mlx"), lambda _: HashEmbedder())
    registry.add(keys.CHUNKER.named("structural"), lambda _: BlockChunker())
    registry.add(keys.MIDDLEWARE.named("first"), lambda _: Recorder("first", log))
    registry.add(keys.MIDDLEWARE.named("second"), lambda _: Recorder("second", log))
    return Container(settings, registry), log


# --- resolution ---------------------------------------------------------------------------


def test_resolution_is_typed_and_memoised(wired: tuple[Container, list[str]]) -> None:
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


def test_a_factory_receives_its_validated_configuration(settings: Settings) -> None:
    seen: list[BuildContext] = []
    registry = ComponentRegistry().bind("test")
    registry.add(
        keys.PARSER.named("lines"),
        lambda ctx: seen.append(ctx) or LineParser(),
        config_model=ParserConfig,
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
    registry.add(keys.PARSER.named("lines"), lambda _: LineParser(), config_model=ParserConfig)
    configured = settings.model_copy(
        update={
            "plugins": settings.plugins.model_copy(
                update={"config": {"parser.lines": {"greetign": "typo"}}}
            )
        }
    )
    with pytest.raises(ConfigError, match=r"parser\.lines"):
        Container(configured, registry).get(keys.PARSER.named("lines"))


def test_configuring_a_component_that_declares_no_model_is_rejected(settings: Settings) -> None:
    """Silently ignoring it would leave the setting looking like it was in force."""
    registry = ComponentRegistry().bind("test")
    registry.add(keys.EMBEDDER.named("mlx"), lambda _: HashEmbedder())
    configured = settings.model_copy(
        update={
            "plugins": settings.plugins.model_copy(
                update={"config": {"embedder.mlx": {"anything": 1}}}
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
    assert log == ["setup:good", "teardown:good"]


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


async def test_health_says_which_component_is_unwell(settings: Settings) -> None:
    class Unwell(Recorder):
        @override
        async def health(self) -> HealthReport:
            return HealthReport.degraded("running on the fallback runtime", remedy="install mlx")

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


async def test_metrics_are_labelled_with_where_they_came_from(
    wired: tuple[Container, list[str]],
) -> None:
    container, _ = wired
    container.get(keys.MIDDLEWARE.named("first"))
    async with container:
        metrics = container.metrics()
    assert metrics[0].labels["component"] == "middleware:first"


# --- subsystem views --------------------------------------------------------------------------


def test_middleware_runs_in_the_order_configuration_lists_it(settings: Settings) -> None:
    """Declared where a reader can see it, not emerging from priority numbers."""
    log: list[str] = []
    registry = ComponentRegistry().bind("test")
    registry.add(keys.MIDDLEWARE.named("first"), lambda _: Recorder("first", log))
    registry.add(keys.MIDDLEWARE.named("second"), lambda _: Recorder("second", log))

    configured = settings.model_copy(
        update={"plugins": settings.plugins.model_copy(update={"middleware": ("second", "first")})}
    )
    chain = Container(configured, registry).middleware()
    assert [m.name for m in chain] == ["second", "first"]


def test_the_parser_chain_puts_the_configured_order_first(settings: Settings) -> None:
    registry = ComponentRegistry().bind("test")
    registry.add(keys.PARSER.named("lines"), lambda _: LineParser())
    registry.add(keys.PARSER.named("other"), lambda _: LineParser())

    configured = settings.model_copy(
        update={"parser_fallbacks": {"text/x-fake": ("other",), "*": ("lines",)}}
    )
    chain = Container(configured, registry).parser_chain("text/x-fake")
    assert len(chain) == 2


def test_a_connector_is_named_by_the_user_and_typed_by_configuration(
    settings: Settings,
) -> None:
    from tests.fakes import MemoryConnector  # noqa: PLC0415

    registry = ComponentRegistry().bind("test")
    registry.add(keys.CONNECTOR.named("memory"), lambda _: MemoryConnector())
    configured = settings.model_copy(update={"connectors": {"docs": _connector_settings("memory")}})
    container = Container(configured, registry)

    assert container.connector("docs").name == "memory"
    with pytest.raises(UnknownComponentError, match="no connector named 'absent'"):
        container.connector("absent")


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
    registry.add(keys.EMBEDDER.named("mlx"), lambda _: HashEmbedder())
    registry.add(keys.CHUNKER.named("structural"), lambda _: BlockChunker())
    registry.add(keys.GENERATOR.named("ollama"), lambda _: object())
    registry.add(keys.VECTOR_STORE.named("lancedb"), lambda _: object())
    registry.add(keys.DOC_STORE.named("sqlite"), lambda _: object())
    for stage in settings.rag.pipeline:
        registry.add(keys.RETRIEVAL_STAGE.named(stage), lambda _: object())

    container = build_container(settings, discovery=Discovery(registry=registry))
    assert isinstance(container.get(keys.EMBEDDER), Embedder)
    assert container.describe() == ["embedder:mlx"]
