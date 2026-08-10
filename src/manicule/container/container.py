"""The container: what turns configuration and installed plugins into a running system.

Assembled at startup and injected. Deliberately **not** a single wiring function that every
feature has to touch — each plugin declares its own components in its own ``register``, and
what remains here is the general machinery: resolve, configure, set up, tear down. Adding a
component to manicule means writing a plugin, not editing this file.

Three properties are worth stating, because they are what the machinery buys:

Construction order is dependency order
    A factory resolves what it needs while it runs, so a dependency is fully built before
    its dependent exists. Setup then walks that order, and teardown walks it backwards.

Everything is torn down, including after a failure
    A component that fails to start does not strand the ones that already started, and one
    that fails to stop does not prevent the rest from stopping.

Misconfiguration fails before construction
    Configuration naming a component that is not installed is an error at startup, with the
    installed alternatives listed — not a surprise at the first document that needs it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Self

from pydantic import BaseModel, ConfigDict, ValidationError

from manicule.config.profiles import profile_config
from manicule.config.settings import Settings
from manicule.container import keys
from manicule.core.errors import (
    CircularDependencyError,
    ConfigError,
    PluginError,
    PolicyError,
    UnknownComponentError,
)
from manicule.core.lifecycle import (
    HealthReport,
    Metric,
    SupportsHealth,
    SupportsMetrics,
    SupportsSetup,
    SupportsTeardown,
)
from manicule.core.protocols import Connector, Middleware, Parser, RetrievalStage
from manicule.plugins.manifest import ComponentKind
from manicule.plugins.registry import (
    BuildContext,
    ComponentKey,
    ComponentRecord,
    ComponentRegistry,
    Discovery,
    discover,
)


class NoConfig(BaseModel):
    """The configuration of a component that declares none."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Container:
    """Resolves components, and owns their lifecycle."""

    def __init__(
        self,
        settings: Settings,
        registry: ComponentRegistry,
        *,
        discovery: Discovery | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.discovery = discovery
        self._instances: dict[tuple[ComponentKind, str], object] = {}
        self._order: list[tuple[ComponentKind, str]] = []
        self._pending: list[tuple[ComponentKind, str]] = []
        self._started: list[tuple[ComponentKind, str]] = []
        self._resolving: list[str] = []

    # --- resolution -----------------------------------------------------------------

    def get[T](self, key: ComponentKey[T]) -> T:
        """Construct ``key``'s component, or return the one already constructed.

        Synchronous, so a factory can resolve its dependencies inline. Anything needing
        ``await`` belongs in :meth:`~manicule.core.lifecycle.Lifecycle.setup`, which the
        container calls in dependency order.
        """
        resolved = self._resolve_name(key)
        slot = (resolved.kind, resolved.name or "")
        existing = self._instances.get(slot)
        if existing is not None:
            return existing  # pyright: ignore[reportReturnType] - key's parameter is the contract

        label = str(resolved)
        if label in self._resolving:
            cycle = " -> ".join([*self._resolving[self._resolving.index(label) :], label])
            msg = f"component dependencies form a cycle: {cycle}"
            raise CircularDependencyError(msg)

        record = self.registry.record(resolved)
        self._resolving.append(label)
        try:
            instance = record.factory(self._context(record))
        finally:
            self._resolving.pop()

        self._instances[slot] = instance
        self._order.append(slot)
        self._pending.append(slot)
        return instance

    async def aget[T](self, key: ComponentKey[T]) -> T:
        """Resolve ``key`` and make sure it, and anything it needed, has been set up."""
        instance = self.get(key)
        await self._setup_pending()
        return instance

    def _resolve_name[T](self, key: ComponentKey[T]) -> ComponentKey[T]:
        if key.name is not None:
            return key
        chosen = self._configured_name(key.kind)
        if chosen is None:
            msg = (
                f"{key.kind.value} must be requested by name; configuration selects no "
                f"default for this kind"
            )
            raise UnknownComponentError(msg)
        return key.named(chosen)

    def _configured_name(self, kind: ComponentKind) -> str | None:
        """Which implementation configuration selects for a kind, if it selects one.

        Kinds absent from this table — parsers, connectors, middleware, retrieval stages —
        are always resolved by name, because configuration lists several of each.
        """
        settings = self.settings
        chosen: Mapping[ComponentKind, str | None] = {
            ComponentKind.EMBEDDER: settings.embedding.provider,
            ComponentKind.GENERATOR: settings.llm.provider,
            ComponentKind.VECTOR_STORE: settings.storage.vector_db,
            ComponentKind.DOC_STORE: settings.storage.db,
            ComponentKind.CHUNKER: settings.rag.chunker,
            ComponentKind.RERANKER: settings.rag.reranker,
        }
        return chosen.get(kind)

    def _context(self, record: ComponentRecord[object]) -> BuildContext:
        return BuildContext(
            settings=self.settings,
            config=self._config_for(record),
            data_dir=self.settings.data_dir,
            cache_dir=self.settings.cache_dir,
            components=self,
        )

    def _config_for(self, record: ComponentRecord[object]) -> BaseModel:
        raw = self.settings.component_config(record.kind.value, record.name)
        slot = f"{record.kind.value}.{record.name}"
        if record.config_model is None:
            if raw:
                msg = (
                    f"plugins.config[{slot!r}] was supplied, but {record.plugin!r} declares no "
                    f"configuration model for it. Remove the setting, or ask its author to "
                    f"declare one."
                )
                raise ConfigError(msg)
            return NoConfig()

        # Reject settings the component has no field for, whatever its model does with
        # extras, unless it opted into accepting them. A model defaulting to "ignore" would
        # otherwise turn a typo into a setting that appears to be in force and is not.
        if record.config_model.model_config.get("extra") != "allow":
            unknown = sorted(set(raw) - set(record.config_model.model_fields))
            if unknown:
                known = ", ".join(sorted(record.config_model.model_fields)) or "none"
                msg = (
                    f"invalid plugins.config[{slot!r}]: no such setting(s) "
                    f"{', '.join(unknown)}. Accepted: {known}"
                )
                raise ConfigError(msg)
        try:
            return record.config_model.model_validate(dict(raw))
        except ValidationError as exc:
            lines = [f"  {'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
            msg = f"invalid plugins.config[{slot!r}]:\n" + "\n".join(lines)
            raise ConfigError(msg) from exc

    # --- subsystem views ------------------------------------------------------------
    #
    # Each is a few lines that read configuration and resolve components. They live here
    # rather than in one startup routine so that a subsystem's wiring stays the size of the
    # subsystem.
    #
    # All of them are async, and none of them is async because it does I/O. They hand a
    # component to a caller who is about to use it, so anything they resolve must already
    # have been set up — and setup is the one part of the lifecycle that can await. The
    # synchronous `get` exists for factories, which run during construction, before setup
    # is due.

    async def parser_chain(self, media_type: str) -> list[Parser]:
        """Parsers to try for ``media_type``, in order.

        The configured chain first, then every parser that declared the media type, then
        the global ``*`` tail. Order is fixed by configuration rather than by which plugin
        happened to register first, so two installations with the same config parse the same
        document the same way.

        Routing reads the declaration each parser made when it registered, so choosing a
        parser does not construct the ones it did not choose. Each parser that *is* chosen
        is checked against its own declaration, so the two cannot drift apart unnoticed.
        """
        fallbacks = self.settings.parser_fallbacks
        names: list[str] = [*fallbacks.get(media_type, ())]
        names.extend(
            record.name
            for record in self.registry.records(ComponentKind.PARSER)
            if media_type in record.media_types and record.name not in names
        )
        names.extend(name for name in fallbacks.get("*", ()) if name not in names)

        chain: list[Parser] = []
        for name in names:
            parser = self.get(keys.PARSER.named(name))
            declared = self.registry.record(keys.PARSER.named(name)).media_types
            if parser.media_types != declared and media_type not in parser.media_types:
                msg = (
                    f"parser {name!r} registered {sorted(declared)} but handles "
                    f"{sorted(parser.media_types)}, and does not handle {media_type!r}. "
                    f"The declaration is what routing uses, so the two must agree."
                )
                raise PluginError(msg)
            chain.append(parser)
        await self._setup_pending()
        return chain

    async def middleware(self) -> list[Middleware]:
        """Configured middleware, in the order it runs."""
        chain = [self.get(keys.MIDDLEWARE.named(n)) for n in self.settings.plugins.middleware]
        await self._setup_pending()
        return chain

    async def retrieval_pipeline(self) -> list[RetrievalStage]:
        """The retrieval stages for the configured profile, in order.

        The reranker is appended when the profile asks for one and configuration names one.
        The result is a plain list of uniform stages, which is what lets the evaluation
        harness compare two pipelines without either of them being special.
        """
        settings = self.settings
        stages: list[RetrievalStage] = [
            self.get(keys.RETRIEVAL_STAGE.named(name)) for name in settings.rag.pipeline
        ]
        profile = profile_config(settings.rag.profile, settings.rag.overrides)
        if profile.rerank and settings.rag.reranker:
            stages.append(self.get(keys.RERANKER.named(settings.rag.reranker)))
        await self._setup_pending()
        return stages

    async def connector(self, instance: str) -> Connector:
        """The connector for a configured source.

        Sources are named by the user and typed by configuration, so two Confluence spaces
        are two connectors of one type rather than one connector serving two configurations.
        """
        configured = self.settings.connectors.get(instance)
        if configured is None:
            known = ", ".join(sorted(self.settings.connectors)) or "none configured"
            msg = f"no connector named {instance!r} in configuration. Configured: {known}"
            raise UnknownComponentError(msg)
        return await self.aget(keys.CONNECTOR.named(configured.type))

    # --- lifecycle ------------------------------------------------------------------

    async def start(self) -> None:
        """Set up every constructed component, in dependency order.

        If one fails, everything already started is torn down before the failure
        propagates, so a failed startup leaves nothing running.
        """
        try:
            await self._setup_pending()
        except BaseException as exc:
            # The startup failure is the one worth reading. A teardown that also fails is
            # recorded against it rather than replacing it.
            try:
                await self.aclose()
            except Exception as during_teardown:  # noqa: BLE001 - attached, not swallowed
                exc.add_note(f"shutting down after the failed start also failed: {during_teardown}")
            raise

    async def _setup_pending(self) -> None:
        while self._pending:
            slot = self._pending.pop(0)
            instance = self._instances[slot]
            # Recorded as started *before* setup runs. A component whose setup raises
            # part-way has usually acquired something, and its teardown is documented as
            # safe to call after a failed setup — which is worth nothing if the container
            # never calls it.
            self._started.append(slot)
            if isinstance(instance, SupportsSetup):
                await instance.setup()

    async def aclose(self) -> None:
        """Tear everything down, in reverse setup order.

        Every component is offered the chance to shut down even if an earlier one refused
        to, because a failure to close a cache must not leave a database handle open.

        Raises:
            ExceptionGroup: If any teardown raised, once they have all been attempted.
        """
        failures: list[Exception] = []
        while self._started:
            slot = self._started.pop()
            instance = self._instances.pop(slot, None)
            if isinstance(instance, SupportsTeardown):
                try:
                    await instance.teardown()
                except Exception as exc:  # noqa: BLE001 - collected, then reported together
                    failures.append(exc)
        self._pending.clear()
        self._instances.clear()
        self._order.clear()
        if failures:
            raise ExceptionGroup("errors while shutting down", failures)

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, tb
        if exc is None:
            await self.aclose()
            return
        # Something is already on its way out, and it is almost certainly more interesting
        # than a shutdown error. Record the shutdown error on it instead of replacing it.
        try:
            await self.aclose()
        except Exception as during_teardown:  # noqa: BLE001 - attached, not swallowed
            exc.add_note(f"shutting down also failed: {during_teardown}")

    # --- observability --------------------------------------------------------------

    async def health(self) -> HealthReport:
        """Ask every started component how it is, and combine the answers.

        A component that raises instead of reporting is itself a health signal, and is
        recorded as failing rather than allowed to break the check.
        """
        reports: dict[str, HealthReport] = {}
        for kind, name in self._started:
            instance = self._instances.get((kind, name))
            if not isinstance(instance, SupportsHealth):
                continue
            label = f"{kind.value}:{name}"
            try:
                reports[label] = await instance.health()
            except Exception as exc:  # noqa: BLE001 - an exception here is the diagnosis
                reports[label] = HealthReport.failing(f"health check raised: {exc}")
        return HealthReport.rollup(reports)

    def metrics(self) -> list[Metric]:
        """Every started component's metrics, labelled with where they came from."""
        collected: list[Metric] = []
        for kind, name in self._started:
            instance = self._instances.get((kind, name))
            if not isinstance(instance, SupportsMetrics):
                continue
            for metric in instance.metrics():
                labels = {**metric.labels, "component": f"{kind.value}:{name}"}
                collected.append(metric.model_copy(update={"labels": labels}))
        return collected

    def describe(self) -> Sequence[str]:
        """What is currently constructed, in construction order."""
        return [f"{kind.value}:{name}" for kind, name in self._order]


def check_wiring(settings: Settings, registry: ComponentRegistry) -> list[str]:
    """Configuration that names components nothing installed provides.

    Reported together, and before anything is constructed. A missing parser found halfway
    through an ingest run has already produced a corpus that a machine with the parser
    installed would have produced differently.
    """
    problems: list[str] = []

    def require(kind: ComponentKind, name: str | None, where: str) -> None:
        if name and not registry.has(kind, name):
            available = ", ".join(registry.names(kind)) or "none installed"
            problems.append(
                f"{where} names {kind.value} {name!r}, which is not installed. "
                f"Available: {available}"
            )

    require(ComponentKind.EMBEDDER, settings.embedding.provider, "embedding.provider")
    require(ComponentKind.GENERATOR, settings.llm.provider, "llm.provider")
    require(ComponentKind.VECTOR_STORE, settings.storage.vector_db, "storage.vector_db")
    require(ComponentKind.DOC_STORE, settings.storage.db, "storage.db")
    require(ComponentKind.CHUNKER, settings.rag.chunker, "rag.chunker")
    require(ComponentKind.RERANKER, settings.rag.reranker, "rag.reranker")

    for name in settings.rag.pipeline:
        require(ComponentKind.RETRIEVAL_STAGE, name, "rag.pipeline")
    for name in settings.plugins.middleware:
        require(ComponentKind.MIDDLEWARE, name, "plugins.middleware")
    for media_type, chain in sorted(settings.parser_fallbacks.items()):
        for name in chain:
            require(ComponentKind.PARSER, name, f"parser_fallbacks[{media_type!r}]")
    for instance, connector in sorted(settings.connectors.items()):
        require(ComponentKind.CONNECTOR, connector.type, f"connectors[{instance!r}].type")

    known = {f"{r.kind.value}.{r.name}" for r in registry.records()}
    for slot in sorted(settings.plugins.config):
        if slot not in known:
            problems.append(
                f"plugins.config[{slot!r}] configures a component that is not installed"
            )

    return problems


def build_container(
    settings: Settings,
    *,
    discovery: Discovery | None = None,
) -> Container:
    """Discover plugins, verify the configuration against them, and return the container.

    The whole composition root. It does not construct components — the container does that
    on demand — so this function does not grow when manicule does.

    Raises:
        PolicyError: The configuration is unrunnable, or names something not installed.
            Everything wrong is listed at once.
    """
    enabled = settings.plugins.enabled
    found = discovery or discover(
        enabled=None if enabled is None else frozenset(enabled),
        disabled=frozenset(settings.plugins.disabled),
    )
    problems = [*settings.policy_problems(), *check_wiring(settings, found.registry)]
    if problems:
        joined = "\n  - ".join(problems)
        msg = f"manicule cannot start:\n  - {joined}"
        raise PolicyError(msg)
    return Container(settings, found.registry, discovery=found)


__all__ = ["Container", "NoConfig", "build_container", "check_wiring"]
