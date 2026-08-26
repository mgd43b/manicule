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

import asyncio
from collections.abc import Mapping, Sequence
from contextlib import suppress
from types import TracebackType
from typing import Final, Self, cast

from pydantic import BaseModel, ConfigDict, ValidationError

from manicule.config.profiles import profile_config
from manicule.config.settings import ConnectorSettings, Settings
from manicule.container import keys
from manicule.core.errors import (
    CircularDependencyError,
    ComponentSetupError,
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
    MetadataContext,
    discover,
)

HEALTH_CHECK_TIMEOUT_S: Final = 2.0
"""How long one component may take to answer before the sweep stops waiting on it.

Two seconds is well past a local check and past a healthy remote one; what it excludes is a
source that is unreachable rather than slow, where the honest report is that nobody knows yet.
"""

HEALTH_DEADLINE_S: Final = 5.0
"""How long the whole sweep may take, whatever the per-check clock allows.

The per-check timeout alone bounds one component. This bounds the answer: with enough
components, enough of them slow, the sum can still outrun whoever asked.
"""

HEALTH_CONCURRENCY: Final = 8
"""How many components are asked at once.

Bounded rather than unlimited because these are outbound requests, and a diagnostic that
opened a connection per configured source would be its own incident on a large installation.
"""


class _MetadataResolver:
    """The descriptor-only dependency view exposed to metadata factories."""

    def __init__(self, container: Container) -> None:
        self._container = container

    def get(self, key: ComponentKey[object]) -> object:
        return self._container.metadata(key)


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
        self._failed: dict[tuple[ComponentKind, str], BaseException] = {}
        """Slots whose ``setup()`` raised, and what it raised.

        Kept because the instance is *not* discarded when setup fails, and it must never be
        handed out as though it had started. ``start()`` wraps ``_setup_pending`` in a teardown
        that pops the instances, so a failed startup is clean — but ``aget`` calls
        ``_setup_pending`` directly, with no such wrapper. There the exception reached the
        caller while the half-built instance stayed in ``_instances`` and its slot had already
        been popped from ``_pending``, so the *next* ``aget`` found it cached, found nothing
        pending, and returned it: a component that never completed ``setup`` handed out as a
        working one, silently, for the rest of the process's life.
        """
        self._resolving: list[str] = []
        self._metadata: dict[tuple[ComponentKind, str], object] = {}
        self._metadata_resolving: list[str] = []

    # --- resolution -----------------------------------------------------------------

    def get[T](self, key: ComponentKey[T]) -> T:
        """Construct ``key``'s component, or return the one already constructed.

        Synchronous, so a factory can resolve its dependencies inline. Anything needing
        ``await`` belongs in :meth:`~manicule.core.lifecycle.Lifecycle.setup`, which the
        container calls in dependency order.
        """
        resolved = self._resolve_name(key)
        slot = (resolved.kind, resolved.name or "")
        failure = self._failed.get(slot)
        if failure is not None:
            # Re-raised rather than retried. A component that failed to start has usually
            # acquired something already, and `setup` is not documented as idempotent; the
            # container's answer to a broken component is to keep saying so.
            msg = f"{resolved} failed to start and cannot be resolved"
            raise ComponentSetupError(msg) from failure
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

    def metadata(self, key: ComponentKey[object]) -> object:
        """Resolve configured metadata without invoking a component factory."""
        resolved = self._resolve_name(key)
        slot = (resolved.kind, resolved.name or "")
        if slot in self._metadata:
            return self._metadata[slot]
        record = self.registry.record(resolved)
        if record.metadata_factory is None:
            raise ConfigError(
                f"{resolved} provides no metadata-only identity; rebuild planning cannot "
                "construct executable components to discover it"
            )
        label = str(resolved)
        if label in self._metadata_resolving:
            cycle = " -> ".join(
                [
                    *self._metadata_resolving[self._metadata_resolving.index(label) :],
                    label,
                ]
            )
            raise CircularDependencyError(f"component metadata dependencies form a cycle: {cycle}")
        self._metadata_resolving.append(label)
        try:
            value = record.metadata_factory(
                MetadataContext(
                    settings=self.settings,
                    config=self._config_for(record),
                    data_dir=self.settings.data_dir,
                    cache_dir=self.settings.cache_dir,
                    components=_MetadataResolver(self),
                )
            )
        finally:
            self._metadata_resolving.pop()
        self._metadata[slot] = value
        return value

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
            ComponentKind.GENERATOR: settings.llm.generator,
            ComponentKind.VECTOR_STORE: settings.storage.vector_db,
            ComponentKind.DOC_STORE: settings.storage.db,
            ComponentKind.CHUNKER: settings.rag.chunker,
            ComponentKind.RERANKER: settings.rag.reranker,
        }
        return chosen.get(kind)

    def _context[T](self, record: ComponentRecord[T]) -> BuildContext:
        return BuildContext(
            settings=self.settings,
            config=self._config_for(record),
            data_dir=self.settings.data_dir,
            cache_dir=self.settings.cache_dir,
            components=self,
        )

    def _config_for[T](self, record: ComponentRecord[T]) -> BaseModel:
        raw = self.settings.component_config(record.kind.value, record.name)
        where = f"plugins.config[{f'{record.kind.value}.{record.name}'!r}]"
        return self._validated_config(record, raw, where)

    def _validated_config[T](
        self,
        record: ComponentRecord[T],
        raw: Mapping[str, object],
        where: str,
    ) -> BaseModel:
        """``raw``, checked against ``record``'s declared model.

        ``where`` names the setting's location in configuration and appears in every error
        this raises. It is a parameter rather than derived from ``record`` because one
        component's settings now arrive from two places — the global
        ``plugins.config."connector.<type>"`` slot and a named instance's own ``options`` —
        and an error naming the wrong one sends the author to a file that is not the problem.
        """
        if record.config_model is None:
            if raw:
                msg = (
                    f"{where} was supplied, but {record.plugin!r} declares no configuration "
                    f"model for it. Remove the setting, or ask its author to declare one."
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
                msg = f"invalid {where}: no such setting(s) {', '.join(unknown)}. Accepted: {known}"
                raise ConfigError(msg)
        try:
            return record.config_model.model_validate(dict(raw))
        except ValidationError as exc:
            lines = [f"  {'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
            msg = f"invalid {where}:\n" + "\n".join(lines)
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

    def parser_chain_names(self, media_type: str) -> list[str]:
        """Registered parser names to try for ``media_type``, in order.

        Synchronous and constructing nothing, because two callers need the *chain* without
        needing the parsers: the ingest pipeline records it on the document before the first
        attempt (``docs/ingest.md`` §5), and each attempt then runs in a worker subprocess
        that resolves only the one parser it was asked for. Resolving lazily instead would
        let a configuration reload mid-chain produce a chain that never existed.
        """
        fallbacks = self.settings.parser_fallbacks
        names: list[str] = [*fallbacks.get(media_type, ())]
        names.extend(
            record.name
            for record in self.registry.records(ComponentKind.PARSER)
            if media_type in record.media_types and record.name not in names
        )
        names.extend(name for name in fallbacks.get("*", ()) if name not in names)
        return names

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
        chain: list[Parser] = []
        for name in self.parser_chain_names(media_type):
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
        # Two stages sharing a name is refused before construction, in `check_wiring`, and
        # again over the built stages by the pipeline runner — the first catches a repeated
        # entry in configuration, the second catches a stage whose own `name` collides with
        # another's. Both matter: candidate scores are keyed by stage name, so a collision
        # means the second stage silently overwrites the first's record and fusion reads a
        # ladder missing half its rungs, which produces a plausible ranking and no error.
        await self._setup_pending()
        return stages

    async def connector(self, instance: str) -> Connector:
        """The connector for a configured source.

        Sources are named by the user and typed by configuration, so two Confluence spaces
        are two connectors of one type rather than one connector serving two configurations.

        **Cached per instance, not per type**, which is what makes the sentence above true.
        The ordinary :meth:`get` path keys its cache on the *component* name, and connectors
        are the one kind where that is the wrong key: configuration lists sources, and two of
        them naming one implementation are two sources. Sharing the constructed object would
        give the second instance the first's root, and — because ``Connector.name`` becomes
        the ``source`` half of ``document_id`` — file its documents under the first's identity.
        """
        configured = self.settings.connectors.get(instance)
        if configured is None:
            known = ", ".join(sorted(self.settings.connectors)) or "none configured"
            msg = f"no connector named {instance!r} in configuration. Configured: {known}"
            raise UnknownComponentError(msg)

        slot = (ComponentKind.CONNECTOR, instance)
        existing = self._instances.get(slot)
        if existing is not None:
            await self._setup_pending()
            return cast("Connector", existing)

        record = self.registry.record(keys.CONNECTOR.named(configured.type))
        built = record.factory(
            BuildContext(
                settings=self.settings,
                config=self._connector_config(record, configured, instance),
                data_dir=self.settings.data_dir,
                cache_dir=self.settings.cache_dir,
                components=self,
                instance=instance,
            )
        )
        self._instances[slot] = built
        self._order.append(slot)
        self._pending.append(slot)
        await self._setup_pending()
        return built

    def _connector_config[T](
        self,
        record: ComponentRecord[T],
        configured: ConnectorSettings,
        instance: str,
    ) -> BaseModel:
        """One instance's settings: its own ``options`` over the type's global defaults.

        Two layers, shallowest first, because both have a job and neither subsumes the other:

        ``plugins.config."connector.<type>"`` supplies defaults for every instance of a type.
            It is also the *only* place settings could go before named instances carried their
            own, so reading it keeps every existing single-instance configuration working
            without its author editing anything.

        ``[connectors.<name>.options]`` is what that source actually is.
            It wins field by field, because it is the more specific statement. A shallow merge
            rather than a replacement so that a global ``include_hidden`` stays in force for an
            instance that only wanted to name its own root — replacing wholesale would silently
            drop settings the author can still see in the file.

        Validated here rather than at first use, so a misspelled option is a startup error
        naming the instance rather than a setting that appears to be in force and is not. When
        both layers carry something the error names both, because the merged value came from
        both and blaming one would send the author to the wrong line half the time.
        """
        globals_ = self.settings.component_config(record.kind.value, record.name)
        merged: dict[str, object] = {**globals_, **configured.options}
        slot = f"plugins.config[{f'{record.kind.value}.{record.name}'!r}]"
        own = f"connectors[{instance!r}].options"
        if configured.options and globals_:
            where = f"{own} merged over {slot}"
        elif configured.options:
            where = own
        else:
            where = slot
        return self._validated_config(record, merged, where)

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
                try:
                    await instance.setup()
                except BaseException as exc:
                    self._failed[slot] = exc
                    raise

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
        """Ask every started component how it is, at once, and combine the answers.

        **Concurrently, and under two clocks, because some of these questions leave the
        machine.** A connector's health check is an outbound request, so asking each in turn
        cost the sum of every remote latency — a local page took seconds to say that local
        storage was fine, and it took longer with each source an operator configured. The
        component that is slow is now the only thing waiting on it.

        Neither clock is a judgment about the component. A check that passes
        :data:`HEALTH_CHECK_TIMEOUT_S` is reported ``degraded`` rather than ``failing``,
        because "it did not answer in two seconds" is not evidence that a source is down; it
        is evidence that this diagnostic cannot say. :data:`HEALTH_DEADLINE_S` bounds the
        whole sweep for the same reason, and what it bounds is the *report*, not the fleet:
        checks that answered are kept, and the ones still outstanding are named as
        outstanding. A sweep that returned only the fast components would be a green report
        that had not looked.

        A component that raises instead of reporting is itself a health signal, and is
        recorded as failing rather than allowed to break the check.
        """
        checkable: list[tuple[str, SupportsHealth]] = []
        for kind, name in self._started:
            instance = self._instances.get((kind, name))
            if isinstance(instance, SupportsHealth):
                checkable.append((f"{kind.value}:{name}", instance))
        if not checkable:
            return HealthReport.rollup({})

        admitted = asyncio.Semaphore(HEALTH_CONCURRENCY)

        async def observe(instance: SupportsHealth) -> HealthReport:
            try:
                async with admitted, asyncio.timeout(HEALTH_CHECK_TIMEOUT_S):
                    return await instance.health()
            except TimeoutError:
                return HealthReport.degraded(
                    f"health check did not answer within {HEALTH_CHECK_TIMEOUT_S:g}s",
                    remedy="Check the component directly; this diagnostic stopped waiting.",
                )
            except Exception as exc:  # noqa: BLE001 - an exception here is the diagnosis
                return HealthReport.failing(f"health check raised: {exc}")

        started = {
            label: asyncio.create_task(observe(instance), name=f"health:{label}")
            for label, instance in checkable
        }
        try:
            await asyncio.wait(started.values(), timeout=HEALTH_DEADLINE_S)
            # Each task's own state decides, rather than the pending set `asyncio.wait` returned.
            # The two agree today — nothing awaits between that partition and this — but a check
            # that answered has to be kept because it answered, not because a set computed a
            # moment earlier still says so. Read before the cancellation below, so a component
            # that made it under the deadline is never reported as one that did not.
            #
            # Built in start order rather than completion order, so that running the same sweep
            # twice does not reorder a report somebody is diffing.
            reports = {
                label: (
                    task.result()
                    if task.done() and not task.cancelled()
                    else HealthReport.degraded(
                        f"health check was still running at the {HEALTH_DEADLINE_S:g}s deadline "
                        "for the whole sweep",
                        remedy="Check the component directly; this diagnostic stopped waiting.",
                    )
                )
                for label, task in started.items()
            }
        finally:
            # `asyncio.wait` does not cancel what it waits on, and neither does a cancellation
            # of whoever called this. Without this the sweep's own tasks would outlive it —
            # still asking remote components on behalf of a request that no longer exists, and
            # touching component instances a shutdown is in the middle of tearing down.
            for task in started.values():
                task.cancel()
            # Awaited while unwinding so the cancellations land here rather than whenever the
            # loop next runs. A cancellation arriving during that is suppressed, not lost: the
            # one already propagating is what continues out of this block.
            with suppress(asyncio.CancelledError):
                await asyncio.gather(*started.values(), return_exceptions=True)
        return HealthReport.rollup(reports)

    def metrics(self) -> list[Metric]:
        """Every started component's metrics, labeled with where they came from."""
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
    require(ComponentKind.GENERATOR, settings.llm.generator, "llm.generator")
    require(ComponentKind.VECTOR_STORE, settings.storage.vector_db, "storage.vector_db")
    require(ComponentKind.DOC_STORE, settings.storage.db, "storage.db")
    require(ComponentKind.CHUNKER, settings.rag.chunker, "rag.chunker")
    require(ComponentKind.RERANKER, settings.rag.reranker, "rag.reranker")

    for name in settings.rag.pipeline:
        require(ComponentKind.RETRIEVAL_STAGE, name, "rag.pipeline")
    repeated = sorted(
        {name for name in settings.rag.pipeline if settings.rag.pipeline.count(name) > 1}
    )
    if repeated:
        problems.append(
            f"rag.pipeline names {', '.join(repeated)} more than once. Candidate scores are "
            f"keyed by stage name, so the second occurrence would overwrite the first's "
            f"record and fusion would read a ladder missing half its rungs — a plausible "
            f"ranking, computed from the wrong numbers"
        )
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
