"""Component registration and entry-point discovery.

Discovery is real discovery: anything installed that advertises a ``manicule.plugins`` entry
point is found, without being named in a configuration file first. Configuration filters
what was discovered; it is not the mechanism by which things are found.

Built-in components arrive the same way. There is no shorter internal path, so the
extension mechanism is exercised by every installation rather than only by the people using
it, and it cannot rot unnoticed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, replace
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, override

from pydantic import BaseModel

from manicule.core.errors import DuplicateComponentError, PluginLoadError, UnknownComponentError
from manicule.core.version import CORE_VERSION
from manicule.plugins.compat import require_compatible, sort_by_load_order
from manicule.plugins.manifest import ComponentKind, Plugin, PluginManifest

if TYPE_CHECKING:
    from manicule.config.settings import Settings

ENTRY_POINT_GROUP = "manicule.plugins"
"""The one group. Built-in and third-party components are indistinguishable here."""


@dataclass(frozen=True, slots=True)
class ComponentKey[T]:
    """Identifies a component, and remembers what type it is.

    The type parameter is what makes resolution typed: ``container.get(keys.EMBEDDER)``
    returns an :class:`~manicule.core.protocols.Embedder`, checked at author time rather
    than discovered by an attribute error.
    """

    kind: ComponentKind
    name: str | None = None
    """``None`` means "whichever one configuration selects for this kind"."""

    def named(self, name: str) -> ComponentKey[T]:
        """This key, bound to a specific implementation."""
        return replace(self, name=name)

    @override
    def __str__(self) -> str:
        return f"{self.kind.value}:{self.name}" if self.name else self.kind.value


class ComponentResolver(Protocol):
    """The part of the container a factory is allowed to see.

    Narrow on purpose. A factory can ask for the components it depends on; it cannot enumerate
    the system, reconfigure it, or reach the lifecycle machinery.
    """

    def get[T](self, key: ComponentKey[T]) -> T:
        """Resolve a dependency. Construction only — never awaits."""
        ...


@dataclass(frozen=True, slots=True)
class BuildContext:
    """Everything a factory is given.

    Dependencies come through here at construction time, which is why
    :meth:`~manicule.core.lifecycle.Lifecycle.setup` needs no context argument of its own.
    """

    settings: Settings
    config: BaseModel
    """This component's own configuration, already validated against its declared model."""

    data_dir: Path
    cache_dir: Path
    components: ComponentResolver


type Factory[T] = Callable[[BuildContext], T]


@dataclass(frozen=True, slots=True)
class ComponentRecord[T]:
    """One registered component: how to build it, and who said so."""

    kind: ComponentKind
    name: str
    plugin: str
    factory: Factory[T]
    config_model: type[BaseModel] | None = None
    summary: str = ""

    @property
    def key(self) -> ComponentKey[T]:
        return ComponentKey(self.kind, self.name)


class ComponentRegistry:
    """What every installed plugin provides.

    A flat table, built once at startup and then read-only in practice. It holds factories,
    never instances — nothing is constructed until something asks for it.
    """

    def __init__(self) -> None:
        self._records: dict[tuple[ComponentKind, str], ComponentRecord[object]] = {}
        self._plugin: str = "unknown"

    def bind(self, plugin: str) -> ComponentRegistry:
        """A view that attributes registrations to ``plugin``.

        Shares the underlying table, so a plugin cannot register components that the rest of
        the system cannot see, and cannot register them anonymously either.
        """
        view = ComponentRegistry()
        view._records = self._records
        view._plugin = plugin
        return view

    def add[T](
        self,
        key: ComponentKey[T],
        factory: Factory[T],
        *,
        config_model: type[BaseModel] | None = None,
        summary: str = "",
    ) -> None:
        """Register a factory under ``key``.

        Args:
            key: Must name an implementation — an unnamed key is a request, not a offering.
            factory: Called with a :class:`BuildContext` to construct the component. Keep
                expensive imports inside it, so that installing a plugin costs nothing until
                it is used.
            config_model: Validates this component's configuration. Without one, the
                component is built with an empty config and any configuration written for it
                is rejected rather than silently ignored.
            summary: One line, shown by diagnostics.

        Raises:
            ValueError: ``key`` has no name.
            DuplicateComponentError: Another plugin already claimed that kind and name.
                Silent shadowing would make behaviour depend on installation order.
        """
        if key.name is None:
            msg = f"cannot register an unnamed component of kind {key.kind.value!r}"
            raise ValueError(msg)
        slot = (key.kind, key.name)
        existing = self._records.get(slot)
        if existing is not None:
            msg = (
                f"{key.kind.value} {key.name!r} is provided by both {existing.plugin!r} and "
                f"{self._plugin!r}. Uninstall one, or ask its author to rename it."
            )
            raise DuplicateComponentError(msg)
        record: ComponentRecord[object] = ComponentRecord(
            kind=key.kind,
            name=key.name,
            plugin=self._plugin,
            factory=factory,
            config_model=config_model,
            summary=summary,
        )
        self._records[slot] = record

    def record[T](self, key: ComponentKey[T]) -> ComponentRecord[T]:
        """The record for ``key``.

        Raises:
            UnknownComponentError: Nothing installed provides it. The message lists what
                does, because the usual cause is a typo or a missing extra.
        """
        if key.name is None:
            msg = f"no implementation named for {key.kind.value!r}"
            raise UnknownComponentError(msg)
        found = self._records.get((key.kind, key.name))
        if found is None:
            available = ", ".join(self.names(key.kind)) or "none installed"
            msg = (
                f"no {key.kind.value} named {key.name!r}. Available: {available}. "
                f"Plugins are found through the {ENTRY_POINT_GROUP!r} entry-point group; "
                f"a plugin that is installed but not listed here did not register it."
            )
            raise UnknownComponentError(msg)
        return found  # pyright: ignore[reportReturnType] - key's type parameter is the contract

    def has(self, kind: ComponentKind, name: str) -> bool:
        return (kind, name) in self._records

    def names(self, kind: ComponentKind) -> list[str]:
        """Registered implementation names for a kind, sorted."""
        return sorted(name for (k, name) in self._records if k is kind)

    def records(self, kind: ComponentKind | None = None) -> list[ComponentRecord[object]]:
        """Every record, or every record of one kind, in a stable order."""
        chosen = [r for r in self._records.values() if kind is None or r.kind is kind]
        return sorted(chosen, key=lambda r: (r.kind.value, r.name))

    def __len__(self) -> int:
        return len(self._records)


@dataclass(frozen=True, slots=True)
class Discovery:
    """The outcome of looking for plugins."""

    registry: ComponentRegistry
    manifests: tuple[PluginManifest, ...] = ()
    disabled: tuple[str, ...] = ()
    """Plugins that were found and deliberately not loaded."""

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(m.name for m in self.manifests)

    def manifest(self, name: str) -> PluginManifest | None:
        return next((m for m in self.manifests if m.name == name), None)


@dataclass(frozen=True, slots=True)
class _Candidate:
    entry_point: EntryPoint
    plugin: Plugin
    manifest: PluginManifest = field(compare=False)


def _load_entry_point(entry_point: EntryPoint) -> Plugin:
    """Import an entry point and coerce it to a plugin.

    Two accepted shapes: a plugin object, or a zero-argument callable returning one.
    """
    try:
        target: object = entry_point.load()
    except Exception as exc:
        msg = (
            f"plugin entry point {entry_point.name!r} ({entry_point.value}) could not be "
            f"imported: {exc}"
        )
        raise PluginLoadError(msg) from exc

    if isinstance(target, Plugin):
        return target
    if callable(target):
        produced: object = target()
        if isinstance(produced, Plugin):
            return produced
        msg = (
            f"plugin entry point {entry_point.name!r} is a callable that returned "
            f"{type(produced).__name__}, which has no 'manifest' and 'register'"
        )
        raise PluginLoadError(msg)
    msg = (
        f"plugin entry point {entry_point.name!r} resolved to {type(target).__name__}, which "
        f"is neither a plugin (an object with 'manifest' and 'register') nor a callable "
        f"returning one"
    )
    raise PluginLoadError(msg)


def _manifest_of(entry_point: EntryPoint, plugin: Plugin) -> PluginManifest:
    manifest = plugin.manifest
    if not isinstance(manifest, PluginManifest):  # pyright: ignore[reportUnnecessaryIsInstance]
        msg = (
            f"plugin {entry_point.name!r} has a 'manifest' of type "
            f"{type(manifest).__name__}, not PluginManifest"
        )
        raise PluginLoadError(msg)
    if manifest.name != entry_point.name:
        msg = (
            f"plugin entry point is named {entry_point.name!r} but its manifest says "
            f"{manifest.name!r}. They must agree: configuration and diagnostics use one name "
            f"for a plugin, and two names for one plugin is a bug waiting to be filed."
        )
        raise PluginLoadError(msg)
    return manifest


def installed_entry_points(group: str = ENTRY_POINT_GROUP) -> list[EntryPoint]:
    """Every plugin entry point in the environment, sorted by name."""
    return sorted(entry_points(group=group), key=lambda ep: ep.name)


def discover(
    *,
    core_version: str = CORE_VERSION,
    enabled: AbstractSet[str] | None = None,
    disabled: AbstractSet[str] = frozenset(),
    points: Iterable[EntryPoint] | None = None,
) -> Discovery:
    """Find, verify and register every installed plugin.

    Args:
        core_version: What plugins declare compatibility against.
        enabled: If given, only these plugins load. ``None`` loads everything discovered.
        disabled: Names to skip. Takes precedence over ``enabled``.
        points: Entry points to use instead of the environment's. For tests.

    Returns:
        The populated registry and the manifests of everything loaded.

    Raises:
        PluginLoadError: An entry point could not be imported or was not a plugin.
        IncompatiblePluginError: A plugin does not support this core version.
        PluginDependencyError: A requirement is missing, a conflict is present, or the
            requirement graph has a cycle.
        DuplicateComponentError: Two plugins claim the same component.

    Every one of these is fatal. A plugin that is installed and cannot load is a
    misconfiguration to fix at startup, not a degradation to discover later from a query
    that quietly returns nothing.
    """
    found = list(points) if points is not None else installed_entry_points()

    seen: dict[str, EntryPoint] = {}
    for entry_point in found:
        clash = seen.get(entry_point.name)
        if clash is not None:
            msg = (
                f"two installed distributions both provide the plugin {entry_point.name!r} "
                f"({clash.value} and {entry_point.value}). Uninstall one."
            )
            raise PluginLoadError(msg)
        seen[entry_point.name] = entry_point

    skipped = tuple(sorted(name for name in seen if _is_disabled(name, enabled, disabled)))
    wanted = [ep for name, ep in sorted(seen.items()) if name not in skipped]

    candidates: list[_Candidate] = []
    for entry_point in wanted:
        plugin = _load_entry_point(entry_point)
        manifest = _manifest_of(entry_point, plugin)
        candidates.append(_Candidate(entry_point=entry_point, plugin=plugin, manifest=manifest))

    present = {c.manifest.name for c in candidates}
    for candidate in candidates:
        require_compatible(candidate.manifest, core_version, present)

    ordered_names = [m.name for m in sort_by_load_order([c.manifest for c in candidates])]
    by_name: Mapping[str, _Candidate] = {c.manifest.name: c for c in candidates}

    registry = ComponentRegistry()
    for name in ordered_names:
        candidate = by_name[name]
        try:
            candidate.plugin.register(registry.bind(name))
        except DuplicateComponentError:
            raise
        except Exception as exc:
            msg = f"plugin {name!r} failed while registering its components: {exc}"
            raise PluginLoadError(msg) from exc

    return Discovery(
        registry=registry,
        manifests=tuple(by_name[name].manifest for name in ordered_names),
        disabled=skipped,
    )


def _is_disabled(name: str, enabled: AbstractSet[str] | None, disabled: AbstractSet[str]) -> bool:
    if name in disabled:
        return True
    return enabled is not None and name not in enabled


def describe(discovery: Discovery) -> Sequence[str]:
    """Human-readable lines describing what loaded. Used by diagnostics."""
    lines = [f"{len(discovery.manifests)} plugin(s) loaded, {len(discovery.registry)} component(s)"]
    for manifest in discovery.manifests:
        lines.append(f"  {manifest.name} {manifest.version} (core {manifest.core_version})")
    for record in discovery.registry.records():
        lines.append(f"    {record.kind.value}:{record.name} from {record.plugin}")
    for name in discovery.disabled:
        lines.append(f"  {name} (disabled)")
    return lines


__all__ = [
    "ENTRY_POINT_GROUP",
    "BuildContext",
    "ComponentKey",
    "ComponentRecord",
    "ComponentRegistry",
    "ComponentResolver",
    "Discovery",
    "Factory",
    "describe",
    "discover",
    "installed_entry_points",
]
