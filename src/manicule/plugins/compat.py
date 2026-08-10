"""Compatibility checking.

A plugin declares the core versions it works with. When that declaration is not satisfied,
manicule refuses to load it and says so — loudly, at startup, naming the plugin, what it
asked for and what it got. The alternative is loading it anyway and discovering the
incompatibility as an ``AttributeError`` somewhere unrelated, hours later.

The version compared against is read from installed distribution metadata, so it cannot
drift from the version actually running.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from manicule.core.errors import IncompatiblePluginError, PluginDependencyError
from manicule.plugins.manifest import PluginManifest


def check_core_version(manifest: PluginManifest, core_version: str) -> str | None:
    """Return a problem description, or ``None`` when the plugin supports this core.

    Pre-release versions of core satisfy a specifier they would otherwise fail, so that
    development builds can exercise plugins written against the coming release.
    """
    try:
        specifier = SpecifierSet(manifest.core_version)
    except InvalidSpecifier:
        return (
            f"declares core_version {manifest.core_version!r}, which is not a valid PEP 440 "
            f"specifier (for example: '>=0.1,<0.2')"
        )
    try:
        version = Version(core_version)
    except InvalidVersion:  # pragma: no cover - only reachable with corrupt metadata
        return f"cannot compare against core version {core_version!r}, which is not valid PEP 440"

    if specifier.contains(version, prereleases=True):
        return None
    return (
        f"requires manicule {manifest.core_version}, but {core_version} is running. "
        f"Upgrade the plugin, or install a manicule version it supports."
    )


def check_dependencies(manifest: PluginManifest, installed: AbstractSet[str]) -> list[str]:
    """Return problem descriptions for unmet requirements and present conflicts.

    ``installed`` is every discovered plugin name, not merely those registered so far.
    Checking against the full set is what makes the outcome independent of load order.
    """
    problems: list[str] = []
    for required in manifest.requires:
        if required not in installed:
            problems.append(f"requires plugin {required!r}, which is not installed")
    for conflicting in manifest.conflicts:
        if conflicting in installed:
            problems.append(f"conflicts with plugin {conflicting!r}, which is installed")
    return problems


def require_compatible(
    manifest: PluginManifest, core_version: str, installed: AbstractSet[str]
) -> None:
    """Raise unless ``manifest`` can be loaded against this core and this plugin set.

    Raises:
        IncompatiblePluginError: The declared core version range is not satisfied.
        PluginDependencyError: A requirement is missing or a conflict is present.
    """
    problem = check_core_version(manifest, core_version)
    if problem is not None:
        raise IncompatiblePluginError(f"plugin {manifest.name!r} {problem}")

    problems = check_dependencies(manifest, installed)
    if problems:
        raise PluginDependencyError(f"plugin {manifest.name!r} " + "; ".join(problems))


def load_order(manifests: Iterable[PluginManifest]) -> list[str]:
    """Order plugin names so that every plugin follows the ones it requires.

    Ties are broken alphabetically, so the order is the same on every machine and a
    reproducible startup stays reproducible.

    Raises:
        PluginDependencyError: The requirement graph contains a cycle.
    """
    by_name: Mapping[str, PluginManifest] = {m.name: m for m in manifests}
    ordered: list[str] = []
    placed: set[str] = set()
    in_progress: list[str] = []

    def visit(name: str) -> None:
        if name in placed:
            return
        if name in in_progress:
            cycle = " -> ".join([*in_progress[in_progress.index(name) :], name])
            msg = f"plugin requirements form a cycle: {cycle}"
            raise PluginDependencyError(msg)
        manifest = by_name.get(name)
        if manifest is None:
            # Missing requirements are reported by check_dependencies with a better message.
            return
        in_progress.append(name)
        for required in sorted(manifest.requires):
            visit(required)
        in_progress.pop()
        placed.add(name)
        ordered.append(name)

    for name in sorted(by_name):
        visit(name)
    return ordered


def sort_by_load_order(manifests: Sequence[PluginManifest]) -> list[PluginManifest]:
    """``manifests`` reordered so requirements come first."""
    by_name = {m.name: m for m in manifests}
    return [by_name[name] for name in load_order(manifests)]


__all__ = [
    "check_core_version",
    "check_dependencies",
    "load_order",
    "require_compatible",
    "sort_by_load_order",
]
