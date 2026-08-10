"""The plugin system: manifests, compatibility checking and entry-point discovery."""

from __future__ import annotations

from manicule.plugins.compat import (
    check_core_version,
    check_dependencies,
    load_order,
    require_compatible,
    sort_by_load_order,
)
from manicule.plugins.manifest import ComponentKind, Plugin, PluginManifest
from manicule.plugins.registry import (
    ENTRY_POINT_GROUP,
    BuildContext,
    ComponentKey,
    ComponentRecord,
    ComponentRegistry,
    ComponentResolver,
    Discovery,
    Factory,
    describe,
    discover,
    installed_entry_points,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "BuildContext",
    "ComponentKey",
    "ComponentKind",
    "ComponentRecord",
    "ComponentRegistry",
    "ComponentResolver",
    "Discovery",
    "Factory",
    "Plugin",
    "PluginManifest",
    "check_core_version",
    "check_dependencies",
    "describe",
    "discover",
    "installed_entry_points",
    "load_order",
    "require_compatible",
    "sort_by_load_order",
]
