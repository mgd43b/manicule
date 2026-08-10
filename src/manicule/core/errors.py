"""The error hierarchy.

Every failure manicule raises on purpose descends from :class:`ManiculeError`, so a caller
can distinguish "this system said no" from "something unexpected broke".
"""

from __future__ import annotations


class ManiculeError(Exception):
    """Base class for every error manicule raises deliberately."""


# --- configuration ---------------------------------------------------------------------


class ConfigError(ManiculeError):
    """Configuration could not be loaded, parsed, or validated."""


class PolicyError(ConfigError):
    """A configuration is internally valid but violates a declared policy.

    Raised before any resource is constructed, so that an impossible combination fails at
    startup rather than at the first request that happens to exercise it.
    """


# --- plugins ---------------------------------------------------------------------------


class PluginError(ManiculeError):
    """Base class for plugin discovery, validation and registration failures."""


class PluginLoadError(PluginError):
    """An entry point could not be imported, or did not resolve to a plugin."""


class IncompatiblePluginError(PluginError):
    """A plugin declares a core version range that the running core does not satisfy."""


class PluginDependencyError(PluginError):
    """A plugin's declared requirements are unmet, or a declared conflict is present."""


class DuplicateComponentError(PluginError):
    """Two plugins registered the same component kind under the same name."""


# --- container -------------------------------------------------------------------------


class ContainerError(ManiculeError):
    """Base class for wiring failures."""


class UnknownComponentError(ContainerError):
    """A component was requested that no installed plugin provides."""


class CircularDependencyError(ContainerError):
    """Resolving a component required resolving itself."""


class ContainerStateError(ContainerError):
    """The container was used outside the lifecycle window it supports."""


# --- pipeline --------------------------------------------------------------------------


class ParseError(ManiculeError):
    """A parser could not handle a document it nominally supports.

    Raising this is how a parser declines a document so that the next parser in the
    fallback chain gets a turn. Any other exception aborts the document.
    """


class FingerprintMismatchError(ManiculeError):
    """An embedding fingerprint does not match the one an index was built with.

    Vectors produced by different models are not comparable, so continuing would silently
    destroy retrieval quality. This is always fatal to the operation that raised it.
    """


class ChunkingError(ManiculeError):
    """A chunker could not produce chunks from a document's blocks."""


__all__ = [
    "ChunkingError",
    "CircularDependencyError",
    "ConfigError",
    "ContainerError",
    "ContainerStateError",
    "DuplicateComponentError",
    "FingerprintMismatchError",
    "IncompatiblePluginError",
    "ManiculeError",
    "ParseError",
    "PluginDependencyError",
    "PluginError",
    "PluginLoadError",
    "PolicyError",
    "UnknownComponentError",
]
