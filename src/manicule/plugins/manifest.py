"""What a plugin declares about itself.

A plugin is any installed distribution that advertises an entry point in the
``manicule.plugins`` group. Built-in components use that same path — there is no privileged
internal register — so the public extension mechanism is the one manicule itself depends on,
and it cannot quietly stop working while everything still runs.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from manicule.plugins.registry import ComponentRegistry


class ComponentKind(StrEnum):
    """The kinds of component a plugin can provide.

    One per extension point. A plugin may provide any number of components of any number of
    kinds — a connector package that also ships the parser for its native format is one
    distribution, not two.
    """

    PARSER = "parser"
    CHUNKER = "chunker"
    EMBEDDER = "embedder"
    VECTOR_STORE = "vector_store"
    DOC_STORE = "doc_store"
    RETRIEVAL_STAGE = "retrieval_stage"
    RERANKER = "reranker"
    GENERATOR = "generator"
    CONNECTOR = "connector"
    MIDDLEWARE = "middleware"


class PluginManifest(BaseModel):
    """A plugin's self-description.

    Every field here is read by something. A field that nothing consumes is a promise to the
    reader that the system does not keep, so this stays small on purpose.

    **There is no ``permissions`` field.** Plugins are imported into the host process and run
    with its full authority: the network, the filesystem, the environment, everything. A
    permissions declaration would describe a boundary that does not exist, and a guarantee
    nothing enforces is worse than an absent one, because it gets believed. Install plugins
    you would run as yourself, because that is what happens.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(
        min_length=1,
        description="Identifier used in configuration, diagnostics and dependency "
        "declarations. Must match the entry-point name, so that what configuration calls a "
        "plugin and what the plugin calls itself cannot drift apart.",
    )
    version: str = Field(min_length=1)
    core_version: str = Field(
        min_length=1,
        description="PEP 440 specifier for the manicule versions this supports, e.g. "
        "``>=0.1,<0.2``. Checked before the plugin registers anything.",
    )
    summary: str = ""
    requires: tuple[str, ...] = Field(
        default=(),
        description="Names of plugins that must also be installed. Checked against the whole "
        "discovered set, so declaration order does not matter.",
    )
    conflicts: tuple[str, ...] = Field(
        default=(),
        description="Names of plugins that must not be installed alongside this one.",
    )

    @field_validator("requires", "conflicts")
    @classmethod
    def _no_self_reference(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            msg = "duplicate plugin names in requires/conflicts"
            raise ValueError(msg)
        return value


@runtime_checkable
class Plugin(Protocol):
    """What an entry point in the ``manicule.plugins`` group must resolve to.

    The entry point may name a :class:`Plugin` instance directly, or a zero-argument
    callable returning one.
    """

    manifest: PluginManifest

    def register(self, registry: ComponentRegistry) -> None:
        """Declare this plugin's components.

        Called once, after the compatibility check passes. Register *factories*, not
        instances: nothing is constructed until configuration asks for it, and the module
        top level stays cheap to import. Heavy imports belong inside the factory, so
        installing a plugin costs nothing until it is used.
        """
        ...


__all__ = ["ComponentKind", "Plugin", "PluginManifest"]
