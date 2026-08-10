"""Parsers that hang, allocate and die, so that a claim about isolation can be checked.

``try``/``except`` covers one of the three ways a document takes down an ingest run. The other
two are not exceptions at all: a parser that hangs never returns to be caught, and a parser
that exhausts memory takes the process with it. Neither can be demonstrated with a
well-behaved parser, and a pipeline whose entire purpose is surviving them is certified by
nothing if they are never made to happen.

So they happen here, in a real plugin, registered through the entry-point group the parse
workers actually discover.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator

from pydantic import BaseModel, Field

from manicule.container import keys
from manicule.core.anchors import Anchor, Unlocated
from manicule.core.content import BlockKind, ParsedBlock, RawDocument
from manicule.core.protocols import Parser
from manicule.plugins import BuildContext, ComponentRegistry, Plugin, PluginManifest

HANGING_MEDIA_TYPE = "text/x-hangs"
GREEDY_MEDIA_TYPE = "text/x-greedy"
CRASHING_MEDIA_TYPE = "text/x-crashes"


class HostileConfig(BaseModel):
    """How far each parser goes before the pipeline stops it."""

    hang_seconds: float = Field(default=3600.0, gt=0)
    chunk_megabytes: int = Field(default=32, ge=1)


class HangingParser:
    """Blocks, and keeps blocking.

    ``time.sleep`` rather than an ``await``: the point is a call that does not yield to the
    event loop, which is what a parser inside a C extension looks like from the outside. A
    timeout written in Python cannot end this. Killing the process can.
    """

    media_types = frozenset({HANGING_MEDIA_TYPE})

    def __init__(self, config: HostileConfig) -> None:
        self._config = config

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        del raw
        # ASYNC251 is right about every other async function in this repository, and wrong
        # about this one: a blocking call that never yields to the event loop is precisely
        # what is being reproduced. An `await asyncio.sleep` would be cancellable, which is
        # the property a parser inside a C extension does not have.
        time.sleep(self._config.hang_seconds)  # noqa: ASYNC251
        yield ParsedBlock(  # pragma: no cover - unreachable in any test that finishes
            kind=BlockKind.PROSE, text="never", anchor=Unlocated(reason="never produced")
        )

    async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
        del anchor, raw
        return None


class GreedyParser:
    """Allocates until something stops it.

    Touches every page it asks for, because resident memory is the quantity the bound is
    about: a reservation nobody reads is not what takes a machine down.
    """

    media_types = frozenset({GREEDY_MEDIA_TYPE})

    def __init__(self, config: HostileConfig) -> None:
        self._config = config

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        del raw
        block = self._config.chunk_megabytes * 1024 * 1024
        held: list[bytearray] = []
        while True:
            grown = bytearray(block)
            grown[::4096] = b"\x01" * len(grown[::4096])
            held.append(grown)
        yield ParsedBlock(  # pragma: no cover - unreachable by construction
            kind=BlockKind.PROSE, text="never", anchor=Unlocated(reason="never produced")
        )

    async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
        del anchor, raw
        return None


class CrashingParser:
    """Ends the interpreter mid-parse, the way a segfault in a native extension does.

    The parent sees a closed pipe rather than an exception, and must attribute that to the
    document it dispatched rather than treating it as the end of the run.
    """

    media_types = frozenset({CRASHING_MEDIA_TYPE})

    def __init__(self, config: HostileConfig) -> None:
        self._config = config

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        del raw
        os._exit(1)  # a crash, deliberately, with no unwinding and no exception to catch
        yield ParsedBlock(  # pragma: no cover - unreachable by construction
            kind=BlockKind.PROSE, text="never", anchor=Unlocated(reason="never produced")
        )

    async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
        del anchor, raw
        return None


class HostilePlugin:
    """The plugin object the entry point resolves to."""

    manifest = PluginManifest(
        name="hostile",
        version="0.1.0",
        core_version=">=0.1,<0.2",
        summary="Parsers that misbehave on purpose, so isolation can be proven.",
    )

    def register(self, registry: ComponentRegistry) -> None:
        registry.add(
            keys.PARSER.named("hanging"),
            lambda context: HangingParser(_config(context)),
            config_model=HostileConfig,
            summary="Blocks forever, in a call no timeout written in Python can end.",
            media_types={HANGING_MEDIA_TYPE},
        )
        registry.add(
            keys.PARSER.named("greedy"),
            lambda context: GreedyParser(_config(context)),
            config_model=HostileConfig,
            summary="Allocates and touches memory until something stops it.",
            media_types={GREEDY_MEDIA_TYPE},
        )
        registry.add(
            keys.PARSER.named("crashing"),
            lambda context: CrashingParser(_config(context)),
            config_model=HostileConfig,
            summary="Ends the interpreter mid-parse, the way a native crash does.",
            media_types={CRASHING_MEDIA_TYPE},
        )


def _config(context: BuildContext) -> HostileConfig:
    config = context.config
    return config if isinstance(config, HostileConfig) else HostileConfig()


PLUGIN = HostilePlugin()

_plugin: Plugin = PLUGIN
_hanging: Parser = HangingParser(HostileConfig())
_greedy: Parser = GreedyParser(HostileConfig())
_crashing: Parser = CrashingParser(HostileConfig())

__all__ = [
    "CRASHING_MEDIA_TYPE",
    "GREEDY_MEDIA_TYPE",
    "HANGING_MEDIA_TYPE",
    "PLUGIN",
    "CrashingParser",
    "GreedyParser",
    "HangingParser",
    "HostileConfig",
    "HostilePlugin",
]
