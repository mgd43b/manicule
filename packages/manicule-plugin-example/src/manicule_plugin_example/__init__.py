"""A complete, minimal manicule plugin.

Four components, one of each of the kinds whose behavior is easiest to demonstrate
without a model or a database. Read top to bottom; it is short on purpose.

The connector is the one to copy if you are bringing a source of your own that will never be
upstreamed: ``docs/writing-a-connector.md`` walks through it.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import override

from pydantic import BaseModel, ConfigDict, Field

from manicule.container import keys
from manicule.core.anchors import Anchor, LineAnchor, Unlocated
from manicule.core.content import BlockKind, Chunk, Document, ParsedBlock, RawDocument
from manicule.core.errors import ManiculeError, ParseError
from manicule.core.lifecycle import HealthReport, Metric
from manicule.core.protocols import Connector, Middleware, Parser, RetrievalStage
from manicule.core.retrieval import Candidate, Query
from manicule.core.sources import DiscoveredDoc, DocRef, SourceId, Watermark
from manicule.plugins import BuildContext, ComponentRegistry, Plugin, PluginManifest

MEDIA_TYPE = "application/x-manicule-example"
"""A made-up format: one block per line. Real parsers claim real IANA media types."""

CONNECTOR_NAME = "example-directory"
"""What ``type = ...`` selects in ``[connectors.<name>]``. One name per implementation."""


class LineParserConfig(BaseModel):
    """Configuration for :class:`LineParser`.

    Set under ``plugins.config."parser.example"`` in the config file. Anything not declared
    here is rejected, so a typo fails at startup rather than doing nothing quietly.
    """

    skip_blank: bool = Field(default=True, description="Drop empty lines rather than emit them.")


class LineParser:
    """Parses a text document into one block per line.

    Every block gets a real :class:`~manicule.core.anchors.LineAnchor`, and
    :meth:`resolve` reads that line back out — which is what makes the round-trip check in
    :func:`manicule.testing.assert_parser_contract` meaningful rather than decorative.
    """

    media_types = frozenset({MEDIA_TYPE})

    def __init__(self, config: LineParserConfig) -> None:
        self._config = config
        self._parsed = 0

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        try:
            text = raw.as_text()
        except UnicodeDecodeError as exc:
            # Declining with ParseError lets the next parser in the chain try. Any other
            # exception would fail the document outright.
            msg = f"not decodable as {raw.encoding}: {exc}"
            raise ParseError(msg) from exc

        self._parsed += 1
        for number, line in enumerate(text.splitlines(), start=1):
            if self._config.skip_blank and not line.strip():
                continue
            yield ParsedBlock(
                kind=BlockKind.PROSE,
                text=line,
                anchor=LineAnchor(start=number, end=number),
            )

    async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
        if isinstance(anchor, Unlocated) or not isinstance(anchor, LineAnchor):
            return None
        lines = raw.as_text().splitlines()
        if anchor.end > len(lines):
            return None
        return "\n".join(lines[anchor.start - 1 : anchor.end])

    async def health(self) -> HealthReport:
        return HealthReport.healthy(f"parsed {self._parsed} document(s)")

    def metrics(self) -> tuple[Metric, ...]:
        return (Metric(name="documents_parsed", value=float(self._parsed)),)


class DirectoryConnectorError(ManiculeError):
    """What this connector raises. Subclassing the base error keeps it catchable by callers
    that handle every manicule failure, and gives a log line this plugin's name to grep for."""


class DirectoryConnectorConfig(BaseModel):
    """Configuration for :class:`DirectoryConnector`.

    Set under ``[connectors.<name>.options]`` for a ``type = "example-directory"`` source:

    .. code-block:: toml

       [connectors.notes]
       type = "example-directory"
       options = { root = "~/notes" }

    ``extra="forbid"`` is the container's default for a model that does not say otherwise;
    it is written out here so a reader sees that a misspelled option is an error, not a
    setting that appears to be in force and is not.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    root: Path = Field(
        description="The directory to index. Required: a source has to be some place."
    )
    suffix: str = Field(default=".txt", description="Only files with this suffix are documents.")


class DirectoryConnector:
    """A directory of text files, as a source.

    The whole of the :class:`~manicule.core.protocols.Connector` protocol, with every
    obligation it carries:

    * **``discover`` is incremental.** The source's own change signal here is the file
      modification time, so a sync costs what changed rather than the whole directory.
    * **The watermark advances when an enumeration finishes, and not before.** It is set on the
      last line of ``discover``. A stream abandoned after one document leaves it where it
      was, which is what :func:`manicule.testing.assert_connector_contract` checks, because the
      alternative loses documents permanently and says nothing.
    * **``reconcile`` lists what still exists.** Incremental sync cannot see a deleted file —
      it simply stops appearing — so without this the index serves it forever.
    * **Nothing is fetched to find out whether it changed.** ``version_token`` is cheap to
      compute from the directory entry, and an unchanged document is never read.
    """

    def __init__(self, name: str, config: DirectoryConnectorConfig) -> None:
        # The name is the *instance's* — the key in ``[connectors.<name>]`` — not the type's.
        # It becomes the ``source`` half of every document's identity, so two instances of
        # this type must not share one.
        self.name = name
        self._root = config.root.expanduser().resolve()
        self._suffix = config.suffix
        self._watermark: Watermark | None = None

    @property
    def watermark(self) -> Watermark | None:
        return self._watermark

    async def discover(self, watermark: Watermark | None) -> AsyncIterator[DiscoveredDoc]:
        # Blocking file-system work goes to a thread so a large directory does not stall
        # everything else in the process.
        entries = await asyncio.to_thread(self._scan)
        since = int(watermark.value) if watermark is not None else None

        newest = 0
        for source_id, path, mtime_ns, size in entries:
            newest = max(newest, mtime_ns)
            # ``>=``, not ``>``: the position is re-read from its own boundary, so a file
            # written in the same tick as the last run's newest one is not skipped. The one
            # document this re-yields costs a version-token comparison, and change detection
            # discards it. Overlapping slightly is how a watermark survives a race it cannot
            # eliminate.
            if since is not None and mtime_ns < since:
                continue
            yield DiscoveredDoc(
                ref=DocRef(source_id=source_id, uri=path.as_uri()),
                version_token=f"{mtime_ns}:{size}",
                title=path.stem,
                media_type=MEDIA_TYPE,
                size_bytes=size,
            )

        # After the loop, deliberately. An enumeration abandoned part-way never gets here.
        # A source with nothing in it offers no position rather than an invented one.
        if entries:
            self._watermark = Watermark(value=str(newest), observed_at=datetime.now(UTC))

    async def fetch(self, ref: DocRef) -> RawDocument:
        path = (self._root / ref.source_id).resolve()
        if not path.is_relative_to(self._root):
            # A source id is data handed back to us, and it is not trusted to stay inside the
            # directory this source was configured with.
            msg = f"{ref.source_id!r} is outside {self._root}"
            raise DirectoryConnectorError(msg)
        content = await asyncio.to_thread(path.read_bytes)
        return RawDocument(
            source_id=ref.source_id,
            uri=ref.uri,
            media_type=MEDIA_TYPE,
            content=content,
        )

    async def reconcile(self) -> AsyncIterator[SourceId]:
        for source_id, _path, _mtime_ns, _size in await asyncio.to_thread(self._scan):
            yield source_id

    def _scan(self) -> list[tuple[SourceId, Path, int, int]]:
        """Every matching file as ``(source id, path, mtime_ns, size)``, in a stable order.

        The source id is the path relative to the root, with ``/`` separators on every
        platform: stable across a move of the whole directory, and not a function of where this
        installation happens to keep it.
        """
        found: list[tuple[SourceId, Path, int, int]] = []
        for path in sorted(self._root.rglob(f"*{self._suffix}")):
            if not path.is_file():
                continue
            stat = path.stat()
            found.append(
                (path.relative_to(self._root).as_posix(), path, stat.st_mtime_ns, stat.st_size)
            )
        return found


class TrimMiddleware(Middleware):
    """Strips trailing whitespace from the text of every chunk before it is stored.

    Middleware is transformational: what a hook returns is what the pipeline continues
    with. Inheriting :class:`~manicule.core.protocols.Middleware` supplies pass-through
    defaults for the hooks this one does not need, so only the interesting method is here.
    """

    name = "trim"

    @override
    async def after_chunk(self, document: Document, chunks: list[Chunk]) -> list[Chunk]:
        del document
        return [
            chunk.model_copy(
                update={"text": chunk.text.rstrip(), "embed_text": chunk.embed_text.rstrip()}
            )
            for chunk in chunks
        ]


class PassthroughStage:
    """A retrieval stage that changes nothing.

    Useful as a control in evaluation runs: a pipeline with this stage added must score
    identically to one without it, and if it does not, the harness is measuring itself.
    """

    name = "passthrough"

    async def run(self, query: Query, candidates: list[Candidate]) -> list[Candidate]:
        del query
        return list(candidates)


class ExamplePlugin:
    """The plugin object the entry point resolves to."""

    manifest = PluginManifest(
        name="example",
        version="0.1.0",
        core_version=">=0.1,<0.3",
        summary="Reference plugin: one connector, one parser, one middleware, one retrieval stage.",
    )

    def register(self, registry: ComponentRegistry) -> None:
        registry.add(
            keys.CONNECTOR.named(CONNECTOR_NAME),
            _build_connector,
            config_model=DirectoryConnectorConfig,
            summary="A directory of text files, read incrementally by modification time.",
            # Not set here, because a directory of notes is not a mailbox. A connector whose
            # content is one person's passes ``sensitive=True`` and the installation then
            # refuses to be served without authentication.
        )
        registry.add(
            keys.PARSER.named("example"),
            _build_parser,
            config_model=LineParserConfig,
            summary="One block per line, with real line anchors.",
            media_types={MEDIA_TYPE},
        )
        registry.add(
            keys.MIDDLEWARE.named("trim"),
            lambda _: TrimMiddleware(),
            summary="Trims trailing whitespace from chunk text.",
        )
        registry.add(
            keys.RETRIEVAL_STAGE.named("passthrough"),
            lambda _: PassthroughStage(),
            summary="A no-op stage, for use as an evaluation control.",
        )


def _build_connector(context: BuildContext) -> DirectoryConnector:
    """Factory. ``context.instance`` is the name the user gave this source."""
    config = context.config
    if not isinstance(
        config, DirectoryConnectorConfig
    ):  # pragma: no cover - the container guarantees it
        msg = "a directory source needs a root: set [connectors.<name>.options] root"
        raise DirectoryConnectorError(msg)
    return DirectoryConnector(context.instance, config)


def _build_parser(context: BuildContext) -> LineParser:
    """Factory. Heavy imports would go here, not at module level."""
    config = context.config
    if not isinstance(config, LineParserConfig):  # pragma: no cover - the container guarantees it
        config = LineParserConfig()
    return LineParser(config)


PLUGIN = ExamplePlugin()

# Checked when this file is type-checked, so the example cannot drift out of conformance
# with the protocols it exists to demonstrate.
_plugin: Plugin = PLUGIN
_connector: Connector = DirectoryConnector("example", DirectoryConnectorConfig(root=Path()))
_parser: Parser = LineParser(LineParserConfig())
_middleware: Middleware = TrimMiddleware()
_stage: RetrievalStage = PassthroughStage()

__all__ = [
    "CONNECTOR_NAME",
    "MEDIA_TYPE",
    "PLUGIN",
    "DirectoryConnector",
    "DirectoryConnectorConfig",
    "DirectoryConnectorError",
    "ExamplePlugin",
    "LineParser",
    "LineParserConfig",
    "PassthroughStage",
    "TrimMiddleware",
]
