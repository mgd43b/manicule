"""A complete, minimal manicule plugin.

Three components, one of each of the kinds whose behaviour is easiest to demonstrate
without a model or a database. Read top to bottom; it is short on purpose.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import override

from pydantic import BaseModel, Field

from manicule.container import keys
from manicule.core.anchors import Anchor, LineAnchor, Unlocated
from manicule.core.content import BlockKind, Chunk, Document, ParsedBlock, RawDocument
from manicule.core.errors import ParseError
from manicule.core.lifecycle import HealthReport, Metric
from manicule.core.protocols import Middleware, Parser, RetrievalStage
from manicule.core.retrieval import Candidate, Query
from manicule.plugins import BuildContext, ComponentRegistry, Plugin, PluginManifest

MEDIA_TYPE = "application/x-manicule-example"
"""A made-up format: one block per line. Real parsers claim real IANA media types."""


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
        core_version=">=0.1,<0.2",
        summary="Reference plugin: one parser, one middleware, one retrieval stage.",
    )

    def register(self, registry: ComponentRegistry) -> None:
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
_parser: Parser = LineParser(LineParserConfig())
_middleware: Middleware = TrimMiddleware()
_stage: RetrievalStage = PassthroughStage()

__all__ = [
    "MEDIA_TYPE",
    "PLUGIN",
    "ExamplePlugin",
    "LineParser",
    "LineParserConfig",
    "PassthroughStage",
    "TrimMiddleware",
]
