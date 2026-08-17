"""The built-in parsing plugin: every parser, and the chunker.

Registered through the public ``manicule.plugins`` entry point, exactly as a third-party
plugin is. There is no shorter internal route, so the extension mechanism is exercised by
every installation rather than only by the people extending it, and it cannot rot unnoticed.

**Nothing here imports a parsing library.** Registration needs two things eagerly — the media
types a parser claims, so a document can be routed without building every installed parser,
and the configuration model, so settings written for a parser are validated rather than
silently ignored. Both live in :mod:`manicule.parsers.config`, which imports nothing heavier
than pydantic. The parser module itself, and the C extension behind it, are imported inside
the factory: an installed plugin nobody has configured costs one cheap import.
"""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING

from pydantic import BaseModel

from manicule.container import keys
from manicule.core.errors import ConfigError, UnknownComponentError
from manicule.parsers import config as parser_config
from manicule.plugins import BuildContext, ComponentRegistry, Plugin, PluginManifest
from manicule.plugins.registry import Factory

if TYPE_CHECKING:
    from manicule.chunking import StructuralChunker
    from manicule.config.settings import Settings
    from manicule.core.fingerprints import ChunkFingerprint
    from manicule.core.protocols import Parser
    from manicule.plugins.registry import MetadataContext

CHUNKER_NAME = "structural"

SOURCE_CODE_NAME = "sourcecode"
"""The registered name of the code parser.

Named once because two things reach for it: the registration below, and the chunker factory,
which reads the code parser's declared language set out of configuration in order to record a
grammar version per language. Two spellings of one name would make the fingerprint silently
fall back to the default language set the moment somebody renamed the parser.
"""


@dataclass(frozen=True, slots=True)
class _Registration:
    """One parser, described without importing it."""

    name: str
    module: str
    factory: str
    config_model: type[BaseModel]
    media_types: AbstractSet[str]
    summary: str


PARSERS: tuple[_Registration, ...] = (
    _Registration(
        name="pdf",
        module="manicule.parsers.pdf",
        factory="PdfParser",
        config_model=parser_config.PdfConfig,
        media_types=parser_config.PDF_MEDIA_TYPES,
        summary="Page and rectangle provenance from pdfium character boxes.",
    ),
    _Registration(
        name=SOURCE_CODE_NAME,
        module="manicule.parsers.sourcecode",
        factory="SourceCodeParser",
        config_model=parser_config.SourceCodeConfig,
        media_types=parser_config.SOURCE_CODE_MEDIA_TYPES,
        summary="Line anchors and symbols from tree-sitter parse trees.",
    ),
    _Registration(
        name="markdown",
        module="manicule.parsers.markdown",
        factory="MarkdownParser",
        config_model=parser_config.MarkdownConfig,
        media_types=parser_config.MARKDOWN_MEDIA_TYPES,
        summary="Heading anchors with synthesized fragments, from markdown-it token maps.",
    ),
    _Registration(
        name="html",
        module="manicule.parsers.web",
        factory="WebParser",
        config_model=parser_config.WebConfig,
        media_types=parser_config.WEB_MEDIA_TYPES,
        summary="Heading anchors, deep-linking only where the author supplied an id.",
    ),
    _Registration(
        name="adf",
        module="manicule.parsers.adf",
        factory="ADFParser",
        config_model=parser_config.ADFConfig,
        media_types=parser_config.ADF_MEDIA_TYPES,
        summary="Confluence Atlassian Document Format, with the source's own anchors.",
    ),
    _Registration(
        name="confluence",
        module="manicule.parsers.confluence",
        factory="ConfluenceStorageParser",
        config_model=parser_config.ConfluenceConfig,
        media_types=parser_config.CONFLUENCE_MEDIA_TYPES,
        summary="Confluence storage format: macros, panels and tasks read as Confluence.",
    ),
    _Registration(
        name="docx",
        module="manicule.parsers.word",
        factory="WordParser",
        config_model=parser_config.WordConfig,
        media_types=parser_config.WORD_MEDIA_TYPES,
        summary="Section anchors from paragraph styles. Never a page number.",
    ),
    _Registration(
        name="pptx",
        module="manicule.parsers.slides",
        factory="SlidesParser",
        config_model=parser_config.SlidesConfig,
        media_types=parser_config.SLIDES_MEDIA_TYPES,
        summary="Slide numbers and shape geometry as normalized rectangles.",
    ),
    _Registration(
        name="spreadsheet",
        module="manicule.parsers.spreadsheet",
        factory="SpreadsheetParser",
        config_model=parser_config.SpreadsheetConfig,
        media_types=parser_config.SPREADSHEET_MEDIA_TYPES,
        summary="Cell anchors for XLSX and CSV alike, once a CSV is given a sheet name.",
    ),
    _Registration(
        name="notebook",
        module="manicule.parsers.notebook",
        factory="NotebookParser",
        config_model=parser_config.NotebookConfig,
        media_types=parser_config.NOTEBOOK_MEDIA_TYPES,
        summary="Heading anchors fragmented by nbformat cell id.",
    ),
    _Registration(
        name="email",
        module="manicule.parsers.mail",
        factory="MailParser",
        config_model=parser_config.MailConfig,
        media_types=parser_config.MAIL_MEDIA_TYPES,
        summary="Line anchors within the canonical body part, headers included.",
    ),
    _Registration(
        name="plaintext",
        module="manicule.parsers.plaintext",
        factory="PlaintextParser",
        config_model=parser_config.PlaintextConfig,
        media_types=parser_config.PLAINTEXT_MEDIA_TYPES,
        summary="Source line anchors, and the refusal that makes the global tail safe.",
    ),
    _Registration(
        name="structured",
        module="manicule.parsers.structured",
        factory="StructuredParser",
        config_model=parser_config.StructuredConfig,
        media_types=parser_config.STRUCTURED_MEDIA_TYPES,
        summary="Line spans and key paths for JSON, YAML and TOML.",
    ),
    _Registration(
        name="archive",
        module="manicule.parsers.archive",
        factory="ArchiveParser",
        config_model=parser_config.ArchiveConfig,
        media_types=parser_config.ARCHIVE_MEDIA_TYPES,
        summary="Expands members into documents of their own. Emits no chunks.",
    ),
)


def _build_parser(registration: _Registration, context: BuildContext) -> Parser:
    """Import the parser's module and construct it.

    The import happens here rather than at module level so that installing the parsing
    plugin costs nothing until a document of that type is routed to it — which for a corpus
    of Markdown means never loading pdfium, tree-sitter, or the Office readers at all.

    Raises:
        ConfigError: The context carries configuration of some other type. The container
            validates against the registered model before calling a factory, so this can only
            be reached by a caller building the parser some other way — and substituting
            defaults there would build a parser whose settings appear to be in force and are
            not, which is the failure configuration validation exists to prevent.
    """
    module = import_module(registration.module)
    parser_type = getattr(module, registration.factory)
    settings = context.config
    if not isinstance(settings, registration.config_model):
        msg = (
            f"parser {registration.name!r} was built with {type(settings).__name__} where it "
            f"declares {registration.config_model.__name__}. Configuration reaching a factory "
            f"is validated against the model the component registered; a factory called "
            f"outside the container has to supply that model itself."
        )
        raise ConfigError(msg)
    built: Parser = parser_type(settings)
    return built


def _build_chunker(context: BuildContext) -> StructuralChunker:
    """Construct the chunker, bound to the embedder when one is configured.

    The embedder is a **construction** dependency rather than something looked up later,
    because the budget check has to happen before ingest: past a model's sequence length the
    input is truncated with no error raised, and the chunk is indexed as its opening tokens
    while still claiming all of its text.

    When no embedder is installed — a parse-only run, a fixture build — the chunker counts
    with a stand-in vocabulary, inflates the result, and marks its chunks provisional so that
    ingest refuses them. That is a narrower thing than falling back: the chunks exist, they
    are usable for inspection, and they cannot reach an index.
    """
    from manicule.chunking import StructuralChunker, TokenCounter  # noqa: PLC0415 - see docstring
    from manicule.chunking.tokens import SupportsTokenCount  # noqa: PLC0415

    policy = context.config
    if not isinstance(policy, parser_config.StructuralChunkerConfig):
        msg = (
            f"structural chunker was built with {type(policy).__name__} where it declares "
            f"{parser_config.StructuralChunkerConfig.__name__}"
        )
        raise ConfigError(msg)

    embedder: SupportsTokenCount | None = None
    try:
        candidate = context.components.get(keys.EMBEDDER)
    except UnknownComponentError:
        candidate = None
    if isinstance(candidate, SupportsTokenCount):
        embedder = candidate

    counter = (
        TokenCounter.bound_to(embedder) if embedder is not None else TokenCounter.provisionally()
    )
    return StructuralChunker(
        counter,
        embedder=embedder,
        max_tokens=policy.max_tokens,
        overlap_tokens=policy.overlap_tokens,
        grammars=_grammar_versions(context.settings),
        version_components=_pinned_versions(),
    )


def _grammar_versions(settings: Settings) -> dict[str, str]:
    """Grammar version by language, for ``ChunkFingerprint.grammars``.

    Read from **configuration**, not from the cache and not from a constructed parser.

    Not the cache, because a map that shrank when a grammar was missing would make the
    fingerprint depend on cache state — a freshly installed machine and a warmed one would
    declare their corpora incompatible for no reason at all, and an air-gapped install with no
    route to the grammar mirror could not build a chunker at all. Recording a grammar version
    must never require the network, and here it does not: the language set is configuration and
    the version is distribution metadata.

    Not a constructed parser, because building the code parser would also configure the
    grammar pack's process-global registry — pointing it at a cache directory and a manifest
    mirror — as a side effect of asking a question about configuration.

    The declared set is validated here rather than trusted. That does import the grammar pack,
    to check the keys against the names it ships; validating late instead would let a typo
    become a fingerprint recording a language that does not exist, on a corpus that happens to
    contain no code and so never builds the parser that would have caught it.
    """
    from manicule.parsers.config import SourceCodeConfig  # noqa: PLC0415 - see module docstring
    from manicule.parsers.grammars import grammar_versions  # noqa: PLC0415

    declared = settings.component_config("parser", SOURCE_CODE_NAME)
    config = SourceCodeConfig.model_validate(dict(declared))
    return dict(grammar_versions(config.languages))


def _pinned_versions() -> dict[str, str]:
    """Other pinned transformations whose output an anchor addresses.

    The HTML-to-text conversion is the one that matters: an email with an HTML-only body has
    line numbers into the *converted* text, so a converter upgrade would shift every anchor
    in every HTML email — round-tripping today and pointing at the wrong paragraph after a
    dependency bump, with no test failing in between.
    """
    return {"html_text": parser_config.html_text_version()}


def _chunker_metadata(context: MetadataContext) -> ChunkFingerprint:
    """The structural chunk identity, derived without constructing its embedder dependency."""
    from manicule.chunking.chunker import (  # noqa: PLC0415
        CHUNKER_NAME as FINGERPRINT_NAME,
    )
    from manicule.chunking.chunker import CHUNKER_VERSION  # noqa: PLC0415
    from manicule.core.embedding import EmbedFingerprint  # noqa: PLC0415
    from manicule.core.fingerprints import ChunkFingerprint  # noqa: PLC0415

    embedding = context.components.get(keys.EMBEDDER)
    if not isinstance(embedding, EmbedFingerprint):
        raise ConfigError("embedder metadata did not declare an embedding fingerprint")
    policy = context.config
    if not isinstance(policy, parser_config.StructuralChunkerConfig):
        raise ConfigError("structural chunker metadata received invalid component configuration")
    components = _pinned_versions()
    suffix = "".join(f";{name}={value}" for name, value in sorted(components.items()))
    return ChunkFingerprint(
        chunker=FINGERPRINT_NAME,
        version=f"{CHUNKER_VERSION}{suffix}",
        max_tokens=policy.max_tokens,
        overlap_tokens=policy.overlap_tokens,
        tokenizer_id=embedding.tokenizer_id,
        grammars=_grammar_versions(context.settings),
    )


class ParsingPlugin:
    """The plugin object the ``parsing`` entry point resolves to."""

    manifest = PluginManifest(
        name="parsing",
        version="0.1.0",
        core_version=">=0.1,<0.2",
        summary="Every built-in parser, and the structure-aware chunker.",
    )

    def register(self, registry: ComponentRegistry) -> None:
        for registration in PARSERS:
            registry.add(
                keys.PARSER.named(registration.name),
                _factory_for(registration),
                config_model=registration.config_model,
                summary=registration.summary,
                media_types=registration.media_types,
            )
        registry.add(
            keys.CHUNKER.named(CHUNKER_NAME),
            _build_chunker,
            config_model=parser_config.StructuralChunkerConfig,
            metadata_factory=_chunker_metadata,
            summary="Configurable final embed_text budget and overlap; structural boundaries.",
        )


def _factory_for(registration: _Registration) -> Factory[Parser]:
    """Bind one registration into a factory the container can call.

    A closure rather than a `partial`, so the registration it captures is visible in a
    traceback when a parser's own import fails.
    """

    def factory(context: BuildContext) -> Parser:
        return _build_parser(registration, context)

    return factory


PLUGIN = ParsingPlugin()

# Checked when this file is type-checked, so the plugin cannot drift out of conformance with
# the protocol every installation loads it through.
_plugin: Plugin = PLUGIN

__all__ = ["CHUNKER_NAME", "PARSERS", "PLUGIN", "SOURCE_CODE_NAME", "ParsingPlugin"]
