"""What registration needs to know about a parser, without importing one.

Registering the parsing plugin needs exactly two things about each parser eagerly: **the
media types it claims**, so a document can be routed without constructing every installed
parser, and **its configuration model**, so settings written for it are validated rather than
silently ignored. Everything else — the parse itself, and the library behind it — is imported
inside the factory that builds the parser.

Both of those things live here, and this module imports nothing heavier than pydantic. That
is the whole point of it. If the media types stayed beside their parsers, importing the
registration table would import pdfium, tree-sitter, python-docx, python-pptx, selectolax,
nbformat, calamine and ruamel.yaml — every one of them, at startup, on an installation whose
corpus is entirely Markdown. ``tests/test_import_boundary.py`` fails the build if that
regresses, and it is worth stating why the cheap thing is not merely nicer: plugin discovery
runs before configuration is read, so the cost is paid by every process that starts, including
``manicule doctor`` on a machine that is not going to parse anything at all.

One deliberate exception, stated so the claim above stays exactly true: ``doctor``'s
``grammars`` check *does* load the grammar pack, because "are the declared grammars on this
machine" cannot be answered without asking the thing that keeps them. That is one native
extension and a directory listing, it happens inside the check rather than during registration,
and it is the only parsing library any diagnostic touches.

Two consequences worth knowing before adding to this file.

- **No module-level lookup of an installed package.** The parsing extras are optional, so an
  install without them still advertises this plugin's entry point and still imports this
  module. A version read at import time would turn a missing optional dependency into a
  crash during discovery instead of a clear refusal at construction. Where a version is part
  of an anchor's identity — see :func:`html_text_version` — it is a function, called when the
  parser that depends on it is built.
- **Configuration models are data, not behavior**, with one deliberate exception:
  :class:`WordConfig` carries the two small style-matching helpers, because they are pure
  functions of the configured values and belong with the fields they interpret rather than
  with the reader that calls them.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from manicule.parsers.expansion import MAX_DEPTH
from manicule.parsers.grammars import DECLARED_LANGUAGES
from manicule.parsers.grammars import MEDIA_TYPES as GRAMMAR_MEDIA_TYPES

__all__ = [
    "ADF_MEDIA_TYPE",
    "ADF_MEDIA_TYPES",
    "ARCHIVE_MEDIA_TYPES",
    "CONFLUENCE_MEDIA_TYPE",
    "CONFLUENCE_MEDIA_TYPES",
    "CSV_MEDIA_TYPE",
    "MAIL_MEDIA_TYPES",
    "MARKDOWN_MEDIA_TYPES",
    "NOTEBOOK_MEDIA_TYPE",
    "NOTEBOOK_MEDIA_TYPES",
    "PDF_MEDIA_TYPES",
    "PLAINTEXT_MEDIA_TYPES",
    "SLIDES_MEDIA_TYPE",
    "SLIDES_MEDIA_TYPES",
    "SOURCE_CODE_MEDIA_TYPES",
    "SPREADSHEET_MEDIA_TYPES",
    "STRUCTURED_MEDIA_TYPES",
    "WEB_MEDIA_TYPES",
    "WORD_MEDIA_TYPE",
    "WORD_MEDIA_TYPES",
    "XLSX_MEDIA_TYPE",
    "ADFConfig",
    "ArchiveConfig",
    "ConfluenceConfig",
    "MailConfig",
    "MarkdownConfig",
    "NotebookConfig",
    "PdfConfig",
    "PlaintextConfig",
    "SlidesConfig",
    "SourceCodeConfig",
    "SpreadsheetConfig",
    "StructuredConfig",
    "WebConfig",
    "WordConfig",
    "html_text_version",
]

# --- media types -------------------------------------------------------------------------
#
# What routes to each parser. Declared narrowly and by name: no parser claims a wildcard, so
# the global fallback tail is a configuration decision rather than a parser winning documents
# a specialized one was installed to handle.

PDF_MEDIA_TYPES = frozenset({"application/pdf"})

SOURCE_CODE_MEDIA_TYPES = frozenset(GRAMMAR_MEDIA_TYPES.values())
"""One media type per declared language. Derived from the language table rather than written
out again, so a language added there cannot arrive with nothing routing to it."""

MARKDOWN_MEDIA_TYPES = frozenset({"text/markdown", "text/x-markdown", "text/mdx"})
"""``text/mdx`` is unregistered but is what the extension map resolves ``.mdx`` to, and MDX is
Markdown with components rather than a format of its own."""

WEB_MEDIA_TYPES = frozenset({"text/html", "application/xhtml+xml"})

ADF_MEDIA_TYPE = "application/json;profile=atlas-doc-format"
"""ADF is not a file type. It is the body format the Confluence Cloud API returns, and it
registers under the profile parameter the API itself uses."""

ADF_MEDIA_TYPES = frozenset({ADF_MEDIA_TYPE})

CONFLUENCE_MEDIA_TYPE = "application/xhtml+xml;profile=confluence-storage"
"""Storage format is not a file type either, and it is not quite XHTML.

The base type is honest — a storage-format body really is XHTML, and reading it with an HTML
engine is the right way to read it — but the profile is what makes it addressable. Without one
there is no way to say "this is Confluence" at routing time, so ``ac:structured-macro`` reaches
a parser with no vocabulary for it and a code macro's language, a panel's severity and a task's
state are flattened to prose. The profile parameter is the device :data:`ADF_MEDIA_TYPE`
already uses for the other Confluence body format, for the same reason.

Deliberately **not** a bare ``application/xhtml+xml``, which :data:`WEB_MEDIA_TYPES` already
claims: resolution is by exact media type, so a bare one would route to the HTML parser and the
distinction this exists to draw would silently not happen.

**A convention of this project, not a registered type.** Atlassian publishes no IANA media type
for storage format, so this string was coined here, exactly as :data:`ADF_MEDIA_TYPE` was for the
other body format. It is stable within manicule and meaningless outside it — recorded because a
convention that reads like a standard is one somebody later cites as though it were, and
``docs/parsing.md`` §2.4 says the same thing where the type is registered."""

CONFLUENCE_MEDIA_TYPES = frozenset({CONFLUENCE_MEDIA_TYPE})

WORD_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

WORD_MEDIA_TYPES = frozenset({WORD_MEDIA_TYPE})

SLIDES_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

SLIDES_MEDIA_TYPES = frozenset({SLIDES_MEDIA_TYPE})

XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

CSV_MEDIA_TYPE = "text/csv"

SPREADSHEET_MEDIA_TYPES = frozenset({XLSX_MEDIA_TYPE, CSV_MEDIA_TYPE})

NOTEBOOK_MEDIA_TYPE = "application/x-ipynb+json"

NOTEBOOK_MEDIA_TYPES = frozenset({NOTEBOOK_MEDIA_TYPE})

MAIL_MEDIA_TYPES = frozenset({"message/rfc822"})
"""``.eml`` only. ``.msg`` is a MAPI compound file rather than a message, its route through
permissively-licensed libraries is specified in ``docs/parsing.md`` §10, and it is
[#21](https://github.com/mgd43b/manicule/issues/21) rather than part of v1."""

PLAINTEXT_MEDIA_TYPES = frozenset({"text/plain"})
"""Declared narrowly. The global fallback tail routes by configuration, not by a parser
claiming every media type for itself — a parser that claimed ``*`` would win documents a real
parser was installed to handle."""

STRUCTURED_MEDIA_TYPES = frozenset(
    {
        "application/json",
        "application/toml",
        "application/x-yaml",
        "application/yaml",
        "text/toml",
        "text/x-yaml",
        "text/yaml",
    }
)
"""Deliberately without ``application/json;profile=atlas-doc-format``: Confluence's document
format is JSON-shaped but is a document tree, and indexing it as a configuration file would
cite key paths where a reader expects sections."""

ARCHIVE_MEDIA_TYPES = frozenset({"application/zip", "application/x-zip-compressed"})
"""Declared narrowly, which is half of the defense in ``docs/parsing.md`` §9.4. The other half
is media type resolution running before parser dispatch, so a ``.docx`` with a correct
extension or a correct declared type never reaches the archive parser at all."""


# --- pinned transformations --------------------------------------------------------------


def html_text_version() -> str:
    """Identity of the pinned HTML-to-text conversion that an HTML-only mail body addresses.

    Two components, because either can move the line numbers. ``web-blocks/1`` is manicule's
    own rule — take the blocks the web parser yields and join them with a blank line — and the
    second is the engine underneath it, whose text extraction is the other half of the answer.
    The installed version is read rather than written down so that an upgrade cannot pass
    unnoticed; the price is that a ``selectolax`` release makes the chunk fingerprint differ
    and ingest say so, which is the explicit, priced operation ``docs/parsing.md`` §1.7 asks
    for instead of silent drift.

    A function rather than a constant: this module is imported during plugin discovery, and an
    install without the parsing extras would otherwise fail there — during discovery, before
    any configuration has been read — rather than at the point something actually needs to
    convert HTML.

    Raises:
        PackageNotFoundError: ``selectolax`` is not installed, so no HTML can be converted at
            all and there is no version to record. Reported where it can be acted on, naming
            the extra to install.
    """
    from importlib.metadata import version  # noqa: PLC0415 - see docstring

    return f"web-blocks/1+selectolax/{version('selectolax')}"


# --- configuration models ----------------------------------------------------------------


class PdfConfig(BaseModel):
    """Configuration for :class:`~manicule.parsers.pdf.PdfParser`."""

    outline_headings: bool = Field(
        default=True,
        description="Take heading paths from the document outline when the PDF has one. A "
        "PDF has no heading semantics — only glyphs with font sizes — so this is the only "
        "honest source; there is deliberately no font-size clustering behind it.",
    )
    max_pages: int = Field(
        default=5000,
        gt=0,
        description="Pages beyond this are not parsed and the document is declined, so one "
        "pathological file cannot occupy an ingest run indefinitely.",
    )


class SourceCodeConfig(BaseModel):
    """Configuration for :class:`~manicule.parsers.sourcecode.SourceCodeParser`.

    Set under ``plugins.config."parser.sourcecode"``. The declared language set decides what
    routes to the parser at all and what the corpus fingerprint records, and the two grammar
    overrides are what a container image and an air-gapped site respectively need in order to
    pre-seed.
    """

    languages: tuple[str, ...] = DECLARED_LANGUAGES
    """The declared language set. Validated against the grammar manifest when the parser is
    built, so a typo — ``c_sharp`` for ``csharp`` — fails at startup with the near misses
    listed, rather than becoming a document that mysteriously never parses."""

    grammar_cache_dir: Path | None = None
    """Where grammars live. ``None`` uses the per-user cache. A container image sets this so
    the grammars pre-seeded at build time are the ones the running process finds."""

    grammar_manifest_url: str | None = None
    """Where the grammar manifest is fetched from. ``None`` uses the public one; a site with
    no route to it points this at an internal mirror."""

    max_block_chars: int = Field(default=1536, gt=0)
    """When a block's source is longer than this, it is split at the next node boundary down.

    Measured in characters rather than tokens because a parser runs before an embedder is
    bound, and a token count is only meaningful with the embedder's own vocabulary
    (``manicule.chunking.tokens``). 1536 is a 512-token budget at three characters per token,
    which is conservative for code — identifiers, punctuation and indentation all tokenize
    densely — so a block under this bound is comfortably under the budget the chunker later
    enforces with the real tokenizer. It is a *pre*-split threshold, not a guarantee: the
    chunker still measures, and this only decides which boundaries it is offered.
    """


class MarkdownConfig(BaseModel):
    """Configuration for :class:`~manicule.parsers.markdown.MarkdownParser`."""

    front_matter: bool = Field(
        default=True,
        description="Treat a leading ``---`` fenced block as metadata rather than content.",
    )
    """Front matter left in place is not inert. CommonMark reads ``title: Something`` followed
    by ``---`` as a setext heading, so the document acquires a top-level heading nobody wrote
    and every heading path below it hangs off it."""

    jsx_media_types: frozenset[str] = Field(
        default=frozenset({"text/mdx"}),
        description="Media types whose documents may contain JSX component tags.",
    )
    """Declared rather than sniffed from the body: a ``.md`` file containing a line that looks
    like a component tag is a Markdown file containing that text, and guessing otherwise would
    drop it from the index."""


class WebConfig(BaseModel):
    """Configuration for :class:`~manicule.parsers.web.WebParser`."""

    drop_tags: frozenset[str] = Field(
        default=frozenset({"script", "style", "noscript", "template"}),
        description="Elements removed, with their contents, before any text is extracted.",
    )
    """These carry no prose. Indexing a script body puts identifiers and punctuation into the
    vector, where they match queries by accident and cite a line no reader ever saw."""


class ConfluenceConfig(BaseModel):
    """Configuration for :class:`~manicule.parsers.confluence.ConfluenceStorageParser`."""

    drop_tags: frozenset[str] = Field(
        default=frozenset({"script", "style", "noscript", "template"}),
        description="Elements removed, with their contents, before any text is extracted.",
    )
    """Storage format is authored through a rich-text editor, but it is not therefore safe: a
    space administrator can place raw HTML on a page, and an export is a file anyone can edit
    before it is ingested. The same elements the HTML parser drops are dropped here, for the
    same reason — a script body in the vector matches queries by accident and cites a line no
    reader ever saw."""

    keep_unsupported_macros: bool = Field(
        default=True,
        description="Emit a named placeholder where a macro has no reader here.",
    )
    """**Off is a real choice and a lossy one.** A placeholder is how a reader learns that the
    page said something this parser could not read; without it the omission is indistinguishable
    from the page having been empty there. Turning it off suits a corpus whose pages carry a
    navigation macro on every one of them, where the placeholders are noise repeated ten
    thousand times — but the content is gone from the index either way, and only the notice is
    configurable."""


class ADFConfig(BaseModel):
    """Configuration for :class:`~manicule.parsers.adf.ADFParser`."""

    keep_card_links: bool = Field(
        default=True,
        description="Emit the target of a block or inline card as text.",
    )
    """Cards are links to other pages, which is a cross-reference graph worth keeping
    (``confluence.md`` §5). A corpus where every page links to a dozen others may prefer the
    prose without them, since a URL contributes little to a sentence's meaning."""


_HEADING_STYLE = re.compile(r"^heading([1-9])$")
"""Word's built-in heading styles, matched against a style key with spacing and case removed.

The style **id** is matched first because Word localizes style names but not ids: a document
authored in German carries ``Heading1`` as the id and ``Überschrift 1`` as the name, and a
parser matching only the name finds no headings at all in it.
"""

_LIST_STYLES: tuple[str, ...] = ("List Bullet", "List Number", "List Paragraph", "List Continue")


def _flatten(key: str) -> str:
    """A style key with spacing and case removed, so an id and a name compare equal."""
    return "".join(key.split()).lower()


class WordConfig(BaseModel):
    """Configuration for :class:`~manicule.parsers.word.WordParser`."""

    extra_heading_styles: dict[str, int] = Field(
        default_factory=dict[str, int],
        description="Style name or id to heading level, for templates that define their own "
        "heading styles. Word's built-in Heading 1-9 are always recognized; a house template "
        "calling its top level 'Chapter Title' has no heading structure without this.",
    )
    list_style_prefixes: tuple[str, ...] = Field(
        default=_LIST_STYLES,
        description="Paragraph styles whose paragraphs are list items. A trailing digit is the "
        "nesting level, matching Word's 'List Bullet 2'. Configurable because the depth a "
        "template supports is a template decision.",
    )
    table_header_rows: int = Field(
        default=1,
        ge=0,
        description="How many leading rows of a table are header rows. The chunker repeats "
        "them into every part of a table too large for one chunk (docs/parsing.md §4.2). "
        "Declared here because WordprocessingML records a repeating header row in a place "
        "python-docx does not expose, and the alternative — reading it off the first row being "
        "bold — is the guess that section forbids.",
    )

    def heading_level(self, keys: Sequence[str]) -> int | None:
        """The heading level a paragraph's style keys imply, or ``None`` if it is not one."""
        for key in keys:
            declared = self.extra_heading_styles.get(key)
            if declared is not None:
                return min(max(declared, 1), 9)
        for key in keys:
            match = _HEADING_STYLE.match(_flatten(key))
            if match is not None:
                return int(match.group(1))
        return None

    def list_level(self, keys: Sequence[str]) -> int | None:
        """The list nesting level a paragraph's style keys imply, or ``None``."""
        for key in keys:
            flat = _flatten(key)
            for prefix in self.list_style_prefixes:
                base = _flatten(prefix)
                if flat == base:
                    return 1
                if flat.startswith(base) and flat[len(base) :].isdigit():
                    return int(flat[len(base) :])
        return None


class SlidesConfig(BaseModel):
    """Configuration for :class:`~manicule.parsers.slides.SlidesParser`."""

    include_speaker_notes: bool = Field(
        default=True,
        description="Index speaker notes as prose on their slide. On by default because the "
        "notes frequently carry the sentence the slide only gestures at; off for decks whose "
        "notes are a rehearsal script nobody should retrieve.",
    )


class SpreadsheetConfig(BaseModel):
    """Configuration for :class:`~manicule.parsers.spreadsheet.SpreadsheetParser`."""

    header_rows: int = Field(
        default=1,
        ge=0,
        description="How many leading rows of a used range are header rows. The chunker repeats "
        "them into every part of a table too large for one chunk (docs/parsing.md §4.2). "
        "Declared rather than detected: the alternative is inferring a header from the first "
        "row being bold, which that section forbids because formatting is not structure.",
    )
    csv_delimiter: str = Field(
        default=",",
        min_length=1,
        max_length=1,
        description="Field separator for CSV. Declared, not sniffed: sniffing reads a sample "
        "and guesses, so the same export could be split into different columns on two "
        "machines, and every cell reference in the corpus would depend on which.",
    )
    include_hidden_sheets: bool = Field(
        default=False,
        description="Index sheets the workbook marks hidden. Off by default because a hidden "
        "sheet is usually working data the author chose not to show; on when it is the "
        "reference table everything else looks up.",
    )


class NotebookConfig(BaseModel):
    """Configuration for :class:`~manicule.parsers.notebook.NotebookParser`."""

    include_outputs: bool = Field(
        default=True,
        description="Index text outputs — stream output, text/plain results, error messages — "
        "as prose under their cell. On by default because the output is often the only place "
        "the answer appears; off for notebooks whose outputs are megabytes of logging.",
    )
    include_raw_cells: bool = Field(
        default=True,
        description="Index raw cells. They hold content a conversion step consumes — LaTeX, "
        "reStructuredText, nbconvert directives — which is prose to a reader and to a query.",
    )


class ArchiveConfig(BaseModel):
    """The four zip-bomb limits, the nesting limit, and nothing else.

    Every default is from ``docs/parsing.md`` §9.3. They are configuration rather than
    constants because the right numbers depend on the corpus — an archive of scanned tiffs is
    legitimately large — and because a test can then exercise the streaming path at a size
    that keeps a suite fast instead of at a gigabyte.
    """

    max_total_bytes: int = Field(
        default=1024**3,
        gt=0,
        description="Uncompressed bytes across the whole container tree. Counted while "
        "streaming, never taken from a member's declared size.",
    )
    max_member_bytes: int = Field(
        default=64 * 1024**2,
        gt=0,
        description="Uncompressed bytes for one member, so a single member cannot exhaust "
        "the tree budget on its own.",
    )
    max_compression_ratio: float = Field(
        default=100.0,
        gt=0.0,
        description="Uncompressed bytes per compressed byte, measured on what was actually "
        "read. Catches the classic single-file bomb.",
    )
    max_members: int = Field(
        default=10_000,
        gt=0,
        description="Members across the whole container tree. Catches the many-tiny-files "
        "variant, which every byte limit passes.",
    )
    max_depth: int = Field(
        default=MAX_DEPTH,
        ge=1,
        description="How far containers may nest, counted from the top-level document.",
    )


class MailConfig(BaseModel):
    """Configuration for :class:`~manicule.parsers.mail.MailParser`."""

    max_block_lines: int = Field(
        default=100,
        ge=1,
        description="Longest run of lines one body block may span before it is split at a "
        "line boundary into parts that each keep their own exact span.",
    )
    """Bounded for the same reason as every other line-anchored parser: when a block does not
    fit the chunk budget the chunker splits its text and each part keeps the block's anchor, so
    every part would resolve to the whole block to address a fraction of it."""

    expand_attachments: bool = Field(
        default=True,
        description="Whether attachments become documents of their own.",
    )
    """Turning this off does not make attachments disappear quietly: each one is reported as a
    failed member naming the setting, because "the message had three attachments and the index
    has none" is not something anyone would otherwise discover."""

    max_depth: int = Field(
        default=MAX_DEPTH,
        ge=1,
        description="How far containers may nest, counted from the top-level document.",
    )

    max_members: int = Field(
        default=1000,
        gt=0,
        description="Attachments across the whole container tree, past which the rest are "
        "reported rather than expanded.",
    )
    """A message is a container, so it needs the limit every container needs.

    The archive parser has counted members since it was written (``docs/parsing.md`` §9.3);
    mail expanded attachments with no ceiling at all, which makes a message with a hundred
    thousand parts a hundred thousand documents. Lower than the archive's ten thousand because
    a message with a thousand attachments is already not a message anybody sent.
    """


class PlaintextConfig(BaseModel):
    """Configuration for :class:`~manicule.parsers.plaintext.PlaintextParser`."""

    max_block_lines: int = Field(
        default=100,
        ge=1,
        description="Longest run of lines one block may span before it is split at a line "
        "boundary into parts that each keep their own exact span.",
    )
    """Bounded because the chunker cannot narrow a line span.

    When a block does not fit the chunk budget the chunker splits its text, and every part
    keeps the *block's* anchor — so four chunks of one 900-line block all cite all 900 lines,
    and each of them resolves to nine hundred lines to address a fifth of them. Splitting here
    instead costs nothing and gives every part a line span that is exactly its own text.
    """


class StructuredConfig(BaseModel):
    """Configuration for :class:`~manicule.parsers.structured.StructuredParser`."""

    max_block_lines: int = Field(
        default=100,
        ge=1,
        description="Longest run of lines one block may span before it is split at the next "
        "level of the document's own structure, and finally at line boundaries.",
    )
    """Bounded because the chunker cannot narrow a line span.

    When a block does not fit the chunk budget the chunker splits its text and every part
    keeps the *block's* anchor, so each part resolves to the whole block to address a fraction
    of it. Splitting here uses the document's own structure instead, which costs nothing and
    deepens ``symbol`` on the way down (``docs/parsing.md`` §11).
    """
