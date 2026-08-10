"""DOCX: a Word document is a flow of paragraphs, and a citation names its section.

A `.docx` records paragraphs, styles and tables. **It does not record pages.** Pagination is
produced by a layout engine at render time from the fonts, the printer metrics and the Word
version, so a page number reported here would be invented rather than read. Explicit page
breaks are stored, but they are a lower bound on the page count and not a pagination. So a
citation into a Word document is "§ Deployment > Rollback", never "p. 7"
(``docs/parsing.md`` §2.5).

Two consequences shape everything below.

**The heading is the location, so it has to be a real heading.** Levels come from the
paragraph style — ``Heading 1``-``Heading 9``, matched on the style id, which Word keeps
language-independent while it localises the display name — never from a font size or a line
that looks like a title. A document whose author styled headings by hand has no heading
structure to recover, and inventing one would put a heading nobody wrote into the breadcrumb,
which reaches the embedder.

**A section resolves to itself, not to its subsections.** :meth:`WordParser.resolve` returns
the span from a heading up to the next heading *of any level*. Including subsections would
make a top-level section resolve to most of the document, which fails the tightness bound in
``docs/parsing.md`` §3.3 on any nested document and, worse, makes a citation of the
introduction quote the whole chapter.
"""

from __future__ import annotations

import io
import re
import zipfile
from collections import Counter
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import docx
from docx.document import Document as DocxDocument
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table
from docx.text.paragraph import Paragraph
from pydantic import BaseModel, Field

from manicule.core.anchors import Anchor, HeadingAnchor, Unlocated
from manicule.core.content import BlockKind, Metadata, ParsedBlock, RawDocument
from manicule.core.errors import ParseError
from manicule.parsers.base import HeadingStack, ParserProfile, SlugAllocator

__all__ = ["MEDIA_TYPE", "WordConfig", "WordParser"]

MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

_HEADING_STYLE = re.compile(r"^heading([1-9])$")
"""Word's built-in heading styles, matched against a style key with spaces removed.

The style **id** is matched first because Word localises style names but not ids: a document
authored in German carries ``Heading1`` as the id and ``Überschrift 1`` as the name, and a
parser matching only the name finds no headings at all in it.
"""

_LIST_STYLES: tuple[str, ...] = ("List Bullet", "List Number", "List Paragraph", "List Continue")

_CELL_SEPARATOR = "\t"
_ROW_SEPARATOR = "\n"


class WordConfig(BaseModel):
    """Configuration for :class:`WordParser`."""

    extra_heading_styles: dict[str, int] = Field(
        default_factory=dict[str, int],
        description="Style name or id to heading level, for templates that define their own "
        "heading styles. Word's built-in Heading 1-9 are always recognised; a house template "
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


def _flatten(key: str) -> str:
    """A style key with spacing and case removed, so an id and a name compare equal."""
    return key.replace(" ", "").lower()


@dataclass(frozen=True, slots=True)
class _Item:
    """One block-to-be: everything but the anchor, which belongs to its section."""

    kind: BlockKind
    text: str
    metadata: Metadata = field(default_factory=Metadata)


@dataclass(frozen=True, slots=True)
class _Heading:
    """A heading paragraph, which opens a section rather than sitting inside one."""

    level: int
    text: str


@dataclass(frozen=True, slots=True)
class _Section:
    """A heading and the content beneath it, up to the next heading of any level."""

    path: tuple[str, ...]
    fragment: str | None
    heading_path: tuple[str, ...]
    items: tuple[_Item, ...]

    @property
    def text(self) -> str:
        """The section as one string. What :meth:`WordParser.resolve` returns."""
        return "\n".join(item.text for item in self.items)


def _style_keys(paragraph: Paragraph) -> tuple[str, ...]:
    """The style id and name of a paragraph, in the order they should be matched."""
    style = paragraph.style
    if style is None:
        return ()
    return tuple(key for key in (style.style_id, style.name) if key)


def _render_content(items: Iterable[Paragraph | Table]) -> list[str]:
    """Render paragraphs and tables to lines, descending into nested tables.

    A table inside a table cell is content like any other. Reading only ``cell.text`` would
    drop it silently, which is the quiet kind of data loss: the citation still resolves, and
    the quotation is simply missing a column of the document.
    """
    lines: list[str] = []
    for item in items:
        if isinstance(item, Table):
            lines.extend(_render_table(item).split("\n"))
            continue
        text = item.text.strip()
        if text:
            lines.append(text)
    return lines


def _render_table(table: Table) -> str:
    """A table as tab-separated cells, one row per line.

    A merged cell reports its text once in the XML but occupies every grid position it spans,
    and ``row.cells`` yields it at each of them. The repeat is kept deliberately: it is what
    the grid shows, and it is what lets a split part of the table (``docs/parsing.md`` §4.2)
    carry the header that applies to *its* columns rather than a blank.
    """
    rows: list[str] = []
    for row in table.rows:
        cells = [" ".join(_render_content(cell.iter_inner_content())) for cell in row.cells]
        rows.append(_CELL_SEPARATOR.join(cells))
    return _ROW_SEPARATOR.join(rows)


def _elements(document: DocxDocument, config: WordConfig) -> list[_Heading | _Item]:
    """Flatten the body into headings and items, in reading order.

    Runs of consecutive list paragraphs collapse into one ``list`` block: a bulleted list is
    one structure, and one block per bullet would hand the chunker a boundary at every item
    where ``docs/parsing.md`` §4.2 asks it to split at top-level items instead.
    """
    out: list[_Heading | _Item] = []
    pending: list[tuple[int, str]] = []

    def flush_list() -> None:
        if not pending:
            return
        out.append(
            _Item(
                kind=BlockKind.LIST,
                text="\n".join(text for _, text in pending),
                metadata={"list_levels": [level for level, _ in pending]},
            )
        )
        pending.clear()

    for content in document.iter_inner_content():
        if isinstance(content, Table):
            flush_list()
            text = _render_table(content)
            if text.strip():
                rows = list(content.rows)
                out.append(
                    _Item(
                        kind=BlockKind.TABLE,
                        text=text,
                        metadata={
                            "rows": len(rows),
                            "columns": max((len(row.cells) for row in rows), default=0),
                            "header_rows": min(config.table_header_rows, len(rows)),
                        },
                    )
                )
            continue

        keys = _style_keys(content)
        text = content.text.strip()
        level = config.heading_level(keys)
        if level is not None:
            flush_list()
            # A heading paragraph with no text names no section, so it opens none. Treating it
            # as one would put an empty string into every path beneath it.
            if text:
                out.append(_Heading(level=level, text=text))
            continue
        if not text:
            continue
        list_level = config.list_level(keys)
        if list_level is not None:
            pending.append((list_level, text))
            continue
        flush_list()
        out.append(_Item(kind=BlockKind.PROSE, text=text))

    flush_list()
    return out


def _sections(elements: Iterable[_Heading | _Item], title: str) -> list[_Section]:
    """Group flattened content into sections, one per heading plus the preamble.

    Content before the first heading is addressed by the document title, which is a location
    the file reports rather than one invented here. A document with no title has nothing to
    address that content with, and it becomes :class:`Unlocated` in :func:`_anchor_for`.
    """
    stack = HeadingStack()
    slugs = SlugAllocator()
    sections: list[_Section] = []
    path: tuple[str, ...] = (title,) if title else ()
    heading_path: tuple[str, ...] = ()
    fragment: str | None = None
    items: list[_Item] = []

    for element in elements:
        if not isinstance(element, _Heading):
            items.append(element)
            continue
        if items:
            sections.append(_Section(path, fragment, heading_path, tuple(items)))
        heading_path = stack.push(element.level, element.text)
        path = heading_path
        fragment = slugs.allocate(element.text)
        items = [
            _Item(kind=BlockKind.HEADING, text=element.text, metadata={"level": element.level})
        ]
    if items:
        sections.append(_Section(path, fragment, heading_path, tuple(items)))
    return sections


def _anchor_for(section: _Section, paths: Mapping[tuple[str, ...], int]) -> Anchor:
    """The anchor for a section, or :class:`Unlocated` when nothing addresses it.

    A section with a fragment is addressable whatever its path does, which is the whole point
    of the fragment (``docs/parsing.md`` §2.3). Without one — a heading whose text slugifies to
    nothing, or the preamble — the path is the only address, and a path that occurs twice
    addresses neither occurrence.
    """
    if section.fragment is not None:
        return HeadingAnchor(path=section.path, fragment=section.fragment)
    if not section.path:
        return Unlocated(
            reason="content precedes the first heading and the document declares no title, so "
            "there is no section to cite. Give the document a title in its properties, or a "
            "heading above this content"
        )
    if paths[section.path] > 1:
        rendered = " > ".join(section.path)
        return Unlocated(
            reason=f"the heading path {rendered!r} addresses {paths[section.path]} places in "
            f"this document and this one has no text a fragment can be built from, so nothing "
            f"tells them apart. Give the heading at least one letter or digit, or rename it"
        )
    return HeadingAnchor(path=section.path, fragment=None)


class WordParser:
    """Parses a `.docx` into sections, tables and lists, anchored by heading.

    ``max_unlocated_ratio`` is 0.05 (``docs/parsing.md`` §3.4): content before the first
    heading in a document with no title has no section to cite, and that is a document
    property rather than a parser fault. There is no page-level budget because this parser
    never emits a :class:`~manicule.core.anchors.PageAnchor` — a DOCX has no pages to number.
    """

    media_types = frozenset({MEDIA_TYPE})
    profile = ParserProfile(name="word", max_unlocated_ratio=0.05, max_pagelevel_ratio=None)

    def __init__(self, config: WordConfig) -> None:
        self._config = config

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        """Yield one block per paragraph run, table and list, in reading order."""
        sections = self._read(raw)
        paths = Counter(section.path for section in sections)
        for section in sections:
            anchor = _anchor_for(section, paths)
            for item in section.items:
                yield ParsedBlock(
                    kind=item.kind,
                    text=item.text,
                    anchor=anchor,
                    heading_path=section.heading_path,
                    metadata=item.metadata,
                )

    async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
        """Return the section ``anchor`` addresses, re-derived from ``raw``.

        Re-derived, never remembered: reading a map left behind by :meth:`parse` would verify
        that the parser agrees with itself, which is the one thing that was never in doubt.
        """
        if not isinstance(anchor, HeadingAnchor):
            return None
        sections = self._read(raw)
        paths = Counter(section.path for section in sections)
        for section in sections:
            if anchor.fragment is not None:
                if section.fragment == anchor.fragment:
                    return section.text
                continue
            if section.fragment is None and section.path == anchor.path and paths[anchor.path] == 1:
                return section.text
        return None

    def _read(self, raw: RawDocument) -> list[_Section]:
        document = _open(raw)
        return _sections(_elements(document, self._config), _title(raw, document))


def _title(raw: RawDocument, document: DocxDocument) -> str:
    """The document's title: what the connector reported, else what the file declares.

    Both are stated rather than guessed. The filename is deliberately not a fallback — a
    connector that knows a title supplies one, and ``report-final-v3.docx`` is not a title.
    """
    declared = raw.metadata.get("title")
    if isinstance(declared, str) and declared.strip():
        return declared.strip()
    return (document.core_properties.title or "").strip()


def _open(raw: RawDocument) -> DocxDocument:
    """Open the package, declining anything that is not a readable WordprocessingML one."""
    try:
        return docx.Document(io.BytesIO(raw.as_bytes()))
    # lxml's XMLSyntaxError subclasses SyntaxError, which is caught here rather than by
    # importing lxml: a truncated word/document.xml must be declined, and taking a direct
    # dependency on python-docx's XML backend to name one exception class would outlive it.
    except (PackageNotFoundError, zipfile.BadZipFile, KeyError, SyntaxError, ValueError) as exc:
        msg = (
            f"{raw.uri}: not a readable .docx package ({type(exc).__name__}: {exc}). Expected "
            f"an OOXML WordprocessingML package containing word/document.xml. Re-export it "
            f"from a word processor, or route this media type to a different parser"
        )
        raise ParseError(msg) from exc
