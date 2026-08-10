"""Markdown and MDX, located by source line numbers.

Markdown is the one format in the set where the source and the citation are the same
characters, so this parser keeps them the same characters: **a block's ``text`` is the exact
run of source lines its anchor addresses.** A heading block is ``## Configuration``, not
``Configuration``; a table block is the pipe table as written. Rendering to plain text first
would mean :meth:`MarkdownParser.resolve` had to reproduce the renderer exactly to get the
same string back, and every divergence between the two would surface as a citation quoting
something the file does not contain.

Three consequences worth stating, because each is a decision rather than an accident:

**Line numbers come from ``token.map`` and are converted once.** ``markdown-it-py`` reports a
0-based, half-open line span; every anchor manicule stores is 1-based and inclusive at both
ends. The conversion happens here, at the parser boundary, and nowhere else.

**Fragments are synthesised, because Markdown defines none.** A ``.md`` file has no anchor
scheme of its own, so the GitHub-style slug in :class:`~manicule.parsers.base.SlugAllocator`
is used — the one every Markdown host derives the same way. Where the source *does* publish
fragments, as Confluence and HTML authors do, this parser is not the one reading them.

**JSX components are markup, not prose.** ``.mdx`` embeds component invocations in the
document body; they are emitted as ``media`` blocks naming the component, so that a component
invocation never reaches the embedder as a sentence or a citation as a quotation. The
Markdown inside a component's children is ordinary Markdown and is parsed as such.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field

from markdown_it import MarkdownIt
from markdown_it.token import Token
from pydantic import BaseModel, Field

from manicule.core.anchors import Anchor, HeadingAnchor, LineAnchor, Unlocated
from manicule.core.content import BlockKind, Metadata, ParsedBlock, RawDocument
from manicule.parsers.base import (
    HeadingStack,
    ParserProfile,
    SlugAllocator,
    decode,
    lines_of,
    resolve_lines,
)

__all__ = ["MARKDOWN_MEDIA_TYPES", "MarkdownConfig", "MarkdownParser"]

MARKDOWN_MEDIA_TYPES = frozenset({"text/markdown", "text/x-markdown", "text/mdx"})
"""What routes here. ``text/mdx`` is unregistered but is what the extension map resolves
``.mdx`` to, and MDX is Markdown with components rather than a format of its own."""

_BLOCK_KINDS: Mapping[str, BlockKind] = {
    "paragraph_open": BlockKind.PROSE,
    "blockquote_open": BlockKind.PROSE,
    "bullet_list_open": BlockKind.LIST,
    "ordered_list_open": BlockKind.LIST,
    "table_open": BlockKind.TABLE,
    "fence": BlockKind.CODE,
    "code_block": BlockKind.CODE,
    "heading_open": BlockKind.HEADING,
}
"""Token type to block kind. A token type absent from here carries no text of its own — a
thematic break, a closing tag — and produces no block rather than an empty one."""

_TEXT_CHILDREN = frozenset({"text", "code_inline"})
"""Inline token types that carry literal characters. Emphasis and link markers do not, so a
heading's rendered title is built from these alone."""

_FRONT_MATTER_FENCE = "---"

_JSX_TAG = re.compile(r"^</?(?P<name>[A-Z][A-Za-z0-9_.]*)(?:\s[^<>]*?)?/?>$")
"""A JSX component tag alone on a line. Capitalised initial letter is what distinguishes a
component from an HTML element in JSX, and it is the rule the MDX compiler itself applies."""

_FENCE_MARKERS = ("```", "~~~")

_MAX_BLOCK_INDENT = 3
"""Indent past which CommonMark stops seeing a block start and starts seeing code."""

_HEADING_LEVELS: Mapping[str, int] = {f"h{level}": level for level in range(1, 7)}


class MarkdownConfig(BaseModel):
    """Configuration for :class:`MarkdownParser`."""

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


@dataclass(frozen=True, slots=True)
class _Section:
    """A heading and the source lines it owns, up to the next heading of **any** level.

    Not including subsections. A section resolved with its subsections inside it is larger
    than the text it addresses by however deep the nesting goes, which fails the tightness
    bound in ``docs/parsing.md`` §3.3 on any document with structure.
    """

    path: tuple[str, ...]
    fragment: str | None
    first_line: int
    last_line: int


@dataclass(frozen=True, slots=True)
class _Draft:
    """A block found in the token stream, before it has been given an anchor."""

    first_line: int
    last_line: int
    kind: BlockKind
    title: str = ""
    lang: str | None = None
    metadata: Metadata = field(default_factory=Metadata)


@dataclass(frozen=True, slots=True)
class _Reading:
    """One pass over the source: the blocks it yields and the sections they live in."""

    text: str
    blocks: tuple[ParsedBlock, ...]
    by_fragment: Mapping[str, _Section]
    by_path: Mapping[tuple[str, ...], tuple[_Section, ...]]


class MarkdownParser:
    """Parses Markdown and MDX into blocks anchored to their source lines."""

    media_types = MARKDOWN_MEDIA_TYPES
    profile = ParserProfile(name="markdown", max_unlocated_ratio=0.00, max_pagelevel_ratio=None)

    def __init__(self, config: MarkdownConfig) -> None:
        self._config = config
        # CommonMark plus tables. HTML is disabled so that a raw tag is indexed as the text it
        # is rather than swallowed into an html_block, which would take the prose after it
        # down with it as far as the next blank line.
        self._md = MarkdownIt("commonmark", {"html": False}).enable("table")

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        """Yield blocks in reading order.

        Raises:
            ParseError: The bytes do not decode, so the next parser in the chain gets a turn.
        """
        for block in self._read(raw).blocks:
            yield block

    async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
        """Return the source lines ``anchor`` addresses, or ``None`` where it addresses none.

        Re-reads the document rather than consulting anything :meth:`parse` left behind: an
        anchor that only resolves against the parser's own memory of a document verifies
        nothing about the document.
        """
        if isinstance(anchor, LineAnchor):
            return resolve_lines(decode(raw), anchor)
        if not isinstance(anchor, HeadingAnchor):
            return None
        reading = self._read(raw)
        section = _locate(reading, anchor)
        if section is None:
            return None
        lines = lines_of(reading.text)
        return "\n".join(lines[section.first_line - 1 : section.last_line])

    def _read(self, raw: RawDocument) -> _Reading:
        text = decode(raw)
        lines = lines_of(text)
        masked, media = self._mask(lines, raw.media_type)
        drafts = _drafts(self._md.parse("\n".join(masked)), lines)
        drafts.extend(media)
        drafts.sort(key=lambda draft: (draft.first_line, draft.last_line))
        sections = _sections(drafts, lines)
        return _assemble(text, lines, drafts, sections)

    def _mask(self, lines: Sequence[str], media_type: str) -> tuple[list[str], list[_Draft]]:
        """Blank the lines Markdown must not see, and return the blocks they become.

        Blanking rather than deleting keeps every surviving line at the number it has in the
        file, so no offset arithmetic is needed anywhere downstream. It also terminates the
        surrounding block, which is what makes the Markdown inside a component's children
        parse as ordinary Markdown instead of being absorbed into the tag.
        """
        masked = list(lines)
        start = _front_matter_end(lines) if self._config.front_matter else 0
        for index in range(start):
            masked[index] = ""
        media: list[_Draft] = []
        if media_type not in self._config.jsx_media_types:
            return masked, media
        for index, name in _component_lines(lines, start):
            masked[index] = ""
            media.append(
                _Draft(
                    first_line=index + 1,
                    last_line=index + 1,
                    kind=BlockKind.MEDIA,
                    metadata={"component": name},
                )
            )
        return masked, media


def _component_lines(lines: Sequence[str], start: int) -> Iterator[tuple[int, str]]:
    """Every line that is a JSX component tag and nothing else, with the component's name.

    Code is skipped, both fenced and indented. An MDX page documenting a component shows its
    tags inside a fence, and treating those as an invocation would blank the lines out of the
    code block — so the page would lose the example it exists to give, and gain ``media``
    blocks for components nobody used.
    """
    fence = ""
    for index in range(start, len(lines)):
        line = lines[index]
        stripped = line.strip()
        if fence:
            if stripped.startswith(fence):
                fence = ""
            continue
        marker = next((mark for mark in _FENCE_MARKERS if stripped.startswith(mark)), "")
        if marker:
            fence = marker
            continue
        if len(line) - len(line.lstrip(" ")) > _MAX_BLOCK_INDENT:
            continue
        match = _JSX_TAG.match(stripped)
        if match is not None:
            yield index, match["name"]


def _front_matter_end(lines: Sequence[str]) -> int:
    """How many leading lines are front matter. Zero when there is none."""
    if not lines or lines[0].strip() != _FRONT_MATTER_FENCE:
        return 0
    for index in range(1, len(lines)):
        if lines[index].strip() == _FRONT_MATTER_FENCE:
            return index + 1
    return 0


def _trim(lines: Sequence[str], first: int, last: int) -> int:
    """Drop blank lines from the end of a 1-based inclusive span.

    ``markdown-it-py`` ends a list's span at the blank line that closed it. Anchoring that
    line would claim a line the block does not contain.
    """
    while last > first and not lines[last - 1].strip():
        last -= 1
    return last


def _inline_text(token: Token | None) -> str:
    """The literal characters of an inline token, with emphasis and link markers dropped."""
    if token is None or token.type != "inline":
        return ""
    if token.children is None:
        return token.content
    return "".join(child.content for child in token.children if child.type in _TEXT_CHILDREN)


def _fence_language(token: Token) -> str | None:
    """The language from a fence info string, or ``None`` when it declares none."""
    info = token.info.strip()
    if not info:
        return None
    return info.split(maxsplit=1)[0]


def _header_rows(tokens: Sequence[Token], index: int) -> Metadata:
    """Header-row count for the pipe table opening at ``index``.

    Recorded rather than inferred downstream: a table too large for one chunk is split by
    rows with its header repeated into every part (``docs/parsing.md`` §4.2), and a part
    without its header is a grid of numbers.
    """
    rows = 0
    for token in tokens[index + 1 :]:
        if token.type in {"thead_close", "table_close"}:
            break
        if token.type == "tr_open":
            rows += 1
    return {"header_rows": rows}


def _drafts(tokens: Sequence[Token], lines: Sequence[str]) -> list[_Draft]:
    """Every top-level block in the token stream, in document order.

    Top level is ``token.level == 0`` with a source map: a nested list or a table cell is
    part of the block that contains it, and closing tokens carry no map at all.
    """
    drafts: list[_Draft] = []
    for index, token in enumerate(tokens):
        kind = _BLOCK_KINDS.get(token.type)
        if kind is None or token.level != 0 or token.map is None:
            continue
        follower = tokens[index + 1] if index + 1 < len(tokens) else None
        # token.map is 0-based and half-open; anchors are 1-based and inclusive.
        first_line = token.map[0] + 1
        drafts.append(
            _Draft(
                first_line=first_line,
                last_line=_trim(lines, first_line, token.map[1]),
                kind=kind,
                title=_inline_text(follower) if kind is BlockKind.HEADING else "",
                lang=_fence_language(token) if token.type == "fence" else None,
                metadata=_header_rows(tokens, index) if kind is BlockKind.TABLE else {},
            )
        )
    return drafts


def _sections(drafts: Sequence[_Draft], lines: Sequence[str]) -> list[_Section]:
    """One section per heading, each ending where the next heading of any level begins."""
    headings = [draft for draft in drafts if draft.kind is BlockKind.HEADING]
    stack = HeadingStack()
    slugs = SlugAllocator()
    sections: list[_Section] = []
    for position, heading in enumerate(headings):
        level = _HEADING_LEVELS.get(_heading_tag(lines, heading), 1)
        path = stack.push(level, heading.title)
        ends = headings[position + 1].first_line - 1 if position + 1 < len(headings) else len(lines)
        last_line = _trim(lines, heading.first_line, ends)
        sections.append(
            _Section(
                path=path,
                fragment=slugs.allocate(heading.title),
                first_line=heading.first_line,
                last_line=last_line,
            )
        )
    return sections


def _heading_tag(lines: Sequence[str], heading: _Draft) -> str:
    """``h1``-``h6`` for a heading draft, read back from its own source line.

    The level is in the markup rather than in the draft because a draft is a line span and a
    kind; carrying a level on every block so that headings can use one would put a field on
    six kinds that have no use for it.
    """
    line = lines[heading.first_line - 1].lstrip()
    if line.startswith("#"):
        return f"h{min(len(line) - len(line.lstrip('#')), 6)}"
    # A setext heading: `===` underlines an h1, `---` an h2.
    underline = lines[heading.last_line - 1].strip()
    return "h1" if underline.startswith("=") else "h2"


def _assemble(
    text: str, lines: Sequence[str], drafts: Sequence[_Draft], sections: Sequence[_Section]
) -> _Reading:
    """Give every draft the anchor its section allows, and index the sections for resolution."""
    by_path: dict[tuple[str, ...], tuple[_Section, ...]] = {}
    for section in sections:
        by_path[section.path] = (*by_path.get(section.path, ()), section)
    by_fragment = {
        section.fragment: section for section in sections if section.fragment is not None
    }

    blocks: list[ParsedBlock] = []
    current: _Section | None = None
    position = 0
    for draft in drafts:
        if draft.kind is BlockKind.HEADING:
            current = sections[position]
            position += 1
        body = "\n".join(lines[draft.first_line - 1 : draft.last_line])
        if not body.strip():
            continue
        blocks.append(
            ParsedBlock(
                kind=draft.kind,
                text=body,
                anchor=_anchor_for(current, by_path, draft),
                heading_path=current.path if current is not None else (),
                lang=draft.lang,
                metadata=draft.metadata,
            )
        )
    return _Reading(text=text, blocks=tuple(blocks), by_fragment=by_fragment, by_path=by_path)


def _anchor_for(
    section: _Section | None,
    by_path: Mapping[tuple[str, ...], tuple[_Section, ...]],
    draft: _Draft,
) -> Anchor:
    """The anchor a block gets, given the section it is in.

    Content before the first heading is anchored by line number rather than to an invented
    root heading: Markdown source lines are exact, and they are already in hand.
    """
    if section is None:
        return LineAnchor(start=draft.first_line, end=draft.last_line)
    if section.fragment is None and len(by_path[section.path]) > 1:
        return Unlocated(
            reason=f"ambiguous heading path {' > '.join(section.path)!r}: it names more than "
            f"one section and the heading text yields no fragment to tell them apart"
        )
    return HeadingAnchor(path=section.path, fragment=section.fragment)


def _locate(reading: _Reading, anchor: HeadingAnchor) -> _Section | None:
    """Find the section an anchor addresses. Fragment first, path second."""
    if anchor.fragment is not None:
        return reading.by_fragment.get(anchor.fragment)
    candidates = reading.by_path.get(anchor.path, ())
    return candidates[0] if len(candidates) == 1 else None
