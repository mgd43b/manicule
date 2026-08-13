"""Confluence's Atlassian Document Format: a typed node tree, not markup.

This is why ADF is worth fetching in preference to rendered storage format. A ``codeBlock``
says it is code and names its language; a ``panel`` says it is a warning; a ``table`` keeps
its rows. Every one of those is a fact the block model can carry, and none of them survives
a pass that flattens the document to a run of words
(``docs/connectors/confluence.md`` §5).

**Fragments are Confluence's own, not ours.** A citation into Confluence deep-links with
``…/pages/{id}/{slug}#{anchor}``, and the anchor is the one Confluence derives from the
heading text — so this parser derives the same one, with the same ``-1``/``-2`` suffixes for
repeated headings, counted in document order (``confluence.md`` §8). An address we invented
instead would look precise and land at the top of the page.

**Content before the first heading is addressed by the page title**, with no fragment,
because that is exactly what it is: text on the page and in no section of it. A page that
arrives without a title, or whose first heading repeats the title, has no honest
document-level address left, and those blocks are
:class:`~manicule.core.anchors.Unlocated` with the reason saying which of the two it was.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

from manicule.core.anchors import Anchor, HeadingAnchor, Unlocated
from manicule.core.content import BlockKind, Metadata, ParsedBlock, RawDocument
from manicule.core.errors import ParseError
from manicule.parsers.base import HeadingStack, ParserProfile, decode
from manicule.parsers.config import ADF_MEDIA_TYPE, ADF_MEDIA_TYPES, ADFConfig

__all__ = ["ADF_MEDIA_TYPE", "ADFConfig", "ADFParser"]

_INLINE_TYPES = frozenset(
    {
        "text",
        "hardBreak",
        "emoji",
        "mention",
        "date",
        "status",
        "inlineCard",
        "placeholder",
        "inlineExtension",
    }
)
"""Node types that render into the sentence around them rather than starting a block."""

_LIST_TYPES = frozenset({"bulletList", "orderedList"})

_INDENT = "  "

_WHITESPACE = re.compile(r"\s+")
_NOT_IN_ANCHOR = re.compile(r"[^\w-]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class _Found:
    """One node the walk recognised, before sections have been worked out."""

    kind: BlockKind
    text: str
    level: int = 0
    lang: str | None = None
    metadata: Metadata = field(default_factory=Metadata)


@dataclass(frozen=True, slots=True)
class _Section:
    """A heading, the anchor Confluence publishes for it, and the text it owns."""

    path: tuple[str, ...]
    fragment: str | None
    text: str


@dataclass(frozen=True, slots=True)
class _Reading:
    """One pass over the document: the blocks it yields and what each anchor resolves to."""

    blocks: tuple[ParsedBlock, ...]
    by_fragment: Mapping[str, _Section]
    by_path: Mapping[tuple[str, ...], tuple[_Section, ...]]
    title: str
    preamble: str


class ADFParser:
    """Parses an Atlassian Document Format body into blocks anchored to its headings."""

    media_types = ADF_MEDIA_TYPES
    profile = ParserProfile(
        name="confluence-adf", max_unlocated_ratio=0.00, max_pagelevel_ratio=None
    )

    def __init__(self, config: ADFConfig) -> None:
        self._config = config

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        """Yield blocks in reading order.

        Raises:
            ParseError: The bytes do not decode, or they are JSON that is not an ADF
                document. Declining lets the next parser in the chain have it, which is the
                right outcome for the plain JSON that shares this media type's prefix.
        """
        for block in self._read(raw).blocks:
            yield block

    async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
        """Return the text ``anchor`` addresses, or ``None`` where it addresses none.

        Re-reads the document rather than consulting anything :meth:`parse` left behind: an
        anchor that only resolves against the parser's own memory of a document verifies
        nothing about the document.
        """
        if not isinstance(anchor, HeadingAnchor):
            return None
        reading = self._read(raw)
        if anchor.fragment is not None:
            section = reading.by_fragment.get(anchor.fragment)
            return section.text if section is not None else None
        candidates = reading.by_path.get(anchor.path, ())
        if len(candidates) == 1:
            return candidates[0].text
        if candidates or anchor.path != (reading.title,):
            return None
        return reading.preamble or None

    def _read(self, raw: RawDocument) -> _Reading:
        document = _document(decode(raw), raw.uri)
        found = list(_walk(_children(document), self._config))
        declared = raw.metadata.get("title")
        title = declared.strip() if isinstance(declared, str) else ""
        return _assemble(found, title)


def _document(text: str, uri: str) -> Mapping[str, object]:
    """The ADF root node, or a refusal naming what arrived instead."""
    try:
        loaded: object = json.loads(text)
    except ValueError as exc:
        msg = (
            f"{uri}: not JSON ({exc}). Confluence bodies arrive as JSON; route this document "
            f"to the parser for the format it is actually in."
        )
        raise ParseError(msg) from exc
    node = _as_node(loaded)
    if node is None or node.get("type") != "doc":
        found = node.get("type") if node else type(loaded).__name__
        msg = (
            f"{uri}: JSON, but not an Atlassian Document Format body: the root node is "
            f"{found!r} and ADF requires 'doc'. Route plain JSON to the structured-data "
            f"parser."
        )
        raise ParseError(msg)
    return node


def _as_node(value: object) -> Mapping[str, object] | None:
    """Narrow a decoded JSON value to an object. JSON keys are strings by construction."""
    if not isinstance(value, dict):
        return None
    return cast("Mapping[str, object]", value)


def _children(node: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
    content = node.get("content")
    if not isinstance(content, list):
        return ()
    entries = cast("Sequence[object]", content)
    return [child for child in (_as_node(entry) for entry in entries) if child is not None]


def _attrs(node: Mapping[str, object]) -> Mapping[str, object]:
    return _as_node(node.get("attrs", {})) or {}


def _text_attr(node: Mapping[str, object], key: str) -> str:
    value = _attrs(node).get(key)
    return value.strip() if isinstance(value, str) else ""


def _node_type(node: Mapping[str, object]) -> str:
    kind = node.get("type")
    return kind if isinstance(kind, str) else ""


def _collapse(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


# --- inline rendering --------------------------------------------------------------------


def _inline_text(nodes: Sequence[Mapping[str, object]], config: ADFConfig) -> str:
    """Everything a run of inline nodes says, marks resolved to their characters."""
    parts: list[str] = []
    for node in nodes:
        kind = _node_type(node)
        if kind == "text":
            literal = node.get("text")
            parts.append(literal if isinstance(literal, str) else "")
        elif kind == "hardBreak":
            parts.append("\n")
        elif kind in {"emoji", "status", "mention"}:
            parts.append(_text_attr(node, "text") or _text_attr(node, "shortName"))
        elif kind == "date":
            parts.append(_text_attr(node, "timestamp"))
        elif kind in {"inlineCard", "inlineExtension"}:
            parts.append(_card_text(node, config))
        else:
            parts.append(_inline_text(_children(node), config))
    return "".join(parts)


def _card_text(node: Mapping[str, object], config: ADFConfig) -> str:
    if not config.keep_card_links:
        return ""
    return _text_attr(node, "url") or _text_attr(node, "extensionKey")


# --- block handlers ----------------------------------------------------------------------

_Handler = Callable[[Mapping[str, object], ADFConfig], Iterator[_Found]]


def _heading(node: Mapping[str, object], config: ADFConfig) -> Iterator[_Found]:
    text = _collapse(_inline_text(_children(node), config))
    if not text:
        return
    level = _attrs(node).get("level")
    yield _Found(kind=BlockKind.HEADING, text=text, level=level if isinstance(level, int) else 1)


def _paragraph(node: Mapping[str, object], config: ADFConfig) -> Iterator[_Found]:
    text = _collapse(_inline_text(_children(node), config))
    if text:
        yield _Found(kind=BlockKind.PROSE, text=text)


def _code_block(node: Mapping[str, object], config: ADFConfig) -> Iterator[_Found]:
    # Not collapsed: indentation and line breaks are part of what the code says.
    text = _inline_text(_children(node), config).strip("\n")
    if text:
        yield _Found(kind=BlockKind.CODE, text=text, lang=_text_attr(node, "language") or None)


def _panel(node: Mapping[str, object], config: ADFConfig) -> Iterator[_Found]:
    """A panel keeps its severity: a warning panel is not ordinary prose."""
    text = "\n".join(found.text for found in _walk(_children(node), config))
    if not text.strip():
        return
    yield _Found(
        kind=BlockKind.PANEL,
        text=text,
        metadata={"panel_type": _text_attr(node, "panelType") or "info"},
    )


def _blockquote(node: Mapping[str, object], config: ADFConfig) -> Iterator[_Found]:
    text = "\n".join(found.text for found in _walk(_children(node), config))
    if text.strip():
        yield _Found(kind=BlockKind.PROSE, text=text)


def _list(node: Mapping[str, object], config: ADFConfig) -> Iterator[_Found]:
    lines = list(_list_lines(node, config, 0))
    if lines:
        yield _Found(kind=BlockKind.LIST, text="\n".join(lines))


def _list_lines(node: Mapping[str, object], config: ADFConfig, depth: int) -> Iterator[str]:
    """A list rendered with its nesting preserved as indentation."""
    marker = "1." if _node_type(node) == "orderedList" else "-"
    for item in _children(node):
        own = [child for child in _children(item) if _node_type(child) not in _LIST_TYPES]
        text = _collapse(" ".join(_inline_text(_children(child), config) for child in own))
        if text:
            yield f"{_INDENT * depth}{marker} {text}"
        for nested in _children(item):
            if _node_type(nested) in _LIST_TYPES:
                yield from _list_lines(nested, config, depth + 1)


def _table(node: Mapping[str, object], config: ADFConfig) -> Iterator[_Found]:
    """A table rendered one row per line, cells separated by pipes.

    ``header_rows`` is counted from the markup rather than guessed, because a table too large
    for one chunk is split by rows with its header repeated into every part
    (``docs/parsing.md`` §4.2), and a part carrying the wrong rows mislabels every column.

    **``rows`` is emitted beside it, and until it was this docstring described something that
    did not happen.** ``header_rows`` alone is not enough: ``_split_table`` reads ``rows`` first
    and falls back to prose splitting when it is absent, so the header-repeating split above was
    never reached and a long table was cut wherever the token budget landed — mid-row, and
    sometimes mid-cell.
    """
    rows: list[str] = []
    header_rows = 0
    still_header = True
    for row in _children(node):
        cells = _children(row)
        rendered = [
            _collapse(" ".join(found.text for found in _walk(_children(cell), config)))
            for cell in cells
        ]
        if not any(rendered):
            continue
        if still_header and all(_node_type(cell) == "tableHeader" for cell in cells):
            header_rows += 1
        else:
            still_header = False
        rows.append(" | ".join(rendered))
    if rows:
        yield _Found(
            kind=BlockKind.TABLE,
            text="\n".join(rows),
            metadata={"header_rows": header_rows, "rows": [*rows]},
        )


def _media(node: Mapping[str, object], config: ADFConfig) -> Iterator[_Found]:
    """An attachment reference. Its alt text or filename is content; its bytes are not."""
    del config
    inner = [node] if _node_type(node) == "media" else _children(node)
    for child in inner:
        text = (
            _text_attr(child, "alt") or _text_attr(child, "__fileName") or _text_attr(child, "id")
        )
        if text:
            metadata: Metadata = {"media_type": _text_attr(child, "type") or "file"}
            yield _Found(kind=BlockKind.MEDIA, text=text, metadata=metadata)


def _card(node: Mapping[str, object], config: ADFConfig) -> Iterator[_Found]:
    text = _card_text(node, config)
    if text:
        yield _Found(kind=BlockKind.MEDIA, text=text, metadata={"url": text})


def _expand(node: Mapping[str, object], config: ADFConfig) -> Iterator[_Found]:
    """Collapsed content is still content, so the body is walked and the title kept."""
    title = _text_attr(node, "title")
    if title:
        yield _Found(kind=BlockKind.PROSE, text=title)
    yield from _walk(_children(node), config)


def _extension(node: Mapping[str, object], config: ADFConfig) -> Iterator[_Found]:
    """A macro with no body of its own: what it is, named, rather than silently dropped."""
    del config
    key = _text_attr(node, "extensionKey")
    if key:
        yield _Found(kind=BlockKind.MEDIA, text=key, metadata={"macro": key})


def _container(node: Mapping[str, object], config: ADFConfig) -> Iterator[_Found]:
    yield from _walk(_children(node), config)


_HANDLERS: Mapping[str, _Handler] = {
    "heading": _heading,
    "paragraph": _paragraph,
    "codeBlock": _code_block,
    "panel": _panel,
    "blockquote": _blockquote,
    "bulletList": _list,
    "orderedList": _list,
    "table": _table,
    "mediaSingle": _media,
    "mediaGroup": _media,
    "media": _media,
    "blockCard": _card,
    "embedCard": _card,
    "expand": _expand,
    "nestedExpand": _expand,
    "bodiedExtension": _container,
    "extension": _extension,
    "layoutSection": _container,
    "layoutColumn": _container,
    "doc": _container,
}


def _walk(nodes: Sequence[Mapping[str, object]], config: ADFConfig) -> Iterator[_Found]:
    """Yield the blocks in a run of ADF nodes, in document order."""
    for node in nodes:
        handler = _HANDLERS.get(_node_type(node))
        if handler is not None:
            yield from handler(node, config)
        else:
            yield from _unknown(node, config)


def _unknown(node: Mapping[str, object], config: ADFConfig) -> Iterator[_Found]:
    """A node type this parser does not know.

    ADF gains node types, and a document containing one is not a broken document. Its
    children are walked so the text inside survives, and a node whose children are all inline
    becomes a paragraph, because that is what a node with inline content is. Refusing the
    document instead would lose everything else in it over one unfamiliar wrapper.
    """
    children = _children(node)
    if children and all(_node_type(child) in _INLINE_TYPES for child in children):
        text = _collapse(_inline_text(children, config))
        if text:
            yield _Found(kind=BlockKind.PROSE, text=text)
        return
    yield from _walk(children, config)


# --- anchors -----------------------------------------------------------------------------


def _confluence_fragment(title: str) -> str:
    """The anchor Confluence derives from a heading, before duplicate numbering.

    Not :func:`~manicule.parsers.base.slugify`, which lowercases: a URL fragment is
    case-sensitive, so a lowercased anchor does not deep-link to a Confluence heading and
    would produce exactly the citation that looks precise and lands at the top of the page.
    """
    hyphenated = _WHITESPACE.sub("-", title.strip())
    return _NOT_IN_ANCHOR.sub("", hyphenated).strip("-")


def _assemble(found: Sequence[_Found], title: str) -> _Reading:
    """Turn the walk's output into blocks, giving each the anchor its section allows."""
    stack = HeadingStack()
    paths: list[tuple[str, ...]] = []
    fragments: list[str | None] = []
    seen: dict[str, int] = {}
    for item in found:
        if item.kind is not BlockKind.HEADING:
            continue
        paths.append(stack.push(item.level, item.text))
        base = _confluence_fragment(item.text)
        if not base:
            fragments.append(None)
            continue
        # Duplicates are numbered from the *second* occurrence, as Confluence numbers them:
        # counting from the first would rename the section that was there before.
        repeat = seen.get(base, 0)
        seen[base] = repeat + 1
        fragments.append(base if repeat == 0 else f"{base}-{repeat}")

    counts: dict[tuple[str, ...], int] = {}
    for path in paths:
        counts[path] = counts.get(path, 0) + 1

    texts = _section_texts(found)
    sections = [
        _Section(path=path, fragment=fragment, text=text)
        for path, fragment, text in zip(paths, fragments, texts[1:], strict=True)
    ]
    anchors = [_section_anchor(section, counts) for section in sections]
    document_anchor = _preamble_anchor(title, counts)

    blocks: list[ParsedBlock] = []
    position = -1
    for item in found:
        if item.kind is BlockKind.HEADING:
            position += 1
        blocks.append(
            ParsedBlock(
                kind=item.kind,
                text=item.text,
                anchor=anchors[position] if position >= 0 else document_anchor,
                heading_path=sections[position].path if position >= 0 else (),
                lang=item.lang,
                metadata=item.metadata,
            )
        )
    return _Reading(
        blocks=tuple(blocks),
        by_fragment={
            section.fragment: section for section in sections if section.fragment is not None
        },
        by_path=_group(sections),
        title=title,
        preamble=texts[0],
    )


def _section_texts(found: Sequence[_Found]) -> list[str]:
    """The text each section owns, the first entry being everything before the first heading.

    A section runs to the next heading of **any** level and does not include its subsections.
    A section resolved with its subsections inside it is larger than the text it addresses by
    however deep the nesting goes, which fails the tightness bound in ``docs/parsing.md`` §3.3
    on any document with structure.
    """
    groups: list[list[str]] = [[]]
    for item in found:
        if item.kind is BlockKind.HEADING:
            groups.append([])
        groups[-1].append(item.text)
    return ["\n\n".join(group) for group in groups]


def _group(sections: Sequence[_Section]) -> Mapping[tuple[str, ...], tuple[_Section, ...]]:
    grouped: dict[tuple[str, ...], tuple[_Section, ...]] = {}
    for section in sections:
        grouped[section.path] = (*grouped.get(section.path, ()), section)
    return grouped


def _section_anchor(section: _Section, counts: Mapping[tuple[str, ...], int]) -> Anchor:
    if section.fragment is None and counts[section.path] > 1:
        return Unlocated(
            reason=f"ambiguous heading path {' > '.join(section.path)!r}: it names more than "
            f"one section and its heading text yields no Confluence anchor to tell them apart"
        )
    return HeadingAnchor(path=section.path, fragment=section.fragment)


def _preamble_anchor(title: str, counts: Mapping[tuple[str, ...], int]) -> Anchor:
    """What content before the first heading is anchored to.

    The page title is a real heading path element — it is what a Confluence breadcrumb ends
    with — so text under it is addressed by it, at page level, with no fragment because there
    is no section to deep-link to. Two cases cannot honestly use it: a page fetched without a
    title, and one where a heading already claims that path.
    """
    if not title:
        return Unlocated(
            reason="content precedes the first heading and the page arrived without a title "
            "to address it by"
        )
    if counts.get((title,)):
        return Unlocated(
            reason=f"content precedes the first heading and a section is itself called "
            f"{title!r}, so the page-level path would name both"
        )
    return HeadingAnchor(path=(title,), fragment=None)
