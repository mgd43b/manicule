"""HTML, located by the heading structure the markup already carries.

The whole reason to parse HTML with a real parser rather than a tag-stripping regular
expression is that ``<h2>``, ``<table>`` and ``<pre>`` are facts. A heading path is taken
from the heading elements the markup provides; a table keeps its rows; code keeps its
indentation. Everything downstream — chunk boundaries, breadcrumbs, citations — is built on
those facts rather than on a guess made after the structure was thrown away.

**Fragments come from the author or not at all.** A ``HeadingAnchor`` carries a fragment only
where the page defines one: an ``id=`` on the heading, or the empty anchor element
immediately before it that older documents use for the same purpose. Synthesising a slug
would produce a citation that looks precise and lands at the top of a page manicule does not
serve, so a heading with no published address gets ``fragment=None`` and cites the document
(``docs/parsing.md`` §2.5). Where that leaves the path ambiguous — two sections with the same
heading path and no fragment to tell them apart — the blocks are
:class:`~manicule.core.anchors.Unlocated` with that as the reason, because an anchor nobody
can resolve is worse than an admission that there is none.

**Only the lexbor backend is imported.** ``selectolax`` ships two engines in one wheel and
the other is LGPL-2.1; manicule imports lexbor, which is Apache-2.0 (``docs/parsing.md``
§12).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from selectolax.lexbor import LexborHTMLParser, LexborNode

from manicule.core.anchors import Anchor, HeadingAnchor, Unlocated
from manicule.core.content import BlockKind, Metadata, ParsedBlock, RawDocument
from manicule.parsers.base import HeadingStack, ParserProfile, decode

__all__ = ["WEB_MEDIA_TYPES", "WebConfig", "WebParser"]

WEB_MEDIA_TYPES = frozenset({"text/html", "application/xhtml+xml"})

_HEADING_LEVELS: Mapping[str, int] = {f"h{level}": level for level in range(1, 7)}

_BLOCK_KINDS: Mapping[str, BlockKind] = {
    "p": BlockKind.PROSE,
    "blockquote": BlockKind.PROSE,
    "pre": BlockKind.CODE,
    "ul": BlockKind.LIST,
    "ol": BlockKind.LIST,
    "dl": BlockKind.LIST,
    "table": BlockKind.TABLE,
    "figure": BlockKind.MEDIA,
    "img": BlockKind.MEDIA,
    "video": BlockKind.MEDIA,
    "audio": BlockKind.MEDIA,
}
"""Elements that are a block on their own. Anything else is either inline, or a container
whose children are walked."""

_INLINE_TAGS = frozenset(
    {
        "a", "abbr", "b", "bdi", "bdo", "br", "cite", "code", "data", "dfn", "em", "i",
        "kbd", "mark", "q", "rp", "rt", "ruby", "s", "samp", "small", "span", "strong",
        "sub", "sup", "time", "u", "var", "wbr",
    }
)  # fmt: skip
"""Elements that are part of the sentence around them. Their text joins the run being
collected rather than starting a block, so ``<p>`` is not split at every ``<strong>``."""

_INDENT = "  "

_TEXT_NODE = "-text"


class WebConfig(BaseModel):
    """Configuration for :class:`WebParser`."""

    drop_tags: frozenset[str] = Field(
        default=frozenset({"script", "style", "noscript", "template"}),
        description="Elements removed, with their contents, before any text is extracted.",
    )
    """These carry no prose. Indexing a script body puts identifiers and punctuation into the
    vector, where they match queries by accident and cite a line no reader ever saw."""


@dataclass(frozen=True, slots=True)
class _Found:
    """One element the walk recognised, before sections have been worked out."""

    kind: BlockKind
    text: str
    level: int = 0
    fragment: str | None = None
    lang: str | None = None
    metadata: Metadata = field(default_factory=Metadata)


@dataclass(frozen=True, slots=True)
class _Section:
    """A heading, the address it publishes, and the text it owns up to the next heading."""

    path: tuple[str, ...]
    fragment: str | None
    text: str


@dataclass(frozen=True, slots=True)
class _Reading:
    """One pass over the source: the blocks it yields and what each anchor resolves to."""

    blocks: tuple[ParsedBlock, ...]
    by_fragment: Mapping[str, _Section]
    by_path: Mapping[tuple[str, ...], tuple[_Section, ...]]
    title: str
    preamble: str


class WebParser:
    """Parses HTML into blocks anchored to the page's own heading structure."""

    media_types = WEB_MEDIA_TYPES
    profile = ParserProfile(name="html", max_unlocated_ratio=0.05, max_pagelevel_ratio=None)

    def __init__(self, config: WebConfig) -> None:
        self._config = config

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        """Yield blocks in reading order.

        Raises:
            ParseError: The bytes do not decode, so the next parser in the chain gets a turn.
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
        tree = LexborHTMLParser(decode(raw))
        for tag in sorted(self._config.drop_tags):
            for node in tree.css(tag):
                node.decompose()
        title = _title(raw, tree)
        body = tree.body
        found = list(_walk(body)) if body is not None else []
        return _assemble(found, title)


def _title(raw: RawDocument, tree: LexborHTMLParser) -> str:
    """The document's title: what the connector reported, else the page's own ``<title>``.

    The connector's is preferred because it is what the rest of the record is keyed on; the
    ``<title>`` element is the honest fallback for a page fetched with nothing alongside it.
    """
    declared = raw.metadata.get("title")
    if isinstance(declared, str) and declared.strip():
        return declared.strip()
    # `css` rather than `css_first`: the stub's first overload defaults to the strict form,
    # so a bare `css_first` is typed as always finding something, which a `<title>` is not.
    element = tree.css("title")
    return _collapse(element[0].text(deep=True)) if element else ""


def _walk(node: LexborNode) -> Iterator[_Found]:
    """Yield the blocks under ``node`` in document order.

    Text that is not wrapped in a block element is collected into the run around it rather
    than dropped: a paragraph written without a ``<p>`` is still a paragraph, and a document
    that loses it indexes as shorter than it is.
    """
    pending: list[str] = []
    for child in node.iter(include_text=True):
        if child.is_text_node:
            pending.append(child.text_content or "")
            continue
        if child.is_comment_node:
            continue
        tag = child.tag or ""
        if tag in _INLINE_TAGS:
            pending.append(child.text(deep=True))
            continue
        yield from _flush(pending)
        if tag in _HEADING_LEVELS:
            heading = _heading(child, tag)
            # A heading of only an image or an anchor names nothing, so it cannot be a path
            # element: an empty one would reach the embedder through the breadcrumb as a
            # heading nobody wrote. Its section is the enclosing one.
            if heading.text:
                yield heading
        elif tag in _BLOCK_KINDS:
            found = _block(child, tag)
            if found is not None:
                yield found
        else:
            yield from _walk(child)
    yield from _flush(pending)


def _flush(pending: list[str]) -> Iterator[_Found]:
    """Emit the loose inline run collected so far, if it holds anything, and clear it."""
    text = _collapse("".join(pending))
    pending.clear()
    if text:
        yield _Found(kind=BlockKind.PROSE, text=text)


def _heading(node: LexborNode, tag: str) -> _Found:
    return _Found(
        kind=BlockKind.HEADING,
        text=_collapse(node.text(deep=True)),
        level=_HEADING_LEVELS[tag],
        fragment=_published_fragment(node),
    )


def _published_fragment(node: LexborNode) -> str | None:
    """The address this heading already has, or ``None`` if the author published none.

    Two forms count, and both are the author's: an ``id`` on the heading itself, and the
    empty anchor element placed immediately before it — the pattern that predates ``id`` on
    arbitrary elements and is still emitted by several documentation generators.
    """
    if node.id:
        return node.id
    previous = node.prev
    while previous is not None and previous.tag == _TEXT_NODE:
        previous = previous.prev
    if previous is None or previous.tag != "a" or previous.text(deep=True).strip():
        return None
    named = previous.attributes.get("id") or previous.attributes.get("name")
    return named or None


def _block(node: LexborNode, tag: str) -> _Found | None:
    """One non-heading block, or ``None`` when the element carries no text to index."""
    kind = _BLOCK_KINDS[tag]
    metadata: Metadata = {}
    lang: str | None = None
    if kind is BlockKind.TABLE:
        text = _table_text(node)
        metadata = {"header_rows": _header_rows(node)}
    elif kind is BlockKind.LIST:
        text = "\n".join(_list_lines(node, 0))
    elif kind is BlockKind.CODE:
        # Not collapsed: indentation is part of what the code says.
        text = node.text(deep=True).strip("\n")
        lang = _code_language(node)
    elif kind is BlockKind.MEDIA:
        text = _media_text(node)
        source = node.attributes.get("src")
        metadata = {"src": source} if source else {}
    else:
        text = _collapse(node.text(deep=True))
    if not text.strip():
        return None
    return _Found(kind=kind, text=text, lang=lang, metadata=metadata)


def _collapse(text: str) -> str:
    """Whitespace as a reader sees it: runs become single spaces, ends are trimmed."""
    return " ".join(text.split())


def _media_text(node: LexborNode) -> str:
    """What an image or figure contributes: its alt text or caption, never its bytes.

    With OCR out of scope, an image with neither is not indexable, and emitting an empty
    block for it would put a vector of nothing into the index.
    """
    alt = node.attributes.get("alt")
    if alt and alt.strip():
        return _collapse(alt)
    return _collapse(node.text(deep=True))


def _code_language(node: LexborNode) -> str | None:
    """The language a ``<pre>`` declares, from the conventional ``language-`` class."""
    inner = node.css("code")
    code = inner[0] if inner else node
    classes = (code.attributes.get("class") or "").split()
    for name in classes:
        for prefix in ("language-", "lang-"):
            if name.startswith(prefix) and len(name) > len(prefix):
                return name[len(prefix) :]
    return None


def _header_rows(node: LexborNode) -> int:
    """How many leading rows are header rows.

    Read from the markup — ``<thead>``, or a first row made of ``<th>`` — rather than guessed
    from the first row being bold, because a split table repeats these rows into every part
    (``docs/parsing.md`` §4.2) and repeating the wrong ones mislabels every column.
    """
    head = node.css("thead")
    if head:
        return len(head[0].css("tr"))
    rows = node.css("tr")
    return 1 if rows and rows[0].css("th") else 0


def _table_text(node: LexborNode) -> str:
    """A table rendered one row per line, cells separated by pipes.

    Keeping the row structure is the point of parsing the table at all: a table flattened to
    a run of words loses which value belongs to which column.
    """
    rows: list[str] = []
    for row in node.css("tr"):
        cells = [_collapse(cell.text(deep=True)) for cell in row.css("th, td")]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _list_lines(node: LexborNode, depth: int) -> Iterator[str]:
    """A list rendered with its nesting preserved as indentation."""
    marker = "1." if node.tag == "ol" else "-"
    for item in node.iter():
        if item.tag not in {"li", "dt", "dd"}:
            continue
        own = _collapse(
            "".join(
                part.text_content or ""
                if part.is_text_node
                else ("" if part.tag in {"ul", "ol"} else part.text(deep=True))
                for part in item.iter(include_text=True)
            )
        )
        if own:
            yield f"{_INDENT * depth}{marker} {own}"
        for nested in item.iter():
            if nested.tag in {"ul", "ol"}:
                yield from _list_lines(nested, depth + 1)


def _assemble(found: Sequence[_Found], title: str) -> _Reading:
    """Turn the walk's output into blocks, giving each the anchor its section allows."""
    stack = HeadingStack()
    paths: list[tuple[str, ...]] = []
    fragments: list[str | None] = []
    taken: set[str] = set()
    for item in found:
        if item.kind is not BlockKind.HEADING:
            continue
        paths.append(stack.push(item.level, item.text))
        # A repeated id addresses the first element that carries it, so it is not this
        # section's address even though the author wrote it here.
        usable = item.fragment if item.fragment and item.fragment not in taken else None
        if usable:
            taken.add(usable)
        fragments.append(usable)

    counts: dict[tuple[str, ...], int] = {}
    for path in paths:
        counts[path] = counts.get(path, 0) + 1

    texts = _section_texts(found)
    sections = [
        _Section(path=path, fragment=fragment, text=text)
        for path, fragment, text in zip(paths, fragments, texts[1:], strict=True)
    ]
    preamble = texts[0]
    anchors = [_section_anchor(section, counts) for section in sections]
    document_anchor = _preamble_anchor(title, counts)

    blocks: list[ParsedBlock] = []
    position = -1
    for item in found:
        if item.kind is BlockKind.HEADING:
            position += 1
        anchor = anchors[position] if position >= 0 else document_anchor
        path = sections[position].path if position >= 0 else ()
        blocks.append(
            ParsedBlock(
                kind=item.kind,
                text=item.text,
                anchor=anchor,
                heading_path=path,
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
        preamble=preamble,
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
            f"one section and the page publishes no id= to tell them apart"
        )
    return HeadingAnchor(path=section.path, fragment=section.fragment)


def _preamble_anchor(title: str, counts: Mapping[tuple[str, ...], int]) -> Anchor:
    """What content before the first heading is anchored to.

    The document title is a real heading path element — it is what a breadcrumb starts with —
    so text under it is addressed by it, at document level, with no fragment because there is
    no section to deep-link to. Two cases cannot honestly use it: a document with no title,
    and one where a real heading already claims that path.
    """
    if not title:
        return Unlocated(
            reason="content precedes the first heading and the document has no title to "
            "address it by"
        )
    if counts.get((title,)):
        return Unlocated(
            reason=f"content precedes the first heading and a section is itself called "
            f"{title!r}, so the document-level path would name both"
        )
    return HeadingAnchor(path=(title,), fragment=None)
