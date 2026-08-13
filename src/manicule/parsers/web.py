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
from typing import Final

from selectolax.lexbor import LexborHTMLParser, LexborNode

from manicule.core.anchors import Anchor, HeadingAnchor, Unlocated
from manicule.core.content import BlockKind, Metadata, ParsedBlock, RawDocument
from manicule.parsers.base import HeadingStack, ParserProfile, decode
from manicule.parsers.config import WEB_MEDIA_TYPES, WebConfig
from manicule.parsers.inline import (
    LINE_BREAK,
    InlinePart,
    collapse,
    collapse_lines,
    collapse_run,
    item_prefix,
)

__all__ = ["WEB_MEDIA_TYPES", "WebConfig", "WebParser", "recover_cdata"]

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

_BREAK = "br"

_ITEM_TAGS: Final = frozenset({"li", "dt", "dd"})
"""Elements that are a list item: bulleted, ordered, or a definition list's term and body."""

_NESTED_LISTS = frozenset({"ul", "ol"})
"""Direct children a list item renders separately rather than as part of its own text."""

_INDENT = "  "

_TEXT_NODE = "-text"


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
        tree = LexborHTMLParser(recover_cdata(decode(raw)))
        for tag in sorted(self._config.drop_tags):
            for node in tree.css(tag):
                node.decompose()
        title = _title(raw, tree)
        body = tree.body
        found = list(_walk(body)) if body is not None else []
        return _assemble(found, title)


_CDATA_OPEN: Final = "<![CDATA["
_CDATA_CLOSE: Final = "]]>"


def recover_cdata(document: str) -> str:
    """Rewrite every ``<![CDATA[…]]>`` section as the text it holds, escaped.

    Public because two callers need it and one of them is a test that asserts the *rendered*
    document contains no element the recovery could have created — a property only checkable on the
    string this returns. A storage-format parser will be the second real caller.

    **This recovers content an HTML parser deletes.** ``<![CDATA[…]]>`` is not a construct HTML has
    outside foreign content, so a conforming parser reparses it as a *bogus comment* — lexbor
    produces ``<!--[CDATA[…]]-->`` — and everything inside is gone from the document. Not degraded:
    absent, with nothing raised and nothing reported.

    It shipped as a live defect rather than a hypothetical. Confluence wraps the body of every
    ``code``, ``noformat`` and ``graphviz`` macro in CDATA, and storage-format bodies are routed as
    ``text/html`` (``docs/connectors/confluence.md`` §4), so every code block on every Server or
    Data Center page has been missing from the index while a fragment of it was indexed as prose.
    The Graphviz case is worse than a clean loss: ``->`` inside the body terminates the bogus
    comment early, so the opening is swallowed and the tail escapes as the page's own words.

    **Not a Confluence special case, and deliberately so.** A CDATA section in any HTML document is
    content its author intended, and losing it is wrong for every input — which matters here
    because there is no way to tell storage format from HTML at this point: storage format *is*
    ``text/html``. A fix that needed to know would have needed a media type, and a media type is
    only honest once a parser gives it meaning.

    The escaping is what keeps this a recovery rather than an injection. CDATA exists precisely so
    a body can contain ``<`` and ``&`` without being markup, so the recovered text is escaped
    before it re-enters the document — otherwise ``<![CDATA[<script>…]]>`` would be *promoted* from
    inert text to a live element, turning a content-loss bug into an execution one.

    Unterminated sections are left exactly as they are. A document whose CDATA never closes is
    malformed, and guessing where the author meant it to end would invent content; the existing
    behaviour — the parser's own error recovery — is the honest answer.
    """
    if _CDATA_OPEN not in document:
        return document
    out: list[str] = []
    rest = document
    while True:
        before, opened, after = rest.partition(_CDATA_OPEN)
        if not opened:
            out.append(before)
            return "".join(out)
        inner, closed, remainder = after.partition(_CDATA_CLOSE)
        if not closed:
            # Unterminated: emit the rest untouched rather than guessing where it ended.
            out.append(before)
            out.append(opened)
            out.append(after)
            return "".join(out)
        out.append(before)
        out.append(_escape(inner))
        rest = remainder


def _escape(text: str) -> str:
    """The three characters that would otherwise make recovered text into markup.

    ``&`` first, or the escapes introduced by the other two would themselves be escaped. Quotes are
    left alone: this text becomes a text node, never an attribute value.
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _title(raw: RawDocument, tree: LexborHTMLParser) -> str:
    """The document's title: what the connector reported, else the page's own ``<title>``.

    The connector's is preferred because it is what the rest of the record is keyed on; the
    ``<title>`` element is the honest fallback for a page fetched with nothing alongside it.

    Flattened rather than walked for breaks, and the exemption is the specification's rather
    than a shortcut: ``<title>`` is RCDATA, so a tokenizer produces one text node from its
    whole content and ``<br/>`` inside one is the six characters an author typed. There is no
    element there to preserve.
    """
    declared = raw.metadata.get("title")
    if isinstance(declared, str) and declared.strip():
        return declared.strip()
    # `css` rather than `css_first`: the stub's first overload defaults to the strict form,
    # so a bare `css_first` is typed as always finding something, which a `<title>` is not.
    element = tree.css("title")
    return collapse(element[0].text(deep=True)) if element else ""


def _walk(node: LexborNode) -> Iterator[_Found]:
    """Yield the blocks under ``node`` in document order.

    Text that is not wrapped in a block element is collected into the run around it rather
    than dropped: a paragraph written without a ``<p>`` is still a paragraph, and a document
    that loses it indexes as shorter than it is.
    """
    pending: list[InlinePart] = []
    for child in node.iter(include_text=True):
        if child.is_text_node:
            pending.append(child.text_content or "")
            continue
        if child.is_comment_node:
            continue
        tag = child.tag or ""
        if tag in _INLINE_TAGS:
            pending.extend(_parts(child))
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


def _parts(node: LexborNode, skip: frozenset[str] = frozenset()) -> Iterator[InlinePart]:
    """What ``node`` contributes to the run around it: its text, and the breaks it draws.

    This is what ``text(deep=True)`` is replaced by wherever the result is inline content, and
    the difference is one element. A ``<br>`` has no text of its own, so asking the tree for
    characters returns none and the break is gone before any collapse could have kept it.

    ``skip`` names direct children the caller renders separately — the nested list inside a
    list item — and applies to that level only, as the flatten it replaces did.
    """
    if node.is_text_node:
        yield node.text_content or ""
        return
    if node.is_comment_node:
        return
    if node.tag == _BREAK:
        yield LINE_BREAK
        return
    for child in node.iter(include_text=True):
        if (child.tag or "") in skip:
            continue
        yield from _parts(child)


def _flush(pending: list[InlinePart]) -> Iterator[_Found]:
    """Emit the loose inline run collected so far, if it holds anything, and clear it."""
    text = collapse_lines(pending)
    pending.clear()
    if text:
        yield _Found(kind=BlockKind.PROSE, text=text)


def _heading(node: LexborNode, tag: str) -> _Found:
    return _Found(
        kind=BlockKind.HEADING,
        text=collapse_run(_parts(node)),
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
        rows = _table_rows(node)
        text = "\n".join(rows)
        metadata = {"header_rows": _header_rows(node), "rows": [*rows]}
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
        text = collapse_lines(_parts(node))
    if not text.strip():
        return None
    return _Found(kind=kind, text=text, lang=lang, metadata=metadata)


def _media_text(node: LexborNode) -> str:
    """What an image or figure contributes: its alt text or caption, never its bytes.

    With OCR out of scope, an image with neither is not indexable, and emitting an empty
    block for it would put a vector of nothing into the index.

    One line, because this is the block's label rather than its prose: a caption is what a
    reader is shown beside the picture, and a media block has no paragraphs to keep apart.
    """
    alt = node.attributes.get("alt")
    if alt and alt.strip():
        return collapse(alt)
    return collapse_run(_parts(node))


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


def _table_rows(node: LexborNode) -> list[str]:
    """A table's rows, one rendered line each, cells separated by pipes.

    Keeping the row structure is the point of parsing the table at all: a table flattened to
    a run of words loses which value belongs to which column.

    Returned as a list so the row boundaries reach the chunker in ``rows`` metadata. Without
    them :meth:`~manicule.chunking.chunker.StructuralChunker._split_table` falls back to prose
    splitting and cuts mid-row, which produces a line that still looks like ``TERM | expansion``
    while its expansion is a fragment.

    A cell is one line, and that is what a break inside one becomes a space for. ``text`` is
    ``"\\n".join(rows)`` (``docs/parsing.md`` §4.4), so a newline inside a cell would put a row
    in ``text`` that is not in ``rows`` and the two would stop describing the same table.
    """
    rows: list[str] = []
    for row in node.css("tr"):
        cells = [collapse_run(_parts(cell)) for cell in row.css("th, td")]
        if any(cells):
            rows.append(" | ".join(cells))
    return rows


def _list_lines(node: LexborNode, depth: int) -> Iterator[str]:
    """A list rendered with its nesting preserved as indentation.

    One line per item, for the reason a table keeps one line per row: the newline is what says
    where an item ends, so a break inside one is a space.
    """
    marker = "1." if node.tag == "ol" else "-"
    for item in node.iter():
        if item.tag not in _ITEM_TAGS:
            continue
        own = collapse_run(_parts(item, skip=_NESTED_LISTS))
        if own:
            yield f"{_INDENT * depth}{item_prefix(item.tag or '', marker)}{own}"
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
