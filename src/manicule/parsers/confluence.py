"""Confluence storage format, read as Confluence rather than as generic HTML.

Storage format is XHTML with Confluence's own ``ac:`` and ``ri:`` vocabulary mixed into it. The
HTML parser reads the XHTML half correctly — headings, paragraphs, lists, tables and ``<pre>``
are the same facts in both — and it is the other half that this module exists for. Routed as
``text/html``, a code macro's declared language, a panel's severity, a task's state and a
Graphviz macro's engine were all flattened into prose, and one thing worse than flattened.

**Macro parameters were being indexed as document text.** ``<ac:parameter ac:name="language">
python</ac:parameter>`` is not something anybody wrote on the page; it is the macro's
configuration. Walked as generic HTML it is an unknown element with a text node inside, so it
became a prose block, went into the vector, and was quotable in a citation as though the page
had said it. The engine name of a diagram, the language of a code block and — the case that
makes this more than untidy — the **JQL query** of a Jira macro were all indexed as things the
document says. This parser consumes parameters as configuration and never as text, with one
exception drawn as an explicit table rather than applied as a judgement: :data:`RENDERED_PARAMETERS`
names which macro's which parameter Confluence actually draws on the page. Anything not in it stays
configuration, so a macro nobody has enumerated fails in the harmless direction.

**Nothing here is rendered, executed or evaluated.** A storage-format body is authored by anyone
with write access to the page, so every string in it is untrusted input: macro bodies, DOT
source, parameter values and link targets alike. Bodies are recovered as *text* and stay text —
:func:`~manicule.parsers.web.recover_cdata` escapes a CDATA section before it re-enters the
document precisely so ``<![CDATA[<script>…]]>`` is not promoted from inert text to a live
element, and parameter values are read out of attributes and text nodes and never written back
into markup. No DOT is laid out, no macro is expanded, no attachment is fetched.

**Unsupported macros leave a placeholder rather than a hole.** A macro this parser has no reader
for still says *something* was there, and a reader who cannot tell "the page was empty here"
from "we could not read this" has been misled by omission. Where such a macro carries a body,
the body is kept as well: the placeholder names what could not be interpreted, and never stands
in for content that could have been preserved.

**Its own table and list rendering, which is not duplication.** The obvious economy is to reuse
the HTML parser's, and it is wrong: those flatten a cell with ``text(deep=True)``, which would
pull an ``ac:parameter`` value straight out of a macro nested inside a table cell. The two
implementations differ exactly where the defect lives, so the parameter rule holds inside cells
and list items and not merely at the top level.

Anchors follow the same rule as the HTML parser: a fragment comes from the author or not at all
(``docs/parsing.md`` §2.5). Storage format publishes them in two ways — an ``id`` on the heading
and the ``anchor`` macro placed before it — and Confluence's own rendered ids are synthesised at
render time from the heading text, which is a guess this parser does not make.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field

from selectolax.lexbor import LexborHTMLParser, LexborNode

from manicule.core.anchors import Anchor, HeadingAnchor, Unlocated
from manicule.core.content import BlockKind, JsonValue, Metadata, ParsedBlock, RawDocument
from manicule.parsers.base import HeadingStack, ParserProfile, decode
from manicule.parsers.config import CONFLUENCE_MEDIA_TYPES, ConfluenceConfig
from manicule.parsers.web import recover_cdata

__all__ = [
    "CONFLUENCE_MEDIA_TYPES",
    "INTERPRETED_MACROS",
    "ConfluenceConfig",
    "ConfluenceStorageParser",
    "close_empty_elements",
    "dot_parse_warning",
]

_HEADING_LEVELS: Mapping[str, int] = {f"h{level}": level for level in range(1, 7)}

MACRO = "ac:structured-macro"
PARAMETER = "ac:parameter"
PLAIN_TEXT_BODY = "ac:plain-text-body"
RICH_TEXT_BODY = "ac:rich-text-body"
TASK_LIST = "ac:task-list"
TASK = "ac:task"
TASK_ID = "ac:task-id"
TASK_STATUS = "ac:task-status"
TASK_BODY = "ac:task-body"
LINK = "ac:link"
LINK_BODY = "ac:link-body"
PLAIN_TEXT_LINK_BODY = "ac:plain-text-link-body"
IMAGE = "ac:image"

_CONFIGURATION_ONLY = frozenset({PARAMETER, TASK_ID, TASK_STATUS})
"""Elements whose text is configuration or machine state, never document text.

The whole of the defect this parser was written for. ``ac:parameter`` holds a macro's settings;
``ac:task-id`` holds a database key; ``ac:task-status`` holds ``complete`` or ``incomplete``,
which is a *property of* a task rather than a line of the page — it appeared in the index as its
own one-word block reading "complete"."""

RENDERED_PARAMETERS: Mapping[str, tuple[str, ...]] = {
    "code": ("title",),
    "noformat": ("title",),
    "expand": ("title",),
    "info": ("title",),
    "note": ("title",),
    "panel": ("title",),
    "tip": ("title",),
    "warning": ("title",),
}
"""The exception to "a parameter is never text": which macro's which parameter a reader sees.

A panel's ``title`` is drawn in the panel's own header, an ``expand``'s is the clickable label,
and a ``code`` macro's is the caption above the block. A reader sees each of them, quotes them and
searches for them, so they are content that happens to be *carried* as a parameter. ``language``,
``engine`` and ``jqlQuery`` are rendered nowhere — they change what the macro does.

**An enumeration rather than a rule applied per macro, and it fails safe.** "Is this rendered?"
answered by judgement gets answered differently by the next person, and it fails in the direction
that puts a JQL query or a user's name in a citation. A table fails the other way: a macro nobody
has added an entry for keeps its parameters out of the index, which is the harmless mistake.

Writing it out found two silent discards that a global rule had hidden — ``expand`` and ``code``
both render a title, and both were being dropped while their bodies were kept. That is the value
of the enumeration: it makes each answer visible enough to be wrong in review."""

_PANEL_SEVERITIES: Mapping[str, str] = {
    "info": "info",
    "note": "note",
    "tip": "tip",
    "warning": "warning",
    "panel": "none",
}
"""Panel macros and the severity each declares.

``panel`` is the unstyled one and says ``none`` rather than being left absent: a reader
filtering for panels that carry a severity should not have to distinguish "no severity" from
"this key was never written"."""

_CODE_MACROS = frozenset({"code", "noformat"})

GRAPHVIZ_MACROS = frozenset({"graphviz", "graphviz-dot"})

DOT_LANGUAGE = "dot"

_DEFAULT_DOT_ENGINE = "dot"
"""What Graphviz itself defaults to when a macro declares no engine.

Recorded explicitly rather than left absent, because "the author did not choose" and "the author
chose dot" produce the same diagram and a consumer should not have to know which happened."""

_ANCHOR_MACRO = "anchor"

_TOC_MACROS = frozenset({"toc", "children", "pagetree"})
"""Macros whose output is generated navigation rather than authored content.

They are unsupported in the sense that matters — there is nothing to preserve, because the page
does not contain the text they would produce — so they get a placeholder and no body."""

INTERPRETED_MACROS: frozenset[str] = (
    _CODE_MACROS | GRAPHVIZ_MACROS | frozenset(_PANEL_SEVERITIES) | frozenset({_ANCHOR_MACRO})
)
"""Every macro name this parser reads as something other than an opaque placeholder.

**Derived from the dispatch tables rather than written out beside them**, so the declaration
cannot come to disagree with the behaviour. One caller needs it and is in another package: the
snapshot connector records which of a page's macros will not be understood, and a second, hand-kept
list of that answer would go stale the first time a macro was taught here and not there — which is
the same reasoning that makes the connector import its media type from the registration module
rather than spelling it again.

``tests/parsers/test_confluence.py`` checks the claim by parsing one of each rather than by reading
this expression, because a set that agrees with itself proves nothing."""


_INLINE_TAGS = frozenset(
    {
        "a", "abbr", "b", "bdi", "bdo", "br", "cite", "code", "data", "dfn", "em", "i",
        "kbd", "mark", "q", "rp", "rt", "ruby", "s", "samp", "small", "span", "strong",
        "sub", "sup", "time", "u", "var", "wbr",
    }
)  # fmt: skip

_ATOMIC = frozenset({"table", "ul", "ol", "dl", "pre", TASK_LIST, MACRO, IMAGE})
"""Elements that are a block on their own. Anything else is either inline, or a container whose
children are walked — and a container's recursion collects its own run and flushes at the end,
which is what makes ``<p>`` a block without making it atomic. A macro nested inside a paragraph
therefore becomes its own block instead of being flattened into the sentence around it."""

_INDENT = "  "


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


class ConfluenceStorageParser:
    """Parses Confluence storage format into blocks anchored to the page's headings."""

    media_types = CONFLUENCE_MEDIA_TYPES
    profile = ParserProfile(
        name="confluence-storage", max_unlocated_ratio=0.05, max_pagelevel_ratio=None
    )
    """The HTML parser's budget, for the reason that parser has it: storage format publishes a
    heading id only where the author asked for one, so two sections that share a path with no
    ``anchor`` macro between them are honestly unlocatable."""

    def __init__(self, config: ConfluenceConfig) -> None:
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
        anchor that only resolves against the parser's own memory of a document verifies nothing
        about the document.
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
        tree = LexborHTMLParser(close_empty_elements(recover_cdata(decode(raw))))
        for tag in sorted(self._config.drop_tags):
            for node in tree.css(tag):
                node.decompose()
        title = _title(raw, tree)
        body = tree.body
        found = list(_walk(body, self._config)) if body is not None else []
        return _assemble(found, title)


_EMPTY_ELEMENT = re.compile(
    r"""<((?:ac|ri):[A-Za-z0-9_.-]+)((?:[^<>"']|"[^"]*"|'[^']*')*?)/>""",
)
"""A self-closed Confluence element, with its attributes captured so they can be kept."""


def close_empty_elements(document: str) -> str:
    """Rewrite ``<ri:page …/>`` as ``<ri:page …></ri:page>`` before an HTML engine sees it.

    **HTML does not honour self-closing syntax on unknown elements, and storage format is full of
    them.** ``<ri:user ri:account-id="…"/>`` is a complete element in XML; to an HTML5 tokenizer
    the trailing slash is meaningless, so the element is left *open* and every following sibling
    becomes its child. A link written the way Confluence writes them —

        <ac:link><ri:page ri:content-title="Runbook"/><ac:plain-text-link-body>…

    — therefore parses with the link body nested inside ``ri:page``, and a parser looking for the
    body among the link's own children finds nothing and falls back to the page title. The text
    is not lost, but the sentence reads as the wrong words, which is the failure mode this
    project exists to avoid: plausible, and wrong.

    Normalising here rather than compensating at each lookup keeps the tree shaped the way the
    author wrote it, so there is one place this quirk is known about instead of one per element.
    Only ``ac:`` and ``ri:`` elements are touched: an HTML void element such as ``<br/>`` is
    already handled correctly, and rewriting it would be the bug rather than the fix.

    The attribute run is matched with quoted strings excluded from the "not a bracket" set, so a
    value containing ``/>`` ends the element where it really ends rather than where it first
    looks like it might.
    """
    return _EMPTY_ELEMENT.sub(r"<\1\2></\1>", document)


def _descendants(node: LexborNode, stop: frozenset[str] = frozenset()) -> Iterator[LexborNode]:
    """Every element under ``node``, without descending through ``stop``.

    Direct children are not enough. A macro whose closing tag never arrives — an export
    truncated mid-write is the ordinary case — leaves its siblings nested inside it, so a body or
    a parameter can sit one level deeper than the markup suggests. ``stop`` is what keeps that
    robustness from becoming a bug: it is always ``{MACRO}`` when the question is "this macro's
    own", so an inner code macro's language cannot answer for the outer panel's title.
    """
    for child in node.iter():
        yield child
        if (child.tag or "") not in stop:
            yield from _descendants(child, stop)


def _title(raw: RawDocument, tree: LexborHTMLParser) -> str:
    """The page's title: what the connector reported, else the document's own ``<title>``.

    A storage-format body is a *fragment* — Confluence stores the body without a title element,
    and the title lives on the page record — so in practice this is the connector's answer. The
    element is still consulted for a body someone has saved to a file by hand.
    """
    declared = raw.metadata.get("title")
    if isinstance(declared, str) and declared.strip():
        return declared.strip()
    element = tree.css("title")
    return _collapse(element[0].text(deep=True)) if element else ""


# --- the walk ----------------------------------------------------------------------------------


def _walk(node: LexborNode, config: ConfluenceConfig) -> Iterator[_Found]:
    """Yield the blocks under ``node`` in document order.

    Loose text is collected into the run around it rather than dropped, as in HTML: a paragraph
    written without a ``<p>`` is still a paragraph. What differs is that the run is interrupted by
    Confluence's own block constructs, and that configuration elements never enter it at all.
    """
    pending: list[str] = []
    refs: list[JsonValue] = []
    fragment: str | None = None
    for child in node.iter(include_text=True):
        tag = child.tag or ""
        _record(child, refs)
        inline = _run_text(child, tag, refs)
        if inline is not None:
            pending.append(inline)
            continue
        yield from _flush(pending, refs)
        if tag in _HEADING_LEVELS:
            heading = _heading(child, tag, fragment)
            fragment = None
            if heading.text:
                yield heading
        elif tag == MACRO:
            name = _macro_name(child)
            if name == _ANCHOR_MACRO:
                # Not content: it publishes an address for the heading that follows it.
                fragment = _anchor_name(child) or fragment
                continue
            yield from _macro(child, name, config)
        elif tag in _ATOMIC:
            found = _atomic(child, tag, config)
            if found is not None:
                yield found
        else:
            yield from _walk(child, config)
    yield from _flush(pending, refs)


def _record(node: LexborNode, refs: list[JsonValue] | None) -> None:
    """Note the thing ``node`` points at, if it points at anything.

    **The target, kept beside the text rather than inside it.** A page link reads as its title and
    that is what the sentence should say, but a title is not an address: two spaces can hold pages
    with the same name, and a reader following a citation needs to know which was meant. Recording
    the reference structurally keeps the cross-reference graph without putting a URL in the middle
    of a sentence, where it is noise in the vector and nonsense read aloud.

    **An attachment is referenced and never fetched**, which is the whole of what this does for
    one: parsing reaches no network, and attachment ingestion stays a separate feature that can be
    added without changing this parser.

    A user reference records only that a person was linked. The account id is deliberately absent
    here as it is from the text — recording it in metadata rather than in text would move a
    directory identifier from one part of the index to another and call it redaction.
    """
    if refs is None:
        return
    reference = _reference(node)
    if reference is not None and reference not in refs:
        refs.append(reference)


_REFERENCE_ATTRIBUTES: Mapping[str, tuple[str, str, str]] = {
    "a": ("external", "href", "href"),
    "ri:page": ("page", "ri:content-title", "title"),
    "ri:attachment": ("attachment", "ri:filename", "filename"),
    "ri:url": ("external", "ri:value", "href"),
}
"""Element to (kind, the attribute holding the target, the field it is recorded under).

A table rather than a chain of conditions, so adding a resource type is one row and the shape of
every reference stays visibly the same. ``ri:user`` is absent on purpose: its identifying attribute
is the one thing that must never be recorded, so it cannot be expressed as "copy this attribute"."""


def _reference(node: LexborNode) -> Metadata | None:
    """One link target, as data. ``None`` for anything that is not a reference."""
    tag = node.tag or ""
    if tag == "ri:user":
        return {"kind": "user"}
    entry = _REFERENCE_ATTRIBUTES.get(tag)
    if entry is None:
        return None
    kind, attribute, field = entry
    value = _collapse(node.attributes.get(attribute) or "")
    if not value:
        return None
    reference: Metadata = {"kind": kind, field: value}
    space = _collapse(node.attributes.get("ri:space-key") or "") if tag == "ri:page" else ""
    if space:
        reference["space"] = space
    return reference


def _run_text(child: LexborNode, tag: str, refs: list[JsonValue] | None = None) -> str | None:
    """What ``child`` adds to the run of text around it, or ``None`` if it starts a block.

    ``""`` and ``None`` mean different things and the difference is the parameter rule: a
    configuration element contributes *nothing to the text* but must not interrupt the sentence
    it sits in, while a table starts a new block.
    """
    if child.is_text_node:
        return child.text_content or ""
    if child.is_comment_node:
        return ""
    if tag in _CONFIGURATION_ONLY:
        # A stray parameter outside a macro. Skipped here as well as inside the macro handler,
        # so the rule holds however malformed the document is.
        return ""
    if tag == LINK:
        return _link_text(child, refs)
    if tag in _INLINE_TAGS:
        return _inline_text(child, refs)
    return None


def _flush(pending: list[str], refs: list[JsonValue] | None = None) -> Iterator[_Found]:
    """Emit the loose inline run collected so far, if it holds anything, and clear it."""
    text = _collapse("".join(pending))
    collected = list(refs) if refs else []
    pending.clear()
    if refs is not None:
        refs.clear()
    if text:
        metadata: Metadata = {"links": collected} if collected else {}
        yield _Found(kind=BlockKind.PROSE, text=text, metadata=metadata)


def _heading(node: LexborNode, tag: str, pending_fragment: str | None) -> _Found:
    return _Found(
        kind=BlockKind.HEADING,
        text=_inline_text(node),
        level=_HEADING_LEVELS[tag],
        fragment=node.id or pending_fragment,
    )


def _atomic(node: LexborNode, tag: str, config: ConfluenceConfig) -> _Found | None:
    """One element that is a block on its own, or ``None`` when it carries nothing to index."""
    metadata: Metadata = {}
    lang: str | None = None
    refs: list[JsonValue] = []
    if tag == "table":
        text = _table_text(node, refs)
        metadata = {"header_rows": _header_rows(node)}
    elif tag == TASK_LIST:
        text, metadata = _task_list(node)
    elif tag in {"ul", "ol", "dl"}:
        text = "\n".join(_list_lines(node, 0, refs))
    elif tag == "pre":
        # Not collapsed: indentation is part of what the code says.
        text = node.text(deep=True).strip("\n")
        lang = _code_language(node)
    else:
        text, metadata = _image(node)
    if not text.strip():
        return None
    if refs:
        metadata = {**metadata, "links": refs}
    kind = _ATOMIC_KINDS[tag]
    return _Found(kind=kind, text=text, lang=lang, metadata=metadata)


_ATOMIC_KINDS: Mapping[str, BlockKind] = {
    "table": BlockKind.TABLE,
    "ul": BlockKind.LIST,
    "ol": BlockKind.LIST,
    "dl": BlockKind.LIST,
    TASK_LIST: BlockKind.LIST,
    "pre": BlockKind.CODE,
    IMAGE: BlockKind.MEDIA,
}


# --- macros ------------------------------------------------------------------------------------


def _macro(node: LexborNode, name: str, config: ConfluenceConfig) -> Iterator[_Found]:
    """What one ``ac:structured-macro`` contributes, which is never its parameters."""
    if name in _CODE_MACROS:
        yield from _code_macro(node, name)
        return
    if name in GRAPHVIZ_MACROS:
        yield from _graphviz_macro(node, name)
        return
    if name in _PANEL_SEVERITIES:
        yield from _panel_macro(node, name, config)
        return
    yield from _unsupported_macro(node, name, config)


def _caption(node: LexborNode, name: str) -> Iterator[_Found]:
    """A verbatim-bodied macro's rendered title, as its own block above the body.

    Merged into the body it captions it would corrupt it: a code or diagram body has to come back
    character for character, and a caption spliced into the first line is no longer the source the
    page holds. Confluence draws it as a bar above the block, so a block above the block is what
    it is.
    """
    caption = _rendered_text(node, name)
    if caption.strip():
        yield _Found(kind=BlockKind.PROSE, text=caption, metadata={"macro": name, "caption": True})


def _code_macro(node: LexborNode, name: str) -> Iterator[_Found]:
    """A ``code`` or ``noformat`` macro: the body verbatim, with the language it declares."""
    body = _plain_text_body(node)
    if not body.strip():
        return
    yield from _caption(node, name)
    language = _parameter(node, "language")
    yield _Found(
        kind=BlockKind.CODE,
        text=body,
        lang=language or None,
        metadata={"macro": name},
    )


def _graphviz_macro(node: LexborNode, name: str) -> Iterator[_Found]:
    """A Graphviz macro: the DOT source exactly as written, inert, with its engine recorded.

    **Nothing lays this out.** The source is preserved as a code block because that is what it is
    — text in a language — and a citation into it must quote what the page says rather than a
    description of a picture nobody generated.

    Invalid DOT is kept with a warning attached rather than dropped. A diagram that does not
    compile is still what the author wrote, it is still searchable, and it is the version somebody
    debugging the page most needs to find.
    """
    body = _plain_text_body(node)
    if not body.strip():
        return
    yield from _caption(node, name)
    metadata: Metadata = {
        "macro": name,
        "engine": _parameter(node, "engine") or _DEFAULT_DOT_ENGINE,
        "rendered": False,
    }
    warning = dot_parse_warning(body)
    if warning is not None:
        metadata["parse_warning"] = warning
    yield _Found(kind=BlockKind.CODE, text=body, lang=DOT_LANGUAGE, metadata=metadata)


def dot_parse_warning(source: str) -> str | None:
    """Why this DOT would not parse, or ``None`` if nothing here can tell.

    **A structural check, not a parser, and deliberately shallow.** It reads the header and
    counts braces; it does not build a graph and it never invokes Graphviz. Anything deeper would
    be a second implementation of a language this project does not otherwise care about, and the
    cost of being wrong is asymmetric: running a real layout engine over untrusted input to find
    out is the thing this parser most exists not to do.

    **Quoted strings are skipped, and that is not shallowness being tidied up.** A record-shaped
    node is written ``a [shape=record, label="{left|right}"]`` — braces inside a string are
    ordinary Graphviz, not structure. Counting them attaches "the diagram body never ends" to a
    diagram that compiles perfectly, and a confident wrong warning on valid content is the exact
    failure this project exists to avoid. The content is kept either way; what would be wrong is
    the sentence next to it.

    Comments are *not* skipped, so a ``/* } */`` still miscounts. Left deliberately: it is rare
    enough that carrying a comment lexer for it would be the second implementation this avoids,
    and the cost is a warning beside content that was preserved regardless.
    """
    stripped = source.strip()
    if not stripped:
        return "empty diagram body"
    head = stripped.split("{", 1)[0].split()
    keywords = [word.lower() for word in head]
    if "graph" not in keywords and "digraph" not in keywords:
        return (
            "does not begin with 'graph' or 'digraph', so Graphviz would reject it before "
            "reading the body"
        )
    depth = 0
    quoted = False
    escaped = False
    for character in stripped:
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == '"':
            quoted = not quoted
        elif quoted:
            continue
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth < 0:
                return "a closing brace appears before the block it would close"
    if depth > 0:
        return f"{depth} unclosed '{{' — the diagram body never ends"
    return None


def _panel_macro(node: LexborNode, name: str, config: ConfluenceConfig) -> Iterator[_Found]:
    """A warning, note, tip, info or plain panel: its prose, with the severity it declares.

    The severity is the point. Flattened to prose, "Do not run this against production" reads as
    an ordinary sentence; kept as a panel with ``severity=warning`` it can be surfaced as the
    warning the author marked it.

    **Structured content inside the panel becomes its own block rather than being flattened into
    it.** A code block inside a warning keeps its language, and a warning inside a note keeps
    ``severity=warning`` instead of being absorbed into the note around it — which is the case
    that matters, because the inner one is usually the more urgent.
    """
    prose, structured = _body_blocks(node, config)
    text = _join_macro_prose((_rendered_text(node, name), *prose))
    if text.strip():
        yield _Found(
            kind=BlockKind.PANEL,
            text=text,
            metadata={"macro": name, "severity": _PANEL_SEVERITIES[name]},
        )
    yield from structured


def _unsupported_macro(node: LexborNode, name: str, config: ConfluenceConfig) -> Iterator[_Found]:
    """A macro with no reader here: a named placeholder, and whatever body it carries.

    Two obligations at once, and the second is the one easily lost. The placeholder stops the
    omission being silent — a reader can tell "we could not read this" from "the page was empty
    here". Keeping the body stops the placeholder becoming an excuse to discard content that
    needed no interpretation at all: an ``expand`` macro's rich text is ordinary prose that
    happens to be behind a disclosure triangle.

    Parameter **names** are recorded and their **values** are not. The names make the omission
    auditable — an operator can see that a ``jira`` macro was skipped and that it had a
    ``jqlQuery`` — without putting the query itself in the index, which is the defect this parser
    exists to end.

    **The rendered-parameter exception applies here too, and it was missed the first time.** An
    ``expand`` macro's ``title`` is the clickable label Confluence draws on the page: a reader
    sees it, quotes it and searches for it, so it is content by the same test a panel's title is.
    Emitting only the placeholder discarded it silently — the macro being unsupported is a
    statement about its *behaviour*, never a licence to drop the words it renders.
    """
    generated = name in _TOC_MACROS
    prose, structured = ([], []) if generated else _body_blocks(node, config)
    body_text = _join_macro_prose((_rendered_text(node, name), *prose))
    plain = "" if generated else _plain_text_body(node)
    if config.keep_unsupported_macros:
        metadata: Metadata = {"macro": name, "unsupported": True}
        names = _parameter_names(node)
        if names:
            metadata["parameters"] = list(names)
        yield _Found(
            kind=BlockKind.MEDIA,
            text=f"[unsupported macro: {name}]",
            metadata=metadata,
        )
    if body_text.strip():
        yield _Found(kind=BlockKind.PROSE, text=body_text, metadata={"macro": name})
    yield from structured
    if plain.strip():
        yield _Found(kind=BlockKind.CODE, text=plain, metadata={"macro": name})


def _macro_name(node: LexborNode) -> str:
    return (node.attributes.get("ac:name") or "").strip().lower() or "unnamed"


def _anchor_name(node: LexborNode) -> str | None:
    """The address an ``anchor`` macro publishes.

    Confluence writes it as an unnamed parameter — ``<ac:parameter ac:name="">Retries</...>`` —
    and older pages write ``ac:name="0"``. Both are read; anything else is not an anchor name.
    """
    for parameter in _descendants(node, frozenset({MACRO})):
        if parameter.tag != PARAMETER:
            continue
        key = (parameter.attributes.get("ac:name") or "").strip()
        if key in {"", "0"}:
            return _collapse(parameter.text(deep=True)) or None
    return None


def _parameter(node: LexborNode, name: str) -> str:
    """One parameter's value, read as configuration.

    Only the macro's **own** parameters count. A nested macro's are its own, and reading through
    to them would let an inner code macro's language decide an outer panel's title.
    """
    for parameter in _own_parameters(node):
        if (parameter.attributes.get("ac:name") or "").strip() == name:
            return _collapse(parameter.text(deep=True))
    return ""


def _parameter_names(node: LexborNode) -> tuple[str, ...]:
    """The names of this macro's own parameters, in document order, values excluded."""
    seen: list[str] = []
    for parameter in _own_parameters(node):
        key = (parameter.attributes.get("ac:name") or "").strip()
        if key and key not in seen:
            seen.append(key)
    return tuple(seen)


def _own_parameters(node: LexborNode) -> Iterator[LexborNode]:
    for child in _descendants(node, frozenset({MACRO})):
        if child.tag == PARAMETER:
            yield child


def _plain_text_body(node: LexborNode) -> str:
    """The macro's own plain-text body, verbatim.

    Verbatim because the body of a code or diagram macro is source: leading spaces are meaning,
    and collapsing whitespace would change what it says. It arrives here as text rather than as
    markup because ``recover_cdata`` escaped it on the way in, which is also what keeps a body
    containing ``<script>`` inert.
    """
    for child in _descendants(node, frozenset({MACRO})):
        if child.tag == PLAIN_TEXT_BODY:
            return child.text(deep=True).strip("\n")
    return ""


_MERGED_INTO_THE_BODY = frozenset({BlockKind.PROSE, BlockKind.HEADING})
"""What a macro body contributes to the text of the block that carries it.

A heading inside a macro body is merged rather than emitted: emitting it would open a section
whose scope is the inside of a panel, and every block after the panel would then be filed under
a heading the page does not have at that level."""


def _body_blocks(node: LexborNode, config: ConfluenceConfig) -> tuple[list[str], list[_Found]]:
    """A macro's rich-text body, split into what it says and what it contains.

    The prose belongs to the carrying block; anything structured — a nested panel, a code macro,
    a table — is its own block, because the properties that make it worth recognising are lost
    the moment it is flattened into the text around it.
    """
    prose: list[str] = []
    structured: list[_Found] = []
    for child in _descendants(node, frozenset({MACRO})):
        if child.tag != RICH_TEXT_BODY:
            continue
        for found in _walk(child, config):
            if found.kind in _MERGED_INTO_THE_BODY:
                if found.text.strip():
                    prose.append(found.text)
            else:
                structured.append(found)
    return prose, structured


def _join_macro_prose(parts: Iterable[str]) -> str:
    """The parts of a macro's rendered text, as separate paragraphs.

    **A blank line, because a single newline is not a paragraph boundary anywhere downstream.**
    :func:`~manicule.chunking.sentences.paragraphs` splits on blank lines and the chunker splits
    an oversized block the same way, so a body joined with ``\\n`` is one paragraph however many
    ``<p>`` elements it came from. Past the budget that paragraph is split into sentences and
    repacked with spaces, and every source paragraph boundary in it is gone — which is a
    structural fact about the page being destroyed by an assembly step, and it is invisible until
    something downstream needs a record to *begin* at a boundary.

    **The rule is about paragraphs and nothing else.** A macro body carries four other kinds of
    line break and none of them becomes a blank line here: a list keeps its one-item-per-line
    rendering, a task list keeps its one-task-per-line rendering, a table keeps its rows, and a
    plain-text body is source that comes back character for character. Each is assembled by its
    own function and reaches this one — if at all — as a single already-structured part, because
    :func:`_body_blocks` emits structured content as its own block rather than merging it.

    **A rendered parameter is a part like any other**, so a panel's title is separated from the
    body beneath it rather than glued to its first sentence. Confluence draws the title in the
    panel's own header; a reader sees two elements, and joining them into one paragraph makes the
    header the opening words of the body. This does *not* extend to the join inside
    :func:`_rendered_text`, which combines two values of the same rendered element and is not a
    paragraph boundary at all.

    Parts are stripped, because a heading merged into the body arrives with the surrounding
    element's whitespace and a blank line either side of it would be two blank lines.
    """
    return "\n\n".join(stripped for part in parts if (stripped := part.strip()))


def _rendered_text(node: LexborNode, macro: str) -> str:
    """The parameters Confluence draws on the page for *this* macro, which are content.

    Keyed by macro rather than by parameter name, so a ``title`` on a macro nobody has enumerated
    stays configuration. That is the safe direction to be wrong in.
    """
    values = [_parameter(node, name) for name in RENDERED_PARAMETERS.get(macro, ())]
    return "\n".join(value for value in values if value.strip())


# --- tasks, links, mentions, media ---------------------------------------------------------------


def _task_list(node: LexborNode) -> tuple[str, Metadata]:
    """A task list rendered one task per line, each carrying its state.

    The state is a property of the task and belongs on the line it describes. Walked as generic
    HTML it became a block of its own reading ``complete``, which is a word the page never says
    and a citation nobody can act on.
    """
    lines: list[str] = []
    complete = 0
    for task in _descendants(node, frozenset({TASK_LIST})):
        if task.tag != TASK:
            continue
        status = ""
        body = ""
        for part in _descendants(task, frozenset({TASK_LIST})):
            if part.tag == TASK_STATUS:
                status = _collapse(part.text(deep=True)).lower()
            elif part.tag == TASK_BODY:
                body = _inline_text(part)
        if not body.strip():
            continue
        done = status == "complete"
        complete += 1 if done else 0
        lines.append(f"- [{'x' if done else ' '}] {body}")
    return "\n".join(lines), {"tasks": len(lines), "complete": complete}


def _inline_text(node: LexborNode, refs: list[JsonValue] | None = None) -> str:
    """The text of an element's inline content, with Confluence's vocabulary understood.

    The single place the parameter rule is enforced for nested content: everything that flattens
    an element to a string goes through here rather than through ``text(deep=True)``, so a macro
    inside a table cell, a list item or a heading cannot leak its configuration into the text.
    """
    return _collapse("".join(_inline_parts(node, refs)))


def _inline_parts(node: LexborNode, refs: list[JsonValue] | None = None) -> Iterator[str]:
    for child in node.iter(include_text=True):
        _record(child, refs)
        if child.is_text_node:
            yield child.text_content or ""
            continue
        if child.is_comment_node:
            continue
        tag = child.tag or ""
        if tag in _CONFIGURATION_ONLY:
            continue
        if tag == LINK:
            yield _link_text(child, refs)
        elif tag == IMAGE:
            yield _image(child)[0]
        elif tag == MACRO:
            yield _inline_macro_text(child)
        elif tag.startswith("ri:"):
            yield _resource_text(child)
        else:
            yield from _inline_parts(child, refs)


def _inline_macro_text(node: LexborNode) -> str:
    """What a macro contributes when it appears inside a run of text.

    A block macro nested in a sentence is rendered as its placeholder rather than expanded: the
    alternative is either dropping it, or splicing a code body into the middle of a paragraph.
    Its parameters stay out of the text here as everywhere.
    """
    name = _macro_name(node)
    if name in _PANEL_SEVERITIES:
        return _collapse(_plain_or_rich(node))
    if name in _CODE_MACROS or name in GRAPHVIZ_MACROS:
        # The body, not a placeholder: a command in a table cell is the cell's whole point, and
        # dropping it here would be the silent discard this parser exists to stop.
        return _collapse(_plain_text_body(node))
    return f"[unsupported macro: {name}]"


def _plain_or_rich(node: LexborNode) -> str:
    for child in _descendants(node, frozenset({MACRO})):
        if child.tag in {RICH_TEXT_BODY, PLAIN_TEXT_BODY}:
            return _inline_text(child)
    return ""


def _link_text(node: LexborNode, refs: list[JsonValue] | None = None) -> str:
    """What an ``ac:link`` contributes: what a reader would see, never an identifier.

    The body the author wrote wins, because it is what the sentence reads as. Failing that the
    referenced thing names itself — a page by its title, an attachment by its filename, a person
    by a display reference.
    """
    for child in _descendants(node, frozenset({LINK})):
        _record(child, refs)
        if child.tag in {LINK_BODY, PLAIN_TEXT_LINK_BODY}:
            body = _inline_text(child)
            if body:
                return body
    for child in _descendants(node, frozenset({LINK})):
        if (child.tag or "").startswith("ri:"):
            return _resource_text(child)
    return ""


def _resource_text(node: LexborNode) -> str:
    """What an ``ri:*`` resource identifier contributes as text.

    **A user is a display reference and never an account id.** ``ri:account-id`` and ``ri:userkey``
    are opaque directory identifiers: they are not what the page shows, they are not what anybody
    would search for, and an index that carries them has taken a personal identifier out of a
    system that governs it and put it somewhere with different rules. Where the export supplies no
    display name there is nothing to show, so the mention says only that a person was mentioned.
    """
    tag = node.tag or ""
    attributes = node.attributes
    if tag == "ri:user":
        display = attributes.get("ri:username") or attributes.get("ri:display-name")
        return f"@{_collapse(display)}" if display and display.strip() else "@user"
    if tag == "ri:page":
        title = attributes.get("ri:content-title") or ""
        return _collapse(title)
    if tag == "ri:attachment":
        return _collapse(attributes.get("ri:filename") or "")
    if tag == "ri:space":
        return _collapse(attributes.get("ri:space-key") or "")
    if tag == "ri:url":
        return _collapse(attributes.get("ri:value") or "")
    return ""


def _image(node: LexborNode) -> tuple[str, Metadata]:
    """An image: what it says in words, and the attachment it names, which is never fetched.

    Keeping the reference in metadata rather than resolving it is what lets attachment ingestion
    be added later without this parser reaching across a network at parse time.
    """
    alt = node.attributes.get("ac:alt") or node.attributes.get("alt") or ""
    metadata: Metadata = {}
    filename = ""
    for child in _descendants(node):
        tag = child.tag or ""
        if tag == "ri:attachment":
            filename = _collapse(child.attributes.get("ri:filename") or "")
            metadata["attachment"] = filename
        elif tag == "ri:url":
            metadata["src"] = _collapse(child.attributes.get("ri:value") or "")
    text = _collapse(alt) or filename or _collapse(str(metadata.get("src") or ""))
    return text, metadata


# --- tables and lists, macro-aware -----------------------------------------------------------


def _table_text(node: LexborNode, refs: list[JsonValue] | None = None) -> str:
    """A table rendered one row per line, cells separated by pipes.

    Cells are flattened through :func:`_inline_text` rather than ``text(deep=True)``, which is the
    difference that keeps a macro's parameters out of a cell.
    """
    rows: list[str] = []
    for row in node.css("tr"):
        cells = [_inline_text(cell, refs) for cell in row.css("th, td")]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _header_rows(node: LexborNode) -> int:
    """How many leading rows are header rows, read from the markup rather than guessed."""
    head = node.css("thead")
    if head:
        return len(head[0].css("tr"))
    rows = node.css("tr")
    return 1 if rows and rows[0].css("th") else 0


def _list_lines(node: LexborNode, depth: int, refs: list[JsonValue] | None = None) -> Iterator[str]:
    """A list rendered with its nesting preserved as indentation."""
    marker = "1." if node.tag == "ol" else "-"
    for item in node.iter():
        if item.tag not in {"li", "dt", "dd"}:
            continue
        own = _collapse(
            "".join(
                part.text_content or ""
                if part.is_text_node
                else ("" if part.tag in {"ul", "ol"} else _inline_text(part, refs))
                for part in item.iter(include_text=True)
            )
        )
        if own:
            yield f"{_INDENT * depth}{marker} {own}"
        for nested in item.iter():
            if nested.tag in {"ul", "ol"}:
                yield from _list_lines(nested, depth + 1, refs)


def _code_language(node: LexborNode) -> str | None:
    """The language a bare ``<pre>`` declares, from the conventional ``language-`` class."""
    inner = node.css("code")
    code = inner[0] if inner else node
    classes = (code.attributes.get("class") or "").split()
    for name in classes:
        for prefix in ("language-", "lang-"):
            if name.startswith(prefix) and len(name) > len(prefix):
                return name[len(prefix) :]
    return None


def _collapse(text: str) -> str:
    """Whitespace as a reader sees it: runs become single spaces, ends are trimmed."""
    return " ".join(text.split())


# --- sections ----------------------------------------------------------------------------------


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
    """The text each section owns, the first entry being everything before the first heading."""
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
            f"one section and the page publishes no anchor to tell them apart"
        )
    return HeadingAnchor(path=section.path, fragment=section.fragment)


def _preamble_anchor(title: str, counts: Mapping[tuple[str, ...], int]) -> Anchor:
    """What content before the first heading is anchored to."""
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
