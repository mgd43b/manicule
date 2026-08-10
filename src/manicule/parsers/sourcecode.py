"""The code parser: blocks and line anchors taken from the parse tree a grammar produces.

Every boundary here is a node boundary. Nothing in this module looks at extracted text to
decide where a block starts, which is what makes ``LineAnchor.start``/``end`` a location the
library reported rather than a heuristic that happens to agree with one most of the time.

Four decisions are worth reading before the code:

**Segmentation asks the grammar, never the tags query.** A top-level child that spans more
than one line becomes its own block; consecutive single-line children group into one. Both
facts come from node row spans. The tags query is used *only* to name a block, so a language
gaining a tags query in a later pack release improves symbols without moving a single chunk
boundary — which is the right way round for something this cosmetic, and is asserted by the
tests rather than left as an intention.

**Splitting is done here, not deferred to the chunker.** ``docs/parsing.md`` §4.2 allows
either; blocks are emitted small enough to chunk because the chunker sees text and this
module sees a tree, and "never split mid-token, mid-string or mid-comment" is a statement
about the tree. A block is only ever cut between two sibling nodes that begin on different
lines, so a cut can never land inside a token — and where no such pair exists, the block is
emitted whole and over budget rather than cut somewhere unsafe.

**A missing grammar stops the document.** It is not a :class:`ParseError`, because declining
would hand the file to the plain-text parser and line-split it — the very outcome the
declared language set exists to prevent. See :mod:`manicule.parsers.grammars`.

**The parse tree is copied into plain Python objects before anything walks it.** This is the
one decision here made for the library's sake rather than the document's, so it is worth
being exact about. Retaining ``tree_sitter.Node`` objects across the yields of a generator
puts them inside reference cycles, and collecting such a cycle segfaults the interpreter on
files of a few hundred kilobytes — reproduced on this version, with and without a tags query,
at the point where the collector runs rather than at any particular call. Copying the parse
tree into :class:`_Item` in one pass that outlives no ``Node`` removes the failure mode
entirely instead of arranging the traversal to avoid it, which would be a fix nobody could
maintain because nothing would fail when it was undone. It also leaves the segmentation
algorithm working on plain data, which is testable without a grammar.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field
from tree_sitter import Node, QueryCursor, Tree

from manicule.core.anchors import Anchor, LineAnchor
from manicule.core.content import BlockKind, ParsedBlock, RawDocument
from manicule.core.errors import ParseError
from manicule.parsers import grammars
from manicule.parsers.base import ParserProfile, decode, lines_of, resolve_lines

__all__ = ["SourceCodeConfig", "SourceCodeParser"]


class SourceCodeConfig(BaseModel):
    """Configuration for :class:`SourceCodeParser`.

    Set under ``plugins.config."parser.sourcecode"``. Every field is load-bearing: the
    language set decides what routes here and what the corpus fingerprint records, and the
    two grammar overrides are what a container image and an air-gapped site respectively
    need in order to pre-seed at all.
    """

    languages: tuple[str, ...] = grammars.DECLARED_LANGUAGES
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


class SourceCodeParser:
    """Parses source code into blocks anchored to the lines they came from.

    Emits :class:`~manicule.core.content.BlockKind.CODE` blocks over the whole file. Code
    between definitions — imports, module-level statements, a trailing comment block — is
    content and is emitted too; dropping it would be data loss dressed up as tidiness.
    """

    profile = ParserProfile(name="sourcecode", max_unlocated_ratio=0.0, max_pagelevel_ratio=None)
    """Declared budgets. Zero unlocated blocks, because tree-sitter reports a row for every
    node: a code block that could not be located would mean the tree was not consulted."""

    def __init__(self, config: SourceCodeConfig) -> None:
        self._config = config
        self._languages = grammars.validate_languages(config.languages)
        grammars.configure_pack(
            self._languages,
            cache_dir=config.grammar_cache_dir,
            manifest_url=config.grammar_manifest_url,
        )
        self.media_types: frozenset[str] = frozenset(
            grammars.MEDIA_TYPES[language] for language in self._languages
        )

    @property
    def languages(self) -> tuple[str, ...]:
        """The declared set, validated and canonically ordered."""
        return self._languages

    def grammar_versions(self) -> dict[str, str]:
        """Grammar version per declared language, for ``ChunkFingerprint.grammars``."""
        return grammars.grammar_versions(self._languages)

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        """Yield code blocks in source order, each anchored to its own lines.

        Raises:
            ParseError: The bytes do not decode, or the media type is not one of the declared
                languages. Both mean "not my kind of document", so the next parser in the
                chain gets a turn.
            GrammarUnavailableError: The declared language's grammar is not cached. Distinct
                from :class:`ParseError` on purpose — see the module docstring.
        """
        language = grammars.language_for_media_type(raw.media_type)
        if language is None or language not in self._languages:
            msg = (
                f"{raw.uri}: media type {raw.media_type!r} is not one of the declared code "
                f"languages ({sorted(self.media_types)}). Declare the language, or route "
                f"this media type to the parser that owns it."
            )
            raise ParseError(msg)

        text = decode(raw)
        parser = grammars.load_parser(language)
        # Parse the *decoded* text re-encoded as UTF-8 rather than the original bytes: node
        # rows then index the same lines `lines_of` produces, which is what lets a row become
        # a LineAnchor without a second, differently-encoded view of the document.
        data = text.encode("utf-8")
        root = _mirror(parser.parse(data), language, data)
        lines = _Lines.of(text)
        separator = grammars.scope_separator(language)
        wrappers = grammars.DEFINITION_WRAPPERS.get(language, frozenset())

        claimed = 0
        for span in _segments(root, lines, self._config.max_block_chars):
            # Clamp to what earlier blocks have not already claimed. A trailing token that
            # closes a construct — a `;` on the line the construct ends on — belongs to the
            # run that follows it, and without this the two blocks would overlap on that
            # line and each would quote a piece of the other.
            first = max(span.first, claimed + 1)
            if first > span.last:
                continue
            block = _block(span, first, lines, language, separator, wrappers)
            if block is None:
                continue
            claimed = span.last
            yield block

    async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
        """The source lines ``anchor`` names, or ``None`` when it names none of these.

        Reads only the bytes it is handed. Nothing here consults state left by
        :meth:`parse`, so an anchor read back from storage months later resolves exactly as
        it did when it was written — which is the only version of this method worth having.
        """
        if not isinstance(anchor, LineAnchor):
            return None
        try:
            text = decode(raw)
        except ParseError:
            return None
        return resolve_lines(text, anchor)


# --- the parse tree, copied ---------------------------------------------------------------


@dataclass(slots=True)
class _Item:
    """One node of the parse tree, as plain data. See the module docstring for why.

    ``name`` is resolved during the copy, while the grammar's own accessors are still in
    hand, so nothing downstream needs a ``Node`` to answer "what is this definition called".
    """

    type: str
    named: bool
    first: int
    last: int
    start_byte: int
    end_byte: int
    name: str | None
    parent: _Item | None = None
    children: list[_Item] = field(default_factory=list["_Item"])

    @property
    def multiline(self) -> bool:
        return self.last > self.first

    @property
    def is_comment(self) -> bool:
        """Whether this is a comment, by the grammar's own name for it.

        Grammars name comment nodes ``comment``, ``line_comment``, ``block_comment`` and
        ``comment_block``; matching on the substring covers all of them. This is a test
        against a node type the library reported, not a pattern match over text.
        """
        return "comment" in self.type


def _first_row(node: Node) -> int:
    """The 1-based line a node starts on.

    tree-sitter rows are 0-based, and ``LineAnchor`` is 1-based and inclusive; this function
    and :func:`_last_row` are the only two places in the parser where that is converted.
    """
    return node.start_point.row + 1


def _last_row(node: Node) -> int:
    """The 1-based line a node's last character is on.

    A node whose end point sits at column 0 of a row ends *before* that row — it is the
    newline that closed the previous line, not content on the next one. Counting it would
    make every such block claim one line more than it holds, and the extra line is the first
    line of the next block, which reads perfectly and cites the wrong thing.
    """
    end = node.end_point
    if end.column == 0 and end.row > node.start_point.row:
        return end.row
    return end.row + 1


_Raw = tuple[str, bool, int, int, int, int, str | None, int]
"""One node reduced to primitives: type, named, first line, last line, start byte, end byte,
definition name, index of its parent (``-1`` for the root)."""


def _mirror(tree: Tree, language: str, data: bytes) -> _Item:
    """Copy a parse tree into :class:`_Item` objects, resolving every definition name.

    Two passes, and the split between them is the whole point. The first reads the tree and
    produces **nothing but tuples of primitives**; the second builds the linked
    :class:`_Item` tree after the parse tree has been released. Building the object graph
    while ``Node`` objects are still reachable is what segfaults the collector — see the
    module docstring — and the split makes that state unreachable rather than merely
    short-lived.
    """
    flat = _flatten(tree, language, data)
    del tree
    return _link(flat)


def _flatten(tree: Tree, language: str, data: bytes) -> list[_Raw]:
    """Read the parse tree into a flat list of primitives, parents before children.

    A node's children occupy consecutive entries in source order, which is what lets
    :func:`_link` rebuild the tree by appending as it goes.
    """
    tagged = _tagged_definitions(language, tree, data)
    rules = grammars.NODE_TYPE_DEFINITIONS.get(language, {})

    def raw(node: Node, parent: int) -> _Raw:
        return (
            node.type,
            node.is_named,
            _first_row(node),
            _last_row(node),
            node.start_byte,
            node.end_byte,
            tagged.get(node.id) or _named_by_rule(node, rules, data),
            parent,
        )

    root_node = tree.root_node
    flat: list[_Raw] = [raw(root_node, -1)]
    pending: list[tuple[Node, int]] = [(root_node, 0)]
    while pending:
        node, index = pending.pop()
        for child in node.children:
            pending.append((child, len(flat)))
            flat.append(raw(child, index))
    return flat


def _link(flat: list[_Raw]) -> _Item:
    """Rebuild the tree from :func:`_flatten`'s output."""
    items = [
        _Item(
            type=type_,
            named=named,
            first=first,
            last=last,
            start_byte=start_byte,
            end_byte=end_byte,
            name=name,
        )
        for type_, named, first, last, start_byte, end_byte, name, _parent in flat
    ]
    for item, entry in zip(items, flat, strict=True):
        parent = entry[7]
        if parent >= 0:
            item.parent = items[parent]
            items[parent].children.append(item)
    return items[0]


def _tagged_definitions(language: str, tree: Tree, data: bytes) -> dict[int, str]:
    """Definition names from the pack's own tags query, indexed by node.

    Run once over the whole tree rather than per block: a query is cheap to run and expensive
    to run n times, and running it once also fixes the answer, so two blocks inside one
    definition cannot disagree about its name.
    """
    query = grammars.tags_query(language)
    if query is None:
        return {}
    tagged: dict[int, str] = {}
    for _pattern, captures in QueryCursor(query).matches(tree.root_node):
        definition = next(
            (
                nodes[0]
                for capture, nodes in captures.items()
                if capture.startswith("definition.") and nodes
            ),
            None,
        )
        named = captures.get("name")
        # Matched against `definition.` rather than simply taking every `@name`: a tags query
        # also captures `@name` under its `reference.*` patterns, so a call inside a function
        # body would otherwise be indexed as a definition and rename the function containing
        # it.
        if definition is not None and named:
            tagged.setdefault(definition.id, _decode(data, named[0]))
    return tagged


def _named_by_rule(
    node: Node, rules: Mapping[str, grammars.DefinitionRule], data: bytes
) -> str | None:
    """This repository's own answer for a node the tags query did not name."""
    rule = rules.get(node.type)
    if rule is None:
        return None
    if rule.field is not None:
        found = node.child_by_field_name(rule.field)
        return _decode(data, found) if found is not None else None
    return next(
        (_decode(data, child) for child in node.children if child.type == rule.child_type),
        None,
    )


def _decode(data: bytes, node: Node) -> str:
    """A node's own text. Sliced from the parsed bytes, which are UTF-8 by construction, and
    at token boundaries, which are codepoint boundaries."""
    return data[node.start_byte : node.end_byte].decode("utf-8")


# --- lines ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Lines:
    """A file's lines, with the arithmetic to measure a range without building it.

    The offsets exist because ``_pack`` asks the length of a candidate range once per
    sibling. Joining the lines to measure them turns segmentation into quadratic string
    building, which on a file of a few hundred kilobytes is most of the parse.
    """

    lines: tuple[str, ...]
    offsets: tuple[int, ...]

    @classmethod
    def of(cls, text: str) -> _Lines:
        lines = tuple(lines_of(text))
        offsets: list[int] = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line) + 1)
        return cls(lines=lines, offsets=tuple(offsets))

    def text(self, first: int, last: int) -> str:
        """The source of lines ``first`` to ``last``, both 1-based and inclusive."""
        return "\n".join(self.lines[first - 1 : last])

    def length(self, first: int, last: int) -> int:
        """How long :meth:`text` would be, without building it."""
        return self.offsets[min(last, len(self.lines))] - self.offsets[first - 1] - 1


# --- segmentation --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Span:
    """A run of sibling items and the 1-based inclusive line range they occupy."""

    items: tuple[_Item, ...]
    first: int
    last: int


def _segments(root: _Item, lines: _Lines, budget: int) -> Iterator[_Span]:
    """Split a file into blocks, taking the highest node boundary that fits.

    Top level first: a child spanning more than one line is a definition in every language
    that has definitions, and gets its own block; single-line children — imports, pragmas,
    module-level statements — group into one, because fifty one-line blocks are fifty
    citations of one import each.
    """
    for run in _top_level_runs(root):
        yield from _fit(run, lines, budget)


def _top_level_runs(root: _Item) -> Iterator[tuple[_Item, ...]]:
    """Group the root's children into runs, in source order."""
    pending: list[_Item] = []
    for child in root.children:
        if child.is_comment:
            pending.append(child)
            continue
        if child.multiline:
            # A comment run immediately before a definition documents it, so it travels with
            # it. Left on its own it becomes a block whose whole text is "# Helpers", which
            # cites nothing useful and, in a file that uses the same separator twice, is two
            # blocks that cannot be told apart.
            head = len(pending)
            while head > 0 and pending[head - 1].is_comment:
                head -= 1
            if pending[:head]:
                yield tuple(pending[:head])
            yield (*pending[head:], child)
            pending = []
        else:
            pending.append(child)
    if pending:
        yield tuple(pending)


def _fit(items: tuple[_Item, ...], lines: _Lines, budget: int) -> Iterator[_Span]:
    """Emit ``items`` as one block, or descend until the parts fit.

    The ladder of ``docs/parsing.md`` §4.2 — top-level definitions, nested definitions,
    statement boundaries, blank-line runs, lines — collapses into one rule once it is applied
    to a tree: **break only between two siblings that begin on different lines**, and where
    that is impossible, descend a level and try again. Every rung of the ladder is a sibling
    boundary at some depth, blank lines fall inside the gaps between siblings, and the rule
    makes a cut inside a token unrepresentable rather than merely discouraged.
    """
    span = _span_of(items)
    if lines.length(span.first, span.last) <= budget:
        yield span
        return

    groups = _pack(items, lines, budget)
    if len(groups) > 1:
        for group in groups:
            yield from _fit(group, lines, budget)
        return

    if len(items) == 1 and items[0].children:
        inner = _pack(tuple(items[0].children), lines, budget)
        if len(inner) > 1:
            for group in inner:
                yield from _fit(group, lines, budget)
            return

    # Over budget with no safe boundary anywhere inside it: a minified line, or a single
    # enormous string or comment. Emitted whole. "Never split mid-token" outranks "then
    # lines" — a chunk cut through the middle of a string literal is not code, and the
    # citation it carries quotes half a token.
    yield span


def _pack(items: tuple[_Item, ...], lines: _Lines, budget: int) -> list[tuple[_Item, ...]]:
    """Greedily group siblings into runs that fit, breaking only at line boundaries."""
    groups: list[list[_Item]] = []
    current: list[_Item] = []
    for item in items:
        breakable = bool(current) and current[-1].last < item.first
        if breakable and lines.length(current[0].first, item.last) > budget:
            groups.append(current)
            current = [item]
            continue
        current.append(item)
    if current:
        groups.append(current)
    return [tuple(group) for group in groups]


def _span_of(items: tuple[_Item, ...]) -> _Span:
    return _Span(
        items=items,
        first=items[0].first,
        last=max(item.last for item in items),
    )


def _block(
    span: _Span,
    first: int,
    lines: _Lines,
    language: str,
    separator: str,
    wrappers: frozenset[str],
) -> ParsedBlock | None:
    """One block for a span, or ``None`` when the span holds no text worth citing."""
    text = lines.text(first, span.last)
    if not text.strip():
        return None
    path = _symbol_chain(span, wrappers)
    return ParsedBlock(
        kind=BlockKind.CODE,
        text=text,
        anchor=LineAnchor(
            start=first,
            end=span.last,
            symbol=separator.join(path) if path else None,
        ),
        heading_path=path,
        lang=language,
    )


# --- symbols -------------------------------------------------------------------------------


def _symbol_chain(span: _Span, wrappers: frozenset[str]) -> tuple[str, ...]:
    """The enclosing definition chain for a block, outermost first.

    Resolved from the **smallest node that contains the whole block**, so the answer is "the
    smallest definition enclosing this text" rather than "whatever the first node happened to
    sit in". A block spanning several top-level statements is contained only by the file
    itself and correctly gets no symbol; a block that is one method gets its class and its
    own name.
    """
    item: _Item | None = _containing(span, wrappers)
    path: list[str] = []
    while item is not None:
        if item.name:
            path.append(item.name)
        item = item.parent
    return tuple(reversed(path))


def _containing(span: _Span, wrappers: frozenset[str]) -> _Item:
    """The smallest item covering the block's code, ignoring any leading comment run.

    The comments are excluded because they were attached to the definition that follows them;
    including them would widen the span to the file and lose the symbol for exactly the
    blocks most worth naming. A wrapper — ``export``, a decorator run — is stepped through
    for the same reason: it contains the definition without being one.
    """
    code = tuple(item for item in span.items if not item.is_comment) or span.items
    start = min(item.start_byte for item in code)
    end = max(item.end_byte for item in code)

    item = code[0]
    while item.parent is not None:
        item = item.parent
    while True:
        inner = next(
            (
                child
                for child in item.children
                if child.start_byte <= start and child.end_byte >= end
            ),
            None,
        )
        if inner is None:
            break
        item = inner
    while item.type in wrappers:
        named = [child for child in item.children if child.named]
        if not named:
            break
        item = named[-1]
    return item
