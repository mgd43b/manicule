"""Diagrams authored as code, read for what they state rather than for how they look.

``docs/parsing.md`` §8.4. A diagram is read with the grammars §8 already ships and answers a
different question with them. For a source file the tree decides *where* chunks begin, and the
text a reader wants is already in the file. For a diagram the tree carries the **meaning**,
because the sentence a reader sees is in no line of the source:

.. code-block:: text

    auth  [label="Auth Service"];
    store [label="Token Store"];
    auth -> store [label="validates against"];

A reader of the rendered diagram sees *Auth Service validates against Token Store*. The embedder
sees two identifiers, an arrow, and two string literals declared three lines away — the join
between an edge and its endpoints' labels is present nowhere in what it reads. That is the same
defect class :mod:`manicule.parsers.confluence` exists for: content a reader sees that the index
does not contain.

**The reading goes to the embedder and never to a citation.** :class:`DiagramMiddleware` rewrites
``Chunk.embed_text`` and leaves ``Chunk.text`` exactly as parsed, so a quotation still shows the
source the page holds. Emitting the reading as a block of its own was rejected for the reason
:mod:`manicule.connectors.macros` gives for unresolved includes: it would put manicule's own words
inside a quotation attributed to the source. The lexical leg indexes ``chunks.text``
(:mod:`manicule.storage.fts`), so the source also stays searchable verbatim — a node id or a
``rankdir`` is no less findable than before.

**Nothing is rendered, and no layout engine runs.** A diagram body is authored by anyone with
write access to the page, so it is untrusted input; ``docs/parsing.md`` §8.4.5 has the reasoning
and it is why everything here is reachable from the source text alone.

**Every failure keeps today's behavior.** A grammar that is not installed, a notation with no
reader, a source that yields no relationships: the chunk is returned unchanged, so ``embed_text``
stays breadcrumb + source and the corpus is what it was. tree-sitter is error-tolerant and this
module only reads node types it names, so a diagram with a syntax error contributes the part that
parsed and invents nothing for the part that did not. A reader that refused loudly here would fail
an ingest over a diagram, which is worse than the diagram embedding the way it does today.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, override

from manicule.core.protocols import Middleware
from manicule.parsers.config import DIAGRAM_LANGUAGES, DIAGRAM_MIDDLEWARE_NAME, DiagramConfig

if TYPE_CHECKING:
    from tree_sitter import Node

    from manicule.core.content import Chunk, Document

__all__ = [
    "DIAGRAM_LANGUAGES",
    "DIAGRAM_MIDDLEWARE_NAME",
    "DiagramConfig",
    "DiagramMiddleware",
    "notations",
    "reading",
]

_ARROW: Final = "→"
_EDGE: Final = "—"

_MAX_NESTING: Final = 64
"""How deep a scope may nest before a reader stops descending.

Far past any diagram a person has drawn, and far short of the interpreter's call stack. It exists
because the body is untrusted rather than because deep nesting is expected."""

_NAMES_PER_LINE: Final = 8
"""How many unconnected node names one line of a reading carries.

Small enough that the bound in :func:`_render` can drop part of an inventory rather than all of
it, and large enough that a handful of stragglers do not become a column of one-name lines."""


@dataclass(frozen=True, slots=True)
class _Edge:
    """One relationship the diagram draws, in the notation's own terms."""

    source: str
    target: str
    label: str = ""
    directed: bool = True


@dataclass(frozen=True, slots=True)
class _Group:
    """A cluster, subgraph or package: a statement about what belongs together."""

    label: str
    members: tuple[str, ...] = ()


@dataclass(slots=True)
class _Graph:
    """What a reader extracted, before it is rendered or bounded."""

    title: str = ""
    labels: dict[str, str] = field(default_factory=dict[str, str])
    edges: list[_Edge] = field(default_factory=list[_Edge])
    groups: list[_Group] = field(default_factory=list[_Group])
    order: list[str] = field(default_factory=list[str])
    """Every identifier seen, in source order, so an unconnected node is still reported."""

    def note(self, identifier: str) -> str:
        if identifier and identifier not in self.order:
            self.order.append(identifier)
        return identifier

    def name(self, identifier: str) -> str:
        """What the diagram draws for this identifier, which is its label where it has one."""
        return self.labels.get(identifier) or identifier


def notations() -> frozenset[str]:
    """Every notation this module actually reads, asked of the dispatch table.

    The declaration registration validates against is :data:`DIAGRAM_LANGUAGES`, and it lives in
    :mod:`manicule.parsers.config` so that registering the middleware costs no import of
    tree-sitter. That makes two statements of one answer, which is exactly the pair that drifts —
    so this is the question a test can put to the module itself rather than a second list to keep
    in step. The same reasoning shapes
    :data:`~manicule.parsers.confluence.INTERPRETED_MACROS`.
    """
    return frozenset(_READERS)


def reading(language: str, source: str, *, budget: int, max_relations: int) -> str | None:
    """The relationships ``source`` draws, as text for an embedder, or ``None`` for nothing.

    Args:
        language: A notation in :data:`DIAGRAM_LANGUAGES`. Anything else returns ``None``.
        source: The diagram exactly as the document holds it.
        budget: Longest reading to return, in characters.
        max_relations: Most relationships to report, before the character budget applies.

    Returns:
        The reading, or ``None`` when the notation has no reader, the grammar is unavailable,
        or the source yielded nothing worth saying. ``None`` means "leave the chunk alone".
    """
    if language not in DIAGRAM_LANGUAGES or not source.strip():
        return None
    from manicule.parsers.grammars import (  # noqa: PLC0415 - a parsing extra, not core
        GrammarUnavailableError,
        GrammarUnusableError,
        load_parser,
    )

    try:
        parser = load_parser(language)
    except (GrammarUnavailableError, GrammarUnusableError):
        # A grammar nobody seeded is not an error here. The chunk keeps the embedding input it
        # has today, which is the behavior every other failure in this module also produces.
        return None
    read = _READERS.get(language)
    if read is None:
        # Declared in `parsers.config` and dispatched here, which is two statements of one
        # answer. A test holds them in step; this keeps the promise above without depending on
        # a test in another file to do it.
        return None
    data = source.encode()
    root = parser.parse(data).root_node
    return _render(read(root, data), budget=budget, max_relations=max_relations)


# --- rendering -------------------------------------------------------------------------------


def _render(graph: _Graph, *, budget: int, max_relations: int) -> str | None:
    """The graph as lines, bounded twice and never truncated mid-fact.

    **The character budget is the length of the source being replaced**, which makes a budget
    regression impossible: the source already satisfied the chunk budget, and prose tokenizes
    more densely than punctuation, so a reading no longer than its source is necessarily no
    larger in tokens. It is a conservative bound in the direction that cannot fail.

    Whole lines are dropped from the end and the count is stated. Truncating mid-line would make
    the tail of the reading a fragment of a relationship, and dropping quietly would make a
    partly-read diagram indistinguishable from a small one — ``docs/connectors/confluence.md`` §5
    settles the same question the same way for macros.
    """
    lines = list(_lines(graph))
    if not lines:
        return None
    dropped = max(0, len(lines) - max_relations)
    kept = lines[: len(lines) - dropped]
    while kept:
        note = _dropped_note(dropped)
        rendered = "\n".join((*kept, note) if note else tuple(kept))
        if len(rendered) <= budget:
            return rendered
        dropped += 1
        kept = kept[:-1]
    return None


def _dropped_note(dropped: int) -> str:
    """What was left out, counted.

    Deliberately not "relationships": the lines dropped may be groupings or the unconnected-node
    line, and a note that named the wrong thing would be a small lie in the one place whose whole
    job is to stop a silent one.
    """
    return f"… and {dropped} more" if dropped else ""


def _lines(graph: _Graph) -> Iterator[str]:
    """Title, then relationships, then groupings, then whatever was never connected."""
    if graph.title:
        yield graph.title
    mentioned: set[str] = set()
    for edge in graph.edges:
        source, target = graph.name(edge.source), graph.name(edge.target)
        if not source or not target:
            continue
        mentioned.update({edge.source, edge.target})
        joint = _ARROW if edge.directed else _EDGE
        yield f"{source} {joint} {target}" + (f": {edge.label}" if edge.label else "")
    for group in graph.groups:
        members = [graph.name(member) for member in group.members if graph.name(member)]
        if group.label and members:
            mentioned.update(group.members)
            yield f'group "{group.label}": {", ".join(members)}'
    unconnected = [graph.name(item) for item in graph.order if item not in mentioned]
    named = [item for item in unconnected if item]
    # A diagram of boxes with no edges still states something, and its labels are the whole of
    # it. Reporting them is the difference between a reading and an empty result that would
    # leave the raw source in the vector.
    #
    # Several lines rather than one, because `_render` bounds by dropping whole lines: an
    # inventory of two hundred boxes emitted as a single line is one line too long to keep, so
    # the diagram this case exists for would be exactly the one that lost all of its labels.
    for start in range(0, len(named), _NAMES_PER_LINE):
        yield f"nodes: {', '.join(named[start : start + _NAMES_PER_LINE])}"


# --- Graphviz --------------------------------------------------------------------------------


def _read_dot(root: Node, data: bytes) -> _Graph:
    """Graphviz, whose tree models nodes, edges, attributes and clusters directly."""
    graph = _Graph()
    for block in _children(root, "block"):
        title, _ = _dot_scope(block, data, graph)
        if title and not graph.title:
            graph.title = title
    return graph


def _dot_scope(block: Node, data: bytes, graph: _Graph, depth: int = 0) -> tuple[str, list[str]]:
    """One ``{ … }`` scope: its own ``label``, and the identifiers declared inside it.

    Recursive because a subgraph is a scope like any other, and an edge inside a cluster is an
    edge on the diagram. The members are returned so that the enclosing scope can record which
    identifiers a cluster groups without walking the same tree twice.

    Bounded for the reason :func:`_walk` is: the body is untrusted, and nesting deeper than
    :data:`_MAX_NESTING` raised ``RecursionError`` out of :func:`reading` rather than leaving the
    chunk alone. Past the bound a scope contributes nothing, which costs a diagram nobody drew.
    """
    own_label = ""
    members: list[str] = []
    if depth > _MAX_NESTING:
        return own_label, members
    for statement in _statements(block):
        kind = statement.type
        if kind == "attribute":
            key, value = _dot_attribute(statement, data)
            if key == "label" and not own_label:
                own_label = value
        elif kind == "node_stmt":
            identifier = _dot_identifier(statement, data)
            if identifier:
                members.append(graph.note(identifier))
                label = _dot_attr_list(statement, data).get("label", "")
                if label:
                    graph.labels[identifier] = label
        elif kind == "edge_stmt":
            members.extend(_dot_edges(statement, data, graph))
        elif kind == "subgraph":
            for inner in _children(statement, "block"):
                label, inside = _dot_scope(inner, data, graph, depth + 1)
                members.extend(inside)
                if label:
                    graph.groups.append(_Group(label, tuple(inside)))
    return own_label, members


def _dot_edges(statement: Node, data: bytes, graph: _Graph) -> list[str]:
    """Every pair in an edge statement, which may be a chain — ``a -> b -> c`` is two edges."""
    endpoints = [_dot_text(child, data) for child in _children(statement, "node_id")]
    operators = [_dot_text(child, data) for child in _children(statement, "edgeop")]
    label = _dot_attr_list(statement, data).get("label", "")
    for identifier in endpoints:
        graph.note(identifier)
    for index, operator in enumerate(operators):
        if index + 1 >= len(endpoints):
            break
        graph.edges.append(
            _Edge(endpoints[index], endpoints[index + 1], label, directed="->" in operator)
        )
    return endpoints


def _dot_attr_list(statement: Node, data: bytes) -> Mapping[str, str]:
    """A statement's bracketed attributes. Everything but ``label`` is styling, and dropped."""
    found: dict[str, str] = {}
    for attributes in _children(statement, "attr_list"):
        for attribute in _children(attributes, "attribute"):
            key, value = _dot_attribute(attribute, data)
            if key and key not in found:
                found[key] = value
    return found


def _dot_attribute(attribute: Node, data: bytes) -> tuple[str, str]:
    """``key=value``, with the value unquoted. Either side may be a quoted string."""
    parts = [child for child in attribute.children if child.type == "id"]
    if len(parts) < 2:  # noqa: PLR2004 - an attribute is a pair; anything else is not one
        return "", ""
    return _dot_text(parts[0], data).lower(), _dot_text(parts[1], data)


def _dot_identifier(statement: Node, data: bytes) -> str:
    for child in _children(statement, "node_id"):
        return _dot_text(child, data)
    return ""


def _dot_text(node: Node, data: bytes) -> str:
    """A DOT identifier as the diagram draws it: unquoted, unescaped, one line."""
    return _unquote(_text(node, data))


# --- Mermaid ---------------------------------------------------------------------------------


def _read_mermaid(root: Node, data: bytes) -> _Graph:
    """Mermaid flowcharts and sequence diagrams, which are the two that dominate a corpus.

    Other diagram types parse to their own ``diagram_*`` subtree and contribute nothing rather
    than being guessed at, which is the same direction of failure as an unreadable source.
    """
    graph = _Graph()
    for node in _walk(root):
        kind = node.type
        if kind == "flow_stmt_vertice":
            _mermaid_flow(node, data, graph)
        elif kind == "flow_stmt_subgraph":
            _mermaid_subgraph(node, data, graph)
        elif kind == "sequence_stmt_participant":
            _mermaid_participant(node, data, graph)
        elif kind == "sequence_stmt_signal":
            _mermaid_signal(node, data, graph)
    return graph


def _mermaid_flow(statement: Node, data: bytes, graph: _Graph) -> None:
    """``a[Auth] -->|validates| b`` — vertices and links alternate, and links carry the text."""
    vertices: list[str] = []
    links: list[tuple[str, bool]] = []
    for child in statement.children:
        if child.type == "flow_node":
            vertices.append(graph.note(_mermaid_vertex(child, data, graph)))
        elif child.type.startswith("flow_link"):
            links.append((_mermaid_link_text(child, data), _mermaid_directed(child, data)))
    for index, (label, directed) in enumerate(links):
        if index + 1 >= len(vertices):
            break
        graph.edges.append(_Edge(vertices[index], vertices[index + 1], label, directed))


def _mermaid_directed(link: Node, data: bytes) -> bool:
    """Whether this link draws an arrowhead, which is a claim and not decoration.

    Mermaid's ``---`` is an open link with no arrowhead: reported as directed it would state a
    direction the diagram does not, which is what the DOT reader already avoids by telling ``--``
    from ``->``. ``<-->`` points both ways and is reported undirected as well — the reading has
    no notation for "both", and claiming less is the only safe direction to be wrong in.

    Read from the arrow token rather than from the whole link, because the link node also spans
    the label, and a label carrying an angle bracket would otherwise decide the direction of the
    relationship above it.
    """
    arrow = next((_text(node, data) for node in _descendants(link, "flow_link_arrow")), "")
    return ">" in arrow and "<" not in arrow


def _mermaid_vertex(node: Node, data: bytes, graph: _Graph) -> str:
    """A vertex's identifier, recording the shape's text as its label where it carries one.

    The shape is read as "whichever child is not the identifier" rather than by enumerating
    ``flow_vertex_square``, ``flow_vertex_cylinder`` and the dozen others: they differ only in
    the brackets around the same text, and an enumeration would silently lose the shapes nobody
    listed. The brackets are stripped as punctuation, which is what they are.
    """
    identifier = ""
    label = ""
    for vertex in _children(node, "flow_vertex"):
        for child in vertex.children:
            if child.type == "flow_vertex_id":
                identifier = _text(child, data)
            elif not label:
                label = _text(child, data).strip("[](){}></\\\"'| ")
    if identifier and label:
        graph.labels[identifier] = label
    return identifier


def _mermaid_link_text(link: Node, data: bytes) -> str:
    for text in _descendants(link, "flow_arrow_text"):
        return _text(text, data).strip("|- ")
    return ""


def _mermaid_subgraph(statement: Node, data: bytes, graph: _Graph) -> None:
    label = ""
    for child in statement.children:
        if child.type == "flow_vertex_text":
            label = _text(child, data).strip("[] ")
            break
    members = [
        _text(node, data)
        for inner in _children(statement, "flow_stmt_subgraph_inner")
        for node in _descendants(inner, "flow_vertex_id")
    ]
    if label and members:
        graph.groups.append(_Group(label, tuple(dict.fromkeys(members))))


def _mermaid_participant(statement: Node, data: bytes, graph: _Graph) -> None:
    """``participant A as Auth Service`` — the alias is the name the diagram draws."""
    actor = ""
    for child in statement.children:
        if child.type == "sequence_actor" and not actor:
            actor = _text(child, data).strip()
        elif child.type == "sequence_alias" and actor:
            graph.labels[actor] = _text(child, data).strip()
    graph.note(actor)


def _mermaid_signal(statement: Node, data: bytes, graph: _Graph) -> None:
    """``A->>B: validate token`` — a message between two actors, which is a relationship."""
    actors = [_text(child, data).strip() for child in _children(statement, "sequence_actor")]
    label = ""
    for child in _children(statement, "sequence_text"):
        label = _text(child, data).strip()
        break
    if len(actors) < 2:  # noqa: PLR2004 - a signal has a sender and a receiver
        return
    for actor in actors[:2]:
        graph.note(actor)
    graph.edges.append(_Edge(actors[0], actors[1], label))


# --- the notations this module reads -----------------------------------------------------------


_READERS: Final[Mapping[str, Callable[[Node, bytes], _Graph]]] = {
    "dot": _read_dot,
    "mermaid": _read_mermaid,
}
"""Every notation with a reader here, which is narrower than the notations with a grammar.

:data:`DIAGRAM_LANGUAGES` is the declaration registration reads, and it lives in
:mod:`manicule.parsers.config` so that registering this middleware costs no import of tree-sitter.
``tests/parsers/test_diagrams.py`` holds this table and that set to each other, because a reader
added here and not declared there would never be reached.

``plantuml`` has a grammar in the same pack and is deliberately absent. Its tree does not model
the language: ``auth --> store : validates against`` parses to a flat ``command`` of
``identifier`` and ``uniqkey`` tokens, with no node, edge or label among them, so a reader would
be re-implementing PlantUML's line grammar on top of a token soup rather than reading a tree. The
gap it would close is also the smallest of the three — PlantUML states a relationship inline, on
one line, so what the embedder loses is the ``[Auth Service] as auth`` aliasing rather than the
whole relation. Measured rather than assumed; ``docs/parsing.md`` §8.4.1 records it."""


# --- tree helpers ----------------------------------------------------------------------------


def _text(node: Node, data: bytes) -> str:
    """A node's source, decoded and flattened onto one line.

    A label may legitimately span lines — DOT writes ``\\n`` inside a quoted string and Mermaid
    accepts a wrapped one — and a reading is line-oriented, so a newline inside one fact would
    read as two.
    """
    raw = data[node.start_byte : node.end_byte].decode("utf-8", "replace")
    return " ".join(raw.split())


def _unquote(raw: str) -> str:
    """A DOT identifier with its quoting removed, or an HTML-like label with its tags removed."""
    value = raw.strip()
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):  # noqa: PLR2004
        return " ".join(value[1:-1].replace('\\"', '"').replace("\\\\", "\\").split())
    if len(value) >= 2 and value.startswith("<") and value.endswith(">"):  # noqa: PLR2004
        return _strip_tags(value[1:-1])
    return value


def _strip_tags(value: str) -> str:
    """The text of an HTML-like DOT label, with its markup removed and never interpreted.

    Scanned rather than parsed: the content is a label, the only question is which characters a
    reader sees, and routing it through an HTML engine would promote untrusted markup from inert
    text to a document — which is the trap ``manicule.parsers.web.recover_cdata`` exists for.
    """
    out: list[str] = []
    depth = 0
    for character in value:
        if character == "<":
            depth += 1
        elif character == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(character)
    return " ".join("".join(out).split())


def _children(node: Node, kind: str) -> Iterator[Node]:
    for child in node.children:
        if child.type == kind:
            yield child


def _descendants(node: Node, kind: str) -> Iterator[Node]:
    for child in _walk(node):
        if child.type == kind:
            yield child


def _walk(node: Node) -> Iterator[Node]:
    """Every descendant, depth-first, without recursing.

    An explicit stack rather than recursion because the input is a diagram body anyone with write
    access to a page can author, and one nested a couple of thousand levels deep raised
    ``RecursionError`` straight out of :func:`reading` — failing the document instead of leaving
    its embedding input alone, which is the opposite of what this module promises. ``docs/
    parsing.md`` §9.3 bounds the same threat for archives; here the bound is free, because the
    walk never needed a call stack.
    """
    stack = list(reversed(node.children))
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))


def _statements(block: Node) -> Iterator[Node]:
    """The statements directly inside a DOT block, which the grammar wraps in a ``stmt_list``."""
    for statements in _children(block, "stmt_list"):
        yield from statements.children


# --- the middleware --------------------------------------------------------------------------


class DiagramMiddleware(Middleware):
    """Replaces a diagram chunk's embedding input with the relationships it draws.

    ``Chunk.text`` is returned untouched, so a citation still quotes the source and the lexical
    leg still matches on it. Only the vector changes, which is why
    :attr:`mutates_embedded_text` is declared: it folds this middleware into the chunk
    fingerprint, and the re-embed that follows is the point rather than a side effect.
    """

    name = DIAGRAM_MIDDLEWARE_NAME
    mutates_embedded_text = True

    def __init__(self, config: DiagramConfig) -> None:
        self._config = config

    @override
    async def after_chunk(self, document: Document, chunks: list[Chunk]) -> list[Chunk]:
        del document
        return [self._rewrite(chunk) for chunk in chunks]

    def _rewrite(self, chunk: Chunk) -> Chunk:
        language = chunk.lang
        if language is None or language not in self._config.languages:
            return chunk
        if not chunk.embed_text.endswith(chunk.text):
            # The breadcrumb is recovered as the prefix `embed_text` carries above `text`
            # (`docs/parsing.md` §5) rather than rebuilt, because rebuilding it needs the
            # document's heading tree, which a chunk does not carry. A chunk some other
            # middleware has already rewritten no longer has that shape, and is left alone
            # instead of being given a breadcrumb this one guessed at.
            return chunk
        prefix = chunk.embed_text[: len(chunk.embed_text) - len(chunk.text)]
        derived = reading(
            language,
            chunk.text,
            budget=len(chunk.text),
            max_relations=self._config.max_relations,
        )
        if derived is None:
            return chunk
        return chunk.model_copy(update={"embed_text": prefix + derived})
