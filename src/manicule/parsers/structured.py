"""JSON, YAML and TOML, anchored to the lines the source actually occupies.

The design goal is that **a block's text is an exact slice of source lines**, which makes its
:class:`~manicule.core.anchors.LineAnchor` correct by construction rather than by
reconstruction. The alternative — pretty-printing the parsed values and then hunting for them
in the source — produces anchors that drift the moment the file's formatting differs from the
printer's, and drift silently, because the reconstructed text still looks like the document.

Three formats, three ways of learning where a key begins, and the differences are the whole
of this module:

**YAML** — ``ruamel.yaml`` in round-trip mode carries line and column marks on every mapping
and sequence, so a top-level key's line is reported rather than searched for. It is used in
preference to PyYAML because PyYAML implements YAML **1.1** and ``ruamel.yaml`` implements
**1.2**. Two consequences, both load-bearing: 1.2 is very nearly a JSON superset, which is the
entire basis of the JSON strategy below, and 1.1 resolves unquoted ``no``, ``off`` and ``yes``
as booleans — so a file documenting the country code ``no`` would be indexed with ``False``
where the source says ``no``, which is a citation reproducing something the document does not
say.

**JSON** — the same mark-bearing reader supplies positions, including for compact one-line
objects with no space after the colon. *Nearly* a superset is not exactly one, and both
divergences are real: duplicate keys are legal JSON and raise in the YAML reader, and ``NaN``
loads as the string ``"NaN"``. So JSON is validated with the standard library first — that
decides whether the document is valid and what its values are — and the mark-bearing parse
supplies positions only. Where the two disagree, or the position parse raises, the document is
still indexed and carries ``Unlocated(reason="JSON source positions unavailable")``. That is
the one thing this parser's wide unlocated budget exists for, and it is not to be spent
elsewhere.

**TOML** — ``tomllib`` reports values and no positions at all. TOML publishes a structural
signal instead: a table header is ``[dotted.name]`` at the start of a line. Every header found
by that scan is **checked against the tables ``tomllib`` actually produced** before it becomes
a boundary, so a ``[not a table]`` line inside a multi-line string cannot invent a section.
That check is what keeps the rule "an anchor comes from a location the library reports" true
for the one format with no position API.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.error import YAMLError

from manicule.core.anchors import Anchor, LineAnchor, Unlocated
from manicule.core.content import BlockKind, ParsedBlock, RawDocument
from manicule.core.errors import ParseError
from manicule.parsers.base import ParserProfile, decode, is_probably_text, lines_of, resolve_lines
from manicule.parsers.config import STRUCTURED_MEDIA_TYPES, StructuredConfig

__all__ = [
    "NO_JSON_POSITIONS",
    "STRUCTURED_MEDIA_TYPES",
    "StructuredConfig",
    "StructuredParser",
    "read_yaml",
]

NO_JSON_POSITIONS = "JSON source positions unavailable"
"""Why a JSON document is indexed without line anchors.

Named because it is asserted in tests and reported by ``doctor``: a file that indexes fine and
loses its anchors is invisible unless the reason has a name.
"""


class _Format(StrEnum):
    JSON = "json"
    TOML = "toml"
    YAML = "yaml"


_FORMATS: dict[str, _Format] = {
    "application/json": _Format.JSON,
    "application/toml": _Format.TOML,
    "application/x-yaml": _Format.YAML,
    "application/yaml": _Format.YAML,
    "text/toml": _Format.TOML,
    "text/x-yaml": _Format.YAML,
    "text/yaml": _Format.YAML,
}

_DIVIDED = 2
"""How many parts a split has to produce before it counts as having divided anything.

One part is the span back again, so recursing on it would not terminate; the caller falls
back to splitting by lines instead.
"""

_MAX_SYMBOL_DEPTH = 16
"""How far a split may deepen a key path before it divides by lines instead.

A sixteen-element key path is longer than any citation would display, and a document nested
deeper than that is a serialised object graph rather than something a person wrote. Splitting
it further would buy a symbol nobody reads at the cost of unbounded recursion on hostile
input.
"""

_BARE_TOML_KEY = re.compile(r"^[A-Za-z0-9_-]+$")
_TOML_TABLE_HEADER = re.compile(r"^[ \t]*(\[\[?)([^\[\]]+)(\]\]?)[ \t]*(?:#.*)?$")
_TOML_KEY_LINE = re.compile(r"^[ \t]*([^=\[\]#]+?)[ \t]*=")


@dataclass(frozen=True, slots=True)
class _Mark:
    """A position the reader reported, and the value that begins there."""

    line: int
    """1-based, converted once from the reader's 0-based marks."""

    symbol: str
    value: object
    """The node whose own marks can divide this span further, or ``None`` for a leaf."""


@dataclass(frozen=True, slots=True)
class _Span:
    """A block-to-be: an inclusive 1-based line range and what addresses it."""

    start: int
    end: int
    symbol: str | None
    value: object = None
    located: bool = True
    """``False`` only where the JSON position parse declined, which is the one path in this
    module that produces an :class:`~manicule.core.anchors.Unlocated` anchor."""

    @property
    def height(self) -> int:
        return self.end - self.start + 1


class StructuredParser:
    """Parses JSON, YAML and TOML into blocks that are exact slices of the source."""

    media_types = STRUCTURED_MEDIA_TYPES
    profile = ParserProfile(name="structured", max_unlocated_ratio=0.10, max_pagelevel_ratio=None)
    """The widest unlocated budget of any parser, for exactly one reason: the JSON position
    path can legitimately decline (§3.4). Every other block here carries a real line span."""

    def __init__(self, config: StructuredConfig) -> None:
        self._config = config

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        """Yield one block per top-level entry, in source order.

        Raises:
            ParseError: The media type is not one of these three, the bytes are not text, or
                the document is not valid in its own format.
        """
        form = _format_of(raw)
        text = _text_of(raw)
        lines = lines_of(text)
        for span in self._spans(form, text, lines, raw.uri):
            yield ParsedBlock(
                # ``code`` rather than ``prose``: a serialisation format is not written to be
                # read as sentences, and the kind is what stops the chunker overlapping two
                # blocks — an overlap window copies the previous chunk's text into this one,
                # which a line span cannot honestly claim (``docs/parsing.md`` §1.5).
                kind=BlockKind.CODE,
                text="\n".join(lines[span.start - 1 : span.end]),
                anchor=_anchor(span),
                lang=form.value,
            )

    async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
        """Return the source lines ``anchor`` addresses, or ``None`` if it addresses none.

        Re-derives everything from ``raw``: an anchor that resolves only against what
        :meth:`parse` remembered has verified nothing about the document.
        """
        if not isinstance(anchor, LineAnchor):
            return None
        return resolve_lines(_text_of(raw), anchor)

    # --- per format ----------------------------------------------------------------------

    def _spans(self, form: _Format, text: str, lines: Sequence[str], uri: str) -> Iterator[_Span]:
        last = _last_content_line(lines)
        if last == 0:
            return
        if form is _Format.TOML:
            yield from self._toml_spans(text, lines, last, uri)
            return
        if form is _Format.JSON:
            yield from self._json_spans(text, lines, last, uri)
            return
        yield from self._mapping_spans(_load_yaml(text, uri), lines, last)

    def _json_spans(self, text: str, lines: Sequence[str], last: int, uri: str) -> Iterator[_Span]:
        """Validate with the standard library, then take positions from the YAML reader.

        The order is the point. ``json`` decides whether the document is valid and what its
        values are; the mark-bearing parse is consulted for line numbers and for nothing else,
        and where it disagrees the document keeps its content and loses its anchors.
        """
        try:
            values = json.loads(text)
        except json.JSONDecodeError as exc:
            msg = (
                f"{uri}: line {exc.lineno} is not valid JSON ({exc.msg}). Fix the document, or "
                f"route this media type to the plaintext parser to index it as text."
            )
            raise ParseError(msg) from exc
        try:
            positioned = _load_yaml(text, uri)
        except ParseError:
            # Duplicate keys are the common cause and are legal JSON. The document is still
            # indexed; only its locations are lost.
            yield _Span(start=1, end=last, symbol=None, located=False)
            return
        if len(positioned) != 1 or positioned[0] != values:
            # ``NaN`` is the case this catches: the YAML reader loads it as the string "NaN",
            # so its marks describe a document with different content from the one we indexed.
            yield _Span(start=1, end=last, symbol=None, located=False)
            return
        yield from self._mapping_spans(positioned, lines, last)

    def _mapping_spans(
        self, roots: Sequence[object], lines: Sequence[str], last: int
    ) -> Iterator[_Span]:
        """Blocks for every top-level entry of every document in the stream.

        A YAML file may hold several documents separated by ``---``, and their marks are
        absolute within the stream, so one ordered list of marks divides the whole file.
        """
        marks = sorted(
            (mark for root in roots for mark in _children_of(root)), key=lambda mark: mark.line
        )
        for span in _partition(marks, first=1, last=last, prefix=None, lines=lines):
            yield from self._divide(span, lines, depth=1)

    def _toml_spans(self, text: str, lines: Sequence[str], last: int, uri: str) -> Iterator[_Span]:
        try:
            values = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            msg = (
                f"{uri}: not valid TOML ({exc}). Fix the document, or route this media type to "
                f"the plaintext parser to index it as text."
            )
            raise ParseError(msg) from exc
        tables = _toml_tables(values)
        marks = _scan(lines, 1, last, _toml_header_reader(tables))
        for span in _partition(marks, first=1, last=last, prefix=None, lines=lines):
            yield from self._divide_toml(span, values, lines)

    def _divide_toml(
        self, span: _Span, values: dict[str, object], lines: Sequence[str]
    ) -> Iterator[_Span]:
        """Split an over-long table at its own keys, checked against the parsed table.

        TOML publishes no position for a key any more than for a table, so the same rule
        applies: scan for the literal ``key =`` at the start of a line, and accept it only
        where ``tomllib`` says that key exists in this table. A ``key = value`` line inside a
        multi-line string therefore cannot invent a boundary.
        """
        if span.height <= self._config.max_block_lines:
            yield from _by_lines(span, self._config.max_block_lines, lines)
            return
        table = _toml_table_at(values, span.symbol)
        keys = frozenset(table) if table is not None else frozenset[str]()
        marks = _scan(lines, span.start + 1, span.end, _toml_key_reader(keys))
        divided = _partition(
            marks, first=span.start, last=span.end, prefix=span.symbol, lines=lines
        )
        if len(divided) < _DIVIDED:
            yield from _by_lines(span, self._config.max_block_lines, lines)
            return
        for part in divided:
            yield from _by_lines(part, self._config.max_block_lines, lines)

    def _divide(self, span: _Span, lines: Sequence[str], *, depth: int) -> Iterator[_Span]:
        """Split a span that is too tall at the next level of the document's own structure."""
        if span.height <= self._config.max_block_lines or depth >= _MAX_SYMBOL_DEPTH:
            yield from _by_lines(span, self._config.max_block_lines, lines)
            return
        children = _partition(
            _children_of(span.value),
            first=span.start,
            last=span.end,
            prefix=span.symbol,
            lines=lines,
        )
        if len(children) < _DIVIDED:
            yield from _by_lines(span, self._config.max_block_lines, lines)
            return
        for child in children:
            yield from self._divide(child, lines, depth=depth + 1)


# --- readers -------------------------------------------------------------------------------


def read_yaml(text: str) -> list[object]:
    """Every document in a YAML stream, read under YAML 1.2.

    Public because the version matters more than the call: under YAML 1.1 the unquoted keys
    ``no``, ``off`` and ``yes`` are booleans, so a file listing country codes would be indexed
    with ``False`` where the source says ``no``. Reading 1.2 is what keeps ``symbol`` a
    reproduction of the document rather than an interpretation of it.

    Raises:
        YAMLError: The stream is not valid YAML.
    """
    reader = YAML(typ="rt")
    return list(cast("Iterator[object]", reader.load_all(text)))


def _load_yaml(text: str, uri: str) -> list[object]:
    """Every document in the stream, or a :class:`ParseError` naming what is wrong with it."""
    try:
        return read_yaml(text)
    except YAMLError as exc:
        msg = (
            f"{uri}: not valid YAML ({exc}). Fix the document, or route this media type to the "
            f"plaintext parser to index it as text."
        )
        raise ParseError(msg) from exc


def _children_of(value: object) -> list[_Mark]:
    """The marks a mapping or a sequence publishes for the entries directly inside it.

    Every line here is converted from the reader's 0-based marks exactly once, which is the
    only place in this module where that conversion happens.
    """
    if isinstance(value, CommentedMap):
        mapping = cast("CommentedMap", value)
        marks = cast("dict[object, Sequence[int]]", mapping.lc.data or {})
        return [
            _Mark(line=marks[key][0] + 1, symbol=str(key), value=mapping[key])
            for key in mapping
            if key in marks
        ]
    if isinstance(value, CommentedSeq):
        sequence = cast("CommentedSeq", value)
        entries = cast("dict[int, Sequence[int]]", sequence.lc.data or {})
        return [
            _Mark(line=entries[index][0] + 1, symbol=f"[{index}]", value=sequence[index])
            for index in range(len(sequence))
            if index in entries
        ]
    return []


def _toml_tables(values: dict[str, object]) -> frozenset[tuple[str, ...]]:
    """Every table path ``tomllib`` produced, as the tuples a header scan is checked against."""
    found: set[tuple[str, ...]] = set()

    def walk(node: object, path: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            table = cast("dict[str, object]", node)
            if path:
                found.add(path)
            for key, child in table.items():
                walk(child, (*path, key))
            return
        if isinstance(node, list):
            entries = cast("list[object]", node)
            if path and all(isinstance(entry, dict) for entry in entries):
                found.add(path)
            for entry in entries:
                walk(entry, path)

    walk(values, ())
    return frozenset(found)


def _toml_table_at(values: dict[str, object], symbol: str | None) -> dict[str, object] | None:
    """The parsed table a header's symbol names, or ``None`` when it names none."""
    if symbol is None:
        return None
    node: object = values
    for part in _symbol_parts(symbol):
        index = _array_index(part)
        name = part if index is None else part[: part.index("[")]
        if not isinstance(node, dict):
            return None
        node = cast("dict[str, object]", node).get(name)
        if index is not None:
            if not isinstance(node, list) or index >= len(cast("list[object]", node)):
                return None
            node = cast("list[object]", node)[index]
    return cast("dict[str, object]", node) if isinstance(node, dict) else None


def _toml_header_reader(tables: frozenset[tuple[str, ...]]) -> Callable[[str], str | None]:
    """Accept ``[name]`` and ``[[name]]`` lines that name a table the document really has."""
    seen: dict[tuple[str, ...], int] = {}

    def read(line: str) -> str | None:
        match = _TOML_TABLE_HEADER.match(line)
        if match is None or len(match.group(1)) != len(match.group(3)):
            return None
        parts = _toml_header_parts(match.group(2))
        if parts is None or parts not in tables:
            return None
        symbol = ".".join(_quoted(part) for part in parts)
        if len(match.group(1)) == 1:
            return symbol
        # Arrays of tables repeat one header, so the occurrence index is the only thing that
        # tells the second entry from the first.
        occurrence = seen.get(parts, 0)
        seen[parts] = occurrence + 1
        return f"{symbol}[{occurrence}]"

    return read


def _toml_key_reader(keys: frozenset[str]) -> Callable[[str], str | None]:
    """Accept ``key =`` lines naming a key the parsed table really has."""

    def read(line: str) -> str | None:
        match = _TOML_KEY_LINE.match(line)
        if match is None:
            return None
        parts = _toml_header_parts(match.group(1))
        if parts is None or len(parts) != 1 or parts[0] not in keys:
            return None
        return _quoted(parts[0])

    return read


def _toml_header_parts(inner: str) -> tuple[str, ...] | None:
    """Split a dotted TOML name into its parts, honouring quoting.

    ``None`` when the text is not a well-formed dotted name, which is the answer that keeps a
    line inside a multi-line string from becoming a section boundary.
    """
    parts: list[str] = []
    current: list[str] = []
    quote: str | None = None
    for character in inner:
        if quote is not None:
            if character == quote:
                quote = None
                continue
            current.append(character)
            continue
        if character in "\"'":
            quote = character
            continue
        if character == ".":
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(character)
    if quote is not None:
        return None
    parts.append("".join(current).strip())
    if any(not part for part in parts):
        return None
    return tuple(parts)


def _quoted(part: str) -> str:
    """A key path element as TOML would write it, quoted only where it has to be."""
    return part if _BARE_TOML_KEY.match(part) else f'"{part}"'


def _symbol_parts(symbol: str) -> list[str]:
    """Split a dotted symbol back into its elements, honouring the quoting above."""
    parts = _toml_header_parts(symbol)
    return list(parts) if parts is not None else [symbol]


def _array_index(part: str) -> int | None:
    if not part.endswith("]") or "[" not in part:
        return None
    digits = part[part.index("[") + 1 : -1]
    return int(digits) if digits.isdigit() else None


# --- spans ---------------------------------------------------------------------------------


def _scan(
    lines: Sequence[str], first: int, last: int, read: Callable[[str], str | None]
) -> list[_Mark]:
    """A forward scan for the boundaries a format publishes as literal text."""
    marks: list[_Mark] = []
    for number in range(first, last + 1):
        symbol = read(lines[number - 1])
        if symbol is not None:
            marks.append(_Mark(line=number, symbol=symbol, value=None))
    return marks


def _partition(
    marks: Sequence[_Mark],
    *,
    first: int,
    last: int,
    prefix: str | None,
    lines: Sequence[str],
) -> list[_Span]:
    """Divide ``first``-``last`` at the reported marks, covering every line in between.

    The spans are contiguous by construction, which is what keeps the chunker honest: a chunk
    merging two blocks resolves to every line between the first block's start and the last
    block's end, so a line belonging to no block would appear in the citation and in no
    chunk's text. Only trailing *blank* lines are trimmed, because whitespace is the one thing
    the round-trip normaliser removes from both sides of that comparison.
    """
    inside = [mark for mark in marks if first <= mark.line <= last]
    if not inside:
        return _trimmed([_Span(start=first, end=last, symbol=None)], lines)

    grouped: list[_Mark] = []
    for mark in inside:
        if grouped and grouped[-1].line == mark.line:
            # Several keys on one line — a compact JSON object. A line range cannot address a
            # fraction of a line, so one block covers them and claims no single key.
            grouped[-1] = _Mark(line=mark.line, symbol="", value=None)
            continue
        grouped.append(mark)

    spans: list[_Span] = []
    for index, mark in enumerate(grouped):
        start = first if index == 0 else mark.line
        end = grouped[index + 1].line - 1 if index + 1 < len(grouped) else last
        symbol = _joined(prefix, mark.symbol) if mark.symbol else None
        spans.append(_Span(start=start, end=min(end, last), symbol=symbol, value=mark.value))
    return _trimmed(spans, lines)


def _joined(prefix: str | None, symbol: str) -> str:
    if prefix is None:
        return symbol
    return f"{prefix}{symbol}" if symbol.startswith("[") else f"{prefix}.{symbol}"


def _trimmed(spans: Sequence[_Span], lines: Sequence[str]) -> list[_Span]:
    """Drop trailing blank lines from each span, and spans that are blank all through."""
    kept: list[_Span] = []
    for span in spans:
        end = span.end
        while end >= span.start and not lines[end - 1].strip():
            end -= 1
        if end >= span.start:
            kept.append(_Span(start=span.start, end=end, symbol=span.symbol, value=span.value))
    return kept


def _by_lines(span: _Span, max_lines: int, lines: Sequence[str]) -> Iterator[_Span]:
    """The last resort: divide at line boundaries, each part keeping its own exact span."""
    if span.height <= max_lines:
        yield span
        return
    for start in range(span.start, span.end + 1, max_lines):
        end = min(start + max_lines - 1, span.end)
        if any(lines[number - 1].strip() for number in range(start, end + 1)):
            yield _Span(start=start, end=end, symbol=span.symbol, value=None)


def _last_content_line(lines: Sequence[str]) -> int:
    """The 1-based number of the last line with anything on it, or ``0`` for none."""
    for number in range(len(lines), 0, -1):
        if lines[number - 1].strip():
            return number
    return 0


def _anchor(span: _Span) -> Anchor:
    if not span.located:
        return Unlocated(reason=NO_JSON_POSITIONS)
    return LineAnchor(start=span.start, end=span.end, symbol=span.symbol)


def _format_of(raw: RawDocument) -> _Format:
    """Which of the three this is, from the media type the pipeline resolved.

    Raises:
        ParseError: The media type is not one of them.
    """
    declared = raw.media_type.split(";", 1)[0].strip().lower()
    form = _FORMATS.get(declared)
    if form is None:
        msg = (
            f"{raw.uri}: declining — {raw.media_type!r} is not structured data this parser "
            f"reads. It reads {', '.join(sorted(STRUCTURED_MEDIA_TYPES))}."
        )
        raise ParseError(msg)
    return form


def _text_of(raw: RawDocument) -> str:
    """The document as text, declining anything that is not.

    Raises:
        ParseError: The bytes are binary, or do not decode as the declared encoding.
    """
    if not is_probably_text(raw.as_bytes()):
        msg = (
            f"{raw.uri}: declining — these bytes are binary, and JSON, YAML and TOML are all "
            f"text formats. Check the media type this document was routed with."
        )
        raise ParseError(msg)
    return decode(raw)
