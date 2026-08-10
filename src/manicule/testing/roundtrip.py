"""The round-trip obligation, as six runnable assertions.

``docs/contracts.md`` §1 calls the round trip "a test obligation on every parser, not a
convention". This module is that obligation. Every failure it catches is silent by nature:
nothing raises, the citation looks right, and the location is wrong.

The six, and what each one alone would let through:

1. **Containment** — the chunk says where it came from, and it is there. Satisfied on its own
   by an anchor pointing at the whole document.
2. **Tightness** — the anchor is not much bigger than what it addresses. This is what stops
   "the whole document" from passing.
3. **Discrimination** — resolving one anchor does not return another chunk's text. This is
   what fires when every page index is 1, when one is off by one, or when a heading fragment
   lands on the wrong section. All three pass containment, and on a short document all three
   can pass tightness.
4. **Determinism** — the same bytes twice give the same blocks. Set iteration order and
   float arithmetic both break this, and a parser that breaks it churns the corpus on every
   re-ingest.
5. **Location budget** — a parser may not satisfy rules 1-3 by returning ``Unlocated`` for
   everything, or by making every anchor page-level. Corpus-wide, and declared per parser.
6. **Idempotence** — re-parsing and re-chunking an unchanged document produces the same
   chunk sequence, so an unchanged document costs no re-embedding.

Assertions 1-4 and 6 run per fixture. Assertion 5 is per corpus and runs once at the end of a
parser's suite, over the reports the others returned.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import NoReturn

from manicule.core.anchors import Anchor, HeadingAnchor, PageAnchor, Unlocated
from manicule.core.content import BlockKind, Chunk, Document, ParsedBlock, RawDocument
from manicule.core.protocols import Chunker, Parser, read_blocks
from manicule.testing.normalise import contains_claimed_text, normalise

TIGHTNESS: dict[str, float] = {
    "line": 1.0,
    "cell": 1.0,
    "page": 1.05,
    "heading": 1.2,
}
"""How much larger than the text it addresses an anchor's resolved span may be.

``line`` and ``cell`` are 1.0 because the span *is* the text. ``page`` gets 5% for glyphs
clipped by the edge of a character box. ``heading`` gets 20% for the whitespace between a
section's blocks. A ``PageAnchor`` with no rectangles is exempt and capped by assertion 5
instead — resolving one returns a whole page, which no bound could usefully constrain.

A resolved section also carries **its own heading line**, and that is a *fixed* overhead
rather than a proportional one: a purely multiplicative bound is met comfortably by a long
section and broken by a short one with a long title, which makes the bound a measure of
section length rather than of anchor quality. So the heading's own text is added to the
denominator — its length is known from :attr:`HeadingAnchor.path`, so nothing has to be
assumed about it — and the multiplier then covers only the whitespace it was meant to cover.
"""

_OVERLAPPING_KINDS = frozenset({BlockKind.PROSE, BlockKind.LIST})
"""Kinds the chunker may overlap (``docs/parsing.md`` §1.5), and therefore the only ones
whose adjacent chunks may legitimately share text."""


@dataclass(frozen=True, slots=True)
class ParserProfile:
    """What a parser declares about the locations it produces.

    Both ratios are ceilings the harness enforces over the parser's whole fixture corpus.
    Without them, "a citation carries a correct location, or none" is satisfied by never
    carrying one. Lowering a ceiling is an ordinary improvement; raising one needs a note
    saying which fixture forced it.
    """

    name: str
    max_unlocated_ratio: float = 0.0
    max_pagelevel_ratio: float | None = None
    """``None`` declares that this parser emits no :class:`PageAnchor` at all, which the
    harness checks: a parser that starts emitting them must declare a budget for them."""


@dataclass(frozen=True, slots=True)
class RoundTripReport:
    """What one fixture produced. Aggregated by :func:`assert_location_budget`."""

    fixture: str
    blocks: int = 0
    unlocated: int = 0
    page_level: int = 0
    chunks: int = 0


def _fail(message: str) -> NoReturn:
    raise AssertionError(message)


def _require(condition: object, message: str) -> None:
    if not condition:
        _fail(message)


def _anchor_key(anchor: Anchor) -> str:
    return anchor.model_dump_json()


def _is_page_level(anchor: Anchor) -> bool:
    return isinstance(anchor, PageAnchor) and not anchor.rects


def _nests(outer: Anchor, inner: Anchor) -> bool:
    """Whether ``inner`` is genuinely inside ``outer`` rather than confused with it.

    Only heading hierarchies qualify, and only by path prefix. A parser resolving a section
    to text that includes its subsections is describing a real containment, not a mistake, so
    excluding it is correct — but the exclusion is derived from the paths rather than
    declared, because a declaration would also excuse the mistakes.
    """
    if not isinstance(outer, HeadingAnchor) or not isinstance(inner, HeadingAnchor):
        return False
    return len(inner.path) > len(outer.path) and inner.path[: len(outer.path)] == outer.path


def _shared_overlap(previous: str, current: str) -> int:
    """How many leading characters of ``current`` repeat the tail of ``previous``.

    Adjacent prose chunks share an overlap window by design (``docs/parsing.md`` §1.5), so
    the duplicated sentences are excluded from discrimination. Everything after them still
    has to discriminate, which is what keeps the exclusion narrow.
    """
    limit = min(len(previous), len(current))
    for size in range(limit, 0, -1):
        if previous.endswith(current[:size]):
            return size
    return 0


def _covered_length(resolved: str, texts: Sequence[str]) -> int:
    """Length of the union of the spans ``texts`` occupy within ``resolved``.

    The union, never the sum. Summing double-counts the overlap between adjacent chunks,
    which inflates the denominator of the tightness bound and makes it *easier* to pass the
    more overlap there is — weakening the assertion exactly where chunking is densest.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for text in texts:
        if not text:
            continue
        found = resolved.find(text, cursor)
        if found < 0:
            found = resolved.find(text)
        if found < 0:
            continue
        spans.append((found, found + len(text)))
        # Advance to the match's *start*, not its end: the next chunk may begin inside this
        # one, and skipping past would lose the overlap the union exists to merge.
        cursor = found
    if not spans:
        return 0
    spans.sort()
    total = 0
    start, end = spans[0]
    for next_start, next_end in spans[1:]:
        if next_start > end:
            total += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    total += end - start
    return total


@dataclass(frozen=True, slots=True)
class _Located:
    """One text with the anchor that claims it, whether it came from a block or a chunk."""

    where: str
    text: str
    anchor: Anchor
    kind: BlockKind


async def _resolve_all(
    parser: Parser, raw: RawDocument, items: Sequence[_Located]
) -> dict[str, str | None]:
    resolved: dict[str, str | None] = {}
    for item in items:
        key = _anchor_key(item.anchor)
        if key not in resolved:
            resolved[key] = await parser.resolve(item.anchor, raw)
    return resolved


def _assert_containment(items: Sequence[_Located], resolved: dict[str, str | None]) -> None:
    for item in items:
        text = resolved[_anchor_key(item.anchor)]
        if isinstance(item.anchor, Unlocated):
            _require(
                text is None,
                f"{item.where}: the anchor is Unlocated but resolve() returned text, so the "
                f"anchor claims less than the parser knows",
            )
            _require(item.anchor.reason, f"{item.where}: Unlocated without a reason")
            continue
        _require(
            text is not None,
            f"{item.where}: anchor {item.anchor!r} does not resolve. An anchor nobody can "
            f"resolve is a citation nobody can check; emit Unlocated instead",
        )
        _require(
            # The same call citation verification makes at answer time. See
            # `contains_claimed_text` for why there is one of these and not two.
            contains_claimed_text(text, item.text),
            f"{item.where}: the anchor resolves to text this does not claim.\n"
            f"  claimed:  {item.text[:160]!r}\n"
            f"  resolved: {(text or '')[:160]!r}",
        )


def _assert_tightness(items: Sequence[_Located], resolved: dict[str, str | None]) -> None:
    groups: dict[str, list[_Located]] = {}
    for item in items:
        groups.setdefault(_anchor_key(item.anchor), []).append(item)

    for key, group in groups.items():
        anchor = group[0].anchor
        if isinstance(anchor, Unlocated) or _is_page_level(anchor):
            continue
        bound = TIGHTNESS.get(anchor.kind)
        if bound is None:  # pragma: no cover - every located kind has a bound
            continue
        text = normalise(resolved[key] or "")
        covered = _covered_length(text, [normalise(item.text) for item in group])
        _require(
            covered > 0,
            f"{group[0].where}: nothing in the resolved text matched, so tightness cannot "
            f"be measured. Containment should have caught this first",
        )
        allowed = covered + _fixed_overhead(anchor)
        _require(
            len(text) <= bound * allowed,
            f"{group[0].where}: anchor {anchor!r} resolves to {len(text)} characters to "
            f"address {covered} of them ({len(text) / allowed:.2f}x, limit {bound}x). An "
            f"anchor much larger than what it addresses cites its neighbours too",
        )


def _fixed_overhead(anchor: Anchor) -> int:
    """Text a resolved location legitimately carries beyond the blocks inside it.

    Only a heading has any: the section's own heading line. Its length comes from the anchor
    itself, so this adds no assumption about the document — and it keeps the multiplier from
    quietly doubling as a section-length check.
    """
    if isinstance(anchor, HeadingAnchor):
        return len(normalise(anchor.path[-1]))
    return 0


def _assert_discrimination(
    items: Sequence[_Located], resolved: dict[str, str | None], *, overlapping: bool
) -> None:
    """Args:
    items: Every located text in document order.
    resolved: What each anchor resolves to.
    overlapping: Whether adjacent items may legitimately share text. True for chunks, which
        carry an overlap window by design; **false for blocks**, which never do. Applying the
        exclusion to blocks would let a parser through whenever one block's text happened to
        end with the next one's — which is exactly what a section anchor confused with its
        neighbour looks like.
    """
    normalised = [normalise(item.text) for item in items]
    for outer_index, outer in enumerate(items):
        if isinstance(outer.anchor, Unlocated) or _is_page_level(outer.anchor):
            continue
        haystack = normalise(resolved[_anchor_key(outer.anchor)] or "")
        outer_key = _anchor_key(outer.anchor)
        for inner_index, inner in enumerate(items):
            if inner_index == outer_index or _anchor_key(inner.anchor) == outer_key:
                continue
            if isinstance(inner.anchor, Unlocated) or _nests(outer.anchor, inner.anchor):
                continue
            needle = normalised[inner_index]
            if (
                overlapping
                and abs(inner_index - outer_index) == 1
                and {outer.kind, inner.kind} <= _OVERLAPPING_KINDS
            ):
                first, second = sorted((outer_index, inner_index))
                shared = _shared_overlap(normalised[first], normalised[second])
                if inner_index > outer_index:
                    needle = needle[shared:]
                if not needle:
                    continue
            _require(
                needle not in haystack,
                f"{outer.where}: resolving {outer.anchor!r} returns the text of "
                f"{inner.where} ({inner.anchor!r}). Two locations that cannot be told apart "
                f"cite each other; this is what an off-by-one page or line index looks like",
            )


def _located_from_blocks(blocks: Sequence[ParsedBlock]) -> list[_Located]:
    return [
        _Located(where=f"block {index}", text=block.text, anchor=block.anchor, kind=block.kind)
        for index, block in enumerate(blocks)
    ]


def _located_from_chunks(chunks: Sequence[Chunk]) -> list[_Located]:
    return [
        _Located(where=f"chunk {index}", text=chunk.text, anchor=chunk.anchor, kind=chunk.kind)
        for index, chunk in enumerate(chunks)
    ]


def _comparable(blocks: Sequence[ParsedBlock]) -> list[str]:
    return [block.model_dump_json() for block in blocks]


async def assert_round_trip(
    parser: Parser,
    raw: RawDocument,
    *,
    fixture: str = "",
    chunker: Chunker | None = None,
    document: Document | None = None,
) -> RoundTripReport:
    """Run assertions 1-4 and 6 against one fixture, and report what it produced.

    Args:
        parser: The parser under test.
        raw: One fixture document.
        fixture: A name for it, used in failure messages.
        chunker: When given, the chunks are checked as well as the blocks. A chunk's anchor
            is a merge of its blocks' anchors, and a merge is exactly where a location goes
            wrong without anything raising.
        document: Required alongside ``chunker``; the document the chunks belong to.

    Returns:
        Counts for :func:`assert_location_budget`, which is the corpus-level assertion.
    """
    name = fixture or raw.uri
    blocks = await read_blocks(parser, raw)

    for index, block in enumerate(blocks):
        _require(
            block.text != "",
            f"{name}: block {index} has empty text; a block with no text is not a block",
        )

    items = _located_from_blocks(blocks)
    resolved = await _resolve_all(parser, raw, items)
    _assert_containment(items, resolved)
    _assert_tightness(items, resolved)
    _assert_discrimination(items, resolved, overlapping=False)

    again = await read_blocks(parser, raw)
    _require(
        _comparable(blocks) == _comparable(again),
        f"{name}: parsing the same bytes twice produced different blocks. Chunk ids are "
        f"derived from content and position, so a parser that varies churns the whole "
        f"document on every re-ingest",
    )

    chunks: list[Chunk] = []
    if chunker is not None:
        if document is None:
            _fail("assert_round_trip needs a document to chunk against")
        chunks = chunker.chunk(document, blocks)
        chunk_items = _located_from_chunks(chunks)
        chunk_resolved = await _resolve_all(parser, raw, chunk_items)
        _assert_containment(chunk_items, chunk_resolved)
        _assert_tightness(chunk_items, chunk_resolved)
        _assert_discrimination(chunk_items, chunk_resolved, overlapping=True)
        repeat = chunker.chunk(document, again)
        _require(
            [chunk.model_dump_json() for chunk in chunks]
            == [chunk.model_dump_json() for chunk in repeat],
            f"{name}: re-parsing and re-chunking an unchanged document produced a different "
            f"chunk sequence, so an unchanged document would cost a full re-embed",
        )

    return RoundTripReport(
        fixture=name,
        blocks=len(blocks),
        unlocated=sum(1 for block in blocks if isinstance(block.anchor, Unlocated)),
        page_level=sum(1 for block in blocks if _is_page_level(block.anchor)),
        chunks=len(chunks),
    )


def assert_location_budget(
    profile: ParserProfile, reports: Iterable[RoundTripReport], *, blocks: int = 0
) -> None:
    """Assertion 5, over a parser's whole fixture corpus.

    Never pooled across parsers: pooling lets a well-behaved parser's fixtures pay for a
    badly-behaved one's, which is how a budget stops measuring anything.

    Args:
        profile: The parser's declared ceilings.
        reports: Every report from that parser's fixtures.
        blocks: A floor on the corpus size, so a suite that silently stopped producing
            fixtures cannot pass by dividing zero by zero.
    """
    collected = list(reports)
    total = sum(report.blocks for report in collected)
    _require(
        total >= blocks,
        f"{profile.name}: the fixture corpus produced {total} blocks, fewer than the {blocks} "
        f"this parser's suite claims to cover. A shrinking corpus passes every ratio",
    )
    if total == 0:
        _fail(f"{profile.name}: the fixture corpus produced no blocks at all")

    unlocated = sum(report.unlocated for report in collected)
    ratio = unlocated / total
    _require(
        ratio <= profile.max_unlocated_ratio + 1e-9,
        f"{profile.name}: {unlocated}/{total} blocks ({ratio:.1%}) are Unlocated, above the "
        f"declared {profile.max_unlocated_ratio:.1%}. Rule 1 — never invent a location — is "
        f"satisfiable by never producing one, and this ceiling is what stops that",
    )

    page_level = sum(report.page_level for report in collected)
    if profile.max_pagelevel_ratio is None:
        _require(
            page_level == 0,
            f"{profile.name}: {page_level} block(s) carry a page-level PageAnchor, but this "
            f"parser declared no page-level budget. Declare one, or emit rectangles",
        )
        return
    page_ratio = page_level / total
    _require(
        page_ratio <= profile.max_pagelevel_ratio + 1e-9,
        f"{profile.name}: {page_level}/{total} blocks ({page_ratio:.1%}) resolve only to a "
        f"page, above the declared {profile.max_pagelevel_ratio:.1%}. Losing the rectangles "
        f"is silent — the citation still names the right page — so the ratio is the only "
        f"signal that box extraction stopped working",
    )


__all__ = [
    "TIGHTNESS",
    "ParserProfile",
    "RoundTripReport",
    "assert_location_budget",
    "assert_round_trip",
]
