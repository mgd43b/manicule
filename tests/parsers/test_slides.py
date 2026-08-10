"""PPTX parsing: the slide number a reader would say, and the box a shape actually occupies.

The interesting assertions here are the three ways a rectangle stops being usable — geometry
that is absent, geometry shared by two shapes, and geometry that belongs to the notes page
rather than the slide — because each one costs a rectangle silently. The citation still names
the right slide, so nothing else would ever say.

:func:`test_shifting_every_slide_number_by_one_fails_the_round_trip` is what makes the harness
load-bearing rather than decorative: an off-by-one slide index is well-formed, plausible, and
wrong.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from manicule.chunking import StructuralChunker
from manicule.core.anchors import Anchor, PageAnchor, Unlocated
from manicule.core.content import BlockKind, ParsedBlock, RawDocument
from manicule.core.errors import ParseError
from manicule.core.protocols import Parser, parsing, read_blocks
from manicule.parsers.slides import SLIDES_MEDIA_TYPE, SlidesConfig, SlidesParser
from manicule.testing import assert_round_trip
from tests.parsers.support import check_corpus, check_fixture, raw_from

HARNESS_FIXTURES = (
    "slides_typical.pptx",
    "slides_structurally_hard.pptx",
    "slides_degenerate_image_only.pptx",
    "slides_degenerate_blank_slide.pptx",
    "slides_degenerate_no_slides.pptx",
    "slides_hostile_astral.pptx",
    "slides-large.pptx",
)

DECLINED_FIXTURES = (
    "slides_degenerate_zero_bytes.pptx",
    "slides_hostile_truncated.pptx",
    "slides_hostile_plain_zip.pptx",
)

SLIDE_WIDTH_INCHES = 10.0
SLIDE_HEIGHT_INCHES = 7.5
"""The default template's slide, in inches. The expected fractions below are computed from
these rather than copied from the parser, so the two are independent answers."""

TOLERANCE = 1e-6


def _parser(config: SlidesConfig | None = None) -> SlidesParser:
    return SlidesParser(config or SlidesConfig())


async def _blocks(path: Path, config: SlidesConfig | None = None) -> list[ParsedBlock]:
    return await read_blocks(_parser(config), raw_from(path, SLIDES_MEDIA_TYPE))


def _pages(blocks: Sequence[ParsedBlock]) -> list[int]:
    return [block.anchor.page for block in blocks if isinstance(block.anchor, PageAnchor)]


# --- slide numbers -----------------------------------------------------------------------


async def test_a_slide_number_is_its_position_in_presentation_order(corpus: Path) -> None:
    """1-based, in the order a viewer shows them, because that is what a person says.

    Positional numbering is not stable across a reorder, and that is the deliberate trade
    (``docs/parsing.md`` §2.4): a reorder changes the document's version token, so the deck is
    re-parsed. What must survive a reorder is the ability to tell a moved slide from a new one,
    and that is ``metadata.slide_id`` — the identifier PowerPoint keeps.
    """
    blocks = await _blocks(corpus / "slides" / "slides_typical.pptx")
    pages = _pages(blocks)

    assert pages == sorted(pages), "blocks must arrive in presentation order"
    assert min(pages) == 1, "slide numbers count from one, as a viewer shows them"
    assert max(pages) == 3

    identifiers = [block.metadata["slide_id"] for block in blocks]
    assert all(isinstance(identifier, int) for identifier in identifiers)
    assert len(set(map(str, identifiers))) == 3, "each slide carries its own stable identifier"


async def test_the_first_slide_title_is_a_heading_block_and_the_heading_path(corpus: Path) -> None:
    """A slide title is the only heading a deck has, so it is both a block and the breadcrumb.

    ``docs/parsing.md`` §2.4 is explicit that a parser whose anchor is not a ``HeadingAnchor``
    still populates ``heading_path`` where it can recover one — the breadcrumb goes into the
    embedding, so a slide's own title is the difference between a retrievable chunk and a
    paragraph about nothing.
    """
    blocks = await _blocks(corpus / "slides" / "slides_typical.pptx")
    first = blocks[0]

    assert first.kind is BlockKind.HEADING
    assert first.text == "Quarterly platform review"
    assert first.heading_path == ("Quarterly platform review",)
    assert all(
        block.heading_path == ("Quarterly platform review",)
        for block in blocks
        if isinstance(block.anchor, PageAnchor) and block.anchor.page == 1
    )


async def test_a_slide_with_no_title_has_an_empty_heading_path(corpus: Path) -> None:
    """An empty path, never "Slide 4": the breadcrumb reaches the embedder.

    A positional label there is a signal about nothing, and a wrong breadcrumb is worse than
    no breadcrumb because it actively moves the vector.
    """
    blocks = await _blocks(corpus / "slides" / "slides_structurally_hard.pptx")
    untitled = [block for block in blocks if isinstance(block.anchor, PageAnchor)]
    first_slide = [block for block in untitled if block.anchor == _anchor_of(blocks[0])]

    assert blocks[0].heading_path == ()
    assert all(block.heading_path == () for block in first_slide)


def _anchor_of(block: ParsedBlock) -> Anchor:
    return block.anchor


# --- rectangles --------------------------------------------------------------------------


async def test_a_rect_is_the_fraction_of_the_slide_its_shape_covers(corpus: Path) -> None:
    """Normalised against the slide's own dimensions, origin top-left, no y-flip.

    python-pptx reports EMU from the top-left, which is already ``Rect``'s convention, so the
    only work is the division — and the expected numbers here come from the inches the fixture
    was built with, not from re-running the parser's arithmetic.
    """
    blocks = await _blocks(corpus / "slides" / "slides_structurally_hard.pptx")
    panel = next(block for block in blocks if block.text.startswith("Backing panel"))

    assert isinstance(panel.anchor, PageAnchor)
    assert len(panel.anchor.rects) == 1
    rect = panel.anchor.rects[0]
    assert rect.x0 == pytest.approx(1.0 / SLIDE_WIDTH_INCHES, abs=TOLERANCE)
    assert rect.y0 == pytest.approx(1.0 / SLIDE_HEIGHT_INCHES, abs=TOLERANCE)
    assert rect.x1 == pytest.approx(5.0 / SLIDE_WIDTH_INCHES, abs=TOLERANCE)
    assert rect.y1 == pytest.approx(2.2 / SLIDE_HEIGHT_INCHES, abs=TOLERANCE)


async def test_overlapping_shapes_each_keep_their_own_box(corpus: Path) -> None:
    """Two boxes that intersect are still two boxes, and neither is merged into an envelope.

    Rule 3 of ``docs/parsing.md`` §2.1: an envelope covers text that was not quoted. Overlap is
    ordinary in a deck — a callout drawn over a panel — and it must not cost either shape its
    own rectangle.
    """
    blocks = await _blocks(corpus / "slides" / "slides_structurally_hard.pptx")
    panel = next(block for block in blocks if block.text.startswith("Backing panel"))
    callout = next(block for block in blocks if block.text.startswith("Callout text"))

    assert isinstance(panel.anchor, PageAnchor)
    assert isinstance(callout.anchor, PageAnchor)
    assert panel.anchor.page == callout.anchor.page
    assert panel.anchor.rects != callout.anchor.rects
    first, second = panel.anchor.rects[0], callout.anchor.rects[0]
    assert first.x1 > second.x0, "the fixture must actually overlap or it tests nothing"


async def test_two_shapes_at_identical_coordinates_are_page_level(corpus: Path) -> None:
    """A box that covers two shapes' text identifies neither of them.

    Stacked shapes at the same coordinates are a real habit — one shape backing another — and
    their box cannot discriminate. Emitting it anyway would give both shapes an anchor that
    resolves to both texts, which is the merged-envelope mistake arriving by a different route;
    page-level is the honest answer, and the 0.20 budget is what stops it becoming the default.
    """
    blocks = await _blocks(corpus / "slides" / "slides_structurally_hard.pptx")
    stacked = [block for block in blocks if "stacked shape" in block.text]

    assert len(stacked) == 2
    for block in stacked:
        assert isinstance(block.anchor, PageAnchor)
        assert block.anchor.rects == (), f"{block.text!r} kept a box it shares with another shape"


async def test_a_shape_with_no_reported_geometry_is_page_level(corpus: Path) -> None:
    """No ``<a:xfrm>`` means no position, and a box at the origin would be a fabrication.

    A rectangle in the top-left corner of a slide the quotation is not in looks deliberate,
    which is worse than no rectangle at all.
    """
    blocks = await _blocks(corpus / "slides" / "slides_structurally_hard.pptx")
    ungeometried = next(block for block in blocks if "no position at all" in block.text)

    assert isinstance(ungeometried.anchor, PageAnchor)
    assert ungeometried.anchor.rects == ()
    assert ungeometried.anchor.page == 5


async def test_a_page_level_anchor_resolves_to_the_whole_slide(corpus: Path) -> None:
    """That is what it claims, and claiming less would break containment.

    A page-level anchor is exempt from the tightness bound for this reason and capped by the
    page-level budget instead (``docs/parsing.md`` §3.3, assertion 5).
    """
    path = corpus / "slides" / "slides_structurally_hard.pptx"
    blocks = await _blocks(path)
    stacked = [block for block in blocks if "stacked shape" in block.text]
    resolved = await _parser().resolve(stacked[0].anchor, raw_from(path, SLIDES_MEDIA_TYPE))

    assert resolved is not None
    assert "First stacked shape" in resolved
    assert "Second stacked shape" in resolved


# --- speaker notes -----------------------------------------------------------------------


async def test_speaker_notes_are_indexed_as_prose_on_their_slide(corpus: Path) -> None:
    """The notes often carry the sentence the slide only gestures at.

    Their anchor is page-level by construction: a notes placeholder's geometry is on the notes
    page, not the slide, so a box taken from it would point at a coordinate on a different
    sheet of paper.
    """
    blocks = await _blocks(corpus / "slides" / "slides_typical.pptx")
    notes = next(block for block in blocks if block.metadata.get("speaker_notes") is True)

    assert notes.kind is BlockKind.PROSE
    assert "maintenance window in April" in notes.text
    assert isinstance(notes.anchor, PageAnchor)
    assert notes.anchor.page == 1
    assert notes.anchor.rects == ()


async def test_speaker_notes_can_be_left_out_by_configuration(corpus: Path) -> None:
    """A rehearsal script nobody should retrieve is a real deck, and a real setting.

    The setting changes what is indexed rather than how it is located, so no anchor moves when
    it is turned off.
    """
    path = corpus / "slides" / "slides_typical.pptx"
    without = await _blocks(path, SlidesConfig(include_speaker_notes=False))
    assert not any(block.metadata.get("speaker_notes") for block in without)
    assert not any("maintenance window" in block.text for block in without)


# --- tables and lists --------------------------------------------------------------------


async def test_a_table_on_a_slide_stays_whole_and_states_its_header_row(corpus: Path) -> None:
    """PowerPoint records a first-row flag, so the header count is read rather than guessed.

    ``metadata["rows"]`` is the rendered lines because that is where a row split cuts, and
    ``header_rows`` is what gets repeated into every part (``docs/parsing.md`` §4.2).
    """
    blocks = await _blocks(corpus / "slides" / "slides_structurally_hard.pptx")
    table = next(block for block in blocks if block.kind is BlockKind.TABLE)

    rows = table.metadata["rows"]
    assert isinstance(rows, list)
    assert "\n".join(str(row) for row in rows) == table.text
    assert table.metadata["header_rows"] == 1
    assert table.text.splitlines()[0] == "Region\tRequests\tErrors"


async def test_an_indented_text_frame_is_a_list_and_records_its_levels(corpus: Path) -> None:
    """The indent level is the only list signal python-pptx resolves.

    Bullet characters live in inherited list styles it does not read, so a frame with indented
    paragraphs is a list and a flat one is prose. Guessing "bulleted" from a placeholder's
    identity would label most prose a list.
    """
    blocks = await _blocks(corpus / "slides" / "slides_structurally_hard.pptx")
    deep = next(block for block in blocks if block.kind is BlockKind.LIST)

    assert deep.metadata["indent_levels"] == [0, 1, 2, 3, 4]
    assert len(deep.text.splitlines()) == 5


# --- the six assertions ------------------------------------------------------------------


async def test_the_corpus_round_trips_and_stays_inside_both_location_budgets(
    corpus: Path, chunker: StructuralChunker
) -> None:
    """Every fixture, blocks and chunks, plus the 0.00 unlocated and 0.20 page-level ceilings.

    The page-level ceiling is the one that matters here. Losing a rectangle is invisible — the
    citation still names the right slide — so the ratio is the only signal that box extraction
    has stopped working.
    """
    raws = [raw_from(corpus / "slides" / name, SLIDES_MEDIA_TYPE) for name in HARNESS_FIXTURES]
    # Chunked with notes off, and checked with notes on below. The chunker currently merges a
    # page-level anchor into a rectangle-bearing one when both are on the same slide, which
    # produces a chunk whose rectangles do not cover its notes text — reported, and a parser
    # cannot avoid it without pretending the notes have a box on the slide.
    reports = await check_corpus(
        _parser(SlidesConfig(include_speaker_notes=False)),
        raws,
        chunker=chunker,
        min_blocks=60,
    )
    with_notes = await check_corpus(_parser(), raws, min_blocks=60)

    blocks = sum(report.blocks for report in with_notes)
    page_level = sum(report.page_level for report in with_notes)
    assert page_level > 0, "the corpus must contain the page-level cases or the budget is untested"
    assert page_level / blocks < 0.20
    assert sum(report.chunks for report in reports) > 0


async def test_shifting_every_slide_number_by_one_fails_the_round_trip(corpus: Path) -> None:
    """An off-by-one slide index is the defect assertion 3 exists to catch.

    Nothing raises: every page number is a page the deck has, and every anchor resolves. Only
    the correspondence between the text and the location is broken, which is why it has to be
    tested rather than reviewed.
    """
    raw = raw_from(corpus / "slides" / "slides_typical.pptx", SLIDES_MEDIA_TYPE)
    with pytest.raises(AssertionError):
        await assert_round_trip(_ShiftedSlideParser(), raw, fixture="shifted")


# --- declines and empties ----------------------------------------------------------------


@pytest.mark.parametrize("name", DECLINED_FIXTURES)
async def test_an_unreadable_package_is_declined_rather_than_indexed(
    corpus: Path, name: str
) -> None:
    """Declining is how the next parser in the chain gets a turn.

    Zero bytes, a truncated ``ppt/presentation.xml`` and a plain zip under a `.pptx` name are
    three ways for the input not to be ours, and all three must name what was expected.
    """
    with pytest.raises(ParseError, match=r"not a readable \.pptx package"):
        await _blocks(corpus / "slides" / name)


@pytest.mark.parametrize(
    "name",
    [
        "slides_degenerate_image_only.pptx",
        "slides_degenerate_blank_slide.pptx",
        "slides_degenerate_no_slides.pptx",
    ],
)
async def test_a_deck_with_nothing_to_read_yields_no_blocks_and_does_not_raise(
    corpus: Path, name: str
) -> None:
    """An image with no text contributes nothing, and that is the honest outcome.

    Optical character recognition is out of scope and python-pptx exposes no alternative text,
    so there is nothing to index. Emitting a block naming the shape — "Picture 3" — would put a
    retrievable vector of noise in the index, and emitting an empty-text block is not a block
    at all. Zero blocks is what ``no_extractable_text`` is for.
    """
    assert await _blocks(corpus / "slides" / name) == []


async def test_astral_text_survives_into_the_title_and_the_body(corpus: Path) -> None:
    """Codepoints above the basic multilingual plane are content, in a title as anywhere."""
    raw = raw_from(corpus / "slides" / "slides_hostile_astral.pptx", SLIDES_MEDIA_TYPE)
    await check_fixture(_parser(), raw)
    blocks = await read_blocks(_parser(), raw)

    assert blocks[0].text == "図表 𝔊raphs"  # noqa: RUF001 - the astral letter is the assertion
    assert "𠀋" in blocks[1].text


async def test_no_block_is_unlocated(corpus: Path) -> None:
    """Every slide has a number, so there is never nothing to say about where text came from.

    The declared unlocated budget is 0.00 for exactly this reason: a PPTX block can be
    page-level when its box is unusable, but it is never unplaceable.
    """
    seen = 0
    for name in HARNESS_FIXTURES:
        blocks = await _blocks(corpus / "slides" / name)
        seen += len(blocks)
        assert not any(isinstance(block.anchor, Unlocated) for block in blocks), name
    assert seen, "every fixture yielded nothing, so the assertion above never ran"


# --- stream lifecycle --------------------------------------------------------------------


async def test_stopping_after_one_block_leaves_nothing_held_open(corpus: Path) -> None:
    """An abandoned generator must not be left suspended holding the presentation it opened.

    CPython finalises a live async generator through the event loop that created it, so one
    still suspended when that loop has closed is finalised against a torn-down runtime — a
    crash inside the interpreter's allocator rather than a warning. This parser reads the whole
    presentation before it yields anything, so nothing is live at a suspension point, and re-parsing
    immediately afterwards checks that nothing was left in a state the next read trips over.
    The pattern is shown catching a release placed after the loop in
    ``tests/parsers/test_word.py`` and in ``tests/test_parser_streams.py``.
    """
    raw = raw_from(corpus / "slides" / "slides_typical.pptx", SLIDES_MEDIA_TYPE)
    parser = _parser()
    async with parsing(parser, raw) as blocks:
        async for _ in blocks:
            break
    assert len(await read_blocks(parser, raw)) > 1


class _ShiftedSlideParser:
    """Numbers every block's slide one higher than the slide it came from.

    The house pattern from ``tests/fakes.py``, in the shape this format makes available: a
    plausible anchor that names the wrong slide. Rectangles are dropped so the anchor stays
    resolvable, which is what an off-by-one actually looks like — the citation opens a real
    slide and quotes text that is not on it.
    """

    media_types = frozenset({SLIDES_MEDIA_TYPE})
    profile = SlidesParser.profile

    def __init__(self) -> None:
        self._real = SlidesParser(SlidesConfig())

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        for block in await read_blocks(self._real, raw):
            if not isinstance(block.anchor, PageAnchor):  # pragma: no cover - all are page anchors
                yield block
                continue
            yield block.model_copy(update={"anchor": PageAnchor(page=block.anchor.page + 1)})

    async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
        return await self._real.resolve(anchor, raw)


_PARSERS: Sequence[Parser] = (SlidesParser(SlidesConfig()), _ShiftedSlideParser())
"""Type-checked conformance: the fake is a parser, not a mock of one."""
