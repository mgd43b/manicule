"""PDF parsing, and above all the coordinate transform.

The transform from pdfium's raw user-space rectangles to normalised page coordinates is the
most expensive mistake available in this parser, because it cannot fail loudly: the wrong
answer is a rectangle that is plausible, on the right page, and in the wrong place. So the
central test here does not check arithmetic against arithmetic. It **renders the page with
pdfium's own renderer** — which composes the page matrix that encodes crop and rotation — and
checks that the ink lands where the parser says the text is. Two independent paths through
the same geometry, one of which is the thing a reader actually sees.

:func:`test_the_naive_transform_lands_the_marker_in_the_wrong_place` is what makes that check
load-bearing rather than decorative: it runs the obvious wrong implementation past the same
assertion and shows it failing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pypdfium2 as pdfium
import pytest

from manicule.chunking import StructuralChunker
from manicule.core.anchors import Anchor, PageAnchor, Rect
from manicule.core.content import BlockKind, ParsedBlock, RawDocument
from manicule.core.errors import ParseError
from manicule.core.protocols import Parser
from manicule.testing import assert_round_trip
from tests.corpus.pdf import MARKER
from tests.parsers.support import check_corpus, check_fixture, raw_from

MEDIA_TYPE = "application/pdf"

GEOMETRY_FIXTURES = (
    "upright.pdf",
    "rotated-90.pdf",
    "rotated-180.pdf",
    "rotated-270.pdf",
    "cropped.pdf",
    "cropped-rotated.pdf",
    "mediabox-offset.pdf",
)

RENDER_SCALE = 4.0
"""Enough resolution that a 12-point glyph is several pixels across, so the ink bounding box
is a real measurement rather than a rounding artefact."""

INK_THRESHOLD = 250
"""How dark a channel must be to count as ink. Anti-aliasing puts grey around every glyph, so
a strict "not pure white" test would grow the measured box by the antialiasing radius."""

BOX_TOLERANCE = 0.02
"""Two percent of the page. The rendered ink box and the character box disagree by the glyph
bearings — the blank margin a font leaves inside each character cell — which is a fraction of
a 12-point glyph on a 612-point page. A transform that is actually wrong is wrong by whole
percentages or by orders of magnitude, never by this."""


def _pdf_parser() -> Parser:
    from manicule.parsers.pdf import PdfConfig, PdfParser  # noqa: PLC0415 - heavy import

    return PdfParser(PdfConfig())


def _ink_box(path: Path) -> Rect:
    """Where the ink is on the first page, as pdfium's renderer draws it.

    Reads the bitmap buffer directly rather than through an image library: the check must not
    depend on a package that is not part of the parser stack, or it would be skipped on the
    machines that most need it.
    """
    document = pdfium.PdfDocument(path.read_bytes())
    try:
        bitmap = document[0].render(scale=RENDER_SCALE)
        pixels = bytes(bitmap.buffer)
        width, height, stride, channels = (
            bitmap.width,
            bitmap.height,
            bitmap.stride,
            bitmap.n_channels,
        )
    finally:
        document.close()

    left, top, right, bottom = width, height, -1, -1
    for row in range(height):
        base = row * stride
        for column in range(width):
            offset = base + column * channels
            if min(pixels[offset : offset + channels]) < INK_THRESHOLD:
                left = min(left, column)
                right = max(right, column)
                top = min(top, row)
                bottom = max(bottom, row)
    if right < 0:
        message = f"{path.name} rendered no ink, so there is nothing to compare a rect against"
        raise AssertionError(message)
    return Rect(
        x0=left / width,
        y0=top / height,
        x1=(right + 1) / width,
        y1=(bottom + 1) / height,
    )


def _covers(outer: Rect, inner: Rect, tolerance: float = BOX_TOLERANCE) -> bool:
    return (
        outer.x0 <= inner.x0 + tolerance
        and outer.y0 <= inner.y0 + tolerance
        and outer.x1 >= inner.x1 - tolerance
        and outer.y1 >= inner.y1 - tolerance
    )


async def _blocks(raw: RawDocument) -> list[ParsedBlock]:
    return [block async for block in _pdf_parser().parse(raw)]


async def _first_rect(parser: Parser, raw: RawDocument) -> Rect:
    blocks = [block async for block in parser.parse(raw)]
    anchors = [block.anchor for block in blocks if isinstance(block.anchor, PageAnchor)]
    if not anchors or not anchors[0].rects:
        message = f"{raw.uri} produced no rectangles, so the transform cannot be checked"
        raise AssertionError(message)
    rects = anchors[0].rects
    return Rect(
        x0=min(rect.x0 for rect in rects),
        y0=min(rect.y0 for rect in rects),
        x1=max(rect.x1 for rect in rects),
        y1=max(rect.y1 for rect in rects),
    )


# --- the coordinate transform ------------------------------------------------------------


@pytest.mark.parametrize("name", GEOMETRY_FIXTURES)
async def test_a_rect_lands_where_pdfium_renders_the_glyphs(corpus: Path, name: str) -> None:
    """The rectangle a citation highlights covers the text a reader sees, on every geometry.

    pdfium reports character boxes in raw user space with ``/Rotate``, the CropBox and the
    MediaBox origin all unapplied, while the page-size call applies all three. Anything that
    conflates the two spaces produces a rectangle on the right page in the wrong place, and
    nothing raises. Rendering is the independent answer.
    """
    path = corpus / "pdf" / name
    measured = await _first_rect(_pdf_parser(), raw_from(path, MEDIA_TYPE))
    assert _covers(measured, _ink_box(path)), (
        f"{name}: the parser says the text is at {measured}, but pdfium renders it at "
        f"{_ink_box(path)}"
    )


def _naive_rect(box: tuple[float, float, float, float], width: float, height: float) -> Rect:
    """``page_height - rect_top``, clamped — the transform this project does not use."""
    x0, y0, x1, y1 = box
    return Rect(
        x0=min(max(x0 / width, 0.0), 1.0),
        y0=min(max((height - y1) / height, 0.0), 1.0),
        x1=min(max(x1 / width, 0.0), 1.0),
        y1=min(max((height - y0) / height, 0.0), 1.0),
    )


class _NaivePdfParser:
    """The obvious wrong transform: flip against the page size, ignore crop and rotation.

    Exists so the render check above is demonstrably load-bearing. This is what an
    implementation that reads the page-size convenience and subtracts looks like, and it is
    right on an upright uncropped page — which is why it survives review and fails in
    production.
    """

    def __init__(self) -> None:
        self._real = _pdf_parser()

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        document = pdfium.PdfDocument(raw.as_bytes())
        try:
            page = document[0]
            width, height = page.get_size()
            textpage = page.get_textpage()
            try:
                count = textpage.count_chars()
                text = textpage.get_text_range(0, count) if count else MARKER
                total = textpage.count_rects(0, count)
                rects = [
                    _naive_rect(textpage.get_rect(index), width, height) for index in range(total)
                ]
            finally:
                textpage.close()
            yield ParsedBlock(
                kind=BlockKind.PROSE,
                text=text,
                anchor=PageAnchor(page=1, rects=tuple(rects)),
            )
        finally:
            document.close()

    async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
        return await self._real.resolve(anchor, raw)

    @property
    def media_types(self) -> frozenset[str]:
        return frozenset({MEDIA_TYPE})


@pytest.mark.parametrize("name", ["rotated-90.pdf", "cropped.pdf", "mediabox-offset.pdf"])
async def test_the_naive_transform_lands_the_marker_in_the_wrong_place(
    corpus: Path, name: str
) -> None:
    """The render check catches the mistake it exists to catch.

    A check nobody has watched fail is a check nobody knows works. This runs the plausible
    wrong implementation past the same assertion and requires it to fail — so if the
    assertion is ever weakened into uselessness, this test goes green in the wrong direction
    and fails.
    """
    path = corpus / "pdf" / name
    measured = await _first_rect(_NaivePdfParser(), raw_from(path, MEDIA_TYPE))
    assert not _covers(measured, _ink_box(path)), (
        f"{name}: the naive transform agreed with the renderer, so this fixture no longer "
        f"distinguishes a correct transform from an incorrect one"
    )


async def test_an_upright_page_is_the_case_where_both_transforms_agree(corpus: Path) -> None:
    """Named because it explains why the wrong transform survives review.

    On an upright, uncropped page the naive flip is correct. Every fixture that is *only*
    upright certifies nothing about rotation or cropping, which is why the geometry fixtures
    above are required rather than nice to have.
    """
    path = corpus / "pdf" / "upright.pdf"
    naive = await _first_rect(_NaivePdfParser(), raw_from(path, MEDIA_TYPE))
    correct = await _first_rect(_pdf_parser(), raw_from(path, MEDIA_TYPE))
    assert abs(naive.x0 - correct.x0) < 1e-6
    assert abs(naive.y0 - correct.y0) < 1e-6


async def test_turning_the_page_moves_the_marker_to_a_different_edge(corpus: Path) -> None:
    """Stated as corners, so the expectation is readable without running a renderer.

    The marker is drawn near the top-left of the upright page. Turning the page 90° clockwise
    must carry it to the right-hand edge; 180° to the bottom-right; 270° to the bottom-left.
    """

    async def corner(name: str) -> tuple[float, float]:
        rect = await _first_rect(_pdf_parser(), raw_from(corpus / "pdf" / name, MEDIA_TYPE))
        return (rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2

    assert await corner("upright.pdf") == pytest.approx((0.15, 0.12), abs=0.1)
    assert await corner("rotated-90.pdf") == pytest.approx((0.88, 0.18), abs=0.1), (
        "a 90° turn must carry the marker to the right-hand edge"
    )
    assert await corner("rotated-180.pdf") == pytest.approx((0.82, 0.88), abs=0.1)
    assert await corner("rotated-270.pdf") == pytest.approx((0.11, 0.82), abs=0.1)


async def test_a_cropped_page_measures_against_the_visible_box_not_the_sheet(
    corpus: Path,
) -> None:
    """A CropBox with an origin away from zero changes both the offset and the scale.

    Not subtracting the origin shifts every rectangle by the crop offset; not dividing by the
    cropped size scales them wrongly. Both survive as plausible fractions, which is why the
    cropped fixture carries an origin *and* a different size.
    """
    cropped = await _first_rect(_pdf_parser(), raw_from(corpus / "pdf" / "cropped.pdf", MEDIA_TYPE))
    upright = await _first_rect(_pdf_parser(), raw_from(corpus / "pdf" / "upright.pdf", MEDIA_TYPE))
    assert cropped.x0 != pytest.approx(upright.x0, abs=1e-3)
    assert _covers(cropped, _ink_box(corpus / "pdf" / "cropped.pdf"))


# --- astral-plane text -------------------------------------------------------------------


async def test_astral_text_addresses_the_glyphs_the_block_claims(corpus: Path) -> None:
    """A surrogate pair is two pdfium characters and one Python character.

    pdfium counts characters in UTF-16 code units, so a page reading ``😀B😀B`` reports six
    characters where Python sees four. A parser that hands a Python offset to pdfium asks
    about a different run of glyphs — right page, wrong place, nothing raised. The block's
    text must contain both emoji and its rectangles must cover all four glyphs.
    """
    path = corpus / "pdf" / "astral.pdf"
    raw = raw_from(path, MEDIA_TYPE)
    blocks = [block async for block in _pdf_parser().parse(raw)]
    assert blocks, "the astral fixture produced no blocks"
    assert blocks[0].text.count("\U0001f600") == 2
    measured = await _first_rect(_pdf_parser(), raw)
    assert _covers(measured, _ink_box(path))


# --- declines, empties and statuses ------------------------------------------------------


async def test_a_page_with_no_text_layer_yields_no_blocks_and_does_not_raise(
    corpus: Path,
) -> None:
    """ "Nothing to extract" and "the parser broke" must stay distinguishable.

    Optical character recognition is out of scope, so a scanned page yields nothing — and
    that has to arrive as an empty parse rather than an exception, because the two lead to
    different document statuses and different remedies: one is a scanning question, the other
    is a bug report.
    """
    raw = raw_from(corpus / "pdf" / "no-text-layer.pdf", MEDIA_TYPE)
    blocks = [block async for block in _pdf_parser().parse(raw)]
    assert blocks == []


async def test_a_user_password_is_declined_and_an_owner_password_is_read(corpus: Path) -> None:
    """Owner-password PDFs are common and refusing them would drop real content.

    A user password genuinely prevents reading. An owner password restricts printing and
    editing, and pdfium reads the document normally — the restriction was never about
    reading, so treating the two alike would lose documents for no benefit.
    """
    locked = raw_from(corpus / "pdf" / "user-password.pdf", MEDIA_TYPE)
    with pytest.raises(ParseError, match="encrypted"):
        await _blocks(locked)

    owner = raw_from(corpus / "pdf" / "owner-password.pdf", MEDIA_TYPE)
    blocks = [block async for block in _pdf_parser().parse(owner)]
    assert any(MARKER in block.text for block in blocks)


@pytest.mark.parametrize("name", ["zero-bytes.pdf", "truncated.pdf"])
async def test_bytes_that_are_not_a_readable_pdf_are_declined(corpus: Path, name: str) -> None:
    """Declining lets the next parser in the chain try; raising anything else fails the
    document outright, which is the wrong answer for a file that may simply be misnamed."""
    unreadable = raw_from(corpus / "pdf" / name, MEDIA_TYPE)
    with pytest.raises(ParseError):
        await _blocks(unreadable)


# --- anchors, headings and the round trip -------------------------------------------------


async def test_page_numbers_count_from_one_as_a_reader_does(corpus: Path) -> None:
    """pdfium indexes pages from zero and every viewer numbers them from one.

    Converted once, at the parser boundary. An off-by-one here produces a citation that
    resolves to the adjacent page, which reads perfectly and is wrong.
    """
    raw = raw_from(corpus / "pdf" / "typical.pdf", MEDIA_TYPE)
    pages = {
        block.anchor.page
        async for block in _pdf_parser().parse(raw)
        if isinstance(block.anchor, PageAnchor)
    }
    assert pages == {1, 2, 3}


async def test_a_paragraph_spanning_lines_keeps_one_rect_per_line(corpus: Path) -> None:
    """Rectangles are stored as measured, never merged into one envelope.

    A quote spanning a line break occupies two boxes with a gap between them. Their union
    covers the space between the lines and, in a two-column layout, the whole gutter and the
    text on the other side of it — a highlight that is confidently wrong.
    """
    raw = raw_from(corpus / "pdf" / "multicolumn.pdf", MEDIA_TYPE)
    blocks = [block async for block in _pdf_parser().parse(raw)]
    multiline = [
        block
        for block in blocks
        if isinstance(block.anchor, PageAnchor) and len(block.anchor.rects) > 1
    ]
    assert multiline, "a multi-line paragraph produced a single rectangle"


async def test_heading_paths_come_from_the_outline_and_are_empty_without_one(
    corpus: Path,
) -> None:
    """A PDF has no heading semantics, only glyphs with font sizes.

    Inferring a hierarchy from font size is a heuristic, and ``heading_path`` feeds the
    breadcrumb, which goes into the embedding — so a wrong heading actively degrades
    retrieval rather than merely failing to help. An outline-less PDF gets an empty path and
    keeps its exact page and rectangle provenance.
    """
    raw = raw_from(corpus / "pdf" / "typical.pdf", MEDIA_TYPE)
    blocks = [block async for block in _pdf_parser().parse(raw)]
    assert blocks
    assert all(block.heading_path == () for block in blocks)


class _OffByOnePdfParser:
    """A parser whose page numbers are one too high — the classic 0-based/1-based slip."""

    media_types = frozenset({MEDIA_TYPE})

    def __init__(self) -> None:
        self._real = _pdf_parser()

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        async for block in self._real.parse(raw):
            anchor = block.anchor
            if isinstance(anchor, PageAnchor):
                yield block.model_copy(
                    update={"anchor": PageAnchor(page=anchor.page + 1, rects=())}
                )
            else:  # pragma: no cover - the PDF parser only produces page anchors
                yield block

    async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
        return await self._real.resolve(anchor, raw)


async def test_an_off_by_one_page_index_is_caught_by_the_round_trip(corpus: Path) -> None:
    """The failure this whole harness exists for: a citation pointing at the wrong page.

    Every page of the fixture carries its own number in the text, so a shifted anchor
    resolves to text the block does not claim. Without the check the citation looks perfect.
    """
    raw = raw_from(corpus / "pdf" / "typical.pdf", MEDIA_TYPE)
    with pytest.raises(AssertionError):
        await assert_round_trip(_OffByOnePdfParser(), raw, fixture="typical.pdf")


async def test_every_pdf_fixture_round_trips_within_the_declared_budgets(
    corpus: Path, chunker: StructuralChunker
) -> None:
    """The whole corpus, through all six assertions and the page-level budget.

    A PDF chunk is never *unlocated* — pdfium reports a page index for every page — so at
    worst it is *page-level*, when box extraction fails for a run of glyphs. Losing the boxes
    is silent, because the citation still names the right page, so the ratio is the only
    signal that it happened.
    """
    readable: Sequence[str] = (
        "typical.pdf",
        "upright.pdf",
        "rotated-90.pdf",
        "rotated-180.pdf",
        "rotated-270.pdf",
        "cropped.pdf",
        "cropped-rotated.pdf",
        "mediabox-offset.pdf",
        "multicolumn.pdf",
        "astral.pdf",
        "owner-password.pdf",
        "many-pages-large.pdf",
    )
    raws = [raw_from(corpus / "pdf" / name, MEDIA_TYPE) for name in readable]
    await check_corpus(_pdf_parser(), raws, chunker=chunker, min_blocks=len(readable))


async def test_the_page_with_no_text_contributes_no_blocks_to_the_budget(
    corpus: Path,
) -> None:
    """A document that yields nothing is a normal outcome and must not be counted as located
    or unlocated — it has nothing to locate."""
    report = await check_fixture(
        _pdf_parser(), raw_from(corpus / "pdf" / "no-text-layer.pdf", MEDIA_TYPE)
    )
    assert report.blocks == 0
