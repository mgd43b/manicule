"""PDF, through pypdfium2.

The library choice is a license decision. PyMuPDF is AGPL-3.0, which an MIT project cannot
take however good it is; pypdfium2 is ``Apache-2.0 OR BSD-3-Clause`` and the pdfium it
bundles is BSD-3-Clause. Its binary wheels ship a ``BUILD_LICENSES/`` directory that must be
carried through into anything manicule redistributes.

**The expensive mistake available in this file is the coordinate transform**, and it is
expensive because it does not fail — it produces rectangles that are plausible, on the right
page, and in the wrong place. :func:`normalize_rect` is where it is prevented; read its
docstring before touching a coordinate.

The other traps, each of which also produces a wrong citation rather than an error, are
marked at the point where they are avoided:

- **Character indices are UTF-16 code units, not codepoints.** An astral character occupies
  two pdfium indices and one Python character. :class:`_TextPage` is the map between them.
- **Rectangle counting is stateful.** The count for a range must be requested before any
  rectangle is read.
- **Page indices are 0-based; ``PageAnchor.page`` is 1-based.** Converted once, in
  :meth:`PdfParser.parse`.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Generator
from dataclasses import dataclass

import pypdfium2 as pdfium

from manicule.core.anchors import Anchor, PageAnchor, Rect
from manicule.core.content import BlockKind, ParsedBlock, RawDocument
from manicule.core.errors import ParseError
from manicule.parsers.base import ParserProfile
from manicule.parsers.config import PDF_MEDIA_TYPES, PdfConfig

QUARTER_TURN, HALF_TURN, THREE_QUARTER_TURN = 90, 180, 270

_BMP_MAX = 0xFFFF
SURROGATE_PAIR_UNITS = 2
"""A codepoint above the basic multilingual plane occupies two UTF-16 code units, and pdfium
counts characters in code units. Both halves report the same character box."""

_BLANK_LINES_END_A_PARAGRAPH = 2
QUARTER_TURNS = (0, QUARTER_TURN, HALF_TURN, THREE_QUARTER_TURN)
"""The only rotations a page can be displayed at.

pdfium normalizes ``/Rotate`` into these four, so a malformed ``/Rotate 45`` arrives here as
``0``. The guard in :func:`normalize_rect` is defense in depth rather than a live branch: it
costs one comparison and it means a future reader who reaches for the raw dictionary value
does not silently produce rectangles rotated by a fraction of a turn.
"""


def normalize_rect(
    rect: tuple[float, float, float, float],
    box: tuple[float, float, float, float],
    rotation: int,
) -> Rect | None:
    """Convert a pdfium rectangle into normalized, rotation-and-crop-aware page coordinates.

    **pdfium reports character and rectangle coordinates in raw PDF user space** — bottom-left
    origin, points, with ``/Rotate`` **not** applied, the CropBox **not** applied, and the
    MediaBox origin **not** subtracted. The same text returns byte-identical rectangles across
    ``/Rotate`` 0, 90, 180 and 270, across a MediaBox of ``[0 0 612 792]`` versus
    ``[50 50 662 842]``, and with a CropBox applied. Content objects stay in user space
    because the page matrix that encodes crop and rotation is only composed in at render time.

    **The page-size call, however, honors both.** It returns ``(792, 612)`` for a
    ``/Rotate 90`` letter page and ``(350, 700)`` for a cropped one.

    So the two live in different coordinate spaces, and the obvious one-liner —
    ``top = page_height - rect_top`` — is silently wrong on every rotated or cropped page. On
    a square, unrotated page it is not even visibly wrong.

    The transform, in order, using the page's bounding box and rotation and **never** the
    page-size convenience:

    1. **Translate** by the box origin, so coordinates are relative to the visible page
       rather than to the media.
    2. **Rotate** by ``/Rotate`` about the box, swapping width and height for 90 and 270. A
       page displayed at ``/Rotate 90`` is turned clockwise, so user-space "up" becomes
       display "right" and user-space "right" becomes display "down".
    3. **Flip y** using the height of the box *after* rotation, and divide through.

    Args:
        rect: ``(x0, y0, x1, y1)`` as pdfium reports it, bottom-left origin.
        box: The page's bounding box in user space — the CropBox intersected with the
            MediaBox, which is what a viewer displays.
        rotation: Degrees clockwise, one of :data:`QUARTER_TURNS`.

    Returns:
        The rectangle in normalized page coordinates, or ``None`` when the page's rotation is
        not a quarter turn or the box is degenerate. ``None`` means the block keeps its page
        number and loses its rectangle, which is a coarser citation rather than a wrong one.
    """
    if rotation not in QUARTER_TURNS:
        return None
    left, bottom, right, top = box
    width, height = right - left, top - bottom
    if width <= 0 or height <= 0:
        return None

    corners = [
        _to_display(x - left, y - bottom, width, height, rotation)
        for x, y in ((rect[0], rect[1]), (rect[2], rect[3]))
    ]
    page_width, page_height = (
        (height, width) if rotation in (QUARTER_TURN, THREE_QUARTER_TURN) else (width, height)
    )
    xs = [x / page_width for x, _ in corners]
    ys = [y / page_height for _, y in corners]
    with contextlib.suppress(ValueError):
        return Rect(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))
    # Outside the page by more than float noise: the rectangle belongs to content the crop
    # removed, so there is no place on the displayed page to point at.
    return None


def _to_display(
    x: float, y: float, width: float, height: float, rotation: int
) -> tuple[float, float]:
    """One point, box-relative and bottom-left, to display coordinates, top-left origin."""
    if rotation == QUARTER_TURN:
        return y, x
    if rotation == HALF_TURN:
        return width - x, y
    if rotation == THREE_QUARTER_TURN:
        return height - y, width - x
    return x, height - y


def denormalize_rect(
    rect: Rect, box: tuple[float, float, float, float], rotation: int
) -> tuple[float, float, float, float]:
    """The inverse of :func:`normalize_rect`, in user space, for resolving an anchor.

    Resolution asks pdfium what text lies inside a region, and pdfium only understands user
    space. Inverting here rather than remembering the original rectangle is deliberate: it
    means :meth:`PdfParser.resolve` re-derives the location from the stored anchor and the
    bytes, so a stored anchor that has drifted from its document shows up as text that does
    not match rather than as a lookup that quietly succeeds.
    """
    left, bottom, right, top = box
    width, height = right - left, top - bottom
    page_width, page_height = (
        (height, width) if rotation in (QUARTER_TURN, THREE_QUARTER_TURN) else (width, height)
    )
    corners = [
        _from_display(x * page_width, y * page_height, width, height, rotation)
        for x, y in ((rect.x0, rect.y0), (rect.x1, rect.y1))
    ]
    xs = [x + left for x, _ in corners]
    ys = [y + bottom for _, y in corners]
    return min(xs), min(ys), max(xs), max(ys)


def _from_display(
    x: float, y: float, width: float, height: float, rotation: int
) -> tuple[float, float]:
    if rotation == QUARTER_TURN:
        return y, x
    if rotation == HALF_TURN:
        return width - x, y
    if rotation == THREE_QUARTER_TURN:
        return width - y, height - x
    return x, height - y


@dataclass(frozen=True, slots=True)
class _TextPage:
    """A page's text, plus the map between Python characters and pdfium char indices.

    pdfium counts characters in **UTF-16 code units**: a page reading ``😀B😀B`` reports six
    characters for four Python ones, and both halves of a surrogate pair carry the same
    character box. So a Python string offset is not a pdfium index, and passing one straight
    through asks for the rectangles of a different run of glyphs — on the right page, in the
    wrong place, with nothing raised. ``offsets[i]`` is the pdfium index of Python character
    ``i``; the list has one trailing entry so a half-open Python range converts directly.
    """

    text: str
    offsets: tuple[int, ...]

    @classmethod
    def read(cls, textpage: pdfium.PdfTextPage) -> _TextPage:
        count = textpage.count_chars()
        text = textpage.get_text_range(0, count) if count else ""
        offsets: list[int] = []
        index = 0
        for character in text:
            offsets.append(index)
            index += SURROGATE_PAIR_UNITS if ord(character) > _BMP_MAX else 1
        offsets.append(index)
        return cls(text=text, offsets=tuple(offsets))

    def native_range(self, start: int, end: int) -> tuple[int, int]:
        """A Python half-open range as a pdfium ``(index, count)`` pair."""
        first = self.offsets[start]
        last = self.offsets[end]
        return first, last - first


@dataclass(frozen=True, slots=True)
class _Page:
    """Everything about one page that both parsing and resolving need."""

    number: int
    box: tuple[float, float, float, float]
    rotation: int
    text: _TextPage


class PdfParser:
    """Extracts text from a PDF with real page and rectangle provenance.

    A :class:`PageAnchor` is only ever constructed from a page index pdfium reports, and its
    rectangles only ever from character boxes pdfium measured. Neither is inferred from the
    shape of the extracted text.
    """

    media_types = PDF_MEDIA_TYPES
    profile = ParserProfile(name="pdf", max_unlocated_ratio=0.00, max_pagelevel_ratio=0.10)

    def __init__(self, config: PdfConfig) -> None:
        self._config = config

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        """Yield one block per paragraph, in the order pdfium reports the text.

        No reordering heuristic is applied to multi-column layouts. pdfium returns text in
        content-stream order, which for two columns can interleave them; a reordering
        heuristic that is wrong produces chunks whose text never appeared contiguously in the
        document, which is a quotation of something nobody wrote.
        """
        with _open(raw) as document:
            outline = _outline_paths(document) if self._config.outline_headings else {}
            if len(document) > self._config.max_pages:
                msg = (
                    f"{raw.uri}: {len(document)} pages, over the {self._config.max_pages}-page "
                    f"limit. Raise parser.pdf.maxPages, or split the document."
                )
                raise ParseError(msg)
            for index in range(len(document)):
                page = _read_page(document, index)
                heading_path = outline.get(index, ())
                for start, end in _paragraph_ranges(page.text.text):
                    body = page.text.text[start:end]
                    yield ParsedBlock(
                        kind=BlockKind.PROSE,
                        text=body,
                        anchor=self._anchor_for(document, page, start, end),
                        heading_path=heading_path,
                    )

    def _anchor_for(
        self, document: pdfium.PdfDocument, page: _Page, start: int, end: int
    ) -> Anchor:
        rects = _rects_for(document, page, start, end)
        return PageAnchor(page=page.number, rects=rects)

    async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
        """Return the text on the page, or inside the rectangles, the anchor names."""
        if not isinstance(anchor, PageAnchor):
            return None
        with _open(raw) as document:
            if anchor.page > len(document):
                return None
            page = _read_page(document, anchor.page - 1)
            if not anchor.rects:
                return page.text.text or None
            raw_page = document[anchor.page - 1]
            textpage = raw_page.get_textpage()
            try:
                pieces = [
                    _bounded_text(textpage, denormalize_rect(rect, page.box, page.rotation))
                    for rect in anchor.rects
                ]
            finally:
                textpage.close()
            joined = "\n".join(piece for piece in pieces if piece)
            return joined or None


BOX_MARGIN = 0.5
"""How far a box is grown before it is quoted back to pdfium, in points.

A character box and the glyph inside it agree to within rounding, and a box handed back
exactly as measured can exclude the very glyph it came from — which reads as a citation
resolving to nothing at all."""


def _bounded_text(
    textpage: pdfium.PdfTextPage, user_rect: tuple[float, float, float, float]
) -> str:
    """The text inside a user-space rectangle, as pdfium reads it back."""
    left, bottom, right, top = user_rect
    return textpage.get_text_bounded(
        left - BOX_MARGIN, bottom - BOX_MARGIN, right + BOX_MARGIN, top + BOX_MARGIN
    )


@contextlib.contextmanager
def _open(raw: RawDocument) -> Generator[pdfium.PdfDocument]:
    """Open a PDF, translating the failures that mean "not for me" into a decline.

    A PDF with a *user* password cannot be read at all. One with only an **owner** password
    can be, and is parsed normally: owner-password PDFs are common and refusing them would
    drop real content over a restriction that was never about reading.
    """
    try:
        document = pdfium.PdfDocument(raw.as_bytes())
    except pdfium.PdfiumError as exc:
        reason = "encrypted" if "password" in str(exc).lower() else str(exc)
        msg = f"{raw.uri}: cannot be opened ({reason})."
        raise ParseError(msg) from exc
    try:
        yield document
    finally:
        document.close()


def _read_page(document: pdfium.PdfDocument, index: int) -> _Page:
    page = document[index]
    textpage = page.get_textpage()
    try:
        text = _TextPage.read(textpage)
    finally:
        textpage.close()
    return _Page(
        number=index + 1,
        box=page.get_bbox(),
        rotation=page.get_rotation(),
        text=text,
    )


def _rects_for(document: pdfium.PdfDocument, page: _Page, start: int, end: int) -> tuple[Rect, ...]:
    """The rectangles covering a character range, one per line pdfium reports.

    Several rectangles is the normal case, not an edge case: a quote spanning a line break
    has one per line, and a quote spanning a column break has rectangles on both sides of the
    gutter. They are stored as they come — merging them into one envelope would highlight
    text that was not quoted.
    """
    first, count = page.text.native_range(start, end)
    if count <= 0:
        return ()
    raw_page = document[page.number - 1]
    textpage = raw_page.get_textpage()
    try:
        # The count must be requested before any rectangle is read: reading them without it
        # returns nothing useful, which produces empty rects — a silently page-level anchor —
        # rather than an error.
        total = textpage.count_rects(first, count)
        boxes = [textpage.get_rect(index) for index in range(total)]
    finally:
        textpage.close()
    normalized = (normalize_rect(box, page.box, page.rotation) for box in boxes)
    return tuple(rect for rect in normalized if rect is not None)


def _paragraph_ranges(text: str) -> list[tuple[int, int]]:
    """Half-open character ranges of the page's paragraphs, in the order they arrived.

    Splitting the *text of a known page* into paragraphs is a content decision; the location
    still comes from pdfium, which measures the character boxes of whichever range is asked
    for. Nothing here decides where a page begins or ends — that is the page index, and it is
    the library's answer.
    """
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    blank = 0
    for index, character in enumerate(text):
        if character == "\n":
            blank += 1
            if blank >= _BLANK_LINES_END_A_PARAGRAPH and start is not None:
                ranges.append((start, index - 1))
                start = None
            continue
        blank = 0
        if not character.isspace() and start is None:
            start = index
    if start is not None:
        ranges.append((start, len(text)))
    return [(begin, end) for begin, end in ranges if text[begin:end].strip()]


def _outline_paths(document: pdfium.PdfDocument) -> dict[int, tuple[str, ...]]:
    """Heading path per page index, taken from the document outline and nowhere else.

    A PDF has no heading semantics — only glyphs with font sizes — so inferring a hierarchy
    from font size is a heuristic, and ``heading_path`` feeds the breadcrumb, which goes into
    the embedding. A wrong heading actively degrades retrieval rather than merely failing to
    help. An outline-less PDF therefore produces an empty heading path, and its page and
    rectangle provenance is exact regardless.
    """
    stack: list[str] = []
    paths: dict[int, tuple[str, ...]] = {}
    try:
        bookmarks: list[pdfium.PdfBookmark] = list(document.get_toc())
    except pdfium.PdfiumError:
        # A malformed outline is not a reason to lose the document's text. The pages keep
        # their exact provenance; only the breadcrumb is poorer for it.
        return {}
    for bookmark in bookmarks:
        title: str = (bookmark.get_title() or "").strip()
        if not title:
            continue
        level: int = bookmark.level
        del stack[level:]
        stack.append(title)
        destination = bookmark.get_dest()
        if destination is None:
            continue
        page_index: int | None = destination.get_index()
        if page_index is not None:
            paths.setdefault(page_index, tuple(stack))
    return _fill_forward(paths)


def _fill_forward(paths: dict[int, tuple[str, ...]]) -> dict[int, tuple[str, ...]]:
    """Carry each outline entry's path forward until the next entry starts.

    A bookmark names where a section *begins*; the pages after it are still in that section
    until another bookmark says otherwise. Leaving them empty would give consecutive pages of
    one section different breadcrumbs.
    """
    if not paths:
        return {}
    filled: dict[int, tuple[str, ...]] = {}
    current: tuple[str, ...] = ()
    for index in range(min(paths), max(paths) + 1):
        current = paths.get(index, current)
        filled[index] = current
    return filled


def outline_page_paths(raw: RawDocument) -> dict[int, tuple[str, ...]]:
    """The outline heading paths for a document, for diagnostics."""
    with _open(raw) as document:
        return _outline_paths(document)


__all__ = [
    "BOX_MARGIN",
    "PDF_MEDIA_TYPES",
    "QUARTER_TURNS",
    "PdfConfig",
    "PdfParser",
    "denormalize_rect",
    "normalize_rect",
    "outline_page_paths",
]
