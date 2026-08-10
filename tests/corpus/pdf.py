"""PDF fixtures.

The rotated and cropped pages are the point of this file. The transform from pdfium's raw
user-space rectangles to normalised page coordinates cannot fail loudly — it produces
rectangles that are plausible, on the right page, and in the wrong place — so it is only
caught by a page where the right answer and the naive answer differ, which is exactly a page
that is rotated or cropped. On a square unrotated page the two agree.

``MARKER`` is drawn at a known user-space position on every geometry fixture, near one
corner, so that "where does this end up when the page is turned" has a visible answer.
"""

from __future__ import annotations

import io
from pathlib import Path

import pypdfium2 as pdfium
from reportlab.lib import pdfencrypt
from reportlab.pdfgen import canvas as rl_canvas

LETTER = (612.0, 792.0)
MARKER = "ANCHORTEST"
MARKER_AT = (72.0, 700.0)
"""Where :data:`MARKER` is drawn, in user space: near the **top-left** of an upright letter
page, since ``y`` counts up from the bottom. Rotating the page must move it somewhere else,
and which corner it lands in is what the tests assert."""

FONT_SIZE = 12


def _page(
    text: list[str],
    *,
    pagesize: tuple[float, float] = LETTER,
    encryption: pdfencrypt.StandardEncryption | None = None,
    pages: int = 1,
) -> bytes:
    buffer = io.BytesIO()
    canvas = rl_canvas.Canvas(buffer, pagesize=pagesize, encrypt=encryption)
    canvas.setCreator("manicule fixtures")
    canvas.setTitle("fixture")
    for page in range(pages):
        canvas.setFont("Helvetica", FONT_SIZE)
        y = MARKER_AT[1]
        for line in text:
            canvas.drawString(MARKER_AT[0], y, line if pages == 1 else f"{line} p{page + 1}")
            y -= FONT_SIZE * 1.4
        canvas.showPage()
    canvas.save()
    return buffer.getvalue()


def _reshape(
    data: bytes,
    *,
    rotation: int | None = None,
    cropbox: tuple[float, float, float, float] | None = None,
    mediabox: tuple[float, float, float, float] | None = None,
) -> bytes:
    """Apply page geometry with pdfium rather than at authoring time.

    Setting ``/Rotate`` through the authoring library also swaps the MediaBox, which would
    hide the very thing these fixtures exist to expose: a portrait MediaBox displayed
    landscape. Applying it afterwards leaves the MediaBox alone.
    """
    document = pdfium.PdfDocument(data)
    try:
        page = document[0]
        if mediabox is not None:
            page.set_mediabox(*mediabox)
        if cropbox is not None:
            page.set_cropbox(*cropbox)
        if rotation is not None:
            page.set_rotation(rotation)
        page.gen_content()
        out = io.BytesIO()
        document.save(out)
        return out.getvalue()
    finally:
        document.close()


_ASTRAL_CMAP = b"""/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CMapName /Astral def
/CMapType 2 def
1 begincodespacerange
<00> <FF>
endcodespacerange
2 beginbfchar
<41> <D83DDE00>
<42> <0042>
endbfchar
endcmap
CMapName currentdict /CMap defineresource pop
end
end"""


def _astral_pdf() -> bytes:
    """A PDF whose ToUnicode CMap maps a byte to U+1F600.

    Hand-written because it is the cheapest way to get astral-plane text into a PDF without
    depending on a system font — and a fixture that depends on a system font would not
    produce the same corpus on two machines, which is the property the whole design protects.

    The page renders ``ABAB`` and extracts as ``😀B😀B``: four Python characters, six pdfium
    character indices, because pdfium counts UTF-16 code units and each emoji is a surrogate
    pair. A parser that passes a Python offset straight to pdfium asks about the wrong glyphs.
    """
    content = b"BT /F1 24 Tf 72 700 Td (ABAB) Tj ET"
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /ToUnicode 6 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Length %d >>\nstream\n" % len(_ASTRAL_CMAP) + _ASTRAL_CMAP + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.7\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    start_xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        start_xref,
    )
    return bytes(out)


def _no_text_layer() -> bytes:
    """A page carrying a drawn rectangle and no text at all.

    Stands in for a scanned page. Optical character recognition is out of scope, and this
    fixture is how "nothing to extract" stays distinguishable from "the parser broke".
    """
    buffer = io.BytesIO()
    canvas = rl_canvas.Canvas(buffer, pagesize=LETTER)
    canvas.rect(100, 100, 200, 200, stroke=1, fill=0)
    canvas.showPage()
    canvas.save()
    return buffer.getvalue()


def _multicolumn() -> bytes:
    """Two columns, so reading order is measured rather than assumed."""
    buffer = io.BytesIO()
    canvas = rl_canvas.Canvas(buffer, pagesize=LETTER)
    canvas.setFont("Helvetica", 10)
    for row in range(20):
        y = 720 - row * 14
        canvas.drawString(60, y, f"Left column line {row + 1} about deployment.")
        canvas.drawString(330, y, f"Right column line {row + 1} about rollback.")
    canvas.showPage()
    canvas.save()
    return buffer.getvalue()


def build(dest: Path) -> None:
    """Write the PDF fixtures into ``dest``."""
    typical = _page(
        [MARKER, "The deployment runs in three stages.", "", "Rollback restores the previous."],
        pages=3,
    )
    (dest / "typical.pdf").write_bytes(typical)

    marker_only = _page([MARKER])
    (dest / "upright.pdf").write_bytes(marker_only)
    for rotation in (90, 180, 270):
        (dest / f"rotated-{rotation}.pdf").write_bytes(_reshape(marker_only, rotation=rotation))
    (dest / "cropped.pdf").write_bytes(_reshape(marker_only, cropbox=(50, 50, 400, 750)))
    (dest / "cropped-rotated.pdf").write_bytes(
        _reshape(marker_only, cropbox=(50, 50, 400, 750), rotation=90)
    )
    (dest / "mediabox-offset.pdf").write_bytes(_reshape(marker_only, mediabox=(50, 50, 662, 842)))

    (dest / "multicolumn.pdf").write_bytes(_multicolumn())
    (dest / "astral.pdf").write_bytes(_astral_pdf())
    (dest / "no-text-layer.pdf").write_bytes(_no_text_layer())
    (dest / "zero-bytes.pdf").write_bytes(b"")
    (dest / "truncated.pdf").write_bytes(typical[: len(typical) // 3])

    # An owner password restricts printing and editing, not reading. Refusing these would
    # drop real content over a restriction that was never about reading, so the parser opens
    # them; a user password genuinely cannot be read and is declined.
    (dest / "owner-password.pdf").write_bytes(
        _page([MARKER], encryption=pdfencrypt.StandardEncryption("", "ownersecret"))
    )
    (dest / "user-password.pdf").write_bytes(
        _page([MARKER], encryption=pdfencrypt.StandardEncryption("usersecret", "ownersecret"))
    )

    # The one deliberate large fixture, exercising the many-page path.
    (dest / "many-pages-large.pdf").write_bytes(
        _page(["Paragraph one about tokens.", "", "Paragraph two about refresh."], pages=60)
    )


__all__ = ["FONT_SIZE", "LETTER", "MARKER", "MARKER_AT", "build"]
