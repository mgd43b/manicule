"""Fixtures for the draw.io parser.

Generated rather than exported from draw.io, so the two encodings a ``<diagram>`` body can use
are both present and both built by the same rules the application uses — ``encodeURIComponent``,
then raw deflate, then base64 — instead of whichever one the author's copy of the editor happened
to write on the day.

The hostile pair is the point of the directory. An attachment is a file from the corpus, so its
compressed payload is untrusted twice over: ``expansion-bomb.drawio`` is a payload that is small
until it is read, and ``doctype.drawio`` is the entity expansion that has no ceiling to stop it.
Both must be refused under the shipped defaults rather than under a test-only configuration, or
the refusal being checked is not the one an operator has.
"""

from __future__ import annotations

import base64
import struct
import zlib
from pathlib import Path
from urllib.parse import quote

TYPICAL_MODEL = """<mxGraphModel dx="1102" dy="798" grid="1" page="1">
  <root>
    <mxCell id="0" />
    <mxCell id="1" parent="0" />
    <mxCell id="auth" value="Auth Service" style="rounded=0;" vertex="1" parent="1">
      <mxGeometry x="80" y="120" width="160" height="60" as="geometry" />
    </mxCell>
    <mxCell id="tokens" value="Token Store" style="rounded=0;" vertex="1" parent="1">
      <mxGeometry x="360" y="120" width="160" height="60" as="geometry" />
    </mxCell>
    <mxCell id="audit" value="Audit Log" style="rounded=0;" vertex="1" parent="1">
      <mxGeometry x="360" y="260" width="160" height="60" as="geometry" />
    </mxCell>
    <mxCell id="e1" value="validates against" style="" edge="1" parent="1"
            source="auth" target="tokens">
      <mxGeometry relative="1" as="geometry" />
    </mxCell>
  </root>
</mxGraphModel>"""
"""One resolved relationship and one node connected to nothing.

Both halves matter: an unconnected node is the case a reading reports as an inventory line
rather than dropping, and a diagram of three boxes and one arrow is what a real architecture
page holds."""

SECOND_MODEL = """<mxGraphModel>
  <root>
    <mxCell id="0" />
    <mxCell id="1" parent="0" />
    <mxCell id="ingest" value="Ingest" vertex="1" parent="1" />
    <mxCell id="index" value="Index" vertex="1" parent="1" />
    <mxCell id="e2" value="" edge="1" parent="1" source="ingest" target="index" />
  </root>
</mxGraphModel>"""

RICH_MODEL = """<mxGraphModel>
  <root>
    <mxCell id="0" />
    <mxCell id="1" parent="0" />
    <object label="&lt;b&gt;Gateway&lt;/b&gt;&lt;br&gt;edge" owner="platform" id="gw">
      <mxCell style="rounded=1;" vertex="1" parent="1">
        <mxGeometry x="40" y="40" width="120" height="60" as="geometry" />
      </mxCell>
    </object>
    <mxCell id="core" value="Core &amp;amp; friends" vertex="1" parent="1" />
    <object label="routes to" id="gwe">
      <mxCell style="" edge="1" parent="1" source="gw" target="core" />
    </object>
  </root>
</mxGraphModel>"""
"""Labels that are rich text, and cells wrapped in ``<object>`` so the wrapper owns the label.

A reader that only looked at ``mxCell`` would see these shapes as unlabeled, and one that did
not strip the markup would embed ``<b>`` as a word."""


def build(dest: Path) -> None:
    (dest / "typical.drawio").write_bytes(
        _mxfile(
            (("architecture", "Architecture", _compressed(TYPICAL_MODEL)),),
        )
    )
    (dest / "multi-page.drawio").write_bytes(
        _mxfile(
            (
                ("architecture", "Architecture", _compressed(TYPICAL_MODEL)),
                ("pipeline", "Pipeline", _compressed(SECOND_MODEL)),
            )
        )
    )
    # Plain rather than compressed: draw.io writes this form when compression is turned off,
    # and a reader that only handled the base64 spelling would refuse half the exports there are.
    (dest / "uncompressed.drawio").write_bytes(_mxfile((("rich", "Rich labels", RICH_MODEL),)))
    (dest / "repeated-names.drawio").write_bytes(
        _mxfile(
            (
                ("first", "Overview", _compressed(SECOND_MODEL)),
                ("second", "Overview", _compressed(TYPICAL_MODEL)),
            )
        )
    )
    (dest / "empty.drawio").write_bytes(_mxfile((("blank", "Blank", ""),)))
    (dest / "typical.drawio.png").write_bytes(
        _png(_mxfile((("architecture", "Architecture", _compressed(TYPICAL_MODEL)),)))
    )
    (dest / "expansion-bomb.drawio").write_bytes(
        _mxfile((("bomb", "Bomb", _encoded(b"0" * (12 * 1024 * 1024))),))
    )
    (dest / "doctype.drawio").write_bytes(
        b'<?xml version="1.0"?>\n'
        b'<!DOCTYPE mxfile [<!ENTITY a "aaaaaaaaaa">'
        b'<!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">'
        b'<!ENTITY c "&b;&b;&b;&b;&b;&b;&b;&b;&b;&b;">]>\n'
        b'<mxfile><diagram id="x" name="&c;"></diagram></mxfile>\n'
    )


def _mxfile(diagrams: tuple[tuple[str, str, str], ...]) -> bytes:
    """An ``<mxfile>`` wrapping one ``<diagram>`` per tuple of id, name and body."""
    pages = "".join(
        f'<diagram id="{identifier}" name="{name}">{body}</diagram>'
        for identifier, name, body in diagrams
    )
    opening = '<mxfile host="app.diagrams.net" agent="fixture" version="24.7.5">'
    return f"{opening}{pages}</mxfile>".encode()


def _compressed(model: str) -> str:
    """A ``<diagram>`` body the way draw.io writes one: URL-encode, raw-deflate, base64."""
    return _encoded(quote(model).encode())


def _encoded(data: bytes) -> str:
    engine = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
    return base64.b64encode(engine.compress(data) + engine.flush()).decode()


def _png(source: bytes) -> bytes:
    """A real 1x1 PNG carrying ``source`` in the ``zTXt`` chunk draw.io writes it to.

    Valid rather than a stub with a signature on the front, because the point of the format is
    that it renders anywhere and still round-trips into the editor. A fixture that only the
    chunk walker could open would not be testing that.
    """
    payload = b"mxfile\x00\x00" + zlib.compress(quote(source.decode()).encode(), 9)
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)),
            _chunk(b"zTXt", payload),
            _chunk(b"IDAT", zlib.compress(b"\x00\x00", 9)),
            _chunk(b"IEND", b""),
        )
    )


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
