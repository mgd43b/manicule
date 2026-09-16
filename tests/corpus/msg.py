"""Fixtures for the Outlook ``.msg`` parser.

Generated rather than exported from Outlook, and generated rather than committed, because the
four shapes that matter here are shapes a real export gives you only by accident. The one that
matters most is the message that never traversed a transport — a draft, or anything in Sent
Items — where there is no header block to read and the addresses have to be assembled from the
sender properties and the recipient table. ``docs/parsing.md`` §10 makes that a required fixture
and §3.5 makes it one of the four kinds every parser owes.

The compound file itself comes from :mod:`tests.corpus._cfbf`, which writes the subset a ``.msg``
uses. ``olefile`` reads compound files and does not write them, and the round-trip in
``tests/parsers/test_msg.py`` is what says the writer produces something a reader accepts rather
than something that merely looks like it.
"""

from __future__ import annotations

import struct
from pathlib import Path

from tests.corpus._cfbf import Storage
from tests.corpus._cfbf import build as compound_file

TRANSPORT_HEADERS = """\
From: "Ana Ruiz" <ana@example.test>
To: "Platform" <platform@example.test>
Cc: "Ben Ito" <ben@example.test>
Subject: Where the retry budget went
Date: Tue, 03 Feb 2026 09:14:00 +0000
Message-ID: <20260203091400.1@example.test>
Content-Type: text/plain; charset="utf-8"
Content-Transfer-Encoding: 7bit
MIME-Version: 1.0
"""
"""A real transport block, including the MIME headers the shim must **not** carry.

Their presence is the point: this message's body is rebuilt out of MAPI properties, so a
``Content-Transfer-Encoding`` copied across from here would be a header asserting an encoding
that no longer describes the bytes it labels.
"""

BODY = """\
The budget is netted down once for the fetch and once more for the parse, so a document sized
to it is refused twice. The fix is to charge it once and compare against what is left.

Numbers are in the attachment, measured at the boundary rather than on short inputs.
"""
"""Two paragraphs, and deliberately no sign-off.

A closing "Ana" would be a body line contained inside the header block's "Ana Ruiz", and the
round-trip suite compares *text*: resolving the header anchor would return a string containing
the whole of the sign-off block, which is indistinguishable from an off-by-one line index. That
collision is real and belongs in a test of the assertion rather than in a fixture whose job is
to exercise the MAPI reader — the same reasoning ``tests/corpus/mail.py`` records for an HTML
heading identical to its subject.
"""

HTML_BODY = """\
<html><body>
  <h1>What moved in the first quarter</h1>
  <p>The budget is netted down twice, so a document sized to it is refused twice.</p>
</body></html>
"""
"""An HTML-only body whose heading deliberately does not repeat the subject.

Real HTML mail very often opens with a heading identical to its subject, and the round-trip
suite compares text: the subject is inside the header block, so an identical heading in the body
would make resolving the header return the heading block's text and fail the discrimination
assertion against a parser behaving perfectly. ``tests/corpus/mail.py`` records the same choice
for the same reason.
"""


def build(dest: Path) -> None:
    (dest / "typical.msg").write_bytes(
        compound_file(
            {
                **_stream("007D", "001F", TRANSPORT_HEADERS),
                **_stream("0037", "001F", "Where the retry budget went"),
                **_stream("1000", "001F", BODY),
            },
            [
                Storage(
                    "__attach_version1.0_#00000000",
                    {
                        **_stream("3707", "001F", "notes.txt"),
                        **_stream("3701", "0102", b"Measured over the whole quarter.\n"),
                    },
                )
            ],
        )
    )
    # The required fixture: no transport block at all, which is every draft and everything in
    # Sent Items. Ben is a `Cc`, and his recipient type lives in the properties stream rather
    # than beside the strings — a reader looking for it as a `__substg1.0_` stream finds nothing
    # and silently promotes him to `To`.
    (dest / "synthesized-headers.msg").write_bytes(
        compound_file(
            {
                **_stream("0037", "001F", "Draft: the retry budget"),
                **_stream("1000", "001F", BODY),
                **_stream("0C1F", "001F", "ana@example.test"),
                **_stream("0C1A", "001F", "Ana Ruiz"),
            },
            [
                Storage(
                    "__recip_version1.0_#00000000",
                    {
                        **_stream("3003", "001F", "platform@example.test"),
                        **_stream("3001", "001F", "Platform"),
                        "__properties_version1.0": _properties({0x0C150003: 1}),
                    },
                ),
                Storage(
                    "__recip_version1.0_#00000001",
                    {
                        **_stream("3003", "001F", "ben@example.test"),
                        **_stream("3001", "001F", "Ben Ito"),
                        "__properties_version1.0": _properties({0x0C150003: 2}),
                    },
                ),
            ],
        )
    )
    # `PidTagHtml` is `PT_BINARY`, so the stream is `…10130102`. A reader that assumed
    # `…1013001F` by analogy with the plain body finds nothing and reports no HTML part.
    (dest / "html-only.msg").write_bytes(
        compound_file(
            {
                **_stream("007D", "001F", TRANSPORT_HEADERS),
                **_stream("0037", "001F", "Quarterly platform review"),
                **_stream("1013", "0102", HTML_BODY.encode()),
            },
            [],
        )
    )
    # Every string property in its 8-bit spelling, which is what a store without
    # `STORE_UNICODE_OK` writes.
    (dest / "ansi.msg").write_bytes(
        compound_file(
            {
                **_stream("007D", "001E", TRANSPORT_HEADERS),
                **_stream("0037", "001E", "Where the retry budget went"),
                **_stream("1000", "001E", BODY),
            },
            [],
        )
    )
    # An attachment with no extension and a MAPI MIME tag that names what it is. Without
    # reading `370E` the reconstituted part is `application/octet-stream`, the mail parser's
    # filename fallback has nothing to work with, and a perfectly ordinary CSV is routed
    # nowhere.
    (dest / "mime-tagged.msg").write_bytes(
        compound_file(
            {
                **_stream("007D", "001F", TRANSPORT_HEADERS),
                **_stream("0037", "001F", "Quarterly numbers"),
                **_stream("1000", "001F", BODY),
            },
            [
                Storage(
                    "__attach_version1.0_#00000000",
                    {
                        **_stream("3707", "001F", "quarterly-numbers"),
                        **_stream("370E", "001F", "text/csv"),
                        **_stream("3701", "0102", b"quarter,throughput\n2026Q1,1.2\n"),
                    },
                ),
                # A tag that is not one plain `type/subtype`. It is attacker-controlled and ends
                # up in a header, so the reader has to refuse it back to the default rather than
                # pass it through — and a fixture that only ever carries a well-formed one
                # exercises none of that.
                Storage(
                    "__attach_version1.0_#00000001",
                    {
                        **_stream("3707", "001F", "notes"),
                        **_stream("370E", "001F", "text/plain\r\nX-Injected: yes"),
                        **_stream("3701", "0102", b"measured at the boundary\n"),
                    },
                ),
            ],
        )
    )
    # Degenerate: named `.msg`, and not a compound file at all. It must be declined rather than
    # read as an empty message, because a message that indexes as empty looks like a success.
    (dest / "not-a-compound-file.msg").write_bytes(b"Subject: this is an .eml in disguise\n\nhi\n")


def _stream(tag: str, kind: str, value: str | bytes) -> dict[str, bytes]:
    if isinstance(value, bytes):
        payload = value
    elif kind == "001F":
        payload = value.encode("utf-16-le")
    else:
        payload = value.encode("cp1252")
    return {f"__substg1.0_{tag}{kind}": payload}


def _properties(values: dict[int, int]) -> bytes:
    """A storage's properties stream: eight reserved bytes, then one entry per property.

    Four bytes of tag, four of flags, eight of value — the fixed-width half of MAPI, which is
    where a ``PT_LONG`` like ``PidTagRecipientType`` actually lives.
    """
    entries = b"".join(
        struct.pack("<IIQ", tag, 0x00000006, value) for tag, value in sorted(values.items())
    )
    return b"\x00" * 8 + entries
