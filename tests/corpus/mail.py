"""Fixtures for the ``.eml`` parser.

Built with the standard library's own message writer rather than written out as literal text,
so the transfer encodings, boundaries and folded headers are the ones a real mail transfer
agent produces instead of the ones a fixture author remembered.

The quoted reply chain in the typical message is deliberate. Quoted text is kept — trimming it
is a retrieval optimisation whose cost is that the quoted passage is frequently the only
statement of the thing being replied to (``docs/parsing.md`` §10) — so it has to be in the
corpus for that to be tested rather than asserted.
"""

from __future__ import annotations

from email.message import EmailMessage
from email.policy import SMTP
from pathlib import Path

_SENT = "Tue, 03 Feb 2026 09:14:00 +0000"
_REPLIED = "Mon, 02 Feb 2026 17:02:00 +0000"

TYPICAL_BODY = """Thanks for looking at this so quickly.

I have taken the watermark change and split it out, so the retry work can land on its own
without waiting for the connector refactor.

On Monday you wrote:

> The thing I cannot tell from the diff is whether an interrupted sync resumes from the last
> page or from the start of the run. If it is the start of the run, the whole point of the
> watermark is lost on exactly the syncs that need it.

It resumes from the last page. The watermark is written after each page rather than at the
end, which is the change in the second commit.
"""

HTML_BODY = """<html>
  <head><title>Ignored, because the subject is the document's title</title></head>
  <body>
    <h1>Where the first quarter went</h1>
    <p>Ingest throughput rose by a fifth over the quarter, with no change to the chunk
       budget and therefore no re-embedding.</p>
    <h2>What moved</h2>
    <p>The reconciler now runs on its own schedule,<br/>which removed the long tail of deletions
       that used to arrive a day late.</p>
    <ul>
      <li>Watermarks are written per page</li>
      <li>Attachments are indexed as documents of their own</li>
    </ul>
  </body>
</html>
"""
"""An HTML-only body, whose ``<h1>`` deliberately does not repeat the subject.

Real HTML mail very often opens with a heading identical to its subject, and the round-trip
suite compares *text*: the subject is inside the header block, so an identical heading in the
body would make resolving the header return the heading block's text and fail assertion 3
against a parser that is behaving perfectly. The realistic collision belongs in a test of the
assertion, not in a fixture whose job is to exercise the pinned HTML conversion.

**The ``<br>`` is the reason this parser's version moves when the web parser's does.** A break
puts a newline inside a block, ``_html_to_text`` joins blocks and hands the result to
``lines_of``, and every ``LineAnchor`` after the break moves by a line. Keeping one in the
fixture means the round-trip suite resolves across it rather than only around it.
"""

ATTACHMENT_TEXT = """Rollout checklist

1. Confirm the chunk fingerprint matches the stored one.
2. Pre-seed the declared grammar set on every worker.
3. Re-run the reconciler before enabling the schedule.
"""

ATTACHMENT_JSON = '{"window": "2026-Q1", "documents": 4821, "failed": 3}\n'

BROKEN_CHARSET = (
    b"From: Ada Okoye <ada@example.invalid>\r\n"
    b"To: Bo Lindqvist <bo@example.invalid>\r\n"
    b"Date: Tue, 03 Feb 2026 09:14:00 +0000\r\n"
    b"Subject: A body that is not the character set it claims\r\n"
    b'Content-Type: text/plain; charset="utf-8"\r\n'
    b"\r\n"
    b"Caf\xe9 written in Latin-1 under a header that says UTF-8.\r\n"
)
"""A message whose body declares one character set and is encoded in another.

What a mail client that mislabels its output produces. Decoding it with replacement characters
would put text into the index that no message contains, under a citation saying it is a
quotation, so the parser fails rather than guessing a second encoding.
"""


DIGEST = (
    b"From: Platform list <list@example.invalid>\r\n"
    b"To: Subscribers <subscribers@example.invalid>\r\n"
    b"Subject: Weekly digest\r\n"
    b"Date: Tue, 03 Feb 2026 09:14:00 +0000\r\n"
    b"MIME-Version: 1.0\r\n"
    b'Content-Type: multipart/digest; boundary="fixture-digest"\r\n'
    b"\r\n"
    b"--fixture-digest\r\n"
    b"\r\n"
    b"From: Ada Okoye <ada@example.invalid>\r\n"
    b"Subject: Retention window\r\n"
    b"Content-Type: text/plain\r\n"
    b"\r\n"
    b"The retention window moves to ninety days at the end of the quarter.\r\n"
    b"--fixture-digest\r\n"
    b"\r\n"
    b"From: Bo Lindqvist <bo@example.invalid>\r\n"
    b"Subject: Reconciler schedule\r\n"
    b"Content-Type: text/plain\r\n"
    b"\r\n"
    b"The reconciler now runs on its own schedule rather than after ingest.\r\n"
    b"--fixture-digest--\r\n"
)
"""A ``multipart/digest``, whose enclosed messages carry no filename and no disposition.

Written out by hand rather than built with :mod:`email`, because what makes it a useful
fixture is precisely the shape a builder normalises away: each part is a bare
``message/rfc822`` with nothing marking it as an attachment. A parser that looks only for a
disposition or a filename finds nothing to expand and nothing to fail, and the digest indexes
as a header block with every enclosed message silently absent.
"""


def build(dest: Path) -> None:
    """Write this format's fixtures into ``dest``."""
    (dest / "typical.eml").write_bytes(_typical())
    (dest / "multipart.eml").write_bytes(_multipart())
    (dest / "html-only.eml").write_bytes(_html_only())
    (dest / "headers-only.eml").write_bytes(_headers_only())
    (dest / "duplicate-attachments.eml").write_bytes(_duplicate_attachments())
    (dest / "not-a-message.eml").write_bytes(
        b"This file has no header fields at all.\nIt is a note somebody renamed.\n"
    )
    (dest / "digest.eml").write_bytes(DIGEST)
    (dest / "empty.eml").write_bytes(b"")
    (dest / "broken-charset.eml").write_bytes(BROKEN_CHARSET)


def _pinned(message: EmailMessage) -> bytes:
    """Serialise with fixed MIME boundaries, so two runs produce identical bytes.

    :mod:`email` mints a random boundary for every multipart part, which would give the same
    fixture a different ``content_hash`` on every build — and a corpus that churns on every
    run cannot be used to assert that a parser does not.
    """
    for index, part in enumerate(message.walk()):
        if part.get_content_maintype() == "multipart":
            part.set_boundary(f"fixture-boundary-{index:02d}")
    return message.as_bytes(policy=SMTP)


def _base(subject: str) -> EmailMessage:
    message = EmailMessage()
    message["From"] = "Ada Okoye <ada@example.invalid>"
    message["To"] = "Bo Lindqvist <bo@example.invalid>"
    message["Cc"] = "platform-team@example.invalid"
    message["Date"] = _SENT
    message["Subject"] = subject
    message["Message-ID"] = "<fixture-0001@example.invalid>"
    return message


def _typical() -> bytes:
    message = _base("Re: watermark on interrupted syncs")
    message["In-Reply-To"] = "<fixture-0000@example.invalid>"
    message["References"] = f"<fixture-0000@example.invalid> ({_REPLIED})"
    message.set_content(TYPICAL_BODY)
    return _pinned(message)


def _multipart() -> bytes:
    """A plain body, an HTML alternative, and two attachments of different kinds.

    The alternative matters: the canonical body is the first ``text/plain`` part in depth-first
    order, so a parser that preferred the richer part would anchor into converted HTML while a
    perfectly good plain body sat above it.
    """
    message = _base("Q1 rollout: checklist and counts attached")
    message.set_content("The checklist and the counts are attached. Nothing else has changed.\n")
    message.add_alternative(
        "<html><body><p>The checklist and the counts are attached.</p></body></html>",
        subtype="html",
    )
    message.add_attachment(
        ATTACHMENT_TEXT.encode("utf-8"),
        maintype="text",
        subtype="plain",
        filename="checklist.txt",
    )
    message.add_attachment(
        ATTACHMENT_JSON.encode("utf-8"),
        maintype="application",
        subtype="json",
        filename="counts.json",
    )
    return _pinned(message)


def _html_only() -> bytes:
    message = _base("Quarterly platform review")
    message.set_content(HTML_BODY, subtype="html")
    return _pinned(message)


def _headers_only() -> bytes:
    """Headers and nothing else, which is a real shape: a calendar decline, a bounce stub."""
    message = _base("No body, only headers")
    return _pinned(message)


def _duplicate_attachments() -> bytes:
    """Two attachments with one filename, which is what forces a second address to exist."""
    message = _base("Two files, one name")
    message.set_content("Both attachments are called report.txt.\n")
    message.add_attachment(
        b"The first report, from the ingest run.\n",
        maintype="text",
        subtype="plain",
        filename="report.txt",
    )
    message.add_attachment(
        b"The second report, from the retrieval evaluation.\n",
        maintype="text",
        subtype="plain",
        filename="report.txt",
    )
    return _pinned(message)
