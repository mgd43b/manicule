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
    <h1>Quarterly platform review</h1>
    <p>Ingest throughput rose by a fifth over the quarter, with no change to the chunk
       budget and therefore no re-embedding.</p>
    <h2>What moved</h2>
    <p>The reconciler now runs on its own schedule, which removed the long tail of deletions
       that used to arrive a day late.</p>
    <ul>
      <li>Watermarks are written per page</li>
      <li>Attachments are indexed as documents of their own</li>
    </ul>
  </body>
</html>
"""

ATTACHMENT_TEXT = """Rollout checklist

1. Confirm the chunk fingerprint matches the stored one.
2. Pre-seed the declared grammar set on every worker.
3. Re-run the reconciler before enabling the schedule.
"""

ATTACHMENT_JSON = '{"window": "2026-Q1", "documents": 4821, "failed": 3}\n'


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
    return message.as_bytes(policy=SMTP)


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
    return message.as_bytes(policy=SMTP)


def _html_only() -> bytes:
    message = _base("Quarterly platform review")
    message.set_content(HTML_BODY, subtype="html")
    return message.as_bytes(policy=SMTP)


def _headers_only() -> bytes:
    """Headers and nothing else, which is a real shape: a calendar decline, a bounce stub."""
    message = _base("No body, only headers")
    return message.as_bytes(policy=SMTP)


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
    return message.as_bytes(policy=SMTP)
