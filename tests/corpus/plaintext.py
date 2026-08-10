"""Fixtures for the plain-text parser.

Every paragraph is deliberately unlike every other paragraph. The round-trip suite asserts
that resolving one anchor does not return another block's text (``docs/parsing.md`` §3.3,
assertion 3), so a fixture with two identical paragraphs would fail a parser that is behaving
perfectly — and the usual repair, weakening the assertion, is how it stops catching the
off-by-one it exists for.
"""

from __future__ import annotations

from pathlib import Path

TYPICAL = """Release notes for the ingest service, week of 3 February.

The connector now records a watermark after every successful page rather than at the end of
a run, so an interrupted sync resumes from where it stopped instead of re-fetching the
whole space.

Two defects are fixed. Attachments larger than the configured limit were skipped without
being recorded, and a page whose title contained a slash produced an unreadable citation.

Nothing in this release changes chunk boundaries, so no re-index is required.
"""

HARD_WRAPPED = "\n".join(
    [
        "A single paragraph hard-wrapped across many short lines, which is what a text file",
        "written in an editor with a fixed margin actually looks like, and which is the case",
        "where a parser that split on blank lines alone would emit one enormous block.",
        *(f"Line {number:03d} of the wrapped run, carrying its own distinct words." for number in range(1, 61)),
        "The run ends here, still with no blank line anywhere inside it.",
    ]
)

DEGENERATE_NO_NEWLINE = (
    "This file ends without a trailing newline.\n\nThe final paragraph is the last line, and "
    "there is no empty element after it."
)

WHITESPACE_ONLY = "   \n\t\n \n\n"

MALFORMED_UTF8 = b"The next byte is not valid UTF-8: \xff\xfe and the sentence continues.\n"

NOT_TEXT = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) + bytes(range(256)) * 4
"""A PNG signature followed by every byte value, NULs included.

This is the document that makes the global fallback tail safe. Handed to the plaintext
parser, it must be declined rather than decoded into a page of replacement characters
(``docs/parsing.md`` §6.2).
"""


def build(dest: Path) -> None:
    """Write this format's fixtures into ``dest``."""
    (dest / "typical.txt").write_text(TYPICAL, encoding="utf-8")
    (dest / "hard-wrapped.txt").write_text(HARD_WRAPPED + "\n", encoding="utf-8")
    (dest / "no-trailing-newline.txt").write_text(DEGENERATE_NO_NEWLINE, encoding="utf-8")
    (dest / "whitespace-only.txt").write_text(WHITESPACE_ONLY, encoding="utf-8")
    (dest / "empty.txt").write_bytes(b"")
    (dest / "malformed-utf8.txt").write_bytes(MALFORMED_UTF8)
    (dest / "not-text.bin").write_bytes(NOT_TEXT)
    (dest / "plaintext-large.txt").write_text(_large(), encoding="utf-8")


def _large() -> str:
    """A generated document past the fixture size cap, to exercise the streaming path.

    Named ``*-large.*`` so the corpus size check can see that the size is deliberate. Every
    paragraph is distinct, for the reason in the module docstring.
    """
    paragraphs = [
        f"Entry {number:04d}. "
        f"The {_WORDS[number % len(_WORDS)]} subsystem recorded {number * 7} events in the "
        f"window beginning at offset {number * 13}, of which {number % 11} were retried and "
        f"{number % 5} were abandoned. No operator action was required."
        for number in range(1, 1400)
    ]
    return "\n\n".join(paragraphs) + "\n"


_WORDS = (
    "ingest",
    "retrieval",
    "embedding",
    "storage",
    "scheduler",
    "watermark",
    "reconciler",
    "fingerprint",
)
