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
        *(
            f"Line {number:03d} of the wrapped run, carrying its own distinct words."
            for number in range(1, 61)
        ),
        "The run ends here, still with no blank line anywhere inside it.",
    ]
)

CHANGELOG = "\n\n".join(
    [
        "Changelog, newest first. Every entry is one paragraph.",
        "0.9.4 — the reconciler runs on its own schedule instead of at the end of a sync.",
        "0.9.3 — attachments over the size limit are recorded with a reason rather than skipped.",
        "0.9.2 — citations into archives use the container separator, so they survive a paste.",
        "0.9.1 — the watermark is written per page, so an interrupted run resumes where it was.",
        "0.9.0 — first release with the chunk fingerprint refusal at startup.",
        "0.8.7 — a page title containing a slash no longer produces an unreadable address.",
        "0.8.6 — the embedder's own tokenizer counts the budget, replacing the estimator.",
        "0.8.5 — empty parses are reported as no extractable text rather than as failures.",
        "0.8.4 — the fallback chain is keyed by media type and records what it attempted.",
        "0.8.3 — heading fragments come from the source where the source publishes one.",
        "0.8.2 — rectangles are stored per line instead of merged into one envelope.",
        "0.8.1 — page indices are converted to one-based at the parser boundary, once.",
        "0.8.0 — first release that stores the source bytes, so re-parsing never re-fetches.",
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
    (dest / "changelog.txt").write_text(CHANGELOG + "\n", encoding="utf-8")
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
