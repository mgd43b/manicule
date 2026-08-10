"""Splitting prose into paragraphs and sentences, without a model.

Segmentation here is deliberately a rule rather than a model. A sentence-segmentation model
would put a second model's version into
:class:`~manicule.core.fingerprints.ChunkFingerprint`, so upgrading it would re-chunk and
re-embed every prose document in the corpus — a large, silent cost for a boundary decision
that a terminator and a capital letter get right.

The rule: a terminator, then whitespace, then something that starts a sentence, unless the
word before the terminator is a known abbreviation.
"""

from __future__ import annotations

import re

_PARAGRAPH = re.compile(r"\n[ \t]*\n")

_ABBREVIATIONS = frozenset(
    {
        "al",
        "approx",
        "cf",
        "dr",
        "e.g",
        "eg",
        "esp",
        "etc",
        "fig",
        "i.e",
        "ie",
        "inc",
        "jr",
        "ltd",
        "mr",
        "mrs",
        "ms",
        "no",
        "pp",
        "prof",
        "sr",
        "st",
        "vs",
        "vol",
    }
)
"""Words that end in a full stop without ending a sentence.

Kept short on purpose. Every entry is a word that would otherwise split a sentence in half,
and a chunk that begins mid-sentence is what the overlap window exists to prevent.
"""

_BOUNDARY = re.compile(r"([.!?][\"'”’)\]]?)(\s+)(?=[\"'“‘(\[]?[A-Z0-9])")  # noqa: RUF001 - typographic quotes are the characters real documents use
"""A terminator, then whitespace, then something that starts a sentence.

Written as two capture groups rather than a look-behind because the terminator and its
optional closing quote are not a fixed width, and a variable-width look-behind is not a
regular expression Python will compile. Group 1 ends the sentence; group 2 is the gap.
"""

_TRAILING_WORD = re.compile(r"([A-Za-z][A-Za-z.]*)[.!?][\"'”’)\]]?\s*$")  # noqa: RUF001 - as above


def paragraphs(text: str) -> list[str]:
    """Split on blank lines, keeping non-empty paragraphs in order."""
    return [part for part in (chunk.strip() for chunk in _PARAGRAPH.split(text)) if part]


def sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, keeping them in order and losing nothing.

    Joined back together with a single space, the result normalises to the input: this is a
    boundary finder, not a rewriter. Anything it cannot split confidently stays whole, which
    costs a little budget and never costs a word.
    """
    stripped = text.strip()
    if not stripped:
        return []
    pieces: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(stripped):
        candidate = stripped[start : match.end(1)]
        if _ends_with_abbreviation(candidate):
            continue
        pieces.append(candidate)
        start = match.end(2)
    tail = stripped[start:]
    if tail:
        pieces.append(tail)
    return pieces


def _ends_with_abbreviation(text: str) -> bool:
    match = _TRAILING_WORD.search(text)
    if match is None:
        return False
    word = match.group(1).rstrip(".").casefold()
    if word in _ABBREVIATIONS:
        return True
    # A single initial — "J. Smith" — is never a sentence end either.
    return len(word) == 1


__all__ = ["paragraphs", "sentences"]
