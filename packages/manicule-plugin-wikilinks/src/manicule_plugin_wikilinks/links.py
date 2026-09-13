"""Finding ``[[wikilinks]]`` in text, normalizing them, and deciding what kind each one is.

Pure functions over strings, with no store and no pipeline in sight, because everything here is
a *rule* rather than a step: what counts as a link, what two spellings of a target have in
common, and where the line runs between a declared relationship and a passing mention. Keeping
them separable is what lets :mod:`manicule_plugin_wikilinks` digest this file and call the digest
the extractor's identity — a rule that changed and a corpus that reports itself current are the
failure the fingerprint exists to prevent, and it can only be prevented if the rules are in a
file somebody can point at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "LINK",
    "Found",
    "Shape",
    "links_in",
    "normalize",
]

LINK = re.compile(r"\[\[(?P<target>[^\[\]\n]+?)\]\]")
"""One wikilink. The target is everything between the brackets, before it is trimmed.

Non-greedy and newline-excluding, so ``[[a]] and [[b]]`` is two links rather than one spanning
both, and an unclosed ``[[`` at the end of a line cannot swallow the rest of a document.
"""

_ALIAS = re.compile(r"[|#].*$", re.DOTALL)
"""An alias or a heading fragment, which name a *display* rather than a target.

``[[project_backoff|the backoff policy]]`` and ``[[project_backoff#Retries]]`` both address the
same document, and an implementation that matched the whole of the brackets would resolve
neither. Stripped rather than parsed, because manicule's citations resolve to a heading through
an anchor the parser produced — reading a fragment here would be a second, weaker answer to a
question already answered properly.
"""

_SEPARATORS = re.compile(r"^[\s,;·•]*(?:and\b)?[\s,;·•]*$", re.IGNORECASE)
"""What may sit between two links on a line and still leave it a list of links rather than prose."""

_LIST_MARKER = re.compile(r"^\s{0,3}(?:[-*+]|\d{1,9}[.)])\s+")
"""A CommonMark list marker at the start of a line, in the indent CommonMark actually allows."""

_LABEL = re.compile("^\\s*(?:[A-Za-z][\\w-]*\\s*){0,2}[:\\-\u2013\u2014]?\\s*")
"""A short leading label or verb: ``relates_to``, ``Related:``, ``See also``, ``See also -``.

At most two words, because that is the difference between labeling a list and writing a sentence.
It is allowed to end without punctuation so that ``relates_to [[x]]`` — the bare-verb form the
corpus uses — is recognized, and bounded at two so that ``The client retries twice, see [[x]]``
is not.
"""


class Shape(StrEnum):
    """How a link was written, which is the whole of what decides its relation type."""

    DECLARED = "declared"
    """The link is the content of its line: a list item, or a label and links and nothing else."""

    PROSE = "prose"
    """The link sits inside a sentence that says something other than the link."""


@dataclass(frozen=True, slots=True)
class Found:
    """One link, as written and as it will be looked up."""

    target: str
    """The normalized target, which is what resolution compares."""

    written: str
    """Exactly what was between the brackets, for a diagnostic that has to quote the source."""

    shape: Shape


def normalize(target: str) -> str:
    """The form two spellings of one target have in common.

    **The corpus is inconsistent and both spellings are in use** —
    ``feedback-commit-everything.md`` and ``feedback_btctrader_hands_off.md`` are both real
    filenames, and links are written both ways — so literal matching would silently disconnect
    roughly half the graph. Every separator run folds to a single hyphen, case folds, and a
    ``.md`` suffix is dropped, because a link written as a filename and a link written as a slug
    name the same document.

    ``casefold`` rather than ``lower``: it is the operation defined for caseless matching, and
    the difference shows up on exactly the characters a lowercasing comparison gets wrong.
    """
    stripped = _ALIAS.sub("", target).strip()
    if stripped.casefold().endswith(".md"):
        stripped = stripped[: -len(".md")]
    return re.sub(r"[\s_\-]+", "-", stripped).strip("-").casefold()


def links_in(text: str) -> list[Found]:
    """Every link in ``text``, in order, each typed by the shape of the line it is on.

    Empty targets are dropped rather than returned unresolvable: ``[[]]`` and ``[[ | alias ]]``
    address nothing, and a target that normalizes to the empty string would match every other
    document that also normalizes to nothing.

    Duplicates within one text are **kept**, because two mentions of the same document on two
    lines can have two different shapes, and it is the store's idempotence rather than this
    function's de-duplication that stops the second one becoming a second row.
    """
    found: list[Found] = []
    for line in text.splitlines():
        shape = _shape_of(line)
        for match in LINK.finditer(line):
            written = match["target"]
            target = normalize(written)
            if target:
                found.append(Found(target=target, written=written, shape=shape))
    return found


def _shape_of(line: str) -> Shape:
    """Whether this line is a list of links or a sentence containing one.

    The two are not the same claim and §5.2 of the design says so: a declared link is the
    author asserting a relationship, while a mention is evidence that two documents are about
    related things. Collapsing them would make the distinction unrecoverable, and it is the
    distinction somebody asking "what is this connected to" wants first.

    A line is declared when, after an optional list marker and an optional short label, nothing
    remains but links and the punctuation that separates them. Everything else is prose —
    including a line whose label ran to three words, which is where this rule deliberately
    stops guessing.
    """
    if not LINK.search(line):
        return Shape.PROSE
    remainder = _LABEL.sub("", _LIST_MARKER.sub("", line), count=1)
    stripped = LINK.sub("", remainder)
    return Shape.DECLARED if _SEPARATORS.match(stripped) else Shape.PROSE
