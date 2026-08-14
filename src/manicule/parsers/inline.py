"""Inline line breaks, and the two joins a parser can honestly give them.

Every inline element is understood by flattening its text into the run around it, and one of
them has no text to flatten. A ``<br>`` contributes no characters, so a flatten that asks the
tree for characters gets none and the element disappears: ``a<br/>b`` read ``ab``, two source
fragments glued into a word neither page contains. That was true of the HTML parser and of the
Confluence storage parser alike, which is why the rule lives here rather than twice.

**A break is an object in the stream, never a character in the string.** The obvious
implementation puts a sentinel character into the text and swaps it for a newline once
whitespace has been collapsed, and it has to answer a question it cannot answer cheaply: which
character can no document produce? ``NUL`` is the usual candidate, and whether it survives is
decided by an HTML tokenizer's error handling — an argument that would have to be re-made every
time the engine changed, about a failure that would surface as a control character inside a
citation. :data:`LINE_BREAK` answers it by construction instead. A :class:`str` is never that
object however the page was written, so no source text can forge one; and none can escape,
because the three joins below are its only consumers and every one of them returns a string.

**Two joins, because a newline does not mean the same thing everywhere.** In prose it is a line
break. In the rendering of a table, a list or a task list it is the *record separator* — one row
per line, one item per line — and in a heading it would put a newline inside a breadcrumb. So a
break becomes a newline only where the text it lands in is prose (:func:`collapse_lines`), and a
space everywhere a newline already carries structure (:func:`collapse_run`). That division is
what keeps a line break from becoming a table row, a list item, a task or a heading path
element, which would be a change to the document's shape rather than to its words.

**The text model, stated once** (``docs/parsing.md`` §4.5):

============================  ===================================================
one break                     one ``\\n`` — a line break inside the same paragraph
two or more adjacent breaks   one blank line — the strongest thing the model says
a break at either end         nothing; there is no text for it to separate
a break in a one-line run     a space
============================  ===================================================

Two adjacent breaks are a blank line because that is what they draw: a reader sees an empty
line and so does :func:`~manicule.chunking.sentences.paragraphs`. More than two draw nothing
further that anything reads — the splitter discards the empty paragraphs between them — so they
are capped rather than carried, which also stops the spacer markup old pages are full of from
spending a chunk's budget on whitespace.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, override

__all__ = [
    "DEFINITION_BODY_MARKER",
    "LINE_BREAK",
    "InlinePart",
    "LineBreak",
    "collapse",
    "collapse_lines",
    "collapse_run",
    "collapse_segments",
    "item_prefix",
]


@dataclass(frozen=True, slots=True)
class LineBreak:
    """The type of :data:`LINE_BREAK`, which is the only value of it anything constructs."""

    @override
    def __repr__(self) -> str:
        return "LINE_BREAK"


LINE_BREAK: Final = LineBreak()
"""What an inline break contributes to the run of text around it."""

DEFINITION_BODY_MARKER: Final = ": "
"""What a ``<dd>`` is written with, and it is what makes a definition list readable.

**A definition list carries a relationship, and rendering every part with the same bullet threw
it away.** ``<dt>NOW</dt><dd>Network Operations Workspace</dd>`` came out as two indistinguishable
``- `` lines, so a reader could not tell which was the term and neither could
:func:`~manicule.ingest.glossary.detect_in_chunk` — measured at 0 of 4 definitions found on a page
whose glossary was written this way. The relationship is the *only* thing a ``<dl>`` adds over a
``<ul>``, and it is what did not survive.

``TERM`` on its own line with ``: definition`` beneath it is the convention Markdown definition
lists already use, so it is what a reader of the chunk sees and what a citation quotes. It is also,
not by coincidence, exactly the shape :data:`~manicule.ingest.glossary._DEFINITION_MARKER_RE` has
always read: that form was implemented and unit-tested and could never fire, because no parser
produced the only rendering it accepts.
"""


def item_prefix(tag: str, marker: str) -> str:
    """What precedes a list item's own text: a bullet, a number, or a definition-list marker.

    **Here rather than in either parser, because two copies of it would have to agree.** The
    storage and HTML parsers keep their own list *flattening* on purpose — one of them must not
    read an ``ac:parameter`` out of a nested macro — but the prefix is the same question with the
    same answer for both, and storage format is XHTML. A ``<dl>`` that rendered one way through a
    Confluence connector and another through a web crawl would reach the corpus as two glossaries
    behaving differently depending on how the page was fetched.

    A ``<dt>`` gets no marker: it is the term, and a bullet in front of it would be structure
    competing with the ``: `` that says what the next line is. A second ``<dd>`` under one ``<dt>``
    is rendered like the first, and detection reads only the first because the line above the
    second is itself a ``: `` line, which is not a term — the conservative outcome, and choosing
    between two definitions is the judgment that feature is forbidden to make.
    """
    if tag == "dt":
        return ""
    if tag == "dd":
        return DEFINITION_BODY_MARKER
    return f"{marker} "


type InlinePart = str | LineBreak
"""One piece of a run: source text, or a break the source drew between two pieces of it."""

_MAX_BLANK_LINES: Final = 1
"""How many empty lines a run of breaks may draw. See the module docstring for why it is one."""


def collapse(text: str) -> str:
    """Whitespace as a reader sees it: runs become single spaces, ends are trimmed."""
    return " ".join(text.split())


def collapse_run(parts: Iterable[InlinePart]) -> str:
    """``parts`` as a single line, a break reading as the space it separates words with.

    For every rendering whose own newline is structural — a table row, a list item, a task, a
    heading path element. A break that produced a newline there would not read as a line break
    at all: it would read as another row, another item, another heading.
    """
    return " ".join(line for line in _lines(parts) if line)


def collapse_lines(parts: Iterable[InlinePart]) -> str:
    """``parts`` as the lines its source drew, under the rule in the module docstring.

    For prose, which is the only text in which a newline means a line break and nothing else.
    """
    kept: list[str] = []
    blank = 0
    for line in _lines(parts):
        if line:
            blank = 0
            kept.append(line)
            continue
        blank += 1
        if kept and blank <= _MAX_BLANK_LINES:
            kept.append(line)
    while kept and not kept[-1]:
        kept.pop()
    return "\n".join(kept)


def collapse_segments(parts: Iterable[InlinePart]) -> tuple[InlinePart, ...]:
    """``parts`` reduced to one collapsed line per line, with the breaks between them kept.

    What an inline element hands back to the run that *contains* it — a link body, a nested
    span — rather than to a block. Handing back raw text instead would let whitespace at the
    element's edges reach the outer collapse, which would change ``a<a> b </a>c`` from ``abc``
    to ``a b c``: a real improvement, an unrelated one, and one that would widen "documents
    whose text changes" from "those containing a break" to something nobody can enumerate.

    Idempotent, because a caller cannot always tell whether the parts it was handed have been
    through here already.
    """
    segments: list[InlinePart] = []
    for index, line in enumerate(_lines(parts)):
        if index:
            segments.append(LINE_BREAK)
        segments.append(line)
    return tuple(segments)


def _lines(parts: Iterable[InlinePart]) -> list[str]:
    """The collapsed lines ``parts`` draws, in order, empty ones included.

    Empty lines are kept for the callers to rule on: :func:`collapse_run` drops them and
    :func:`collapse_lines` counts them, and a shared step that discarded them would have
    decided the question here for both.
    """
    lines: list[list[str]] = [[]]
    for part in parts:
        if isinstance(part, str):
            lines[-1].append(part)
        else:
            lines.append([])
    return [collapse("".join(line)) for line in lines]
