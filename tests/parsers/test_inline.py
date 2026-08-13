"""The inline-break rule both HTML parsers obey, checked where it is stated.

Every case here is a sentence of ``docs/parsing.md`` §4.5.1 run rather than read. The rule is
small and every part of it is a decision somebody could reasonably have made differently, so
each one is asserted separately: what one break contributes, what two contribute, what a run
longer than two is capped to, what a break at a container's edge contributes, and what happens
where a newline already means something other than a line break.

:func:`test_no_source_string_can_forge_a_break` is the one that is about the *implementation*
rather than the rule. A sentinel character would have to be a character no document can
produce, and the case below is what says this design never has to answer that question.
"""

from __future__ import annotations

import pytest

from manicule.chunking.sentences import paragraphs
from manicule.parsers.inline import (
    LINE_BREAK,
    InlinePart,
    collapse,
    collapse_lines,
    collapse_run,
    collapse_segments,
)


def test_one_break_is_one_newline() -> None:
    assert collapse_lines(["alpha", LINE_BREAK, "beta"]) == "alpha\nbeta"


def test_one_break_is_not_a_paragraph_boundary() -> None:
    """The distinction §4.5 exists for, asserted on the splitter rather than on the string.

    A substring assertion passes on ``\\n\\n`` too. What must be true is that the splitter the
    chunker uses agrees the two fragments are one paragraph.
    """
    assert paragraphs(collapse_lines(["alpha", LINE_BREAK, "beta"])) == ["alpha\nbeta"]


def test_two_adjacent_breaks_are_one_blank_line() -> None:
    """Two breaks draw an empty line on the page, and an empty line is a paragraph boundary."""
    text = collapse_lines(["alpha", LINE_BREAK, LINE_BREAK, "beta"])

    assert text == "alpha\n\nbeta"
    assert paragraphs(text) == ["alpha", "beta"], "which is what a reader sees"


@pytest.mark.parametrize("count", [3, 4, 10])
def test_a_longer_run_of_breaks_is_capped_at_one_blank_line(count: int) -> None:
    """Beyond two there is nothing further the text model can say.

    ``paragraphs`` discards the empty paragraphs a longer run would produce, so carrying them
    would put whitespace nothing reads into text that is cited, shown and counted against a
    chunk's budget — which is what decorative spacer markup is made of.
    """
    parts: list[InlinePart] = ["alpha", *([LINE_BREAK] * count), "beta"]

    assert collapse_lines(parts) == "alpha\n\nbeta"


def test_a_break_at_either_edge_contributes_nothing() -> None:
    """There is no text on the other side of it for it to separate."""
    assert collapse_lines([LINE_BREAK, "alpha"]) == "alpha"
    assert collapse_lines(["alpha", LINE_BREAK]) == "alpha"
    assert collapse_lines([LINE_BREAK]) == ""


def test_whitespace_around_a_break_produces_no_extra_space() -> None:
    """The collapse happens per line, so the spaces either side of a break are line ends."""
    assert collapse_lines(["alpha  ", LINE_BREAK, "\n  beta"]) == "alpha\nbeta"


def test_whitespace_between_two_breaks_is_not_content() -> None:
    """Otherwise ``<br> <br>`` and ``<br><br>`` would say different things, and they do not."""
    assert collapse_lines(["alpha", LINE_BREAK, "   ", LINE_BREAK, "beta"]) == "alpha\n\nbeta"


def test_a_run_renders_a_break_as_a_space() -> None:
    """Where a newline is already the record separator, a break cannot be one.

    A table row, a list item, a task and a heading path element are all one line each. A
    newline there would read as another row, another item, another heading — which is a change
    to the document's shape rather than to its words.
    """
    assert collapse_run(["alpha", LINE_BREAK, "beta"]) == "alpha beta"
    assert collapse_run(["alpha", LINE_BREAK, LINE_BREAK, "beta"]) == "alpha beta"
    assert collapse_run([LINE_BREAK, "alpha", LINE_BREAK]) == "alpha"


@pytest.mark.parametrize(
    "source",
    [
        "",
        "   ",
        "alpha",
        "  alpha  ",
        "alpha  beta",
        "alpha\n\nbeta",
        "alpha\tbeta",
        "alpha \xa0 beta",
        "  alpha\n\tbeta \xa0 gamma  delta  ",
        ".  alpha  .",
    ],
)
def test_a_run_with_no_break_collapses_exactly_as_it_always_did(source: str) -> None:
    """The bound on the whole change: text with no break in it is untouched.

    Both joins reduce to :func:`collapse` on a single line, which is the function every parser
    already flattened with. Without this the fingerprint bumps could not claim that only
    documents containing a break produce different text — so the claim is checked across the
    whole surface of the collapse rather than on one string, where a divergence in leading
    whitespace, in an interior run or at a punctuation edge would pass unnoticed.

    ``\xa0`` is spelled by escape because a no-break space is indistinguishable from a space
    in a source file, and it is here because ``str.split`` treats it as whitespace.
    """
    assert collapse_lines([source]) == collapse(source)
    assert collapse_run([source]) == collapse(source)


def test_collapse_is_the_reader_s_whitespace_and_nothing_more() -> None:
    """What the two joins reduce to, pinned so the equalities above cannot both drift."""
    assert collapse("  alpha\n\tbeta \xa0 gamma  delta  ") == "alpha beta gamma delta"


def test_no_source_string_can_forge_a_break() -> None:
    """The property a sentinel character would have had to be chosen to have.

    A break is an object, so the question "which character can no document produce?" is never
    asked. Every candidate answer is tried here as ordinary text — the repr, a NUL, the
    replacement character, a private-use codepoint — and each stays one line.
    """
    for forgery in ("LINE_BREAK", repr(LINE_BREAK), "\x00", "�", "", "\\n"):
        assert "\n" not in collapse_lines([f"alpha{forgery}beta"]), forgery


def test_segments_hand_back_collapsed_lines_with_the_breaks_between_them() -> None:
    """What an inline element gives the run containing it: one collapsed line per line."""
    assert collapse_segments(["  alpha ", LINE_BREAK, " beta  "]) == ("alpha", LINE_BREAK, "beta")


def test_segments_are_idempotent() -> None:
    """A caller cannot always tell whether what it was handed has been through here already."""
    once = collapse_segments(["  alpha ", LINE_BREAK, LINE_BREAK, " beta  "])

    assert collapse_segments(once) == once
    assert collapse_lines(once) == collapse_lines(["  alpha ", LINE_BREAK, LINE_BREAK, " beta  "])


def test_a_run_with_nothing_in_it_produces_nothing() -> None:
    """So a container holding only breaks yields no block rather than an empty one."""
    assert collapse_lines([]) == ""
    assert collapse_lines(["", LINE_BREAK, "  "]) == ""
    assert collapse_run(["", LINE_BREAK, "  "]) == ""
