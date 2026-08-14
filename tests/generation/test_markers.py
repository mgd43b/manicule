"""The marker syntax, and the scanner that finds one in a stream arriving in pieces."""

from __future__ import annotations

import pytest

from manicule.generation.markers import (
    ATTEMPT_PREFIX,
    MARKER_MAX_LEN,
    MarkerScanner,
    ScanEventKind,
    escape_markers,
    parse_slots,
    render_marker,
)


def scan(*chunks: str) -> list[tuple[ScanEventKind, str, tuple[int, ...]]]:
    scanner = MarkerScanner()
    events = [event for chunk in chunks for event in scanner.feed(chunk)]
    events.extend(scanner.finish())
    return [(event.kind, event.text, event.slots) for event in events]


def text_of(events: list[tuple[ScanEventKind, str, tuple[int, ...]]]) -> str:
    """Everything the reader would see if nothing were deleted."""
    return "".join(text for _, text, _ in events)


def test_a_marker_split_across_chunks_is_still_one_marker() -> None:
    """The normal case, not an edge case: providers do not align chunks to token boundaries."""
    events = scan("Roll back", f"{ATTEMPT_PREFIX[:3]}", f"{ATTEMPT_PREFIX[3:]}:", "3,5]] now")

    assert [kind for kind, _, _ in events] == [
        ScanEventKind.TEXT,
        ScanEventKind.MARKER,
        ScanEventKind.TEXT,
    ]
    assert events[1][2] == (3, 5)


@pytest.mark.parametrize(
    "sample",
    [
        "argv[1] and items[0] are ordinary code",
        "See [1] and [2] in the bibliography",
        "A footnote reference[^3] in Markdown",
        "A [[wiki link]] and [[another]]",
        "An unmatched [ bracket at the end",
    ],
)
def test_ordinary_prose_and_code_pass_through_untouched(sample: str) -> None:
    """``[1]`` was the obvious marker and it is unusable: a binder scanning for it eats the
    answer, because the corpus is full of code blocks by design."""
    events = scan(sample)

    assert text_of(events) == sample
    assert all(kind is ScanEventKind.TEXT for kind, _, _ in events)


def test_a_wiki_link_is_released_without_waiting_for_the_marker_bound() -> None:
    """``[[`` that is not an attempt is decided as soon as it diverges, so a wiki link does
    not stall the stream for 64 characters."""
    scanner = MarkerScanner()
    emitted = "".join(event.text for event in scanner.feed("start [[Some Page"))

    assert emitted == "start [[Some Page"


@pytest.mark.parametrize(
    ("written", "slots"),
    [
        (f"{ATTEMPT_PREFIX}:3]]", (3,)),
        (f"{ATTEMPT_PREFIX}: 3]]", (3,)),
        (f"{ATTEMPT_PREFIX}:3 ]]", (3,)),
        (f"{ATTEMPT_PREFIX}:3 , 5]]", (3, 5)),
        (f"{ATTEMPT_PREFIX}:3,3]]", (3,)),
        ("[[CITE:2]]", (2,)),
    ],
)
def test_the_one_normalization_is_of_syntax_the_binder_defined_itself(
    written: str, slots: tuple[int, ...]
) -> None:
    events = scan(written)

    assert events == [(ScanEventKind.MARKER, written, slots)]
    assert render_marker(slots) == f"{ATTEMPT_PREFIX}:{','.join(str(s) for s in slots)}]]"


@pytest.mark.parametrize("payload", [":", ":abc", " 3", ":3;5", ":-1", ""])
def test_a_closed_attempt_that_is_not_a_slot_list_is_malformed_rather_than_prose(
    payload: str,
) -> None:
    events = scan(f"{ATTEMPT_PREFIX}{payload}]]")

    assert [kind for kind, _, _ in events] == [ScanEventKind.MALFORMED]


def test_an_attempt_that_never_closes_is_released_verbatim_rather_than_stalling() -> None:
    """Without the bound one unterminated ``[[cite:`` holds the stream forever.

    Released rather than deleted: the scanner has no evidence about where it was meant to
    end, and deleting to a guessed boundary is the sentence-level surgery this design refuses.
    """
    overrun = f"{ATTEMPT_PREFIX}:" + "x" * (MARKER_MAX_LEN + 10) + "]] tail"

    events = scan(overrun)

    assert events[0][0] is ScanEventKind.UNTERMINATED
    assert text_of(events) == overrun


def test_a_stream_that_ends_mid_marker_loses_nothing() -> None:
    """``finish`` is what turns a truncated answer into one that still shows what it said."""
    events = scan(f"Roll back{ATTEMPT_PREFIX}:1")

    assert text_of(events) == f"Roll back{ATTEMPT_PREFIX}:1"


def test_slot_zero_parses_so_that_range_checking_can_reject_it() -> None:
    """Zero is a well-formed integer and an invented slot. It must reach level 0 to be
    counted as invention rather than swallowed as a syntax error."""
    assert parse_slots(":0") == (0,)


def test_escaping_neutralizes_exactly_what_the_scanner_would_bind() -> None:
    """manicule's own documentation describes this syntax and is exactly the sort of thing
    somebody indexes. A passage containing a literal marker that the model quoted back would
    otherwise bind — to a real passage, so it would even verify."""
    for sample in (f"{ATTEMPT_PREFIX}:3]]", "[[Cite:4]]", f"prose {ATTEMPT_PREFIX}:1,2]] more"):
        escaped = escape_markers(sample)

        assert all(kind is ScanEventKind.TEXT for kind, _, _ in scan(escaped))
        assert text_of(scan(escaped)) == escaped


def test_escaping_leaves_text_without_marker_syntax_alone() -> None:
    sample = "items[0], [[wiki]], and a citation-shaped word: cite"

    assert escape_markers(sample) == sample
