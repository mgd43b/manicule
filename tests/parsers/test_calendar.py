"""The iCalendar parser: one composed text, and anchors that resolve against it."""

from __future__ import annotations

from pathlib import Path

import pytest

from manicule.chunking import StructuralChunker
from manicule.core.errors import ParseError
from manicule.core.protocols import read_blocks
from manicule.parsers.calendar import CalendarConfig, CalendarParser
from tests.parsers.support import check_corpus, raw_from, raw_of


@pytest.fixture
def parser() -> CalendarParser:
    return CalendarParser(CalendarConfig())


async def test_the_fixtures_round_trip(
    parser: CalendarParser, corpus: Path, chunker: StructuralChunker
) -> None:
    raws = [
        raw_from(corpus / "calendar" / name, "text/calendar")
        for name in ("invitation.ics", "multi.ics")
    ]
    await check_corpus(parser, raws, chunker=chunker, min_blocks=4)


async def test_an_invitation_reads_as_the_sentences_a_person_searches_for(
    parser: CalendarParser, corpus: Path
) -> None:
    blocks = await read_blocks(
        parser, raw_from(corpus / "calendar" / "invitation.ics", "text/calendar")
    )

    text = "\n".join(block.text for block in blocks)
    assert "Event: Quarterly planning, capacity and budget" in text, "escapes are undone"
    assert "When: 2026-03-02 14:00 UTC to 2026-03-02 15:00 UTC" in text
    assert "Where: Room 4; second floor" in text
    assert "Organizer: Pat Example <pat@example.org>" in text
    assert "Attendees: Lee Example <lee@example.org>, sam@example.org" in text
    assert "Repeats: FREQ=WEEKLY;COUNT=4" in text
    assert (
        "Agenda for the quarter.\nBring the retention figures and the outstanding hardware" in text
    ), "a folded line is one line, and an escaped newline is a real one"
    assert "Reminder text" not in text, "an alarm is machinery, not content"


async def test_dates_tasks_and_zones_are_rendered_as_written(
    parser: CalendarParser, corpus: Path
) -> None:
    blocks = await read_blocks(parser, raw_from(corpus / "calendar" / "multi.ics", "text/calendar"))

    text = "\n".join(block.text for block in blocks)
    assert "When: 2026-10-12 to 2026-10-14" in text
    assert "When: 2026-11-04 09:30 (Europe/London)" in text
    assert "Task: Send the minutes" in text
    assert "When: 2026-11-01 08:00:37 UTC to 2026-11-03 17:00 UTC" in text, (
        "seconds are kept when they are not zero, and a task with a start still shows its due date"
    )
    assert "When: 2026-11-05 17:00 UTC" in text


async def test_something_that_is_not_a_calendar_is_declined(
    parser: CalendarParser, corpus: Path
) -> None:
    with pytest.raises(ParseError, match="not an iCalendar"):
        await read_blocks(
            parser, raw_from(corpus / "calendar" / "not-a-calendar.ics", "text/calendar")
        )


async def test_a_calendar_past_the_component_ceiling_is_declined_not_truncated() -> None:
    events = "".join(f"BEGIN:VEVENT\r\nSUMMARY:e{i}\r\nEND:VEVENT\r\n" for i in range(5))
    raw = raw_of(f"BEGIN:VCALENDAR\r\n{events}END:VCALENDAR\r\n".encode(), "text/calendar")

    with pytest.raises(ParseError, match="more than 3 events"):
        await read_blocks(CalendarParser(CalendarConfig(max_components=3)), raw)


async def test_a_fold_inside_a_multibyte_character_is_restored_before_decoding(
    parser: CalendarParser, corpus: Path
) -> None:
    raw = raw_from(corpus / "calendar" / "folded-multibyte.ics", "text/calendar")

    blocks = await read_blocks(parser, raw)

    assert "Event: Café sync with the team" in "\n".join(block.text for block in blocks)


async def test_a_header_with_many_separators_is_read_in_linear_time(
    parser: CalendarParser,
) -> None:
    hostile = "ATTENDEE" + ";X=1" * 200_000 + ":mailto:a@example.org"
    raw = raw_of(
        f"BEGIN:VCALENDAR\r\nBEGIN:VEVENT\r\nSUMMARY:s\r\n{hostile}\r\nEND:VEVENT\r\nEND:VCALENDAR\r\n".encode(),
        "text/calendar",
    )

    blocks = await read_blocks(parser, raw)

    assert "Attendees: a@example.org" in "\n".join(block.text for block in blocks)
