"""An Outlook event rendered as iCalendar (``docs/connectors/outlook.md`` §7.2).

The renderer is pure; the calendar parser reading what it writes is the other half of the
promise, so these round-trip through :class:`~manicule.parsers.calendar.CalendarParser`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from manicule.connectors.outlook_ics import render_event
from manicule.core.content import RawDocument
from manicule.parsers.calendar import CalendarParser
from manicule.parsers.config import CalendarConfig


def event(**changes: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "e1",
        "iCalUId": "uid-1",
        "subject": "Quarterly planning",
        "isAllDay": False,
        "start": {"dateTime": "2026-03-02T14:00:00.0000000", "timeZone": "UTC"},
        "end": {"dateTime": "2026-03-02T15:00:00.0000000", "timeZone": "UTC"},
        "location": {"displayName": "Room 4"},
        "organizer": {"emailAddress": {"name": "Pat Example", "address": "pat@example.org"}},
        "attendees": [{"emailAddress": {"name": "Lee Example", "address": "lee@example.org"}}],
        "body": {"contentType": "text", "content": "Agenda for the quarter."},
        "isCancelled": False,
    }
    return {**base, **changes}


async def parsed(fields: Mapping[str, Any]) -> list[str]:
    raw = RawDocument(
        source_id="x",
        uri="https://x.example/",
        media_type="text/calendar",
        content=render_event(fields),
    )
    return [block.text async for block in CalendarParser(CalendarConfig()).parse(raw)]


async def test_what_the_parser_reads_is_what_a_person_would_search_for() -> None:
    (stanza,) = await parsed(event())

    assert stanza == (
        "Event: Quarterly planning\n"
        "When: 2026-03-02 14:00 UTC to 2026-03-02 15:00 UTC\n"
        "Where: Room 4\n"
        "Organizer: Pat Example <pat@example.org>\n"
        "Attendees: Lee Example <lee@example.org>\n"
        "Status: CONFIRMED\n"
        "Description:\n"
        "Agenda for the quarter."
    )


async def test_a_canceled_event_says_so() -> None:
    (stanza,) = await parsed(event(isCancelled=True))

    assert "Status: CANCELLED" in stanza


async def test_an_all_day_event_keeps_a_date_and_no_time() -> None:
    (stanza,) = await parsed(
        event(
            isAllDay=True,
            start={"dateTime": "2026-03-03T00:00:00.0000000", "timeZone": "UTC"},
            end={"dateTime": "2026-03-04T00:00:00.0000000", "timeZone": "UTC"},
        )
    )

    assert "When: 2026-03-03 to 2026-03-04" in stanza


async def test_commas_semicolons_backslashes_and_newlines_survive_the_round_trip() -> None:
    (stanza,) = await parsed(
        event(
            subject=r"Q1, Q2; the \ plan",
            location={"displayName": "Room 4, 2nd floor; east"},
            body={"content": "line one\r\nline two, with a comma\nline three"},
        )
    )

    assert r"Event: Q1, Q2; the \ plan" in stanza
    assert "Where: Room 4, 2nd floor; east" in stanza
    assert "line one\nline two, with a comma\nline three" in stanza


async def test_a_long_description_is_folded_and_read_back_whole() -> None:
    body = "word " * 400
    raw = render_event(event(body={"content": body}))

    assert all(len(line.encode()) <= 75 for line in raw.decode().split("\r\n"))
    (stanza,) = await parsed(event(body={"content": body}))
    assert stanza.endswith(body.strip())


async def test_a_fold_never_falls_inside_a_multibyte_character() -> None:
    body = "é" * 200 + "日本語" * 40
    (stanza,) = await parsed(event(body={"content": body}))

    assert stanza.endswith(body)


async def test_a_property_graph_did_not_send_is_not_written() -> None:
    raw = render_event({"id": "e1", "subject": "Bare"}).decode()

    for absent in ("DTSTART", "DTEND", "LOCATION", "ORGANIZER", "ATTENDEE", "DESCRIPTION"):
        assert absent not in raw
    assert "UID:e1" in raw, "the id stands in for a missing iCalUId"


async def test_a_time_that_is_not_one_of_graphs_shapes_is_left_out_not_reinterpreted() -> None:
    raw = render_event(event(start={"dateTime": "next tuesday"}, end="nonsense")).decode()

    assert "DTSTART" not in raw
    assert "DTEND" not in raw


async def test_an_attendee_without_an_address_keeps_the_name_and_one_with_nothing_is_dropped() -> (
    None
):
    raw = render_event(
        event(
            attendees=[
                {"emailAddress": {"name": "No Address"}},
                {"emailAddress": {}},
                {"nothing": True},
            ]
        )
    ).decode()

    assert raw.count("ATTENDEE") == 1
    assert 'ATTENDEE;CN="No Address":' in raw


async def test_a_quote_in_a_display_name_cannot_break_out_of_its_parameter() -> None:
    raw = render_event(
        event(
            organizer={"emailAddress": {"name": 'Pat "The Boss":evil', "address": "p@example.org"}}
        )
    ).decode()

    assert 'ORGANIZER;CN="Pat The Boss:evil":mailto:p@example.org' in raw


@pytest.mark.parametrize("categories", [["Red", "Blue, dark"], []])
async def test_categories_are_escaped_and_omitted_when_empty(categories: list[str]) -> None:
    raw = render_event(event(categories=categories)).decode()

    assert ("CATEGORIES:Red,Blue\\, dark" in raw) is bool(categories)
    assert ("CATEGORIES" in raw) is bool(categories)


def test_the_output_is_crlf_terminated_utf8() -> None:
    raw = render_event(event())

    assert raw.endswith(b"\r\n")
    assert raw.startswith(b"BEGIN:VCALENDAR\r\n")
    assert b"\n" not in raw.replace(b"\r\n", b"")
