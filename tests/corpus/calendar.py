"""Fixtures for the iCalendar parser.

Every event is deliberately unlike every other one (``tests/corpus/plaintext.py`` says why).
Addresses are at ``example.org``.
"""

from __future__ import annotations

from pathlib import Path

INVITATION = "\r\n".join(
    [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Example//Calendar//EN",
        "METHOD:REQUEST",
        "BEGIN:VEVENT",
        "UID:planning-1@example.org",
        "DTSTART:20260302T140000Z",
        "DTEND:20260302T150000Z",
        "SUMMARY:Quarterly planning\\, capacity and budget",
        r"LOCATION:Room 4\; second floor",
        'ORGANIZER;CN="Pat Example":mailto:pat@example.org',
        "ATTENDEE;CN=Lee Example;PARTSTAT=ACCEPTED:mailto:lee@example.org",
        "ATTENDEE;PARTSTAT=NEEDS-ACTION:mailto:sam@example.org",
        "STATUS:CONFIRMED",
        "RRULE:FREQ=WEEKLY;COUNT=4",
        "DESCRIPTION:Agenda for the quarter.\\nBring the retention figures and the",
        "  outstanding hardware requests.",
        "BEGIN:VALARM",
        "TRIGGER:-PT15M",
        "DESCRIPTION:Reminder text that must not be rendered",
        "END:VALARM",
        "END:VEVENT",
        "END:VCALENDAR",
        "",
    ]
)

MULTI = "\r\n".join(
    [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "BEGIN:VEVENT",
        "SUMMARY:Offsite in the autumn",
        "DTSTART;VALUE=DATE:20261012",
        "DTEND;VALUE=DATE:20261014",
        "LOCATION:Lakeside lodge",
        "END:VEVENT",
        "BEGIN:VEVENT",
        "SUMMARY:Retention review",
        "DTSTART;TZID=Europe/London:20261104T093000",
        "DESCRIPTION:Walk the table of tiers.",
        "END:VEVENT",
        "BEGIN:VTODO",
        "SUMMARY:Send the minutes",
        "DUE:20261105T170000Z",
        "END:VTODO",
        "END:VCALENDAR",
        "",
    ]
)

NOT_A_CALENDAR = "Plain words that mention nothing of the format.\n"


def build(dest: Path) -> None:
    """Write this format's fixtures into ``dest``."""
    (dest / "invitation.ics").write_bytes(INVITATION.encode())
    (dest / "multi.ics").write_bytes(MULTI.encode())
    (dest / "not-a-calendar.ics").write_text(NOT_A_CALENDAR, encoding="utf-8")
