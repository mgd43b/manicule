"""An Outlook event rendered as iCalendar, so the calendar parser can read it.

``docs/connectors/outlook.md`` §7.2. Graph has no MIME or ICS representation of an event in v1.0,
only JSON, and the repository already has a ``text/calendar`` parser whose canonical text is one
stanza per event (title, when, where, organizer, attendees, status, description). Rendering to
that format here, rather than adding a JSON parser, keeps one calendar reader and one citation
shape for events that arrive as invitations in mail and events that arrive from a calendar.

**Times are written in UTC with a ``Z``.** The client asks Graph for UTC
(``Prefer: outlook.timezone="UTC"``), and the parser prints a ``Z`` time as ``UTC`` rather than
converting a zone it cannot verify. An all-day event is a ``DATE`` value and keeps no time.

**Nothing is guessed.** A property Graph did not send is not written, and a time that is not one
of Graph's two shapes is left out rather than reinterpreted.

The text is escaped and folded as RFC 5545 asks, so any other reader of the retained bytes sees a
valid calendar.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Final, cast

__all__ = ["render_event"]

type Json = Mapping[str, object]

_LOCAL: Final = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?Z?$")
_FOLD: Final = 75
"""Octets a content line may hold before it is folded."""


def render_event(event: Json) -> bytes:
    """The event as a one-event ``VCALENDAR``, UTF-8 with CRLF line ends."""
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//manicule//outlook//EN", "BEGIN:VEVENT"]
    uid = _text(event.get("iCalUId")) or _text(event.get("id"))
    if uid:
        lines.append(f"UID:{_escape(uid)}")
    lines.append(f"SUMMARY:{_escape(_text(event.get('subject')))}")
    all_day = event.get("isAllDay") is True
    for name, key in (("DTSTART", "start"), ("DTEND", "end")):
        rendered = _when(name, event.get(key), all_day=all_day)
        if rendered:
            lines.append(rendered)
    where = _name(event.get("location"), "displayName")
    if where:
        lines.append(f"LOCATION:{_escape(where)}")
    organizer = _person("ORGANIZER", _mapping(event.get("organizer")))
    if organizer:
        lines.append(organizer)
    for attendee in _sequence(event.get("attendees")):
        rendered = _person("ATTENDEE", _mapping(attendee))
        if rendered:
            lines.append(rendered)
    lines.append("STATUS:CANCELLED" if event.get("isCancelled") is True else "STATUS:CONFIRMED")
    categories = [c for c in _sequence(event.get("categories")) if isinstance(c, str) and c]
    if categories:
        lines.append("CATEGORIES:" + ",".join(_escape(c) for c in categories))
    body = _mapping(event.get("body"))
    description = _text(body.get("content")).strip() if body is not None else ""
    if description:
        lines.append(f"DESCRIPTION:{_escape(description)}")
    lines.extend(["END:VEVENT", "END:VCALENDAR"])
    return ("\r\n".join(_fold(line) for line in lines) + "\r\n").encode("utf-8")


def _when(name: str, value: object, *, all_day: bool) -> str:
    moment = _mapping(value)
    text = _text(moment.get("dateTime")) if moment is not None else ""
    match = _LOCAL.match(text)
    if match is None:
        return ""
    year, month, day, hour, minute, second = match.groups()
    if all_day:
        return f"{name};VALUE=DATE:{year}{month}{day}"
    return f"{name}:{year}{month}{day}T{hour}{minute}{second}Z"


def _person(prop: str, holder: Json | None) -> str:
    address = _mapping(holder.get("emailAddress")) if holder is not None else None
    if address is None:
        return ""
    mail = _text(address.get("address"))
    name = _text(address.get("name"))
    if not mail and not name:
        return ""
    common = f';CN="{_param(name)}"' if name else ""
    return f"{prop}{common}:mailto:{mail}" if mail else f"{prop}{common}:"


def _param(value: str) -> str:
    """A parameter value: quoted by the caller, so only the quote and controls are removed."""
    return "".join(c for c in value if c != '"' and c.isprintable())


def _escape(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\r\n", "\\n")
        .replace("\n", "\\n")
        .replace("\r", "\\n")
    )


def _fold(line: str) -> str:
    """Fold at 75 octets, never inside a multibyte character, continuation lines led by a space."""
    encoded = line.encode("utf-8")
    if len(encoded) <= _FOLD:
        return line
    pieces: list[str] = []
    current = ""
    budget = _FOLD
    for char in line:
        size = len(char.encode("utf-8"))
        if len(current.encode("utf-8")) + size > budget:
            pieces.append(current)
            current = char
            budget = _FOLD - 1
        else:
            current += char
    pieces.append(current)
    return "\r\n ".join(pieces)


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _mapping(value: object) -> Json | None:
    return cast("Json", value) if isinstance(value, dict) else None


def _sequence(value: object) -> Sequence[object]:
    return cast("Sequence[object]", value) if isinstance(value, list) else ()


def _name(value: object, key: str) -> str:
    holder = _mapping(value)
    return _text(holder.get(key)) if holder is not None else ""
