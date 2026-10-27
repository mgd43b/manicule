"""iCalendar (``text/calendar``): an invitation read as the sentences a person would search for.

Mail carries calendar invitations as ``text/calendar`` parts and ``.ics`` attachments, and the
raw format is hostile to search: folded lines, escaped commas, ``ATTENDEE;CN=...;PARTSTAT=...``
parameters that bury a name in punctuation. So this parser does what :class:`~manicule.parsers.
mail.MailParser` does for a message, and for the same reason (``docs/parsing.md`` §10): the
anchors address **one canonical text, composed here**, and :meth:`CalendarParser.resolve`
rebuilds it from the same bytes by the same rules. A line anchor into the source file would
cite folded fragments of a property, and its text would not be what the reader sees.

The canonical text is one stanza per event (or task), separated by a blank line::

    Event: Quarterly planning
    When: 2026-03-02 14:00 UTC to 2026-03-02 15:00 UTC
    Where: Room 4
    Organizer: Pat Example <pat@example.org>
    Attendees: Lee Example <lee@example.org>, sam@example.org
    Status: CONFIRMED
    Repeats: FREQ=WEEKLY;COUNT=4
    Description:
    Agenda for the quarter.

Times are rendered as written. Converting a ``TZID`` time to UTC needs the zone database and the
sender's idea of the zone, and a conversion that is quietly wrong is worse than a time that is
printed with the zone it was written in.

Standard library only: unfolding, unescaping and nesting are a page of code, and a calendar
library would decide what the stored text is without this module being able to say so.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from manicule.core.anchors import Anchor, LineAnchor
from manicule.core.content import BlockKind, ParsedBlock, RawDocument
from manicule.core.errors import ParseError
from manicule.parsers.base import ParserProfile, decode, lines_of, resolve_lines
from manicule.parsers.config import CALENDAR_MEDIA_TYPES, CalendarConfig
from manicule.parsers.plaintext import paragraph_spans

__all__ = ["CALENDAR_MEDIA_TYPES", "CalendarConfig", "CalendarParser"]

_COMPONENTS = {"VEVENT": "Event", "VTODO": "Task", "VJOURNAL": "Note"}
"""The components that are about something. ``VTIMEZONE`` and ``VALARM`` are machinery."""

_UNFOLD = re.compile(r"\r?\n[ \t]")
_ESCAPED = re.compile(r"\\([nN,;\\])")
_DATE = re.compile(r"^(\d{4})(\d{2})(\d{2})(?:T(\d{2})(\d{2})(\d{2})?(Z)?)?$")


@dataclass(slots=True)
class _Property:
    name: str
    params: dict[str, str]
    value: str


@dataclass(slots=True)
class _Component:
    kind: str
    properties: list[_Property] = field(default_factory=list[_Property])

    def first(self, name: str) -> _Property | None:
        return next((p for p in self.properties if p.name == name), None)

    def all(self, name: str) -> list[_Property]:
        return [p for p in self.properties if p.name == name]


class CalendarParser:
    """Parses an iCalendar stream into one block per event paragraph."""

    media_types = CALENDAR_MEDIA_TYPES
    profile = ParserProfile(name="calendar", max_unlocated_ratio=0.00, max_pagelevel_ratio=None)
    """No unlocated budget: every block is a span of the canonical text."""

    def __init__(self, config: CalendarConfig) -> None:
        self._config = config

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        """Yield the canonical text's paragraphs, in the order the events appear.

        Raises:
            ParseError: The bytes are not text, or are not an iCalendar stream.
        """
        lines = self._lines(raw)
        for start, end in paragraph_spans(lines, max_lines=self._config.max_block_lines):
            yield ParsedBlock(
                kind=BlockKind.PROSE,
                text="\n".join(lines[start - 1 : end]),
                anchor=LineAnchor(start=start, end=end),
            )

    async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
        """The canonical text ``anchor`` addresses, recomposed from ``raw``."""
        if not isinstance(anchor, LineAnchor):
            return None
        return resolve_lines("\n".join(self._lines(raw)), anchor)

    def _lines(self, raw: RawDocument) -> list[str]:
        components = _read(decode(raw), self._config.max_components, raw.uri)
        stanzas = [_stanza(component) for component in components]
        lines: list[str] = []
        for stanza in stanzas:
            if lines:
                lines.append("")
            lines.extend(stanza)
        return lines


def _read(text: str, limit: int, uri: str) -> list[_Component]:
    """The components worth rendering, with properties unfolded and unescaped.

    Raises:
        ParseError: There is no ``BEGIN:VCALENDAR``, so this is not a calendar and the next
            parser in the chain may be able to read it as text.
    """
    if "BEGIN:VCALENDAR" not in text[:4096].upper():
        msg = f"{uri}: declining — no BEGIN:VCALENDAR, so this is not an iCalendar stream."
        raise ParseError(msg)
    found: list[_Component] = []
    stack: list[_Component | None] = []
    for line in _UNFOLD.sub("", text.replace("\r\n", "\n")).split("\n"):
        if not line.strip():
            continue
        prop = _property(line)
        if prop is None:
            continue
        if prop.name == "BEGIN":
            kind = prop.value.strip().upper()
            component = _Component(kind) if kind in _COMPONENTS else None
            stack.append(component)
            if component is not None:
                found.append(component)
                if len(found) > limit:
                    msg = f"{uri}: more than {limit} events. Raise maxComponents to index the rest."
                    raise ParseError(msg)
        elif prop.name == "END":
            if stack:
                stack.pop()
        elif stack and stack[-1] is not None:
            stack[-1].properties.append(prop)
    return found


def _property(line: str) -> _Property | None:
    """``NAME;PARAM=value:value``, split where the first unquoted colon is."""
    in_quotes = False
    colon = -1
    for index, char in enumerate(line):
        if char == '"':
            in_quotes = not in_quotes
        elif char == ":" and not in_quotes:
            colon = index
            break
    if colon < 1:
        return None
    head, value = line[:colon], line[colon + 1 :]
    name, _, rest = head.partition(";")
    params: dict[str, str] = {}
    for part in re.split(r';(?=(?:[^"]*"[^"]*")*[^"]*$)', rest) if rest else []:
        key, _, val = part.partition("=")
        params[key.upper()] = val.strip('"')
    return _Property(name.strip().upper(), params, value)


def _unescaped(value: str) -> str:
    return _ESCAPED.sub(lambda m: "\n" if m.group(1) in "nN" else m.group(1), value)


def _time(prop: _Property) -> str:
    value = prop.value.strip()
    match = _DATE.match(value)
    if match is None:
        return _unescaped(value)
    year, month, day, hour, minute, _second, zulu = match.groups()
    out = f"{year}-{month}-{day}"
    if hour is not None:
        out += f" {hour}:{minute}"
        if zulu:
            out += " UTC"
        elif "TZID" in prop.params:
            out += f" ({prop.params['TZID']})"
    return out


def _person(prop: _Property) -> str:
    address = prop.value.strip()
    if address.lower().startswith("mailto:"):
        address = address[7:]
    name = prop.params.get("CN", "").strip()
    return f"{name} <{address}>" if name and address else name or address


def _stanza(component: _Component) -> list[str]:
    summary = component.first("SUMMARY")
    lines = [
        f"{_COMPONENTS[component.kind]}: {_one_line(summary.value) if summary else ''}".rstrip()
    ]
    start, end = component.first("DTSTART") or component.first("DUE"), component.first("DTEND")
    if start is not None:
        when = _time(start)
        if end is not None:
            when += f" to {_time(end)}"
        lines.append(f"When: {when}")
    for label, name in (("Where", "LOCATION"),):
        prop = component.first(name)
        if prop is not None and prop.value.strip():
            lines.append(f"{label}: {_one_line(prop.value)}")
    organizer = component.first("ORGANIZER")
    if organizer is not None:
        lines.append(f"Organizer: {_person(organizer)}")
    attendees = [_person(p) for p in component.all("ATTENDEE")]
    if attendees:
        lines.append(f"Attendees: {', '.join(attendees)}")
    for label, name in (("Status", "STATUS"), ("Repeats", "RRULE")):
        prop = component.first(name)
        if prop is not None and prop.value.strip():
            lines.append(f"{label}: {prop.value.strip()}")
    description = component.first("DESCRIPTION")
    if description is not None and description.value.strip():
        lines.append("Description:")
        lines.extend(lines_of(_unescaped(description.value).strip()))
    return lines


def _one_line(value: str) -> str:
    return " ".join(_unescaped(value).split())
