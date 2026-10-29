"""A streaming reader for unix mbox files: where a message begins and ends, and what it is.

``docs/connectors/mbox.md`` §3 and §4 are the design. This module is the part that decides what
every message in an archive *is*, so it is small, has no dependency beyond the standard
library, and is read by two callers that must agree byte for byte: a scan, which walks a file once
and yields each message's position and identity, and a read, which goes back to one position and
returns the message. Both run the same :func:`_read_message`, so the digest a scan computes and the
digest a read verifies cannot be derived by two pieces of code that drifted.

**Why not :mod:`mailbox`.** It builds a table of every message's offsets before it returns the
first one, hands a message back only after a second read, ignores ``Content-Length``, and splits
on the ``From `` prefix without looking at the next line (``mbox.md`` §3.5).

**Memory.** A file is read a line at a time with a bound on the line (a base64 body with no
newline is read in fragments), so the working set is a line, one message's header block and the
hash state. A message's body is only ever accumulated by :func:`read`, which is asked for one.
"""

from __future__ import annotations

import hashlib
import re
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, BinaryIO, Final

from manicule.core.mail_ids import normalize_message_id

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

__all__ = [
    "HEADER_LIMIT",
    "Located",
    "MessageChangedError",
    "is_mbox",
    "read",
    "scan",
]

HEADER_LIMIT: Final = 1 << 20
"""How much of a message is read as its header block before the rest is treated as body."""

MAX_LINE: Final = 1 << 20
"""The longest fragment read at once; a longer line arrives as several and only the first can
begin a message."""

_HASH_CHUNK: Final = 1 << 16

_SNIFF_BYTES: Final = 4096

_HEADER_LIKE: Final = re.compile(rb"[\x21-\x39\x3b-\x7e]+:")
"""A header field's start: a token with no space in it, then a colon."""

_TIME_OF_DAY: Final = re.compile(rb"\b\d{1,2}:\d{2}\b")
"""The clock time every envelope line carries, and prose beginning ``From`` rarely does."""

_QUOTED_FROM: Final = re.compile(rb">+From ")

_BLANK: Final = (b"\n", b"\r\n")

_UNSTORED_HEADERS: Final = frozenset(
    {
        b"status",
        b"x-status",
        b"x-mozilla-status",
        b"x-mozilla-status2",
        b"x-mozilla-keys",
        b"x-uid",
        b"x-keywords",
        b"x-evolution",
        b"content-length",
        b"lines",
    }
)
"""Headers that record what a mail client did to *its copy*, which are left out of the digest.

They are delivered with the message and left out of its identity (``mbox.md`` §4), so one message
filed in two folders, read in one and not the other, is one message.
"""

_PLACEHOLDER: Final = "FOLDER INTERNAL DATA"
_EXPUNGED: Final = 0x0008


class MessageChangedError(Exception):
    """The bytes at a message's recorded position are not the message that was scanned there."""


@dataclass(frozen=True, slots=True)
class Located:
    """One message found by :func:`scan`: where it is and what it is."""

    offset: int
    """Where the message's ``From `` line begins."""
    length: int
    """Bytes from the offset to the next message, or to the end of what was scanned."""
    digest: str
    """SHA-256, in hex, of the canonical bytes (``mbox.md`` §4): the message's identity."""
    size: int
    """Bytes of the message as delivered."""
    message_id: str | None
    """The normalized ``Message-ID``, if the message has a usable one."""


@dataclass(frozen=True, slots=True)
class _Line:
    offset: int
    data: bytes
    starts: bool
    """Whether this is the start of a line. A fragment of a very long line is not."""

    @property
    def blank(self) -> bool:
        return self.starts and self.data in _BLANK


class _Lines:
    """Lines of a file up to a limit, with two lines of lookahead and byte-exact positions."""

    def __init__(self, handle: BinaryIO, limit: int, *, offset: int = 0) -> None:
        self._handle = handle
        self.limit = limit
        self._position = offset
        self._ahead: deque[_Line] = deque()
        self._continued = False
        handle.seek(offset)

    def peek(self, index: int = 0) -> _Line | None:
        while len(self._ahead) <= index:
            line = self._read()
            if line is None:
                return None
            self._ahead.append(line)
        return self._ahead[index]

    def pop(self) -> _Line | None:
        line = self.peek()
        if line is not None:
            self._ahead.popleft()
        return line

    @property
    def position(self) -> int:
        """Where the next unread line begins, counting lines already peeked at as unread."""
        return self._ahead[0].offset if self._ahead else self._position

    def seek(self, offset: int) -> None:
        """Continue from ``offset`` (a line start), forgetting any lookahead."""
        self._ahead.clear()
        self._position = offset
        self._continued = False
        self._handle.seek(offset)

    def raw_at(self, offset: int, count: int) -> bytes:
        """``count`` bytes at ``offset`` within the limit, without moving the reader."""
        wanted = max(0, min(count, self.limit - offset))
        if wanted == 0:
            return b""
        resume = self._handle.tell()
        self._handle.seek(offset)
        data = self._handle.read(wanted)
        self._handle.seek(resume)
        return data

    def _read(self) -> _Line | None:
        room = self.limit - self._position
        if room <= 0:
            return None
        data = self._handle.readline(min(MAX_LINE, room))
        if not data:
            return None
        line = _Line(self._position, data, starts=not self._continued)
        self._position += len(data)
        self._continued = not data.endswith(b"\n")
        return line


def _separator_at(lines: _Lines) -> bool:
    """Whether the next line begins a message (``mbox.md`` §3.1 (1) and (3)).

    The blank line before it (2) is the caller's to know, because it depends on what was read.
    """
    head = lines.peek()
    if head is None or not head.starts or not head.data.startswith(b"From "):
        return False
    following = lines.peek(1)
    return (
        following is not None
        and following.starts
        and _HEADER_LIKE.match(following.data) is not None
    )


class _Facts:
    """The few header fields that decide what a message is, read without a full parse."""

    _WANTED: Final = frozenset(
        {b"message-id", b"content-length", b"x-mozilla-status", b"x-status", b"subject", b"x-imap"}
    )

    def __init__(self) -> None:
        self.values: dict[bytes, bytes] = {}
        self._current: bytes | None = None

    def add(self, data: bytes) -> None:
        if data[:1] in (b" ", b"\t"):
            if self._current is not None:
                self.values[self._current] += b" " + data.strip()
            return
        name, colon, value = data.partition(b":")
        key = name.strip().lower()
        if colon and key in self._WANTED and key not in self.values:
            self._current = key
            self.values[key] = value.strip()
        else:
            self._current = None

    @property
    def message_id(self) -> str | None:
        raw = self.values.get(b"message-id")
        return normalize_message_id(raw.decode("ascii", "replace")) if raw is not None else None

    @property
    def content_length(self) -> int | None:
        raw = self.values.get(b"content-length")
        if raw is None or not raw.isdigit():
            return None
        try:
            return int(raw)
        except ValueError:  # more digits than the interpreter will convert
            return None

    @property
    def dropped(self) -> bool:
        """Flagged deleted by the client that wrote the file, or a folder's placeholder."""
        status = self.values.get(b"x-mozilla-status", b"")
        try:
            if int(status, 16) & _EXPUNGED:
                return True
        except ValueError:
            pass
        if b"D" in self.values.get(b"x-status", b"").upper():
            return True
        if b"x-imap" in self.values:
            return True
        return _PLACEHOLDER in self.values.get(b"subject", b"").decode("ascii", "replace").upper()


class _Canonical:
    """The message as delivered, and the digest of it less what a mail client recorded."""

    def __init__(self, *, collect: bool) -> None:
        self._hash = hashlib.sha256()
        self._pending: list[bytes] = []
        self._pending_size = 0
        self._parts: list[bytes] | None = [] if collect else None
        self.size = 0
        self._continuing_excluded = False

    def header(self, data: bytes) -> None:
        data = _lf(data)
        if data[:1] in (b" ", b"\t"):
            excluded = self._continuing_excluded
        else:
            name = data.partition(b":")[0].strip().lower()
            excluded = name in _UNSTORED_HEADERS
        self._continuing_excluded = excluded
        self._deliver(data, hashed=not excluded)

    def blank(self, data: bytes) -> None:
        self._deliver(_lf(data), hashed=True)

    def body(self, data: bytes, *, unquote: bool) -> None:
        data = _lf(data)
        if unquote and data[:1] == b">" and _QUOTED_FROM.match(data):
            data = data[1:]
        self._deliver(data, hashed=True)

    def _deliver(self, data: bytes, *, hashed: bool) -> None:
        self.size += len(data)
        if self._parts is not None:
            self._parts.append(data)
        if hashed:
            self._pending.append(data)
            self._pending_size += len(data)
            if self._pending_size >= _HASH_CHUNK:
                self._flush()

    def _flush(self) -> None:
        if self._pending:
            self._hash.update(b"".join(self._pending))
            self._pending.clear()
            self._pending_size = 0

    @property
    def digest(self) -> str:
        self._flush()
        return self._hash.hexdigest()

    @property
    def content(self) -> bytes:
        if self._parts is None:  # pragma: no cover - read() always collects
            raise RuntimeError("this reader did not collect the message")
        return b"".join(self._parts)


def _lf(data: bytes) -> bytes:
    return data[:-2] + b"\n" if data.endswith(b"\r\n") else data


@dataclass(frozen=True, slots=True)
class _Message:
    offset: int
    end: int
    canonical: _Canonical
    facts: _Facts


def _read_message(lines: _Lines, *, collect: bool) -> _Message:
    """Read the message whose ``From `` line is next, up to the next message or the limit."""
    separator = lines.pop()
    if separator is None:  # pragma: no cover - callers peeked at it
        raise MessageChangedError
    canonical = _Canonical(collect=collect)
    facts = _Facts()
    header_bytes = 0
    body_follows = False
    blank_ended = False
    while (line := lines.peek()) is not None:
        if header_bytes and _separator_at(lines):
            return _Message(separator.offset, line.offset, canonical, facts)
        lines.pop()
        if line.blank:
            canonical.blank(line.data)
            body_follows = blank_ended = True
            break
        header_bytes += len(line.data)
        if header_bytes > HEADER_LIMIT:
            # Headers that do not end are delivered as read, and the rest is body.
            canonical.body(line.data, unquote=True)
            body_follows = True
            break
        canonical.header(line.data)
        facts.add(line.data)
    if body_follows:
        length = facts.content_length if blank_ended else None
        if length is not None and _body_by_length(lines, canonical, length):
            return _Message(separator.offset, lines.position, canonical, facts)
        _body_by_scan(lines, canonical, after_blank=blank_ended)
    head = lines.peek()
    return _Message(separator.offset, head.offset if head else lines.position, canonical, facts)


def _starts_message(data: bytes) -> bool:
    """Whether ``data`` begins with a complete separator: ``From ``, then a header field."""
    first, newline, following = data.partition(b"\n")
    return (
        first.startswith(b"From ") and bool(newline) and _HEADER_LIKE.match(following) is not None
    )


def _body_by_length(lines: _Lines, canonical: _Canonical, length: int) -> bool:
    """Read ``length`` bytes of body if that lands on a message boundary (``mbox.md`` §3.2).

    Believed only when what follows is the end of the file, or the blank line and ``From `` of the
    next message. ``False`` leaves the reader where the body begins, for the scan to read instead.
    """
    start = lines.position
    lines.seek(start)
    end = start + length
    if end > lines.limit:
        return False
    after = lines.raw_at(end, _SNIFF_BYTES)
    blank = 2 if after[:2] == b"\r\n" else 1 if after[:1] == b"\n" else 0
    rest = after[blank:]
    if rest and not _starts_message(rest):
        return False
    limit = lines.limit
    lines.limit = end
    while (line := lines.pop()) is not None:
        canonical.body(line.data, unquote=False)
    lines.limit = limit
    lines.seek(end + blank)
    return True


def _body_by_scan(lines: _Lines, canonical: _Canonical, *, after_blank: bool) -> None:
    held: bytes | None = None
    previous_blank = after_blank
    while (line := lines.peek()) is not None:
        if previous_blank and _separator_at(lines):
            return
        lines.pop()
        if held is not None:
            canonical.body(held, unquote=True)
            held = None
        if line.blank:
            held = line.data
        else:
            canonical.body(line.data, unquote=True)
        previous_blank = line.blank
    # A blank line still held at the end of the file belongs to the separator that was not there.


def scan(path: Path, limit: int) -> Iterator[Located]:
    """Every live message in the first ``limit`` bytes of an mbox file, in file order.

    Messages a mail client flagged deleted, and a folder's placeholder message, are not yielded
    (``mbox.md`` §3.4). Raises :class:`OSError` if the file cannot be read.
    """
    with path.open("rb", buffering=1 << 20) as handle:
        lines = _Lines(handle, limit)
        previous_blank = True
        while True:
            line = lines.peek()
            if line is None:
                return
            if previous_blank and _separator_at(lines):
                message = _read_message(lines, collect=False)
                if not message.facts.dropped:
                    yield Located(
                        offset=message.offset,
                        length=message.end - message.offset,
                        digest=message.canonical.digest,
                        size=message.canonical.size,
                        message_id=message.facts.message_id,
                    )
                previous_blank = True
                continue
            lines.pop()
            previous_blank = line.blank


def read(path: Path, offset: int, length: int, digest: str) -> bytes:
    """The message at ``offset``, canonical, if it is still the message whose digest is given.

    Raises:
        MessageChangedError: What is there is not that message.
        OSError: The file cannot be read.
    """
    with path.open("rb") as handle:
        lines = _Lines(handle, offset + length, offset=offset)
        head = lines.peek()
        if head is None or not head.data.startswith(b"From "):
            raise MessageChangedError
        message = _read_message(lines, collect=True)
    if message.canonical.digest != digest:
        raise MessageChangedError
    return message.canonical.content


def is_mbox(path: Path) -> bool:
    """Whether a file starts like an mbox (``mbox.md`` §2). Raises :class:`OSError` if it cannot be
    opened: an unreadable file is not a file that is not mail, and the walk decides what it means.

    A ``From `` line that carries a time of day, as every writer's envelope line does, then a
    header field. The time is what keeps a text file that starts ``From the desk of…`` and happens
    to have a line like ``Thanks:`` next from being read as mail.
    """
    with path.open("rb") as handle:
        head = handle.read(_SNIFF_BYTES)
    if not head.startswith(b"From "):
        return False
    first, newline, rest = head.partition(b"\n")
    if not newline or _TIME_OF_DAY.search(first) is None:
        return False
    return _HEADER_LIKE.match(rest) is not None
