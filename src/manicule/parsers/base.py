"""Helpers every parser shares, and the rules they encode.

Three of these exist because getting them wrong is invisible:

**Ordinals are converted once, here.** Every library underneath counts from zero — pdfium
page indices, tree-sitter node rows, ``markdown-it-py``'s ``token.map`` (which is
additionally half-open). Every anchor manicule stores counts from one and is inclusive at
both ends, because that is what ``token.py:42`` means to a person and what every editor
shows. Converting at the call site means converting in twenty places, and an off-by-one
produces a citation that resolves to adjacent text and reads perfectly.

**Slugs are allocated, not derived.** Two sections called "Overview" produce one slug unless
something counts them, and a fragment that addresses two places addresses neither.

**Text is decoded once, with an explicit refusal.** A parser handed bytes it cannot decode
declines so the next parser in the chain gets a turn; it never indexes replacement
characters, which match queries by accident and cite nothing.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from manicule.core.anchors import HeadingAnchor, LineAnchor
from manicule.core.content import RawDocument
from manicule.core.errors import ParseError
from manicule.testing.roundtrip import ParserProfile

__all__ = [
    "HeadingStack",
    "ParserProfile",
    "SlugAllocator",
    "decode",
    "heading_anchor",
    "is_probably_text",
    "line_range_of",
    "lines_of",
    "merge_line_anchors",
    "resolve_lines",
    "slugify",
]


_NOT_SLUGGABLE = re.compile(r"[^\w\- ]+", re.UNICODE)
_SPACES = re.compile(r"[\s-]+")


def slugify(text: str) -> str:
    """GitHub-style slug: lowercase, non-alphanumerics to hyphens, runs collapsed, trimmed.

    Used only where the source defines no fragment of its own. Where it does — Confluence
    heading anchors, an HTML author's ``id=`` — that one is used instead, because a citation
    that deep-links has to use the address the source publishes.
    """
    stripped = _NOT_SLUGGABLE.sub("", text.strip().lower())
    return _SPACES.sub("-", stripped).strip("-")


@dataclass(slots=True)
class SlugAllocator:
    """Hands out unique fragments, in document order.

    Duplicates get ``-1``, ``-2``… counted from the *second* occurrence, which is what
    Confluence and GitHub both do. Counting from the first would rename the section that was
    there before a duplicate arrived.
    """

    _seen: dict[str, int] = field(default_factory=dict[str, int])

    def allocate(self, text: str) -> str | None:
        """A unique fragment for a heading, or ``None`` when nothing sluggable is left.

        ``None`` rather than a positional fallback like ``section-3``: a fragment is a
        promise that following it lands on this heading, and a number invented here keeps
        that promise only by accident.
        """
        base = slugify(text)
        if not base:
            return None
        count = self._seen.get(base, 0)
        self._seen[base] = count + 1
        return base if count == 0 else f"{base}-{count}"

    def reserve(self, fragment: str) -> None:
        """Record a fragment the source supplied, so a synthesized one cannot collide."""
        self._seen[fragment] = self._seen.get(fragment, 0) + 1


@dataclass(slots=True)
class HeadingStack:
    """The current heading path, maintained as headings arrive in document order.

    Levels need not be contiguous. A document that jumps from ``h1`` to ``h3`` produces a
    two-element path rather than one padded with an empty string, because the padding would
    reach the embedder through the breadcrumb as a heading nobody wrote.
    """

    _entries: list[tuple[int, str]] = field(default_factory=list[tuple[int, str]])

    def push(self, level: int, title: str) -> tuple[str, ...]:
        """Record a heading and return the path *including* it."""
        while self._entries and self._entries[-1][0] >= level:
            self._entries.pop()
        self._entries.append((level, title))
        return self.path

    @property
    def path(self) -> tuple[str, ...]:
        """The path to the current section, outermost first."""
        return tuple(title for _, title in self._entries)

    def reset(self) -> None:
        self._entries.clear()


def heading_anchor(path: Sequence[str], fragment: str | None) -> HeadingAnchor | None:
    """A :class:`HeadingAnchor` for ``path``, or ``None`` when there is no path yet.

    ``None`` is the "text before the first heading" case. A caller turns it into whatever
    anchor its format can honestly produce, rather than inventing a root heading.
    """
    if not path:
        return None
    return HeadingAnchor(path=tuple(path), fragment=fragment)


def decode(raw: RawDocument) -> str:
    """Decode a document's bytes, declining rather than producing replacement characters.

    Raises:
        ParseError: The bytes do not decode. Declining lets the next parser in the chain
            try; indexing mojibake instead would match queries by accident and cite nothing.
    """
    try:
        return raw.as_text()
    except (UnicodeDecodeError, LookupError) as exc:
        msg = (
            f"{raw.uri}: not decodable as {raw.encoding!r} ({exc}). Set the source's encoding, "
            f"or route this media type to a parser that reads bytes."
        )
        raise ParseError(msg) from exc


_TEXT_CONTROL = bytes(range(0x00, 0x09)) + bytes(range(0x0E, 0x20)) + b"\x7f"
_TEXT_CONTROL_ALLOWED = b"\t\n\r\x0b\x0c\x1b"

_MAX_CONTROL_RATIO = 0.05
"""How many unexpected control bytes a text file may contain before it is judged binary.

Above a few per hundred, the file is a container whose bytes happen to decode rather than
text with an odd character in it."""


def is_probably_text(data: bytes) -> bool:
    """Whether ``data`` is text rather than a binary file that happens to reach us.

    The plaintext parser is the global tail of every fallback chain (``docs/parsing.md``
    §6.2), so it is handed everything nothing else claimed. Without this refusal a JPEG
    would be indexed as a page of replacement characters — retrievable, meaningless, and
    citing an image — and ``unsupported_media_type`` would become unreachable because some
    parser always claimed every document.

    A NUL byte is decisive: no text encoding manicule supports produces one, and every
    binary container is full of them.
    """
    if not data:
        return True
    if b"\x00" in data:
        return False
    sample = data[:8192]
    control = sum(
        1 for byte in sample if byte in _TEXT_CONTROL and byte not in _TEXT_CONTROL_ALLOWED
    )
    if control / len(sample) > _MAX_CONTROL_RATIO:
        return False
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError as exc:
        # A truncated multi-byte sequence at the sample boundary is not a binary file.
        return exc.start >= len(sample) - 4
    return True


def lines_of(text: str) -> list[str]:
    """Split into lines the way every anchor in manicule counts them.

    ``str.splitlines`` treats ``U+2028``, ``U+0085`` and friends as line breaks; no editor
    does, and neither does any of the libraries whose row numbers become
    :class:`LineAnchor`. Splitting on ``\\n`` alone keeps this module's idea of a line and
    theirs identical.
    """
    return text.split("\n")


def line_range_of(text: str, start: int, end: int) -> tuple[int, int]:
    """Convert a half-open character range into a 1-based inclusive line range.

    Args:
        text: The whole source.
        start: Character offset of the first character, 0-based.
        end: Character offset one past the last character.

    Returns:
        ``(first_line, last_line)``, both 1-based, the last inclusive.
    """
    first = text.count("\n", 0, start) + 1
    last_char = max(start, end - 1)
    last = text.count("\n", 0, last_char) + 1
    return first, max(first, last)


def resolve_lines(text: str, anchor: LineAnchor) -> str | None:
    """The source lines a :class:`LineAnchor` addresses, or ``None`` if it does not fit.

    ``None`` rather than a clamped range: an anchor naming lines this document does not have
    is an anchor that has diverged from its document, and returning the last few lines
    instead would hide that behind a plausible quotation.
    """
    lines = lines_of(text)
    if anchor.end > len(lines):
        return None
    return "\n".join(lines[anchor.start - 1 : anchor.end])


def merge_line_anchors(anchors: Iterable[LineAnchor]) -> LineAnchor:
    """The smallest :class:`LineAnchor` covering all of ``anchors``.

    ``symbol`` survives only when every anchor agrees on it. A merged chunk covering two
    functions belongs to neither, and naming one of them would put a wrong symbol into the
    breadcrumb, which reaches the embedder.
    """
    collected = list(anchors)
    symbols = {anchor.symbol for anchor in collected}
    return LineAnchor(
        start=min(anchor.start for anchor in collected),
        end=max(anchor.end for anchor in collected),
        symbol=symbols.pop() if len(symbols) == 1 else None,
    )
