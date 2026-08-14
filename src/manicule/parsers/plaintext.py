"""Plain text, and the refusal that makes the global fallback tail safe.

Paragraphs are runs of non-blank lines, and a block's :class:`~manicule.core.anchors.LineAnchor`
is the run's own line numbers. Nothing here is inferred: the text between two blank lines is a
paragraph because that is what a blank line means in a text file, and the line numbers come
from counting lines.

**This parser is the ``"*"`` tail of every fallback chain** (``docs/parsing.md`` §6.2), so it
is handed every document nothing else claimed — including the ones that are not text at all.
It therefore declines non-text bytes rather than decoding them. Without that refusal a shipped
global tail would index every unrecognized binary as mojibake, so a JPEG would become a page
of replacement characters that matches queries by accident and cites an image, and
``unsupported_media_type`` would become unreachable because some parser would always claim
every document.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

from manicule.core.anchors import Anchor, LineAnchor
from manicule.core.content import BlockKind, ParsedBlock, RawDocument
from manicule.core.errors import ParseError
from manicule.parsers.base import (
    ParserProfile,
    decode,
    is_probably_text,
    lines_of,
    resolve_lines,
)
from manicule.parsers.config import PLAINTEXT_MEDIA_TYPES, PlaintextConfig

__all__ = [
    "PLAINTEXT_MEDIA_TYPES",
    "PlaintextConfig",
    "PlaintextParser",
    "paragraph_spans",
]


class PlaintextParser:
    """Parses a text document into paragraph blocks with real line anchors."""

    media_types = PLAINTEXT_MEDIA_TYPES
    profile = ParserProfile(name="plaintext", max_unlocated_ratio=0.00, max_pagelevel_ratio=None)
    """No unlocated budget at all: a line number is always available for text that exists, so
    an ``Unlocated`` block here would mean the parser stopped counting."""

    def __init__(self, config: PlaintextConfig) -> None:
        self._config = config

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        """Yield one block per paragraph, in reading order.

        A document of nothing but whitespace yields no blocks, which is not an error — it is
        how ``no_extractable_text`` is reached honestly (§6.5).

        Raises:
            ParseError: The bytes are not text, or do not decode. Declining lets the next
                parser in the chain try.
        """
        for block in self._blocks(_text_of(raw)):
            yield block

    async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
        """Return the source lines ``anchor`` addresses, or ``None`` if it addresses none.

        Re-derives everything from ``raw``. Nothing :meth:`parse` computed is consulted: an
        anchor that only resolves against the parser's memory of a document has verified
        nothing about the document.
        """
        if not isinstance(anchor, LineAnchor):
            return None
        return resolve_lines(_text_of(raw), anchor)

    def _blocks(self, text: str) -> Iterator[ParsedBlock]:
        lines = lines_of(text)
        for start, end in paragraph_spans(lines, max_lines=self._config.max_block_lines):
            yield ParsedBlock(
                kind=BlockKind.PROSE,
                text="\n".join(lines[start - 1 : end]),
                anchor=LineAnchor(start=start, end=end),
            )


def paragraph_spans(lines: list[str], *, max_lines: int) -> Iterator[tuple[int, int]]:
    """Yield ``(first, last)`` 1-based inclusive line ranges, one per paragraph.

    A paragraph is a run of lines that are not blank; the blank lines between them belong to
    no block. That is the only division a text file publishes, and inventing another — an
    indentation rule, a heading regular expression — would put structure into the index that
    the document does not contain.

    A run longer than ``max_lines`` is divided at line boundaries into consecutive parts. The
    division is coarse, but each part's span is exactly its own text, which is the property
    every anchor in manicule is held to.

    Args:
        lines: The document, split the way :func:`manicule.parsers.base.lines_of` splits it.
        max_lines: Longest span one range may cover.
    """
    start: int | None = None
    for index, line in enumerate(lines, start=1):
        if line.strip():
            if start is None:
                start = index
            continue
        if start is not None:
            yield from _divided(start, index - 1, max_lines)
            start = None
    if start is not None:
        yield from _divided(start, len(lines), max_lines)


def _divided(start: int, end: int, max_lines: int) -> Iterator[tuple[int, int]]:
    for first in range(start, end + 1, max_lines):
        yield first, min(first + max_lines - 1, end)


def _text_of(raw: RawDocument) -> str:
    """The document as text, declining anything that is not.

    Raises:
        ParseError: The bytes are binary, or do not decode as the declared encoding.
    """
    if not is_probably_text(raw.as_bytes()):
        msg = (
            f"{raw.uri}: declining — these bytes are binary, not text. The plaintext parser "
            f"is the global fallback tail and is offered every unclaimed document, so it "
            f"reads only what is text. Register a parser for this media type, or exclude it "
            f"at the connector."
        )
        raise ParseError(msg)
    return decode(raw)
