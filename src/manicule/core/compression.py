"""Expanding a compressed payload against a ceiling, enforced while it expands.

Two places in this codebase read compressed bytes that came from outside it — a draw.io
attachment's deflate stream, and a gzipped sitemap — and both need the same thing:
``docs/parsing.md`` §9.3's ruling that a compressed payload needs a bound on what it expands
*to*, not on what it arrived as. A declared size is a field the sender wrote, and by the time
``len()`` can be taken the memory is already spent.

Stdlib only, so this can live in ``core`` and be reached from a parser and a connector alike
without either importing the other.
"""

from __future__ import annotations

import zlib
from typing import Final

from manicule.core.errors import DecompressionError, DecompressionLimitError

_CHUNK: Final = 64 * 1024
"""How much output one ``decompress`` call may produce before the total is re-checked."""

GZIP_WBITS: Final = 16 + zlib.MAX_WBITS
"""A gzip member, header and all — what a ``.gz`` file holds."""

RAW_DEFLATE_WBITS: Final = -zlib.MAX_WBITS
"""A bare deflate stream with no wrapper — what ``encodeURIComponent``-then-deflate produces."""


def inflate(data: bytes, *, max_bytes: int, wbits: int, what: str) -> bytes:
    """Expand ``data``, refusing the moment it passes ``max_bytes``.

    Args:
        data: The compressed bytes.
        max_bytes: Most bytes the result may reach. Checked as output accumulates rather than
            after, which is the whole point.
        wbits: :data:`GZIP_WBITS` or :data:`RAW_DEFLATE_WBITS`, or any value :mod:`zlib`
            accepts.
        what: Named in the refusal, so an operator is told which payload was refused.

    Raises:
        DecompressionLimitError: The output passed ``max_bytes``.
        DecompressionError: The stream is unreadable, ends early, or stops making progress.

    Three ways this ends badly and they are told apart, because only the first is a ceiling an
    operator can raise:

    **Past the ceiling.** Refused with the number, while the memory is still bounded by it.

    **Truncated.** An empty ``unconsumed_tail`` alone does not mean the input ran out — capping
    the *output* leaves the rest buffered inside the decompressor with every byte of input
    already consumed, which is the ordinary state for any stream larger than one chunk. Reading
    the first half of that pair as the whole of it refuses every payload over about 128 KiB.

    **Stalled.** No output and no input consumed: the tail never shrinks and the total never
    grows, so neither of the checks above can ever fire again. Without this the loop is the one
    unbounded thing in a function whose entire job is bounding untrusted input.
    """
    engine = zlib.decompressobj(wbits)
    out = bytearray()
    pending = data
    while True:
        try:
            part = engine.decompress(pending, _CHUNK)
        except zlib.error as exc:
            msg = f"{what} does not hold a readable compressed stream ({exc})"
            raise DecompressionError(msg) from exc
        out += part
        if len(out) > max_bytes:
            msg = f"{what} expands past the {max_bytes}-byte ceiling"
            raise DecompressionLimitError(msg)
        if engine.eof:
            return bytes(out)
        if not part and engine.unconsumed_tail == pending:
            msg = f"{what} stopped producing output before its stream ended"
            raise DecompressionError(msg)
        pending = engine.unconsumed_tail
        if not pending and not part:
            msg = f"{what} holds a truncated compressed stream"
            raise DecompressionError(msg)


__all__ = ["GZIP_WBITS", "RAW_DEFLATE_WBITS", "inflate"]
