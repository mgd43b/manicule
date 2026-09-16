"""The shared bounded decompressor, and the three ways it must refuse.

Two callers read compressed bytes that came from outside the process — a draw.io attachment and
a gzipped sitemap — and ``docs/parsing.md`` §9.3's ruling governs both: bound what the payload
expands *to*, while it expands. The refusals are told apart because only one of them is a
ceiling an operator can raise.
"""

from __future__ import annotations

import zlib

import pytest

from manicule.core.compression import GZIP_WBITS, RAW_DEFLATE_WBITS, inflate
from manicule.core.errors import DecompressionError, DecompressionLimitError


def raw_deflate(data: bytes) -> bytes:
    engine = zlib.compressobj(9, zlib.DEFLATED, RAW_DEFLATE_WBITS)
    return engine.compress(data) + engine.flush()


@pytest.mark.parametrize("size", [1, 2, 65_535, 65_536, 65_537, 131_072, 131_079, 400_000])
def test_a_valid_stream_of_any_size_round_trips(size: int) -> None:
    """The bound is on the total, not on one call's output.

    Capping the output leaves the rest buffered inside the decompressor with every byte of
    input already consumed, so an empty ``unconsumed_tail`` is the ordinary state for anything
    over one chunk. Reading that as truncation refused every payload past about 128 KiB.
    """
    payload = b"ab" * size

    assert inflate(raw_deflate(payload), max_bytes=1 << 30, wbits=RAW_DEFLATE_WBITS, what="p") == (
        payload
    )


def test_expanding_past_the_ceiling_is_refused_with_the_number() -> None:
    """Refused while the memory is still bounded by the ceiling, not after it is spent."""
    with pytest.raises(DecompressionLimitError, match="16-byte ceiling"):
        inflate(raw_deflate(b"x" * 4096), max_bytes=16, wbits=RAW_DEFLATE_WBITS, what="p")


def test_a_truncated_stream_is_refused_rather_than_returned_short() -> None:
    """Returning what arrived would hand a caller half a document and call it whole."""
    blob = raw_deflate(b"y" * 4096)

    with pytest.raises(DecompressionError, match="truncated"):
        inflate(blob[: len(blob) // 2], max_bytes=1 << 30, wbits=RAW_DEFLATE_WBITS, what="p")


def test_unreadable_bytes_are_refused_rather_than_raising_zlib() -> None:
    """A caller catches this module's vocabulary, not :mod:`zlib`'s."""
    with pytest.raises(DecompressionError, match="readable compressed stream"):
        inflate(b"not deflate at all", max_bytes=1 << 30, wbits=RAW_DEFLATE_WBITS, what="p")


def test_a_gzip_member_reads_under_the_gzip_window() -> None:
    """A ``.gz`` sitemap is a gzip member, header and all — a different window from raw deflate."""
    payload = b"<urlset/>" * 100

    assert (
        inflate(
            zlib.compress(payload, wbits=GZIP_WBITS), max_bytes=1 << 20, wbits=GZIP_WBITS, what="p"
        )
        == payload
    )
