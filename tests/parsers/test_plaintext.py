"""The plain-text parser, and the refusal that makes the global fallback tail safe."""

from __future__ import annotations

import itertools
from collections.abc import AsyncIterator
from pathlib import Path
from typing import override

import pytest

from manicule.chunking import StructuralChunker
from manicule.core.anchors import LineAnchor
from manicule.core.content import ParsedBlock, RawDocument
from manicule.core.errors import ParseError
from manicule.core.protocols import aclose, parsing, read_blocks
from manicule.parsers.plaintext import PlaintextConfig, PlaintextParser
from manicule.testing import assert_round_trip
from tests.parsers.support import check_corpus, check_fixture, document_for, raw_from, raw_of

READABLE = (
    "typical.txt",
    "changelog.txt",
    "hard-wrapped.txt",
    "no-trailing-newline.txt",
    "whitespace-only.txt",
    "empty.txt",
)
"""Fixtures the parser accepts. The two it declines are tested by name below, because a
refusal is a result and not a gap in coverage."""


@pytest.fixture
def parser() -> PlaintextParser:
    return PlaintextParser(PlaintextConfig())


def _raws(corpus: Path) -> list[RawDocument]:
    return [raw_from(corpus / "plaintext" / name, "text/plain") for name in READABLE]


async def test_every_readable_fixture_round_trips_within_its_declared_location_budget(
    parser: PlaintextParser, corpus: Path, chunker: StructuralChunker
) -> None:
    """Every block cites lines that contain it, and no block gives up on a location.

    This parser declares ``max_unlocated_ratio = 0.00``: a line number is always available for
    text that exists, so an ``Unlocated`` block here would mean the parser stopped counting.
    """
    reports = await check_corpus(parser, _raws(corpus), chunker=chunker, min_blocks=20)
    assert sum(report.unlocated for report in reports) == 0


async def test_binary_bytes_are_declined_so_a_shipped_fallback_tail_cannot_index_a_jpeg(
    parser: PlaintextParser, corpus: Path
) -> None:
    """The refusal the ``"*"`` chain entry depends on.

    Without it a binary would be indexed as a page of replacement characters that matches
    queries by accident and cites an image, and ``unsupported_media_type`` would be
    unreachable because some parser would always claim every document.
    """
    raw = raw_from(corpus / "plaintext" / "not-text.bin", "text/plain")
    with pytest.raises(ParseError, match="binary, not text"):
        await read_blocks(parser, raw)


async def test_bytes_that_do_not_decode_are_declined_rather_than_indexed_as_mojibake(
    parser: PlaintextParser, corpus: Path
) -> None:
    """A file with an invalid UTF-8 sequence is somebody else's, or nobody's.

    Decoding it with replacement characters would put text into the index that no source
    contains, under a citation claiming it is a quotation.
    """
    raw = raw_from(corpus / "plaintext" / "malformed-utf8.txt", "text/plain")
    with pytest.raises(ParseError):
        await read_blocks(parser, raw)


async def test_a_declined_document_says_it_is_declining_rather_than_that_it_broke(
    parser: PlaintextParser, corpus: Path
) -> None:
    """Declining and failing end in different document statuses (§6.3).

    A parser that reports "this is not mine" leaves the chain able to end in
    ``unsupported_media_type``; one that reports a breakage makes the document ``failed``. The
    two are told apart by what the message says, so the message is part of the contract.
    """
    raw = raw_from(corpus / "plaintext" / "not-text.bin", "text/plain")
    with pytest.raises(ParseError) as caught:
        await read_blocks(parser, raw)
    assert "declining" in str(caught.value)


async def test_a_file_of_only_whitespace_yields_no_blocks_rather_than_an_empty_one(
    parser: PlaintextParser, corpus: Path
) -> None:
    """Zero blocks is how ``no_extractable_text`` is reached honestly.

    A single block of whitespace would make the document look parsed, put a vector of nothing
    into the index, and consume a slot in every result list.
    """
    raw = raw_from(corpus / "plaintext" / "whitespace-only.txt", "text/plain")
    assert await read_blocks(parser, raw) == []


async def test_zero_bytes_yields_no_blocks_and_does_not_raise(
    parser: PlaintextParser, corpus: Path
) -> None:
    """An empty document is a legitimate outcome, not a failure.

    Raising here would make an empty file ``failed``, which says the tooling broke when in
    fact it worked and there was nothing to find.
    """
    raw = raw_from(corpus / "plaintext" / "empty.txt", "text/plain")
    assert await read_blocks(parser, raw) == []


async def test_a_last_line_with_no_trailing_newline_is_still_in_its_block(
    parser: PlaintextParser, corpus: Path
) -> None:
    """The end of a file without a final newline is content, not a delimiter.

    Splitting on newlines and discarding the remainder loses the last paragraph of every file
    an editor did not terminate, which is a whole class of file.
    """
    raw = raw_from(corpus / "plaintext" / "no-trailing-newline.txt", "text/plain")
    blocks = await read_blocks(parser, raw)
    assert blocks[-1].text.endswith("no empty element after it.")


async def test_a_run_longer_than_the_block_budget_splits_into_parts_that_each_own_their_span(
    corpus: Path,
) -> None:
    """A hard-wrapped paragraph does not become one block the chunker would have to split.

    When a block exceeds the chunk budget the chunker splits its *text* and every part keeps
    the *block's* anchor, so each part would resolve to the whole run to address a fraction of
    it. Splitting here gives every part a span that is exactly its own text.
    """
    parser = PlaintextParser(PlaintextConfig(max_block_lines=20))
    raw = raw_from(corpus / "plaintext" / "hard-wrapped.txt", "text/plain")
    blocks = await read_blocks(parser, raw)
    spans = [block.anchor for block in blocks]
    assert len(blocks) > 1
    assert all(isinstance(span, LineAnchor) and span.end - span.start < 20 for span in spans)
    for earlier, later in itertools.pairwise(spans):
        assert isinstance(earlier, LineAnchor)
        assert isinstance(later, LineAnchor)
        assert later.start == earlier.end + 1


async def test_resolve_reads_the_document_rather_than_anything_parse_left_behind(
    parser: PlaintextParser,
) -> None:
    """An anchor that only resolves against the parser's memory verifies nothing.

    Resolving before parsing has ever run must return the same text as resolving after, or the
    round-trip check is measuring the parser against itself.
    """
    raw = raw_of("first line\n\nsecond paragraph\n", "text/plain")
    anchor = LineAnchor(start=3, end=3)
    assert await parser.resolve(anchor, raw) == "second paragraph"


async def test_an_anchor_one_line_out_is_caught_by_the_round_trip_check(
    corpus: Path, chunker: StructuralChunker
) -> None:
    """The guard is load-bearing, proved by breaking it.

    A parser whose anchors are one line late produces citations that are well-formed, resolve
    to real text, and are wrong. Nothing raises; only this assertion fires.
    """
    raw = raw_from(corpus / "plaintext" / "typical.txt", "text/plain")
    with pytest.raises(AssertionError):
        await assert_round_trip(
            _OffByOneParser(PlaintextConfig()),
            raw,
            fixture="off-by-one",
            chunker=chunker,
            document=document_for(raw),
        )


async def test_the_large_fixture_round_trips_at_block_level(
    parser: PlaintextParser, corpus: Path
) -> None:
    """The generated large document exercises the path a real corpus takes.

    Checked at block level: the chunk-level obligation for a document this size is the
    chunker's overlap behavior rather than the parser's, and it is covered where it belongs.
    """
    raw = raw_from(corpus / "plaintext" / "plaintext-large.txt", "text/plain")
    report = await check_fixture(parser, raw)
    assert report.blocks > 1000
    assert report.unlocated == 0


class _OffByOneParser(PlaintextParser):
    """Anchors every block one line further down than the text it claims.

    The exact defect the round-trip contract exists to catch, and the reason a fixture corpus
    is not enough on its own: without a parser that is known to be wrong, a suite that passes
    is evidence only that nothing was checked.
    """

    @override
    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        async for block in super().parse(raw):
            anchor = block.anchor
            if not isinstance(anchor, LineAnchor):  # pragma: no cover - always a LineAnchor
                yield block
                continue
            moved = LineAnchor(start=anchor.start + 1, end=anchor.end + 1, symbol=anchor.symbol)
            yield block.model_copy(update={"anchor": moved})


async def test_a_block_stream_stopped_after_one_block_is_not_left_suspended(
    parser: PlaintextParser, corpus: Path
) -> None:
    """A consumer that stops early must not strand the generator.

    An abandoned async generator stays suspended until CPython finalizes it through the loop
    that created it — and if that loop has closed, the finalization runs against a torn-down
    runtime and crashes inside the allocator, on a stack naming nothing anybody here wrote.
    """
    raw = raw_from(corpus / "plaintext" / "typical.txt", "text/plain")
    stream = parser.parse(raw)
    async for _ in stream:
        break
    await aclose(stream)
    assert await _is_closed(stream)


async def test_an_exception_between_blocks_still_closes_the_stream(
    parser: PlaintextParser, corpus: Path
) -> None:
    """A failing assertion mid-iteration is an early exit like any other, and the common one."""
    raw = raw_from(corpus / "plaintext" / "typical.txt", "text/plain")
    seen: list[AsyncIterator[ParsedBlock]] = []
    with pytest.raises(_StopEarlyError):
        await _fail_between_blocks(parser, raw, seen)
    assert await _is_closed(seen[0])


class _StopEarlyError(RuntimeError):
    """Stands in for an assertion failing between two blocks."""


async def _fail_between_blocks(
    parser: PlaintextParser, raw: RawDocument, seen: list[AsyncIterator[ParsedBlock]]
) -> None:
    async with parsing(parser, raw) as blocks:
        seen.append(blocks)
        async for _ in blocks:
            raise _StopEarlyError


async def _is_closed(stream: AsyncIterator[object]) -> bool:
    """Whether a stream is finished with, rather than merely paused.

    A generator that has been closed raises ``StopAsyncIteration`` at once instead of resuming
    where it left off, which is the observable difference between "released" and "suspended
    holding whatever it had open".
    """
    try:
        await anext(stream)
    except StopAsyncIteration:
        return True
    return False
