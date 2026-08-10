"""Closing a parser's block stream, on every path including the early ones.

``Parser.parse`` returns an ``AsyncIterator``, and every parser here implements it as an
async generator. A generator abandoned part-way stays suspended holding whatever it had open
at the ``yield`` — a document handle, a native text page, a decompression stream — and
CPython finalises it later, through the event loop that created it. When that loop has
already closed, the finalisation runs against a torn-down runtime, and the symptom is not a
warning: it is a crash inside the interpreter's allocator, on a stack naming no library
anybody here wrote.

So iteration goes through :func:`manicule.core.protocols.parsing`, which closes in a
``finally``, and parsers hold their resources in ``try``/``finally`` around the ``yield``.
Both halves are checked below, each against a fake that omits it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import override

import pytest

from manicule.core.anchors import Anchor, LineAnchor
from manicule.core.content import BlockKind, ParsedBlock, RawDocument
from manicule.core.protocols import aclose, parsing, read_blocks
from tests.parsers.support import raw_from, raw_of

MEDIA_TYPE = "text/plain"


@dataclass
class Ledger:
    """Counts resources taken and given back, so a leak is a number rather than a hunch."""

    opened: int = 0
    closed: int = 0
    events: list[str] = field(default_factory=list[str])

    @property
    def leaked(self) -> int:
        return self.opened - self.closed


class TidyParser:
    """Releases what it holds in a ``finally``, so ``aclose`` cannot strand it."""

    media_types = frozenset({MEDIA_TYPE})

    def __init__(self, ledger: Ledger) -> None:
        self._ledger = ledger

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        self._ledger.opened += 1
        try:
            for number, line in enumerate(raw.as_text().splitlines(), start=1):
                yield ParsedBlock(
                    kind=BlockKind.PROSE, text=line, anchor=LineAnchor(start=number, end=number)
                )
        finally:
            # aclose() throws GeneratorExit in at the suspension point, so only a finally
            # runs after it. A release placed after the loop never happens on an early close.
            self._ledger.closed += 1

    async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
        del raw
        return "" if isinstance(anchor, LineAnchor) else None


class LeakyParser(TidyParser):
    """Releases only after its last block — which an early close never reaches."""

    @override
    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        self._ledger.opened += 1
        for number, line in enumerate(raw.as_text().splitlines(), start=1):
            yield ParsedBlock(
                kind=BlockKind.PROSE, text=line, anchor=LineAnchor(start=number, end=number)
            )
        self._ledger.closed += 1


def _raw() -> RawDocument:
    return raw_of("alpha\nbeta\ngamma\ndelta", MEDIA_TYPE)


async def test_a_stream_stopped_after_one_block_still_releases_what_it_held() -> None:
    """The path that strands a generator: a consumer that stops early."""
    ledger = Ledger()
    async with parsing(TidyParser(ledger), _raw()) as blocks:
        async for _ in blocks:
            break
    assert ledger.leaked == 0


async def test_a_parser_that_releases_after_its_last_block_is_caught_by_that_test() -> None:
    """Proof the check above is load-bearing rather than decorative.

    A release placed after the loop looks correct and is correct on a full drain, which is
    every ordinary run. It is only wrong on the path that is hard to notice.
    """
    ledger = Ledger()
    async with parsing(LeakyParser(ledger), _raw()) as blocks:
        async for _ in blocks:
            break
    assert ledger.leaked == 1


class ConsumerError(RuntimeError):
    """Stands in for an assertion failing between two blocks."""


async def _fail_after_one_block(ledger: Ledger) -> None:
    async with parsing(TidyParser(ledger), _raw()) as blocks:
        async for _ in blocks:
            raise ConsumerError


async def test_an_exception_in_the_loop_body_still_closes_the_stream() -> None:
    """A failing assertion between blocks is an early exit like any other."""
    ledger = Ledger()
    with pytest.raises(ConsumerError):
        await _fail_after_one_block(ledger)
    assert ledger.leaked == 0


async def test_a_full_drain_closes_the_stream_too() -> None:
    ledger = Ledger()
    blocks = await read_blocks(TidyParser(ledger), _raw())
    assert len(blocks) == 4
    assert ledger.leaked == 0


async def test_closing_a_stream_that_is_not_a_generator_is_not_an_error() -> None:
    """``parse`` promises an ``AsyncIterator``, which is weaker than an async generator.

    A parser returning a hand-written iterator satisfies the protocol, so the close asks
    whether the stream can be closed rather than assuming it can.
    """

    class Handwritten:
        def __init__(self) -> None:
            self._left = 1

        def __aiter__(self) -> Handwritten:
            return self

        async def __anext__(self) -> ParsedBlock:
            if self._left <= 0:
                raise StopAsyncIteration
            self._left -= 1
            return ParsedBlock(kind=BlockKind.PROSE, text="only", anchor=LineAnchor(start=1, end=1))

    await aclose(Handwritten())


async def test_the_pdf_parser_releases_its_document_when_closed_part_way(corpus: Path) -> None:
    """The case that matters most: a generator holding a native handle across a ``yield``.

    A PDF document left open by a stranded generator is closed by the interpreter at some
    later moment of its choosing, from whichever loop is running then. Closing deterministically
    is what keeps that out of the runtime's hands, and re-parsing afterwards proves nothing
    was left in a state the next read trips over.
    """
    raw = raw_from(corpus / "pdf" / "typical.pdf", "application/pdf")
    from manicule.parsers.pdf import PdfConfig, PdfParser  # noqa: PLC0415 - heavy import

    parser = PdfParser(PdfConfig())
    async with parsing(parser, raw) as blocks:
        async for _ in blocks:
            break
    assert len(await read_blocks(parser, raw)) > 0
