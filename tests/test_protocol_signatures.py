"""The signature-conformance check itself.

``@runtime_checkable`` gives false confidence by design: it checks that an attribute exists
and never what it accepts. Every ``isinstance(x, SomeProtocol)`` in this repository is
therefore weaker than it looks, and this is the check that closes the gap.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pytest

from manicule.testing import assert_protocol_signatures


@runtime_checkable
class Example(Protocol):
    """A stand-in with the shape that caused the bug this check exists for."""

    async def search(self, text: str, k: int, filter: str | None = None) -> list[str]: ...  # noqa: A002

    def close(self) -> None: ...


class Faithful:
    async def search(self, text: str, k: int, filter: str | None = None) -> list[str]:  # noqa: A002
        del text, k, filter
        return []

    def close(self) -> None:
        return


class RenamedParameter:
    async def search(self, text_query: str, k: int, filter: str | None = None) -> list[str]:  # noqa: A002
        del text_query, k, filter
        return []

    def close(self) -> None:
        return


class ExtraOptional:
    async def search(
        self,
        text: str,
        k: int,
        filter: str | None = None,  # noqa: A002
        trace: bool = False,
    ) -> list[str]:
        del text, k, filter, trace
        return []

    def close(self) -> None:
        return


class ExtraRequired:
    async def search(
        self,
        text: str,
        k: int,
        filter: str | None = None,  # noqa: A002
        *,
        workspace: str,
    ) -> list[str]:
        del text, k, filter, workspace
        return []

    def close(self) -> None:
        return


class Missing:
    async def search(self, text: str, k: int, filter: str | None = None) -> list[str]:  # noqa: A002
        del text, k, filter
        return []


def test_a_faithful_implementation_passes() -> None:
    """The check must not reject the thing it is meant to admit."""
    assert_protocol_signatures(Faithful(), Example)


def test_a_renamed_parameter_is_caught_even_though_isinstance_passes() -> None:
    """The exact bug: ``text_query`` where the protocol says ``text``.

    ``isinstance`` reports success and the first keyword call fails at run time, possibly in
    a caller written months later.
    """
    assert isinstance(RenamedParameter(), Example), "isinstance cannot see this"
    with pytest.raises(AssertionError, match="isinstance\\(\\) cannot see this"):
        assert_protocol_signatures(RenamedParameter(), Example)


def test_an_extra_optional_parameter_is_allowed() -> None:
    """A caller working from the protocol will never pass it, so it cannot break them."""
    assert_protocol_signatures(ExtraOptional(), Example)


def test_an_extra_required_parameter_is_rejected() -> None:
    """A protocol-shaped call would fail with a missing argument."""
    with pytest.raises(AssertionError, match="extra required parameter"):
        assert_protocol_signatures(ExtraRequired(), Example)


def test_a_missing_method_is_reported_by_name() -> None:
    """Which method is absent is the first thing anyone needs to know."""
    with pytest.raises(AssertionError, match=r"Example\.close is not implemented"):
        assert_protocol_signatures(Missing(), Example)


def test_the_shipped_protocols_are_checkable() -> None:
    """The check has to work against the real protocols, not only a stand-in.

    A fake that satisfies its protocol structurally is the cheapest way to prove that, and
    ``tests/fakes.py`` already maintains those.
    """
    from manicule.core.protocols import (  # noqa: PLC0415
        Chunker,
        Connector,
        Embedder,
        Parser,
        RetrievalStage,
        VectorStore,
    )
    from tests.fakes import (  # noqa: PLC0415
        BlockChunker,
        HashEmbedder,
        LineParser,
        MemoryConnector,
        MemoryVectorStore,
        TopKStage,
    )

    assert_protocol_signatures(LineParser(), Parser)
    assert_protocol_signatures(BlockChunker(), Chunker)
    assert_protocol_signatures(HashEmbedder(), Embedder)
    assert_protocol_signatures(MemoryVectorStore(), VectorStore)
    assert_protocol_signatures(TopKStage(), RetrievalStage)
    assert_protocol_signatures(MemoryConnector(), Connector)


# --- async iterator lifetime ------------------------------------------------------------


async def test_closing_finalizes_a_generator_abandoned_part_way() -> None:
    """A generator suspended at a yield holds whatever it had open until it is finalized.

    Drained bare, that happens at garbage-collection time through the event loop's
    async-generator hook — possibly after the loop has closed. It leaks at best and has been
    observed to crash CPython 3.13 at worst.
    """
    from collections.abc import AsyncIterator  # noqa: PLC0415

    from manicule.testing import closing  # noqa: PLC0415

    released: list[str] = []

    async def source() -> AsyncIterator[int]:
        try:
            for value in range(100):
                yield value
        finally:
            released.append("closed")

    async with closing(source()) as numbers:
        async for value in numbers:
            if value == 2:
                break

    assert released == ["closed"], "the generator must be finalized when the block exits"


async def test_closing_accepts_an_iterator_that_cannot_be_closed() -> None:
    """The protocols declare ``AsyncIterator``, so an implementation may hand back a plain one.

    ``contextlib.aclosing`` would raise on it; narrowing the protocol to ``AsyncGenerator``
    to satisfy the helper would be the tail wagging the dog.
    """
    from manicule.testing import closing  # noqa: PLC0415

    class PlainIterator:
        def __init__(self) -> None:
            self._values = iter((1, 2, 3))

        def __aiter__(self) -> PlainIterator:
            return self

        async def __anext__(self) -> int:
            try:
                return next(self._values)
            except StopIteration as stop:
                raise StopAsyncIteration from stop

    async with closing(PlainIterator()) as numbers:
        collected = [value async for value in numbers]
    assert collected == [1, 2, 3]
