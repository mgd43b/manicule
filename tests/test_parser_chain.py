"""The fallback chain, and the four statuses that come out of it.

Every one of these distinctions exists because collapsing it loses information somebody
needs. "The tooling worked and there was nothing to find" and "the tooling broke" call for
different actions — a scanning question and a bug report — and a chain that reports both as
an error makes a corpus of scanned documents look like an indexing outage.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import override

import pytest

from manicule.core.anchors import Anchor, LineAnchor
from manicule.core.content import BlockKind, DocumentStatus, ParsedBlock, PipelineStage, RawDocument
from manicule.core.errors import ConfigError, ParseError
from manicule.parsers.chain import Outcome, ParserChain, container_result

MEDIA_TYPE = "application/pdf"


def raw(media_type: str = MEDIA_TYPE) -> RawDocument:
    return RawDocument(source_id="s", uri="doc", media_type=media_type, content=b"bytes")


class Working:
    """A parser that produces text."""

    media_types = frozenset({MEDIA_TYPE})

    def __init__(self, text: str = "real content") -> None:
        self._text = text

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        del raw
        yield ParsedBlock(kind=BlockKind.PROSE, text=self._text, anchor=LineAnchor(start=1, end=1))

    async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
        del anchor, raw
        return self._text


class Empty(Working):
    """Returns zero text-bearing blocks without raising — a scanned page, an empty file."""

    @override
    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        del raw
        return
        yield  # pragma: no cover - unreachable, and what makes this an async generator


class Whitespace(Working):
    """Produces blocks with nothing in them but whitespace, which is not text."""

    @override
    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        del raw
        yield ParsedBlock(kind=BlockKind.PROSE, text="  \n\f ", anchor=LineAnchor(start=1, end=1))


class Declining(Working):
    """Inspected the input and reported that it is not its kind."""

    @override
    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        msg = f"{raw.uri}: not a PDF"
        raise ParseError(msg)
        yield  # pragma: no cover - unreachable, and what makes this an async generator


class Broken(Working):
    """Raised something that is not a decline — a bug, not a judgement."""

    @override
    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        del raw
        msg = "index out of range"
        raise IndexError(msg)
        yield  # pragma: no cover - unreachable, and what makes this an async generator


class Degraded(Working):
    """Produces text with no usable location at all — and has still succeeded."""

    @override
    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        del raw
        yield ParsedBlock(
            kind=BlockKind.PROSE,
            text="text with a coarse location",
            anchor=LineAnchor(start=1, end=1),
        )


def chain(parsers: dict[str, Working], order: tuple[str, ...]) -> ParserChain:
    return ParserChain(parsers=dict(parsers), chains={MEDIA_TYPE: order})


# --- what advances the chain ---------------------------------------------------------------


async def test_empty_output_advances_so_a_second_parser_gets_a_turn() -> None:
    """This is the entire purpose of putting a heavier parser behind a fast one."""
    result = await chain({"fast": Empty(), "thorough": Working()}, ("fast", "thorough")).run(raw())
    assert result.status is DocumentStatus.PARSED
    assert result.parser_used == "thorough"
    assert [attempt.outcome for attempt in result.attempts] == [Outcome.EMPTY, Outcome.PARSED]


async def test_a_decline_advances_and_is_recorded_as_a_decline_not_a_failure() -> None:
    """A parser that declined is reporting that the document is not its kind, which is
    information. One that raised is reporting that something broke."""
    result = await chain({"a": Declining(), "b": Working()}, ("a", "b")).run(raw())
    assert result.status is DocumentStatus.PARSED
    assert result.attempts[0].outcome is Outcome.DECLINED


async def test_degraded_output_does_not_advance_the_chain() -> None:
    """A parser that produced text but coarse locations **has succeeded**.

    Falling back on quality grounds makes the chain non-deterministic, makes results depend
    on thresholds nobody tuned, and doubles parse cost on exactly the documents that are
    already slow. If a parser's quality is unacceptable, reorder the chain.
    """
    result = await chain({"a": Degraded(), "b": Working()}, ("a", "b")).run(raw())
    assert result.parser_used == "a"
    assert len(result.attempts) == 1


# --- the statuses ----------------------------------------------------------------------------


async def test_a_chain_that_finds_nothing_anywhere_is_no_extractable_text() -> None:
    """Never an empty document that looks successfully indexed.

    Optical character recognition is out of scope, and this status is what keeps that
    decision visible instead of silent — it is also the selector for the re-parse pass the
    day the decision is revisited.
    """
    result = await chain({"a": Empty(), "b": Whitespace()}, ("a", "b")).run(raw())
    assert result.status is DocumentStatus.NO_EXTRACTABLE_TEXT
    assert result.failed_stage is None
    assert "no text" in result.status_detail


async def test_whitespace_only_output_counts_as_no_text() -> None:
    """A statement about content, not about how many objects a parser happened to yield."""
    result = await chain({"a": Whitespace()}, ("a",)).run(raw())
    assert result.status is DocumentStatus.NO_EXTRACTABLE_TEXT


async def test_a_chain_where_everything_declined_is_an_unsupported_media_type() -> None:
    result = await chain({"a": Declining(), "b": Declining()}, ("a", "b")).run(raw())
    assert result.status is DocumentStatus.UNSUPPORTED_MEDIA_TYPE


async def test_a_chain_where_a_parser_broke_is_failed_at_the_parse_stage() -> None:
    result = await chain({"a": Broken()}, ("a",)).run(raw())
    assert result.status is DocumentStatus.FAILED
    assert result.failed_stage is PipelineStage.PARSE
    assert "IndexError" in result.status_detail


async def test_one_broken_parser_and_one_empty_one_is_failed_not_no_extractable_text() -> None:
    """The mixed case, which an implementation guesses at.

    A parser that broke leaves us genuinely not knowing whether text was there.
    ``no_extractable_text`` means something specific — the tooling worked and there was
    nothing to find — and widening it to cover "something broke and the rest found nothing"
    would make the scanned-corpus warning fire on library bugs.
    """
    result = await chain({"a": Broken(), "b": Empty()}, ("a", "b")).run(raw())
    assert result.status is DocumentStatus.FAILED
    assert result.failed_stage is PipelineStage.PARSE


async def test_a_container_is_its_own_status_rather_than_an_empty_document() -> None:
    """Nothing failed and nothing was missing. Conflating an archive with a document that
    yielded no text would put every archive into the bucket that triggers the scanned-corpus
    warning, which is how a diagnostic stops meaning anything."""
    result = container_result(members=4)
    assert result.status is DocumentStatus.CONTAINER
    assert result.blocks == []
    assert "4 member" in result.status_detail


# --- resolution ------------------------------------------------------------------------------


def test_a_named_parser_that_is_not_installed_is_a_startup_error() -> None:
    """A chain whose behaviour depends on what happens to be installed indexes the same
    document differently on different machines — the same hazard the OCR decision avoids."""
    with pytest.raises(ConfigError, match="docling"):
        ParserChain(parsers={"pdf": Working()}, chains={MEDIA_TYPE: ("pdf", "docling")})


def test_the_wildcard_supplies_a_tail_appended_to_every_chain() -> None:
    """Shipping a plaintext tail means an unknown text-ish file is indexed with real line
    anchors rather than skipped."""
    resolved = ParserChain(
        parsers={"pdf": Working(), "plaintext": Working()},
        chains={MEDIA_TYPE: ("pdf",), "*": ("plaintext",)},
    ).resolve(MEDIA_TYPE)
    assert resolved == ("pdf", "plaintext")


def test_a_parser_already_in_a_chain_is_not_repeated_by_the_tail() -> None:
    resolved = ParserChain(
        parsers={"plaintext": Working()},
        chains={"text/plain": ("plaintext",), "*": ("plaintext",)},
    ).resolve("text/plain")
    assert resolved == ("plaintext",)


def test_media_type_parameters_do_not_defeat_chain_lookup() -> None:
    """A chain that matched only when the charset happened to be spelled the same way would
    look configured and silently not apply."""
    resolved = ParserChain(parsers={"html": Working()}, chains={"text/html": ("html",)}).resolve(
        "text/HTML; charset=utf-8"
    )
    assert resolved == ("html",)


async def test_a_media_type_with_no_chain_at_all_is_unsupported_rather_than_failed() -> None:
    result = await ParserChain(parsers={}, chains={}).run(raw("image/png"))
    assert result.status is DocumentStatus.UNSUPPORTED_MEDIA_TYPE


async def test_every_attempt_is_recorded_so_a_result_can_be_explained_later() -> None:
    """Falling back is not a status; it is metadata. Status stays coarse enough to filter on
    while the detail remains available to diagnostics."""
    result = await chain({"a": Declining(), "b": Empty(), "c": Working()}, ("a", "b", "c")).run(
        raw()
    )
    attempted = result.metadata["parsers_attempted"]
    assert isinstance(attempted, list)
    assert len(attempted) == 3
    assert result.metadata["parser_used"] == "c"
