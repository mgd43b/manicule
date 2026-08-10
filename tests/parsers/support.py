"""Shared scaffolding for the parser suites.

One place builds the documents and one place runs the checks, so that a parser cannot pass by
being tested more gently than its neighbours.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from manicule.chunking import StructuralChunker, TokenCounter
from manicule.core.content import Document, DocumentStatus, Metadata, RawDocument
from manicule.core.ids import content_hash, document_id
from manicule.core.protocols import Chunker, Parser
from manicule.testing import (
    ParserProfile,
    RoundTripReport,
    assert_location_budget,
    assert_parser_contract,
    assert_round_trip,
)

TOKENIZER_ID = "test/whitespace"
"""Identity of the counter below.

Real ingest counts with the embedder's own tokenizer. A test counts with something
deterministic and cheap, and records *which* — because a chunk budget agreed under one
vocabulary means something else under another, and a suite that hid that would be measuring
its own arithmetic.
"""


def count_tokens(text: str) -> int:
    """A deterministic stand-in tokenizer: whitespace-separated words.

    Not provisional. Provisional counting inflates by a safety factor and marks its chunks
    for ingest to refuse; a fixture suite wants exact, repeatable boundaries, so it declares
    a tokenizer identity of its own instead.
    """
    return len(text.split())


def make_counter() -> TokenCounter:
    return TokenCounter(TOKENIZER_ID, count_tokens, provisional=False)


def make_chunker(**overrides: object) -> StructuralChunker:
    """The chunker every parser suite chunks with: 512 tokens, 64 overlap."""
    return StructuralChunker(make_counter(), **overrides)  # pyright: ignore[reportArgumentType] - test-only keyword pass-through


def raw_of(
    content: bytes | str,
    media_type: str,
    *,
    uri: str = "fixture",
    encoding: str = "utf-8",
    title: str = "",
) -> RawDocument:
    """A :class:`RawDocument` from literal content."""
    metadata: Metadata = {"title": title} if title else {}
    return RawDocument(
        source_id=uri,
        uri=uri,
        media_type=media_type,
        content=content,
        encoding=encoding,
        metadata=metadata,
    )


def raw_from(path: Path, media_type: str, *, encoding: str = "utf-8") -> RawDocument:
    """A :class:`RawDocument` from a fixture file, titled after its stem."""
    return raw_of(
        path.read_bytes(),
        media_type,
        uri=path.name,
        encoding=encoding,
        title=path.stem.replace("-", " "),
    )


def document_for(raw: RawDocument, *, title: str = "") -> Document:
    """The stored record a parser's blocks would be chunked against."""
    resolved = title or _title_of(raw)
    return Document(
        id=document_id("fixtures", raw.source_id),
        source="fixtures",
        source_id=raw.source_id,
        uri=raw.uri,
        title=resolved,
        content_hash=content_hash(raw.as_bytes()),
        media_type=raw.media_type,
        status=DocumentStatus.PARSED,
    )


def _title_of(raw: RawDocument) -> str:
    value = raw.metadata.get("title")
    return value if isinstance(value, str) else ""


async def check_fixture(
    parser: Parser, raw: RawDocument, *, chunker: Chunker | None = None
) -> RoundTripReport:
    """Run every per-fixture obligation against one document.

    Both suites, not one: :func:`assert_parser_contract` is the shipped promise a third-party
    parser is held to, and :func:`assert_round_trip` is the six assertions. Running only the
    first would let an anchor pointing at the whole document through.
    """
    await assert_parser_contract(parser, raw)
    return await assert_round_trip(
        parser,
        raw,
        fixture=raw.uri,
        chunker=chunker,
        document=document_for(raw) if chunker is not None else None,
    )


async def check_corpus(
    parser: Parser,
    raws: Sequence[RawDocument],
    *,
    chunker: Chunker | None = None,
    min_blocks: int = 0,
) -> list[RoundTripReport]:
    """Run every fixture, then the corpus-level location budget.

    The budget is the assertion that stops a parser satisfying the other five by never
    producing a location. It is asserted per parser against that parser's own corpus, never
    pooled — pooling lets a well-behaved parser's fixtures pay for a badly-behaved one's.
    """
    reports = [await check_fixture(parser, raw, chunker=chunker) for raw in raws]
    assert_location_budget(_profile_of(parser), reports, blocks=min_blocks)
    return reports


def _profile_of(parser: Parser) -> ParserProfile:
    profile = getattr(parser, "profile", None)
    if not isinstance(profile, ParserProfile):
        message = (
            f"{type(parser).__name__} declares no ParserProfile. Both location ratios are "
            f"declared by the parser and enforced by the harness; without them 'a citation "
            f"carries a correct location, or none' is satisfied by never carrying one"
        )
        raise AssertionError(message)  # noqa: TRY004 - a failed check, not a type error
    return profile


__all__ = [
    "TOKENIZER_ID",
    "check_corpus",
    "check_fixture",
    "count_tokens",
    "document_for",
    "make_chunker",
    "make_counter",
    "raw_from",
    "raw_of",
]
