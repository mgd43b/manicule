"""The structured-data parser: JSON, YAML and TOML, anchored to real source lines."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast, override

import pytest

from manicule.chunking import StructuralChunker
from manicule.core.anchors import LineAnchor, Unlocated
from manicule.core.content import ParsedBlock, RawDocument
from manicule.core.errors import ParseError
from manicule.core.protocols import aclose, read_blocks
from manicule.parsers.structured import (
    NO_JSON_POSITIONS,
    StructuredConfig,
    StructuredParser,
    read_yaml,
)
from manicule.testing import assert_round_trip
from tests.parsers.support import check_corpus, check_fixture, document_for, raw_from, raw_of

MEDIA_TYPES = {
    ".json": "application/json",
    ".toml": "application/toml",
    ".yaml": "application/yaml",
}

READABLE = (
    "typical.yaml",
    "typical.json",
    "typical.toml",
    "norway.yaml",
    "nested.yaml",
    "multi-document.yaml",
    "deep.json",
    "compact.json",
    "duplicate-keys.json",
    "nan.json",
    "flat.toml",
    "tricky.toml",
    "empty.yaml",
    "empty.json",
)
"""Everything the parser reads. The three it refuses are named in their own tests."""


@pytest.fixture
def parser() -> StructuredParser:
    return StructuredParser(StructuredConfig())


def _raw(corpus: Path, name: str) -> RawDocument:
    path = corpus / "structured" / name
    return raw_from(path, MEDIA_TYPES[path.suffix])


async def _blocks(parser: StructuredParser, raw: RawDocument) -> list[ParsedBlock]:
    """Drained through ``read_blocks``, which closes the stream in a ``finally``."""
    return await read_blocks(parser, raw)


async def test_every_readable_fixture_round_trips_within_its_declared_location_budget(
    parser: StructuredParser, corpus: Path, chunker: StructuralChunker
) -> None:
    """Blocks are exact slices, so every anchor resolves to precisely the block's own text.

    The corpus deliberately contains the two JSON files whose positions cannot be recovered,
    so the unlocated budget is measured against real declines rather than against none.
    """
    reports = await check_corpus(
        parser, [_raw(corpus, name) for name in READABLE], chunker=chunker, min_blocks=25
    )
    assert sum(report.unlocated for report in reports) == 2


async def test_a_block_is_an_exact_slice_of_the_source_rather_than_a_reprinted_value(
    parser: StructuredParser, corpus: Path
) -> None:
    """The property that makes the anchor correct by construction.

    Pretty-printing the parsed values and hunting for them in the source would produce anchors
    that drift the moment the file's formatting differs from the printer's — and drift
    silently, because the reprinted text still looks like the document.
    """
    raw = _raw(corpus, "typical.yaml")
    lines = raw.as_text().split("\n")
    blocks = await _blocks(parser, raw)
    assert blocks, "no blocks, so the slice comparison below never ran"
    for block in blocks:
        assert isinstance(block.anchor, LineAnchor)
        assert block.text == "\n".join(lines[block.anchor.start - 1 : block.anchor.end])


# --- YAML ---------------------------------------------------------------------------------


async def test_unquoted_no_and_yes_are_read_as_the_words_the_document_contains(
    parser: StructuredParser, corpus: Path
) -> None:
    """The Norway problem, which is why this parser reads YAML 1.2 and not 1.1.

    Under 1.1 the keys ``no``, ``off``, ``yes`` and ``on`` resolve to booleans, so the symbol
    on a country-code block would read ``False`` where the source plainly says ``no``. That is
    a citation reproducing something the document does not say.
    """
    raw = _raw(corpus, "norway.yaml")
    blocks = await _blocks(parser, raw)
    symbols = [block.anchor.symbol for block in blocks if isinstance(block.anchor, LineAnchor)]
    assert symbols == ["no", "off", "yes", "on"]

    document = read_yaml(raw.as_text())[0]
    assert isinstance(document, dict)
    keys: list[object] = list(cast("dict[object, object]", document))
    assert keys == ["no", "off", "yes", "on"]
    assert all(isinstance(key, str) for key in keys)


async def test_a_multi_document_stream_is_divided_by_every_document_s_own_keys(
    parser: StructuredParser, corpus: Path
) -> None:
    """A ``.yaml`` holding several documents is one file with one line numbering.

    Treating the stream as a single opaque value would give a Kubernetes manifest one block
    covering everything, and a citation into it would name the file rather than the object.
    """
    blocks = await _blocks(parser, _raw(corpus, "multi-document.yaml"))
    symbols = [block.anchor.symbol for block in blocks if isinstance(block.anchor, LineAnchor)]
    assert symbols.count("kind") == 2
    assert "stringData" in symbols


async def test_yaml_that_does_not_parse_reports_a_failure_rather_than_a_decline(
    parser: StructuredParser, corpus: Path
) -> None:
    """A broken document is not "not ours", and the two end in different statuses (§6.3).

    An unterminated quoted scalar — what a copied spreadsheet cell arrives as — means the
    document is malformed, so the message says the document is invalid rather than that this
    parser does not handle it.
    """
    with pytest.raises(ParseError, match="not valid YAML"):
        await _blocks(parser, _raw(corpus, "unterminated-quote.yaml"))


# --- JSON ---------------------------------------------------------------------------------


async def test_a_compact_object_becomes_one_block_because_a_line_range_has_no_columns(
    parser: StructuredParser, corpus: Path
) -> None:
    """Every key on one line means one block, not one block per key.

    A block per key would give them all the same ``LineAnchor`` while claiming different text,
    so each would resolve to the whole line — a citation quoting its neighbors.
    """
    blocks = await _blocks(parser, _raw(corpus, "compact.json"))
    assert len(blocks) == 1
    assert blocks[0].anchor == LineAnchor(start=1, end=1)


async def test_duplicate_keys_keep_the_document_and_lose_only_its_positions(
    parser: StructuredParser, corpus: Path
) -> None:
    """Legal JSON that the mark-bearing reader refuses.

    The last value wins and the file is perfectly valid, so refusing it would drop real
    content. Indexing it with invented positions would be worse. The document is kept and the
    decline is named, which is what makes it countable in ``doctor``.
    """
    blocks = await _blocks(parser, _raw(corpus, "duplicate-keys.json"))
    assert [block.anchor for block in blocks] == [Unlocated(reason=NO_JSON_POSITIONS)]
    assert "retries" in blocks[0].text


async def test_a_nan_literal_keeps_the_document_and_loses_only_its_positions(
    parser: StructuredParser, corpus: Path
) -> None:
    """``json`` reads a float where the mark-bearing reader reads the string ``"NaN"``.

    The two disagree about what the document contains, so its marks describe a different
    document and cannot be trusted for this one. Silent is the failure mode being avoided: the
    file indexes fine either way, and only the anchors would have been wrong.
    """
    blocks = await _blocks(parser, _raw(corpus, "nan.json"))
    assert [block.anchor for block in blocks] == [Unlocated(reason=NO_JSON_POSITIONS)]


async def test_valid_json_keeps_real_line_anchors_so_the_decline_is_not_the_normal_path(
    parser: StructuredParser, corpus: Path
) -> None:
    """The unlocated budget exists for the two files above and for nothing else.

    Without this, a parser that returned ``Unlocated`` for every JSON document would pass both
    tests above and the corpus budget, having stopped locating anything at all.
    """
    blocks = await _blocks(parser, _raw(corpus, "typical.json"))
    assert len(blocks) > 1
    assert all(isinstance(block.anchor, LineAnchor) for block in blocks)


async def test_json_that_does_not_parse_reports_a_failure_naming_the_line(
    parser: StructuredParser,
) -> None:
    """An actionable error names what was wrong and where.

    ``json.loads`` knows the line; discarding it and reporting "invalid JSON" makes a large
    configuration file a hunt.
    """
    raw = raw_of('{\n  "a": 1,\n  "b":\n}\n', "application/json", uri="broken.json")
    with pytest.raises(ParseError, match="line 4 is not valid JSON"):
        await _blocks(StructuredParser(StructuredConfig()), raw)


# --- TOML ---------------------------------------------------------------------------------


async def test_table_headers_become_symbols_and_arrays_of_tables_carry_their_occurrence(
    parser: StructuredParser, corpus: Path
) -> None:
    """``[[connector]]`` repeats one header, so the index is the only thing telling them apart.

    Without it the second entry's citation would name the first, which reads correctly and
    points at the wrong table.
    """
    blocks = await _blocks(parser, _raw(corpus, "typical.toml"))
    symbols = [block.anchor.symbol for block in blocks if isinstance(block.anchor, LineAnchor)]
    assert symbols == [None, "chunking", "storage.sqlite", "connector[0]", "connector[1]"]


async def test_the_keys_before_the_first_table_header_are_a_block_of_their_own(
    parser: StructuredParser, corpus: Path
) -> None:
    """The root table is content, not a preamble to the first section.

    Folding it into the first table would give those lines a symbol naming a table they are
    not in, which is a citation that names the wrong section of the file.
    """
    blocks = await _blocks(parser, _raw(corpus, "typical.toml"))
    assert blocks[0].anchor == LineAnchor(start=1, end=3)
    assert "schema_version" in blocks[0].text


async def test_a_file_with_no_table_headers_is_one_block_with_a_whole_file_span(
    parser: StructuredParser, corpus: Path
) -> None:
    """A flat key-value TOML file has one section, and it is the file."""
    blocks = await _blocks(parser, _raw(corpus, "flat.toml"))
    assert len(blocks) == 1
    assert blocks[0].anchor == LineAnchor(start=1, end=5)


async def test_a_table_header_inside_a_multi_line_string_does_not_become_a_section(
    parser: StructuredParser, corpus: Path
) -> None:
    """Every scanned header is checked against the tables ``tomllib`` actually produced.

    Scanning alone is a heuristic over extracted text, which is exactly what rule 1 of §2.1
    forbids as a source of anchors. Checking the scan against the parsed structure is what
    keeps the rule true for the one format with no position API.
    """
    blocks = await _blocks(parser, _raw(corpus, "tricky.toml"))
    symbols = [block.anchor.symbol for block in blocks if isinstance(block.anchor, LineAnchor)]
    assert "not a table" not in symbols
    assert symbols == [None, "service", "service.limits"]
    banner = next(block for block in blocks if "[not a table]" in block.text)
    assert banner.anchor == LineAnchor(start=2, end=7, symbol="service")


async def test_toml_that_does_not_parse_reports_a_failure_rather_than_a_decline(
    parser: StructuredParser, corpus: Path
) -> None:
    """A malformed document is broken, not somebody else's."""
    with pytest.raises(ParseError, match="not valid TOML"):
        await _blocks(parser, _raw(corpus, "invalid.toml"))


# --- size, refusals and the guard itself ---------------------------------------------------


async def test_an_oversized_key_splits_at_the_next_level_with_its_symbol_deepening(
    corpus: Path,
) -> None:
    """§11's rule for very large files, and the reason it is not a line split.

    Dividing a huge mapping at line boundaries would give every part the parent's symbol.
    Dividing it at the child keys the reader reported gives each part a symbol that addresses
    what it actually contains.
    """
    parser = StructuredParser(StructuredConfig(max_block_lines=40))
    blocks = await _blocks(parser, _raw(corpus, "structured-large.yaml"))
    symbols = [block.anchor.symbol for block in blocks if isinstance(block.anchor, LineAnchor)]
    assert "hosts.host-0500" in symbols
    assert all(
        isinstance(block.anchor, LineAnchor) and block.anchor.end - block.anchor.start < 40
        for block in blocks
    )


async def test_the_large_fixture_round_trips_through_the_chunker(
    parser: StructuredParser, corpus: Path, chunker: StructuralChunker
) -> None:
    """A document of nine hundred blocks still cites exactly what it quotes.

    Structured blocks are ``code``, so the chunker never overlaps them — an overlap window
    copies the previous chunk's text into this one, which a line span cannot honestly claim.
    """
    report = await check_fixture(parser, _raw(corpus, "structured-large.yaml"), chunker=chunker)
    assert report.blocks > 800
    assert report.chunks > 1


async def test_a_media_type_this_parser_does_not_read_is_declined(parser: StructuredParser) -> None:
    """Declining names what this parser reads, so the next step is obvious.

    Confluence's document format is JSON-shaped and registers under its own profiled media
    type; indexing it here would cite key paths where a reader expects sections.
    """
    raw = raw_of("{}", "application/json;profile=atlas-doc-format", uri="page.json")
    with pytest.raises(ParseError, match="declining"):
        await _blocks(parser, raw)


async def test_binary_bytes_are_declined_rather_than_decoded(parser: StructuredParser) -> None:
    """JSON, YAML and TOML are all text formats, so bytes that are not text are not ours."""
    raw = raw_of(b"\x00\x01\x02binary\x00", "application/json", uri="thing.json")
    with pytest.raises(ParseError, match="declining"):
        await _blocks(parser, raw)


async def test_resolve_reads_the_document_rather_than_anything_parse_left_behind(
    parser: StructuredParser,
) -> None:
    """Resolving against the parser's own memory of a document verifies nothing about it."""
    raw = raw_of("alpha: 1\nbeta: 2\n", "application/yaml")
    assert await parser.resolve(LineAnchor(start=2, end=2, symbol="beta"), raw) == "beta: 2"


async def test_an_anchor_one_line_out_is_caught_by_the_round_trip_check(
    corpus: Path, chunker: StructuralChunker
) -> None:
    """The guard is load-bearing, proved by breaking it.

    A key path is a plausible-looking symbol attached to the wrong lines, and nothing about it
    raises; only this assertion fires.
    """
    raw = _raw(corpus, "typical.yaml")
    with pytest.raises(AssertionError):
        await assert_round_trip(
            _OffByOneParser(StructuredConfig()),
            raw,
            fixture="off-by-one",
            chunker=chunker,
            document=document_for(raw),
        )


class _OffByOneParser(StructuredParser):
    """Anchors every block one line further down than the slice it claims."""

    @override
    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        async for block in super().parse(raw):
            anchor = block.anchor
            if not isinstance(anchor, LineAnchor):
                yield block
                continue
            moved = LineAnchor(start=anchor.start + 1, end=anchor.end + 1, symbol=anchor.symbol)
            yield block.model_copy(update={"anchor": moved})


async def test_a_block_stream_stopped_after_one_block_is_not_left_suspended(
    parser: StructuredParser, corpus: Path
) -> None:
    """A consumer that stops early must not strand the generator.

    A suspended generator is finalized later, through the loop that created it; when that loop
    has closed the finalization runs against a torn-down runtime and crashes the interpreter
    rather than warning.
    """
    stream = parser.parse(_raw(corpus, "structured-large.yaml"))
    async for _ in stream:
        break
    await aclose(stream)
    assert await _is_closed(stream)


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
