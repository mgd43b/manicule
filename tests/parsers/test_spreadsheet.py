"""XLSX and CSV parsing: one parser, one anchor, two readers.

The assertions that carry the most weight are the ones about *where* a cell is:
:func:`test_a_reference_counts_from_the_used_range_origin_rather_than_from_a1`, because a sheet
with a title row above its table is ordinary and an A1 assumption gets every cell in it wrong
while still looking plausible, and
:func:`test_a_split_table_part_addresses_its_own_rows_and_its_header`, because splitting a
table is the one place in this project where chunking *improves* provenance.

:func:`test_calamine_cannot_read_a_csv_which_is_why_the_stdlib_reader_exists` is here to keep a
library correction from being rediscovered as a crash.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from manicule.chunking import StructuralChunker
from manicule.core.anchors import Anchor, CellAnchor, Unlocated
from manicule.core.content import BlockKind, ParsedBlock, RawDocument
from manicule.core.errors import ParseError
from manicule.core.protocols import Parser, parsing, read_blocks
from manicule.parsers.spreadsheet import (
    CSV_MEDIA_TYPE,
    XLSX_MEDIA_TYPE,
    SpreadsheetConfig,
    SpreadsheetParser,
)
from manicule.testing import assert_round_trip
from tests.parsers.support import check_corpus, check_fixture, document_for, raw_from

WORKBOOK_FIXTURES = (
    "spreadsheet_typical.xlsx",
    "spreadsheet_structurally_hard.xlsx",
    "spreadsheet_degenerate_single_cell.xlsx",
    "spreadsheet_degenerate_empty_sheet.xlsx",
    "spreadsheet_hidden_sheet.xlsx",
    "spreadsheet_hostile_astral.xlsx",
    "spreadsheet-large.xlsx",
)

CSV_FIXTURES = (
    "spreadsheet_typical.csv",
    "spreadsheet_structurally_hard.csv",
    "spreadsheet_degenerate_zero_bytes.csv",
    "spreadsheet_hostile_wide_unterminated.csv",
)


def _parser(config: SpreadsheetConfig | None = None) -> SpreadsheetParser:
    return SpreadsheetParser(config or SpreadsheetConfig())


def _media_type(path: Path) -> str:
    return CSV_MEDIA_TYPE if path.suffix == ".csv" else XLSX_MEDIA_TYPE


async def _blocks(path: Path, config: SpreadsheetConfig | None = None) -> list[ParsedBlock]:
    return await read_blocks(_parser(config), raw_from(path, _media_type(path)))


def _anchor_of(block: ParsedBlock) -> CellAnchor:
    assert isinstance(block.anchor, CellAnchor), f"expected a cell anchor, got {block.anchor}"
    return block.anchor


# --- where a cell is ---------------------------------------------------------------------


async def test_a_reference_counts_from_the_used_range_origin_rather_than_from_a1(
    corpus: Path,
) -> None:
    """An absolute reference is the used-range origin plus the row and column offset.

    The fixture's content starts at C3, which is what a sheet with a title row above its table
    and a margin column beside it looks like. A parser that assumes A1 produces references that
    are well-formed, inside the sheet, and pointing at the wrong cells — the failure this whole
    document exists to prevent, in its quietest form.
    """
    blocks = await _blocks(corpus / "spreadsheet" / "spreadsheet_structurally_hard.xlsx")
    assert len(blocks) == 1
    anchor = _anchor_of(blocks[0])

    assert anchor.sheet == "Offset"
    assert anchor.ref == "C3:E6"
    assert anchor.a1 == "Offset!C3:E6"
    assert blocks[0].metadata["first_row"] == 3
    assert blocks[0].metadata["first_column"] == 3


async def test_a_single_cell_is_addressed_as_a_cell_and_not_as_a_range(corpus: Path) -> None:
    """``Solo!A1``, not ``Solo!A1:A1``. The reference a citation shows is one a reader reads."""
    blocks = await _blocks(corpus / "spreadsheet" / "spreadsheet_degenerate_single_cell.xlsx")
    assert _anchor_of(blocks[0]).a1 == "Solo!A1"
    assert blocks[0].text == "only value"


async def test_a_csv_takes_its_sheet_name_from_its_file_stem(corpus: Path) -> None:
    """A CSV has no sheets, so ``CellAnchor.sheet`` needs one supplied.

    ``docs/parsing.md`` §2.4 settles it as the file stem — not the whole filename, because the
    extension is not part of what the sheet is called, and the reference a citation shows
    (``spreadsheet_typical!A1:C4``) then reads like one a spreadsheet would accept.
    """
    path = corpus / "spreadsheet" / "spreadsheet_typical.csv"
    blocks = await _blocks(path)

    assert _anchor_of(blocks[0]).sheet == "spreadsheet_typical"
    assert _anchor_of(blocks[0]).ref == "A1:C4"
    assert blocks[0].heading_path == ("spreadsheet_typical",)


async def test_the_sheet_name_is_a_heading_path_element(corpus: Path) -> None:
    """The anchor is not a ``HeadingAnchor`` and the sheet name is still a breadcrumb element.

    ``docs/parsing.md`` §2.4 requires it: the breadcrumb goes into the embedding, and "EMEA" on
    a sheet called "Regional" retrieves for a query about regions only if the sheet name is
    there.
    """
    blocks = await _blocks(corpus / "spreadsheet" / "spreadsheet_typical.xlsx")
    assert [block.heading_path for block in blocks] == [("Regional",), ("Notes",)]


# --- the two readers ---------------------------------------------------------------------


async def test_calamine_cannot_read_a_csv_which_is_why_the_stdlib_reader_exists(
    corpus: Path,
) -> None:
    """A correction to ``PLAN.md`` §5, kept as a test so it cannot be rediscovered as a crash.

    ``python-calamine`` handles Excel and ODF only; the Rust crate underneath has no CSV
    support and reports "cannot detect file format". So the same parser reads CSV with the
    standard library, chosen by media type — and a `.csv` routed to the workbook reader must
    fail with a message that says which path it took.
    """
    path = corpus / "spreadsheet" / "spreadsheet_typical.csv"

    with pytest.raises(ParseError, match="not a readable spreadsheet"):
        await read_blocks(_parser(), raw_from(path, XLSX_MEDIA_TYPE))

    blocks = await read_blocks(_parser(), raw_from(path, CSV_MEDIA_TYPE))
    assert blocks, "the same bytes read as text/csv must parse"


async def test_a_whole_number_is_not_quoted_as_a_decimal(corpus: Path) -> None:
    """calamine types every number as a float, and a cell reading ``12`` must not become ``12.0``.

    A citation reproduces what the document says. Rendering the float verbatim would make the
    quotation say something the spreadsheet does not, which is the same defect class as an
    anchor pointing at the wrong cell and considerably easier to introduce.
    """
    blocks = await _blocks(corpus / "spreadsheet" / "spreadsheet_typical.xlsx")
    first_data_row = blocks[0].text.splitlines()[1]

    assert first_data_row == "EMEA\t48200\t12\tPriya"
    assert ".0" not in blocks[0].text


async def test_ragged_csv_rows_are_padded_so_a_column_index_means_one_thing(corpus: Path) -> None:
    """Rows of differing width would make the same column index mean two positions.

    The reference has to describe a rectangle, so short rows are padded to the widest. The
    fixture also carries a newline inside a quoted field, which is flattened for the same
    reason a table row is one line: a row split cuts at line ends.
    """
    blocks = await _blocks(corpus / "spreadsheet" / "spreadsheet_structurally_hard.csv")
    rows = blocks[0].text.splitlines()

    assert _anchor_of(blocks[0]).ref == "A1:D4"
    assert len({len(row.split("\t")) for row in rows}) == 1, "every row has the same width"
    assert "raised after the October incident" in blocks[0].text


async def test_an_unterminated_quote_is_read_rather_than_raising(corpus: Path) -> None:
    """Two hundred columns and a quote that never closes are a file, not an exception.

    The standard library reads the unterminated quote as one field running to the end of the
    file. That is a hostile input the parser has to survive, because the alternative — failing
    the document — loses two hundred columns of real data over one missing character.
    """
    blocks = await _blocks(corpus / "spreadsheet" / "spreadsheet_hostile_wide_unterminated.csv")

    assert len(blocks) == 1
    assert _anchor_of(blocks[0]).ref.endswith("GR3")
    assert "column_200" in blocks[0].text


# --- merged cells and headers ------------------------------------------------------------


async def test_a_merged_header_is_repeated_across_every_column_it_covers(corpus: Path) -> None:
    """A merged cell stores its value once and is read as empty in the cells it spans.

    Left alone, a table split by rows repeats a header that is blank for two of its three
    columns, so the part is a grid of numbers with nothing saying what they measure. The merged
    ranges calamine reports are what makes the repeat correct rather than invented.
    """
    blocks = await _blocks(corpus / "spreadsheet" / "spreadsheet_structurally_hard.xlsx")
    first_row = blocks[0].text.splitlines()[0]

    assert first_row == "Forecast window\tForecast window\tRegion"
    assert blocks[0].metadata["merged_ranges"] == ["C3:D3"]


async def test_a_table_describes_its_rows_and_their_references(corpus: Path) -> None:
    """The chunker splits at row boundaries and narrows the anchor to the rows it kept.

    It needs both halves: the rendered lines, so a split cuts where a row ends, and one
    reference per line, so a part can address exactly its own cells. A count of rows would
    leave it guessing at both.
    """
    blocks = await _blocks(corpus / "spreadsheet" / "spreadsheet_typical.xlsx")
    metadata = blocks[0].metadata

    rows, refs = metadata["rows"], metadata["row_refs"]
    assert isinstance(rows, list)
    assert isinstance(refs, list)
    assert "\n".join(str(row) for row in rows) == blocks[0].text
    assert len(refs) == len(rows)
    assert refs[0] == "A1:D1"
    assert refs[-1] == "A4:D4"
    assert metadata["header_rows"] == 1


async def test_the_header_row_count_is_declared_rather_than_detected(corpus: Path) -> None:
    """Nothing here reads formatting to decide what a header is.

    ``docs/parsing.md`` §4.2 forbids inferring it from the first row being bold, because
    formatting is not structure — a sheet whose first row is data and happens to be bold would
    lose that row from every split part. So it is configuration, and a sheet with two header
    rows says so.
    """
    blocks = await _blocks(
        corpus / "spreadsheet" / "spreadsheet_structurally_hard.xlsx",
        SpreadsheetConfig(header_rows=2),
    )
    assert blocks[0].metadata["header_rows"] == 2

    none = await _blocks(
        corpus / "spreadsheet" / "spreadsheet_structurally_hard.xlsx",
        SpreadsheetConfig(header_rows=0),
    )
    assert none[0].metadata["header_rows"] == 0


# --- hidden sheets -----------------------------------------------------------------------


async def test_a_hidden_sheet_is_skipped_unless_it_is_asked_for(corpus: Path) -> None:
    """The workbook states which sheets are hidden, so this is a decision and not a guess.

    A hidden sheet is usually working data the author chose not to show, and indexing it by
    default surfaces figures nobody published. Turning it on is one setting, for the case where
    the hidden sheet is the reference table everything else looks up.
    """
    path = corpus / "spreadsheet" / "spreadsheet_hidden_sheet.xlsx"

    default = await _blocks(path)
    assert [_anchor_of(block).sheet for block in default] == ["Shown"]

    included = await _blocks(path, SpreadsheetConfig(include_hidden_sheets=True))
    assert [_anchor_of(block).sheet for block in included] == ["Shown", "Working"]
    assert "hidden scratch figure" in included[1].text


# --- the six assertions ------------------------------------------------------------------


async def test_the_corpus_round_trips_and_stays_inside_its_location_budget(
    corpus: Path, chunker: StructuralChunker
) -> None:
    """Every fixture, blocks and chunks, at the strictest budgets in ``docs/parsing.md`` §3.4.

    Both are zero: every cell in a spreadsheet has a reference, so nothing here is ever
    unlocated, and there is no page anchor to be page-level. A sheet with no used range
    produces no block, which is a different thing from an unlocated one.
    """
    names = [*WORKBOOK_FIXTURES, *CSV_FIXTURES]
    raws = [raw_from(corpus / "spreadsheet" / name, _media_type(Path(name))) for name in names]
    reports = await check_corpus(_parser(), raws, chunker=chunker, min_blocks=10)

    assert sum(report.unlocated for report in reports) == 0
    assert sum(report.chunks for report in reports) > len(reports), (
        "the large fixture must split into several chunks or the row-split path is untested"
    )


async def test_a_split_table_part_addresses_its_own_rows_and_its_header(
    corpus: Path, chunker: StructuralChunker
) -> None:
    """Splitting a spreadsheet table improves provenance instead of costing it.

    Each part addresses exactly the rows it kept *and* the header rows repeated into it, as
    comma-separated A1 areas — Excel's own multi-area syntax, so the reference a citation shows
    is one a spreadsheet accepts. A whole-table anchor on every part would resolve to all four
    hundred rows and fail the tightness bound at ``k=1.0``.
    """
    raw = raw_from(corpus / "spreadsheet" / "spreadsheet-large.xlsx", XLSX_MEDIA_TYPE)
    blocks = await read_blocks(_parser(), raw)
    chunks = chunker.chunk(document_for(raw), blocks)

    assert len(chunks) > 1, "the fixture must exceed one chunk or this test measures nothing"
    refs = [chunk.anchor.ref for chunk in chunks if isinstance(chunk.anchor, CellAnchor)]
    assert len(refs) == len(chunks)
    assert len(set(refs)) == len(refs), "every part addresses a different range"
    assert all("," in ref for ref in refs[1:]), (
        "a part after the first repeats the header row, so its reference needs both areas"
    )

    for chunk in chunks:
        resolved = await _parser().resolve(chunk.anchor, raw)
        assert resolved is not None
        assert chunk.text.splitlines()[0] == "Index\tCheckpoint\tLatency\tRegion", (
            "every part carries the header, or it is a grid of numbers"
        )
        assert resolved == chunk.text


async def test_shifting_every_reference_down_one_row_fails_the_round_trip(corpus: Path) -> None:
    """An off-by-one row is well-formed, inside the sheet, and quoting the wrong cells.

    This is the fake that proves the harness is load-bearing: nothing raises, the reference
    parses, and the citation is wrong by exactly one row — which is what a used-range origin
    applied at the wrong end looks like.
    """
    raw = raw_from(corpus / "spreadsheet" / "spreadsheet_typical.xlsx", XLSX_MEDIA_TYPE)
    with pytest.raises(AssertionError):
        await assert_round_trip(_ShiftedRowParser(), raw, fixture="shifted")


# --- declines and empties ----------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["spreadsheet_degenerate_zero_bytes.xlsx", "spreadsheet_hostile_plain_zip.xlsx"]
)
async def test_an_unreadable_workbook_is_declined_rather_than_indexed(
    corpus: Path, name: str
) -> None:
    """Zero bytes and a plain zip under an `.xlsx` name are both "not ours", said out loud."""
    with pytest.raises(ParseError, match="not a readable spreadsheet"):
        await _blocks(corpus / "spreadsheet" / name)


async def test_a_sheet_with_no_used_range_yields_no_blocks_and_does_not_raise(
    corpus: Path,
) -> None:
    """An empty sheet is empty, not broken, and it has no reference to cite.

    Emitting a block for it would mean a chunk with no text, which is retrievable and cites
    nothing — the placeholder ``docs/parsing.md`` §6.4 refuses.
    """
    assert await _blocks(corpus / "spreadsheet" / "spreadsheet_degenerate_empty_sheet.xlsx") == []


async def test_an_empty_csv_yields_no_blocks_and_does_not_raise(corpus: Path) -> None:
    """Zero bytes of CSV is a document with no cells, which is not a parse failure."""
    assert await _blocks(corpus / "spreadsheet" / "spreadsheet_degenerate_zero_bytes.csv") == []


async def test_astral_text_survives_into_the_cells_and_the_sheet_name(corpus: Path) -> None:
    """A sheet name and a cell can both hold codepoints above the basic multilingual plane."""
    path = corpus / "spreadsheet" / "spreadsheet_hostile_astral.xlsx"
    await check_fixture(_parser(), raw_from(path, XLSX_MEDIA_TYPE))
    blocks = await _blocks(path)

    assert _anchor_of(blocks[0]).sheet == "宇宙"
    assert "🌐" in blocks[0].text
    assert "𠀋" in blocks[0].text


async def test_no_block_is_unlocated(corpus: Path) -> None:
    """Every cell has a reference, so a spreadsheet block is never unplaceable."""
    for name in [*WORKBOOK_FIXTURES, *CSV_FIXTURES]:
        blocks = await _blocks(corpus / "spreadsheet" / name)
        assert not any(isinstance(block.anchor, Unlocated) for block in blocks), name
        assert all(block.kind is BlockKind.TABLE for block in blocks), name


# --- stream lifecycle --------------------------------------------------------------------


async def test_stopping_after_one_block_leaves_nothing_held_open(corpus: Path) -> None:
    """An abandoned generator must not be left suspended holding the workbook it opened.

    CPython finalises a live async generator through the event loop that created it, so one
    still suspended when that loop has closed is finalised against a torn-down runtime — a
    crash inside the interpreter's allocator rather than a warning. This parser reads the whole
    workbook before it yields anything, so nothing is live at a suspension point, and re-parsing
    immediately afterwards checks that nothing was left in a state the next read trips over.
    The pattern is shown catching a release placed after the loop in
    ``tests/parsers/test_word.py`` and in ``tests/test_parser_streams.py``.
    """
    raw = raw_from(corpus / "spreadsheet" / "spreadsheet_typical.xlsx", XLSX_MEDIA_TYPE)
    parser = _parser()
    async with parsing(parser, raw) as blocks:
        async for _ in blocks:
            break
    assert len(await read_blocks(parser, raw)) > 1


class _ShiftedRowParser:
    """Moves every reference down one row.

    The house pattern from ``tests/fakes.py``, in the shape a spreadsheet makes available. The
    anchors it emits are syntactically perfect and semantically one row out, which is what an
    origin conversion applied in the wrong direction produces.
    """

    media_types = frozenset({XLSX_MEDIA_TYPE, CSV_MEDIA_TYPE})
    profile = SpreadsheetParser.profile

    def __init__(self) -> None:
        self._real = SpreadsheetParser(SpreadsheetConfig())

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        for block in await read_blocks(self._real, raw):
            if not isinstance(block.anchor, CellAnchor):  # pragma: no cover - all are cells
                yield block
                continue
            shifted = ",".join(_down_one(area) for area in block.anchor.ref.split(","))
            yield block.model_copy(
                update={"anchor": CellAnchor(sheet=block.anchor.sheet, ref=shifted)}
            )

    async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
        return await self._real.resolve(anchor, raw)


def _down_one(area: str) -> str:
    """One A1 area with every row number incremented."""
    return ":".join(_increment(reference) for reference in area.split(":"))


def _increment(reference: str) -> str:
    digits = "".join(character for character in reference if character.isdigit())
    letters = reference[: len(reference) - len(digits)]
    return f"{letters}{int(digits) + 1}"


_PARSERS: Sequence[Parser] = (SpreadsheetParser(SpreadsheetConfig()), _ShiftedRowParser())
"""Type-checked conformance: the fake is a parser, not a mock of one."""
