"""XLSX and CSV: one parser, because a cell reference is the same address in both.

``docs/parsing.md`` §2.4 pairs these two deliberately. ``CellAnchor`` and the block model are
identical once a CSV is given a sheet name, and **a CSV's sheet name is its file stem** — so
sharing the parser means the two formats cannot drift apart in how they cite a range.

They do not share a *reader*. ``python-calamine`` reads Excel and ODF only; handed a `.csv` it
raises "cannot detect file format", because the Rust crate underneath has no CSV support at
all. That is a correction to ``PLAN.md`` §5, recorded in ``docs/parsing.md`` §2.4, and the
alternative to writing it down is discovering it as a crash on the first `.csv`. CSV therefore
goes through the standard library's ``csv``, which handles the quoting and embedded-newline
cases that actually occur.

Three things about the XLSX path are worth stating, because each one produces a wrong
citation rather than an error:

**The used range does not have to start at A1.** calamine reports the used range's top-left
corner, and an absolute reference is that origin plus the row and column offset. A parser that
assumes A1 cites the wrong cells on every sheet with a title row above the table or a margin
column to the left of it — and the reference still looks perfectly plausible.

**Merged cells report their value once.** A merged header spanning three columns is stored in
its top-left cell and read as empty in the other two. Left alone, a table split by rows
(``docs/parsing.md`` §4.2) repeats a header that is blank for two of its three columns, so the
part is a grid of numbers with no idea what they measure. The merged ranges calamine reports
are used to fill the covered cells, which is what the reader sees on screen.

**A whole number is not a decimal.** calamine returns every number as a float, so a cell
containing ``1`` arrives as ``1.0``. Rendering that verbatim makes a quotation say something
the document does not, which is exactly the class of defect the round-trip contract exists to
catch — so an integral float is rendered without its fractional part.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import urlsplit

from pydantic import BaseModel, Field
from python_calamine import (
    CalamineError,
    CalamineWorkbook,
    PasswordError,
    SheetVisibleEnum,
)

from manicule.core.anchors import Anchor, CellAnchor
from manicule.core.content import BlockKind, Metadata, ParsedBlock, RawDocument
from manicule.core.errors import ParseError
from manicule.parsers.base import ParserProfile, decode

__all__ = [
    "CSV_MEDIA_TYPE",
    "XLSX_MEDIA_TYPE",
    "SpreadsheetConfig",
    "SpreadsheetParser",
]

XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
CSV_MEDIA_TYPE = "text/csv"

_CELL_SEPARATOR = "\t"
_ROW_SEPARATOR = "\n"

_AREA = re.compile(
    r"^(?P<c0>[A-Z]{1,3})(?P<r0>[1-9][0-9]*)(?::(?P<c1>[A-Z]{1,3})(?P<r1>[1-9][0-9]*))?$"
)

_CellValue = int | float | str | bool | dt.time | dt.date | dt.datetime | dt.timedelta
"""What calamine hands back per cell. Spelled out so the renderer below is exhaustive."""


class SpreadsheetConfig(BaseModel):
    """Configuration for :class:`SpreadsheetParser`."""

    header_rows: int = Field(
        default=1,
        ge=0,
        description="How many leading rows of a used range are header rows. The chunker repeats "
        "them into every part of a table too large for one chunk (docs/parsing.md §4.2). "
        "Declared rather than detected: the alternative is inferring a header from the first "
        "row being bold, which that section forbids because formatting is not structure.",
    )
    csv_delimiter: str = Field(
        default=",",
        min_length=1,
        max_length=1,
        description="Field separator for CSV. Declared, not sniffed: sniffing reads a sample "
        "and guesses, so the same export could be split into different columns on two "
        "machines, and every cell reference in the corpus would depend on which.",
    )
    include_hidden_sheets: bool = Field(
        default=False,
        description="Index sheets the workbook marks hidden. Off by default because a hidden "
        "sheet is usually working data the author chose not to show; on when it is the "
        "reference table everything else looks up.",
    )


def _column_letters(index: int) -> str:
    """A 1-based column index as A1 letters: 1 to ``A``, 27 to ``AA``."""
    letters = ""
    remaining = index
    while remaining > 0:
        remaining, offset = divmod(remaining - 1, 26)
        letters = chr(ord("A") + offset) + letters
    return letters


def _column_index(letters: str) -> int:
    """A1 column letters as a 1-based index. The inverse of :func:`_column_letters`."""
    index = 0
    for letter in letters:
        index = index * 26 + (ord(letter) - ord("A") + 1)
    return index


@dataclass(frozen=True, slots=True)
class _Area:
    """One rectangle of cells, 1-based and inclusive at both ends."""

    first_row: int
    first_column: int
    last_row: int
    last_column: int

    @property
    def ref(self) -> str:
        """A1 notation, collapsed to a single cell reference when it covers one cell."""
        start = f"{_column_letters(self.first_column)}{self.first_row}"
        if (self.first_row, self.first_column) == (self.last_row, self.last_column):
            return start
        return f"{start}:{_column_letters(self.last_column)}{self.last_row}"


def _parse_ref(ref: str) -> list[_Area] | None:
    """Parse comma-separated A1 areas, or ``None`` if the reference is not that.

    Several areas is Excel's own multi-area syntax and the shape a split table's anchor takes
    — ``A1:D1,A25:D48`` addresses the header rows *and* the part's own rows without claiming
    the twenty-three rows in between (``docs/parsing.md`` §4.2).
    """
    areas: list[_Area] = []
    for piece in ref.split(","):
        match = _AREA.match(piece.strip())
        if match is None:
            return None
        first_row, first_column = int(match.group("r0")), _column_index(match.group("c0"))
        last_row = int(match.group("r1")) if match.group("r1") else first_row
        last_column = _column_index(match.group("c1")) if match.group("c1") else first_column
        areas.append(
            _Area(
                first_row=min(first_row, last_row),
                first_column=min(first_column, last_column),
                last_row=max(first_row, last_row),
                last_column=max(first_column, last_column),
            )
        )
    return areas


def _render_cell(value: _CellValue) -> str:
    """One cell as the text a citation should quote.

    ``bool`` is checked before ``int`` because it is one, and ``TRUE``/``FALSE`` is what a
    spreadsheet shows. An integral float loses its ``.0``: calamine types every number as a
    float, and a cell reading ``1`` must not be quoted as ``1.0``.
    """
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else repr(value)
    if isinstance(value, dt.timedelta):
        return str(value)
    if isinstance(value, dt.date | dt.time):
        return value.isoformat()
    return value


@dataclass(frozen=True, slots=True)
class _Region:
    """A sheet's used range: where it starts, and the text of every cell in it."""

    sheet: str
    index: int
    first_row: int
    first_column: int
    rows: tuple[tuple[str, ...], ...]
    merged: tuple[str, ...]

    @property
    def area(self) -> _Area:
        """The whole used range as one area."""
        width = max((len(row) for row in self.rows), default=0)
        return _Area(
            first_row=self.first_row,
            first_column=self.first_column,
            last_row=self.first_row + len(self.rows) - 1,
            last_column=self.first_column + max(width, 1) - 1,
        )

    def text(self, area: _Area | None = None) -> str | None:
        """The rendered cells of ``area``, or ``None`` when it is not inside this region.

        ``None`` rather than a clamped range: an anchor naming cells the sheet does not have is
        an anchor that has diverged from its document, and returning the rows that do exist
        would hide that behind a plausible quotation.
        """
        whole = self.area
        target = whole if area is None else area
        inside = (
            target.first_row >= whole.first_row
            and target.last_row <= whole.last_row
            and target.first_column >= whole.first_column
            and target.last_column <= whole.last_column
        )
        if not inside:
            return None
        lines: list[str] = []
        for number in range(target.first_row, target.last_row + 1):
            row = self.rows[number - whole.first_row]
            cells = [
                row[column - whole.first_column] if column - whole.first_column < len(row) else ""
                for column in range(target.first_column, target.last_column + 1)
            ]
            lines.append(_CELL_SEPARATOR.join(cells))
        return _ROW_SEPARATOR.join(lines)


def _rectangular(rows: Sequence[Sequence[str]]) -> tuple[tuple[str, ...], ...]:
    """Pad rows to a common width, so a column index means the same thing on every row."""
    width = max((len(row) for row in rows), default=0)
    return tuple(tuple(row) + ("",) * (width - len(row)) for row in rows)


def _drop_trailing_blank(rows: list[list[str]]) -> list[list[str]]:
    """Drop blank rows from the end. A file ending in newlines does not claim them."""
    while rows and not any(cell.strip() for cell in rows[-1]):
        rows.pop()
    return rows


def _sheet_name_for_csv(raw: RawDocument) -> str:
    """A CSV's sheet name: its file stem (``docs/parsing.md`` §2.4).

    The stem rather than the whole filename, because the extension is not part of what the
    sheet is called, and the reference a citation shows — ``forecast!A1:D12`` — reads like one
    a spreadsheet would accept.
    """
    path = urlsplit(raw.uri).path or raw.uri
    stem = PurePosixPath(path.replace("\\", "/")).stem
    return stem or raw.source_id


def _csv_regions(raw: RawDocument, config: SpreadsheetConfig) -> list[_Region]:
    """The single region a CSV describes, or none when it holds no cells."""
    text = decode(raw)
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=config.csv_delimiter)
    try:
        rows = [list(row) for row in reader]
    except csv.Error as exc:
        msg = (
            f"{raw.uri}: unreadable as CSV ({exc}). Expected delimiter-separated rows with "
            f"balanced quoting; a field longer than the reader's limit or a NUL byte will do "
            f"this. Check the file's delimiter and quoting, or route it to another parser"
        )
        raise ParseError(msg) from exc
    trimmed = _drop_trailing_blank(rows)
    if not trimmed:
        return []
    return [
        _Region(
            sheet=_sheet_name_for_csv(raw),
            index=1,
            first_row=1,
            first_column=1,
            rows=_rectangular(trimmed),
            merged=(),
        )
    ]


def _merged_refs(
    merged: Sequence[tuple[tuple[int, int], tuple[int, int]]], first_row: int, first_column: int
) -> tuple[str, ...]:
    """Merged ranges as A1 references, for ``metadata`` — 0-based from calamine, 1-based here."""
    return tuple(
        _Area(
            first_row=start[0] + 1,
            first_column=start[1] + 1,
            last_row=end[0] + 1,
            last_column=end[1] + 1,
        ).ref
        for start, end in merged
        if end[0] + 1 >= first_row and end[1] + 1 >= first_column
    )


def _fill_merged(
    grid: list[list[str]],
    merged: Sequence[tuple[tuple[int, int], tuple[int, int]]],
    first_row: int,
    first_column: int,
) -> None:
    """Copy each merged range's value across the cells it covers, in place.

    The covered cells are empty in the file — the value is stored once, in the top-left. What
    the reader sees is the value across the whole span, and that is what a header repeated into
    a split part has to say.
    """
    for start, end in merged:
        top, left = start[0] + 1 - first_row, start[1] + 1 - first_column
        bottom, right = end[0] + 1 - first_row, end[1] + 1 - first_column
        if top < 0 or left < 0 or top >= len(grid) or left >= len(grid[top]):
            continue
        value = grid[top][left]
        if not value:
            continue
        for row in range(top, min(bottom, len(grid) - 1) + 1):
            for column in range(left, right + 1):
                if column < len(grid[row]) and not grid[row][column]:
                    grid[row][column] = value


def _xlsx_regions(raw: RawDocument, config: SpreadsheetConfig) -> list[_Region]:
    """One region per sheet that has a used range, in workbook order."""
    try:
        workbook = CalamineWorkbook.from_filelike(io.BytesIO(raw.as_bytes()))
    except PasswordError as exc:
        msg = (
            f"{raw.uri}: the workbook is encrypted ({exc}). manicule reads no passwords; "
            f"remove the encryption or index a decrypted copy"
        )
        raise ParseError(msg) from exc
    except CalamineError as exc:
        msg = (
            f"{raw.uri}: not a readable spreadsheet ({type(exc).__name__}: {exc}). Expected an "
            f"XLSX workbook. A `.csv` reaches this parser as text/csv and is read by a "
            f"different code path, so a 'cannot detect file format' here means the bytes are "
            f"neither; check the declared media type"
        )
        raise ParseError(msg) from exc

    regions: list[_Region] = []
    for index, metadata in enumerate(workbook.sheets_metadata, start=1):
        hidden = metadata.visible is not SheetVisibleEnum.Visible
        if hidden and not config.include_hidden_sheets:
            continue
        sheet = workbook.get_sheet_by_name(metadata.name)
        start = sheet.start
        values = sheet.to_python()
        if start is None or not values:
            continue
        first_row, first_column = start[0] + 1, start[1] + 1
        grid = [[_render_cell(cell) for cell in row] for row in values]
        merged = sheet.merged_cell_ranges or []
        _fill_merged(grid, merged, first_row, first_column)
        regions.append(
            _Region(
                sheet=metadata.name,
                index=index,
                first_row=first_row,
                first_column=first_column,
                rows=_rectangular(grid),
                merged=_merged_refs(merged, first_row, first_column),
            )
        )
    return regions


class SpreadsheetParser:
    """Parses `.xlsx` and `.csv` into one ``table`` block per sheet, anchored by cell range.

    Both budgets in ``docs/parsing.md`` §3.4 are the strict ones: every cell in a spreadsheet
    has a reference, so nothing here is ever unlocated, and no page anchor exists to be
    page-level. A sheet with no used range produces no block, which is not the same as an
    unlocated one.
    """

    media_types = frozenset({XLSX_MEDIA_TYPE, CSV_MEDIA_TYPE})
    profile = ParserProfile(name="spreadsheet", max_unlocated_ratio=0.0, max_pagelevel_ratio=None)

    def __init__(self, config: SpreadsheetConfig) -> None:
        self._config = config

    async def parse(self, raw: RawDocument) -> AsyncIterator[ParsedBlock]:
        """Yield one ``table`` block per sheet, in workbook order."""
        for region in self._regions(raw):
            text = region.text()
            if text is None or not text.strip():
                continue
            yield ParsedBlock(
                kind=BlockKind.TABLE,
                text=text,
                anchor=CellAnchor(sheet=region.sheet, ref=region.area.ref),
                # The sheet name is a heading path element even though the anchor is not a
                # HeadingAnchor: it is what the breadcrumb needs to say which sheet a range is
                # on (docs/parsing.md §2.4).
                heading_path=(region.sheet,),
                metadata=self._metadata(region),
            )

    async def resolve(self, anchor: Anchor, raw: RawDocument) -> str | None:
        """Return the cells ``anchor`` addresses, re-derived from ``raw``."""
        if not isinstance(anchor, CellAnchor):
            return None
        areas = _parse_ref(anchor.ref)
        if areas is None:
            return None
        for region in self._regions(raw):
            if region.sheet != anchor.sheet:
                continue
            lines: list[str] = []
            for area in areas:
                text = region.text(area)
                if text is None:
                    return None
                lines.append(text)
            return _ROW_SEPARATOR.join(lines) or None
        return None

    def _regions(self, raw: RawDocument) -> list[_Region]:
        if raw.media_type == CSV_MEDIA_TYPE:
            return _csv_regions(raw, self._config)
        return _xlsx_regions(raw, self._config)

    def _metadata(self, region: _Region) -> Metadata:
        area = region.area
        return {
            "sheet": region.sheet,
            "sheet_index": region.index,
            "header_rows": min(self._config.header_rows, len(region.rows)),
            "rows": [area.first_row, area.last_row],
            "columns": [area.first_column, area.last_column],
            "merged_ranges": list(region.merged),
        }
