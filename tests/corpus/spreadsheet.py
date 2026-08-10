"""Generated `.xlsx` and `.csv` fixtures: typical, structurally hard, degenerate and hostile.

The workbooks are written by the small SpreadsheetML writer below rather than by a spreadsheet
library, for two reasons that both matter to a fixture:

**Exact control of the used range.** The interesting case is a sheet whose content does *not*
start at A1, because an absolute cell reference is the used-range origin plus an offset and a
parser that assumes A1 gets every such sheet wrong while still looking plausible. Writing the
``sheetData`` directly puts the origin where the fixture says.

**Byte determinism.** Every member is stored with a fixed timestamp, so building the corpus
twice produces identical files. A spreadsheet library stamps the current time into
``docProps`` and the zip entries, which makes "the same fixture" a moving target.

The CSVs are written as text, because that is what a CSV is.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Sequence
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

__all__ = ["build"]

_FIXED_TIMESTAMP = (2026, 1, 1, 0, 0, 0)
"""One timestamp for every zip member, so the bytes are the same on every run."""

_SHEET_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_RELATIONSHIPS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PACKAGE_RELATIONSHIPS = "http://schemas.openxmlformats.org/package/2006/relationships"
_CONTENT_TYPES = "http://schemas.openxmlformats.org/package/2006/content-types"

_Cell = str | int | float | None
_Rows = Sequence[Sequence[_Cell]]


def build(dest: Path) -> None:
    """Write this format's fixtures into ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    _typical(dest / "spreadsheet_typical.xlsx")
    _structurally_hard(dest / "spreadsheet_structurally_hard.xlsx")
    _single_cell(dest / "spreadsheet_degenerate_single_cell.xlsx")
    _empty_sheet(dest / "spreadsheet_degenerate_empty_sheet.xlsx")
    (dest / "spreadsheet_degenerate_zero_bytes.xlsx").write_bytes(b"")
    _hidden_sheet(dest / "spreadsheet_hidden_sheet.xlsx")
    _astral(dest / "spreadsheet_hostile_astral.xlsx")
    _large(dest / "spreadsheet-large.xlsx")
    _plain_zip(dest / "spreadsheet_hostile_plain_zip.xlsx")

    _csv_typical(dest / "spreadsheet_typical.csv")
    (dest / "spreadsheet_degenerate_zero_bytes.csv").write_bytes(b"")
    _csv_ragged(dest / "spreadsheet_structurally_hard.csv")
    _csv_hostile(dest / "spreadsheet_hostile_wide_unterminated.csv")


def _typical(path: Path) -> None:
    path.write_bytes(
        _workbook(
            [
                _Sheet(
                    name="Regional",
                    origin=(1, 1),
                    rows=[
                        ["Region", "Requests", "Errors", "Owner"],
                        ["EMEA", 48200, 12, "Priya"],
                        ["APAC", 31750, 31, "Tomás"],
                        ["AMER", 52640, 8, "Dara"],
                    ],
                ),
                _Sheet(
                    name="Notes",
                    origin=(1, 1),
                    rows=[["Note"], ["Errors exclude synthetic probes."]],
                ),
            ]
        )
    )


def _structurally_hard(path: Path) -> None:
    """A used range starting at C3, and a merged header spanning two columns."""
    path.write_bytes(
        _workbook(
            [
                _Sheet(
                    name="Offset",
                    origin=(3, 3),
                    rows=[
                        ["Forecast window", None, "Region"],
                        ["Low", "High", "Territory"],
                        [110, 240, "Nordics"],
                        [305, 410, "Iberia"],
                    ],
                    # The merged header stores "Forecast window" once and covers C3:D3, so a
                    # part split off the bottom of this table has to be given the header that
                    # applies to both of its first two columns.
                    merged=["C3:D3"],
                )
            ]
        )
    )


def _single_cell(path: Path) -> None:
    """One cell, whose reference is a single cell rather than a range."""
    path.write_bytes(_workbook([_Sheet(name="Solo", origin=(1, 1), rows=[["only value"]])]))


def _empty_sheet(path: Path) -> None:
    """A workbook whose only sheet has no cells. No blocks, and that is not a failure."""
    path.write_bytes(_workbook([_Sheet(name="Nothing", origin=(1, 1), rows=[])]))


def _hidden_sheet(path: Path) -> None:
    """A visible sheet and a hidden one, which are different kinds of content.

    A hidden sheet is usually working data the author chose not to show, so indexing it is a
    decision rather than a default — and the workbook states which is which, so nothing here
    has to guess.
    """
    path.write_bytes(
        _workbook(
            [
                _Sheet(
                    name="Shown",
                    origin=(1, 1),
                    rows=[["Metric", "Value"], ["visible headroom", 42]],
                ),
                _Sheet(
                    name="Working",
                    origin=(1, 1),
                    rows=[["Scratch", "Value"], ["hidden scratch figure", 99]],
                    hidden=True,
                ),
            ]
        )
    )


def _astral(path: Path) -> None:
    path.write_bytes(
        _workbook(
            [
                _Sheet(
                    name="宇宙",
                    origin=(1, 1),
                    rows=[
                        ["Glyph", "Plane"],
                        ["🌐", "supplementary"],
                        ["𠀋", "extension B"],
                        ["𝕄", "mathematical"],  # noqa: RUF001 - astral glyphs are the fixture
                    ],
                )
            ]
        )
    )


def _large(path: Path) -> None:
    """A sheet with enough rows to exceed one chunk, so the row split gets exercised."""
    rows: list[list[_Cell]] = [["Index", "Checkpoint", "Latency", "Region"]]
    rows.extend(
        [index, f"checkpoint {index:04d} verified", index * 3, f"region {index % 7}"]
        for index in range(1, 401)
    )
    path.write_bytes(_workbook([_Sheet(name="Checkpoints", origin=(1, 1), rows=rows)]))


def _plain_zip(path: Path) -> None:
    """A zip that is not a workbook at all, under an `.xlsx` name."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(_info("sheets.txt"), "not a workbook")
    path.write_bytes(buffer.getvalue())


def _csv_typical(path: Path) -> None:
    path.write_text(
        "Service,Owner,Budget\n"
        "checkout,payments,4200\n"
        "search,discovery,3100\n"
        'ledger,"finance, core",5800\n',
        encoding="utf-8",
    )


def _csv_ragged(path: Path) -> None:
    """Ragged rows, an embedded newline inside a quoted field, and trailing blank lines."""
    path.write_text(
        "Key,Value,Comment\n"
        'retry.limit,5,"raised after\nthe October incident"\n'
        "retry.window,30\n"
        "timeout,2,seconds,extra\n"
        "\n\n",
        encoding="utf-8",
    )


def _csv_hostile(path: Path) -> None:
    """Two hundred columns and a quote that never closes.

    Neither is an error to the standard library: the unterminated quote swallows the rest of
    the file into one field, which is what a reader has to cope with rather than crash on.
    """
    header = ",".join(f"column_{index:03d}" for index in range(1, 201))
    body = ",".join(str(index) for index in range(1, 201))
    path.write_text(
        f'{header}\n{body}\n{"x" * 8},"unterminated quote runs to the end\n', encoding="utf-8"
    )


class _Sheet:
    """One worksheet to write: its name, where its used range starts, and its cells."""

    def __init__(
        self,
        *,
        name: str,
        origin: tuple[int, int],
        rows: _Rows,
        merged: Sequence[str] = (),
        hidden: bool = False,
    ) -> None:
        self.name = name
        self.origin = origin
        self.rows = rows
        self.merged = tuple(merged)
        self.hidden = hidden


def _column_letters(index: int) -> str:
    letters = ""
    remaining = index
    while remaining > 0:
        remaining, offset = divmod(remaining - 1, 26)
        letters = chr(ord("A") + offset) + letters
    return letters


def _sheet_xml(sheet: _Sheet) -> str:
    first_row, first_column = sheet.origin
    parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        f'<worksheet xmlns="{_SHEET_MAIN}">',
    ]
    if sheet.rows:
        width = max(len(row) for row in sheet.rows)
        last = f"{_column_letters(first_column + width - 1)}{first_row + len(sheet.rows) - 1}"
        parts.append(f'<dimension ref="{_column_letters(first_column)}{first_row}:{last}"/>')
    parts.append("<sheetData>")
    for offset, row in enumerate(sheet.rows):
        number = first_row + offset
        cells = "".join(
            _cell_xml(f"{_column_letters(first_column + column)}{number}", value)
            for column, value in enumerate(row)
        )
        parts.append(f'<row r="{number}">{cells}</row>')
    parts.append("</sheetData>")
    if sheet.merged:
        merges = "".join(f'<mergeCell ref="{ref}"/>' for ref in sheet.merged)
        parts.append(f'<mergeCells count="{len(sheet.merged)}">{merges}</mergeCells>')
    parts.append("</worksheet>")
    return "".join(parts)


def _cell_xml(reference: str, value: _Cell) -> str:
    """One cell. Strings are inline, which avoids a shared-strings table entirely."""
    if value is None or value == "":
        return ""
    if isinstance(value, bool):  # pragma: no cover - no fixture needs a boolean cell yet
        return f'<c r="{reference}" t="b"><v>{int(value)}</v></c>'
    if isinstance(value, int | float):
        return f'<c r="{reference}"><v>{value}</v></c>'
    return (
        f'<c r="{reference}" t="inlineStr"><is><t xml:space="preserve">{escape(value)}</t></is></c>'
    )


def _workbook(sheets: Sequence[_Sheet]) -> bytes:
    """A minimal SpreadsheetML package: content types, relationships, workbook, worksheets."""
    overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
        f'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.'
        f'worksheet+xml"/>'
        for index in range(1, len(sheets) + 1)
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Types xmlns="{_CONTENT_TYPES}">'
        '<Default Extension="rels" ContentType="application/'
        'vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/'
        'vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        f"{overrides}</Types>"
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Relationships xmlns="{_PACKAGE_RELATIONSHIPS}">'
        f'<Relationship Id="rId1" Type="{_RELATIONSHIPS}/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )
    sheet_tags = "".join(
        f'<sheet name={quoteattr(sheet.name)} sheetId="{index}" r:id="rId{index}"'
        f"{' state=' + quoteattr('hidden') if sheet.hidden else ''}/>"
        for index, sheet in enumerate(sheets, start=1)
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<workbook xmlns="{_SHEET_MAIN}" xmlns:r="{_RELATIONSHIPS}">'
        f"<sheets>{sheet_tags}</sheets></workbook>"
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Relationships xmlns="{_PACKAGE_RELATIONSHIPS}">'
        + "".join(
            f'<Relationship Id="rId{index}" Type="{_RELATIONSHIPS}/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
            for index in range(1, len(sheets) + 1)
        )
        + "</Relationships>"
    )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(_info("[Content_Types].xml"), content_types)
        archive.writestr(_info("_rels/.rels"), root_rels)
        archive.writestr(_info("xl/workbook.xml"), workbook)
        archive.writestr(_info("xl/_rels/workbook.xml.rels"), workbook_rels)
        for index, sheet in enumerate(sheets, start=1):
            archive.writestr(_info(f"xl/worksheets/sheet{index}.xml"), _sheet_xml(sheet))
    return buffer.getvalue()


def _info(name: str) -> zipfile.ZipInfo:
    """A zip entry with a fixed timestamp, so the package bytes are reproducible."""
    info = zipfile.ZipInfo(name, date_time=_FIXED_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    return info
