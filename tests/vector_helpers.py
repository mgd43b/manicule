"""How to damage one Lance row the way hardware does, in one place.

Every numerical-integrity test needs the same unusual thing: a stored row with exactly one
piece changed and everything else left alone. Going through
:meth:`~manicule.storage.vectors.LanceVectorStore.upsert` cannot produce it — a write
recomputes the checksum from the vector it is given, so the row comes out self-consistent
again, which is the opposite of the state being tested. So these reach lancedb directly, and
they live here rather than in two suites because a second copy is a second thing to get subtly
wrong, and a corruption helper that quietly corrupts the wrong column produces a test that
passes for the wrong reason.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
#
# lancedb annotates its surface in terms of `pyarrow`, which ships no type information, so
# every call here is "partially unknown" through no fault of this code. The same note and the
# same narrow suppression are in `manicule.storage.vectors`; values crossing back out are
# converted explicitly.

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Any

import lancedb

from manicule.storage.vectors import CHUNK_ID_COLUMN, ID_COLUMN, table_name

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from manicule.core.embedding import EmbedFingerprint


async def rewrite_row(
    directory: Path,
    embed: EmbedFingerprint,
    chunk_id: str,
    columns: Mapping[str, Any],
) -> None:
    """Replace named columns of one physical row, leaving every other column untouched.

    Delete and re-add rather than ``update``, because Lance declines to plan a SQL update
    against a ``fixed_size_list`` column — the vector cannot be written as a literal, so
    replacing the whole row is the only way to change it from outside the store.
    """
    connection = await lancedb.connect_async(directory)
    table = await connection.open_table(table_name(embed))
    found: list[dict[str, Any]] = await table.query().to_list()
    rows = [row for row in found if str(row[CHUNK_ID_COLUMN]) == chunk_id]
    assert rows, f"no row for {chunk_id!r} to rewrite"
    for row in rows:
        row.update(columns)
        await table.delete(f"{ID_COLUMN} = {_quoted(str(row[ID_COLUMN]))}")
        await table.add([row])
    connection.close()


async def read_column(directory: Path, embed: EmbedFingerprint, chunk_id: str, column: str) -> Any:
    """One column of one row, read without going through the store's own classification.

    ``Any`` because a Lance column is whatever Arrow made of it — a string, a float, a list —
    and a caller reading one already knows which.
    """
    connection = await lancedb.connect_async(directory)
    table = await connection.open_table(table_name(embed))
    found: list[dict[str, Any]] = await table.query().to_list()
    rows = [row for row in found if str(row[CHUNK_ID_COLUMN]) == chunk_id]
    connection.close()
    assert rows, f"no row for {chunk_id!r}"
    return rows[0][column]


async def rows_of(directory: Path, embed: EmbedFingerprint, predicate: str) -> list[dict[str, Any]]:
    """Every physical row matching ``predicate``, as plain dictionaries."""
    connection = await lancedb.connect_async(directory)
    table = await connection.open_table(table_name(embed))
    found: list[dict[str, Any]] = await table.query().where(predicate).to_list()
    connection.close()
    return found


def nudged(vector: Sequence[float], index: int = 0) -> list[float]:
    """``vector`` with one component moved to the adjacent float32 value.

    The smallest possible corruption, and the one that makes the point: the result is finite,
    the same length, the same order of magnitude, and — at a distance a cosine metric can
    barely see — still ranks the row roughly where it did. Nothing but a checksum notices.
    """
    values = [float(value) for value in vector]
    bits = struct.unpack(">I", struct.pack(">f", values[index]))[0]
    values[index] = struct.unpack(">f", struct.pack(">I", bits + 1))[0]
    return values


def _quoted(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
