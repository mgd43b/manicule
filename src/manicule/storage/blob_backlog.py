"""The unpublished-blob total, kept by triggers so that admitting a blob does not scan them.

``BlobStore`` refuses a new blob when the bytes already retained and not yet published, plus the
new one, would pass ``max_acquired_blob_backlog_bytes``. That total used to be recomputed on every
admission by scanning ``blobs`` against ``documents`` and ``document_versions``. During a first
sync nothing is published, so the scan visited every blob the run had retained so far and the work
per fetched message grew with the run (a 200,000-message mailbox spent about half a second of CPU
per message on it, under the one writer lock).

The total is a singleton row, ``blob_backlog_totals``, and **triggers maintain it**, for the
reason ``manicule.storage.fts`` gives: a trigger runs in the transaction of the row change and
cannot be bypassed by a migration, a repair path or a write someone forgot. A blob counts while
its ``algo`` is not ``gc_pending:...`` and no document or version references its hash. A blob
enters the total when its row appears unreferenced or loses its last reference, and leaves it when
its row is deleted or gains its first reference. ``documents.original_ref`` and
``document_versions.original_ref`` are indexed so that "is this the first or the last reference"
costs two index probes, not a scan.

**The total is a fast path, never the only path.** It is trusted when it is an integer and
``unsized_rows`` is zero. SQLite turns integer overflow into a real number and the schema does not
stop a text ``stored_bytes``, so a total that overflowed or saw such a row is not used, and
``BlobStore`` computes the exact sum by scanning, as it always did.
"""

# ruff: noqa: S608 - trigger DDL assembled from module constants and trigger row references; no
# caller supplies any part of it.

from __future__ import annotations

from typing import Final

__all__ = [
    "CREATE_TRIGGERS",
    "DROP_TRIGGERS",
    "TRIGGER_NAMES",
    "initial_totals_sql",
]

_TOTALS = "blob_backlog_totals"
_PENDING = "gc_pending:%"
"""The ``LIKE`` pattern of ``blobs.GC_PENDING_PREFIX``; a test pins that they agree."""


def _unreferenced(hash_sql: str, *, except_table: str | None = None, except_id: str = "") -> str:
    """No document or version references ``hash_sql``, other than the row ``except_id``."""
    clauses: list[str] = []
    for table in ("documents", "document_versions"):
        other = f" AND id <> {except_id}" if table == except_table else ""
        clauses.append(f"NOT EXISTS (SELECT 1 FROM {table} WHERE original_ref = {hash_sql}{other})")
    return " AND ".join(clauses)


def _counted(algo_sql: str, stored_sql: str, *, sized: bool) -> str:
    kind = "=" if sized else "<>"
    return f"{algo_sql} NOT LIKE '{_PENDING}' AND typeof({stored_sql}) {kind} 'integer'"


def _move(sign: str, algo_sql: str, stored_sql: str, unreferenced: str) -> str:
    """Add (``+``) or take away (``-``) one blob given by values, if it is unreferenced."""
    sized = _counted(algo_sql, stored_sql, sized=True)
    unsized = _counted(algo_sql, stored_sql, sized=False)
    return (
        f"UPDATE {_TOTALS} SET unpublished_bytes = unpublished_bytes {sign} {stored_sql} "
        f"WHERE id = 1 AND {sized} AND {unreferenced};\n"
        f"UPDATE {_TOTALS} SET unsized_rows = unsized_rows {sign} 1 "
        f"WHERE id = 1 AND {unsized} AND {unreferenced};"
    )


def _move_row(sign: str, hash_sql: str) -> str:
    """Add or take away the blob row named ``hash_sql``, if there is one and it counts."""
    row = f"FROM blobs WHERE hash = {hash_sql}"
    sized = f"algo NOT LIKE '{_PENDING}' AND typeof(stored_bytes) = 'integer'"
    unsized = f"algo NOT LIKE '{_PENDING}' AND typeof(stored_bytes) <> 'integer'"
    return (
        f"UPDATE {_TOTALS} SET unpublished_bytes = unpublished_bytes {sign} "
        f"(SELECT stored_bytes {row} AND {sized}) "
        f"WHERE id = 1 AND EXISTS (SELECT 1 {row} AND {sized});\n"
        f"UPDATE {_TOTALS} SET unsized_rows = unsized_rows {sign} 1 "
        f"WHERE id = 1 AND EXISTS (SELECT 1 {row} AND {unsized});"
    )


def _pending_move(sign: str, algo_sql: str) -> str:
    return (
        f"UPDATE {_TOTALS} SET gc_pending_rows = gc_pending_rows {sign} 1 "
        f"WHERE id = 1 AND {algo_sql} LIKE '{_PENDING}';"
    )


def _blob_triggers() -> tuple[tuple[str, str], ...]:
    old, new = "old", "new"
    gone = _unreferenced(f"{old}.hash")
    here = _unreferenced(f"{new}.hash")
    return (
        (
            "blobs_backlog_ai",
            "CREATE TRIGGER blobs_backlog_ai AFTER INSERT ON blobs BEGIN\n"
            + _move("+", f"{new}.algo", f"{new}.stored_bytes", here)
            + "\n"
            + _pending_move("+", f"{new}.algo")
            + "\nEND",
        ),
        (
            "blobs_backlog_ad",
            "CREATE TRIGGER blobs_backlog_ad AFTER DELETE ON blobs BEGIN\n"
            + _move("-", f"{old}.algo", f"{old}.stored_bytes", gone)
            + "\n"
            + _pending_move("-", f"{old}.algo")
            + "\nEND",
        ),
        (
            "blobs_backlog_au",
            "CREATE TRIGGER blobs_backlog_au AFTER UPDATE OF hash, algo, stored_bytes ON blobs "
            "BEGIN\n"
            + _move("-", f"{old}.algo", f"{old}.stored_bytes", gone)
            + "\n"
            + _pending_move("-", f"{old}.algo")
            + "\n"
            + _move("+", f"{new}.algo", f"{new}.stored_bytes", here)
            + "\n"
            + _pending_move("+", f"{new}.algo")
            + "\nEND",
        ),
    )


def _reference_triggers(table: str, prefix: str) -> tuple[tuple[str, str], ...]:
    """A reference appearing takes its blob out of the total; the last one going puts it back."""
    old, new = "old", "new"
    first_new = _unreferenced(f"{new}.original_ref", except_table=table, except_id=f"{new}.id")
    none_old = _unreferenced(f"{old}.original_ref")
    return (
        (
            f"{prefix}_backlog_ai",
            f"CREATE TRIGGER {prefix}_backlog_ai AFTER INSERT ON {table} "
            f"WHEN {new}.original_ref IS NOT NULL AND {first_new} BEGIN\n"
            + _move_row("-", f"{new}.original_ref")
            + "\nEND",
        ),
        (
            f"{prefix}_backlog_ad",
            f"CREATE TRIGGER {prefix}_backlog_ad AFTER DELETE ON {table} "
            f"WHEN {old}.original_ref IS NOT NULL AND {none_old} BEGIN\n"
            + _move_row("+", f"{old}.original_ref")
            + "\nEND",
        ),
        (
            f"{prefix}_backlog_au",
            f"CREATE TRIGGER {prefix}_backlog_au AFTER UPDATE OF original_ref ON {table} "
            f"WHEN {old}.original_ref IS NOT {new}.original_ref BEGIN\n"
            + _move_row(
                "+",
                f"CASE WHEN {old}.original_ref IS NOT NULL AND {none_old} "
                f"THEN {old}.original_ref END",
            )
            + "\n"
            + _move_row(
                "-",
                f"CASE WHEN {new}.original_ref IS NOT NULL AND {first_new} "
                f"THEN {new}.original_ref END",
            )
            + "\nEND",
        ),
    )


_TRIGGERS: Final = (
    *_blob_triggers(),
    *_reference_triggers("documents", "documents"),
    *_reference_triggers("document_versions", "document_versions"),
)

TRIGGER_NAMES: Final[tuple[str, ...]] = tuple(name for name, _ in _TRIGGERS)
CREATE_TRIGGERS: Final[tuple[str, ...]] = tuple(sql for _, sql in _TRIGGERS)
DROP_TRIGGERS: Final[tuple[str, ...]] = tuple(
    f"DROP TRIGGER IF EXISTS {name}" for name in TRIGGER_NAMES
)


def initial_totals_sql() -> str:
    """The singleton row for an empty ``blobs`` table; a migration then adds what is there."""
    return (
        f"INSERT INTO {_TOTALS} (id, unpublished_bytes, unsized_rows, gc_pending_rows) "
        "VALUES (1, 0, 0, 0)"
    )
