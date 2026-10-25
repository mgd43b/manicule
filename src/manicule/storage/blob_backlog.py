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
its row is deleted or gains a reference. **The set of blobs in the total is a table of its own,
``blob_backlog_members``, holding what each contributed**, so entering and leaving are idempotent:
a document and its saved version that share a blob are deleted together by a cascade, and each
deletion finds the blob unreferenced and enters it, once between them.
``documents.original_ref`` and ``document_versions.original_ref`` are indexed so that "is this
the first or the last reference" costs two index probes, not a scan.

**The total is a fast path, never the only path.** It is trusted when ``inexact`` and
``unsized_rows`` are zero. The schema does not stop a text ``stored_bytes``, and a sum can pass
what SQLite stores as an integer (it then becomes a real number that an integer column would
turn back into an integer once enough was taken away, with the precision gone). Either marks the
total untrusted, ``inexact`` for good, and ``BlobStore`` computes the exact sum by scanning, as it
always did. A conflict deletion by ``INSERT OR REPLACE`` fires no delete trigger while
``recursive_triggers`` is off; no writer of these tables uses one.
"""

# ruff: noqa: S608 - trigger DDL assembled from module constants and trigger row references; no
# caller supplies any part of it.

from __future__ import annotations

from typing import Final

__all__ = [
    "CREATE_TRIGGERS",
    "DROP_TRIGGERS",
    "MEMBERS_FROM_BLOBS",
    "TRIGGER_NAMES",
    "initial_totals_sql",
]

_TOTALS = "blob_backlog_totals"
_MEMBERS = "blob_backlog_members"
_PENDING = "gc_pending:%"
"""The ``LIKE`` pattern of ``blobs.GC_PENDING_PREFIX``; a test pins that they agree."""
_INT_MAX = 2**63 - 1


def _unreferenced(hash_sql: str) -> str:
    """No document or version references ``hash_sql``."""
    return " AND ".join(
        f"NOT EXISTS (SELECT 1 FROM {table} WHERE original_ref = {hash_sql})"
        for table in ("documents", "document_versions")
    )


def _enter(hash_sql: str) -> str:
    """Count the blob ``hash_sql`` if it exists, is not pending, is unreferenced and is not yet in.

    Idempotent through the membership row, which is written last and records what was added.
    """
    row = f"FROM blobs WHERE hash = {hash_sql} AND algo NOT LIKE '{_PENDING}'"
    sized = f"{row} AND typeof(stored_bytes) = 'integer'"
    unsized = f"{row} AND typeof(stored_bytes) <> 'integer'"
    fresh = f"NOT EXISTS (SELECT 1 FROM {_MEMBERS} WHERE hash = {hash_sql}) AND " + _unreferenced(
        hash_sql
    )
    return (
        f"UPDATE {_TOTALS} SET inexact = 1 WHERE id = 1 AND typeof(unpublished_bytes) = 'integer' "
        f"AND EXISTS (SELECT 1 {sized}) AND {fresh} "
        f"AND unpublished_bytes > {_INT_MAX} - (SELECT stored_bytes {sized});\n"
        f"UPDATE {_TOTALS} SET unpublished_bytes = unpublished_bytes + "
        f"(SELECT stored_bytes {sized}) "
        f"WHERE id = 1 AND EXISTS (SELECT 1 {sized}) AND {fresh};\n"
        f"UPDATE {_TOTALS} SET unsized_rows = unsized_rows + 1 "
        f"WHERE id = 1 AND EXISTS (SELECT 1 {unsized}) AND {fresh};\n"
        f"INSERT INTO {_MEMBERS} (hash, bytes, sized) SELECT hash, "
        f"CASE WHEN typeof(stored_bytes) = 'integer' THEN stored_bytes ELSE 0 END, "
        f"typeof(stored_bytes) = 'integer' {row} AND {fresh};"
    )


def _leave(hash_sql: str) -> str:
    """Take the blob ``hash_sql`` out of the total by what it contributed, if it is in."""
    member = f"FROM {_MEMBERS} WHERE hash = {hash_sql}"
    return (
        f"UPDATE {_TOTALS} SET unpublished_bytes = unpublished_bytes - "
        f"(SELECT bytes {member} AND sized = 1) "
        f"WHERE id = 1 AND EXISTS (SELECT 1 {member} AND sized = 1);\n"
        f"UPDATE {_TOTALS} SET unsized_rows = unsized_rows - 1 "
        f"WHERE id = 1 AND EXISTS (SELECT 1 {member} AND sized = 0);\n"
        f"DELETE {member};"
    )


def _pending_move(sign: str, algo_sql: str) -> str:
    return (
        f"UPDATE {_TOTALS} SET gc_pending_rows = gc_pending_rows {sign} 1 "
        f"WHERE id = 1 AND {algo_sql} LIKE '{_PENDING}';"
    )


def _blob_triggers() -> tuple[tuple[str, str], ...]:
    old, new = "old", "new"
    return (
        (
            "blobs_backlog_ai",
            "CREATE TRIGGER blobs_backlog_ai AFTER INSERT ON blobs BEGIN\n"
            + _enter(f"{new}.hash")
            + "\n"
            + _pending_move("+", f"{new}.algo")
            + "\nEND",
        ),
        (
            "blobs_backlog_ad",
            "CREATE TRIGGER blobs_backlog_ad AFTER DELETE ON blobs BEGIN\n"
            + _leave(f"{old}.hash")
            + "\n"
            + _pending_move("-", f"{old}.algo")
            + "\nEND",
        ),
        (
            "blobs_backlog_au",
            "CREATE TRIGGER blobs_backlog_au AFTER UPDATE OF hash, algo, stored_bytes ON blobs "
            "BEGIN\n"
            + _leave(f"{old}.hash")
            + "\n"
            + _pending_move("-", f"{old}.algo")
            + "\n"
            + _enter(f"{new}.hash")
            + "\n"
            + _pending_move("+", f"{new}.algo")
            + "\nEND",
        ),
    )


def _reference_triggers(table: str) -> tuple[tuple[str, str], ...]:
    """A reference appearing takes its blob out of the total; the last one going puts it back."""
    old, new = "old", "new"
    return (
        (
            f"{table}_backlog_ai",
            f"CREATE TRIGGER {table}_backlog_ai AFTER INSERT ON {table} "
            f"WHEN {new}.original_ref IS NOT NULL BEGIN\n"
            + _leave(f"{new}.original_ref")
            + "\nEND",
        ),
        (
            f"{table}_backlog_ad",
            f"CREATE TRIGGER {table}_backlog_ad AFTER DELETE ON {table} "
            f"WHEN {old}.original_ref IS NOT NULL BEGIN\n"
            + _enter(f"{old}.original_ref")
            + "\nEND",
        ),
        (
            f"{table}_backlog_au",
            f"CREATE TRIGGER {table}_backlog_au AFTER UPDATE OF original_ref ON {table} "
            f"WHEN {old}.original_ref IS NOT {new}.original_ref BEGIN\n"
            + _enter(f"{old}.original_ref")
            + "\n"
            + _leave(f"{new}.original_ref")
            + "\nEND",
        ),
    )


_TRIGGERS: Final = (
    *_blob_triggers(),
    *_reference_triggers("documents"),
    *_reference_triggers("document_versions"),
)

TRIGGER_NAMES: Final[tuple[str, ...]] = tuple(name for name, _ in _TRIGGERS)
CREATE_TRIGGERS: Final[tuple[str, ...]] = tuple(sql for _, sql in _TRIGGERS)
DROP_TRIGGERS: Final[tuple[str, ...]] = tuple(
    f"DROP TRIGGER IF EXISTS {name}" for name in TRIGGER_NAMES
)

MEMBERS_FROM_BLOBS: Final = (
    f"INSERT INTO {_MEMBERS} (hash, bytes, sized) SELECT hash, "
    "CASE WHEN typeof(stored_bytes) = 'integer' THEN stored_bytes ELSE 0 END, "
    "typeof(stored_bytes) = 'integer' FROM blobs "
    f"WHERE algo NOT LIKE '{_PENDING}' AND " + _unreferenced("blobs.hash")
)
"""Every blob the triggers would count, for a migration to start from what is stored."""


def initial_totals_sql() -> str:
    """The singleton row for an empty ``blobs`` table; a migration then adds what is there."""
    return (
        f"INSERT INTO {_TOTALS} (id, unpublished_bytes, unsized_rows, gc_pending_rows, inexact) "
        "VALUES (1, 0, 0, 0, 0)"
    )
