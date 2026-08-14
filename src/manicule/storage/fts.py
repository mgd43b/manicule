"""The lexical index: an FTS5 table over ``chunks``, kept in step by triggers.

Two decisions carry the whole design.

**External content.** The virtual table stores only the inverted index and reads the text
back from ``chunks``. One copy of the corpus, in the authoritative store, and a whole class
of "the two copies disagree" bug that cannot occur.

**Triggers, not application code.** A trigger runs inside the same transaction as the row
change and cannot be bypassed by a migration, a repair path, or a write someone forgot about.
Application-level synchronization covers only the write paths its author remembered.
"""

from __future__ import annotations

from manicule.storage.models import FTS_TOKENIZER

FTS_TABLE = "chunks_fts"

FTS_SHADOW_TABLES = (
    "chunks_fts",
    "chunks_fts_data",
    "chunks_fts_idx",
    "chunks_fts_content",
    "chunks_fts_docsize",
    "chunks_fts_config",
)
"""What FTS5 creates behind the virtual table.

Alembic's autogenerate does not model virtual tables, so it sees every one of these as a
table present in the database and absent from the models — and helpfully emits
``drop_table``. ``env.py`` filters on this tuple.
"""

CREATE_FTS = f"""
CREATE VIRTUAL TABLE {FTS_TABLE} USING fts5(
    text,
    heading_text,
    content='chunks',
    content_rowid='seq',
    tokenize='{FTS_TOKENIZER}'
)
"""

CREATE_TRIGGERS = (
    """
    CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(rowid, text, heading_text)
        VALUES (new.seq, new.text, new.heading_text);
    END
    """,
    """
    CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, text, heading_text)
        VALUES ('delete', old.seq, old.text, old.heading_text);
        INSERT OR IGNORE INTO vector_tombstones(chunk_id, deleted_at)
        VALUES (old.id, datetime('now'));
    END
    """,
    """
    CREATE TRIGGER chunks_au AFTER UPDATE OF text, heading_text ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, text, heading_text)
        VALUES ('delete', old.seq, old.text, old.heading_text);
        INSERT INTO chunks_fts(rowid, text, heading_text)
        VALUES (new.seq, new.text, new.heading_text);
    END
    """,
)
"""Insert, delete and update, all inside the caller's transaction.

The delete trigger also records a vector tombstone. It is the one place a SQL-side timestamp
is used, because nothing orders ``vector_tombstones`` against another table's timestamps —
the sweep reads the whole table.

Hard-deleting a document reaches ``chunks`` through ``ON DELETE CASCADE``, and these fire on
cascaded deletes: verified directly on SQLite 3.51 at two cascade levels, with and without
``PRAGMA recursive_triggers``, which does not govern this case. ``tests/test_storage_fts.py``
holds it.
"""

DROP_TRIGGERS = (
    "DROP TRIGGER IF EXISTS chunks_au",
    "DROP TRIGGER IF EXISTS chunks_ad",
    "DROP TRIGGER IF EXISTS chunks_ai",
)

REBUILD_FTS = f"INSERT INTO {FTS_TABLE}({FTS_TABLE}) VALUES('rebuild')"
"""Rung 1 of the blast-radius ladder: rebuild the lexical index from ``chunks``, free."""

INTEGRITY_CHECK_FTS = f"INSERT INTO {FTS_TABLE}({FTS_TABLE}, rank) VALUES('integrity-check', 1)"
"""The ``rank`` argument is the whole check.

Two obvious alternatives do not work, and both were in an earlier draft of the design:

* ``COUNT(*)`` on each table can never disagree. ``chunks_fts`` is external-content, so
  counting it reads through to ``chunks``; the two are the same number by construction.
* A bare ``integrity-check`` passes on a completely empty index, because it verifies only
  that the index is internally consistent — and an empty index is consistent with itself.

Only the ``1`` argument compares the index against the content table. Checked directly: with
the triggers dropped and rows inserted, both counts agree, ``MATCH`` returns nothing, and a
bare ``integrity-check`` passes; this form raises. A silently empty lexical index halves
hybrid retrieval while every health check reports green.
"""

SEARCH_SQL = """
SELECT c.id AS chunk_id, bm25(chunks_fts, 1.0, 0.4) AS rank_score
FROM chunks_fts
JOIN chunks c ON c.seq = chunks_fts.rowid
JOIN documents d ON d.id = c.document_id
WHERE chunks_fts MATCH :match
  AND d.deleted_at IS NULL
  AND d.status = 'indexed'
  {extra}
ORDER BY bm25(chunks_fts, 1.0, 0.4)
LIMIT :limit
"""
"""One statement, so ``LIMIT`` applies *after* the joins and filters.

Running ``MATCH … LIMIT k`` first and filtering afterwards silently returns fewer than ``k``
live rows: deletion is deferred, so chunks of soft-deleted documents are still in the index
and still compete for those slots, as are chunks from other workspaces — which ``chunks_fts``
cannot distinguish, because the workspace lives on ``documents``. Measured on a fixture with
three matching live chunks against five soft-deleted and five cross-workspace, ``k = 5``
returned **zero** live results this way and all three the joined way.

Column weights ``1.0`` and ``0.4``: the breadcrumb has to be searchable for the same reason
it is prefixed to ``embed_text``, but it repeats on every chunk of a page, so at full weight
it floods term frequencies and depresses the IDF of the terms that identify the page.

``bm25()`` returns a *negative* number and a better match is more negative, so the ordering is
ascending. Its first argument must be the table's real name — it is parsed as a column
reference, so a query that aliases the FTS table fails with ``no such column``.
"""


def escape_match_query(text: str) -> str:
    """Turn user input into an FTS5 ``MATCH`` expression that cannot be an operator.

    FTS5's query language has operators (``AND``, ``OR``, ``NOT``, ``NEAR``, ``*``, ``:``,
    parentheses). A user typing ``NOT`` means the word, and a user typing an unbalanced
    quote means nothing at all — both currently produce a syntax error rather than a search.
    Each token is quoted, which makes it a literal phrase.

    Args:
        text: Whatever the user typed.

    Returns:
        A ``MATCH`` expression, or the empty string when nothing searchable remains.
    """
    tokens: list[str] = []
    for raw in text.split():
        cleaned = "".join(char for char in raw if char.isalnum() or char in "-_.'")
        if not cleaned:
            continue
        tokens.append('"' + cleaned.replace('"', '""') + '"')
    return " ".join(tokens)


__all__ = [
    "CREATE_FTS",
    "CREATE_TRIGGERS",
    "DROP_TRIGGERS",
    "FTS_SHADOW_TABLES",
    "FTS_TABLE",
    "INTEGRITY_CHECK_FTS",
    "REBUILD_FTS",
    "SEARCH_SQL",
    "escape_match_query",
]
