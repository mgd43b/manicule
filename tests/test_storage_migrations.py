"""Migrations, and the two checks that keep them honest.

These are the enforcement half of #18. Without them, a model edited without a migration and a
downgrade that has never run both look exactly like a healthy repository.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import text

from manicule.storage.autogen import include_name, include_object
from manicule.storage.engine import create_engine
from manicule.storage.fts import FTS_SHADOW_TABLES
from manicule.storage.migrator import alembic_config, current, downgrade, head_revision, upgrade
from manicule.storage.models import Base

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.engine import Connection
    from sqlalchemy.ext.asyncio import AsyncEngine


def _diff(connection: Connection) -> list[object]:
    context = MigrationContext.configure(
        connection,
        opts={
            "compare_type": True,
            "compare_server_default": True,
            "include_name": include_name,
            "include_object": include_object,
            "target_metadata": Base.metadata,
        },
    )
    return list(compare_metadata(context, Base.metadata))


@pytest.mark.contract
async def test_the_models_and_the_migrations_agree(engine: AsyncEngine) -> None:
    """``alembic check``, as a test.

    A model edited without a migration produces a schema that exists only in the developer's
    working tree; the first fresh database finds the columns missing.
    """
    async with engine.connect() as connection:
        differences = await connection.run_sync(_diff)
    assert differences == [], (
        f"the models have drifted from the migrations: {differences}. "
        f"Generate a revision with `uv run alembic revision --autogenerate`."
    )


@pytest.mark.contract
async def test_every_revision_downgrades_and_upgrades_back_to_the_same_schema(
    data_dir: Path,
) -> None:
    """A downgrade that has never run is not a downgrade path.

    It is untested code that will be needed exactly once, under pressure. This matters more
    on SQLite than elsewhere: batch mode implements every constraint change as
    create-copy-swap, and a batch downgrade that cannot name the constraint it drops fails in
    a way only running it reveals.
    """
    script = ScriptDirectory.from_config(alembic_config())
    revisions = [revision.revision for revision in script.walk_revisions()]

    engine = create_engine(data_dir)
    try:
        await upgrade(engine)
        assert await current(engine) == head_revision()
        before = await _schema_snapshot(engine)

        for revision in revisions:
            down_to = script.get_revision(revision).down_revision or "base"
            await downgrade(engine, down_to if isinstance(down_to, str) else "base")
            await upgrade(engine)

        assert await current(engine) == head_revision()
        assert await _schema_snapshot(engine) == before, (
            "the schema after a downgrade/upgrade round trip differs from the schema before it"
        )
    finally:
        await engine.dispose()


@pytest.mark.contract
async def test_a_migration_over_a_populated_database_keeps_the_rows(data_dir: Path) -> None:
    """Every revision runs over a database with content in it, not an empty one.

    An empty database hides the failure mode that matters most on SQLite. A batch rebuild is
    CREATE-temp, INSERT…SELECT, ``DROP TABLE``, RENAME — and with ``PRAGMA foreign_keys=ON``,
    which this project sets on every connection, that ``DROP`` performs an implicit
    ``DELETE FROM`` and fires every ``ON DELETE CASCADE`` pointing at the table. Children are
    emptied and the migration reports success; where the cascade reaches ``chunks`` it also
    trips the FTS trigger, and the half-finished rebuild leaves a temporary table that makes
    every retry fail.

    So this seeds one row in each table a cascade would reach, migrates to head and back, and
    counts them. It is deliberately not a test of one revision: it is what stops the *next*
    table rebuild shipping the same way.
    """
    engine = create_engine(data_dir)
    try:
        await upgrade(engine, revision=_first_revision())
        await _seed(engine)
        before = await _row_counts(engine)
        assert before["chunks"] == 2, "the seed must actually seed, or this test proves nothing"
        columns = await _document_values(engine)
        assert columns["chunk_fp"] == _SEEDED_CHUNK_FP, "the seed must set the lineage it checks"

        await upgrade(engine)
        assert await _row_counts(engine) == before, (
            "a migration must not delete rows it was not asked to delete"
        )
        assert await _document_values(engine) == columns, (
            "a migration must not rewrite the columns it was not asked to touch"
        )

        await downgrade(engine, _first_revision())
        assert await _row_counts(engine) == before, "and neither must the downgrade"
        assert await _document_values(engine) == columns, "and neither must the downgrade"
    finally:
        await engine.dispose()


@pytest.mark.contract
async def test_parse_lineage_arrives_empty_and_leaves_the_rest_alone(data_dir: Path) -> None:
    """``documents.parse_fp`` over a database with documents already in it.

    Two claims, and the second is the one an empty database cannot make. The column arrives
    ``NULL`` on every existing row — not backfilled with today's parser versions, because
    nothing knows which versions produced text extracted months ago, and a lineage that
    asserts what nobody knows is worse than one that admits it does not. And adding it leaves
    every other column of every existing row exactly as it was, including the two lineages
    that were already there.

    The downgrade is checked the same way and for a sharper reason: it drops an *indexed*
    column, which SQLite refuses outright unless the index goes first. A downgrade nobody has
    run is not a downgrade path, and this one would fail at the moment it was needed.
    """
    engine = create_engine(data_dir)
    try:
        await upgrade(engine, revision=_first_revision())
        await _seed(engine)
        counts = await _row_counts(engine)
        original = await _document_values(engine)

        await upgrade(engine)
        assert await _document_values(engine, "parse_fp") == {"parse_fp": None}, (
            "an existing document must not be claimed to have been parsed by today's libraries"
        )

        async with engine.begin() as connection:
            await connection.execute(
                text("UPDATE documents SET parse_fp = :fp"), {"fp": _SEEDED_PARSE_FP}
            )
        assert await _document_values(engine, "parse_fp") == {"parse_fp": _SEEDED_PARSE_FP}

        await downgrade(engine, _first_revision())
        assert await _row_counts(engine) == counts, "the downgrade must not cascade"
        assert await _document_values(engine) == original, (
            "dropping parse_fp must not disturb the columns beside it"
        )

        await upgrade(engine)
        assert await _document_values(engine, "parse_fp") == {"parse_fp": None}
        assert await _row_counts(engine) == counts
    finally:
        await engine.dispose()


_SEEDED_GLOSSARY = (
    "INSERT INTO glossary_entries (id, document_id, chunk_id, acronym, display, expansion, "
    "location, form, confidence, created_at) "
    "VALUES ('g1', 'd1', 'c1', 'NOW', 'NOW', 'Network Operations Workspace', 'Glossary', "
    "'em_dash', 0.95, '2026-01-01T00:00:00+00:00')",
    "INSERT INTO glossary_aliases (entry_id, \"key\") VALUES ('g1', 'NETOPS')",
)
"""One entry and one alias, hung off the seeded document and its first chunk.

Both foreign keys are exercised, which is the point: the entry points at ``chunks`` and the
alias points at the entry, so this is the shape whose ``DROP`` has somewhere to cascade to.
"""


@pytest.mark.contract
async def test_the_glossary_tables_arrive_empty_and_their_removal_cascades_nowhere(
    data_dir: Path,
) -> None:
    """The glossary revision, over a database that already has documents in it.

    Three claims, and the third is the one an empty database cannot make.

    **The tables arrive empty.** Definitions come from parsing text, so an existing corpus gains
    a glossary on its next ingest. Deriving entries inside a migration would freeze one copy of
    the detector's rules into a revision that can never be changed.

    **Adding them disturbs nothing.** Every seeded row and every column of the seeded document
    is still exactly as it was.

    **Removing them cascades nowhere.** ``glossary_entries`` has foreign keys to ``documents``
    and to ``chunks``, and the cascades point *inwards* — but a downgrade drops the table while
    it is populated, and this project's worst near-miss was a migration whose ``DROP`` emptied
    four tables and reported success. So the glossary is populated before the downgrade runs,
    and the counts are read on the other side of it.
    """
    engine = create_engine(data_dir)
    try:
        await upgrade(engine, revision=_first_revision())
        await _seed(engine)
        counts = await _row_counts(engine)
        columns = await _document_values(engine)

        await upgrade(engine)
        assert await _glossary_counts(engine) == {"glossary_entries": 0, "glossary_aliases": 0}, (
            "an existing corpus must not be claimed to have a glossary nobody extracted"
        )
        assert await _row_counts(engine) == counts
        assert await _document_values(engine) == columns

        async with engine.begin() as connection:
            for statement in _SEEDED_GLOSSARY:
                await connection.execute(text(statement))
        assert await _glossary_counts(engine) == {"glossary_entries": 1, "glossary_aliases": 1}, (
            "the seed must actually seed, or the downgrade below proves nothing"
        )

        await downgrade(engine, _first_revision())
        assert await _row_counts(engine) == counts, (
            "dropping a populated glossary must not take the documents or chunks with it"
        )
        assert await _document_values(engine) == columns

        await upgrade(engine)
        assert await _glossary_counts(engine) == {"glossary_entries": 0, "glossary_aliases": 0}
        assert await _row_counts(engine) == counts
    finally:
        await engine.dispose()


async def _glossary_counts(engine: AsyncEngine) -> dict[str, int]:
    async with engine.connect() as connection:
        return {
            # S608: both names are literals in this function, and a table name cannot be a
            # bind parameter.
            table: (
                await connection.execute(text(f"SELECT count(*) FROM {table}"))  # noqa: S608
            ).scalar_one()
            for table in ("glossary_entries", "glossary_aliases")
        }


def _first_revision() -> str:
    script = ScriptDirectory.from_config(alembic_config())
    return next(r.revision for r in script.walk_revisions() if r.down_revision is None)


_SEEDED_CHUNK_FP = '{"chunker":"structural","max_tokens":512}'
_SEEDED_EMBED_FP = '{"dimension":1024,"model_id":"BAAI/bge-m3"}'
_SEEDED_PARSE_FP = '{"libraries":{"pypdfium2":"5.12.1"},"parser":"pdf","version":"1"}'
_SEEDED_METADATA = '{"parser_used": "markdown"}'
"""Lineage values the seed writes, so the assertions compare against something specific.

Placeholders would defeat the test twice over: a migration that rewrote every ``chunk_fp`` to
the empty string would pass an assertion that only checked the column still existed, and a
migration that swapped two columns would pass one where both held the same string.
"""


async def _seed(engine: AsyncEngine) -> None:
    """One row in every table a cascade from ``documents`` would reach.

    The document carries a value in every column an assertion later reads — lineage,
    version token, metadata — because a rebuild that keeps the row and empties a column
    counts identically to one that did nothing wrong.
    """
    document = (
        "INSERT INTO documents (id, workspace_id, source, source_id, uri, title, media_type, "
        "content_hash, version_token, status, status_detail, chunk_fp, embed_fp, metadata, "
        "created_at, updated_at, indexed_at) "
        "VALUES ('d1', 'w', 'fs', 's1', 'file:///a.md', 'A', 'text/markdown', 'h', 'v7', "
        "'indexed', NULL, :chunk_fp, :embed_fp, :metadata, "
        "'2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', "
        "'2026-01-02T00:00:00+00:00')"
    )
    # Bound rather than interpolated because a canonical fingerprint contains colons, and
    # ``text()`` reads ``:512`` in ``{"max_tokens":512}`` as a bind parameter it was never
    # given — which fails loudly here and would fail identically in any code that built a
    # statement around one.
    statements = (
        "INSERT INTO workspaces (id, name, mode, settings, created_at) "
        "VALUES ('w', 'w', 'personal', '{}', '2026-01-01T00:00:00+00:00')",
        "INSERT INTO chunks (id, document_id, text, embed_text, heading_text, heading_path, "
        "kind, position, token_count, anchor, metadata, created_at) "
        "VALUES ('c1', 'd1', 'one', 'one', '', '[]', 'prose', 0, 1, '{}', '{}', "
        "'2026-01-01T00:00:00+00:00')",
        "INSERT INTO chunks (id, document_id, text, embed_text, heading_text, heading_path, "
        "kind, position, token_count, anchor, metadata, created_at) "
        "VALUES ('c2', 'd1', 'two', 'two', '', '[]', 'prose', 1, 1, '{}', '{}', "
        "'2026-01-01T00:00:00+00:00')",
        "INSERT INTO document_versions (id, document_id, version, content_hash, created_at) "
        "VALUES ('v1', 'd1', 1, 'h', '2026-01-01T00:00:00+00:00')",
        "INSERT INTO tags (id, workspace_id, name) VALUES ('t1', 'w', 'tag')",
        "INSERT INTO document_tags (document_id, tag_id) VALUES ('d1', 't1')",
        "INSERT INTO collections (id, workspace_id, name, created_at) "
        "VALUES ('k1', 'w', 'k', '2026-01-01T00:00:00+00:00')",
        "INSERT INTO collection_documents (collection_id, document_id) VALUES ('k1', 'd1')",
    )
    async with engine.begin() as connection:
        await connection.execute(text(statements[0]))
        await connection.execute(
            text(document),
            {
                "chunk_fp": _SEEDED_CHUNK_FP,
                "embed_fp": _SEEDED_EMBED_FP,
                "metadata": _SEEDED_METADATA,
            },
        )
        for statement in statements[1:]:
            await connection.execute(text(statement))


_CHECKED_COLUMNS = (
    "id",
    "workspace_id",
    "source",
    "source_id",
    "uri",
    "title",
    "media_type",
    "content_hash",
    "version_token",
    "status",
    "chunk_fp",
    "embed_fp",
    "metadata",
    "indexed_at",
)
"""What the seeded document must still say after a migration and after its downgrade.

Named explicitly rather than read with ``SELECT *``: the column set changes across revisions,
so a wildcard would compare a different shape on each side of the round trip and quietly
compare nothing.
"""


async def _document_values(engine: AsyncEngine, *columns: str) -> dict[str, object]:
    """The seeded document's columns, as stored. Defaults to :data:`_CHECKED_COLUMNS`."""
    wanted = columns or _CHECKED_COLUMNS
    async with engine.connect() as connection:
        row = (
            # S608: every name comes from a literal tuple in this module, and a column name
            # cannot be a bind parameter.
            await connection.execute(
                text(f"SELECT {', '.join(wanted)} FROM documents WHERE id = 'd1'")  # noqa: S608
            )
        ).one()
    return dict(zip(wanted, row, strict=True))


async def _row_counts(engine: AsyncEngine) -> dict[str, int]:
    tables = (
        "documents",
        "chunks",
        "document_versions",
        "document_tags",
        "collection_documents",
    )
    async with engine.connect() as connection:
        return {
            # S608: every name comes from the literal tuple above, and a table name cannot be
            # a bind parameter.
            table: (
                await connection.execute(text(f"SELECT count(*) FROM {table}"))  # noqa: S608
            ).scalar_one()
            for table in tables
        }


async def test_downgrading_to_base_leaves_nothing_of_ours_behind(data_dir: Path) -> None:
    """A downgrade that leaves tables behind makes the next upgrade fail on a fresh start."""
    engine = create_engine(data_dir)
    try:
        await upgrade(engine)
        await downgrade(engine, "base")
        async with engine.connect() as connection:
            names = set(
                (
                    await connection.execute(
                        text("SELECT name FROM sqlite_master WHERE type='table'")
                    )
                )
                .scalars()
                .all()
            )
        assert "documents" not in names
        assert "chunks_fts" not in names, "the virtual table must be dropped with the rest"
        assert not any(name.startswith("chunks_fts") for name in names)
    finally:
        await engine.dispose()


def test_autogenerate_never_proposes_dropping_the_lexical_index() -> None:
    """Alembic does not model virtual tables, so it sees them as tables to remove.

    Without this filter every autogenerated revision begins by deleting the FTS index and its
    five shadow tables.
    """
    for name in FTS_SHADOW_TABLES:
        assert not include_name(name, "table", {})
        assert not include_object(Base.metadata, name, "table", True, None)
    assert include_name("documents", "table", {})
    assert include_name("id", "column", {})


def test_every_constraint_has_a_name_a_migration_can_refer_to() -> None:
    """Batch mode must *name* the constraint it drops, and SQLite generates anonymous ones.

    A project that discovers this later cannot write the migration it needs.
    """
    unnamed: list[str] = []
    for table in Base.metadata.sorted_tables:
        for constraint in table.constraints:
            if constraint.name is None or str(constraint.name).startswith("_unnamed_"):
                unnamed.append(f"{table.name}.{type(constraint).__name__}")
        for index in table.indexes:
            if index.name is None:
                unnamed.append(f"{table.name}.Index")
    assert unnamed == []


async def _schema_snapshot(engine: AsyncEngine) -> list[tuple[str, str]]:
    """Every object the schema declares, as text, ordered.

    Comparing the SQL rather than a reflection because a round trip that silently drops a
    ``CHECK`` still reflects as a table with the right columns.
    """
    async with engine.connect() as connection:
        rows = (
            await connection.execute(
                text(
                    "SELECT type, name, COALESCE(sql, '') FROM sqlite_master "
                    "WHERE name NOT LIKE 'sqlite_%' AND name <> 'alembic_version' "
                    "ORDER BY type, name"
                )
            )
        ).all()
    return [(f"{row[0]}:{row[1]}", _canonical_ddl(str(row[2]))) for row in rows]


def _canonical_ddl(sql: str) -> str:
    """DDL with its table-level constraints sorted, and everything else left alone.

    Constraint *order* in a ``CREATE TABLE`` carries no meaning, and SQLAlchemy's batch
    rebuild does not preserve it: the constraints of a reflected table live in a set, so two
    runs of the same migration can emit the two foreign keys either way round. Comparing raw
    text would make every batch migration's round trip fail intermittently — a flaky test,
    which is a bug in the test.

    Sorting is the narrowest possible relaxation. Column definitions keep their order, every
    constraint still has to be present, and its text still has to match character for
    character — so a rebuild that drops a ``CHECK``, widens a column or loses a cascade is
    caught exactly as before.
    """
    collapsed = " ".join(sql.split())
    opened = collapsed.find("(")
    if not collapsed.upper().startswith("CREATE TABLE") or opened < 0:
        return collapsed
    head, body = collapsed[:opened], collapsed[opened + 1 : collapsed.rfind(")")]

    clauses: list[str] = []
    depth = 0
    current = ""
    for character in body:
        if character == "," and depth == 0:
            clauses.append(current.strip())
            current = ""
            continue
        depth += (character == "(") - (character == ")")
        current += character
    clauses.append(current.strip())

    columns = [clause for clause in clauses if not clause.upper().startswith("CONSTRAINT")]
    constraints = sorted(clause for clause in clauses if clause.upper().startswith("CONSTRAINT"))
    return f"{head}( {', '.join([*columns, *constraints])} )"
