"""Migrations, and the two checks that keep them honest.

These are the enforcement half of #18. Without them, a model edited without a migration and a
downgrade that has never run both look exactly like a healthy repository.
"""

from __future__ import annotations

import importlib
import json
from typing import TYPE_CHECKING

import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import text

from manicule.core.provenance import PROVENANCE_KEY
from manicule.core.retrieval import Filter
from manicule.retrieval.hydration import visible_documents
from manicule.storage.autogen import include_name, include_object
from manicule.storage.engine import create_engine
from manicule.storage.fts import FTS_SHADOW_TABLES
from manicule.storage.migrator import alembic_config, current, downgrade, head_revision, upgrade
from manicule.storage.models import Base
from manicule.storage.vectors import LanceVectorStore
from tests.fakes import HashEmbedder
from tests.storage_helpers import make_chunk, make_document

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
async def test_marker_name_index_upgrades_from_already_applied_inventory_revision(
    data_dir: Path,
) -> None:
    """The successor must upgrade a database that already recorded e83 as applied."""
    engine = create_engine(data_dir)
    try:
        await upgrade(engine, revision="e83a21f96c40")
        async with engine.connect() as connection:
            before = (
                await connection.execute(text("PRAGMA table_info(acquisition_records)"))
            ).all()
        assert "marker_name" not in {row[1] for row in before}

        await upgrade(engine)
        async with engine.connect() as connection:
            after = (
                await connection.execute(text("PRAGMA table_info(acquisition_records)"))
            ).all()
            indexes = (
                await connection.execute(text("PRAGMA index_list(acquisition_records)"))
            ).all()
        assert "marker_name" in {row[1] for row in after}
        assert "ix_acquisition_records_marker_name" in {row[1] for row in indexes}
        assert await current(engine) == head_revision()
    finally:
        await engine.dispose()


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
        assert await _document_values(engine, "publication_id") == {"publication_id": "legacy"}, (
            "pre-publication documents must retain vectors under the legacy generation"
        )
        async with engine.connect() as connection:
            migrated_vector_ids = (
                await connection.execute(text("SELECT count(*) FROM chunks WHERE vector_id = id"))
            ).scalar_one()
        assert migrated_vector_ids == before["chunks"], (
            "every legacy chunk's logical id was also its physical vector id"
        )
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
async def test_downgrade_refuses_to_expose_retired_real_vector_generations(
    data_dir: Path,
) -> None:
    """Without the SQLite pointer, old code could hydrate either row for one logical chunk."""
    from manicule.storage.docstore import SqliteDocStore  # noqa: PLC0415

    engine = create_engine(data_dir)
    vectors = LanceVectorStore(data_dir / "vectors")
    try:
        await upgrade(engine)
        store = SqliteDocStore(engine)
        await store.ensure_workspace()
        document = make_document(source_id="generated").model_copy(
            update={"publication_id": "active-publication"}
        )
        document = await store.upsert_document(document)
        active = make_chunk(document, 0, "current text")
        retired = active.model_copy(update={"embed_text": "retired embedding input"})
        await store.replace_chunks(document.id, [active])
        await vectors.ensure_ready(HashEmbedder().fingerprint)
        await vectors.upsert([retired], [[0.1] * 5], publication_id="retired-publication")
        await vectors.upsert([active], [[0.2] * 5], publication_id=document.publication_id)

        with pytest.raises(RuntimeError, match="refusing to downgrade atomic publications"):
            await downgrade(engine, "d4a90c7e15b3")

        assert await current(engine) == head_revision()
        candidates = await vectors.search([0.0] * 5, 10)
        visible = await visible_documents(
            store,
            Filter(workspace_ids=frozenset({store.workspace_id})),
            {document.id},
        )
        admitted = [
            candidate
            for candidate in candidates
            if visible.get(candidate.chunk.document_id) == candidate.publication_id
        ]
        assert [candidate.publication_id for candidate in admitted] == [document.publication_id]
    finally:
        await vectors.teardown()
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


async def test_glossary_lineage_arrives_empty_so_the_first_release_repairs_rather_than_trusts(
    data_dir: Path,
) -> None:
    """``documents.glossary_fp`` over a database whose glossary was detected by older rules.

    **Not backfilled, and that is the decision the revision is making.** Writing the installed
    fingerprint into existing rows would be one statement and would assert that entries detected
    before anything recorded a detector came out of the rules installed now — which is false for
    every corpus indexed before this column, and is exactly the plausible falsehood the
    fingerprints exist to prevent. ``NULL`` means "nobody has looked", the repair selects on it,
    and the price is one sweep over stored chunks with no parser and no embedder in it.

    The seeded glossary row is left alone on the way through, both directions. Losing a
    definition to a lineage migration would be the migration causing the harm it is adding a
    column to detect.
    """
    engine = create_engine(data_dir)
    try:
        await upgrade(engine, revision=_first_revision())
        await _seed(engine)
        counts = await _row_counts(engine)
        original = await _document_values(engine)

        await upgrade(engine)
        assert await _document_values(engine, "glossary_fp") == {"glossary_fp": None}, (
            "an existing document must not be claimed to have been read by today's detector"
        )
        assert await _row_counts(engine) == counts, "no entry may be lost to a column"

        async with engine.begin() as connection:
            await connection.execute(
                text("UPDATE documents SET glossary_fp = :fp"), {"fp": _SEEDED_GLOSSARY_FP}
            )
        assert await _document_values(engine, "glossary_fp") == {"glossary_fp": _SEEDED_GLOSSARY_FP}

        # An indexed column, so the downgrade has to drop the index first or SQLite refuses —
        # at the moment somebody is downgrading under pressure, which is the worst time to find
        # out that a path nobody has run is not a path.
        await downgrade(engine, _first_revision())
        assert await _row_counts(engine) == counts, "the downgrade must not cascade"
        assert await _document_values(engine) == original, (
            "dropping glossary_fp must not disturb the columns beside it"
        )

        await upgrade(engine)
        assert await _document_values(engine, "glossary_fp") == {"glossary_fp": None}
        assert await _row_counts(engine) == counts
    finally:
        await engine.dispose()


_SEEDED_GLOSSARY_FP = (
    '{"detector":"deterministic","middleware":[],"rules":"sha256:0000000000000000"}'
)
"""A glossary lineage value the assertions above compare against something specific.

A placeholder would defeat the test the same way :data:`_SEEDED_PARSE_FP`'s docstring describes:
a migration that rewrote every lineage column to the empty string would pass an assertion that
only checked the column still existed.
"""


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


_REVISION = "manicule.storage.migrations.versions.20260813_b2e6d0c94a17_page_keyed_identity"
"""The re-key revision, addressed by its module path.

Imported through :func:`importlib.import_module` rather than with an ``import`` statement,
because a revision's module name begins with a date and is therefore not an identifier. Imported
at all, rather than repeating its constant here, so that renaming the key in the migration breaks
this test instead of quietly leaving it asserting a string nothing writes.
"""


def _previous_identity_key() -> str:
    return str(importlib.import_module(_REVISION).PREVIOUS_IDENTITY)


_PAGE = "1002"
_CORPUS_PATH = "/corpus/pages/1002.html"
_DECLARED_METADATA = json.dumps(
    {
        "parser_used": "web",
        PROVENANCE_KEY: {
            "source": {
                "title": "Retry Runbook",
                "canonical_uri": "https://docs.example.test/pages/1002",
                "source_id": _PAGE,
                "version": "7",
                "created_at": None,
                "modified_at": None,
                "content_type": "",
                "section_path": ["ENG"],
            },
            "snapshot": {"path": "pages/1002.html", "retrieved_at": None},
            "unavailable_reason": "",
        },
    }
)
"""A document keyed on its path whose manifest declares a page id: the shape that moves.

The record is spelled out as the stored JSON rather than built by calling the application, so
this fixture states what is *on disk* in a corpus somebody upgraded — which is what the migration
reads, and which a helper that happened to change shape would silently stop describing.
"""


async def _seed_declaring(engine: AsyncEngine) -> None:
    """A second document, path-keyed, declaring a page id, with curation hung off it.

    Curation is the point. ``collection_documents`` and ``document_tags`` are what a re-key
    destroys if it is done by inserting the new row and deleting the old, so both are seeded and
    both are read back on the other side.
    """
    statements = (
        "INSERT INTO documents (id, workspace_id, source, source_id, uri, title, media_type, "
        "content_hash, version_token, status, status_detail, chunk_fp, embed_fp, metadata, "
        "created_at, updated_at, indexed_at) "
        "VALUES ('d2', 'w', 'handbook', :source_id, 'file:///corpus/pages/1002.html', "
        "'Retry Runbook', 'text/html', 'h2', 'v9', 'indexed', NULL, NULL, NULL, :metadata, "
        "'2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', "
        "'2026-01-02T00:00:00+00:00')",
        "INSERT INTO chunks (id, document_id, text, embed_text, heading_text, heading_path, "
        "kind, position, token_count, anchor, metadata, created_at) "
        "VALUES ('c3', 'd2', 'body', 'body', '', '[]', 'prose', 0, 1, '{}', '{}', "
        "'2026-01-01T00:00:00+00:00')",
        "INSERT INTO document_versions (id, document_id, version, content_hash, created_at) "
        "VALUES ('v2', 'd2', 1, 'h2', '2026-01-01T00:00:00+00:00')",
        "INSERT INTO document_tags (document_id, tag_id) VALUES ('d2', 't1')",
        "INSERT INTO collection_documents (collection_id, document_id) VALUES ('k1', 'd2')",
        "INSERT INTO glossary_entries (id, document_id, chunk_id, acronym, display, expansion, "
        "location, form, confidence, created_at) "
        "VALUES ('g2', 'd2', 'c3', 'RPS', 'RPS', 'Retries Per Second', 'Body', 'em_dash', "
        "0.9, '2026-01-01T00:00:00+00:00')",
    )
    async with engine.begin() as connection:
        await connection.execute(
            text(statements[0]), {"source_id": _CORPUS_PATH, "metadata": _DECLARED_METADATA}
        )
        for statement in statements[1:]:
            await connection.execute(text(statement))


async def _children_of(engine: AsyncEngine, identifier: str) -> dict[str, list[str]]:
    """Every child row pointing at ``identifier``, by table, as the ids they carry.

    Ids rather than counts. A migration that re-pointed a child at the *wrong* document keeps the
    count identical, and a count is what this project's worst near-miss would also have passed —
    on an empty database, where every count was zero on both sides.
    """
    reads = {
        "chunks": "SELECT id FROM chunks WHERE document_id = :id ORDER BY id",
        "document_versions": (
            "SELECT id FROM document_versions WHERE document_id = :id ORDER BY id"
        ),
        "glossary_entries": "SELECT id FROM glossary_entries WHERE document_id = :id ORDER BY id",
        "collection_documents": (
            "SELECT collection_id FROM collection_documents WHERE document_id = :id "
            "ORDER BY collection_id"
        ),
        "document_tags": "SELECT tag_id FROM document_tags WHERE document_id = :id ORDER BY tag_id",
    }
    async with engine.connect() as connection:
        return {
            table: [row[0] for row in (await connection.execute(text(sql), {"id": identifier}))]
            for table, sql in reads.items()
        }


@pytest.mark.contract
async def test_a_declared_page_identity_is_re_keyed_and_keeps_its_curation(
    data_dir: Path,
) -> None:
    """The re-key, over a populated database, up and back.

    **Values, not counts**, and running the mutations showed what that is worth and what it is
    not. Three of the ways this migration could go wrong are already caught by something else:
    :func:`~manicule.storage.migrator.upgrade` runs ``PRAGMA foreign_key_check`` after every
    migration, so a child left behind is a ``StorageMigrationError`` before any assertion here
    runs; and SQLite's own UNIQUE constraints catch a child moved onto a key another row holds.
    Claiming this test catches the cascade would have been claiming a colleague's work.

    What it catches that neither does: a child re-pointed at a **valid, existing, wrong**
    document. The foreign key resolves, no constraint is violated, the row counts are identical
    on both sides — and a person's collection now contains a page they never put in it.
    Confirmed by sending one child table to the other seeded document:

        AssertionError: the curation did not travel with the document
        {'glossary_entries': []} != {'glossary_entries': ['g2']}

    That is why the curation is read back *by the ids it points at* rather than counted.
    """
    from manicule.core.ids import document_id  # noqa: PLC0415 - the derivation under test

    engine = create_engine(data_dir)
    try:
        await upgrade(engine, revision="5f1c8a34b7d9")
        await _seed(engine)
        await _seed_declaring(engine)
        before = await _children_of(engine, "d2")
        assert before["collection_documents"] == ["k1"], (
            "the seed must actually seed the curation, or this test proves nothing"
        )
        assert before["document_tags"] == ["t1"]
        assert before["chunks"] == ["c3"]
        assert before["glossary_entries"] == ["g2"]
        assert before["document_versions"] == ["v2"]
        counts = await _row_counts(engine)

        await upgrade(engine)

        moved = document_id("w", "handbook", _PAGE)
        assert await _row_counts(engine) == counts, "the re-key deleted rows"
        row = await _document_row(engine, moved)
        assert row is not None, "the document was not re-keyed onto its declared identity"
        assert row["source_id"] == _PAGE
        assert row["title"] == "Retry Runbook", "the re-key rewrote a column it was not asked to"
        assert row["content_hash"] == "h2"
        recorded = json.loads(str(row["metadata"]))[_previous_identity_key()]
        assert recorded["source_id"] == _CORPUS_PATH
        assert recorded["document_id"] == "d2"
        assert recorded["content_hash"] == "h2", (
            "the seeded document is text/html, so its stored text is about to be wrong and the "
            "record has to say so — this is what `doctor` reads to know a sync is still owed"
        )
        assert await _children_of(engine, moved) == before, (
            "the curation did not travel with the document, which is the whole point"
        )
        assert await _children_of(engine, "d2") == {table: [] for table in before}, (
            "children were left pointing at an identity that no longer exists"
        )
        # The document this revision has no business touching is untouched, including its key.
        assert await _document_row(engine, "d1") is not None

        await downgrade(engine, "5f1c8a34b7d9")

        assert await _row_counts(engine) == counts, "the downgrade deleted rows"
        back = await _document_row(engine, "d2")
        assert back is not None, "the downgrade did not put the document back"
        assert back["source_id"] == _CORPUS_PATH
        assert _previous_identity_key() not in json.loads(str(back["metadata"]))
        assert await _children_of(engine, "d2") == before
    finally:
        await engine.dispose()


@pytest.mark.contract
async def test_a_re_key_onto_an_identity_already_taken_is_refused(data_dir: Path) -> None:
    """Two documents cannot become one, so the second stays where it is.

    Overwriting an occupied primary key would destroy a document this revision has no way to put
    back — and it is the shape that actually arises: a page ingested under its page id from one
    root and under its path from another, or two manifests declaring one id. Left alone and
    logged, which is what ``doctor``'s ``document-identity`` check keeps reporting.
    """
    engine = create_engine(data_dir)
    try:
        await upgrade(engine, revision="5f1c8a34b7d9")
        await _seed(engine)
        await _seed_declaring(engine)
        # A third document already sitting on the identity `d2` would move onto.
        from manicule.core.ids import document_id  # noqa: PLC0415

        occupied = document_id("w", "handbook", _PAGE)
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    "INSERT INTO documents (id, workspace_id, source, source_id, uri, title, "
                    "media_type, content_hash, status, metadata, created_at, updated_at) "
                    "VALUES (:id, 'w', 'handbook', :page, 'https://docs.example.test/pages/1002', "
                    "'Retry Runbook', 'text/html', 'h3', 'indexed', '{}', "
                    "'2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')"
                ),
                {"id": occupied, "page": _PAGE},
            )
        counts = await _row_counts(engine)
        before = await _children_of(engine, "d2")

        await upgrade(engine)

        assert await _row_counts(engine) == counts, "a collision cost a document"
        stayed = await _document_row(engine, "d2")
        assert stayed is not None, "the colliding document was moved onto an occupied key"
        assert stayed["source_id"] == _CORPUS_PATH
        assert await _children_of(engine, "d2") == before
        occupant = await _document_row(engine, occupied)
        assert occupant is not None
        assert occupant["content_hash"] == "h3", "the occupant was overwritten"
    finally:
        await engine.dispose()


async def _document_row(engine: AsyncEngine, identifier: str) -> dict[str, object] | None:
    async with engine.connect() as connection:
        row = (
            (
                await connection.execute(
                    text(
                        "SELECT id, source, source_id, title, content_hash, metadata "
                        "FROM documents WHERE id = :id"
                    ),
                    {"id": identifier},
                )
            )
            .mappings()
            .first()
        )
        return dict(row) if row is not None else None


@pytest.mark.contract
async def test_a_page_keyed_on_its_path_on_purpose_is_not_re_keyed(data_dir: Path) -> None:
    """The migration must not move a document the connector will never look up that way.

    An enriched export with no manifest declares its page id in its provenance record and is
    keyed on its path deliberately, because identity is settled at discovery and discovery reads
    the manifest rather than the document. Re-keying it moves the row onto an identity nothing
    queries: the next sync discovers the file under its path, finds no row, and creates a second
    document — the exact duplication this migration exists to prevent.

    Found by running an ingest and a migration over a real corpus. Nothing failed; ``doctor``
    simply said something that was not true, and following it to the migration showed why.
    """
    from manicule.connectors.enriched import ENRICHED_KEY, AdapterOutcome  # noqa: PLC0415

    engine = create_engine(data_dir)
    try:
        await upgrade(engine, revision="5f1c8a34b7d9")
        await _seed(engine)
        await _seed_declaring(engine)
        held = json.loads(_DECLARED_METADATA)
        held[ENRICHED_KEY] = {"outcome": AdapterOutcome.IDENTITY_NOT_APPLIED.value}
        async with engine.begin() as connection:
            await connection.execute(
                text("UPDATE documents SET metadata = :metadata WHERE id = 'd2'"),
                {"metadata": json.dumps(held)},
            )
        before = await _children_of(engine, "d2")

        await upgrade(engine)

        stayed = await _document_row(engine, "d2")
        assert stayed is not None, "a page keyed on its path on purpose was moved"
        assert stayed["source_id"] == _CORPUS_PATH
        assert _previous_identity_key() not in json.loads(str(stayed["metadata"])), (
            "the migration recorded a move it did not make"
        )
        assert await _children_of(engine, "d2") == before
    finally:
        await engine.dispose()


@pytest.mark.contract
async def test_a_document_whose_parse_is_unaffected_records_no_staleness(data_dir: Path) -> None:
    """A mirrored PDF with a manifest is re-keyed and its chunks stay exactly right.

    Recording a staleness marker for it would make ``doctor`` report work that never needs doing
    and never clears, because its ``content_hash`` never moves — a warning nobody can satisfy,
    which is the shape this project keeps refusing. Only ``text/html`` is affected, because that
    is the routing this change moves.
    """
    engine = create_engine(data_dir)
    try:
        await upgrade(engine, revision="5f1c8a34b7d9")
        await _seed(engine)
        await _seed_declaring(engine)
        async with engine.begin() as connection:
            await connection.execute(
                text("UPDATE documents SET media_type = 'application/pdf' WHERE id = 'd2'")
            )

        await upgrade(engine)

        from manicule.core.ids import document_id  # noqa: PLC0415

        row = await _document_row(engine, document_id("w", "handbook", _PAGE))
        assert row is not None, "a manifest-bearing PDF was not re-keyed"
        recorded = json.loads(str(row["metadata"]))[_previous_identity_key()]
        assert recorded["document_id"] == "d2", "the move itself was not recorded"
        assert "content_hash" not in recorded, (
            "a document whose parse is unaffected was marked as owing a re-parse"
        )
    finally:
        await engine.dispose()
