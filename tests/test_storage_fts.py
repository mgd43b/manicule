"""The lexical index, and the three ways it fails silently if nobody checks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text

from manicule.core.content import DocumentStatus
from manicule.storage.docstore import SqliteDocStore
from manicule.storage.fts import (
    DROP_TRIGGERS,
    INTEGRITY_CHECK_FTS,
    REBUILD_FTS,
    escape_match_query,
)
from tests.storage_helpers import make_chunk, make_document

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


async def _seed(store: SqliteDocStore, source_id: str, texts: list[str]) -> str:
    document = make_document(source_id=source_id)
    await store.upsert_document(document)
    await store.replace_chunks(
        document.id, [make_chunk(document, i, body) for i, body in enumerate(texts)]
    )
    return document.id


async def test_a_query_stem_matches_a_different_surface_form(store: SqliteDocStore) -> None:
    """Porter stemming is why the lexical leg is not a list of exact strings.

    Without it, "authenticate" scores zero against a chunk containing "authentication", and
    because the lexical leg is half of a hybrid merge that removes a candidate rather than
    reordering one.
    """
    await _seed(store, "s1", ["the service handles authentication and token rotation"])
    hits = await store.search_lexical("authenticate", k=5)
    assert [candidate.chunk.text for candidate in hits]


async def test_the_breadcrumb_is_searchable_but_does_not_outrank_the_body(
    store: SqliteDocStore,
) -> None:
    """A section called "Configuration" is unfindable without knowing what it configures.

    Weighted below the body, because the breadcrumb repeats on every chunk of a page.
    """
    await _seed(store, "s1", ["nothing relevant here at all"])
    hits = await store.search_lexical("tokens", k=5)
    assert hits, "the heading breadcrumb should be searchable"


async def test_deleting_a_document_removes_its_rows_from_the_lexical_index(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    """The cascade reaches chunks, and the chunk trigger has to fire on a cascaded delete.

    Verified rather than assumed: if it did not, deleted documents would keep matching
    queries forever with nothing to notice.
    """
    document_id = await _seed(store, "s1", ["authentication and rotation"])
    assert await store.search_lexical("authentication", k=5)

    await store.delete_document(document_id)
    assert await store.search_lexical("authentication", k=5) == []

    async with engine.connect() as connection:
        remaining = (await connection.execute(text("SELECT count(*) FROM chunks"))).scalar_one()
        tombstones = (
            await connection.execute(text("SELECT count(*) FROM vector_tombstones"))
        ).scalar_one()
    assert remaining == 0
    assert tombstones == 1, "the delete trigger must record a tombstone for the vector sweep"


async def test_a_soft_deleted_document_stops_matching_without_touching_the_index(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    """Restoring is clearing a timestamp — no re-embed, no re-parse, no re-fetch."""
    document_id = await _seed(store, "s1", ["authentication and rotation"])
    await store.soft_delete_document(document_id)

    assert await store.search_lexical("authentication", k=5) == []
    async with engine.connect() as connection:
        assert (await connection.execute(text("SELECT count(*) FROM chunks"))).scalar_one() == 1


async def test_asking_for_k_returns_k_live_rows_not_k_rows_of_which_some_are_dead(
    store: SqliteDocStore,
) -> None:
    """The reason the lexical query is one joined statement.

    Deferred deletion leaves soft-deleted chunks in the index competing for the same top-k
    slots. Filtering after a ``LIMIT`` silently returns fewer live rows than asked for — on
    the fixture below, MATCH-then-filter returns none at all.
    """
    dead = await _seed(store, "dead", ["auth token rotation"] * 5)
    await _seed(store, "live", ["auth token rotation"] * 3)
    await store.soft_delete_document(dead)

    hits = await store.search_lexical("auth", k=5)
    assert len(hits) == 3, "every returned candidate must belong to a live document"


async def test_a_document_that_is_not_indexed_is_not_searchable(
    store: SqliteDocStore,
) -> None:
    """Only ``indexed`` is servable, and the join is what enforces it."""
    document = make_document(source_id="pending", status=DocumentStatus.PENDING)
    await store.upsert_document(document)
    await store.replace_chunks(document.id, [make_chunk(document, 0, "auth token rotation")])
    assert await store.search_lexical("auth", k=5) == []


async def test_the_integrity_check_catches_an_index_that_stopped_being_updated(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    """Two obvious checks cannot detect this, and were in an earlier draft of the design.

    ``COUNT(*)`` on an external-content FTS table reads through to the content table, so the
    two counts agree by construction; and a bare ``integrity-check`` passes on an empty index
    because an empty index is internally consistent. Only the ``rank`` argument compares the
    index against the content table.
    """
    async with engine.begin() as connection:
        for statement in DROP_TRIGGERS:
            await connection.execute(text(statement))

    await _seed(store, "s1", ["authentication and rotation"])

    async with engine.connect() as connection:
        chunks = (await connection.execute(text("SELECT count(*) FROM chunks"))).scalar_one()
        fts = (await connection.execute(text("SELECT count(*) FROM chunks_fts"))).scalar_one()
    assert chunks == fts, "the counts agree even though the index is empty"
    assert await store.search_lexical("authentication", k=5) == [], "the index really is empty"

    from sqlalchemy.exc import DatabaseError  # noqa: PLC0415

    async with engine.begin() as connection:
        with pytest.raises(DatabaseError):
            await connection.execute(text(INTEGRITY_CHECK_FTS))


async def test_rebuilding_the_index_repairs_it_from_the_chunks_table(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    """Rung 1 of the ladder: seconds, no network, no model."""
    async with engine.begin() as connection:
        for statement in DROP_TRIGGERS:
            await connection.execute(text(statement))
    await _seed(store, "s1", ["authentication and rotation"])
    assert await store.search_lexical("authentication", k=5) == []

    async with engine.begin() as connection:
        await connection.execute(text(REBUILD_FTS))
    assert await store.search_lexical("authentication", k=5)


@pytest.mark.parametrize(
    "raw",
    ["NOT", "AND OR", 'unbalanced " quote', "col:value", "a*", "(unclosed"],
)
async def test_query_syntax_never_reaches_fts5_as_an_operator(
    store: SqliteDocStore, raw: str
) -> None:
    """A user typing NOT means the word. Unescaped, these are syntax errors, not searches."""
    await _seed(store, "s1", ["ordinary prose about tokens"])
    await store.search_lexical(raw, k=5)


def test_an_empty_query_produces_no_match_expression() -> None:
    """Punctuation alone must not become a match-everything expression."""
    assert escape_match_query("   ") == ""
    assert escape_match_query("!!! ???") == ""


async def test_search_is_scoped_to_the_stores_workspace(engine: AsyncEngine) -> None:
    """Workspace isolation is a property of the handle, so a query cannot forget it."""
    first = SqliteDocStore(engine, workspace_id="alpha")
    second = SqliteDocStore(engine, workspace_id="beta")
    await first.ensure_workspace()
    await second.ensure_workspace()

    await _seed(first, "s1", ["auth token rotation"])
    assert await first.search_lexical("auth", k=5)
    assert await second.search_lexical("auth", k=5) == []
