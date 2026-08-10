"""The relational store: identity, chunk replacement, sync state and workspace scope."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from manicule.core.content import BlockKind, DocumentStatus, PipelineStage
from manicule.core.protocols import DocStore
from manicule.core.retrieval import Filter
from manicule.core.sources import Watermark
from manicule.storage.docstore import SqliteDocStore
from tests.storage_helpers import make_chunk, make_document

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


@pytest.mark.contract
async def test_the_store_satisfies_the_docstore_protocol(store: SqliteDocStore) -> None:
    """Structural conformance, so the container can wire it without a base class."""
    assert isinstance(store, DocStore)


async def test_a_document_round_trips_through_storage_unchanged(store: SqliteDocStore) -> None:
    """Anything storage silently drops is a field nothing downstream can rely on."""
    document = make_document()
    await store.upsert_document(document)
    loaded = await store.get_document(document.id)
    assert loaded == document


async def test_a_document_is_found_by_source_identity_not_by_uri(
    store: SqliteDocStore,
) -> None:
    """A URI is display data a source may change; identity must be what it can promise."""
    document = make_document(source="confluence", source_id="12345")
    await store.upsert_document(document)

    renamed = document.model_copy(update={"uri": "https://wiki/spaces/ENG/pages/12345/New-Title"})
    await store.upsert_document(renamed)

    assert len(await store.list_documents()) == 1
    found = await store.find_document("confluence", "12345")
    assert found is not None
    assert found.uri.endswith("New-Title")


async def test_reparsing_replaces_chunks_rather_than_accumulating_them(
    store: SqliteDocStore,
) -> None:
    """Re-parsing is not additive; a document with both old and new chunks cites both."""
    document = make_document()
    await store.upsert_document(document)
    await store.replace_chunks(
        document.id, [make_chunk(document, i, f"body {i}") for i in range(3)]
    )
    assert await store.count_chunks(document.id) == 3

    await store.replace_chunks(document.id, [make_chunk(document, 0, "only one now")])
    assert await store.count_chunks(document.id) == 1


async def test_an_unchanged_chunk_keeps_its_id_across_a_reparse(
    store: SqliteDocStore,
) -> None:
    """Content-derived ids are what let an unchanged chunk keep its vector.

    Re-parsing a document that changed in one place should not re-embed the rest of it.
    """
    document = make_document()
    await store.upsert_document(document)
    first = [make_chunk(document, i, f"paragraph {i}") for i in range(3)]
    await store.replace_chunks(document.id, first)

    edited = [
        make_chunk(document, 0, "paragraph 0"),
        make_chunk(document, 1, "paragraph 1 but edited"),
        make_chunk(document, 2, "paragraph 2"),
    ]
    await store.replace_chunks(document.id, edited)

    unchanged = {first[0].id, first[2].id}
    stored = {chunk.id for chunk in await store.get_chunks([c.id for c in edited])}
    assert unchanged <= stored
    assert first[1].id not in stored


async def test_a_chunk_round_trips_with_its_anchor_intact(store: SqliteDocStore) -> None:
    """Anchors are the product. A citation that survives storage lossily is not a citation."""
    document = make_document()
    await store.upsert_document(document)
    chunk = make_chunk(document, 0, "a located paragraph", heading_path=("Auth", "Tokens"))
    await store.replace_chunks(document.id, [chunk])

    loaded = await store.get_chunks([chunk.id])
    assert list(loaded) == [chunk]
    assert loaded[0].anchor == chunk.anchor
    assert loaded[0].heading_path == ("Auth", "Tokens")


async def test_an_unlocated_anchor_round_trips_as_unlocated(store: SqliteDocStore) -> None:
    """ "We do not know" has to survive storage, or it becomes "nobody asked"."""
    from manicule.core.anchors import Unlocated  # noqa: PLC0415

    document = make_document()
    await store.upsert_document(document)
    chunk = make_chunk(document, 0, "unplaceable text", located=False)
    await store.replace_chunks(document.id, [chunk])

    loaded = (await store.get_chunks([chunk.id]))[0]
    assert isinstance(loaded.anchor, Unlocated)
    assert loaded.anchor.reason


async def test_setting_a_status_clears_failed_stage_when_it_no_longer_applies(
    store: SqliteDocStore,
) -> None:
    """A stale ``failed_stage`` makes "everything that died in parse" return successes."""
    document = make_document(status=DocumentStatus.FAILED)
    await store.upsert_document(document)
    assert (await store.get_document(document.id)) is not None

    await store.set_status(document.id, DocumentStatus.INDEXED)
    loaded = await store.get_document(document.id)
    assert loaded is not None
    assert loaded.status is DocumentStatus.INDEXED
    assert loaded.failed_stage is None


async def test_a_failed_document_records_which_stage_failed(store: SqliteDocStore) -> None:
    """One ``failed`` member plus a stage, rather than a member per stage."""
    document = make_document(status=DocumentStatus.FAILED)
    await store.upsert_document(document)
    loaded = await store.get_document(document.id)
    assert loaded is not None
    assert loaded.failed_stage is PipelineStage.PARSE
    assert loaded.status_detail


async def test_setting_the_status_of_a_document_that_vanished_is_not_an_error(
    store: SqliteDocStore,
) -> None:
    """One bad document never aborts a batch, including one deleted underneath the run."""
    await store.set_status("no-such-document", DocumentStatus.INDEXED)


async def test_deleting_a_document_twice_is_not_an_error(store: SqliteDocStore) -> None:
    """Idempotence is what makes a retry safe after a partial failure."""
    document = make_document()
    await store.upsert_document(document)
    await store.delete_document(document.id)
    await store.delete_document(document.id)
    assert await store.get_document(document.id) is None


async def test_a_soft_deleted_document_is_invisible_but_recoverable(
    store: SqliteDocStore,
) -> None:
    """Restore is clearing a timestamp, which is the cheapest rung there is."""
    document = make_document()
    await store.upsert_document(document)
    await store.replace_chunks(document.id, [make_chunk(document, 0, "text")])
    await store.soft_delete_document(document.id)

    assert await store.get_document(document.id) is None
    assert await store.count_chunks(document.id) == 1, "chunks survive for a free restore"


async def test_listing_documents_can_be_filtered(store: SqliteDocStore) -> None:
    """Selection has to be a query; scanning the corpus in Python does not scale."""
    await store.upsert_document(make_document(source="fs", source_id="a"))
    await store.upsert_document(
        make_document(source="confluence", source_id="b", media_type="text/html")
    )

    assert len(await store.list_documents()) == 2
    assert len(await store.list_documents(Filter(source="confluence"))) == 1
    assert len(await store.list_documents(Filter(media_types=frozenset({"text/html"})))) == 1


async def test_filtering_lexical_search_by_kind_pushes_into_the_query(
    store: SqliteDocStore,
) -> None:
    """Filtering in Python after the LIMIT is the bug this whole query shape avoids."""
    document = make_document()
    await store.upsert_document(document)
    await store.replace_chunks(
        document.id,
        [
            make_chunk(document, 0, "authentication in prose", kind=BlockKind.PROSE),
            make_chunk(document, 1, "authentication in code", kind=BlockKind.CODE),
        ],
    )
    only_code = await store.search_lexical(
        "authentication", k=5, filter=Filter(kinds=frozenset({BlockKind.CODE}))
    )
    assert [candidate.chunk.kind for candidate in only_code] == [BlockKind.CODE]


async def test_a_document_id_containing_a_quote_cannot_break_the_filter(
    store: SqliteDocStore,
) -> None:
    """Filter values are bound parameters, never interpolated text."""
    document = make_document()
    await store.upsert_document(document)
    await store.replace_chunks(document.id, [make_chunk(document, 0, "authentication")])

    hostile = "' OR 1=1 --"
    results = await store.search_lexical(
        "authentication", k=5, filter=Filter(document_ids=frozenset({hostile}))
    )
    assert results == []


async def test_a_watermark_is_stored_and_handed_back_uninterpreted(
    store: SqliteDocStore,
) -> None:
    """Opaque and connector-defined: a CQL timestamp, a page token, a commit SHA."""
    assert await store.get_watermark("confluence") is None

    watermark = Watermark(
        value="2026-08-10 14:30",
        observed_at=datetime(2026, 8, 10, 14, 30, tzinfo=UTC),
        metadata={"space": "ENG"},
    )
    await store.set_watermark("confluence", watermark)
    assert await store.get_watermark("confluence") == watermark


async def test_a_watermark_is_replaced_rather_than_appended(store: SqliteDocStore) -> None:
    """Two watermarks for one connector is two answers to "where did we get to"."""
    first = Watermark(value="a", observed_at=datetime(2026, 8, 1, tzinfo=UTC))
    second = Watermark(value="b", observed_at=datetime(2026, 8, 2, tzinfo=UTC))
    await store.set_watermark("confluence", first)
    await store.set_watermark("confluence", second)
    assert await store.get_watermark("confluence") == second


async def test_known_source_ids_yields_what_reconciliation_diffs_against(
    store: SqliteDocStore,
) -> None:
    """Whatever this omits gets deleted from the index, so it must be complete."""
    await store.upsert_document(make_document(source="confluence", source_id="1"))
    await store.upsert_document(make_document(source="confluence", source_id="2"))
    await store.upsert_document(make_document(source="fs", source_id="3"))

    seen = {source_id async for source_id in store.known_source_ids("confluence")}
    assert seen == {"1", "2"}


async def test_a_soft_deleted_document_is_absent_from_reconciliation(
    store: SqliteDocStore,
) -> None:
    """Otherwise a deleted document is repeatedly "rediscovered" and never settles."""
    document = make_document(source="confluence", source_id="1")
    await store.upsert_document(document)
    await store.soft_delete_document(document.id)
    assert [source_id async for source_id in store.known_source_ids("confluence")] == []


async def test_two_workspaces_cannot_see_each_others_documents(engine: AsyncEngine) -> None:
    """Workspace isolation is a security boundary, so it is bound to the handle."""
    alpha = SqliteDocStore(engine, workspace_id="alpha")
    beta = SqliteDocStore(engine, workspace_id="beta")
    await alpha.ensure_workspace()
    await beta.ensure_workspace()

    document = make_document()
    await alpha.upsert_document(document)

    assert await alpha.get_document(document.id) is not None
    assert await beta.get_document(document.id) is None
    assert await beta.list_documents() == []


async def test_the_same_source_identity_can_exist_in_two_workspaces(
    engine: AsyncEngine,
) -> None:
    """Uniqueness is per workspace; two tenants indexing one wiki is not a conflict."""
    alpha = SqliteDocStore(engine, workspace_id="alpha")
    beta = SqliteDocStore(engine, workspace_id="beta")
    await alpha.ensure_workspace()
    await beta.ensure_workspace()

    await alpha.upsert_document(make_document(source="confluence", source_id="shared"))
    other = make_document(source="confluence", source_id="shared")
    await beta.upsert_document(other.model_copy(update={"id": other.id + "-beta"}))

    assert len(await alpha.list_documents()) == 1
    assert len(await beta.list_documents()) == 1


async def test_getting_no_chunks_asks_the_database_nothing(store: SqliteDocStore) -> None:
    """An empty id list is a legitimate call, not a query for every chunk there is."""
    assert list(await store.get_chunks([])) == []
