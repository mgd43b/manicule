"""The relational store: identity, chunk replacement, sync state and workspace scope."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import event, func, select

from manicule.core.acquisition import AcquisitionFence
from manicule.core.content import BlockKind, DocumentStatus, PipelineStage
from manicule.core.protocols import DocStore
from manicule.core.retrieval import Filter
from manicule.core.sources import Watermark
from manicule.storage import models
from manicule.storage.docstore import (
    DEFAULT_WORKSPACE,
    CrossWorkspaceCollisionError,
    SqliteDocStore,
)
from manicule.testing import assert_protocol_signatures, closing
from tests.storage_helpers import make_chunk, make_document

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


def scoped(**restrictions: object) -> Filter:
    """A filter scoped to the workspace the ``store`` fixture serves.

    Every filter carries a workspace now — the field is required precisely so that no query
    can be written without one — so the tests say it once here rather than in every call.
    """
    return Filter(workspace_ids=frozenset({DEFAULT_WORKSPACE}), **restrictions)  # pyright: ignore[reportArgumentType]


@pytest.mark.contract
async def test_the_store_satisfies_the_docstore_protocol(store: SqliteDocStore) -> None:
    """Structural conformance, so the container can wire it without a base class.

    Both halves are needed. ``isinstance`` checks the attributes exist;
    ``assert_protocol_signatures`` checks they accept what the protocol says they accept,
    which ``@runtime_checkable`` deliberately does not.
    """
    assert isinstance(store, DocStore)
    assert_protocol_signatures(store, DocStore)


async def test_a_document_round_trips_through_storage_unchanged(store: SqliteDocStore) -> None:
    """Anything storage silently drops is a field nothing downstream can rely on.

    ``indexed_at`` is held out of the equality and asserted on its own line, because it is the
    one field on the model that storage **writes** rather than round-trips:
    :func:`~manicule.storage.rows.apply_document` stamps it when a document arrives ``indexed``,
    and the domain object the pipeline built cannot know it. Inside the equality it would compare
    a value nothing supplied against one storage invented. The honest claim is two claims — every
    other field survives the trip, and this one comes back filled in — and splitting them is what
    lets the second one fail on its own if the stamp ever stops happening.
    """
    document = make_document()
    assert document.indexed_at is None, "the fixture must not pre-supply the stamped field"
    await store.upsert_document(document)
    loaded = await store.get_document(document.id)
    assert loaded is not None
    assert loaded.model_copy(update={"indexed_at": None}) == document
    assert loaded.indexed_at is not None, (
        "storage stamps indexed_at for an indexed document, and a citation reports it as the one "
        "of a mirrored document's three timestamps that describes this installation"
    )


@pytest.mark.parametrize(("chunks", "writes"), [(100, 1), (101, 2)])
async def test_vector_cleanup_evidence_is_staged_in_bounded_transactional_batches(
    store: SqliteDocStore,
    engine: AsyncEngine,
    chunks: int,
    writes: int,
) -> None:
    document = make_document()
    statements: list[str] = []

    def observe(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        if "INSERT INTO vector_tombstones" in statement:
            statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", observe)
    try:
        await store.stage_vectors(
            "synthetic-publication",
            [make_chunk(document, position, f"generated {position}") for position in range(chunks)],
        )
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", observe)

    assert len(statements) == writes
    async with engine.connect() as connection:
        count = await connection.scalar(select(func.count(models.VectorTombstone.chunk_id)))
    assert count == chunks


async def test_fenced_publication_refreshes_an_expired_lease_before_commit(
    store: SqliteDocStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease_start = datetime(2026, 8, 15, 12, tzinfo=UTC)
    await store.create_acquisition_run("long-publication", "wiki")
    claimed = await store.claim_acquisition_run(
        "long-publication",
        "worker",
        now=lease_start,
        expires_at=lease_start + timedelta(seconds=1),
    )
    assert claimed is not None
    commit_time = lease_start + timedelta(seconds=5)
    monkeypatch.setattr("manicule.storage.acquisition.utcnow", lambda: commit_time)
    document = make_document(source="wiki", source_id="synthetic-long-publication")

    committed = await store.fenced_publish_document(
        AcquisitionFence(
            run_id=claimed.id,
            owner="worker",
            generation=claimed.lease_generation,
            now=lease_start,
            lease_ttl_seconds=30,
        ),
        document,
        [],
        expected=None,
        chunk_fp=None,
        embed_fp=None,
        parse_fp=None,
        glossary_entries=None,
        glossary_fp=None,
        original_omitted_reason=None,
    )

    assert committed.committed
    durable = await store.get_acquisition_run(claimed.id)
    assert durable is not None
    assert durable.lease_owner == "worker"
    assert durable.lease_generation == claimed.lease_generation
    assert durable.lease_expires_at == commit_time + timedelta(seconds=30)


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
    assert len(await store.list_documents(scoped(sources=frozenset({"confluence"})))) == 1
    assert len(await store.list_documents(scoped(media_types=frozenset({"text/html"})))) == 1


async def test_listing_documents_refuses_a_field_it_has_no_column_for(
    store: SqliteDocStore,
) -> None:
    """A chunk restriction on a list of documents has no meaning, so it is not guessed at.

    Applying the rest of the filter and dropping this one would return documents the caller
    asked to exclude, in a listing that still looks like it worked.
    """
    with pytest.raises(ValueError, match="kinds"):
        await store.list_documents(scoped(kinds=frozenset({BlockKind.CODE})))


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
        "authentication", k=5, filter=scoped(kinds=frozenset({BlockKind.CODE}))
    )
    assert [candidate.chunk.kind for candidate in only_code] == [BlockKind.CODE]


async def test_lexical_search_filters_on_the_language_column_it_promotes(
    store: SqliteDocStore,
) -> None:
    """The authoritative store has to hold what the derived one filters on.

    ``langs`` resolves against ``chunks.lang`` here and against the Lance ``lang`` column in
    the dense leg. Both are promoted out of the chunk's metadata by the same rule, because two
    legs of one query that disagreed about what language a chunk is in would return two
    different corpora for the same filter.
    """
    document = make_document()
    await store.upsert_document(document)
    await store.replace_chunks(
        document.id,
        [
            make_chunk(document, 0, "authentication in english", lang="en"),
            make_chunk(document, 1, "authentication en francais", lang="fr"),
        ],
    )

    french = await store.search_lexical(
        "authentication", k=5, filter=scoped(langs=frozenset({"fr"}))
    )

    assert [candidate.chunk.text for candidate in french] == ["authentication en francais"]


async def test_lexical_search_applies_every_document_level_restriction_inline(
    store: SqliteDocStore,
) -> None:
    """One statement, so ``LIMIT`` lands after the filters rather than before them."""
    wanted = make_document(source="confluence", source_id="a", media_type="text/html")
    other = make_document(source="fs", source_id="b")
    for document in (wanted, other):
        await store.upsert_document(document)
        await store.replace_chunks(document.id, [make_chunk(document, 0, "authentication")])

    by_source = await store.search_lexical(
        "authentication", k=5, filter=scoped(sources=frozenset({"confluence"}))
    )
    by_media_type = await store.search_lexical(
        "authentication", k=5, filter=scoped(media_types=frozenset({"text/html"}))
    )
    before_the_epoch = await store.search_lexical(
        "authentication", k=5, filter=scoped(updated_before=datetime(2000, 1, 1, tzinfo=UTC))
    )
    since_the_epoch = await store.search_lexical(
        "authentication", k=5, filter=scoped(updated_after=datetime(2000, 1, 1, tzinfo=UTC))
    )

    assert [candidate.chunk.document_id for candidate in by_source] == [wanted.id]
    assert [candidate.chunk.document_id for candidate in by_media_type] == [wanted.id]
    # Both bounds, because "returned nothing" is what a timestamp that reached the driver in
    # the wrong encoding looks like as well.
    assert before_the_epoch == []
    assert len(since_the_epoch) == 2


async def test_lexical_search_refuses_a_field_it_cannot_apply(store: SqliteDocStore) -> None:
    """``collection_ids`` needs a join this statement does not make.

    The retrieval layer resolves it into ``document_ids`` before either store is reached, so
    arriving here with one set is a caller who believes a restriction is in force that is not.
    """
    with pytest.raises(ValueError, match="collection_ids"):
        await store.search_lexical("anything", k=5, filter=scoped(collection_ids=frozenset({"c"})))


async def test_a_document_id_containing_a_quote_cannot_break_the_filter(
    store: SqliteDocStore,
) -> None:
    """Filter values are bound parameters, never interpolated text."""
    document = make_document()
    await store.upsert_document(document)
    await store.replace_chunks(document.id, [make_chunk(document, 0, "authentication")])

    hostile = "' OR 1=1 --"
    results = await store.search_lexical(
        "authentication", k=5, filter=scoped(document_ids=frozenset({hostile}))
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

    async with closing(store.known_source_ids("confluence")) as ids:
        seen = {source_id async for source_id in ids}
    assert seen == {"1", "2"}


async def test_a_soft_deleted_document_is_absent_from_reconciliation(
    store: SqliteDocStore,
) -> None:
    """Otherwise a deleted document is repeatedly "rediscovered" and never settles."""
    document = make_document(source="confluence", source_id="1")
    await store.upsert_document(document)
    await store.soft_delete_document(document.id)
    async with closing(store.known_source_ids("confluence")) as ids:
        assert [source_id async for source_id in ids] == []


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


async def test_two_workspaces_can_index_the_same_source_independently(
    engine: AsyncEngine,
) -> None:
    """The same wiki synced into two workspaces is two documents, and that is correct.

    Written without adjusting either id. An earlier version of this test hand-edited the
    second document's id to make it pass, which quietly worked around the defect it was
    supposed to prove absent: ``document_id`` did not include the workspace, so both tenants
    computed the same id and the second write landed on the first tenant's row.
    """
    alpha = SqliteDocStore(engine, workspace_id="alpha")
    beta = SqliteDocStore(engine, workspace_id="beta")
    await alpha.ensure_workspace()
    await beta.ensure_workspace()

    in_alpha = make_document(source="confluence", source_id="shared", workspace_id="alpha")
    in_beta = make_document(source="confluence", source_id="shared", workspace_id="beta")
    assert in_alpha.id != in_beta.id, "workspace is part of the identity"

    await alpha.upsert_document(in_alpha)
    await beta.upsert_document(in_beta)

    assert len(await alpha.list_documents()) == 1
    assert len(await beta.list_documents()) == 1
    assert await alpha.get_document(in_beta.id) is None
    assert await beta.get_document(in_alpha.id) is None


async def test_an_id_built_without_the_workspace_cannot_land_on_another_tenants_row(
    engine: AsyncEngine,
) -> None:
    """The guard behind the id scheme, for a caller that computed an id some other way.

    Unreachable through ``document_id``, which is the point: it costs one comparison per
    write, and the bug it catches costs a tenant's data — an overwrite of content the writer
    cannot read, with the writer's own document apparently vanishing.
    """
    alpha = SqliteDocStore(engine, workspace_id="alpha")
    beta = SqliteDocStore(engine, workspace_id="beta")
    await alpha.ensure_workspace()
    await beta.ensure_workspace()

    private = make_document(source="confluence", source_id="123", title="alpha private")
    await alpha.upsert_document(private)

    with pytest.raises(CrossWorkspaceCollisionError, match="belongs to workspace"):
        await beta.upsert_document(private.model_copy(update={"title": "beta overwrote it"}))

    survived = await alpha.get_document(private.id)
    assert survived is not None
    assert survived.title == "alpha private"


async def test_a_filter_naming_another_workspace_is_refused(engine: AsyncEngine) -> None:
    """Silently ignoring it would answer a question nobody asked.

    A subset check rather than an equality one, because the field is set-valued: cross-workspace
    search is one handle per workspace merged, so a handle that is offered a set reaching past
    its own is being asked for somebody else's corpus.
    """
    alpha = SqliteDocStore(engine, workspace_id="alpha")
    await alpha.ensure_workspace()

    for named in (frozenset({"beta"}), frozenset({"alpha", "beta"})):
        with pytest.raises(CrossWorkspaceCollisionError, match="this store serves"):
            await alpha.list_documents(Filter(workspace_ids=named))
        with pytest.raises(CrossWorkspaceCollisionError, match="this store serves"):
            await alpha.search_lexical("anything", k=5, filter=Filter(workspace_ids=named))

    assert await alpha.list_documents(Filter(workspace_ids=frozenset({"alpha"}))) == []


async def test_lexical_search_takes_its_query_under_the_protocols_parameter_name(
    store: SqliteDocStore,
) -> None:
    """A caller using the protocol's keyword must reach the implementation."""
    assert await store.search_lexical(text="nothing indexed yet", k=5) == []


async def test_getting_no_chunks_asks_the_database_nothing(store: SqliteDocStore) -> None:
    """An empty id list is a legitimate call, not a query for every chunk there is."""
    assert list(await store.get_chunks([])) == []


# --- parse lineage ------------------------------------------------------------------------


async def test_parse_lineage_round_trips_and_is_not_written_by_an_upsert(
    store: SqliteDocStore,
) -> None:
    """Lineage moves through ``set_lineage`` and through nothing else.

    The pipeline builds a fresh :class:`Document` for every ingest and cannot know a parse
    fingerprint before the chain has chosen a parser, so a store that wrote lineage from the
    domain object would clear it at the start of every run — and "which documents need
    re-parsing" would then answer "all of them" for the wrong reason, on every sync, forever.
    """
    document = await store.upsert_document(make_document())
    await store.set_lineage(document.id, chunk_fp=None, embed_fp=None, parse_fp="pdf@5.12.1")

    assert (await store.get_document(document.id)).parse_fp == "pdf@5.12.1"  # pyright: ignore[reportOptionalMemberAccess]

    await store.upsert_document(document.model_copy(update={"title": "renamed"}))

    reread = await store.get_document(document.id)
    assert reread is not None
    assert reread.title == "renamed"
    assert reread.parse_fp == "pdf@5.12.1", "an ordinary upsert must not clear the lineage"


async def test_a_null_parse_lineage_leaves_a_stored_one_alone(store: SqliteDocStore) -> None:
    """``None`` means "leave this one alone", not "clear it".

    Re-embedding moves only the embedding lineage. A store reading ``None`` as a clear would
    make every re-embed silently mark the whole corpus as needing a re-parse.
    """
    document = await store.upsert_document(make_document())
    await store.set_lineage(document.id, chunk_fp=None, embed_fp=None, parse_fp="pdf@5.12.1")

    await store.set_lineage(document.id, chunk_fp="chunker@1", embed_fp="model@1")

    assert (await store.get_document(document.id)).parse_fp == "pdf@5.12.1"  # pyright: ignore[reportOptionalMemberAccess]


async def test_selecting_on_parse_lineage_finds_the_stale_and_the_unrecorded(
    store: SqliteDocStore,
) -> None:
    """The re-parse query, in SQL, in both directions.

    A ``NULL`` lineage is in the selection deliberately: no recorded fingerprint means no
    evidence the stored text is current, and a selector that assumed it was would skip
    precisely the documents that predate the column.
    """
    stale = await store.upsert_document(make_document(source_id="stale"))
    fresh = await store.upsert_document(make_document(source_id="fresh"))
    unrecorded = await store.upsert_document(make_document(source_id="plugin"))
    await store.set_lineage(stale.id, chunk_fp=None, embed_fp=None, parse_fp="pdf@5.12.1")
    await store.set_lineage(fresh.id, chunk_fp=None, embed_fp=None, parse_fp="pdf@5.13.0")

    chosen = await store.select_documents(parse_fp_current=["pdf@5.13.0"])

    assert {document.id for document in chosen} == {stale.id, unrecorded.id}
    assert fresh.id not in {document.id for document in chosen}


async def test_not_selecting_on_parse_lineage_selects_everything(store: SqliteDocStore) -> None:
    """``None`` and an empty collection are different questions, so they are kept apart."""
    await store.upsert_document(make_document(source_id="a"))

    assert len(await store.select_documents()) == 1
    assert len(await store.select_documents(parse_fp_current=[])) == 1


async def test_paging_the_selection_returns_each_document_once_and_in_one_order(
    store: SqliteDocStore,
) -> None:
    """What a corpus-wide repair pages through, asserted as a partition rather than a count.

    A page size and an offset that agreed on how many rows there were but not on *which* would
    still add up to the total while returning one document twice and another never — which is a
    repair that silently skips documents, and the only failure this could have. So the pages
    are concatenated and compared to the unpaged answer, element by element and in order.
    """
    for index in range(5):
        await store.upsert_document(make_document(source_id=f"page-{index}"))
    every = [document.id for document in await store.select_documents()]

    paged = [
        document.id
        for offset in (0, 2, 4)
        for document in await store.select_documents(limit=2, offset=offset)
    ]

    assert len(every) == 5
    assert paged == every, "two pages disagreed about which documents lie between them"
    assert await store.select_documents(offset=5) == []
