"""Organization on top of the corpus: collections, tags, versions, relations, the trash.

Three properties carry most of the weight here, and each of them fails in silence.

**A grouping never widens a workspace.** ``collection_documents``, ``document_tags`` and
``chunk_relations`` all reach documents and chunks by id, and none of them has a workspace
column — so tenancy is enforced by the code that writes them or it is not enforced at all.

**A grouping never deletes what it groups.** The cascades run from collection to membership and
from tag to application, and one foreign key pointed a table further takes the corpus with it.

**A citation is correct or it is absent.** Versioning is what makes the second half of that
sentence sayable: a chunk id derived from text a re-ingest replaced no longer exists, and the
store says *why* rather than returning whatever now sits at that position.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text as sql

from manicule.core.content import Document, DocumentStatus
from manicule.core.errors import NameInUseError, UnknownEntityError
from manicule.core.ids import content_hash
from manicule.core.organization import (
    ChunkRelationType,
    CitationState,
    CollectionRule,
)
from manicule.core.protocols import (
    ChunkRelationStore,
    CollectionStore,
    TagStore,
    TrashStore,
    VersionStore,
)
from manicule.core.retrieval import Filter
from manicule.storage import organization
from manicule.storage.blobs import BlobStore
from manicule.storage.docstore import SqliteDocStore
from manicule.storage.organization import normalize_name, resolve_filter
from manicule.storage.scoped import DEFAULT_WORKSPACE
from manicule.testing import (
    assert_chunk_relation_store_contract,
    assert_collection_store_contract,
    assert_protocol_signatures,
    assert_tag_store_contract,
    assert_trash_store_contract,
    assert_version_store_contract,
)
from tests.storage_helpers import make_chunk, make_document

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine


async def _seed(
    store: SqliteDocStore, count: int = 2, *, source: str = "fs", workspace: str | None = None
) -> list[Document]:
    """Live, indexed documents in ``store``'s workspace."""
    made: list[Document] = []
    for index in range(count):
        document = make_document(
            source=source,
            source_id=f"s{index}",
            workspace_id=workspace or store.workspace_id,
            uri=f"file:///{index}.md",
            body=f"body {index}".encode(),
        )
        made.append(await store.upsert_document(document))
    return made


async def _foreign(engine: AsyncEngine) -> tuple[SqliteDocStore, Document]:
    """A document in another workspace, and the handle that owns it."""
    other = SqliteDocStore(engine, workspace_id="other-tenant")
    await other.ensure_workspace()
    document = (await _seed(other, 1, source="fs"))[0]
    return other, document


async def _with_chunks(store: SqliteDocStore, document: Document, count: int = 2) -> list[str]:
    chunks = [make_chunk(document, index, f"chunk {index}") for index in range(count)]
    await store.replace_chunks(document.id, chunks)
    return [chunk.id for chunk in chunks]


# --- the protocols themselves -------------------------------------------------------------


@pytest.mark.contract
def test_one_store_satisfies_every_organization_protocol(store: SqliteDocStore) -> None:
    """``DocStore`` was left partial on the promise that these would arrive separately.

    Structural conformance *and* signature conformance, because ``@runtime_checkable`` checks
    only that an attribute exists: an implementation whose parameter is spelled differently
    passes ``isinstance`` and fails at the first keyword call.
    """
    for protocol in (CollectionStore, TagStore, VersionStore, TrashStore, ChunkRelationStore):
        assert isinstance(store, protocol), f"the store does not satisfy {protocol.__name__}"
        assert_protocol_signatures(store, protocol)


# --- collections ----------------------------------------------------------------------------


@pytest.mark.contract
async def test_the_sqlite_store_passes_the_collection_suite(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    """The shipped suite, against the real database rather than a double."""
    documents = await _seed(store, 2)
    _, outsider = await _foreign(engine)
    await assert_collection_store_contract(
        store,
        store,
        document_ids=[document.id for document in documents],
        foreign_document_id=outsider.id,
    )


async def test_a_rule_driven_collection_picks_up_documents_indexed_after_it(
    store: SqliteDocStore,
) -> None:
    """A rule is a saved query, not a snapshot taken the day it was written.

    Materializing membership at write time would make "everything from the runbooks space"
    quietly mean "what was there in March", and nothing about the collection would say so.
    """
    collection = await store.create_collection(
        "wiki", rule=CollectionRule(sources=frozenset({"confluence"}))
    )
    assert not await store.collection_documents(collection.id)

    later = await _seed(store, 1, source="confluence")
    members = {document.id for document in await store.collection_documents(collection.id)}
    assert members == {later[0].id}


async def test_a_collection_never_lists_a_member_that_is_in_the_trash(
    store: SqliteDocStore,
) -> None:
    """Soft delete is enforced at every read or at none of them.

    Deletion is deferred, so the membership row survives the delete by design. If the
    collection listing did not apply the same predicate as retrieval, a deleted document would
    still be reachable through the one view built for browsing a corpus.
    """
    documents = await _seed(store, 2)
    collection = await store.create_collection("everything")
    await store.add_to_collection(collection.id, [document.id for document in documents])

    await store.soft_delete_document(documents[0].id)

    members = {document.id for document in await store.collection_documents(collection.id)}
    assert members == {documents[1].id}
    assert not await store.collections_for(documents[0].id)


async def test_a_membership_batch_naming_a_foreign_document_writes_nothing(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    """All or nothing, because a partial write reports a success it did not have.

    The refused id is the interesting one — a typo, or a document from another tenant — and a
    caller told "added two of three" without being told which is a caller who cannot act.
    """
    mine = await _seed(store, 1)
    _, outsider = await _foreign(engine)
    collection = await store.create_collection("mixed")

    with pytest.raises(UnknownEntityError, match="Nothing was written"):
        await store.add_to_collection(collection.id, [mine[0].id, outsider.id])

    assert not await store.collection_documents(collection.id), (
        "the document the handle *could* see was written before the batch was refused"
    )


async def test_a_stored_rule_cannot_name_a_workspace() -> None:
    """A saved query that could widen its own scope is a cross-tenant leak with a schema.

    The rule is stored and re-executed later by whichever handle reads the collection. The
    workspace comes from that handle, always, and the type refuses to carry one.
    """
    with pytest.raises(ValueError, match="workspace_ids"):
        CollectionRule.model_validate({"sources": ["fs"], "workspace_ids": ["somebody-else"]})


async def test_a_rule_that_restricts_nothing_is_refused() -> None:
    """An empty rule selects the whole workspace, which is never what adding a rule meant."""
    with pytest.raises(ValueError, match="restrict something"):
        CollectionRule()


async def test_a_collection_name_is_taken_only_once(store: SqliteDocStore) -> None:
    """Two collections under one name merge two people's sets, silently and permanently."""
    await store.create_collection("Runbooks")
    with pytest.raises(NameInUseError, match="already has a collection"):
        await store.create_collection("  Runbooks  ")


# --- tags -----------------------------------------------------------------------------------


@pytest.mark.contract
async def test_the_sqlite_store_passes_the_tag_suite(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    documents = await _seed(store, 1)
    _, outsider = await _foreign(engine)
    await assert_tag_store_contract(
        store, store, document_id=documents[0].id, foreign_document_id=outsider.id
    )


async def test_two_encodings_of_one_label_are_one_tag(store: SqliteDocStore) -> None:
    """``café`` typed two ways is one label to a reader and two rows without normalization.

    A precomposed ``é`` and an ``e`` followed by a combining acute are different byte strings
    and identical on screen. Filtering by one would then return half the documents, with both
    tags looking correct in every listing.
    """
    composed = await store.ensure_tag("café")
    decomposed = await store.ensure_tag("café")
    assert decomposed.id == composed.id
    assert len(await store.list_tags()) == 1


async def test_a_tag_cannot_be_applied_to_another_workspaces_document(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    """Tags resolve into document ids before a search runs, so a foreign application is a leak."""
    _, outsider = await _foreign(engine)
    tag = await store.ensure_tag("shared")
    with pytest.raises(UnknownEntityError):
        await store.tag_document(outsider.id, [tag.id])


async def test_normalizing_refuses_a_name_that_is_only_whitespace() -> None:
    """A nameless label cannot be found again, and would collide with the next one."""
    with pytest.raises(ValueError, match="empty once whitespace is collapsed"):
        normalize_name("   \n ")


# --- chunk relations --------------------------------------------------------------------------


@pytest.mark.contract
async def test_the_sqlite_store_passes_the_relation_suite(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    documents = await _seed(store, 1)
    chunk_ids = await _with_chunks(store, documents[0])
    other, outsider = await _foreign(engine)
    foreign_chunks = await _with_chunks(other, outsider, 1)
    await assert_chunk_relation_store_contract(
        store, chunk_ids=chunk_ids, foreign_chunk_id=foreign_chunks[0]
    )


async def test_replacing_a_documents_chunks_takes_their_edges_with_them(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    """The cascade is the whole point of chunks being a table, so it is checked on real rows.

    ``chunk_relations`` carries foreign keys with ``ON DELETE CASCADE`` in both directions.
    Without them, orphan cleanup is a pattern match over formatted identifiers — which is what
    a design without a chunks table is reduced to, and what it gets wrong.
    """
    documents = await _seed(store, 1)
    chunk_ids = await _with_chunks(store, documents[0], 2)
    await store.relate(chunk_ids[0], chunk_ids[1], ChunkRelationType.PARENT)

    await store.replace_chunks(documents[0].id, [])

    async with engine.connect() as connection:
        remaining = (
            await connection.execute(sql("SELECT count(*) FROM chunk_relations"))
        ).scalar_one()
    assert remaining == 0, "an edge outlived both of the chunks it joined"


async def test_an_edge_into_the_trash_is_not_returned(store: SqliteDocStore) -> None:
    """A soft-deleted document must not reach a reader through a neighbor that is still live.

    Deletion is deferred, so the far chunk and its row are both still there. Relations are a
    context-expansion mechanism; one that ignored the soft-delete predicate would put deleted
    content back into an answer by the side door.
    """
    first, second = await _seed(store, 2)
    left = await _with_chunks(store, first, 1)
    right = await _with_chunks(store, second, 1)
    await store.relate(left[0], right[0], ChunkRelationType.SIBLING)
    assert len(await store.related(left[0])) == 1

    await store.soft_delete_document(second.id)
    assert not await store.related(left[0])


# --- versions -------------------------------------------------------------------------------


@pytest.mark.contract
async def test_the_sqlite_store_passes_the_version_suite(store: SqliteDocStore) -> None:
    await assert_version_store_contract(
        store, store, document=make_document(source="fs", source_id="versioned")
    )


async def test_a_write_that_does_not_change_the_bytes_records_no_version(
    store: SqliteDocStore,
) -> None:
    """Ingest writes a document row far more often than a document changes.

    A version per upsert would fill the history with rows recording nothing — a status
    transition, a re-seen skip, a repair marking a document indexed — and each of them would
    pin a blob against collection for as long as it stayed.
    """
    document = (await _seed(store, 1))[0]
    await store.upsert_document(document.model_copy(update={"title": "renamed"}))
    await store.set_status(document.id, DocumentStatus.PARSED)

    assert not await store.list_versions(document.id)
    assert await store.current_version(document.id) == 1


async def test_a_version_keeps_the_bytes_and_the_chunk_count_of_the_state_it_left(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path
) -> None:
    """A version row describes the state the document *left*, completely.

    Recording the incoming state instead would write a row whose ``original_ref`` is filled in
    a moment later by ``set_original`` — a history whose most recent entry is the one that
    might still be wrong.
    """
    blobs = BlobStore(engine, data_dir)
    retained = await blobs.retain(b"the first draft", "text/markdown")
    document = (await _seed(store, 1))[0]
    await store.set_original(document.id, ref=retained.ref, omitted_reason=None)
    await _with_chunks(store, document, 3)

    reloaded = await store.get_document(document.id)
    assert reloaded is not None
    await store.upsert_document(
        reloaded.model_copy(
            update={"content_hash": content_hash(b"the second draft"), "title": "second"}
        )
    )

    versions = await store.list_versions(document.id)
    assert len(versions) == 1
    assert versions[0].content_hash == document.content_hash
    assert versions[0].original_ref == retained.ref
    assert versions[0].chunk_count == 3
    assert versions[0].changes["title"] == {"from": "A", "to": "second"}


async def test_releasing_expired_versions_lets_the_blob_collector_reclaim_them(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path
) -> None:
    """Recording a version must not grow the blob store without bound.

    ``document_versions.original_ref`` is in the mark-and-sweep query, so a version row pins
    its bytes for as long as it is set. Without a release, adding version history would repeal
    the retention policy by accident — every superseded draft of every page kept for ever, with
    nothing in the code saying that was the decision.
    """
    blobs = BlobStore(engine, data_dir)
    retained = await blobs.retain(b"the first draft", "text/markdown")
    assert retained.ref is not None
    document = (await _seed(store, 1))[0]
    await store.set_original(document.id, ref=retained.ref, omitted_reason=None)

    reloaded = await store.get_document(document.id)
    assert reloaded is not None
    await store.upsert_document(
        reloaded.model_copy(
            update={"content_hash": content_hash(b"the second draft"), "original_ref": None}
        )
    )
    assert await blobs.collect_garbage() == [], "the version still needs those bytes"

    released = await store.release_expired_versions(datetime.now(UTC) + timedelta(seconds=1))
    assert released == 1
    assert await blobs.collect_garbage() == [retained.ref]

    versions = await store.list_versions(document.id)
    assert versions[0].original_ref is None
    assert "bytes_released_at" in versions[0].changes, (
        "'never retained' and 'retained and since reclaimed' are different facts, and a bare "
        "NULL is both"
    )


async def test_a_citation_into_a_live_chunk_resolves(store: SqliteDocStore) -> None:
    document = (await _seed(store, 1))[0]
    chunk_ids = await _with_chunks(store, document, 1)
    resolution = await store.resolve_citation(document.id, chunk_ids[0])
    assert resolution.state is CitationState.PRESENT
    assert resolution.resolved
    assert resolution.chunk is not None
    assert resolution.chunk.id == chunk_ids[0]


async def test_a_citation_into_the_trash_says_so_rather_than_reporting_it_missing(
    store: SqliteDocStore,
) -> None:
    """ "Deleted" and "superseded" have different remedies, so they are different answers."""
    document = (await _seed(store, 1))[0]
    chunk_ids = await _with_chunks(store, document, 1)
    await store.soft_delete_document(document.id)

    resolution = await store.resolve_citation(document.id, chunk_ids[0])
    assert resolution.state is CitationState.DELETED
    assert "grace period" in resolution.reason


async def test_a_re_ingest_that_keeps_a_chunk_keeps_its_citation(store: SqliteDocStore) -> None:
    """The other half of the versioning rule, and the reason ids are content-derived.

    A chunk that survives a re-parse unchanged keeps its id, so its citation keeps resolving
    and its vector is never recomputed. Only the chunk that actually changed dangles.
    """
    document = (await _seed(store, 1))[0]
    unchanged = make_chunk(document, 0, "the stable paragraph")
    edited = make_chunk(document, 1, "the paragraph that changes")
    await store.replace_chunks(document.id, [unchanged, edited])

    rewritten = make_chunk(document, 1, "the paragraph after the edit")
    await store.replace_chunks(document.id, [unchanged, rewritten])
    reloaded = await store.get_document(document.id)
    assert reloaded is not None
    await store.upsert_document(
        reloaded.model_copy(update={"content_hash": content_hash(b"edited body")})
    )

    kept = await store.resolve_citation(document.id, unchanged.id)
    assert kept.state is CitationState.PRESENT

    lost = await store.resolve_citation(document.id, edited.id)
    assert lost.state is CitationState.SUPERSEDED
    assert lost.chunk is None, "a superseded citation must not be answered with its replacement"


# --- the trash ------------------------------------------------------------------------------


@pytest.mark.contract
async def test_the_sqlite_store_passes_the_trash_suite(store: SqliteDocStore) -> None:
    documents = await _seed(store, 1)
    await assert_trash_store_contract(store, store, document_id=documents[0].id)


async def test_restoring_after_the_sweep_names_the_rung_the_repair_lands_on(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path
) -> None:
    """The sweep took the content; the row comes back empty and the caller has to be told.

    A restore that reported plain success here would hand back a document with no chunks,
    invisible to every search, with nothing to explain why. ``pending`` is what that state
    already means everywhere else, and ``needs_reparse`` is what names the repair.

    Both documents need their content re-derived and they need *different* operations to get
    it: one has retained bytes and is a single-document re-parse on this machine, the other has
    none and can only come back through a forced re-sync that talks to the source and may fail.
    A restore that reported the same sentence for both would send an operator to run a command
    that cannot work.
    """
    blobs = BlobStore(engine, data_dir)
    retained = await blobs.retain(b"the bytes that were kept", "text/markdown")
    kept, lost = await _seed(store, 2)
    await store.set_original(kept.id, ref=retained.ref, omitted_reason=None)

    for document in (kept, lost):
        await _with_chunks(store, document, 2)
        await store.soft_delete_document(document.id)
        # Exactly what `sweep_vectors` does once the grace period has passed.
        await store.replace_chunks(document.id, [])
        await store.set_status(document.id, DocumentStatus.DELETED, "grace period expired")

    repairable = await store.restore_document(kept.id)
    assert repairable.restored
    assert repairable.needs_reparse
    assert "Re-parse it from its retained bytes" in repairable.reason

    unrepairable = await store.restore_document(lost.id)
    assert unrepairable.needs_reparse
    assert "re-sync from the source" in unrepairable.reason

    restored = await store.get_document(kept.id)
    assert restored is not None
    assert restored.status is DocumentStatus.PENDING


async def test_a_restored_document_is_no_longer_offered_to_the_sweep(
    store: SqliteDocStore,
) -> None:
    """Restore and the sweep must not fight, and the only shared state is ``deleted_at``.

    The sweep selects on ``deleted_at IS NOT NULL``; a restore that cleared a status but left
    the timestamp would put a live document back on the purge list, and the purge would happen
    quietly a month later.
    """
    document = (await _seed(store, 1))[0]
    await store.soft_delete_document(document.id)
    cutoff = datetime.now(UTC) + timedelta(seconds=1)
    assert list(await store.soft_deleted_before(cutoff)) == [document.id]

    await store.restore_document(document.id)
    assert not await store.soft_deleted_before(cutoff)


async def test_the_trash_lists_the_longest_deleted_first(store: SqliteDocStore) -> None:
    """The order the sweep will take them in, so a listing predicts what goes next."""
    first, second = await _seed(store, 2)
    await store.soft_delete_document(first.id)
    await store.soft_delete_document(second.id)

    entries = await store.list_trash(grace_s=60.0)
    assert [entry.document.id for entry in entries] == [first.id, second.id]
    assert all(entry.free_restore for entry in entries)
    assert entries[0].restorable_until == entries[0].deleted_at + timedelta(seconds=60)


# --- resolving a filter -----------------------------------------------------------------------


async def test_an_empty_collection_refuses_the_query_rather_than_widening_it(
    store: SqliteDocStore,
) -> None:
    """The trap this resolution exists to avoid, and it inverts the restriction.

    ``Filter.document_ids`` defaults to empty and an empty field restricts nothing. So
    resolving an empty collection into ``document_ids=frozenset()`` would turn the narrowest
    request anyone can make into a search of the entire workspace — ranked, plausible, and
    exactly backwards.
    """
    await _seed(store, 2)
    empty = await store.create_collection("nothing yet")
    scope = Filter(
        workspace_ids=frozenset({DEFAULT_WORKSPACE}), collection_ids=frozenset({empty.id})
    )

    assert await resolve_filter(scope, collections=store, tags=store) is None


async def test_resolving_a_collection_and_a_tag_keeps_only_documents_in_both(
    store: SqliteDocStore,
) -> None:
    """Conjunction between fields, disjunction within one — the same rule ``Filter`` states."""
    first, second = await _seed(store, 2)
    collection = await store.create_collection("both")
    await store.add_to_collection(collection.id, [first.id, second.id])
    tag = await store.ensure_tag("only-one")
    await store.tag_document(first.id, [tag.id])

    scope = Filter(
        workspace_ids=frozenset({DEFAULT_WORKSPACE}),
        collection_ids=frozenset({collection.id}),
        tag_ids=frozenset({tag.id}),
    )
    resolved = await resolve_filter(scope, collections=store, tags=store)

    assert resolved is not None
    assert resolved.document_ids == frozenset({first.id})
    assert not resolved.collection_ids
    assert not resolved.tag_ids


async def test_a_collection_too_large_to_carry_is_refused_rather_than_truncated(
    store: SqliteDocStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same silent wrongness as an empty collection, arriving from the other end.

    A truncated id set is a filter that looks complete while excluding documents that are in
    the collection — and the caller has nothing to notice it by. It cannot degrade gracefully
    either: the ids reach SQLite as one bind parameter each, against a limit of 32 766 on a
    modern build, so a large list fails somewhere that reads as a bug in search.
    """
    documents = await _seed(store, 3)
    collection = await store.create_collection("large")
    await store.add_to_collection(collection.id, [document.id for document in documents])
    monkeypatch.setattr(organization, "MAX_RESOLVED_DOCUMENTS", 2)

    scope = Filter(
        workspace_ids=frozenset({DEFAULT_WORKSPACE}), collection_ids=frozenset({collection.id})
    )
    with pytest.raises(ValueError, match="more than 2 documents"):
        await resolve_filter(scope, collections=store, tags=store)


async def test_a_filter_naming_neither_is_returned_untouched(store: SqliteDocStore) -> None:
    """Resolution is a no-op for the filters that need none, so it can be applied everywhere."""
    scope = Filter(workspace_ids=frozenset({DEFAULT_WORKSPACE}), sources=frozenset({"fs"}))
    assert await resolve_filter(scope, collections=store, tags=store) is scope


async def test_the_store_still_refuses_a_filter_it_cannot_resolve(store: SqliteDocStore) -> None:
    """The refusal is what makes the resolution step mandatory rather than advisory.

    If ``list_documents`` quietly ignored ``collection_ids``, a caller who forgot to resolve
    would get every document in the workspace back and no indication that the restriction was
    dropped.
    """
    collection = await store.create_collection("unused")
    scope = Filter(
        workspace_ids=frozenset({DEFAULT_WORKSPACE}), collection_ids=frozenset({collection.id})
    )
    with pytest.raises(ValueError, match="collection_ids"):
        await store.list_documents(scope)
