"""Purging a soft-deleted document gives its retained bytes back, and only its own.

``docs/storage.md`` §8.2. The sweep used to flip a soft-deleted document to ``deleted`` and stop,
leaving ``original_ref`` on the document and on every version of it; blob collection treats
those as roots whatever the status, so the raw bytes of a deleted document were kept for ever.
:meth:`~manicule.storage.history.TrashMixin.purge_document` is the one transaction that ends
that, and every test here runs it against the real store and the real blob store.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import func, select, update

from manicule.core.content import DocumentStatus
from manicule.core.ids import content_hash
from manicule.core.organization import CitationState
from manicule.storage import history, models
from manicule.storage.blobs import BlobStore
from manicule.storage.docstore import SqliteDocStore
from manicule.storage.engine import session_factory
from manicule.storage.history import RELEASED_AFTER_GRACE_REASON
from tests.storage_helpers import make_chunk, make_document

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine

REASON = "grace period expired; purged"


def _after() -> datetime:
    return datetime.now(UTC) + timedelta(seconds=1)


async def _retained_document(
    store: SqliteDocStore, blobs: BlobStore, body: bytes = b"the first draft"
) -> tuple[str, str]:
    """A live document with bytes retained, one chunk, and one superseded version.

    The version is what makes this the realistic case: a document that has been edited holds a
    reference to its current bytes *and* one to the bytes it left, and a purge that released
    only the first would leave the second pinning a blob for the life of the install.
    """
    first = await blobs.retain(body, "text/markdown")
    assert first.ref is not None
    document = (await store.upsert_document(make_document())).model_copy()
    await store.set_original(document.id, ref=first.ref, omitted_reason=None)
    reloaded = await store.get_document(document.id)
    assert reloaded is not None
    second = await blobs.retain(b"the second draft", "text/markdown")
    assert second.ref is not None
    await store.upsert_document(
        reloaded.model_copy(
            update={"content_hash": content_hash(b"the second draft"), "original_ref": None}
        )
    )
    await store.set_original(document.id, ref=second.ref, omitted_reason=None)
    await store.replace_chunks(document.id, [make_chunk(document, 0, "second draft text")])
    return document.id, first.ref


async def _chunk_count(engine: AsyncEngine, document_id: str) -> int:
    """Chunk rows for a document, trashed or not — the store's own reads hide trashed ones."""
    async with session_factory(engine)() as session:
        return (
            await session.execute(
                select(func.count())
                .select_from(models.Chunk)
                .where(models.Chunk.document_id == document_id)
            )
        ).scalar_one()


async def _row(engine: AsyncEngine, document_id: str) -> models.Document:
    async with session_factory(engine)() as session:
        row = await session.get(models.Document, document_id)
        assert row is not None
        session.expunge(row)
        return row


async def test_purge_releases_the_document_its_versions_and_its_chunks_together(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path
) -> None:
    """The whole claim: nothing the document or its history referenced is still a root."""
    blobs = BlobStore(engine, data_dir)
    document_id, old_ref = await _retained_document(store, blobs)
    current = await _row(engine, document_id)
    assert current.original_ref is not None
    current_ref = current.original_ref
    assert await blobs.collect_garbage() == [], "everything is referenced while the document lives"

    await store.soft_delete_document(document_id)
    assert await blobs.collect_garbage() == [], "a soft delete alone releases nothing"
    assert await store.purge_document(document_id, reason=REASON) is True

    row = await _row(engine, document_id)
    assert row.status is DocumentStatus.DELETED
    assert row.status_detail == REASON
    assert row.deleted_at is not None
    assert row.original_ref is None
    assert row.original_omitted_reason == RELEASED_AFTER_GRACE_REASON
    assert await _chunk_count(engine, document_id) == 0
    versions = await store.list_versions(document_id)
    assert len(versions) == 1
    assert versions[0].original_ref is None
    assert "bytes_released_at" in versions[0].changes
    async with session_factory(engine)() as session:
        version_row = await session.get(models.DocumentVersion, versions[0].id)
        assert version_row is not None
        assert version_row.bytes_released_at is not None

    assert sorted(await blobs.collect_garbage()) == sorted([old_ref, current_ref]), (
        "both the document's bytes and the version's are now garbage"
    )


async def test_the_purge_leaves_tombstones_for_the_vectors_it_orphaned(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path
) -> None:
    """The chunk delete and the status flip commit together, so a crash cannot lose the vectors.

    The vector store is not part of the transaction. What makes that safe is the trigger
    that writes a tombstone for every chunk deleted: the sweep that purged, or the next one,
    reads them.
    """
    blobs = BlobStore(engine, data_dir)
    document_id, _ = await _retained_document(store, blobs)
    assert not await store.take_tombstones(10)
    await store.soft_delete_document(document_id)

    assert await store.purge_document(document_id, reason=REASON)

    assert len(await store.take_tombstones(10)) == 1


async def test_a_document_whose_bytes_were_never_kept_keeps_its_own_reason(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    """'Never retained' and 'retained and released' must stay different facts."""
    document = await store.upsert_document(make_document())
    await store.set_original(document.id, ref=None, omitted_reason="over the retention cap")
    await store.soft_delete_document(document.id)

    assert await store.purge_document(document.id, reason=REASON)

    assert (await _row(engine, document.id)).original_omitted_reason == "over the retention cap"


async def test_purging_twice_is_a_no_op_that_says_so(store: SqliteDocStore) -> None:
    document = await store.upsert_document(make_document())
    await store.soft_delete_document(document.id)

    assert await store.purge_document(document.id, reason=REASON) is True
    assert await store.purge_document(document.id, reason=REASON) is False


async def test_a_live_document_cannot_be_purged(store: SqliteDocStore, engine: AsyncEngine) -> None:
    document = await store.upsert_document(make_document())
    await store.replace_chunks(document.id, [make_chunk(document, 0, "live text")])

    assert await store.purge_document(document.id, reason=REASON) is False

    row = await _row(engine, document.id)
    assert row.status is DocumentStatus.INDEXED
    assert await _chunk_count(engine, document.id) == 1


async def test_a_restore_that_lands_before_the_purge_wins(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path
) -> None:
    """The sweep selects, a person restores, the sweep purges: the person must win.

    Selection and purge are separate steps with the vector store in between, so a restore can
    land in the gap. The purge re-checks inside its own transaction; if it only trusted the
    selection it would take the chunks and the bytes from a document somebody had just put
    back, and mark a live document ``deleted``.
    """
    blobs = BlobStore(engine, data_dir)
    document_id, _ = await _retained_document(store, blobs)
    await store.soft_delete_document(document_id)
    selected = await store.soft_deleted_before(_after())
    assert list(selected) == [document_id]
    before = await _row(engine, document_id)

    restoration = await store.restore_document(document_id)
    assert restoration.restored
    assert not restoration.needs_reparse
    assert await store.purge_document(document_id, reason=REASON, deleted_before=_after()) is False

    row = await _row(engine, document_id)
    assert row.status is before.status
    assert row.deleted_at is None
    assert row.original_ref == before.original_ref
    assert await _chunk_count(engine, document_id) == 1
    assert await blobs.collect_garbage() == []


async def test_a_document_deleted_again_after_a_restore_gets_a_fresh_grace_period(
    store: SqliteDocStore, engine: AsyncEngine
) -> None:
    """The cutoff the sweep selected with is part of the re-check.

    Restore, then delete again, inside the window between selection and purge: ``deleted_at`` is
    set again and is recent. Without the cutoff in the predicate the purge would see "in the
    trash" and take a document that has been there for a second.
    """
    document = await store.upsert_document(make_document())
    await store.soft_delete_document(document.id)
    cutoff = datetime.now(UTC) + timedelta(seconds=10)
    assert list(await store.soft_deleted_before(cutoff)) == [document.id]
    await store.restore_document(document.id)
    await store.soft_delete_document(document.id)
    async with session_factory(engine).begin() as session:
        await session.execute(
            update(models.Document)
            .where(models.Document.id == document.id)
            .values(deleted_at=cutoff + timedelta(seconds=5))
        )

    assert await store.purge_document(document.id, reason=REASON, deleted_before=cutoff) is False

    assert (await _row(engine, document.id)).status is DocumentStatus.INDEXED


async def test_the_purge_is_one_transaction(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure at the last step must undo the first.

    ``soft_deleted_before`` skips ``deleted`` rows, so a purge that committed the status flip and
    then failed to release the bytes would never be retried and would pin them for ever.
    ``utcnow`` is called by the release step, after the status update and the chunk delete have
    run; making it fail is how this reaches the last step without reaching into the store.
    """
    blobs = BlobStore(engine, data_dir)
    document_id, _ = await _retained_document(store, blobs)
    await store.soft_delete_document(document_id)
    before = await _row(engine, document_id)

    def fail() -> datetime:
        message = "the release step failed"
        raise RuntimeError(message)

    monkeypatch.setattr(history, "utcnow", fail)
    with pytest.raises(RuntimeError, match="release step"):
        await store.purge_document(document_id, reason=REASON)
    monkeypatch.undo()

    row = await _row(engine, document_id)
    assert row.status is before.status, "the status flip must not have survived the failure"
    assert row.original_ref == before.original_ref
    assert await _chunk_count(engine, document_id) == 1
    assert list(await store.soft_deleted_before(_after())) == [document_id], "so a retry finds it"


async def test_a_derived_reset_document_keeps_its_bytes_through_the_sweep(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path
) -> None:
    """``reset_derived`` also leaves documents ``deleted`` with ``deleted_at`` set.

    Their retained bytes are what the rebuild reads, so nothing may release them: not the sweep's
    selection (it skips ``deleted``), not the purge itself (its predicate does), and not blob
    collection (it was deliberately not given a status predicate for this reason).
    """
    blobs = BlobStore(engine, data_dir)
    kept = await blobs.retain(b"authoritative", "text/plain")
    assert kept.ref is not None
    document = await store.upsert_document(make_document(source="wiki", source_id="page-1"))
    await store.set_original(document.id, ref=kept.ref, omitted_reason=None)

    await store.reset_derived()

    row = await _row(engine, document.id)
    assert row.status is DocumentStatus.DELETED
    assert row.deleted_at is not None
    assert row.original_ref == kept.ref
    assert not await store.soft_deleted_before(_after()), "the sweep never selects it"
    assert await store.purge_document(document.id, reason=REASON) is False
    assert (await _row(engine, document.id)).original_ref == kept.ref
    assert await blobs.collect_garbage() == [], "and collection still treats it as a root"


# --- what a restore and a citation say about each kind of purged row ---------------------------


async def test_restore_after_a_release_says_only_a_resync_can_help(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path
) -> None:
    blobs = BlobStore(engine, data_dir)
    document_id, _ = await _retained_document(store, blobs)
    await store.soft_delete_document(document_id)
    await store.purge_document(document_id, reason=REASON)

    resolution = await store.resolve_citation(document_id, "any-chunk")
    assert resolution.state is CitationState.DELETED
    assert "released" in resolution.reason
    assert "re-parse" not in resolution.reason

    restoration = await store.restore_document(document_id)

    assert restoration.restored
    assert restoration.needs_reparse
    assert "released" in restoration.reason
    assert "re-sync" in restoration.reason
    assert "re-parse it from its retained bytes" not in restoration.reason


async def test_restore_of_a_row_purged_before_bytes_were_released_still_offers_the_reparse(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path
) -> None:
    """Rows the sweep flipped to ``deleted`` before this existed still hold their reference.

    The sweep never revisits them, so for them the cheap repair is real, and the message must
    not claim a release that never happened.
    """
    blobs = BlobStore(engine, data_dir)
    document_id, _ = await _retained_document(store, blobs)
    await store.soft_delete_document(document_id)
    await store.replace_chunks(document_id, [])
    await store.set_status(document_id, DocumentStatus.DELETED, "purged by an older sweep")

    resolution = await store.resolve_citation(document_id, "any-chunk")
    assert "re-parse it from its retained bytes" in resolution.reason

    restoration = await store.restore_document(document_id)

    assert restoration.needs_reparse
    assert "re-parse it from its retained bytes" in restoration.reason
    assert "released" not in restoration.reason


async def test_restore_of_a_purged_row_that_never_kept_bytes_says_so(
    store: SqliteDocStore,
) -> None:
    document = await store.upsert_document(make_document())
    await store.set_original(document.id, ref=None, omitted_reason="over the retention cap")
    await store.soft_delete_document(document.id)
    await store.purge_document(document.id, reason=REASON)

    restoration = await store.restore_document(document.id)

    assert restoration.needs_reparse
    assert "not retained" in restoration.reason
    assert "released" not in restoration.reason


# --- the age floor that lets collection run beside ingest --------------------------------------


async def test_collection_with_an_age_floor_leaves_a_fresh_unreferenced_blob_alone(
    engine: AsyncEngine, data_dir: Path
) -> None:
    """A blob is written before the document that references it commits.

    Between the two nothing references it, and a collection that deleted it there would leave a
    publication pointing at bytes that no longer exist. The scheduled sweep passes a floor so
    that window is never the collector's to act on; the default of zero keeps immediate
    collection for callers with no concurrent writer.
    """
    blobs = BlobStore(engine, data_dir)
    fresh = await blobs.retain(b"just written, not yet referenced", "text/plain")
    old = await blobs.retain(b"written long ago, never referenced", "text/plain")
    assert fresh.ref is not None
    assert old.ref is not None
    async with session_factory(engine).begin() as session:
        await session.execute(
            update(models.Blob)
            .where(models.Blob.hash == old.ref)
            .values(created_at=datetime.now(UTC) - timedelta(days=2))
        )

    assert await blobs.collect_garbage(min_age_s=24 * 3600.0) == [old.ref]
    assert blobs.path_for(fresh.ref).exists()

    assert await blobs.collect_garbage() == [fresh.ref]
