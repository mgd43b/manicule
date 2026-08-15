from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast, override

import pytest
from pydantic import JsonValue
from sqlalchemy import delete, select, text, update

from manicule.core.acquisition import (
    AcquiredSource,
    AcquisitionRecordState,
    AcquisitionSource,
)
from manicule.core.anchors import Unlocated
from manicule.core.content import Chunk, Document, DocumentStatus, RawDocument
from manicule.core.embedding import EmbedFingerprint, Pooling
from manicule.core.fingerprints import ChunkFingerprint
from manicule.core.ids import chunk_id, content_hash
from manicule.core.rebuild import (
    DerivedReplacement,
    RebuildCheckpoint,
    RebuildRefusalCode,
    RebuildState,
    RebuildTarget,
)
from manicule.core.sources import DiscoveredDoc, DocRef, Watermark
from manicule.ingest.rebuild import OfflineDeriver, OfflineGenerationRebuilder
from manicule.storage import models
from manicule.storage.blobs import BlobStore, StoredBlob
from manicule.storage.docstore import SqliteDocStore
from manicule.storage.engine import session_factory
from manicule.storage.rebuild import RebuildLeaseConflictError, SqliteRebuildStore
from manicule.storage.vectors import LanceVectorStore
from tests.storage_helpers import make_chunk, make_document

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

NOW = datetime(2026, 8, 15, 12, tzinfo=UTC)


class ReplayBarrierRebuildStore(SqliteRebuildStore):
    """Pause after the serialized binding read but before published replay lookup."""

    replay_observed: asyncio.Event
    replay_release: asyncio.Event

    @override
    async def _published_replay(
        self,
        session: AsyncSession,
        *,
        snapshot_run_id: str,
        target_digest: str,
        vector_table: str | None,
        vector_inventory_digest: str | None,
    ) -> models.DerivedGeneration | None:
        self.replay_observed.set()
        await self.replay_release.wait()
        return await super()._published_replay(
            session,
            snapshot_run_id=snapshot_run_id,
            target_digest=target_digest,
            vector_table=vector_table,
            vector_inventory_digest=vector_inventory_digest,
        )


async def promoted_snapshot(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    *,
    run_id: str = "promoted-run",
    connector: str = "wiki",
    scope_fingerprint: str = "scope-v1",
    source_id: str = "page-1",
) -> tuple[str, str, RawDocument]:
    raw = RawDocument(
        source_id=source_id,
        uri=f"https://source.invalid/{source_id}",
        media_type="text/plain",
        content="replacement searchable text",
    )
    source = AcquisitionSource.from_discovered(
        DiscoveredDoc(
            ref=DocRef(source_id=raw.source_id, uri=raw.uri),
            version_token="v2",  # noqa: S106 - source revision, not a credential
            media_type=raw.media_type,
            size_bytes=len(raw.as_bytes()),
        )
    )
    run = await store.create_acquisition_run(
        run_id,
        connector,
        source_scope=f"scope:{scope_fingerprint}",
        scope_fingerprint=scope_fingerprint,
    )
    claimed = await store.claim_acquisition_run(
        run.id, "worker", now=NOW, expires_at=NOW + timedelta(minutes=5)
    )
    assert claimed is not None
    await store.append_acquisition_record(
        run.id,
        0,
        source,
        lease_owner="worker",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
    await store.complete_acquisition_enumeration(
        run.id,
        Watermark(value="v2", observed_at=NOW),
        lease_owner="worker",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
    blob = await BlobStore(engine, data_dir).put(raw.as_bytes(), raw.media_type)
    assert isinstance(blob, StoredBlob)
    await store.transition_acquisition_record(
        run.id,
        raw.source_id,
        AcquisitionRecordState.DISCOVERED,
        AcquisitionRecordState.ACQUIRING,
        lease_owner="worker",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
    await store.transition_acquisition_record(
        run.id,
        raw.source_id,
        AcquisitionRecordState.ACQUIRING,
        AcquisitionRecordState.ACQUIRED,
        lease_owner="worker",
        lease_generation=claimed.lease_generation,
        now=NOW,
        blob_ref=blob.hash,
        acquired_source=AcquiredSource.from_raw(raw),
        fetched_version_token="v2",  # noqa: S106 - source revision, not a credential
    )
    await store.complete_snapshot_acquisition(
        run.id,
        lease_owner="worker",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
    await store.promote_snapshot_and_commit_watermark(
        run.id,
        expected_scope_fingerprint=scope_fingerprint,
        lease_owner="worker",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
    return run.id, blob.hash, raw


async def publish_one_replacement(
    rebuilds: SqliteRebuildStore,
    vectors: LanceVectorStore,
    *,
    estimate_id: str,
    target: RebuildTarget,
    document: Document,
    raw: RawDocument,
    blob_ref: str,
    owner: str,
) -> RebuildCheckpoint:
    """Stage and publish one retained snapshot member for composition regressions."""
    claimed = await rebuilds.claim_generation(
        estimate_id,
        owner,
        now=NOW,
        expires_at=NOW + timedelta(minutes=5),
    )
    replacement_document = document.model_copy(
        update={
            "publication_id": estimate_id,
            "content_hash": content_hash(raw.as_bytes()),
            "version_token": "v2",
            "status": DocumentStatus.INDEXED,
            "original_ref": blob_ref,
        }
    )
    replacement_chunk = Chunk(
        id=chunk_id(replacement_document.id, 0, raw.as_text()),
        document_id=replacement_document.id,
        text=raw.as_text(),
        embed_text=raw.as_text(),
        anchor=Unlocated(reason="plain text"),
        position=0,
        token_count=3,
    )
    replacement = DerivedReplacement(
        document=replacement_document,
        chunks=(replacement_chunk,),
        parse_fingerprint="plain@2",
        vector_embedded=1,
    )
    await vectors.upsert(
        [replacement_chunk],
        [[1.0, 0.0, 0.0, 0.0]],
        publication_id=claimed.vector_publication_id,
    )
    await rebuilds.stage_replacements(
        estimate_id,
        [(0, replacement)],
        expected_next_sequence=0,
        owner=owner,
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
    await rebuilds.begin_validation(
        estimate_id,
        owner=owner,
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
    await rebuilds.validate_generation(estimate_id)
    return await rebuilds.publish_generation(
        estimate_id,
        owner=owner,
        lease_generation=claimed.lease_generation,
        now=NOW,
    )


async def test_live_vector_swap_gets_a_new_plan_and_published_replay_is_idempotent(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    run_id, blob_ref, raw = await promoted_snapshot(store, engine, data_dir)
    old = make_document(
        source="wiki",
        source_id=raw.source_id,
        body=b"old text",
        uri=raw.uri,
        media_type=raw.media_type,
    ).model_copy(update={"original_ref": blob_ref})
    await store.upsert_document(old)
    await store.replace_chunks(old.id, [make_chunk(old, 0, "old text")])
    embed = EmbedFingerprint(
        model_id="test/embed",
        dimension=4,
        pooling=Pooling.MEAN,
        normalized=True,
        tokenizer_id="test/tokenizer",
        max_sequence_length=128,
    )
    target = RebuildTarget(
        parser_routing="routing-v2",
        parser_set=("plain@2",),
        chunk_fingerprint="chunk-v2",
        embedding_fingerprint=embed.canonical(),
        glossary_fingerprint="glossary-v2",
        fts_tokenizer="unicode61",
        batch_documents=1,
        max_memory_bytes=1_000_000,
        max_temporary_bytes=1_000_000,
    )
    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(embed)
    rebuilds = SqliteRebuildStore(
        engine,
        workspace_id=store.workspace_id,
        blobs=BlobStore(engine, data_dir),
        vectors=vectors,
    )
    stale = await rebuilds.plan_rebuild(run_id, target, missing_limit=10)
    sessions = session_factory(engine)
    async with sessions.begin() as session:
        state = await session.get(models.IndexState, 1)
        if state is None:
            session.add(
                models.IndexState(
                    id=1,
                    vector_table="reembed-winner",
                    vector_inventory_digest="winner-inventory",
                )
            )
        else:
            state.vector_table = "reembed-winner"
            state.vector_inventory_digest = "winner-inventory"

    winner = await rebuilds.plan_rebuild(run_id, target, missing_limit=10)
    assert winner.generation_id != stale.generation_id
    published = await publish_one_replacement(
        rebuilds,
        vectors,
        estimate_id=winner.generation_id,
        target=target,
        document=old,
        raw=raw,
        blob_ref=blob_ref,
        owner="winner-worker",
    )
    assert published.state is RebuildState.PUBLISHED

    runner = OfflineGenerationRebuilder(
        store=rebuilds,
        blobs=BlobStore(engine, data_dir),
        deriver=cast("OfflineDeriver", object()),
    )
    repeated = await runner.dry_run(run_id, target, missing_limit=10)
    assert repeated.generation_id == winner.generation_id
    replay = await runner.run(run_id, target, owner="replay-worker")
    assert replay.state is RebuildState.PUBLISHED

    # A #187 swap that starts after the binding read cannot commit ahead of the replay lookup.
    # The planner holds SQLite's writer slot across both observations, so it returns one coherent
    # old decision; the next plan observes the fully committed winner and creates a new identity.
    barrier = ReplayBarrierRebuildStore(
        engine,
        workspace_id=store.workspace_id,
        blobs=BlobStore(engine, data_dir),
        vectors=vectors,
    )
    barrier.replay_observed = asyncio.Event()
    barrier.replay_release = asyncio.Event()
    planning = asyncio.create_task(barrier.plan_rebuild(run_id, target, missing_limit=10))
    await barrier.replay_observed.wait()
    swap_started = asyncio.Event()

    async def swap_live_pointer() -> None:
        async with sessions.begin() as session:
            swap_started.set()
            await session.execute(
                update(models.IndexState)
                .where(models.IndexState.id == 1)
                .values(
                    vector_table="later-reembed-winner",
                    vector_inventory_digest="later-winner-inventory",
                )
            )

    swapping = asyncio.create_task(swap_live_pointer())
    await swap_started.wait()
    await asyncio.sleep(0)
    assert not swapping.done()
    barrier.replay_release.set()
    serialized = await planning
    assert serialized.generation_id == winner.generation_id
    await swapping
    after_swap = await rebuilds.plan_rebuild(run_id, target, missing_limit=10)
    assert after_swap.generation_id not in {stale.generation_id, winner.generation_id}


async def test_second_connector_promotion_fences_a_single_scope_rebuild(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    first_run, _, _ = await promoted_snapshot(store, engine, data_dir)
    embed = EmbedFingerprint(
        model_id="test/embed",
        dimension=4,
        pooling=Pooling.MEAN,
        normalized=True,
        tokenizer_id="test/tokenizer",
        max_sequence_length=128,
    )
    target = RebuildTarget(
        parser_routing="routing-v2",
        parser_set=("plain@2",),
        chunk_fingerprint="chunk-v2",
        embedding_fingerprint=embed.canonical(),
        glossary_fingerprint="glossary-v2",
        fts_tokenizer="unicode61",
        batch_documents=1,
        max_memory_bytes=1_000_000,
        max_temporary_bytes=1_000_000,
    )
    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(embed)
    rebuilds = SqliteRebuildStore(
        engine,
        workspace_id=store.workspace_id,
        blobs=BlobStore(engine, data_dir),
        vectors=vectors,
    )
    first = await rebuilds.plan_rebuild(first_run, target, missing_limit=10)
    assert first.runnable
    claimed = await rebuilds.claim_generation(
        first.generation_id,
        "first-worker",
        now=NOW,
        expires_at=NOW + timedelta(minutes=5),
    )

    second_run, _, _ = await promoted_snapshot(
        store,
        engine,
        data_dir,
        run_id="drive-promoted-run",
        connector="drive",
        scope_fingerprint="folder-v1",
        source_id="drive-page-1",
    )
    refused_first = await rebuilds.plan_rebuild(first_run, target, missing_limit=10)
    refused_second = await rebuilds.plan_rebuild(second_run, target, missing_limit=10)
    assert refused_first.refusal is RebuildRefusalCode.WORKSPACE_SCOPE_CHANGED
    assert refused_second.refusal is RebuildRefusalCode.WORKSPACE_SCOPE_CHANGED
    with pytest.raises(RebuildLeaseConflictError, match="workspace_scope_changed"):
        await rebuilds.assert_generation_lease(
            first.generation_id,
            "first-worker",
            claimed.lease_generation,
            now=NOW,
        )
    await rebuilds.begin_validation(
        first.generation_id,
        owner="first-worker",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
    with pytest.raises(RuntimeError, match="workspace_scope_changed"):
        await rebuilds.publish_generation(
            first.generation_id,
            owner="first-worker",
            lease_generation=claimed.lease_generation,
            now=NOW,
        )


async def test_shadow_generation_is_invisible_until_one_atomic_publication(  # noqa: PLR0915
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    run_id, blob_ref, raw = await promoted_snapshot(store, engine, data_dir)
    old = make_document(
        source="wiki",
        source_id=raw.source_id,
        body=b"old searchable text",
        uri=raw.uri,
        media_type=raw.media_type,
    )
    old = old.model_copy(update={"original_ref": blob_ref})
    await store.upsert_document(old)
    await store.replace_chunks(old.id, [make_chunk(old, 0, "old searchable text")])

    embed = EmbedFingerprint(
        model_id="test/embed",
        dimension=4,
        pooling=Pooling.MEAN,
        normalized=True,
        tokenizer_id="test/tokenizer",
        max_sequence_length=128,
    )
    chunking = ChunkFingerprint(
        chunker="structural",
        version="2",
        max_tokens=64,
        overlap_tokens=8,
        tokenizer_id="test/tokenizer",
    )
    target = RebuildTarget(
        parser_routing="routing-v2",
        parser_set=("plain@2",),
        chunk_fingerprint=chunking.canonical(),
        embedding_fingerprint=embed.canonical(),
        glossary_fingerprint="glossary-v2",
        fts_tokenizer="unicode61",
        batch_documents=1,
        max_memory_bytes=1_000_000,
        max_temporary_bytes=1_000_000,
    )
    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(embed)
    rebuilds = SqliteRebuildStore(
        engine,
        workspace_id=store.workspace_id,
        blobs=BlobStore(engine, data_dir),
        vectors=vectors,
    )
    estimate = await rebuilds.plan_rebuild(run_id, target, missing_limit=10)
    assert estimate.runnable
    assert await rebuilds.plan_rebuild(run_id, target, missing_limit=10) == estimate
    assert (await rebuilds.checkpoint(estimate.generation_id)).state is RebuildState.PLANNED
    claimed = await rebuilds.claim_generation(
        estimate.generation_id,
        "worker",
        now=NOW,
        expires_at=NOW + timedelta(minutes=5),
    )
    replacement_document = old.model_copy(
        update={
            "publication_id": estimate.generation_id,
            "content_hash": content_hash(raw.as_bytes()),
            "version_token": "v2",
            "status": DocumentStatus.INDEXED,
        }
    )
    replacement_chunk = Chunk(
        id=chunk_id(old.id, 0, raw.as_text()),
        document_id=old.id,
        text=raw.as_text(),
        embed_text=raw.as_text(),
        anchor=Unlocated(reason="plain text"),
        position=0,
        token_count=3,
    )
    replacement = DerivedReplacement(
        document=replacement_document,
        chunks=(replacement_chunk,),
        parse_fingerprint="plain@2",
        vector_embedded=1,
    )
    await vectors.upsert(
        [replacement_chunk],
        [[1.0, 0.0, 0.0, 0.0]],
        publication_id=claimed.vector_publication_id,
    )
    await rebuilds.stage_replacements(
        estimate.generation_id,
        [(0, replacement)],
        expected_next_sequence=0,
        owner="worker",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )

    # Shadow rows and vectors exist, but every ordinary reader still sees one coherent old
    # publication until validation and the relational flip commit.
    still_old = await store.get_document(old.id)
    assert still_old is not None
    assert still_old.publication_id == old.publication_id
    assert (await store.document_chunks(old.id))[0].text == "old searchable text"

    await rebuilds.begin_validation(
        estimate.generation_id,
        owner="worker",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
    await rebuilds.validate_generation(estimate.generation_id)
    sessions = session_factory(engine)

    # A #187 pointer swap after staging must win the CAS without publishing relational rows.
    async with sessions.begin() as session:
        index_state = await session.get(models.IndexState, 1)
        index_state_was_missing = index_state is None
        expected_vector_table = None if index_state is None else index_state.vector_table
        expected_inventory = None if index_state is None else index_state.vector_inventory_digest
        if index_state is None:
            session.add(
                models.IndexState(
                    id=1,
                    vector_table="reembed-interleaving-winner",
                    vector_inventory_digest="interleaving-inventory",
                )
            )
        else:
            index_state.vector_table = "reembed-interleaving-winner"
            index_state.vector_inventory_digest = "interleaving-inventory"
    with pytest.raises(RebuildLeaseConflictError, match="live vector publication changed"):
        await rebuilds.publish_generation(
            estimate.generation_id,
            owner="worker",
            lease_generation=claimed.lease_generation,
            now=NOW,
        )
    async with sessions.begin() as session:
        index_state = await session.get(models.IndexState, 1)
        assert index_state is not None
        if index_state_was_missing:
            await session.delete(index_state)
        else:
            index_state.vector_table = expected_vector_table
            index_state.vector_inventory_digest = expected_inventory

    # Validation is not a trust boundary by itself: publication rechecks the canonical
    # promoted manifest inside the same transaction as the relational flip.
    async with sessions.begin() as session:
        record = (
            await session.execute(
                select(models.AcquisitionRecord).where(models.AcquisitionRecord.run_id == run_id)
            )
        ).scalar_one()
        original_source_record = cast("dict[str, JsonValue]", record.source_record).copy()
        record.source_record = cast(
            "JsonValue", {**original_source_record, "uri": "https://tampered.invalid"}
        )
    with pytest.raises(RuntimeError, match="snapshot_changed"):
        await rebuilds.publish_generation(
            estimate.generation_id,
            owner="worker",
            lease_generation=claimed.lease_generation,
            now=NOW,
        )
    async with sessions.begin() as session:
        record = (
            await session.execute(
                select(models.AcquisitionRecord).where(models.AcquisitionRecord.run_id == run_id)
            )
        ).scalar_one()
        record.source_record = cast("JsonValue", original_source_record)

    newer_plan = await rebuilds.plan_rebuild(
        run_id,
        target.model_copy(update={"parser_routing": "routing-v3"}),
        missing_limit=10,
    )
    newer_claim = await rebuilds.claim_generation(
        newer_plan.generation_id,
        "new-worker",
        now=NOW,
        expires_at=NOW + timedelta(minutes=5),
    )
    with pytest.raises(RebuildLeaseConflictError, match="newer rebuild"):
        await rebuilds.publish_generation(
            estimate.generation_id,
            owner="worker",
            lease_generation=claimed.lease_generation,
            now=NOW,
        )
    await rebuilds.cancel_generation(
        newer_plan.generation_id,
        owner="new-worker",
        lease_generation=newer_claim.lease_generation,
        now=NOW,
    )
    published = await rebuilds.publish_generation(
        estimate.generation_id,
        owner="worker",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
    assert published.state is RebuildState.PUBLISHED
    current = await store.get_document(old.id)
    assert current is not None
    assert current.publication_id == claimed.vector_publication_id
    assert (await store.document_chunks(old.id))[0].text == raw.as_text()
    async with engine.connect() as connection:
        fts_sql = (
            await connection.execute(
                text("SELECT sql FROM sqlite_master WHERE name = 'chunks_fts'")
            )
        ).scalar_one()
        inventory = (
            await connection.execute(
                select(models.IndexState.vector_inventory_digest).where(models.IndexState.id == 1)
            )
        ).scalar_one()
    assert "tokenize='unicode61'" in fts_sql
    assert inventory is not None
    assert inventory != expected_inventory

    # Idempotent retries do not create another generation or duplicate derived rows.
    repeated = await rebuilds.publish_generation(
        estimate.generation_id,
        owner="worker",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
    assert repeated == published
    assert await store.count_chunks(old.id) == 1

    async with sessions.begin() as session:
        await session.execute(
            delete(models.DerivedGenerationItem).where(
                models.DerivedGenerationItem.generation_id == estimate.generation_id
            )
        )
    with pytest.raises(RuntimeError, match="exact and contiguous"):
        await rebuilds.validate_generation(estimate.generation_id)


async def test_expired_owner_is_fenced_after_takeover(  # noqa: PLR0915 - one takeover timeline
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id, blob_ref, raw = await promoted_snapshot(store, engine, data_dir)
    embed = EmbedFingerprint(
        model_id="test/embed",
        dimension=4,
        pooling=Pooling.MEAN,
        normalized=True,
        tokenizer_id="test/tokenizer",
        max_sequence_length=128,
    )
    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(embed)
    rebuilds = SqliteRebuildStore(
        engine,
        workspace_id=store.workspace_id,
        blobs=BlobStore(engine, data_dir),
        vectors=vectors,
    )
    plan = await rebuilds.plan_rebuild(
        run_id,
        RebuildTarget(
            parser_routing="routing-v2",
            parser_set=("plain@2",),
            chunk_fingerprint="chunk-v2",
            embedding_fingerprint=embed.canonical(),
            glossary_fingerprint="glossary-v2",
            fts_tokenizer="unicode61",
            batch_documents=1,
            max_memory_bytes=1_000_000,
            max_temporary_bytes=1_000_000,
        ),
        missing_limit=10,
    )
    first = await rebuilds.claim_generation(
        plan.generation_id, "first", now=NOW, expires_at=NOW + timedelta(seconds=1)
    )
    document = make_document(
        source="wiki",
        source_id=raw.source_id,
        body=raw.as_bytes(),
        uri=raw.uri,
        media_type=raw.media_type,
    ).model_copy(
        update={
            "publication_id": plan.generation_id,
            "original_ref": blob_ref,
            "version_token": "v2",
            "status": DocumentStatus.INDEXED,
        }
    )
    staged_chunk = make_chunk(document, 0, raw.as_text())
    replacement = DerivedReplacement(
        document=document,
        chunks=(staged_chunk,),
        vector_embedded=1,
    )
    await vectors.upsert(
        [staged_chunk], [[1.0, 0.0, 0.0, 0.0]], publication_id=first.vector_publication_id
    )
    await rebuilds.stage_replacements(
        plan.generation_id,
        [(0, replacement)],
        expected_next_sequence=0,
        owner="first",
        lease_generation=first.lease_generation,
        now=NOW,
    )
    takeover_now = datetime.now(UTC)
    second = await rebuilds.claim_generation(
        plan.generation_id,
        "second",
        now=takeover_now,
        expires_at=takeover_now + timedelta(seconds=1),
    )
    assert second.lease_generation == first.lease_generation + 1
    assert second.fence_generation == first.fence_generation
    assert second.vector_publication_id != first.vector_publication_id
    assert second.predecessor_vector_publication_id == first.vector_publication_id
    original_copy = vectors.copy_publication

    async def certified_publication() -> str | None:
        sessions = session_factory(engine)
        async with sessions() as session:
            return (
                await session.execute(
                    select(models.DerivedGeneration.vector_publication_id).where(
                        models.DerivedGeneration.id == plan.generation_id
                    )
                )
            ).scalar_one()

    async def crash_after_page(*args: object, **kwargs: object) -> None:
        await original_copy(*args, **kwargs)  # pyright: ignore[reportArgumentType]
        raise RuntimeError("worker crashed after the first replay page")

    monkeypatch.setattr(vectors, "copy_publication", crash_after_page)
    with pytest.raises(RuntimeError, match="crashed after the first replay page"):
        await rebuilds.copy_checkpointed_vectors(
            plan.generation_id,
            first.vector_publication_id,
            owner="second",
            lease_generation=second.lease_generation,
            now=takeover_now,
        )
    assert await certified_publication() == first.vector_publication_id
    third = await rebuilds.claim_generation(
        plan.generation_id,
        "third",
        now=takeover_now + timedelta(seconds=2),
        expires_at=takeover_now + timedelta(seconds=3),
    )
    assert third.predecessor_vector_publication_id == first.vector_publication_id

    event = asyncio.Event()

    async def cancel_after_page(*args: object, **kwargs: object) -> None:
        await original_copy(*args, **kwargs)  # pyright: ignore[reportArgumentType]
        event.set()

    monkeypatch.setattr(vectors, "copy_publication", cancel_after_page)
    with pytest.raises(asyncio.CancelledError):
        await rebuilds.copy_checkpointed_vectors(
            plan.generation_id,
            first.vector_publication_id,
            owner="third",
            lease_generation=third.lease_generation,
            now=takeover_now + timedelta(seconds=2),
            cancel=event,
        )
    assert await certified_publication() == first.vector_publication_id

    fourth = await rebuilds.claim_generation(
        plan.generation_id,
        "fourth",
        now=takeover_now + timedelta(seconds=4),
        expires_at=takeover_now + timedelta(seconds=5),
    )
    assert fourth.predecessor_vector_publication_id == first.vector_publication_id

    started = asyncio.Event()

    async def wait_after_page(*args: object, **kwargs: object) -> None:
        await original_copy(*args, **kwargs)  # pyright: ignore[reportArgumentType]
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(vectors, "copy_publication", wait_after_page)
    copying = asyncio.create_task(
        rebuilds.copy_checkpointed_vectors(
            plan.generation_id,
            first.vector_publication_id,
            owner="fourth",
            lease_generation=fourth.lease_generation,
            now=takeover_now + timedelta(seconds=4),
        )
    )
    await started.wait()
    copying.cancel()
    with pytest.raises(asyncio.CancelledError):
        await copying
    assert await certified_publication() == first.vector_publication_id

    fifth = await rebuilds.claim_generation(
        plan.generation_id,
        "fifth",
        now=takeover_now + timedelta(seconds=6),
        expires_at=takeover_now + timedelta(minutes=2),
    )
    assert fifth.predecessor_vector_publication_id == first.vector_publication_id
    monkeypatch.setattr(vectors, "copy_publication", original_copy)
    await rebuilds.copy_checkpointed_vectors(
        plan.generation_id,
        first.vector_publication_id,
        owner="fifth",
        lease_generation=fifth.lease_generation,
        now=takeover_now + timedelta(seconds=6),
    )
    assert await vectors.publication_is_complete(
        fifth.vector_publication_id,
        [staged_chunk],
        embedding_fingerprint=embed.canonical(),
    )
    assert await certified_publication() == fifth.vector_publication_id
    with pytest.raises(RebuildLeaseConflictError, match="lease changed"):
        await rebuilds.renew_generation(
            plan.generation_id,
            "first",
            first.lease_generation,
            now=takeover_now + timedelta(seconds=6),
            expires_at=takeover_now + timedelta(minutes=3),
        )
