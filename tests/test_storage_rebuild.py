from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast, override

import pytest
from pydantic import JsonValue
from sqlalchemy import delete, event, select, text, update

from manicule.core.acquisition import (
    AcquiredSource,
    AcquisitionDiagnostic,
    AcquisitionFailureCode,
    AcquisitionRecordState,
    AcquisitionRunState,
    AcquisitionSource,
    AcquisitionStage,
    SnapshotCompleteness,
    SnapshotPromotionPolicy,
)
from manicule.core.anchors import Unlocated
from manicule.core.content import Chunk, Document, DocumentStatus, RawDocument
from manicule.core.embedding import EmbedFingerprint, Pooling
from manicule.core.fingerprints import ChunkFingerprint
from manicule.core.glossary import DefinitionForm, GlossaryEntry
from manicule.core.ids import chunk_id, content_hash
from manicule.core.rebuild import (
    DerivedReplacement,
    RebuildCheckpoint,
    RebuildPublicationConflictError,
    RebuildPublicationValidationError,
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
from manicule.storage.vectors import LanceVectorStore, VectorStoreStateError
from tests.storage_helpers import make_chunk, make_document

if TYPE_CHECKING:
    from collections.abc import Sequence
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


class FailingGlossaryPublicationStore(SqliteRebuildStore):
    """Fail after a later evidence page has added its relational replacement."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.published_items = 0

    @override
    async def _publish_item(self, *args: object, **kwargs: object) -> str:
        result = await super()._publish_item(*args, **kwargs)  # pyright: ignore[reportArgumentType]
        self.published_items += 1
        if self.published_items == 2:
            raise RuntimeError("synthetic glossary publication failure")
        return result


class FailingSettlementPublicationStore(SqliteRebuildStore):
    """Inject a failure after journal settlement but before the shared commit."""

    @override
    async def _settle_published_generation(
        self,
        session: AsyncSession,
        generation: models.DerivedGeneration,
        *,
        now: datetime,
    ) -> None:
        await super()._settle_published_generation(session, generation, now=now)
        raise RuntimeError("synthetic settlement commit failure")


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
        uri=f"https://wiki.example.test/content/{source_id}",
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


async def promoted_snapshot_many(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    raws: tuple[RawDocument, ...],
) -> tuple[str, tuple[str, ...]]:
    """Promote several retained inputs so publication must cross evidence pages."""
    run = await store.create_acquisition_run(
        "promoted-glossary-run",
        "wiki",
        source_scope="scope:glossary-v1",
        scope_fingerprint="glossary-v1",
    )
    claimed = await store.claim_acquisition_run(
        run.id, "worker", now=NOW, expires_at=NOW + timedelta(minutes=5)
    )
    assert claimed is not None
    for sequence, raw in enumerate(raws):
        source = AcquisitionSource.from_discovered(
            DiscoveredDoc(
                ref=DocRef(source_id=raw.source_id, uri=raw.uri),
                version_token="v2",  # noqa: S106 - source revision, not a credential
                title="Synthetic glossary",
                media_type=raw.media_type,
                size_bytes=len(raw.as_bytes()),
            )
        )
        await store.append_acquisition_record(
            run.id,
            sequence,
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
    blobs = BlobStore(engine, data_dir)
    blob_refs: list[str] = []
    for raw in raws:
        blob = await blobs.put(raw.as_bytes(), raw.media_type)
        assert isinstance(blob, StoredBlob)
        blob_refs.append(blob.hash)
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
        expected_scope_fingerprint="glossary-v1",
        lease_owner="worker",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
    return run.id, tuple(blob_refs)


async def promoted_partial_snapshot(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> tuple[str, tuple[RawDocument, RawDocument], tuple[str, str]]:
    """Promote retained members on either side of one typed omission."""
    run = await store.create_acquisition_run(
        "promoted-partial-run",
        "wiki",
        source_scope="scope:partial-v1",
        scope_fingerprint="partial-v1",
        promotion_policy=SnapshotPromotionPolicy.ALLOW_OMISSIONS,
    )
    claimed = await store.claim_acquisition_run(
        run.id, "partial-worker", now=NOW, expires_at=NOW + timedelta(minutes=5)
    )
    assert claimed is not None
    raws = (
        RawDocument(
            source_id="retained-zero",
            uri="https://wiki.example.test/content/retained-zero",
            media_type="text/plain",
            content="retained zero",
        ),
        RawDocument(
            source_id="retained-two",
            uri="https://wiki.example.test/content/retained-two",
            media_type="text/plain",
            content="retained two",
        ),
    )
    source_ids = (raws[0].source_id, "omitted-one", raws[1].source_id)
    for sequence, source_id in enumerate(source_ids):
        raw = raws[0] if sequence == 0 else raws[1] if sequence == 2 else None
        source = AcquisitionSource.from_discovered(
            DiscoveredDoc(
                ref=DocRef(
                    source_id=source_id,
                    uri=f"https://wiki.example.test/content/{source_id}",
                ),
                version_token="v2",  # noqa: S106 - source revision, not a credential
                media_type="text/plain",
                size_bytes=0 if raw is None else len(raw.as_bytes()),
            )
        )
        await store.append_acquisition_record(
            run.id,
            sequence,
            source,
            lease_owner="partial-worker",
            lease_generation=claimed.lease_generation,
            now=NOW,
        )
    await store.complete_acquisition_enumeration(
        run.id,
        Watermark(value="partial-v2", observed_at=NOW),
        lease_owner="partial-worker",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
    blobs = BlobStore(engine, data_dir)
    refs: list[str] = []
    for raw in raws:
        blob = await blobs.put(raw.as_bytes(), raw.media_type)
        assert isinstance(blob, StoredBlob)
        refs.append(blob.hash)
        await store.transition_acquisition_record(
            run.id,
            raw.source_id,
            AcquisitionRecordState.DISCOVERED,
            AcquisitionRecordState.ACQUIRING,
            lease_owner="partial-worker",
            lease_generation=claimed.lease_generation,
            now=NOW,
        )
        await store.transition_acquisition_record(
            run.id,
            raw.source_id,
            AcquisitionRecordState.ACQUIRING,
            AcquisitionRecordState.ACQUIRED,
            lease_owner="partial-worker",
            lease_generation=claimed.lease_generation,
            now=NOW,
            blob_ref=blob.hash,
            acquired_source=AcquiredSource.from_raw(raw),
            fetched_version_token="v2",  # noqa: S106 - source revision, not a credential
        )
    await store.transition_acquisition_record(
        run.id,
        "omitted-one",
        AcquisitionRecordState.DISCOVERED,
        AcquisitionRecordState.RETRY,
        lease_owner="partial-worker",
        lease_generation=claimed.lease_generation,
        now=NOW,
        diagnostic=AcquisitionDiagnostic(
            stage=AcquisitionStage.ACQUISITION,
            code=AcquisitionFailureCode.AUTHENTICATION,
        ),
    )
    completed = await store.complete_snapshot_acquisition(
        run.id,
        lease_owner="partial-worker",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
    assert completed.omission_count == 1
    promoted = await store.promote_snapshot_and_commit_watermark(
        run.id,
        expected_scope_fingerprint="partial-v1",
        lease_owner="partial-worker",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
    assert promoted.completeness is SnapshotCompleteness.PARTIAL
    return run.id, raws, (refs[0], refs[1])


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


def rebuild_target() -> tuple[RebuildTarget, EmbedFingerprint]:
    """One stable same-target commitment for publication/reset regressions."""
    embed = EmbedFingerprint(
        model_id="test/embed",
        dimension=4,
        pooling=Pooling.MEAN,
        normalized=True,
        tokenizer_id="test/tokenizer",
        max_sequence_length=128,
    )
    return (
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
        embed,
    )


async def staged_glossary_generation(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    *,
    rebuild_type: type[SqliteRebuildStore] = SqliteRebuildStore,
) -> tuple[SqliteRebuildStore, RebuildCheckpoint, tuple[Document, ...]]:
    """Stage two aliased replacements under a real FK-on SQLite generation."""
    raws = (
        RawDocument(
            source_id="glossary-one",
            uri="https://wiki.example.test/glossary-one",
            media_type="text/plain",
            content="Network Operations Workspace (NOW, NETOPS)",
        ),
        RawDocument(
            source_id="glossary-two",
            uri="https://wiki.example.test/glossary-two",
            media_type="text/plain",
            content="Network Operations Service (NOS, NETOPS)",
        ),
    )
    run_id, blob_refs = await promoted_snapshot_many(store, engine, data_dir, raws)
    target, embed = rebuild_target()
    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(embed)
    rebuilds = rebuild_type(
        engine,
        workspace_id=store.workspace_id,
        blobs=BlobStore(engine, data_dir),
        vectors=vectors,
    )
    plan = await rebuilds.plan_rebuild(run_id, target, missing_limit=10)
    claimed = await rebuilds.claim_generation(
        plan.generation_id,
        "glossary-publisher",
        now=NOW,
        expires_at=NOW + timedelta(minutes=5),
    )
    replacements: list[tuple[int, DerivedReplacement]] = []
    documents: list[Document] = []
    for sequence, (raw, blob_ref) in enumerate(zip(raws, blob_refs, strict=True)):
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
        chunk = make_chunk(document, 0, raw.as_text())
        acronym = "NOW" if sequence == 0 else "NOS"
        entries = [
            GlossaryEntry(
                acronym=acronym,
                display=acronym,
                expansion=(
                    "Network Operations Workspace"
                    if sequence == 0
                    else "Network Operations Service"
                ),
                document_id=document.id,
                chunk_id=chunk.id,
                form=DefinitionForm.PARENTHETICAL,
                confidence=0.9,
                aliases=("NETOPS",),
            )
        ]
        if sequence == 0:
            entries.append(
                GlossaryEntry(
                    acronym="SOC",
                    display="SOC",
                    expansion="Security Operations Center",
                    document_id=document.id,
                    chunk_id=chunk.id,
                    form=DefinitionForm.PARENTHETICAL,
                    confidence=0.9,
                    aliases=("SECOPS",),
                )
            )
        replacement = DerivedReplacement(
            document=document,
            chunks=(chunk,),
            glossary=tuple(entries),
            parse_fingerprint="plain@2",
            vector_embedded=1,
        )
        replacements.append((sequence, replacement))
        documents.append(document)
        await vectors.upsert(
            [chunk],
            [[1.0, 0.0, 0.0, 0.0]],
            publication_id=claimed.vector_publication_id,
        )
    checkpoint = await rebuilds.stage_replacements(
        plan.generation_id,
        replacements,
        expected_next_sequence=0,
        owner="glossary-publisher",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
    await rebuilds.begin_validation(
        plan.generation_id,
        owner="glossary-publisher",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
    await rebuilds.validate_generation(plan.generation_id)
    return rebuilds, checkpoint, tuple(documents)


async def test_atomic_publication_flushes_glossary_parents_before_alias_evidence_pages(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    rebuilds, claimed, documents = await staged_glossary_generation(store, engine, data_dir)

    published = await rebuilds.publish_generation(
        claimed.generation_id,
        owner="glossary-publisher",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )

    assert published.state is RebuildState.PUBLISHED
    assert [len(await store.glossary_entries(document.id)) for document in documents] == [2, 1]
    async with engine.connect() as connection:
        assert (await connection.execute(text("PRAGMA foreign_key_check"))).all() == []
        aliases = (
            (
                await connection.execute(
                    select(models.GlossaryAlias.key).order_by(models.GlossaryAlias.entry_id)
                )
            )
            .scalars()
            .all()
        )
    assert sorted(aliases) == ["NETOPS", "NETOPS", "SECOPS"], (
        "the repeated alias crosses the one-row evidence pages without orphaning either parent"
    )


async def test_glossary_publication_failure_rolls_back_every_relational_live_row(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    rebuilds, claimed, documents = await staged_glossary_generation(
        store, engine, data_dir, rebuild_type=FailingGlossaryPublicationStore
    )

    with pytest.raises(RuntimeError, match="synthetic glossary publication failure"):
        await rebuilds.publish_generation(
            claimed.generation_id,
            owner="glossary-publisher",
            lease_generation=claimed.lease_generation,
            now=NOW,
        )

    assert [await store.get_document(document.id) for document in documents] == [None, None]
    async with engine.connect() as connection:
        assert (await connection.execute(select(models.GlossaryEntry.id))).scalars().all() == []
        assert (await connection.execute(select(models.GlossaryAlias.key))).scalars().all() == []
        assert (await connection.execute(text("PRAGMA foreign_key_check"))).all() == []
        assert (
            await connection.execute(select(models.IndexState.id).where(models.IndexState.id == 1))
        ).scalar_one_or_none() is None
    assert (await rebuilds.checkpoint(claimed.generation_id)).state is RebuildState.VALIDATING


async def test_publication_and_acquisition_settlement_share_one_rollback_boundary(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    rebuilds, claimed, documents = await staged_glossary_generation(
        store, engine, data_dir, rebuild_type=FailingSettlementPublicationStore
    )

    with pytest.raises(RuntimeError, match="synthetic settlement commit failure"):
        await rebuilds.publish_generation(
            claimed.generation_id,
            owner="glossary-publisher",
            lease_generation=claimed.lease_generation,
            now=NOW,
        )

    async with engine.connect() as connection:
        generation_state = await connection.scalar(
            select(models.DerivedGeneration.state).where(
                models.DerivedGeneration.id == claimed.generation_id
            )
        )
        snapshot_id = await connection.scalar(
            select(models.DerivedGeneration.snapshot_run_id).where(
                models.DerivedGeneration.id == claimed.generation_id
            )
        )
        run_state = await connection.scalar(
            select(models.AcquisitionRun.state).where(models.AcquisitionRun.id == snapshot_id)
        )
        record_states = (
            (
                await connection.execute(
                    select(models.AcquisitionRecord.state).where(
                        models.AcquisitionRecord.run_id == snapshot_id
                    )
                )
            )
            .scalars()
            .all()
        )
    assert generation_state is RebuildState.VALIDATING
    assert run_state is AcquisitionRunState.ACQUIRING
    assert set(record_states) == {AcquisitionRecordState.ACQUIRED}
    assert [await store.get_document(document.id) for document in documents] == [None, None]


async def test_allowed_partial_publication_derives_only_evidence_and_keeps_omission_pending(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    run_id, raws, blob_refs = await promoted_partial_snapshot(store, engine, data_dir)
    target, embed = rebuild_target()
    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(embed)
    rebuilds = SqliteRebuildStore(
        engine,
        workspace_id=store.workspace_id,
        blobs=BlobStore(engine, data_dir),
        vectors=vectors,
    )
    plan = await rebuilds.plan_rebuild(run_id, target, missing_limit=10)
    assert plan.runnable
    assert plan.documents == plan.expected_items == 2
    assert plan.missing_count == 0
    batch = await rebuilds.snapshot_inputs(plan.generation_id, after_sequence=-1, limit=10)
    assert [(item.sequence, item.source.source_id) for item in batch] == [
        (0, "retained-zero"),
        (1, "retained-two"),
    ]
    first = await rebuilds.snapshot_inputs(plan.generation_id, after_sequence=-1, limit=1)
    assert [(item.sequence, item.source.source_id) for item in first] == [(0, "retained-zero")]

    claimed = await rebuilds.claim_generation(
        plan.generation_id,
        "partial-publisher",
        now=NOW,
        expires_at=NOW + timedelta(minutes=5),
    )
    replacements: list[tuple[int, DerivedReplacement]] = []
    for sequence, (raw, blob_ref) in enumerate(zip(raws, blob_refs, strict=True)):
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
        chunk = make_chunk(document, 0, raw.as_text())
        replacements.append(
            (
                sequence,
                DerivedReplacement(
                    document=document,
                    chunks=(chunk,),
                    parse_fingerprint="plain@2",
                    vector_embedded=1,
                ),
            )
        )
        await vectors.upsert(
            [chunk],
            [[1.0, 0.0, 0.0, 0.0]],
            publication_id=claimed.vector_publication_id,
        )
    staged = await rebuilds.stage_replacements(
        plan.generation_id,
        replacements,
        expected_next_sequence=0,
        owner="partial-publisher",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
    second = await rebuilds.snapshot_inputs(plan.generation_id, after_sequence=0, limit=1)
    assert [(item.sequence, item.source.source_id) for item in second] == [(1, "retained-two")]
    await rebuilds.begin_validation(
        plan.generation_id,
        owner="partial-publisher",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
    await rebuilds.validate_generation(plan.generation_id)
    published = await rebuilds.publish_generation(
        plan.generation_id,
        owner="partial-publisher",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )

    assert staged.documents_built == 2
    assert published.state is RebuildState.PUBLISHED
    persisted = await store.get_acquisition_run(run_id)
    records = await store.list_acquisition_records(run_id)
    assert persisted is not None
    assert persisted.state is AcquisitionRunState.SETTLED
    assert persisted.omission_count == 1
    assert [record.state for record in records] == [
        AcquisitionRecordState.SETTLED,
        AcquisitionRecordState.OMITTED,
        AcquisitionRecordState.SETTLED,
    ]
    metadata = await store.connector_metadata("wiki")
    last_run = cast("dict[str, Any]", metadata["last_run"])
    lifecycle = cast("dict[str, Any]", last_run["lifecycle"])
    assert last_run["outcome"] == "incomplete"
    assert last_run["retry_required"] is True
    assert lifecycle["pending_items"] == lifecycle["backlog_items"] == 1
    assert lifecycle["omitted_items"] == 1
    assert lifecycle["snapshot_completeness"] == "partial"
    assert lifecycle["reproducibility_policy"] == "allow_omissions"
    assert lifecycle["can_continue_offline"] is False
    assert await store.verify_snapshot_manifest(run_id)
    assert await store.latest_unsettled_acquisition_run("wiki") is None
    rendered = json.dumps(metadata, sort_keys=True)
    assert "retained zero" not in rendered
    assert "wiki.example.test" not in rendered


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
async def test_publication_rechecks_retained_blob_integrity_before_settlement(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    damage: str,
) -> None:
    rebuilds, claimed, _ = await staged_glossary_generation(store, engine, data_dir)
    async with engine.connect() as connection:
        snapshot_id = await connection.scalar(
            select(models.DerivedGeneration.snapshot_run_id).where(
                models.DerivedGeneration.id == claimed.generation_id
            )
        )
        blob_ref = await connection.scalar(
            select(models.AcquisitionRecord.blob_ref)
            .where(models.AcquisitionRecord.run_id == snapshot_id)
            .order_by(models.AcquisitionRecord.sequence)
            .limit(1)
        )
    assert isinstance(snapshot_id, str)
    assert isinstance(blob_ref, str)
    path = BlobStore(engine, data_dir).path_for(blob_ref)
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"synthetic corrupt evidence")

    with pytest.raises(RebuildPublicationConflictError) as caught:
        await rebuilds.publish_generation(
            claimed.generation_id,
            owner="glossary-publisher",
            lease_generation=claimed.lease_generation,
            now=NOW,
        )

    assert caught.value.code is RebuildRefusalCode.SNAPSHOT_CHANGED
    assert (await rebuilds.checkpoint(claimed.generation_id)).state is RebuildState.VALIDATING
    persisted = await store.get_acquisition_run(snapshot_id)
    assert persisted is not None
    assert persisted.state is AcquisitionRunState.ACQUIRING
    assert {record.state for record in await store.list_acquisition_records(snapshot_id)} == {
        AcquisitionRecordState.ACQUIRED
    }


async def test_published_replay_repairs_a_legacy_unsettled_handoff_exactly_once(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    rebuilds, claimed, _ = await staged_glossary_generation(store, engine, data_dir)
    published = await rebuilds.publish_generation(
        claimed.generation_id,
        owner="glossary-publisher",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
    sessions = session_factory(engine)
    async with sessions.begin() as session:
        generation = await session.get(models.DerivedGeneration, claimed.generation_id)
        assert generation is not None
        run = await session.get(models.AcquisitionRun, generation.snapshot_run_id)
        assert run is not None
        run.state = AcquisitionRunState.INDEXING
        run.indexed_count = 0
        run.acquired_blob_bytes = 1
        await session.execute(
            update(models.AcquisitionRecord)
            .where(models.AcquisitionRecord.run_id == run.id)
            .values(state=AcquisitionRecordState.INDEXING)
        )
        snapshot_id = run.id

    replayed = await rebuilds.claim_generation(
        claimed.generation_id,
        "stale-response-owner",
        now=NOW + timedelta(seconds=1),
        expires_at=NOW + timedelta(minutes=1),
    )
    repaired = await store.get_acquisition_run(snapshot_id)
    records = await store.list_acquisition_records(snapshot_id)

    assert replayed == published
    assert repaired is not None
    assert repaired.state is AcquisitionRunState.SETTLED
    assert repaired.indexed_count == repaired.discovered_count
    assert repaired.retry_count == 0
    assert repaired.acquired_blob_bytes == 0
    assert {record.state for record in records} == {AcquisitionRecordState.SETTLED}
    assert await store.verify_snapshot_manifest(snapshot_id)


@pytest.mark.asyncio
async def test_derived_reset_turns_same_target_published_replay_into_actionable_plan(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    run_id, blob_ref, raw = await promoted_snapshot(store, engine, data_dir)
    document = make_document(
        source="wiki",
        source_id=raw.source_id,
        body=b"old text",
        uri=raw.uri,
        media_type=raw.media_type,
    ).model_copy(update={"original_ref": blob_ref})
    await store.upsert_document(document)
    await store.replace_chunks(document.id, [make_chunk(document, 0, "old text")])
    target, embed = rebuild_target()
    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(embed)
    rebuilds = SqliteRebuildStore(
        engine,
        workspace_id=store.workspace_id,
        blobs=BlobStore(engine, data_dir),
        vectors=vectors,
    )
    planned = await rebuilds.plan_rebuild(run_id, target, missing_limit=10)
    published = await publish_one_replacement(
        rebuilds,
        vectors,
        estimate_id=planned.generation_id,
        target=target,
        document=document,
        raw=raw,
        blob_ref=blob_ref,
        owner="publisher",
    )
    replay = await rebuilds.plan_rebuild(run_id, target, missing_limit=10)
    assert replay.generation_id == published.generation_id
    assert (await rebuilds.checkpoint(replay.generation_id)).state is RebuildState.PUBLISHED

    await store.reset_derived()
    async for page in store.obsolete_generation_publications():
        for generation in page:
            await store.cleanup_obsolete_generation(generation.generation_id)

    actionable = await rebuilds.plan_rebuild(run_id, target, missing_limit=10)
    assert actionable.generation_id == published.generation_id
    assert (await rebuilds.checkpoint(actionable.generation_id)).state is RebuildState.PLANNED


async def test_live_vector_swap_gets_a_new_plan_and_published_replay_is_idempotent(  # noqa: PLR0915
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
    record_selects = 0

    def count_record_selects(*args: object) -> None:
        nonlocal record_selects
        statement = args[2]
        if (
            isinstance(statement, str)
            and statement.lstrip().upper().startswith("SELECT")
            and "FROM acquisition_records" in statement
        ):
            record_selects += 1

    event.listen(engine.sync_engine, "before_cursor_execute", count_record_selects)
    try:
        stale = await rebuilds.plan_rebuild(run_id, target, missing_limit=10)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", count_record_selects)
    assert record_selects == 2, "one manifest verification and one bounded planning cursor"
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
    with pytest.raises(RebuildPublicationConflictError) as caught:
        await rebuilds.publish_generation(
            first.generation_id,
            owner="first-worker",
            lease_generation=claimed.lease_generation,
            now=NOW,
        )
    assert caught.value.code is RebuildRefusalCode.WORKSPACE_SCOPE_CHANGED


async def test_resume_refuses_a_snapshot_that_lost_promotion_with_a_bounded_conflict(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    run_id, _, _ = await promoted_snapshot(store, engine, data_dir)
    target, embed = rebuild_target()
    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(embed)
    rebuilds = SqliteRebuildStore(
        engine,
        workspace_id=store.workspace_id,
        blobs=BlobStore(engine, data_dir),
        vectors=vectors,
    )
    plan = await rebuilds.plan_rebuild(run_id, target, missing_limit=10)
    await rebuilds.claim_generation(
        plan.generation_id,
        "resume-worker",
        now=NOW,
        expires_at=NOW + timedelta(minutes=5),
    )
    async with session_factory(engine).begin() as session:
        await session.execute(
            update(models.AcquisitionRun)
            .where(models.AcquisitionRun.id == run_id)
            .values(promoted_at=None, watermark_committed_at=None)
        )

    with pytest.raises(RebuildPublicationConflictError) as caught:
        await rebuilds.snapshot_inputs(plan.generation_id, after_sequence=-1, limit=1)
    assert caught.value.code is RebuildRefusalCode.SNAPSHOT_CHANGED


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
    with pytest.raises(RebuildPublicationConflictError) as moved:
        await rebuilds.stage_replacements(
            estimate.generation_id,
            [(0, replacement)],
            expected_next_sequence=0,
            owner="worker",
            lease_generation=claimed.lease_generation,
            now=NOW,
        )
    assert moved.value.code is RebuildRefusalCode.PUBLICATION_CONFLICT

    sessions = session_factory(engine)
    async with sessions.begin() as session:
        await session.execute(
            update(models.DerivedGeneration)
            .where(models.DerivedGeneration.id == estimate.generation_id)
            .values(next_sequence=0)
        )
    changed = replacement.model_copy(
        update={"document": replacement.document.model_copy(update={"title": "changed retry"})}
    )
    with pytest.raises(RebuildPublicationValidationError) as invalid:
        await rebuilds.stage_replacements(
            estimate.generation_id,
            [(0, changed)],
            expected_next_sequence=0,
            owner="worker",
            lease_generation=claimed.lease_generation,
            now=NOW,
        )
    assert invalid.value.code is RebuildRefusalCode.INVALID_REPLACEMENT
    async with sessions.begin() as session:
        await session.execute(
            update(models.DerivedGeneration)
            .where(models.DerivedGeneration.id == estimate.generation_id)
            .values(next_sequence=1)
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
    with pytest.raises(RebuildPublicationConflictError) as caught:
        await rebuilds.publish_generation(
            estimate.generation_id,
            owner="worker",
            lease_generation=claimed.lease_generation,
            now=NOW,
        )
    assert caught.value.code is RebuildRefusalCode.PUBLICATION_CONFLICT
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
    with pytest.raises(RebuildPublicationConflictError) as caught:
        await rebuilds.publish_generation(
            estimate.generation_id,
            owner="worker",
            lease_generation=claimed.lease_generation,
            now=NOW,
        )
    assert caught.value.code is RebuildRefusalCode.SNAPSHOT_CHANGED
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
    with pytest.raises(RebuildPublicationConflictError) as caught:
        await rebuilds.publish_generation(
            estimate.generation_id,
            owner="worker",
            lease_generation=claimed.lease_generation,
            now=NOW,
        )
    assert caught.value.code is RebuildRefusalCode.PUBLICATION_CONFLICT
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
    with pytest.raises(RebuildPublicationValidationError) as caught:
        await rebuilds.validate_generation(estimate.generation_id)
    assert caught.value.code is RebuildRefusalCode.INVALID_REPLACEMENT


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
    original_page_complete = vectors.publication_page_is_complete
    original_row_count = vectors.publication_row_count

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

    async def corrupt_source_page(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise VectorStoreStateError("private vector path /private/vectors")

    monkeypatch.setattr(vectors, "copy_publication", corrupt_source_page)
    with pytest.raises(RebuildPublicationValidationError) as invalid:
        await rebuilds.copy_checkpointed_vectors(
            plan.generation_id,
            first.vector_publication_id,
            owner="second",
            lease_generation=second.lease_generation,
            now=takeover_now,
        )
    assert invalid.value.code is RebuildRefusalCode.INVALID_REPLACEMENT
    assert "/private" not in str(invalid.value)
    assert await certified_publication() == first.vector_publication_id
    monkeypatch.setattr(vectors, "copy_publication", original_copy)

    async def corrupt_replay_page(
        publication_id: str,
        chunks: Sequence[Chunk],
        *,
        embedding_fingerprint: str,
    ) -> bool:
        if publication_id == second.vector_publication_id:
            return False
        return await original_page_complete(
            publication_id,
            chunks,
            embedding_fingerprint=embedding_fingerprint,
        )

    monkeypatch.setattr(vectors, "publication_page_is_complete", corrupt_replay_page)
    with pytest.raises(RebuildPublicationValidationError) as invalid:
        await rebuilds.copy_checkpointed_vectors(
            plan.generation_id,
            first.vector_publication_id,
            owner="second",
            lease_generation=second.lease_generation,
            now=takeover_now,
        )
    assert invalid.value.code is RebuildRefusalCode.INVALID_REPLACEMENT
    assert await certified_publication() == first.vector_publication_id

    async def unavailable_replay_page(
        publication_id: str,
        chunks: Sequence[Chunk],
        *,
        embedding_fingerprint: str,
    ) -> bool:
        if publication_id == second.vector_publication_id:
            raise ValueError("private page bound /private/vectors")
        return await original_page_complete(
            publication_id,
            chunks,
            embedding_fingerprint=embedding_fingerprint,
        )

    monkeypatch.setattr(vectors, "publication_page_is_complete", unavailable_replay_page)
    with pytest.raises(RebuildPublicationValidationError) as invalid:
        await rebuilds.copy_checkpointed_vectors(
            plan.generation_id,
            first.vector_publication_id,
            owner="second",
            lease_generation=second.lease_generation,
            now=takeover_now,
        )
    assert invalid.value.code is RebuildRefusalCode.INVALID_REPLACEMENT
    assert "/private" not in str(invalid.value)
    assert await certified_publication() == first.vector_publication_id

    monkeypatch.setattr(vectors, "publication_page_is_complete", original_page_complete)

    async def corrupt_replay_inventory(*args: object, **kwargs: object) -> int:
        del args, kwargs
        return 2

    monkeypatch.setattr(vectors, "publication_row_count", corrupt_replay_inventory)
    with pytest.raises(RebuildPublicationValidationError) as invalid:
        await rebuilds.copy_checkpointed_vectors(
            plan.generation_id,
            first.vector_publication_id,
            owner="second",
            lease_generation=second.lease_generation,
            now=takeover_now,
        )
    assert invalid.value.code is RebuildRefusalCode.INVALID_REPLACEMENT
    assert await certified_publication() == first.vector_publication_id

    async def unavailable_replay_inventory(*args: object, **kwargs: object) -> int:
        del args, kwargs
        raise VectorStoreStateError("private inventory /private/vectors")

    monkeypatch.setattr(vectors, "publication_row_count", unavailable_replay_inventory)
    with pytest.raises(RebuildPublicationValidationError) as invalid:
        await rebuilds.copy_checkpointed_vectors(
            plan.generation_id,
            first.vector_publication_id,
            owner="second",
            lease_generation=second.lease_generation,
            now=takeover_now,
        )
    assert invalid.value.code is RebuildRefusalCode.INVALID_REPLACEMENT
    assert "/private" not in str(invalid.value)
    assert await certified_publication() == first.vector_publication_id

    monkeypatch.setattr(vectors, "publication_row_count", original_row_count)
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
