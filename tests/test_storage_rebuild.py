from __future__ import annotations

import asyncio
import json
import os
import stat
import tracemalloc
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any, cast, override

import pytest
from pydantic import JsonValue
from sqlalchemy import delete, event, select, text, update

import manicule.storage.rebuild as rebuild_storage
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
from manicule.core.ids import chunk_id, content_hash, document_id
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
    from collections.abc import Callable, Sequence

    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

NOW = datetime(2026, 8, 15, 12, tzinfo=UTC)


def open_descriptor_count() -> int:
    return len(list(Path("/dev/fd").iterdir()))


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
        snapshot_membership_hash: str,
        target_digest: str,
        vector_table: str | None,
        vector_inventory_digest: str | None,
    ) -> models.DerivedGeneration | None:
        self.replay_observed.set()
        await self.replay_release.wait()
        return await super()._published_replay(
            session,
            snapshot_run_id=snapshot_run_id,
            snapshot_membership_hash=snapshot_membership_hash,
            target_digest=target_digest,
            vector_table=vector_table,
            vector_inventory_digest=vector_inventory_digest,
        )


class CountingEvidenceBlobStore(BlobStore):
    """Count full representation hashes separately from cheap fence probes."""

    full_verifications = 0
    cheap_probes = 0
    pin_probes = 0
    final_probes = 0

    @override
    async def evidence_identity(
        self,
        digest: str,
        *,
        size_bytes: int,
        stored_bytes: int,
        compression: str,
        verify_content: bool,
    ) -> str | None:
        if verify_content:
            self.full_verifications += 1
        else:
            self.cheap_probes += 1
        return await super().evidence_identity(
            digest,
            size_bytes=size_bytes,
            stored_bytes=stored_bytes,
            compression=compression,
            verify_content=verify_content,
        )

    @override
    def pin_evidence_representation(
        self,
        digest: str,
        pin: Path,
        size_bytes: int,
        stored_bytes: int,
        compression: str,
    ) -> str | None:
        self.pin_probes += 1
        return super().pin_evidence_representation(
            digest, pin, size_bytes, stored_bytes, compression
        )

    @override
    def validate_evidence_pin(
        self,
        digest: str,
        pin: Path,
        size_bytes: int,
        stored_bytes: int,
        compression: str,
    ) -> str | None:
        self.final_probes += 1
        return super().validate_evidence_pin(digest, pin, size_bytes, stored_bytes, compression)


class GatedEvidenceBlobStore(CountingEvidenceBlobStore):
    """Pause after one full streamed hash and before fence persistence."""

    verification_started: asyncio.Event
    verification_release: asyncio.Event
    armed = False
    blocked = False
    active_digest: str | None = None

    @override
    async def evidence_identity(
        self,
        digest: str,
        *,
        size_bytes: int,
        stored_bytes: int,
        compression: str,
        verify_content: bool,
    ) -> str | None:
        identity = await super().evidence_identity(
            digest,
            size_bytes=size_bytes,
            stored_bytes=stored_bytes,
            compression=compression,
            verify_content=verify_content,
        )
        if verify_content and self.armed and not self.blocked:
            self.blocked = True
            self.active_digest = digest
            self.verification_started.set()
            await self.verification_release.wait()
        return identity


class EarlyAliasMutationBlobStore(BlobStore):
    """Mutate a retired alias after the first member's locked final probe."""

    mutation: Callable[[str], None] | None = None
    final_probes = 0

    @override
    def validate_evidence_pin(
        self,
        digest: str,
        pin: Path,
        size_bytes: int,
        stored_bytes: int,
        compression: str,
    ) -> str | None:
        identity = super().validate_evidence_pin(digest, pin, size_bytes, stored_bytes, compression)
        self.final_probes += 1
        if self.final_probes == 1 and self.mutation is not None:
            self.mutation(digest)
            self.mutation = None
        return identity


class ArmingEvidenceVerificationStore(SqliteRebuildStore):
    """Arm a gated blob store only for the durable validation pass, not planning."""

    def __init__(self, *args: Any, blobs: GatedEvidenceBlobStore, **kwargs: Any) -> None:
        self.gated_blobs = blobs
        super().__init__(*args, blobs=blobs, **kwargs)

    @override
    async def _record_evidence_verification(self, generation_id: str) -> None:
        self.gated_blobs.armed = True
        await super()._record_evidence_verification(generation_id)


class FailingGlossaryPublicationStore(SqliteRebuildStore):
    """Fail after a later evidence page has added its relational replacement."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.published_items = 0

    @override
    async def _publish_item(self, *args: object, **kwargs: object) -> str:
        result = await super()._publish_item(*args, **kwargs)
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


class MutatingAfterEvidenceProbeStore(SqliteRebuildStore):
    """Change the public representation after pins exist but before SQLite publication."""

    mutation: Callable[[], None] | None = None

    @override
    async def _publish_generation_atomic(self, *args: Any, **kwargs: Any) -> RebuildCheckpoint:
        if self.mutation is not None:
            self.mutation()
            self.mutation = None
        return await super()._publish_generation_atomic(*args, **kwargs)


class GatedAfterEvidenceProbeStore(SqliteRebuildStore):
    """Pause after durable pins are established and before SQLite gets a writer."""

    probe_complete: asyncio.Event
    publication_release: asyncio.Event

    @override
    async def _publish_generation_atomic(self, *args: Any, **kwargs: Any) -> RebuildCheckpoint:
        self.probe_complete.set()
        await self.publication_release.wait()
        return await super()._publish_generation_atomic(*args, **kwargs)


async def promoted_snapshot(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    *,
    run_id: str = "promoted-run",
    connector: str = "wiki",
    scope_fingerprint: str = "scope-v1",
    source_id: str = "page-1",
    emits_watermark: bool = True,
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
        Watermark(value="v2", observed_at=NOW) if emits_watermark else None,
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
        # Comfortably past any real elapsed time this suite could ever run under: validation's
        # per-page checkpoint commit fences against a live clock (`utcnow()`), not `NOW`, so the
        # claimed lease has to stay valid against real wall-clock time, not just the fixed `NOW`
        # this fixture otherwise reasons about.
        expires_at=NOW + timedelta(days=36500),
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
    await rebuilds.validate_generation(
        estimate_id, owner=owner, lease_generation=claimed.lease_generation, now=NOW
    )
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
    blobs: BlobStore | None = None,
    large_evidence: bool = False,
) -> tuple[SqliteRebuildStore, RebuildCheckpoint, tuple[Document, ...]]:
    """Stage two aliased replacements under a real FK-on SQLite generation."""
    raws = (
        RawDocument(
            source_id="glossary-one",
            uri="https://wiki.example.test/glossary-one",
            media_type="text/plain",
            content="Network Operations Workspace (NOW, NETOPS)"
            + ("\n" + "x" * 2_500_000 if large_evidence else ""),
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
    if large_evidence:
        target = target.model_copy(
            update={"max_memory_bytes": 20_000_000, "max_temporary_bytes": 50_000_000}
        )
    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(embed)
    rebuilds = rebuild_type(
        engine,
        workspace_id=store.workspace_id,
        blobs=blobs or BlobStore(engine, data_dir),
        vectors=vectors,
    )
    plan = await rebuilds.plan_rebuild(run_id, target, missing_limit=10)
    claimed = await rebuilds.claim_generation(
        plan.generation_id,
        "glossary-publisher",
        now=NOW,
        # See `publish_one_replacement`: validation's checkpoint commit fences against a live
        # clock, so this has to stay valid against real wall-clock time, not just `NOW`.
        expires_at=NOW + timedelta(days=36500),
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
        chunk = make_chunk(
            document,
            0,
            "Synthetic bounded derived chunk" if large_evidence else raw.as_text(),
        )
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
    await rebuilds.validate_generation(
        plan.generation_id,
        owner="glossary-publisher",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
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
            await connection.execute(
                select(models.IndexState.workspace_id).where(
                    models.IndexState.workspace_id == "default"
                )
            )
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


async def test_allowed_partial_publication_derives_only_evidence_and_keeps_omission_pending(  # noqa: PLR0915
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    old_complete_run, _, _ = await promoted_snapshot(
        store,
        engine,
        data_dir,
        run_id="old-complete-scope",
        scope_fingerprint="old-complete-v1",
        source_id="old-scope-page",
    )
    async with session_factory(engine).begin() as session:
        await session.execute(
            update(models.AcquisitionRecord)
            .where(models.AcquisitionRecord.run_id == old_complete_run)
            .values(state=AcquisitionRecordState.SETTLED)
        )
        await session.execute(
            update(models.AcquisitionRun)
            .where(models.AcquisitionRun.id == old_complete_run)
            .values(state=AcquisitionRunState.SETTLED)
        )
    run_id, raws, blob_refs = await promoted_partial_snapshot(store, engine, data_dir)
    omitted_live = make_document(
        source="wiki",
        source_id="omitted-one",
        body=b"live content retained while the current scope is partial",
        uri="https://wiki.example.test/content/omitted-one",
        media_type="text/plain",
    )
    await store.upsert_document(omitted_live)
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
        expires_at=NOW + timedelta(days=36500),
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
    await rebuilds.validate_generation(
        plan.generation_id,
        owner="partial-publisher",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
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
    assert await store.get_document(omitted_live.id) is not None
    assert await store.verify_snapshot_manifest(run_id)
    assert await store.latest_unsettled_acquisition_run("wiki") is None
    rendered = json.dumps(metadata, sort_keys=True)
    assert "retained zero" not in rendered
    assert "wiki.example.test" not in rendered


async def test_a_released_generation_keeps_its_prefix_and_is_taken_over_by_the_next_run(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    """The difference between a settlement and an ending, on the one row where it costs.

    A storage failure says nothing about the documents already derived — a writer held SQLite
    for six seconds, a disk was briefly full — so it records why the attempt stopped and gives
    the generation up. `fail_generation` would be the other thing: `claim_generation` refuses a
    failed generation forever, so ending one over contention means cleanup and deriving the
    whole corpus again for a condition that had already passed.

    Two attempts release here, because the count is what tells an operator that a failure they
    were told was transient is not.
    """
    rebuilds, claimed, _ = await staged_glossary_generation(store, engine, data_dir)
    # Read the row rather than trusting the claim: the fixture staged and began validating
    # after it, and what release has to preserve is where the generation actually got to.
    before = await rebuilds.checkpoint(claimed.generation_id)
    built = before.documents_built
    assert built, "the fixture has to have committed something for losing it to mean anything"

    released = await rebuilds.release_generation(
        claimed.generation_id,
        RebuildRefusalCode.STORAGE_FAILED,
        owner="glossary-publisher",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )

    assert released.state is before.state, (
        "released is not ended: the state it was in is the state it stays in, which is what "
        "makes the next claim a takeover rather than a refusal"
    )
    assert released.state is not RebuildState.FAILED
    assert released.diagnostic_code is RebuildRefusalCode.STORAGE_FAILED
    assert released.diagnostic_count == 1
    assert released.lease_owner is None, "nobody owns it, which is what status reports"
    assert released.lease_expires_at is None
    assert released.documents_built == built

    successor = await rebuilds.claim_generation(
        claimed.generation_id,
        "successor",
        now=NOW + timedelta(minutes=1),
        expires_at=NOW + timedelta(minutes=6),
    )

    assert successor.state is RebuildState.BUILDING
    assert successor.documents_built == built, "the takeover resumes rather than starting over"
    assert successor.next_sequence == before.next_sequence
    assert successor.lease_generation > claimed.lease_generation
    assert successor.lease_owner == "successor"

    again = await rebuilds.release_generation(
        claimed.generation_id,
        RebuildRefusalCode.STORAGE_FAILED,
        owner="successor",
        lease_generation=successor.lease_generation,
        now=NOW + timedelta(minutes=2),
    )

    assert again.diagnostic_count == 2, (
        "one storage failure is contention; the same one twice is a machine somebody has to look at"
    )


async def test_releasing_a_generation_another_owner_holds_is_refused(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    """Giving up a generation is still a write, and a lost lease is the loss of that right."""
    rebuilds, claimed, _ = await staged_glossary_generation(store, engine, data_dir)
    # Past `staged_glossary_generation`'s own (real-time-safe) claim expiry, so this simulates
    # "the original lease has now expired" the same way the pre-existing 5-minute-window
    # version of this test did against its shorter expiry.
    await rebuilds.claim_generation(
        claimed.generation_id,
        "successor",
        now=NOW + timedelta(days=36500, minutes=10),
        expires_at=NOW + timedelta(days=36500, minutes=15),
    )

    with pytest.raises(RebuildLeaseConflictError):
        await rebuilds.release_generation(
            claimed.generation_id,
            RebuildRefusalCode.STORAGE_FAILED,
            owner="glossary-publisher",
            lease_generation=claimed.lease_generation,
            now=NOW + timedelta(days=36500, minutes=11),
        )


@pytest.mark.parametrize("damage", ["missing", "corrupt", "symlink", "ref", "manifest"])
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
    blobs = BlobStore(engine, data_dir)
    path = blobs.evidence_path_for(blob_ref)
    if damage == "missing":
        path.unlink()
    elif damage == "corrupt":
        path.chmod(0o600)
        path.write_bytes(b"synthetic corrupt evidence")
    elif damage == "symlink":
        replacement = path.with_name(f"{path.name}.synthetic-replacement")
        replacement.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(replacement)
    elif damage == "ref":
        replacement = await blobs.put(b"synthetic replacement evidence", "text/plain")
        assert isinstance(replacement, StoredBlob)
        sessions = session_factory(engine)
        async with sessions.begin() as session:
            await session.execute(
                update(models.AcquisitionRecord)
                .where(
                    models.AcquisitionRecord.run_id == snapshot_id,
                    models.AcquisitionRecord.blob_ref == blob_ref,
                )
                .values(blob_ref=replacement.hash)
            )
    else:
        sessions = session_factory(engine)
        async with sessions.begin() as session:
            record = (
                await session.execute(
                    select(models.AcquisitionRecord)
                    .where(models.AcquisitionRecord.run_id == snapshot_id)
                    .order_by(models.AcquisitionRecord.sequence)
                    .limit(1)
                )
            ).scalar_one()
            source_record = dict(cast("dict[str, Any]", record.source_record))
            source_record["title"] = "Synthetic changed manifest title"
            record.source_record = cast("Any", source_record)

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


async def test_slow_evidence_verification_does_not_hold_sqlite_writer_slot(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    blobs = GatedEvidenceBlobStore(engine, data_dir)
    blobs.verification_started = asyncio.Event()
    blobs.verification_release = asyncio.Event()
    build = asyncio.create_task(
        staged_glossary_generation(
            store,
            engine,
            data_dir,
            rebuild_type=ArmingEvidenceVerificationStore,
            blobs=blobs,
            large_evidence=True,
        )
    )
    await asyncio.wait_for(blobs.verification_started.wait(), timeout=5)

    started = monotonic()
    sessions = session_factory(engine)
    async with asyncio.timeout(5), sessions.begin() as session:
        await session.execute(
            update(models.Connector)
            .where(models.Connector.workspace_id == store.workspace_id)
            .values(error_message="synthetic unrelated writer probe")
        )
    elapsed = monotonic() - started

    assert elapsed < 5

    assert blobs.active_digest is not None
    blocked_shard = blobs.evidence_lock_shard(blobs.active_digest)
    candidate = 0
    while True:
        unrelated_payload = f"unrelated managed blob {candidate}".encode()
        unrelated_digest = content_hash(unrelated_payload)
        if blobs.evidence_lock_shard(unrelated_digest) != blocked_shard:
            break
        candidate += 1
    started = monotonic()
    async with asyncio.timeout(5):
        unrelated = await blobs.put(unrelated_payload, "application/octet-stream")
        assert isinstance(unrelated, StoredBlob)
        token = await blobs._mark_gc_candidate(  # pyright: ignore[reportPrivateUsage]
            unrelated.hash
        )
        assert token is not None
        assert await blobs._run_gc_intent(  # pyright: ignore[reportPrivateUsage]
            unrelated.hash, token
        )
    assert monotonic() - started < 5

    blobs.verification_release.set()
    rebuilds, claimed, _ = await build
    generation = await rebuilds.checkpoint(claimed.generation_id)
    assert generation.state is RebuildState.VALIDATING


async def test_early_alias_replacement_during_locked_final_scan_uses_canonical_bytes(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    blobs = EarlyAliasMutationBlobStore(engine, data_dir)
    rebuilds, claimed, documents = await staged_glossary_generation(
        store, engine, data_dir, blobs=blobs
    )
    blob_refs = {document.original_ref for document in documents}
    assert None not in blob_refs
    retained = {ref: await blobs.get(ref) for ref in cast("set[str]", blob_refs)}
    assert all(body is not None for body in retained.values())
    for ref in retained:
        alias = blobs.path_for(ref)
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.write_bytes(b"retired alias before final scan")
    mutated: list[str] = []

    def replace_alias(blob_ref: str) -> None:
        alias = blobs.path_for(blob_ref)
        alias.unlink()
        alias.write_bytes(b"retired alias replaced after early probe")
        mutated.append(blob_ref)

    blobs.mutation = replace_alias
    published = await rebuilds.publish_generation(
        claimed.generation_id,
        owner="glossary-publisher",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )

    assert published.state is RebuildState.PUBLISHED
    assert blobs.final_probes >= 4
    assert len(mutated) == 1
    assert await blobs.get(mutated[0]) == retained[mutated[0]]
    assert blobs.path_for(mutated[0]).read_bytes() == b"retired alias replaced after early probe"


@pytest.mark.parametrize("mutation", ["in_place", "unlink", "replacement", "symlink"])
async def test_canonical_fence_refuses_pin_changes_but_ignores_retired_alias(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    mutation: str,
) -> None:
    blobs = BlobStore(engine, data_dir)
    rebuilds, claimed, documents = await staged_glossary_generation(
        store,
        engine,
        data_dir,
        rebuild_type=MutatingAfterEvidenceProbeStore,
        blobs=blobs,
    )
    assert isinstance(rebuilds, MutatingAfterEvidenceProbeStore)
    blob_ref = documents[0].original_ref
    assert blob_ref is not None
    alias = blobs.path_for(blob_ref)
    pin = blobs.evidence_path_for(blob_ref)
    before = pin.stat()
    stored = pin.read_bytes()
    retained = await blobs.get(blob_ref)
    assert retained is not None
    alias.parent.mkdir(parents=True, exist_ok=True)
    alias.write_bytes(stored)

    def mutate() -> None:
        if mutation == "in_place":
            changed = bytes([stored[0] ^ 1]) + stored[1:]
            pin.chmod(0o600)
            pin.write_bytes(changed)
            os.utime(pin, ns=(before.st_atime_ns, before.st_mtime_ns))
            pin.chmod(stat.S_IRUSR)
        elif mutation == "unlink":
            alias.unlink()
        elif mutation == "replacement":
            alias.unlink()
            alias.write_bytes(stored[::-1])
            os.utime(alias, ns=(before.st_atime_ns, before.st_mtime_ns))
        else:
            alias.unlink()
            alias.symlink_to(pin)

    rebuilds.mutation = mutate
    if mutation == "in_place":
        with pytest.raises(RebuildPublicationConflictError) as caught:
            await rebuilds.publish_generation(
                claimed.generation_id,
                owner="glossary-publisher",
                lease_generation=claimed.lease_generation,
                now=NOW,
            )
        assert caught.value.code is RebuildRefusalCode.SNAPSHOT_CHANGED
        assert (await rebuilds.checkpoint(claimed.generation_id)).state is RebuildState.VALIDATING
    else:
        published = await rebuilds.publish_generation(
            claimed.generation_id,
            owner="glossary-publisher",
            lease_generation=claimed.lease_generation,
            now=NOW,
        )
        assert published.state is RebuildState.PUBLISHED
        assert await blobs.get(blob_ref) == retained


async def test_canceled_publication_after_pin_probe_retries_from_durable_pins(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    blobs = BlobStore(engine, data_dir)
    rebuilds, claimed, _ = await staged_glossary_generation(
        store,
        engine,
        data_dir,
        rebuild_type=GatedAfterEvidenceProbeStore,
        blobs=blobs,
    )
    assert isinstance(rebuilds, GatedAfterEvidenceProbeStore)
    rebuilds.probe_complete = asyncio.Event()
    rebuilds.publication_release = asyncio.Event()
    publication = asyncio.create_task(
        rebuilds.publish_generation(
            claimed.generation_id,
            owner="glossary-publisher",
            lease_generation=claimed.lease_generation,
            now=NOW,
        )
    )
    await asyncio.wait_for(rebuilds.probe_complete.wait(), timeout=5)
    publication.cancel()
    with pytest.raises(asyncio.CancelledError):
        await publication

    retry = SqliteRebuildStore(
        engine,
        workspace_id=store.workspace_id,
        blobs=BlobStore(engine, data_dir),
        vectors=LanceVectorStore(data_dir / "vectors"),
    )
    published = await retry.publish_generation(
        claimed.generation_id,
        owner="glossary-publisher",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
    assert published.state is RebuildState.PUBLISHED


async def test_canceled_evidence_verification_leaves_no_fence_and_retry_recovers(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    blobs = GatedEvidenceBlobStore(engine, data_dir)
    blobs.verification_started = asyncio.Event()
    blobs.verification_release = asyncio.Event()
    build = asyncio.create_task(
        staged_glossary_generation(
            store,
            engine,
            data_dir,
            rebuild_type=ArmingEvidenceVerificationStore,
            blobs=blobs,
        )
    )
    await asyncio.wait_for(blobs.verification_started.wait(), timeout=5)
    build.cancel()
    with pytest.raises(asyncio.CancelledError):
        await build

    sessions = session_factory(engine)
    async with sessions() as session:
        row = (await session.execute(select(models.DerivedGeneration))).scalar_one()
        generation_id = row.id
        lease_generation = row.lease_generation
        assert row.state is RebuildState.VALIDATING
        assert row.evidence_verification_digest is None

    target, embed = rebuild_target()
    del target
    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(embed)
    rebuilds = SqliteRebuildStore(
        engine,
        workspace_id=store.workspace_id,
        blobs=BlobStore(engine, data_dir),
        vectors=vectors,
    )
    await rebuilds.validate_generation(
        generation_id, owner="glossary-publisher", lease_generation=lease_generation, now=NOW
    )
    published = await rebuilds.publish_generation(
        generation_id,
        owner="glossary-publisher",
        lease_generation=lease_generation,
        now=NOW,
    )
    assert published.state is RebuildState.PUBLISHED


async def test_successor_takeover_fences_slow_stale_evidence_verifier(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    blobs = GatedEvidenceBlobStore(engine, data_dir)
    blobs.verification_started = asyncio.Event()
    blobs.verification_release = asyncio.Event()
    build = asyncio.create_task(
        staged_glossary_generation(
            store,
            engine,
            data_dir,
            rebuild_type=ArmingEvidenceVerificationStore,
            blobs=blobs,
        )
    )
    await asyncio.wait_for(blobs.verification_started.wait(), timeout=5)
    sessions = session_factory(engine)
    async with sessions() as session:
        generation = (await session.execute(select(models.DerivedGeneration))).scalar_one()
        generation_id = generation.id

    successor = SqliteRebuildStore(
        engine,
        workspace_id=store.workspace_id,
        blobs=BlobStore(engine, data_dir),
    )
    # Past `staged_glossary_generation`'s own (real-time-safe) claim expiry — see
    # `test_releasing_a_generation_another_owner_holds_is_refused`.
    claimed = await successor.claim_generation(
        generation_id,
        "successor-verifier",
        now=NOW + timedelta(days=36500, minutes=10),
        expires_at=NOW + timedelta(days=36500, minutes=20),
    )
    blobs.verification_release.set()

    with pytest.raises(RebuildLeaseConflictError, match="lease changed during verification"):
        await build
    async with sessions() as session:
        generation = await session.get(models.DerivedGeneration, generation_id)
    assert generation is not None
    assert generation.lease_owner == "successor-verifier"
    assert generation.lease_generation == claimed.lease_generation
    assert generation.evidence_verification_digest is None


async def test_publication_uses_one_durable_hash_fence_without_rehashing(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    blobs = CountingEvidenceBlobStore(engine, data_dir)
    rebuilds, claimed, _ = await staged_glossary_generation(store, engine, data_dir, blobs=blobs)
    full_before_publication = blobs.full_verifications
    pins_before_publication = blobs.pin_probes
    final_before_publication = blobs.final_probes
    # One planning pass and one durable verification pass over two unique retained blobs.
    assert full_before_publication == 4

    published = await rebuilds.publish_generation(
        claimed.generation_id,
        owner="glossary-publisher",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )

    assert published.state is RebuildState.PUBLISHED
    assert blobs.full_verifications == full_before_publication
    assert blobs.cheap_probes == 0
    assert blobs.pin_probes == pins_before_publication + 2
    # One locked pre-transaction scan and one immediately-precommit scan; both are stat-only.
    assert blobs.final_probes == final_before_publication + 4
    sessions = session_factory(engine)
    async with sessions() as session:
        generation = await session.get(models.DerivedGeneration, claimed.generation_id)
    assert generation is not None
    assert generation.evidence_inventory_digest
    assert generation.evidence_verification_digest
    assert generation.evidence_verification_lease_generation == claimed.lease_generation


async def test_many_unique_evidence_streams_with_fixed_shard_and_descriptor_state(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    item_count = 300
    raws = tuple(
        RawDocument(
            source_id=f"bounded-{index:04d}",
            uri=f"https://wiki.example.test/bounded/{index:04d}",
            media_type="text/plain",
            content=f"unique synthetic retained representation {index:04d}",
        )
        for index in range(item_count)
    )
    run_id, _ = await promoted_snapshot_many(store, engine, data_dir, raws)
    target, _ = rebuild_target()
    blobs = BlobStore(engine, data_dir)
    rebuilds = SqliteRebuildStore(
        engine,
        workspace_id=store.workspace_id,
        blobs=blobs,
    )
    plan = await rebuilds.plan_rebuild(run_id, target, missing_limit=1)
    sessions = session_factory(engine)
    record_selects = 0

    def count_record_selects(*args: object) -> None:
        nonlocal record_selects
        statement = args[2]
        if isinstance(statement, str) and "FROM acquisition_records" in statement:
            record_selects += 1

    event.listen(engine.sync_engine, "before_cursor_execute", count_record_selects)
    tracemalloc.start()
    try:
        descriptors_before = await asyncio.to_thread(open_descriptor_count)
        async with blobs.evidence_fence() as pins, sessions() as session:
            generation = await session.get(models.DerivedGeneration, plan.generation_id)
            assert generation is not None
            digest = await rebuilds._evidence_inventory_digest(  # pyright: ignore[reportPrivateUsage]
                session, generation
            )
            verified = await rebuilds._evidence_verification_digest(  # pyright: ignore[reportPrivateUsage]
                session,
                generation,
                digest,
                verify_content=True,
                pins=pins,
            )
            shard_bitmap = pins._shard_bitmap  # pyright: ignore[reportPrivateUsage]
            identity_rows = len(session.identity_map)
        descriptors_after = await asyncio.to_thread(open_descriptor_count)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        event.remove(engine.sync_engine, "before_cursor_execute", count_record_selects)

    assert digest
    assert verified
    assert record_selects == 2
    assert identity_rows <= 1
    assert shard_bitmap.bit_count() <= 256
    assert descriptors_after <= descriptors_before + 2
    assert len(list((blobs.root / "evidence-pins" / "by-digest").iterdir())) == item_count
    assert peak < 12 * 1024 * 1024


async def test_evidence_page_batches_one_join_per_page_not_per_document(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_evidence_page` must not round-trip once per document for its snapshot lookup.

    `_EVIDENCE_PAGE` was `1` for exactly this reason: reading N documents took 2N SQLite round
    trips — N document-item reads plus N per-item snapshot reads, one document at a time. With a
    real page size, reading a whole page costs one query for the items and one bounded join for
    their snapshots, regardless of how many documents the page holds.
    """
    monkeypatch.setattr(rebuild_storage, "_EVIDENCE_PAGE", 5)
    item_count = 5
    raws = tuple(
        RawDocument(
            source_id=f"batched-{index}",
            uri=f"https://wiki.example.test/batched/{index}",
            media_type="text/plain",
            content=f"batched evidence document {index}",
        )
        for index in range(item_count)
    )
    run_id, blob_refs = await promoted_snapshot_many(store, engine, data_dir, raws)
    target, embed = rebuild_target()
    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(embed)
    rebuilds = SqliteRebuildStore(
        engine, workspace_id=store.workspace_id, blobs=BlobStore(engine, data_dir), vectors=vectors
    )
    plan = await rebuilds.plan_rebuild(run_id, target, missing_limit=10)
    claimed = await rebuilds.claim_generation(
        plan.generation_id, "batch-worker", now=NOW, expires_at=NOW + timedelta(minutes=5)
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
            (sequence, DerivedReplacement(document=document, chunks=(chunk,), vector_embedded=1))
        )
        await vectors.upsert(
            [chunk], [[1.0, 0.0, 0.0, 0.0]], publication_id=claimed.vector_publication_id
        )
    await rebuilds.stage_replacements(
        plan.generation_id,
        replacements,
        expected_next_sequence=0,
        owner="batch-worker",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )

    sessions = session_factory(engine)
    item_selects = 0
    record_selects = 0

    def count_selects(*args: object) -> None:
        nonlocal item_selects, record_selects
        statement = args[2]
        if not isinstance(statement, str):
            return
        if "FROM derived_generation_items" in statement:
            item_selects += 1
        if "FROM acquisition_records" in statement:
            record_selects += 1

    event.listen(engine.sync_engine, "before_cursor_execute", count_selects)
    try:
        async with sessions() as session:
            generation = await session.get(models.DerivedGeneration, plan.generation_id)
            assert generation is not None
            pairs = await rebuilds._evidence_page(  # pyright: ignore[reportPrivateUsage]
                session, generation, after=-1
            )
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", count_selects)

    assert len(pairs) == item_count
    assert item_selects == 1
    assert record_selects == 1, "one bounded join for the whole page, not one per document"


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
        generation.evidence_inventory_digest = None
        generation.evidence_verification_digest = None
        generation.evidence_verification_lease_generation = None
        generation.evidence_verified_at = None
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


async def test_slow_legacy_published_replay_verification_does_not_block_writer(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    rebuilds, claimed, _ = await staged_glossary_generation(
        store, engine, data_dir, large_evidence=True
    )
    await rebuilds.publish_generation(
        claimed.generation_id,
        owner="glossary-publisher",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
    sessions = session_factory(engine)
    async with sessions.begin() as session:
        generation = await session.get(models.DerivedGeneration, claimed.generation_id)
        assert generation is not None
        generation.evidence_inventory_digest = None
        generation.evidence_verification_digest = None
        generation.evidence_verification_lease_generation = None
        generation.evidence_verified_at = None
        run = await session.get(models.AcquisitionRun, generation.snapshot_run_id)
        assert run is not None
        run.state = AcquisitionRunState.INDEXING
        await session.execute(
            update(models.AcquisitionRecord)
            .where(models.AcquisitionRecord.run_id == run.id)
            .values(state=AcquisitionRecordState.INDEXING)
        )

    blobs = GatedEvidenceBlobStore(engine, data_dir)
    blobs.verification_started = asyncio.Event()
    blobs.verification_release = asyncio.Event()
    blobs.armed = True
    repair_store = SqliteRebuildStore(
        engine,
        workspace_id=store.workspace_id,
        blobs=blobs,
    )
    repair = asyncio.create_task(
        repair_store.claim_generation(
            claimed.generation_id,
            "legacy-repair",
            now=NOW + timedelta(seconds=1),
            expires_at=NOW + timedelta(minutes=1),
        )
    )
    await asyncio.wait_for(blobs.verification_started.wait(), timeout=5)

    started = monotonic()
    async with asyncio.timeout(5), sessions.begin() as session:
        await session.execute(
            update(models.Connector)
            .where(models.Connector.workspace_id == store.workspace_id)
            .values(error_message="synthetic replay writer probe")
        )
    assert monotonic() - started < 5
    blobs.verification_release.set()

    repaired = await repair
    assert repaired.state is RebuildState.PUBLISHED


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
        state = await session.get(models.IndexState, "default")
        if state is None:
            session.add(
                models.IndexState(
                    workspace_id="default",
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
                .where(models.IndexState.workspace_id == "default")
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


async def test_second_connector_promotion_replans_one_workspace_generation_and_fences_old(
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
    workspace_from_first = await rebuilds.plan_rebuild(first_run, target, missing_limit=10)
    workspace_from_second = await rebuilds.plan_rebuild(second_run, target, missing_limit=10)
    assert workspace_from_first.runnable
    assert workspace_from_second.runnable
    assert workspace_from_first.documents == 2
    assert workspace_from_second.documents == 2
    assert workspace_from_first.generation_id == workspace_from_second.generation_id
    assert workspace_from_first.generation_id != first.generation_id
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


async def test_multi_source_generation_resumes_and_publishes_once_atomically(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    wiki_run, wiki_blob, wiki_raw = await promoted_snapshot(store, engine, data_dir)
    drive_run, drive_blob, drive_raw = await promoted_snapshot(
        store,
        engine,
        data_dir,
        run_id="drive-promoted-run",
        connector="drive",
        scope_fingerprint="folder-v1",
        source_id="drive-page-1",
    )
    target, embed = rebuild_target()
    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(embed)
    rebuilds = SqliteRebuildStore(
        engine,
        workspace_id=store.workspace_id,
        blobs=BlobStore(engine, data_dir),
        vectors=vectors,
    )
    plan = await rebuilds.plan_rebuild(wiki_run, target, missing_limit=10)
    assert plan.runnable
    assert plan.documents == 2
    assert (await rebuilds.plan_rebuild(drive_run, target, missing_limit=10)).generation_id == (
        plan.generation_id
    )
    inputs = await rebuilds.snapshot_inputs(plan.generation_id, after_sequence=-1, limit=10)
    assert [(item.sequence, item.connector) for item in inputs] == [(0, "drive"), (1, "wiki")]

    retained = {
        "drive": (drive_raw, drive_blob),
        "wiki": (wiki_raw, wiki_blob),
    }

    def replacement(sequence: int) -> tuple[DerivedReplacement, Chunk]:
        item = inputs[sequence]
        raw, blob_ref = retained[item.connector]
        document = Document(
            id=document_id(store.workspace_id, item.connector, raw.source_id),
            publication_id=plan.generation_id,
            source=item.connector,
            source_id=raw.source_id,
            uri=raw.uri,
            content_hash=content_hash(raw.as_bytes()),
            original_ref=blob_ref,
            media_type=raw.media_type,
            version_token="v2",  # noqa: S106 - source revision, not a credential
            status=DocumentStatus.INDEXED,
        )
        chunk = Chunk(
            id=chunk_id(document.id, 0, raw.as_text()),
            document_id=document.id,
            text=raw.as_text(),
            embed_text=raw.as_text(),
            anchor=Unlocated(reason="plain text"),
            position=0,
            token_count=3,
        )
        return (
            DerivedReplacement(
                document=document,
                chunks=(chunk,),
                parse_fingerprint="plain@2",
                vector_embedded=1,
            ),
            chunk,
        )

    first = await rebuilds.claim_generation(
        plan.generation_id,
        "first-worker",
        now=NOW,
        expires_at=NOW + timedelta(seconds=1),
    )
    first_replacement, first_chunk = replacement(0)
    await vectors.upsert(
        [first_chunk], [[1.0, 0.0, 0.0, 0.0]], publication_id=first.vector_publication_id
    )
    await rebuilds.stage_replacements(
        plan.generation_id,
        [(0, first_replacement)],
        expected_next_sequence=0,
        owner="first-worker",
        lease_generation=first.lease_generation,
        now=NOW,
    )

    resumed_at = NOW + timedelta(seconds=2)
    resumed = await rebuilds.claim_generation(
        plan.generation_id,
        "resumed-worker",
        now=resumed_at,
        expires_at=resumed_at + timedelta(days=30),
    )
    assert resumed.next_sequence == 1
    assert resumed.predecessor_vector_publication_id == first.vector_publication_id
    await rebuilds.copy_checkpointed_vectors(
        plan.generation_id,
        first.vector_publication_id,
        owner="resumed-worker",
        lease_generation=resumed.lease_generation,
        now=resumed_at,
    )
    second_replacement, second_chunk = replacement(1)
    await vectors.upsert(
        [second_chunk],
        [[0.0, 1.0, 0.0, 0.0]],
        publication_id=resumed.vector_publication_id,
    )
    await rebuilds.stage_replacements(
        plan.generation_id,
        [(1, second_replacement)],
        expected_next_sequence=1,
        owner="resumed-worker",
        lease_generation=resumed.lease_generation,
        now=resumed_at,
    )

    # Shadow staging remains invisible until one publication commits the full source set.
    assert await store.get_document(first_replacement.document.id) is None
    assert await store.get_document(second_replacement.document.id) is None
    await rebuilds.begin_validation(
        plan.generation_id,
        owner="resumed-worker",
        lease_generation=resumed.lease_generation,
        now=resumed_at,
    )
    await rebuilds.validate_generation(
        plan.generation_id,
        owner="resumed-worker",
        lease_generation=resumed.lease_generation,
        now=resumed_at,
    )
    published = await rebuilds.publish_generation(
        plan.generation_id,
        owner="resumed-worker",
        lease_generation=resumed.lease_generation,
        now=resumed_at,
    )
    assert published.state is RebuildState.PUBLISHED
    assert await store.get_document(first_replacement.document.id) is not None
    assert await store.get_document(second_replacement.document.id) is not None


async def test_current_promoted_scope_without_a_watermark_is_still_plannable(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    await promoted_snapshot(
        store,
        engine,
        data_dir,
        run_id="prior-watermarked-scope",
        scope_fingerprint="watermarked-scope",
        source_id="prior-watermarked-page",
    )
    run_id, _, raw = await promoted_snapshot(
        store,
        engine,
        data_dir,
        run_id="new-no-watermark-scope",
        scope_fingerprint="no-watermark-scope",
        source_id="new-no-watermark-page",
        emits_watermark=False,
    )
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
    assert plan.runnable, plan
    inputs = await rebuilds.snapshot_inputs(plan.generation_id, after_sequence=-1, limit=10)

    assert plan.documents == 1
    assert [item.source.source_id for item in inputs] == [raw.source_id]


async def test_complete_snapshot_publication_removes_stale_same_source_documents(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    run_id, blob_ref, raw = await promoted_snapshot(store, engine, data_dir)
    stale = make_document(
        source="wiki",
        source_id="stale-outside-manifest",
        body=b"stale live content",
        uri="https://wiki.example.test/content/stale-outside-manifest",
        media_type="text/plain",
    )
    await store.upsert_document(stale)
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
    replacement = make_document(
        source="wiki",
        source_id=raw.source_id,
        body=raw.as_bytes(),
        uri=raw.uri,
        media_type=raw.media_type,
    )

    published = await publish_one_replacement(
        rebuilds,
        vectors,
        estimate_id=plan.generation_id,
        target=target,
        document=replacement,
        raw=raw,
        blob_ref=blob_ref,
        owner="complete-cleanup-publisher",
    )

    assert published.state is RebuildState.PUBLISHED
    assert await store.get_document(stale.id) is None


async def test_generation_bound_snapshot_is_not_removed_by_history_cleanup(
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
    await rebuilds.plan_rebuild(run_id, target, missing_limit=10)
    sessions = session_factory(engine)
    old = datetime(2020, 1, 1, tzinfo=UTC)
    async with sessions.begin() as session:
        await session.execute(
            update(models.AcquisitionRecord)
            .where(models.AcquisitionRecord.run_id == run_id)
            .values(state=AcquisitionRecordState.SETTLED, updated_at=old)
        )
        await session.execute(
            update(models.AcquisitionRun)
            .where(models.AcquisitionRun.id == run_id)
            .values(state=AcquisitionRunState.SETTLED, updated_at=old)
        )

    removed = await store.cleanup_acquisition_history(datetime(2100, 1, 1, tzinfo=UTC), limit=10)

    assert removed == 0
    assert await store.get_acquisition_run(run_id) is not None


async def test_rebuild_plan_reports_aggregate_chunk_budget_diagnostics(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    run_id, _, raw = await promoted_snapshot(store, engine, data_dir)
    current = ChunkFingerprint(
        chunker="structural",
        version="3",
        max_tokens=512,
        overlap_tokens=64,
        tokenizer_id="test/tokenizer",
    )
    target_chunk = current.model_copy(update={"max_tokens": 768, "overlap_tokens": 96})
    document = make_document(
        source="wiki",
        source_id=raw.source_id,
        body=b"stored body",
        uri=raw.uri,
        media_type=raw.media_type,
    )
    await store.upsert_document(document)
    await store.replace_chunks(
        document.id,
        [make_chunk(document, 0, "stored body").model_copy(update={"token_count": 513})],
    )
    sessions = session_factory(engine)
    async with sessions.begin() as session:
        state = await session.get(models.IndexState, "default")
        if state is None:
            session.add(
                models.IndexState(workspace_id="default", chunk_fingerprint=current.canonical())
            )
        else:
            state.chunk_fingerprint = current.canonical()

    target, embed = rebuild_target()
    target = target.model_copy(update={"chunk_fingerprint": target_chunk.canonical()})
    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(embed)
    rebuilds = SqliteRebuildStore(
        engine,
        workspace_id=store.workspace_id,
        blobs=BlobStore(engine, data_dir),
        vectors=vectors,
    )
    estimate = await rebuilds.plan_rebuild(run_id, target, missing_limit=10, persist=False)

    assert estimate.runnable
    assert estimate.current_chunk_fingerprint == current.canonical()
    assert estimate.target_chunk_fingerprint == target_chunk.canonical()
    assert estimate.over_budget_chunks == 1
    assert estimate.max_stored_chunk_tokens == 513
    assert estimate.estimated_embedding_chunks == estimate.estimated_chunks
    assert estimate.network_required is False
    rendered = estimate.model_dump_json()
    assert raw.source_id not in rendered
    assert raw.uri not in rendered


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
        # Real-time-safe: validation's per-page checkpoint commit fences against a live clock.
        expires_at=NOW + timedelta(days=36500),
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
    await rebuilds.validate_generation(
        estimate.generation_id,
        owner="worker",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
    # A #187 pointer swap after staging must win the CAS without publishing relational rows.
    async with sessions.begin() as session:
        index_state = await session.get(models.IndexState, "default")
        index_state_was_missing = index_state is None
        expected_vector_table = None if index_state is None else index_state.vector_table
        expected_inventory = None if index_state is None else index_state.vector_inventory_digest
        if index_state is None:
            session.add(
                models.IndexState(
                    workspace_id="default",
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
        index_state = await session.get(models.IndexState, "default")
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
                select(models.IndexState.vector_inventory_digest).where(
                    models.IndexState.workspace_id == "default"
                )
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
        await rebuilds.validate_generation(
            estimate.generation_id,
            owner="worker",
            lease_generation=claimed.lease_generation,
            now=NOW,
        )
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

    # The scenarios above all held "second"'s own lease generation, and the last of them —
    # `unavailable_replay_inventory` — copied and verified the single staged page before its
    # invented total-count mismatch raised, durably checkpointing that page under this exact
    # lease generation. A resumable replay is supposed to trust that: rather than proving a
    # worker crash mid-copy, reusing "second" here would prove only that an already-checkpointed
    # page is not recopied. So this scenario claims its own fresh takeover, like the ones after
    # it, to give the crash something in-flight to crash on.
    monkeypatch.setattr(vectors, "publication_row_count", original_row_count)
    second_crash = await rebuilds.claim_generation(
        plan.generation_id,
        "second-crash",
        now=takeover_now + timedelta(seconds=1),
        expires_at=takeover_now + timedelta(seconds=1, milliseconds=500),
    )
    assert second_crash.predecessor_vector_publication_id == first.vector_publication_id
    monkeypatch.setattr(vectors, "copy_publication", crash_after_page)
    with pytest.raises(RuntimeError, match="crashed after the first replay page"):
        await rebuilds.copy_checkpointed_vectors(
            plan.generation_id,
            first.vector_publication_id,
            owner="second-crash",
            lease_generation=second_crash.lease_generation,
            now=takeover_now + timedelta(seconds=1),
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


# --- durable, resumable replay and validation checkpoints -------------------------------------


async def _two_document_generation(
    store: SqliteDocStore, engine: AsyncEngine, data_dir: Path, *, owner: str
) -> tuple[
    SqliteRebuildStore, LanceVectorStore, RebuildCheckpoint, tuple[Document, ...], tuple[str, ...]
]:
    """Stage two single-chunk replacements and enter ``VALIDATING`` without validating them."""
    raws = (
        RawDocument(
            source_id="resumable-one",
            uri="https://wiki.example.test/resumable-one",
            media_type="text/plain",
            content="first resumable document",
        ),
        RawDocument(
            source_id="resumable-two",
            uri="https://wiki.example.test/resumable-two",
            media_type="text/plain",
            content="second resumable document",
        ),
    )
    run_id, blob_refs = await promoted_snapshot_many(store, engine, data_dir, raws)
    target, embed = rebuild_target()
    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(embed)
    rebuilds = SqliteRebuildStore(
        engine, workspace_id=store.workspace_id, blobs=BlobStore(engine, data_dir), vectors=vectors
    )
    plan = await rebuilds.plan_rebuild(run_id, target, missing_limit=10)
    # Real time, not the fixed `NOW`: validation's per-page checkpoint commit fences against a
    # live clock, and some callers of this fixture take the generation over via a later,
    # real-time-anchored claim — a claim expiry fixed to `NOW` would already read as expired to
    # both.
    claim_now = datetime.now(UTC)
    claimed = await rebuilds.claim_generation(
        plan.generation_id, owner, now=claim_now, expires_at=claim_now + timedelta(minutes=5)
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
        replacements.append(
            (sequence, DerivedReplacement(document=document, chunks=(chunk,), vector_embedded=1))
        )
        documents.append(document)
        await vectors.upsert(
            [chunk], [[1.0, 0.0, 0.0, 0.0]], publication_id=claimed.vector_publication_id
        )
    await rebuilds.stage_replacements(
        plan.generation_id,
        replacements,
        expected_next_sequence=0,
        owner=owner,
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
    await rebuilds.begin_validation(
        plan.generation_id, owner=owner, lease_generation=claimed.lease_generation, now=NOW
    )
    return rebuilds, vectors, claimed, tuple(documents), blob_refs


async def test_validation_resumes_past_a_durably_checkpointed_page(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupted validation resumes after its last durable page, not from the beginning.

    Two documents, one relational evidence page each (`_EVIDENCE_PAGE` forced to one). The first
    call is made to fail while checking the second document's vectors, after the first document's
    page has already committed. The second call must verify only the second document — the whole
    point of a durable per-page checkpoint is that the first is never asked about again.
    """
    monkeypatch.setattr(rebuild_storage, "_EVIDENCE_PAGE", 1)
    rebuilds, vectors, claimed, documents, _ = await _two_document_generation(
        store, engine, data_dir, owner="resumable-worker"
    )
    original_page_complete = vectors.publication_page_is_complete
    calls: list[str] = []

    async def fail_on_second_document(
        publication_id: str, chunks: Sequence[Chunk], *, embedding_fingerprint: str
    ) -> bool:
        calls.append(chunks[0].document_id)
        if chunks[0].document_id == documents[1].id:
            raise ValueError("private injected failure /private/path")
        return await original_page_complete(
            publication_id, chunks, embedding_fingerprint=embedding_fingerprint
        )

    monkeypatch.setattr(vectors, "publication_page_is_complete", fail_on_second_document)
    with pytest.raises(RebuildPublicationValidationError):
        await rebuilds.validate_generation(
            claimed.generation_id,
            owner="resumable-worker",
            lease_generation=claimed.lease_generation,
            now=NOW,
        )
    assert calls == [documents[0].id, documents[1].id]

    sessions = session_factory(engine)
    async with sessions() as session:
        row = (await session.execute(select(models.DerivedGeneration))).scalar_one()
        assert row.validation_lease_generation == claimed.lease_generation
        assert row.validation_checkpoint_sequence == 0, "only the first document's page committed"
        assert row.validated_vector_count == 1
        assert row.last_progress_at is not None

    calls.clear()
    monkeypatch.setattr(vectors, "publication_page_is_complete", original_page_complete)
    resumed_calls: list[str] = []

    async def counting_page_complete(
        publication_id: str, chunks: Sequence[Chunk], *, embedding_fingerprint: str
    ) -> bool:
        resumed_calls.append(chunks[0].document_id)
        return await original_page_complete(
            publication_id, chunks, embedding_fingerprint=embedding_fingerprint
        )

    monkeypatch.setattr(vectors, "publication_page_is_complete", counting_page_complete)
    await rebuilds.validate_generation(
        claimed.generation_id,
        owner="resumable-worker",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
    assert resumed_calls == [documents[1].id], "resume must not re-verify the checkpointed page"

    published = await rebuilds.publish_generation(
        claimed.generation_id,
        owner="resumable-worker",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
    assert published.state is RebuildState.PUBLISHED


async def test_validation_coalesces_vector_checks_for_one_evidence_page(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One bounded Lance proof covers all staged documents in the relational page."""
    monkeypatch.setattr(rebuild_storage, "_EVIDENCE_PAGE", 2)
    rebuilds, vectors, claimed, documents, _ = await _two_document_generation(
        store, engine, data_dir, owner="coalesced-validation-worker"
    )
    original_page_complete = vectors.publication_page_is_complete
    checked: list[tuple[str, ...]] = []

    async def tracking_page_complete(
        publication_id: str, chunks: Sequence[Chunk], *, embedding_fingerprint: str
    ) -> bool:
        checked.append(tuple(chunk.document_id for chunk in chunks))
        return await original_page_complete(
            publication_id, chunks, embedding_fingerprint=embedding_fingerprint
        )

    monkeypatch.setattr(vectors, "publication_page_is_complete", tracking_page_complete)
    await rebuilds.validate_generation(
        claimed.generation_id,
        owner="coalesced-validation-worker",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )

    assert checked == [tuple(document.id for document in documents)]


async def test_publication_batches_relational_deletes_for_one_evidence_page(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publishing a page must not issue one delete cycle for every document it contains."""
    monkeypatch.setattr(rebuild_storage, "_EVIDENCE_PAGE", 2)
    rebuilds, _vectors, claimed, _documents, _ = await _two_document_generation(
        store, engine, data_dir, owner="batched-publication-worker"
    )
    await rebuilds.validate_generation(
        claimed.generation_id,
        owner="batched-publication-worker",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
    statements = {"chunk_deletes": 0, "glossary_deletes": 0, "chunk_inserts": 0}

    def count_deletes(*args: object) -> None:
        statement = args[2]
        if not isinstance(statement, str):
            return
        normalized = statement.upper()
        if "DELETE FROM CHUNKS" in normalized:
            statements["chunk_deletes"] += 1
        if "DELETE FROM GLOSSARY_ENTRIES" in normalized:
            statements["glossary_deletes"] += 1
        if "INSERT INTO CHUNKS (" in normalized:
            statements["chunk_inserts"] += 1

    event.listen(engine.sync_engine, "before_cursor_execute", count_deletes)
    try:
        published = await rebuilds.publish_generation(
            claimed.generation_id,
            owner="batched-publication-worker",
            lease_generation=claimed.lease_generation,
            now=NOW,
        )
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", count_deletes)

    assert published.state is RebuildState.PUBLISHED
    assert statements == {"chunk_deletes": 1, "glossary_deletes": 1, "chunk_inserts": 1}


async def test_live_chunk_inventory_digest_streams_beyond_one_page(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    """The final publication inventory uses one cursor, rather than keyset-querying each page."""
    for index in range(101):
        document = make_document(
            source="wiki",
            source_id=f"inventory-stream-{index:03d}",
            body=f"inventory document {index}".encode(),
            uri=f"https://wiki.example.test/inventory/{index}",
            media_type="text/plain",
        )
        await store.upsert_document(document)
        await store.replace_chunks(
            document.id, [make_chunk(document, 0, f"inventory document {index}")]
        )
    rebuilds = SqliteRebuildStore(
        engine, workspace_id=store.workspace_id, blobs=BlobStore(engine, data_dir)
    )
    statements = 0

    def count_statements(*args: object) -> None:
        nonlocal statements
        if isinstance(args[2], str):
            statements += 1

    event.listen(engine.sync_engine, "before_cursor_execute", count_statements)
    try:
        async with session_factory(engine)() as session:
            digest = await rebuilds._live_chunk_inventory_digest(  # pyright: ignore[reportPrivateUsage]
                session
            )
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", count_statements)

    assert digest
    assert statements == 1


async def test_takeover_invalidates_a_stale_validation_checkpoint(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A checkpoint recorded under a superseded lease generation is never resumed from.

    Trusting it would mean trusting page evidence checked against a physical vector namespace
    the new owner may not even share — the same reasoning :meth:`copy_checkpointed_vectors`
    applies to its own replay checkpoint.
    """
    monkeypatch.setattr(rebuild_storage, "_EVIDENCE_PAGE", 1)
    rebuilds, vectors, claimed, documents, _ = await _two_document_generation(
        store, engine, data_dir, owner="stale-worker"
    )
    await rebuilds.validate_generation(
        claimed.generation_id,
        owner="stale-worker",
        lease_generation=claimed.lease_generation,
        now=NOW,
    )
    sessions = session_factory(engine)
    async with sessions() as session:
        before = (await session.execute(select(models.DerivedGeneration))).scalar_one()
        assert before.validation_lease_generation == claimed.lease_generation
        assert before.validation_checkpoint_sequence == 1

    # Past `_two_document_generation`'s own claim expiry (real time plus five minutes), so this
    # is a genuine takeover rather than a live-owner conflict.
    takeover_now = datetime.now(UTC) + timedelta(minutes=10)
    taken_over = await rebuilds.claim_generation(
        claimed.generation_id,
        "new-worker",
        now=takeover_now,
        expires_at=takeover_now + timedelta(minutes=5),
    )
    assert taken_over.lease_generation == claimed.lease_generation + 1
    await rebuilds.copy_checkpointed_vectors(
        claimed.generation_id,
        claimed.vector_publication_id,
        owner="new-worker",
        lease_generation=taken_over.lease_generation,
        now=takeover_now,
    )
    # A takeover's claim resets state to `BUILDING`, exactly as it does the document loop's own
    # checkpoint — replay is one more thing a new owner must redo before it may re-enter
    # validation.
    await rebuilds.begin_validation(
        claimed.generation_id,
        owner="new-worker",
        lease_generation=taken_over.lease_generation,
        now=takeover_now,
    )
    original_page_complete = vectors.publication_page_is_complete
    seen: list[str] = []

    async def tracking_page_complete(
        publication_id: str, chunks: Sequence[Chunk], *, embedding_fingerprint: str
    ) -> bool:
        seen.append(chunks[0].document_id)
        return await original_page_complete(
            publication_id, chunks, embedding_fingerprint=embedding_fingerprint
        )

    monkeypatch.setattr(vectors, "publication_page_is_complete", tracking_page_complete)
    await rebuilds.validate_generation(
        claimed.generation_id,
        owner="new-worker",
        lease_generation=taken_over.lease_generation,
        now=takeover_now,
    )
    assert seen == [documents[0].id, documents[1].id], (
        "a stale checkpoint from a superseded lease generation must not be trusted"
    )


async def test_replay_resumes_past_an_already_copied_page_under_the_same_lease(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker that crashes and retries under its own still-valid lease does not recopy.

    Two documents replay one page at a time. The first attempt is made to crash while copying
    the second page, after the first has already copied, verified and durably checkpointed. The
    retry, still holding the same lease, must copy only the second.
    """
    monkeypatch.setattr(rebuild_storage, "_EVIDENCE_PAGE", 1)
    rebuilds, vectors, claimed, documents, _ = await _two_document_generation(
        store, engine, data_dir, owner="replay-worker"
    )

    # Past `_two_document_generation`'s own claim expiry (real time plus five minutes), so this
    # is a genuine takeover rather than a live-owner conflict.
    takeover_now = datetime.now(UTC) + timedelta(minutes=10)
    taken_over = await rebuilds.claim_generation(
        claimed.generation_id,
        "replay-two",
        now=takeover_now,
        expires_at=takeover_now + timedelta(minutes=5),
    )
    assert taken_over.predecessor_vector_publication_id == claimed.vector_publication_id

    original_copy = vectors.copy_publication
    copied: list[str] = []

    async def crash_on_second_document(
        source_publication_id: str, target_publication_id: str, chunks: Sequence[Chunk]
    ) -> None:
        copied.append(chunks[0].document_id)
        await original_copy(source_publication_id, target_publication_id, chunks)
        if chunks[0].document_id == documents[1].id:
            raise VectorStoreStateError("private crash /private/path")

    monkeypatch.setattr(vectors, "copy_publication", crash_on_second_document)
    with pytest.raises(RebuildPublicationValidationError):
        await rebuilds.copy_checkpointed_vectors(
            claimed.generation_id,
            claimed.vector_publication_id,
            owner="replay-two",
            lease_generation=taken_over.lease_generation,
            now=takeover_now,
        )
    assert copied == [documents[0].id, documents[1].id]

    sessions = session_factory(engine)
    async with sessions() as session:
        row = (await session.execute(select(models.DerivedGeneration))).scalar_one()
        assert row.replay_lease_generation == taken_over.lease_generation
        assert row.replay_checkpoint_sequence == 0
        assert row.replayed_vector_count == 1

    copied.clear()
    monkeypatch.setattr(vectors, "copy_publication", original_copy)
    resumed_copies: list[str] = []

    async def counting_copy(
        source_publication_id: str, target_publication_id: str, chunks: Sequence[Chunk]
    ) -> None:
        resumed_copies.append(chunks[0].document_id)
        await original_copy(source_publication_id, target_publication_id, chunks)

    monkeypatch.setattr(vectors, "copy_publication", counting_copy)
    await rebuilds.copy_checkpointed_vectors(
        claimed.generation_id,
        claimed.vector_publication_id,
        owner="replay-two",
        lease_generation=taken_over.lease_generation,
        now=takeover_now,
    )
    assert resumed_copies == [documents[1].id], "resume must not recopy the checkpointed page"


async def test_last_progress_at_is_distinct_from_lease_renewal(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
) -> None:
    """A heartbeat that only renews a lease must not read as content progress.

    ``renew_generation`` is the exact call the timer-driven lease heartbeat makes; it must move
    ``lease_expires_at`` without moving ``last_progress_at``. Only a durable staged batch does.
    """
    run_id, blob_ref, raw = await promoted_snapshot(store, engine, data_dir)
    target, embed = rebuild_target()
    vectors = LanceVectorStore(data_dir / "vectors")
    await vectors.ensure_ready(embed)
    rebuilds = SqliteRebuildStore(
        engine, workspace_id=store.workspace_id, blobs=BlobStore(engine, data_dir), vectors=vectors
    )
    plan = await rebuilds.plan_rebuild(run_id, target, missing_limit=10)
    claimed = await rebuilds.claim_generation(
        plan.generation_id, "heartbeat-worker", now=NOW, expires_at=NOW + timedelta(minutes=5)
    )
    checkpoint = await rebuilds.checkpoint(claimed.generation_id)
    assert checkpoint.last_progress_at is None

    renewed = await rebuilds.renew_generation(
        plan.generation_id,
        "heartbeat-worker",
        claimed.lease_generation,
        now=NOW + timedelta(minutes=1),
        expires_at=NOW + timedelta(minutes=6),
    )
    assert renewed.last_progress_at is None, "a lease renewal alone is not progress"
    assert renewed.lease_expires_at == NOW + timedelta(minutes=6)

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
    await vectors.upsert(
        [chunk], [[1.0, 0.0, 0.0, 0.0]], publication_id=claimed.vector_publication_id
    )
    staged_at = NOW + timedelta(minutes=2)
    staged = await rebuilds.stage_replacements(
        plan.generation_id,
        [(0, DerivedReplacement(document=document, chunks=(chunk,), vector_embedded=1))],
        expected_next_sequence=0,
        owner="heartbeat-worker",
        lease_generation=claimed.lease_generation,
        now=staged_at,
    )
    assert staged.last_progress_at == staged_at, "a durable staged batch is real progress"


# --- lease renewal during checkpoint replay ---------------------------------------------------


async def _replayable_takeover(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    *,
    items: int,
    lease_seconds: int,
) -> tuple[SqliteRebuildStore, LanceVectorStore, RebuildCheckpoint, RebuildCheckpoint, str]:
    """A generation with ``items`` staged replacements and a fresh short-leased takeover.

    Several items rather than one because replay pages per item: a single-item checkpoint
    replays in one bounded copy and can never outlive any lease, which is exactly the shape
    the existing takeover coverage already has and the shape this defect hides behind.
    """
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
            max_temporary_bytes=10_000_000,
        ),
        missing_limit=10,
    )
    first = await rebuilds.claim_generation(
        plan.generation_id, "first", now=NOW, expires_at=NOW + timedelta(minutes=10)
    )
    for sequence in range(items):
        document = make_document(
            source="wiki",
            source_id=f"{raw.source_id}-{sequence}",
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
        chunk = make_chunk(document, 0, f"{raw.as_text()} {sequence}")
        await vectors.upsert(
            [chunk], [[1.0, 0.0, 0.0, 0.0]], publication_id=first.vector_publication_id
        )
        await rebuilds.stage_replacements(
            plan.generation_id,
            [(sequence, DerivedReplacement(document=document, chunks=(chunk,), vector_embedded=1))],
            expected_next_sequence=sequence,
            owner="first",
            lease_generation=first.lease_generation,
            now=NOW,
        )
    takeover_at = NOW + timedelta(minutes=20)
    second = await rebuilds.claim_generation(
        plan.generation_id,
        "second",
        now=takeover_at,
        expires_at=takeover_at + timedelta(seconds=lease_seconds),
    )
    assert second.predecessor_vector_publication_id == first.vector_publication_id
    return rebuilds, vectors, first, second, plan.generation_id


class ReplayClock:
    """A clock the replay itself advances, so a long replay needs no wall-clock time.

    Advancing on the vector store's own bounded copy is what makes this deterministic: replay
    duration is a function of how many pages it copies, not of how the event loop happened to
    schedule. ``asyncio.sleep`` would make the same test pass or fail on machine load.
    """

    def __init__(self, start: datetime, *, per_page: timedelta) -> None:
        self.now = start
        self._per_page = per_page
        self.pages = 0

    def __call__(self) -> datetime:
        return self.now

    def advance_one_page(self) -> None:
        self.pages += 1
        self.now += self._per_page


async def test_replay_enforces_the_lease_it_was_handed_rather_than_extending_it(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay fences on the lease and never extends it; keeping it alive is the caller's job.

    Asserted so the division stays where it is. The store's business is refusing to write into a
    namespace this worker no longer owns, and it does that whether or not anybody is renewing —
    which is what makes the fencing worth trusting. Moving renewal *into* here would make the
    check and the thing it checks the same code.

    This is also the defect's shape, seen from below: nothing renewed, so a replay longer than
    one lease could never finish, and every retry began another full replay under another finite
    lease. :func:`test_replay_across_several_leases_completes_when_the_lease_is_renewed` is the
    same replay with a heartbeat over it, and
    ``tests/ingest/test_generation_rebuild.py`` covers the worker that now provides one.
    """
    rebuilds, vectors, first, second, generation_id = await _replayable_takeover(
        store, engine, data_dir, items=6, lease_seconds=5
    )
    clock = ReplayClock(NOW + timedelta(minutes=20), per_page=timedelta(seconds=3))
    original_copy = vectors.copy_publication

    async def slow_copy(*args: object, **kwargs: object) -> None:
        clock.advance_one_page()
        await original_copy(*args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(vectors, "copy_publication", slow_copy)

    with pytest.raises(RebuildLeaseConflictError, match="lease"):
        await rebuilds.copy_checkpointed_vectors(
            generation_id,
            first.vector_publication_id,
            owner="second",
            lease_generation=second.lease_generation,
            now=clock.now,
            clock=clock,
        )

    assert clock.pages >= 2, "replay must get far enough to outlive the lease"


async def test_replay_across_several_leases_completes_when_the_lease_is_renewed(
    store: SqliteDocStore,
    engine: AsyncEngine,
    data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same replay, renewed between pages, against the real store and a real vector table.

    The worker's heartbeat is a timer over a fake store elsewhere; this is the half that fake
    cannot answer — that a renewal landing *while* replay is mid-flight is tolerated by the
    production path rather than tripping one of the guards replay carries. Replay re-reads the
    generation row every page and compares ``next_sequence`` against the value it started with,
    so a renewal that touched anything but the expiry would surface here as a checkpoint that
    moved under the worker.

    Renewal runs in a task of its own, handed the turn by the replay itself: no sleeps and no
    scheduling luck, but a genuine second task writing to SQLite while replay is mid-page, which
    is the contention an inline renewal would quietly avoid testing.
    """
    lease_seconds = 5
    rebuilds, vectors, first, second, generation_id = await _replayable_takeover(
        store, engine, data_dir, items=6, lease_seconds=lease_seconds
    )
    clock = ReplayClock(NOW + timedelta(minutes=20), per_page=timedelta(seconds=3))
    monkeypatch.setattr(rebuild_storage, "utcnow", clock)
    original_copy = vectors.copy_publication

    page_reached = asyncio.Event()
    renewed = asyncio.Event()

    async def heartbeat() -> None:
        while True:
            await page_reached.wait()
            page_reached.clear()
            await rebuilds.renew_generation(
                generation_id,
                "second",
                second.lease_generation,
                now=clock.now,
                expires_at=clock.now + timedelta(seconds=lease_seconds),
            )
            renewed.set()

    async def slow_copy(*args: object, **kwargs: object) -> None:
        clock.advance_one_page()
        renewed.clear()
        page_reached.set()
        await renewed.wait()
        await original_copy(*args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(vectors, "copy_publication", slow_copy)
    beat = asyncio.ensure_future(heartbeat())
    try:
        await rebuilds.copy_checkpointed_vectors(
            generation_id,
            first.vector_publication_id,
            owner="second",
            lease_generation=second.lease_generation,
            now=clock.now,
        )
    finally:
        beat.cancel()
        with suppress(asyncio.CancelledError):
            await beat

    elapsed = clock.now - (NOW + timedelta(minutes=20))
    assert elapsed > timedelta(seconds=lease_seconds * 2), (
        "the replay has to outlast more than two leases for this to be the case under test"
    )
    checkpoint = await rebuilds.checkpoint(generation_id)
    assert checkpoint.lease_owner == "second", "renewal must not move ownership"
    assert checkpoint.lease_generation == second.lease_generation, "nor the fencing generation"
    assert checkpoint.lease_expires_at is not None
    assert checkpoint.lease_expires_at > clock.now, "and the expiry has to have moved with it"
    assert checkpoint.vector_publication_id == second.vector_publication_id, (
        "a completed replay certifies the takeover's own namespace"
    )
