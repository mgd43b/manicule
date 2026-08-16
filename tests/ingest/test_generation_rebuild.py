from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, override

import pytest
from manicule_plugin_hostile import HangingMiddleware, HostileConfig
from sqlalchemy.exc import IntegrityError

from manicule.core.acquisition import AcquiredSource
from manicule.core.anchors import Unlocated
from manicule.core.content import Chunk, Document, DocumentStatus, PipelineStage, RawDocument
from manicule.core.errors import ParseError
from manicule.core.ids import content_hash
from manicule.core.rebuild import (
    DerivedReplacement,
    MissingSnapshotInput,
    RebuildCheckpoint,
    RebuildDerivationError,
    RebuildEstimate,
    RebuildLeaseConflictError,
    RebuildLeaseError,
    RebuildRefusalCode,
    RebuildRefusedError,
    RebuildState,
    RebuildStorageError,
    RebuildTarget,
    RebuildTerminalError,
    RebuildTerminalGenerationError,
    RebuildValidationError,
    SnapshotRebuildInput,
)
from manicule.ingest.glossary_lineage import glossary_fingerprint
from manicule.ingest.middleware import MiddlewareRunner
from manicule.ingest.rebuild import (
    OfflineGenerationRebuilder,
    ParserChunkerRelationalDeriver,
    PreparedReplacement,
)
from manicule.ingest.workers import MEGABYTE, InProcessRunner, WorkerConfig, WorkerPool
from manicule.parsers.chain import ParserChain
from manicule.parsers.config import PlaintextConfig
from manicule.parsers.expansion import ExpandedMember, MemberFailure
from manicule.parsers.plaintext import PlaintextParser
from manicule.parsers.versions import parse_fingerprint
from tests.ingest.fakes import BlockChunker, PassThrough

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from pathlib import Path


class OneMemberContainer:
    async def expand(self, raw: RawDocument) -> AsyncIterator[ExpandedMember]:
        yield ExpandedMember(
            source_id=f"zip:{raw.source_id}!/inside.txt",
            uri=f"zip:{raw.uri}!/inside.txt",
            raw=RawDocument(
                source_id=f"zip:{raw.source_id}!/inside.txt",
                uri=f"zip:{raw.uri}!/inside.txt",
                media_type="text/plain",
                content="member text",
            ),
            depth=1,
        )


class FailureContainer:
    async def expand(self, raw: RawDocument) -> AsyncIterator[MemberFailure]:
        yield MemberFailure(
            source_id=f"zip:{raw.source_id}!/encrypted.bin",
            uri=f"zip:{raw.uri}!/encrypted.bin",
            status=DocumentStatus.FAILED,
            reason="member is encrypted",
            depth=1,
        )
        yield MemberFailure(
            source_id=f"zip:{raw.source_id}!/too-deep.zip",
            uri=f"zip:{raw.uri}!/too-deep.zip",
            status=DocumentStatus.UNSUPPORTED_MEDIA_TYPE,
            reason="container depth exceeds the configured limit",
            depth=4,
        )


def target(*, batch: int = 2, memory: int = 8192) -> RebuildTarget:
    return RebuildTarget(
        parser_routing="routing-v2",
        parser_set=("markdown@2",),
        chunk_fingerprint="chunker-v2/tokenizer-v3/size-64/overlap-8",
        embedding_fingerprint="embedder-v1",
        glossary_fingerprint="glossary-v2",
        fts_tokenizer="unicode61",
        batch_documents=batch,
        max_memory_bytes=memory,
        max_temporary_bytes=8192,
    )


def source(sequence: int, body: str) -> tuple[SnapshotRebuildInput, bytes]:
    raw = RawDocument(
        source_id=f"page-{sequence}",
        uri=f"https://never-logged.invalid/{sequence}",
        media_type="text/plain",
        content=body,
    )
    envelope = AcquiredSource.from_raw(raw)
    return (
        SnapshotRebuildInput(sequence=sequence, blob_ref=f"blob-{sequence}", source=envelope),
        raw.as_bytes(),
    )


@dataclass
class FakeStore:
    items: list[SnapshotRebuildInput]
    fail_stage_once: bool = False
    missing: bool = False
    staged: dict[int, DerivedReplacement] = field(default_factory=dict[int, DerivedReplacement])
    published: bool = False
    old_reads: int = 0
    stage_calls: int = 0
    late_cancel: asyncio.Event | None = None
    publish_started: asyncio.Event | None = None
    publish_release: asyncio.Event | None = None
    claimed_owners: list[str] = field(default_factory=list[str])
    failed_code: RebuildRefusalCode | None = None

    def _checkpoint(self, state: RebuildState | None = None) -> RebuildCheckpoint:
        next_sequence = 0
        while next_sequence in self.staged:
            next_sequence += 1
        return RebuildCheckpoint(
            generation_id="generation-v2",
            state=state or (RebuildState.PUBLISHED if self.published else RebuildState.BUILDING),
            expected_items=len(self.items),
            next_sequence=next_sequence,
            documents_built=len(self.staged),
            chunks_built=sum(len(value.chunks) for value in self.staged.values()),
            vectors_reused=sum(value.vector_reused for value in self.staged.values()),
            vectors_embedded=sum(value.vector_embedded for value in self.staged.values()),
            lease_owner="worker",
            lease_generation=1,
            fence_generation=1,
        )

    async def plan_rebuild(
        self,
        snapshot_run_id: str,
        target: RebuildTarget,
        *,
        missing_limit: int,
        persist: bool = True,
    ) -> RebuildEstimate:
        del target, missing_limit, persist
        if self.missing:
            return RebuildEstimate(
                generation_id="generation-v2",
                snapshot_run_id=snapshot_run_id,
                documents=len(self.items),
                expected_items=len(self.items),
                known_source_bytes=0,
                estimated_chunks=0,
                estimated_seconds=0,
                estimated_peak_memory_bytes=0,
                estimated_temporary_bytes=0,
                missing_count=len(self.items),
                missing=tuple(MissingSnapshotInput(sequence=item.sequence) for item in self.items),
            )
        return RebuildEstimate(
            generation_id="generation-v2",
            snapshot_run_id=snapshot_run_id,
            documents=len(self.items),
            expected_items=len(self.items),
            known_source_bytes=20,
            estimated_chunks=len(self.items),
            estimated_seconds=0.5,
            estimated_peak_memory_bytes=1024,
            estimated_temporary_bytes=2048,
            missing_count=0,
        )

    async def checkpoint(self, generation_id: str) -> RebuildCheckpoint:
        assert generation_id == "generation-v2"
        return self._checkpoint()

    async def claim_generation(
        self, generation_id: str, owner: str, *, now: object, expires_at: object
    ) -> RebuildCheckpoint:
        del generation_id, now, expires_at
        self.claimed_owners.append(owner)
        return self._checkpoint()

    async def renew_generation(
        self,
        generation_id: str,
        owner: str,
        lease_generation: int,
        *,
        now: object,
        expires_at: object,
    ) -> RebuildCheckpoint:
        del generation_id, owner, lease_generation, now, expires_at
        return self._checkpoint()

    async def assert_generation_lease(
        self,
        generation_id: str,
        owner: str,
        lease_generation: int,
        *,
        now: object,
    ) -> None:
        del generation_id, owner, lease_generation, now

    async def copy_checkpointed_vectors(
        self,
        generation_id: str,
        source_publication_id: str,
        *,
        owner: str,
        lease_generation: int,
        now: object,
        cancel: asyncio.Event | None = None,
    ) -> None:
        del generation_id, source_publication_id, owner, lease_generation, now, cancel

    async def snapshot_inputs(
        self, generation_id: str, *, after_sequence: int, limit: int
    ) -> Sequence[SnapshotRebuildInput]:
        assert generation_id == "generation-v2"
        self.old_reads += 1
        return [item for item in self.items if item.sequence > after_sequence][:limit]

    async def check_capacity(
        self, generation_id: str, replacements: Sequence[DerivedReplacement]
    ) -> None:
        del generation_id, replacements

    async def stage_replacements(
        self,
        generation_id: str,
        replacements: Sequence[tuple[int, DerivedReplacement]],
        *,
        expected_next_sequence: int,
        owner: str,
        lease_generation: int,
        now: object,
    ) -> RebuildCheckpoint:
        del owner, lease_generation, now
        assert generation_id == "generation-v2"
        self.stage_calls += 1
        if self.fail_stage_once:
            self.fail_stage_once = False
            raise RuntimeError("simulated checkpoint crash")
        assert expected_next_sequence == self._checkpoint().next_sequence
        for sequence, replacement in replacements:
            existing = self.staged.get(sequence)
            if existing is not None and existing != replacement:
                raise RuntimeError("non-deterministic retry")
            self.staged[sequence] = replacement
        return self._checkpoint()

    async def begin_validation(
        self,
        generation_id: str,
        *,
        owner: str,
        lease_generation: int,
        now: object,
    ) -> RebuildCheckpoint:
        del owner, lease_generation, now
        assert len(self.staged) == len(self.items)
        return self._checkpoint(RebuildState.VALIDATING)

    async def validate_generation(self, generation_id: str) -> None:
        assert generation_id == "generation-v2"
        assert not self.published
        if self.late_cancel is not None:
            self.late_cancel.set()

    async def publish_generation(
        self,
        generation_id: str,
        *,
        owner: str,
        lease_generation: int,
        now: object,
    ) -> RebuildCheckpoint:
        del owner, lease_generation, now
        assert generation_id == "generation-v2"
        if self.publish_started is not None:
            self.publish_started.set()
        if self.publish_release is not None:
            await self.publish_release.wait()
        self.published = True
        return self._checkpoint()

    async def cancel_generation(
        self,
        generation_id: str,
        *,
        owner: str,
        lease_generation: int,
        now: object,
    ) -> RebuildCheckpoint:
        del owner, lease_generation, now
        return self._checkpoint(RebuildState.CANCELED)

    async def fail_generation(
        self,
        generation_id: str,
        code: RebuildRefusalCode,
        *,
        owner: str,
        lease_generation: int,
        now: object,
    ) -> RebuildCheckpoint:
        del generation_id, owner, lease_generation, now
        self.failed_code = code
        return self._checkpoint(RebuildState.FAILED)


@dataclass
class FakeBlobs:
    bodies: dict[str, bytes]

    async def get_bounded(self, digest: str, *, max_bytes: int) -> bytes | None:
        value = self.bodies.get(digest)
        return value if value is None or len(value) <= max_bytes else None


@dataclass
class FakeDeriver:
    calls: list[str] = field(default_factory=list[str])

    async def prepare(
        self,
        raw: RawDocument,
        target: RebuildTarget,
        *,
        generation_id: str,
        blob_ref: str,
        title: str,
        version_token: str | None,
    ) -> PreparedReplacement:
        del blob_ref, title, version_token
        self.calls.append(raw.source_id)
        document = Document(
            id=f"document-{raw.source_id}",
            publication_id=generation_id,
            source="wiki",
            source_id=raw.source_id,
            uri=raw.uri,
            content_hash="retained-hash",
            original_ref=f"blob-{raw.source_id.removeprefix('page-')}",
            media_type=raw.media_type,
            status=DocumentStatus.INDEXED,
        )
        text = raw.as_text()
        chunk = Chunk(
            id=f"chunk-{raw.source_id}",
            document_id=document.id,
            text=text,
            embed_text=text,
            anchor=Unlocated(reason="fixture"),
            position=0,
            token_count=1,
        )
        replacement = DerivedReplacement(
            document=document,
            chunks=(chunk,),
            parse_fingerprint=target.parser_set[0],
            vector_reused=1,
        )
        return PreparedReplacement(
            replacement=replacement,
            vectors=(),
            memory_bytes=len(raw.as_bytes()) + 128,
            temporary_bytes=256,
        )

    async def stage(self, prepared: PreparedReplacement, *, publication_id: str) -> None:
        del prepared, publication_id


class StorageFailureStore(FakeStore):
    @override
    async def publish_generation(
        self,
        generation_id: str,
        *,
        owner: str,
        lease_generation: int,
        now: object,
    ) -> RebuildCheckpoint:
        del generation_id, owner, lease_generation, now
        raise IntegrityError(
            "INSERT INTO private_table VALUES (?)",
            ("https://wiki.example.test/private?cookie=secret",),
            RuntimeError("/private/machine/workspace.sqlite"),
        )


class ValidationFailureStore(FakeStore):
    @override
    async def validate_generation(self, generation_id: str) -> None:
        del generation_id
        raise RuntimeError("invalid row https://wiki.example.test/private token=secret")


class ValidationStorageFailureStore(FakeStore):
    @override
    async def validate_generation(self, generation_id: str) -> None:
        del generation_id
        raise IntegrityError(
            "SELECT private_column FROM private_table WHERE secret = ?",
            ("cookie=secret",),
            RuntimeError("/private/machine/workspace.sqlite"),
        )


@dataclass
class ClaimFailureStore(FakeStore):
    claim_failure: Exception = field(default_factory=RebuildLeaseConflictError)

    @override
    async def claim_generation(
        self, generation_id: str, owner: str, *, now: object, expires_at: object
    ) -> RebuildCheckpoint:
        del generation_id, owner, now, expires_at
        raise self.claim_failure


class DerivationFailure(FakeDeriver):
    @override
    async def prepare(
        self,
        raw: RawDocument,
        target: RebuildTarget,
        *,
        generation_id: str,
        blob_ref: str,
        title: str,
        version_token: str | None,
    ) -> PreparedReplacement:
        del raw, target, generation_id, blob_ref, title, version_token
        raise ParseError("secret body at /private/path from https://wiki.example.test")


@pytest.mark.asyncio
async def test_publication_integrity_failure_is_bounded_and_marks_generation_failed() -> None:
    item, body = source(0, "private source body")
    store = StorageFailureStore([item])

    with pytest.raises(RebuildStorageError) as caught:
        await OfflineGenerationRebuilder(
            store=store,
            blobs=FakeBlobs({item.blob_ref: body}),
            deriver=FakeDeriver(),
        ).run("promoted-run", target())

    assert str(caught.value) == "offline rebuild storage failed"
    assert store.failed_code is RebuildRefusalCode.STORAGE_FAILED
    assert store.published is False
    rendered = str(caught.value).lower()
    for private in ("insert", "secret", "wiki.example.test", "/private", "sqlite"):
        assert private not in rendered


@pytest.mark.asyncio
async def test_validation_failure_is_bounded_and_marks_generation_failed() -> None:
    item, body = source(0, "body")
    store = ValidationFailureStore([item])

    with pytest.raises(RebuildValidationError, match="offline rebuild validation failed"):
        await OfflineGenerationRebuilder(
            store=store,
            blobs=FakeBlobs({item.blob_ref: body}),
            deriver=FakeDeriver(),
        ).run("promoted-run", target())

    assert store.failed_code is RebuildRefusalCode.INVALID_REPLACEMENT
    assert store.published is False


@pytest.mark.asyncio
async def test_validation_storage_failure_is_bounded_and_marks_generation_failed() -> None:
    item, body = source(0, "body")
    store = ValidationStorageFailureStore([item])

    with pytest.raises(RebuildStorageError, match="offline rebuild storage failed"):
        await OfflineGenerationRebuilder(
            store=store,
            blobs=FakeBlobs({item.blob_ref: body}),
            deriver=FakeDeriver(),
        ).run("promoted-run", target())

    assert store.failed_code is RebuildRefusalCode.STORAGE_FAILED
    assert store.published is False


@pytest.mark.asyncio
async def test_derivation_failure_is_bounded_and_marks_generation_failed() -> None:
    item, body = source(0, "body")
    store = FakeStore([item])

    with pytest.raises(RebuildDerivationError, match="offline rebuild derivation failed"):
        await OfflineGenerationRebuilder(
            store=store,
            blobs=FakeBlobs({item.blob_ref: body}),
            deriver=DerivationFailure(),
        ).run("promoted-run", target())

    assert store.failed_code is RebuildRefusalCode.DERIVATION_FAILED
    assert store.published is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("internal", "public", "message"),
    [
        (
            RebuildLeaseConflictError("owner secret"),
            RebuildLeaseError,
            "offline rebuild lease was lost",
        ),
        (
            RebuildTerminalGenerationError("terminal secret"),
            RebuildTerminalError,
            "offline rebuild generation is terminal",
        ),
    ],
)
async def test_internal_claim_failures_cross_as_bounded_public_errors(
    internal: Exception, public: type[Exception], message: str
) -> None:
    item, body = source(0, "body")

    with pytest.raises(public, match=message):
        await OfflineGenerationRebuilder(
            store=ClaimFailureStore([item], claim_failure=internal),
            blobs=FakeBlobs({item.blob_ref: body}),
            deriver=FakeDeriver(),
        ).run("promoted-run", target())


@pytest.mark.asyncio
async def test_rebuild_is_bounded_offline_and_publishes_only_after_validation() -> None:
    pairs = [source(index, f"body {index}") for index in range(5)]
    store = FakeStore([item for item, _ in pairs])
    blobs = FakeBlobs({item.blob_ref: body for item, body in pairs})
    deriver = FakeDeriver()
    result = await OfflineGenerationRebuilder(store=store, blobs=blobs, deriver=deriver).run(
        "promoted-run", target()
    )

    assert result.state is RebuildState.PUBLISHED
    assert result.documents_built == 5
    assert store.stage_calls == 5, "each document reaches a durable bounded checkpoint"
    assert store.published
    assert deriver.calls == [f"page-{index}" for index in range(5)]


@pytest.mark.asyncio
async def test_default_worker_owner_is_unique_per_invocation() -> None:
    item, body = source(0, "body")
    store = FakeStore([item])
    runner = OfflineGenerationRebuilder(
        store=store, blobs=FakeBlobs({item.blob_ref: body}), deriver=FakeDeriver()
    )
    await runner.run("promoted-run", target())
    await runner.run("promoted-run", target())
    assert len(store.claimed_owners) == 2
    assert store.claimed_owners[0] != store.claimed_owners[1]


@pytest.mark.asyncio
async def test_checkpoint_crash_retries_without_duplicate_generation_or_rows() -> None:
    pairs = [source(index, f"body {index}") for index in range(3)]
    store = FakeStore([item for item, _ in pairs], fail_stage_once=True)
    runner = OfflineGenerationRebuilder(
        store=store,
        blobs=FakeBlobs({item.blob_ref: body for item, body in pairs}),
        deriver=FakeDeriver(),
    )
    with pytest.raises(RuntimeError, match="checkpoint crash"):
        await runner.run("promoted-run", target())
    result = await runner.run("promoted-run", target())
    assert result.state is RebuildState.PUBLISHED
    assert sorted(store.staged) == [0, 1, 2]


@pytest.mark.asyncio
async def test_missing_input_is_a_typed_aggregate_safe_refusal() -> None:
    item, _ = source(0, "secret body")
    store = FakeStore([item], missing=True)
    with pytest.raises(RebuildRefusedError) as caught:
        await OfflineGenerationRebuilder(
            store=store, blobs=FakeBlobs({}), deriver=FakeDeriver()
        ).run("promoted-run", target())
    assert caught.value.code is RebuildRefusalCode.MISSING_LOCAL_INPUT
    rendered = repr(caught.value.estimate.model_dump())
    assert "secret" not in rendered
    assert "never-logged" not in rendered


@pytest.mark.asyncio
async def test_cancellation_never_publishes_a_partial_generation() -> None:
    item, body = source(0, "body")
    store = FakeStore([item])
    event = __import__("asyncio").Event()
    event.set()
    result = await OfflineGenerationRebuilder(
        store=store, blobs=FakeBlobs({item.blob_ref: body}), deriver=FakeDeriver()
    ).run("promoted-run", target(), cancel=event)
    assert result.state is RebuildState.CANCELED
    assert not store.published
    assert not store.staged


@pytest.mark.asyncio
async def test_cancel_arriving_after_validation_fences_publication() -> None:
    item, body = source(0, "body")
    event = asyncio.Event()
    store = FakeStore([item], late_cancel=event)
    result = await OfflineGenerationRebuilder(
        store=store, blobs=FakeBlobs({item.blob_ref: body}), deriver=FakeDeriver()
    ).run("promoted-run", target(), cancel=event)
    assert result.state is RebuildState.CANCELED
    assert not store.published


@pytest.mark.asyncio
async def test_task_cancellation_joins_an_in_flight_atomic_publication() -> None:
    item, body = source(0, "body")
    started = asyncio.Event()
    release = asyncio.Event()
    store = FakeStore([item], publish_started=started, publish_release=release)
    running = asyncio.create_task(
        OfflineGenerationRebuilder(
            store=store, blobs=FakeBlobs({item.blob_ref: body}), deriver=FakeDeriver()
        ).run("promoted-run", target())
    )
    await started.wait()
    running.cancel()
    await asyncio.sleep(0)
    assert not running.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert store.published


@pytest.mark.asyncio
async def test_dry_run_refuses_estimated_memory_before_claim_or_derivation() -> None:
    item, body = source(0, "body")
    deriver = FakeDeriver()
    with pytest.raises(RebuildRefusedError) as caught:
        await OfflineGenerationRebuilder(
            store=FakeStore([item]),
            blobs=FakeBlobs({item.blob_ref: body}),
            deriver=deriver,
        ).run("promoted-run", target(memory=100))
    assert caught.value.code is RebuildRefusalCode.MEMORY_BOUND
    assert not deriver.calls


def test_replacement_refuses_mixed_document_identities_and_incomplete_vectors() -> None:
    item, _ = source(0, "body")
    document = Document(
        id="document-a",
        publication_id="generation-v2",
        source="wiki",
        source_id=item.source.source_id,
        uri=item.source.uri,
        content_hash=item.source.content_hash,
        media_type=item.source.media_type,
        status=DocumentStatus.INDEXED,
    )
    chunk = Chunk(
        id="chunk-a",
        document_id="document-b",
        text="body",
        embed_text="body",
        anchor=Unlocated(reason="fixture"),
        position=0,
        token_count=1,
    )
    replacement = DerivedReplacement(document=document, chunks=(chunk,), vector_reused=1)
    with pytest.raises(ValueError, match="another document"):
        replacement.validate_identity()


@pytest.mark.asyncio
async def test_production_relational_deriver_runs_parser_routing_chunker_and_glossary_offline() -> (
    None
):
    parser = ParserChain(
        parsers={"plaintext": PlaintextParser(PlaintextConfig())},
        chains={"text/plain": ("plaintext",)},
    )
    chunker = BlockChunker()
    middleware = MiddlewareRunner((PassThrough(),))
    installed_parse = parse_fingerprint("plaintext")
    assert installed_parse is not None
    requested = RebuildTarget(
        parser_routing="plain-routing-v1",
        parser_set=(installed_parse.canonical(),),
        chunk_fingerprint=chunker.fingerprint.canonical(),
        embedding_fingerprint="embed-v1",
        glossary_fingerprint=glossary_fingerprint(middleware=middleware.chain()).canonical(),
        fts_tokenizer="unicode61",
        batch_documents=1,
        max_memory_bytes=1_000_000,
        max_temporary_bytes=1_000_000,
    )
    raw = RawDocument(
        source_id="page-1",
        uri="https://snapshot.invalid/page-1",
        media_type="text/plain",
        content="NOW — Network Operations Workspace",
    )
    replacement = await ParserChunkerRelationalDeriver(
        workspace_id="default",
        source="wiki",
        parser_chain=parser,
        routing_identity="plain-routing-v1",
        chunker=chunker,
        middleware=middleware,
        parse_runner=InProcessRunner(
            {"plaintext": PlaintextParser(PlaintextConfig())},
            middleware=middleware,
            chunker=chunker,
        ),
    ).derive(
        raw,
        requested,
        generation_id="generation-v2",
        blob_ref="retained-blob",
        title="Operations glossary",
        version_token="v2",  # noqa: S106 - source revision, not a credential
    )
    assert replacement.document.status is DocumentStatus.INDEXED
    assert replacement.document.original_ref == "retained-blob"
    assert replacement.document.version_token == "v2"  # noqa: S105
    assert len(replacement.chunks) == 1
    assert [entry.acronym for entry in replacement.glossary] == ["NOW"]


@pytest.mark.asyncio
async def test_production_deriver_expands_container_members_through_the_same_runner() -> None:
    plaintext = PlaintextParser(PlaintextConfig())
    parser = ParserChain(
        parsers={"container": OneMemberContainer(), "plaintext": plaintext},  # pyright: ignore[reportArgumentType]
        chains={"application/zip": ("container",), "text/plain": ("plaintext",)},
    )
    chunker = BlockChunker()
    installed = tuple(
        fingerprint.canonical()
        for name in ("archive", "plaintext")
        if (fingerprint := parse_fingerprint(name)) is not None
    )
    requested = RebuildTarget(
        parser_routing="container-routing-v1",
        parser_set=installed,
        chunk_fingerprint=chunker.fingerprint.canonical(),
        embedding_fingerprint="embed-v1",
        glossary_fingerprint=glossary_fingerprint().canonical(),
        fts_tokenizer="unicode61",
        batch_documents=1,
        max_memory_bytes=1_000_000,
        max_temporary_bytes=1_000_000,
    )
    raw = RawDocument(
        source_id="archive",
        uri="https://snapshot.invalid/archive.zip",
        media_type="application/zip",
        content=b"container bytes",
    )
    replacement = await ParserChunkerRelationalDeriver(
        workspace_id="default",
        source="wiki",
        parser_chain=parser,
        routing_identity="container-routing-v1",
        chunker=chunker,
        parse_runner=InProcessRunner(
            {"container": OneMemberContainer(), "plaintext": plaintext},
            middleware=MiddlewareRunner(()),
            chunker=chunker,
        ),
    ).derive(
        raw,
        requested,
        generation_id="generation-v2",
        blob_ref="retained-container",
        title="Archive",
        version_token=None,
    )
    assert len(replacement.members) == 1
    assert replacement.members[0].document.source_id == "zip:archive!/inside.txt"
    assert replacement.members[0].document.status is DocumentStatus.INDEXED
    assert replacement.members[0].chunks[0].text == "member text"


async def test_non_resource_stage_timeout_maps_to_bounded_derivation_failure(
    tmp_path: Path, manicule_environment: Path
) -> None:
    del manicule_environment
    middleware = MiddlewareRunner((HangingMiddleware(HostileConfig(hang_seconds=60)),))
    chunker = BlockChunker()
    plaintext = PlaintextParser(PlaintextConfig())
    parser = ParserChain(parsers={"plaintext": plaintext}, chains={"text/plain": ("plaintext",)})
    installed = parse_fingerprint("plaintext")
    assert installed is not None
    requested = RebuildTarget(
        parser_routing="plain-routing-v1",
        parser_set=(installed.canonical(),),
        chunk_fingerprint=chunker.fingerprint.canonical(),
        embedding_fingerprint="embed-v1",
        glossary_fingerprint=glossary_fingerprint(middleware=middleware.chain()).canonical(),
        fts_tokenizer="unicode61",
        batch_documents=1,
        max_memory_bytes=4096 * MEGABYTE,
        max_temporary_bytes=4096 * MEGABYTE,
    )
    config = WorkerConfig(
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        middleware=("hanging-stage",),
        plugin_config={"middleware.hanging-stage": {"hang_seconds": 60}},
        memory_limit_bytes=4096 * MEGABYTE,
    )
    raw = RawDocument(
        source_id="page-1",
        uri="https://snapshot.invalid/page-1",
        media_type="text/plain",
        content="source",
    )
    async with WorkerPool(config, workers=1, timeout_s=0.1, poll_interval_s=0.01) as pool:
        deriver = ParserChunkerRelationalDeriver(
            workspace_id="default",
            source="wiki",
            parser_chain=parser,
            routing_identity="plain-routing-v1",
            chunker=chunker,
            middleware=middleware,
            parse_runner=pool,
        )
        with pytest.raises(ValueError, match=RebuildRefusalCode.DERIVATION_FAILED.value):
            await deriver.derive(
                raw,
                requested,
                generation_id="generation-v2",
                blob_ref="retained",
                title="Timeout",
                version_token=None,
            )


@pytest.mark.asyncio
async def test_container_member_failure_matches_live_identity_and_parse_stage() -> None:
    parser = ParserChain(
        parsers={"container": FailureContainer()},  # pyright: ignore[reportArgumentType]
        chains={"application/zip": ("container",)},
    )
    chunker = BlockChunker()
    archive_fingerprint = parse_fingerprint("archive")
    assert archive_fingerprint is not None
    target = RebuildTarget(
        parser_routing="container-routing-v1",
        parser_set=(archive_fingerprint.canonical(),),
        chunk_fingerprint=chunker.fingerprint.canonical(),
        embedding_fingerprint="embed-v1",
        glossary_fingerprint=glossary_fingerprint().canonical(),
        fts_tokenizer="unicode61",
        batch_documents=1,
        max_memory_bytes=1_000_000,
        max_temporary_bytes=1_000_000,
    )
    raw = RawDocument(
        source_id="archive",
        uri="https://snapshot.invalid/archive.zip",
        media_type="application/zip",
        content=b"container bytes",
    )
    replacement = await ParserChunkerRelationalDeriver(
        workspace_id="default",
        source="wiki",
        parser_chain=parser,
        routing_identity="container-routing-v1",
        chunker=chunker,
        parse_runner=InProcessRunner(
            {"container": FailureContainer()},
            middleware=MiddlewareRunner(()),
            chunker=chunker,
        ),
    ).derive(
        raw,
        target,
        generation_id="generation-v2",
        blob_ref="retained-container",
        title="Archive",
        version_token=None,
    )
    failed = replacement.members[0].document
    assert failed.content_hash == content_hash(failed.uri)
    assert failed.failed_stage is PipelineStage.PARSE
    assert failed.metadata["reason"] == "member is encrypted"
    assert failed.metadata["parsers_attempted"] == []
    depth_limited = replacement.members[1].document
    assert depth_limited.status is DocumentStatus.UNSUPPORTED_MEDIA_TYPE
    assert depth_limited.failed_stage is None
    assert depth_limited.content_hash == content_hash(depth_limited.uri)
    assert depth_limited.metadata["reason"] == "container depth exceeds the configured limit"
