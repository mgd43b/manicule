from __future__ import annotations

import asyncio
import errno
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, cast, override

import pytest
from manicule_plugin_hostile import HangingMiddleware, HostileConfig
from sqlalchemy.exc import IntegrityError, OperationalError

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
    RebuildPublicationConflictError,
    RebuildPublicationValidationError,
    RebuildRefusalCode,
    RebuildRefusedError,
    RebuildState,
    RebuildStorageCause,
    RebuildStorageDiagnostic,
    RebuildStorageError,
    RebuildStorageStage,
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
    from collections.abc import AsyncIterator, Callable, Sequence
    from pathlib import Path


class _RebuildDiagnosticLogRecord(Protocol):
    rebuild_storage: dict[str, Any]


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
    released_code: RebuildRefusalCode | None = None

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
        clock: Callable[[], object] | None = None,
    ) -> None:
        del generation_id, source_publication_id, owner, lease_generation, now, cancel, clock

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

    async def validate_generation(
        self,
        generation_id: str,
        *,
        owner: str,
        lease_generation: int,
        now: object,
        cancel: asyncio.Event | None = None,
        clock: Callable[[], object] | None = None,
    ) -> None:
        del owner, lease_generation, now, cancel, clock
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

    async def release_generation(
        self,
        generation_id: str,
        code: RebuildRefusalCode,
        *,
        owner: str,
        lease_generation: int,
        now: object,
    ) -> RebuildCheckpoint:
        del generation_id, owner, lease_generation, now
        self.released_code = code
        # Deliberately not a terminal state: the point of releasing is that the generation is
        # still there to be claimed.
        return self._checkpoint()


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
        connector: str | None = None,
    ) -> PreparedReplacement:
        del blob_ref, title, version_token, connector
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


class PublicationRuntimeFailureStore(FakeStore):
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
        raise RuntimeError(
            "replacement vectors incomplete at /private/path for "
            "https://wiki.example.test?cookie=secret"
        )


class PublicationConflictStore(FakeStore):
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
        raise RebuildPublicationConflictError(RebuildRefusalCode.SNAPSHOT_CHANGED)


class ResumeVectorValidationStore(FakeStore):
    @override
    async def claim_generation(
        self, generation_id: str, owner: str, *, now: object, expires_at: object
    ) -> RebuildCheckpoint:
        checkpoint = await super().claim_generation(
            generation_id, owner, now=now, expires_at=expires_at
        )
        return checkpoint.model_copy(
            update={"predecessor_vector_publication_id": "private-predecessor"}
        )

    @override
    async def copy_checkpointed_vectors(
        self,
        generation_id: str,
        source_publication_id: str,
        *,
        owner: str,
        lease_generation: int,
        now: object,
        cancel: asyncio.Event | None = None,
        clock: Callable[[], object] | None = None,
    ) -> None:
        del generation_id, source_publication_id, owner, lease_generation, now, cancel, clock
        raise RebuildPublicationValidationError


class ResumeMemoryBoundStore(ResumeVectorValidationStore):
    @override
    async def copy_checkpointed_vectors(
        self,
        generation_id: str,
        source_publication_id: str,
        *,
        owner: str,
        lease_generation: int,
        now: object,
        cancel: asyncio.Event | None = None,
        clock: Callable[[], object] | None = None,
    ) -> None:
        del generation_id, source_publication_id, owner, lease_generation, now, cancel, clock
        raise RebuildPublicationValidationError(RebuildRefusalCode.MEMORY_BOUND)


class ManifestChangedStore(FakeStore):
    @override
    async def snapshot_inputs(
        self, generation_id: str, *, after_sequence: int, limit: int
    ) -> list[SnapshotRebuildInput]:
        del generation_id, after_sequence, limit
        raise RebuildPublicationConflictError(RebuildRefusalCode.SNAPSHOT_CHANGED)


class NoncontiguousManifestStore(FakeStore):
    @override
    async def snapshot_inputs(
        self, generation_id: str, *, after_sequence: int, limit: int
    ) -> list[SnapshotRebuildInput]:
        items = list(
            await super().snapshot_inputs(generation_id, after_sequence=after_sequence, limit=limit)
        )
        return [items[0].model_copy(update={"sequence": items[0].sequence + 1})]


class CorruptCheckpointStore(FakeStore):
    @override
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
        del generation_id, replacements, expected_next_sequence, owner, lease_generation, now
        raise RebuildPublicationValidationError


class PublicationLeaseLossStore(FakeStore):
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
        raise RebuildLeaseConflictError("private replacement owner")


@dataclass
class AssertLeaseSeamStore(FakeStore):
    conflict_at: int = 3
    true_lease_loss: bool = False
    successor_took_over_before_fail: bool = False
    assert_calls: int = 0

    @override
    async def assert_generation_lease(
        self,
        generation_id: str,
        owner: str,
        lease_generation: int,
        *,
        now: object,
    ) -> None:
        del generation_id, owner, lease_generation, now
        self.assert_calls += 1
        if self.assert_calls != self.conflict_at:
            return
        if self.true_lease_loss:
            raise RebuildLeaseConflictError("private successor owner")
        raise RebuildPublicationConflictError(RebuildRefusalCode.WORKSPACE_SCOPE_CHANGED)

    @override
    async def fail_generation(
        self,
        generation_id: str,
        code: RebuildRefusalCode,
        *,
        owner: str,
        lease_generation: int,
        now: object,
    ) -> RebuildCheckpoint:
        if self.successor_took_over_before_fail:
            raise RebuildLeaseConflictError("private successor owner")
        return await super().fail_generation(
            generation_id,
            code,
            owner=owner,
            lease_generation=lease_generation,
            now=now,
        )


class ValidationFailureStore(FakeStore):
    @override
    async def validate_generation(
        self,
        generation_id: str,
        *,
        owner: str,
        lease_generation: int,
        now: object,
        cancel: asyncio.Event | None = None,
        clock: Callable[[], object] | None = None,
    ) -> None:
        del generation_id, owner, lease_generation, now, cancel, clock
        raise RuntimeError("invalid row https://wiki.example.test/private token=secret")


class ValidationConflictStore(FakeStore):
    @override
    async def validate_generation(
        self,
        generation_id: str,
        *,
        owner: str,
        lease_generation: int,
        now: object,
        cancel: asyncio.Event | None = None,
        clock: Callable[[], object] | None = None,
    ) -> None:
        del generation_id, owner, lease_generation, now, cancel, clock
        raise RebuildPublicationConflictError(RebuildRefusalCode.SNAPSHOT_CHANGED)


class ValidationStorageFailureStore(FakeStore):
    @override
    async def validate_generation(
        self,
        generation_id: str,
        *,
        owner: str,
        lease_generation: int,
        now: object,
        cancel: asyncio.Event | None = None,
        clock: Callable[[], object] | None = None,
    ) -> None:
        del generation_id, owner, lease_generation, now, cancel, clock
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
        connector: str | None = None,
    ) -> PreparedReplacement:
        del raw, target, generation_id, blob_ref, title, version_token, connector
        raise ParseError("secret body at /private/path from https://wiki.example.test")


class MidBuildStorageFailureStore(FakeStore):
    """A driver failure from the middle of the build, which nothing along the way wrapped.

    Publication failures were always settled. This one is raised by an ordinary checkpoint
    write, which is where a locked, corrupted or full database actually shows up.
    """

    @override
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
        del generation_id, replacements, expected_next_sequence, owner, lease_generation, now
        raise IntegrityError(
            "UPDATE private_table SET body = ? WHERE id = ?",
            ("https://wiki.example.test/private?cookie=secret",),
            RuntimeError("/private/machine/workspace.sqlite"),
        )


class UnsettleableStore(MidBuildStorageFailureStore):
    """The store cannot be told about the failure, because the lease is already gone."""

    @override
    async def release_generation(
        self,
        generation_id: str,
        code: RebuildRefusalCode,
        *,
        owner: str,
        lease_generation: int,
        now: object,
    ) -> RebuildCheckpoint:
        del generation_id, code, owner, lease_generation, now
        raise RebuildLeaseConflictError("generation lease changed or expired")


class _BusySqliteError(sqlite3.OperationalError):
    sqlite_errorcode = sqlite3.SQLITE_BUSY


@dataclass
class RetryableValidationStore(FakeStore):
    validation_attempts: int = 0
    storage_diagnostics: list[RebuildStorageDiagnostic] = field(
        default_factory=list[RebuildStorageDiagnostic]
    )

    @override
    async def validate_generation(
        self,
        generation_id: str,
        *,
        owner: str,
        lease_generation: int,
        now: object,
        cancel: asyncio.Event | None = None,
        clock: Callable[[], object] | None = None,
    ) -> None:
        await super().validate_generation(
            generation_id,
            owner=owner,
            lease_generation=lease_generation,
            now=now,
            cancel=cancel,
            clock=clock,
        )
        self.validation_attempts += 1
        if self.validation_attempts == 1:
            raise OperationalError(
                "SELECT private_body FROM private_table WHERE credential = ?",
                ("https://example.test/private?token=secret",),
                _BusySqliteError("database is locked at /private/storage.sqlite"),
            )

    async def record_storage_diagnostic(
        self,
        generation_id: str,
        diagnostic: RebuildStorageDiagnostic,
        *,
        owner: str,
        lease_generation: int,
        now: object,
    ) -> None:
        del generation_id, owner, lease_generation, now
        self.storage_diagnostics.append(diagnostic)


@dataclass
class CapacityValidationStore(RetryableValidationStore):
    @override
    async def validate_generation(
        self,
        generation_id: str,
        *,
        owner: str,
        lease_generation: int,
        now: object,
        cancel: asyncio.Event | None = None,
        clock: Callable[[], object] | None = None,
    ) -> None:
        del generation_id, owner, lease_generation, now, cancel, clock
        self.validation_attempts += 1
        raise OSError(errno.ENOSPC, "no space at /private/storage.sqlite token=secret")


@pytest.mark.asyncio
async def test_a_storage_failure_mid_build_gives_the_generation_up_without_ending_it() -> None:
    """What the run leaves behind is the whole point, not what it raises.

    A driver failure from a checkpoint write used to unwind past every handler in the build
    and out through the typed boundary, which reported it correctly and settled nothing. The
    durable row stayed ``building`` with the counters it had reached, an owner that no longer
    existed and a lease that expired minutes later, so status went on describing a rebuild as
    running with no worker, no diagnostic, and no way to tell the difference from here.

    Released rather than failed, and the difference is thousands of documents. A writer that
    held SQLite past the busy timeout says nothing about the work already committed, and
    `fail_generation` is terminal — `claim_generation` refuses a failed generation forever, so
    ending one over contention means cleanup and deriving the whole corpus again. This records
    why the attempt stopped, drops the lease, and leaves the generation there to be taken over.
    """
    item, body = source(0, "private source body")
    store = MidBuildStorageFailureStore([item])

    with pytest.raises(RebuildStorageError) as caught:
        await OfflineGenerationRebuilder(
            store=store,
            blobs=FakeBlobs({item.blob_ref: body}),
            deriver=FakeDeriver(),
        ).run("promoted-run", target())

    assert store.released_code is RebuildRefusalCode.STORAGE_FAILED, (
        "a claimed generation the worker is abandoning has to carry why"
    )
    assert store.failed_code is None, (
        "contention is not a diagnosis of the work, so it must not end the generation"
    )
    assert store.published is False
    rendered = str(caught.value).lower()
    for private in ("update", "secret", "wiki.example.test", "/private", "sqlite"):
        assert private not in rendered


@pytest.mark.asyncio
async def test_a_settlement_the_lease_no_longer_permits_does_not_rewrite_the_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Losing the lease is exactly the loss of the right to write the row.

    If the lease lapsed while this worker was inside the failing call, another owner may hold
    the generation now, and that owner alone decides what it says. So the settlement is an
    attempt rather than a guarantee: it is dropped, the storage failure the caller has to act
    on survives unchanged, and the unsettled row is left to lease recovery — which is a state
    the surfaces can describe, because an unowned generation no longer reads as running.
    """
    item, body = source(0, "private source body")
    store = UnsettleableStore([item])

    with pytest.raises(RebuildStorageError):
        await OfflineGenerationRebuilder(
            store=store,
            blobs=FakeBlobs({item.blob_ref: body}),
            deriver=FakeDeriver(),
        ).run("promoted-run", target())

    assert store.released_code is None, "the store refused the settlement, so nothing recorded it"
    assert store.failed_code is None
    assert store.published is False
    diagnostics: list[dict[str, Any]] = [
        cast("_RebuildDiagnosticLogRecord", record).rebuild_storage
        for record in caplog.records
        if hasattr(record, "rebuild_storage")
    ]
    assert [diagnostic["event"] for diagnostic in diagnostics] == ["primary", "settlement"]
    assert diagnostics[0]["correlation_id"] == diagnostics[1]["correlation_id"]
    assert diagnostics[1]["stage"] == RebuildStorageStage.RELEASE.value
    assert not any(
        marker in str(diagnostics).lower()
        for marker in ("private", "secret", "sqlite", "example.test")
    )


@pytest.mark.asyncio
async def test_publication_integrity_failure_is_bounded_and_gives_the_generation_up() -> None:
    item, body = source(0, "private source body")
    store = StorageFailureStore([item])

    with pytest.raises(RebuildStorageError) as caught:
        await OfflineGenerationRebuilder(
            store=store,
            blobs=FakeBlobs({item.blob_ref: body}),
            deriver=FakeDeriver(),
        ).run("promoted-run", target())

    assert str(caught.value) == "offline rebuild storage failed"
    assert store.released_code is RebuildRefusalCode.STORAGE_FAILED, (
        "publication is one transaction, so a rolled-back one leaves a generation worth keeping"
    )
    assert store.failed_code is None
    assert store.published is False
    rendered = str(caught.value).lower()
    for private in ("insert", "secret", "wiki.example.test", "/private", "sqlite"):
        assert private not in rendered


@pytest.mark.asyncio
async def test_publication_runtime_failure_is_bounded_and_marks_generation_failed() -> None:
    item, body = source(0, "private source body")
    store = PublicationRuntimeFailureStore([item])

    with pytest.raises(RebuildValidationError) as caught:
        await OfflineGenerationRebuilder(
            store=store,
            blobs=FakeBlobs({item.blob_ref: body}),
            deriver=FakeDeriver(),
        ).run("promoted-run", target())

    assert store.failed_code is RebuildRefusalCode.INVALID_REPLACEMENT
    assert store.published is False
    rendered = str(caught.value).lower()
    for private in ("incomplete", "secret", "wiki.example.test", "/private"):
        assert private not in rendered


@pytest.mark.asyncio
async def test_publication_snapshot_conflict_is_bounded_and_marks_generation_failed() -> None:
    item, body = source(0, "body")
    store = PublicationConflictStore([item])

    with pytest.raises(RebuildLeaseError, match="offline rebuild lease was lost"):
        await OfflineGenerationRebuilder(
            store=store,
            blobs=FakeBlobs({item.blob_ref: body}),
            deriver=FakeDeriver(),
        ).run("promoted-run", target())

    assert store.failed_code is RebuildRefusalCode.SNAPSHOT_CHANGED
    assert store.published is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("store_type", "expected_code"),
    [
        (ResumeVectorValidationStore, RebuildRefusalCode.INVALID_REPLACEMENT),
        (ResumeMemoryBoundStore, RebuildRefusalCode.MEMORY_BOUND),
    ],
)
async def test_corrupt_resume_vectors_are_bounded_and_mark_generation_failed(
    store_type: type[FakeStore], expected_code: RebuildRefusalCode
) -> None:
    item, body = source(0, "body")
    store = store_type([item])

    with pytest.raises(RebuildValidationError, match="offline rebuild validation failed"):
        await OfflineGenerationRebuilder(
            store=store,
            blobs=FakeBlobs({item.blob_ref: body}),
            deriver=FakeDeriver(),
        ).run("promoted-run", target())

    assert store.failed_code is expected_code
    assert store.published is False


@pytest.mark.asyncio
@pytest.mark.parametrize("store_type", [ManifestChangedStore, NoncontiguousManifestStore])
async def test_changed_resume_manifest_is_bounded_and_marks_generation_failed(
    store_type: type[FakeStore],
) -> None:
    item, body = source(0, "body")
    store = store_type([item])

    with pytest.raises(RebuildLeaseError, match="offline rebuild lease was lost"):
        await OfflineGenerationRebuilder(
            store=store,
            blobs=FakeBlobs({item.blob_ref: body}),
            deriver=FakeDeriver(),
        ).run("promoted-run", target())

    assert store.failed_code is RebuildRefusalCode.SNAPSHOT_CHANGED
    assert store.published is False


@pytest.mark.asyncio
@pytest.mark.parametrize("conflict_at", [3, 4], ids=["pre-validation", "pre-publication"])
async def test_owned_assert_lease_conflict_fails_the_generation_at_both_seams(
    conflict_at: int,
) -> None:
    item, body = source(0, "body")
    store = AssertLeaseSeamStore([item], conflict_at=conflict_at)

    with pytest.raises(RebuildLeaseError, match="offline rebuild lease was lost"):
        await OfflineGenerationRebuilder(
            store=store,
            blobs=FakeBlobs({item.blob_ref: body}),
            deriver=FakeDeriver(),
        ).run("promoted-run", target())

    assert store.failed_code is RebuildRefusalCode.WORKSPACE_SCOPE_CHANGED
    assert store.published is False


@pytest.mark.asyncio
@pytest.mark.parametrize("conflict_at", [3, 4], ids=["pre-validation", "pre-publication"])
async def test_true_assert_lease_loss_never_mutates_successor_state(conflict_at: int) -> None:
    item, body = source(0, "body")
    store = AssertLeaseSeamStore([item], conflict_at=conflict_at, true_lease_loss=True)

    with pytest.raises(RebuildLeaseError, match="offline rebuild lease was lost"):
        await OfflineGenerationRebuilder(
            store=store,
            blobs=FakeBlobs({item.blob_ref: body}),
            deriver=FakeDeriver(),
        ).run("promoted-run", target())

    assert store.failed_code is None
    assert store.published is False


@pytest.mark.asyncio
async def test_successor_takeover_between_owned_conflict_and_failure_never_mutates_it() -> None:
    item, body = source(0, "body")
    store = AssertLeaseSeamStore([item], conflict_at=3, successor_took_over_before_fail=True)

    with pytest.raises(RebuildLeaseError, match="offline rebuild lease was lost"):
        await OfflineGenerationRebuilder(
            store=store,
            blobs=FakeBlobs({item.blob_ref: body}),
            deriver=FakeDeriver(),
        ).run("promoted-run", target())

    assert store.failed_code is None
    assert store.published is False


@pytest.mark.asyncio
async def test_corrupt_checkpoint_is_bounded_and_marks_generation_failed() -> None:
    item, body = source(0, "body")
    store = CorruptCheckpointStore([item])

    with pytest.raises(RebuildValidationError, match="offline rebuild validation failed"):
        await OfflineGenerationRebuilder(
            store=store,
            blobs=FakeBlobs({item.blob_ref: body}),
            deriver=FakeDeriver(),
        ).run("promoted-run", target())

    assert store.failed_code is RebuildRefusalCode.INVALID_REPLACEMENT
    assert store.published is False


@pytest.mark.asyncio
async def test_publication_lease_loss_does_not_fail_a_generation_owned_by_another_worker() -> None:
    item, body = source(0, "body")
    store = PublicationLeaseLossStore([item])

    with pytest.raises(RebuildLeaseError, match="offline rebuild lease was lost"):
        await OfflineGenerationRebuilder(
            store=store,
            blobs=FakeBlobs({item.blob_ref: body}),
            deriver=FakeDeriver(),
        ).run("promoted-run", target())

    assert store.failed_code is None
    assert store.published is False


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
async def test_validation_snapshot_conflict_is_bounded_and_marks_generation_failed() -> None:
    item, body = source(0, "body")
    store = ValidationConflictStore([item])

    with pytest.raises(RebuildLeaseError, match="offline rebuild lease was lost"):
        await OfflineGenerationRebuilder(
            store=store,
            blobs=FakeBlobs({item.blob_ref: body}),
            deriver=FakeDeriver(),
        ).run("promoted-run", target())

    assert store.failed_code is RebuildRefusalCode.SNAPSHOT_CHANGED
    assert store.published is False


@pytest.mark.asyncio
async def test_validation_storage_failure_is_bounded_and_gives_the_generation_up() -> None:
    """A storage failure at validation is the same class as one mid-build, and settles the same.

    Validation reads what is already staged. A driver failure there is a statement about the
    database, not about the generation, so it releases rather than ends it — while a validation
    failure that actually finds bad output still marks the generation failed, because retrying
    that would produce the same bad output.
    """
    item, body = source(0, "body")
    store = ValidationStorageFailureStore([item])

    with pytest.raises(RebuildStorageError, match="offline rebuild storage failed"):
        await OfflineGenerationRebuilder(
            store=store,
            blobs=FakeBlobs({item.blob_ref: body}),
            deriver=FakeDeriver(),
        ).run("promoted-run", target())

    assert store.released_code is RebuildRefusalCode.STORAGE_FAILED
    assert store.failed_code is None
    assert store.published is False


@pytest.mark.asyncio
async def test_busy_validation_retries_under_the_same_lease_without_rebuilding() -> None:
    item, body = source(0, "body")
    store = RetryableValidationStore([item])

    result = await OfflineGenerationRebuilder(
        store=store,
        blobs=FakeBlobs({item.blob_ref: body}),
        deriver=FakeDeriver(),
    ).run("promoted-run", target())

    assert result.state is RebuildState.PUBLISHED
    assert store.validation_attempts == 2
    assert store.stage_calls == 1, "the retry must start from durable validation evidence"
    assert store.released_code is None
    assert len(store.storage_diagnostics) == 1
    diagnostic = store.storage_diagnostics[0]
    assert diagnostic.stage is RebuildStorageStage.VALIDATION_EVIDENCE_READ
    assert diagnostic.cause is RebuildStorageCause.BUSY
    assert diagnostic.retryable is True
    assert diagnostic.namespace_usable is True
    assert diagnostic.next_retry_at is not None
    rendered = str(diagnostic.model_dump(mode="json")).lower()
    assert not any(value in rendered for value in ("private", "secret", "sqlite", "example.test"))


@pytest.mark.asyncio
async def test_capacity_failure_never_enters_an_automatic_restart_loop() -> None:
    item, body = source(0, "body")
    store = CapacityValidationStore([item])

    with pytest.raises(RebuildStorageError) as caught:
        await OfflineGenerationRebuilder(
            store=store,
            blobs=FakeBlobs({item.blob_ref: body}),
            deriver=FakeDeriver(),
        ).run("promoted-run", target())

    assert store.validation_attempts == 1
    assert store.released_code is RebuildRefusalCode.STORAGE_FAILED
    diagnostic = caught.value.diagnostic
    assert diagnostic is not None
    assert diagnostic.stage is RebuildStorageStage.VALIDATION_EVIDENCE_READ
    assert diagnostic.cause is RebuildStorageCause.CAPACITY
    assert diagnostic.retryable is False
    assert diagnostic.next_retry_at is None
    rendered = str(caught.value).lower()
    assert not any(value in rendered for value in ("private", "secret", "sqlite", "example.test"))


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
            RebuildPublicationConflictError(RebuildRefusalCode.SNAPSHOT_CHANGED),
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
    assert store.failed_code is None
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
async def test_heartbeat_storage_error_does_not_hide_committed_atomic_publication() -> None:
    """A second SQLite connection can be busy behind the publication writer lock."""

    @dataclass
    class PublicationContentionStore(FakeStore):
        renewal_attempted: asyncio.Event = field(default_factory=asyncio.Event)

        @override
        async def renew_generation(
            self,
            generation_id: str,
            owner: str,
            lease_generation: int,
            *,
            now: object,
            expires_at: object,
        ) -> RebuildCheckpoint:
            if self.publish_started is not None and self.publish_started.is_set():
                self.renewal_attempted.set()
                raise OSError(errno.EAGAIN, "publication holds the writer lock")
            return await super().renew_generation(
                generation_id,
                owner,
                lease_generation,
                now=now,
                expires_at=expires_at,
            )

        @override
        async def publish_generation(
            self,
            generation_id: str,
            *,
            owner: str,
            lease_generation: int,
            now: object,
        ) -> RebuildCheckpoint:
            if self.publish_started is not None:
                self.publish_started.set()
            await asyncio.wait_for(self.renewal_attempted.wait(), timeout=5)
            return await super().publish_generation(
                generation_id,
                owner=owner,
                lease_generation=lease_generation,
                now=now,
            )

    item, body = source(0, "body")
    store = PublicationContentionStore([item], publish_started=asyncio.Event())
    result = await OfflineGenerationRebuilder(
        store=store, blobs=FakeBlobs({item.blob_ref: body}), deriver=FakeDeriver()
    ).run("promoted-run", target(), lease_seconds=1)

    assert result.state is RebuildState.PUBLISHED
    assert store.published
    assert store.released_code is None


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


# --- checkpoint replay holds its own lease ----------------------------------------------------


@dataclass
class ReplayingStore(FakeStore):
    """A takeover whose checkpoint replay outlives the lease it was claimed under.

    Replay blocks until the heartbeat has renewed ``renewals_wanted`` times, so the test asserts
    on a *count* rather than on elapsed time: a build that cannot renew does not make this flaky,
    it makes it hang and fail. That is the same shape as the acquisition heartbeat's coverage,
    and for the same reason — a wall-clock barrier turns a scheduling accident into a green run.
    """

    renewals_wanted: int = 2
    renewals: int = 0
    expiries: list[datetime] = field(default_factory=list[datetime])
    replay_renewals: int = 0
    replay_expiries: list[datetime] = field(default_factory=list[datetime])
    """Renewals seen while replay was running, apart from the document loop's own.

    Scoped deliberately: the loop below renews twice per document, so a total would report a
    healthy heartbeat for a build that never renewed during replay at all — which is exactly
    the defect.
    """
    replay_finished: bool = False
    renewal_error: Exception | None = None
    renewed_enough: asyncio.Event = field(default_factory=asyncio.Event)
    replay_started: asyncio.Event = field(default_factory=asyncio.Event)

    @override
    async def claim_generation(
        self, generation_id: str, owner: str, *, now: object, expires_at: object
    ) -> RebuildCheckpoint:
        checkpoint = await super().claim_generation(
            generation_id, owner, now=now, expires_at=expires_at
        )
        return checkpoint.model_copy(
            update={"predecessor_vector_publication_id": "predecessor-publication"}
        )

    @override
    async def renew_generation(
        self,
        generation_id: str,
        owner: str,
        lease_generation: int,
        *,
        now: object,
        expires_at: object,
    ) -> RebuildCheckpoint:
        if self.renewal_error is not None:
            raise self.renewal_error
        self.renewals += 1
        if isinstance(expires_at, datetime):
            self.expiries.append(expires_at)
        if self.renewals >= self.renewals_wanted:
            self.renewed_enough.set()
        return await super().renew_generation(
            generation_id, owner, lease_generation, now=now, expires_at=expires_at
        )

    @override
    async def copy_checkpointed_vectors(
        self,
        generation_id: str,
        source_publication_id: str,
        *,
        owner: str,
        lease_generation: int,
        now: object,
        cancel: asyncio.Event | None = None,
        clock: Callable[[], object] | None = None,
    ) -> None:
        del generation_id, source_publication_id, owner, lease_generation, now, clock
        self.replay_started.set()
        waiters = [asyncio.ensure_future(self.renewed_enough.wait())]
        if cancel is not None:
            waiters.append(asyncio.ensure_future(cancel.wait()))
        try:
            # Bounded, and the bound is not a timing assumption. A working heartbeat renews
            # every third of a one-second lease, so it clears this in well under a second; the
            # margin exists only so that a build which renews *never* fails the suite in ten
            # seconds instead of hanging it forever, which is what an unbounded wait on an
            # event nothing will set would do to CI.
            await asyncio.wait(waiters, timeout=10, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for waiter in waiters:
                waiter.cancel()
        if not self.renewed_enough.is_set() and (cancel is None or not cancel.is_set()):
            msg = "replay waited for renewals that never came"
            raise AssertionError(msg)
        if cancel is not None and cancel.is_set():
            raise asyncio.CancelledError
        self.replay_renewals = self.renewals
        self.replay_expiries = list(self.expiries)
        self.replay_finished = True


def _replaying(*, renewals_wanted: int) -> tuple[ReplayingStore, OfflineGenerationRebuilder]:
    """A takeover rebuilder over one synthetic document, ready to replay."""
    item, body = source(0, "alpha\nbeta")
    store = ReplayingStore(items=[item], renewals_wanted=renewals_wanted)
    rebuilder = OfflineGenerationRebuilder(
        store=store,
        blobs=FakeBlobs({item.blob_ref: body}),
        deriver=FakeDeriver(),
    )
    return store, rebuilder


async def test_replay_renews_its_lease_for_as_long_as_replay_takes() -> None:
    """The defect: a takeover replaying a large checkpoint outlived its own lease.

    The expiry was fixed when the generation was claimed and nothing moved it while replay ran,
    so a checkpoint big enough to take longer than one lease could never be replayed — and every
    retry started another full replay under another finite, unrenewed lease. No competing worker
    is involved anywhere in this test; the worker lost a lease nobody else wanted.
    """
    store, rebuilder = _replaying(renewals_wanted=2)

    checkpoint = await rebuilder.run("promoted-run", target(), lease_seconds=1)

    assert store.replay_finished, "replay has to finish rather than lose its lease part way"
    assert store.replay_renewals >= 2, (
        "a replay spanning more than one lease has to renew more than once"
    )
    assert checkpoint.state is RebuildState.PUBLISHED


async def test_one_transient_busy_heartbeat_round_does_not_cancel_replay() -> None:
    @dataclass
    class TransientBusyStore(ReplayingStore):
        busy_rounds: int = 1

        @override
        async def renew_generation(
            self,
            generation_id: str,
            owner: str,
            lease_generation: int,
            *,
            now: object,
            expires_at: object,
        ) -> RebuildCheckpoint:
            if self.busy_rounds:
                self.busy_rounds -= 1
                raise OSError(errno.EAGAIN, "synthetic transient writer contention")
            return await super().renew_generation(
                generation_id,
                owner,
                lease_generation,
                now=now,
                expires_at=expires_at,
            )

    item, body = source(0, "alpha\nbeta")
    store = TransientBusyStore(items=[item], renewals_wanted=1)
    checkpoint = await OfflineGenerationRebuilder(
        store=store,
        blobs=FakeBlobs({item.blob_ref: body}),
        deriver=FakeDeriver(),
    ).run("promoted-run", target(), lease_seconds=1)

    assert store.busy_rounds == 0
    assert store.replay_finished
    assert checkpoint.state is RebuildState.PUBLISHED


async def test_the_durable_expiry_advances_while_owner_and_generation_stay_put() -> None:
    """Renewing is not fencing: the expiry moves and the identity behind it does not.

    A renewal that changed the owner or the lease generation would be a takeover wearing a
    heartbeat's clothes, and would make every fenced assertion downstream meaningless.
    """
    store, rebuilder = _replaying(renewals_wanted=3)

    await rebuilder.run("promoted-run", target(), lease_seconds=1)

    assert len(store.replay_expiries) >= 3
    assert store.replay_expiries == sorted(store.replay_expiries), (
        "each renewal must move the expiry forward"
    )
    assert store.replay_expiries[-1] > store.replay_expiries[0]


async def test_a_takeover_during_replay_stops_the_superseded_worker() -> None:
    """The case fencing exists for, and the one a storage-only except list would miss.

    A real takeover increments the lease generation, so the renewal is refused with
    ``RebuildLeaseConflictError`` — a ``RuntimeError`` subclass, not a storage failure. The
    heartbeat has to recognize it, stop the worker, and let the surface report a lost lease
    rather than leaving a superseded worker copying into a namespace it no longer owns.
    """
    store, rebuilder = _replaying(renewals_wanted=99)
    store.renewal_error = RebuildLeaseConflictError("generation lease changed or expired")

    with pytest.raises(RebuildLeaseError):
        await rebuilder.run("promoted-run", target(), lease_seconds=1)

    assert not store.replay_finished, "a superseded worker must not finish replaying"
    assert not store.published, "and must not publish"


async def test_a_renewal_failure_is_reported_safely_and_leaves_the_generation_resumable() -> None:
    """A storage failure under the heartbeat is diagnosable without leaking its driver text.

    The failed renewal is distinct from an actual successor takeover.  It releases the still
    resumable generation with a bounded `lease_renewal` storage diagnosis; only a true ownership
    conflict crosses as `RebuildLeaseError`.
    """
    store, rebuilder = _replaying(renewals_wanted=99)
    store.renewal_error = OSError("/private/data/manicule.db is unreadable")

    with pytest.raises(RebuildStorageError) as caught:
        await rebuilder.run("promoted-run", target(), lease_seconds=1)

    assert "/private" not in str(caught.value)
    assert "manicule.db" not in str(caught.value)
    assert caught.value.diagnostic is not None
    assert caught.value.diagnostic.stage is RebuildStorageStage.LEASE_RENEWAL
    assert caught.value.diagnostic.cause is RebuildStorageCause.IO
    assert store.released_code is RebuildRefusalCode.STORAGE_FAILED
    assert store.failed_code is None
    assert not store.published


async def test_canceling_during_replay_stays_a_cancellation() -> None:
    """A run canceled while its heartbeat is healthy is canceled, not fenced.

    The two arrive at the same place — a ``CancelledError`` out of replay — and mean opposite
    things. Reporting a cancellation as a lost lease would send somebody looking for a competing
    worker that never existed.
    """
    store, rebuilder = _replaying(renewals_wanted=99)
    cancel = asyncio.Event()

    async def cancel_once_replaying() -> None:
        await store.replay_started.wait()
        cancel.set()

    watcher = asyncio.ensure_future(cancel_once_replaying())
    with pytest.raises(asyncio.CancelledError):
        await rebuilder.run("promoted-run", target(), lease_seconds=1, cancel=cancel)
    await watcher

    assert not store.replay_finished
    assert not store.published, "the predecessor publication is still the certified one"


async def test_a_replay_shorter_than_the_lease_renews_nothing() -> None:
    """The heartbeat is paced by the lease, so an ordinary takeover pays for no renewals.

    Worth pinning because the cheap fix — renewing at every page — would pass every test above
    while turning a small replay into a burst of writes proportional to the checkpoint.
    """
    store, rebuilder = _replaying(renewals_wanted=0)
    store.renewed_enough.set()

    await rebuilder.run("promoted-run", target(), lease_seconds=300)

    assert store.replay_finished
    assert store.replay_renewals == 0


@dataclass
class CountingStore(FakeStore):
    """A generation with no predecessor to replay, counting only what the heartbeat renews."""

    renewals: int = 0
    renewed_enough: asyncio.Event = field(default_factory=asyncio.Event)
    renewals_wanted: int = 2

    @override
    async def renew_generation(
        self,
        generation_id: str,
        owner: str,
        lease_generation: int,
        *,
        now: object,
        expires_at: object,
    ) -> RebuildCheckpoint:
        self.renewals += 1
        if self.renewals >= self.renewals_wanted:
            self.renewed_enough.set()
        return await super().renew_generation(
            generation_id, owner, lease_generation, now=now, expires_at=expires_at
        )


@dataclass
class SlowPreparingDeriver(FakeDeriver):
    """A document whose preparation outlasts the lease it is being prepared under."""

    renewed_enough: asyncio.Event | None = None

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
        connector: str | None = None,
    ) -> PreparedReplacement:
        if self.renewed_enough is not None:
            # Bounded for the reason the replay fake gives: a build that renews never should
            # fail the suite rather than hang it. A working heartbeat clears this in a third
            # of a second.
            await asyncio.wait_for(self.renewed_enough.wait(), timeout=10)
        return await super().prepare(
            raw,
            target,
            generation_id=generation_id,
            blob_ref=blob_ref,
            title=title,
            version_token=version_token,
            connector=connector,
        )


async def test_a_document_slower_than_its_lease_still_holds_it() -> None:
    """The sibling of the replay defect, in the loop rather than before it.

    Preparation — parse, chunk, exact token counts, embedding — runs *before* the renewal that
    follows it, so the covering renewal for one document is the one that preceded it, and for
    the first document it is the claim. A single large document can therefore outlast the lease
    on its own, and the acquisition side already measured that shape: 513 seconds of preparation
    against a 300-second lease, none of five renewals fired, and the run was dead while the
    snapshot was complete and resumable.

    The heartbeat covers the whole build rather than only the replay for exactly this reason.
    """
    item, body = source(0, "alpha\nbeta")
    store = CountingStore(items=[item], renewals_wanted=2)
    deriver = SlowPreparingDeriver(renewed_enough=store.renewed_enough)

    checkpoint = await OfflineGenerationRebuilder(
        store=store,
        blobs=FakeBlobs({item.blob_ref: body}),
        deriver=deriver,
    ).run("promoted-run", target(), lease_seconds=1)

    assert store.renewals >= 2, "a document slower than its lease has to renew during preparation"
    assert checkpoint.state is RebuildState.PUBLISHED
    assert deriver.calls == ["page-0"], "and it must be prepared once, not retried"
