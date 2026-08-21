"""Vocabulary for offline, generation-safe corpus rebuilds."""

from __future__ import annotations

import hashlib
from enum import StrEnum

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from manicule.core.acquisition import AcquiredSource
from manicule.core.content import Chunk, Document
from manicule.core.errors import ManiculeError
from manicule.core.glossary import GlossaryEntry


class RebuildState(StrEnum):
    """Forward-only lifecycle of a replacement derived generation."""

    PLANNED = "planned"
    BUILDING = "building"
    VALIDATING = "validating"
    PUBLISHED = "published"
    FAILED = "failed"
    CANCELED = "canceled"


class RebuildRefusalCode(StrEnum):
    """Bounded reasons a rebuild can refuse before parsing any content."""

    SNAPSHOT_NOT_PROMOTED = "snapshot_not_promoted"
    SNAPSHOT_CHANGED = "snapshot_changed"
    WORKSPACE_SCOPE_CHANGED = "workspace_scope_changed"
    MISSING_LOCAL_INPUT = "missing_local_input"
    MEMORY_BOUND = "memory_bound"
    TEMP_DISK_BOUND = "temp_disk_bound"
    INVALID_REPLACEMENT = "invalid_replacement"
    DERIVATION_FAILED = "derivation_failed"
    STORAGE_FAILED = "storage_failed"
    PUBLICATION_CONFLICT = "publication_conflict"


class RebuildTarget(BaseModel):
    """Every producer identity and resource bound that defines a rebuild."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    parser_routing: str = Field(min_length=1)
    parser_set: tuple[str, ...]
    chunk_fingerprint: str = Field(min_length=1)
    embedding_fingerprint: str = Field(min_length=1)
    embedding_config: str = ""
    """Full serialized embedder configuration for publication; identity remains canonical."""
    glossary_fingerprint: str = Field(min_length=1)
    fts_tokenizer: str = Field(min_length=1)
    batch_documents: int = Field(default=32, gt=0, le=1024)
    max_memory_bytes: int = Field(gt=0)
    max_temporary_bytes: int = Field(gt=0)


class MissingSnapshotInput(BaseModel):
    """Aggregate-safe identity of one manifest position that cannot be rebuilt locally."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sequence: int = Field(ge=0)
    code: RebuildRefusalCode = RebuildRefusalCode.MISSING_LOCAL_INPUT


class RebuildEstimate(BaseModel):
    """Deterministic, bounded dry-run result; it contains no source text or URI."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    generation_id: str = Field(min_length=1)
    snapshot_run_id: str = Field(min_length=1)
    documents: int = Field(ge=0)
    expected_items: int = Field(ge=0)
    known_source_bytes: int = Field(ge=0)
    estimated_chunks: int = Field(ge=0)
    estimated_seconds: float = Field(ge=0)
    estimated_peak_memory_bytes: int = Field(ge=0)
    estimated_temporary_bytes: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    missing: tuple[MissingSnapshotInput, ...] = ()
    missing_truncated: bool = False
    refusal: RebuildRefusalCode | None = None
    current_chunk_fingerprint: str | None = None
    target_chunk_fingerprint: str = ""
    over_budget_chunks: int = Field(default=0, ge=0)
    max_stored_chunk_tokens: int = Field(default=0, ge=0)
    estimated_embedding_chunks: int = Field(default=0, ge=0)
    network_required: bool = False

    @property
    def runnable(self) -> bool:
        return self.missing_count == 0 and self.refusal is None


class SnapshotRebuildInput(BaseModel):
    """One immutable member of the promoted acquisition manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sequence: int = Field(ge=0)
    connector: str = ""
    blob_ref: str = Field(min_length=1)
    source: AcquiredSource
    title: str = ""
    version_token: str | None = None


class DerivedReplacement(BaseModel):
    """All relational output for one document in an unpublished generation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document: Document
    chunks: tuple[Chunk, ...] = ()
    glossary: tuple[GlossaryEntry, ...] = ()
    members: tuple[DerivedReplacement, ...] = ()
    parse_fingerprint: str | None = None
    vector_reused: int = Field(default=0, ge=0)
    vector_embedded: int = Field(default=0, ge=0)

    def validate_identity(self) -> None:
        """Refuse a mixed or internally inconsistent staged document."""
        if any(chunk.document_id != self.document.id for chunk in self.chunks):
            raise ValueError("a replacement chunk names another document")
        if tuple(chunk.position for chunk in self.chunks) != tuple(range(len(self.chunks))):
            raise ValueError("replacement chunk positions must be contiguous from zero")
        chunk_ids = {chunk.id for chunk in self.chunks}
        if any(
            entry.document_id != self.document.id or entry.chunk_id not in chunk_ids
            for entry in self.glossary
        ):
            raise ValueError("a replacement glossary entry is outside its document's chunks")
        if self.vector_reused + self.vector_embedded != len(self.chunks):
            raise ValueError("every replacement chunk must have exactly one staged vector")
        for member in self.members:
            member.validate_identity()
        flattened = self.flattened()
        document_ids = [item.document.id for item in flattened]
        chunk_ids = [chunk.id for item in flattened for chunk in item.chunks]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("container members must have distinct document identities")
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("container members must have distinct chunk identities")

    def flattened(self) -> tuple[DerivedReplacement, ...]:
        """Breadth-first durable documents represented by one snapshot member."""
        result: list[DerivedReplacement] = []
        queue: list[DerivedReplacement] = [self]
        while queue:
            current = queue.pop(0)
            result.append(current)
            queue.extend(current.members)
        return tuple(result)

    def flattened_chunks(self) -> tuple[Chunk, ...]:
        return tuple(chunk for replacement in self.flattened() for chunk in replacement.chunks)


class RebuildCheckpoint(BaseModel):
    """Durable progress returned after an idempotent batch commit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    generation_id: str
    state: RebuildState
    expected_items: int = Field(default=0, ge=0)
    next_sequence: int = Field(ge=0)
    documents_built: int = Field(ge=0)
    chunks_built: int = Field(ge=0)
    vectors_reused: int = Field(ge=0)
    vectors_embedded: int = Field(ge=0)
    lease_owner: str | None = None
    lease_generation: int = Field(default=0, ge=0)
    # Aware, because a status surface asks whether this lease is still live and a naive
    # timestamp cannot answer that without guessing a timezone.
    lease_expires_at: AwareDatetime | None = None
    # Distinct from `lease_expires_at`: only a durable replay page, validation page, staged
    # batch, or publication commit moves this, so a healthy heartbeat renewing a stalled
    # worker's lease cannot read as content progress. Retained-source evidence verification
    # does not move it — it also runs as a read-only repair pass over an already-published
    # generation, where nothing should read as new progress.
    last_progress_at: AwareDatetime | None = None
    replayed_items: int = Field(default=0, ge=0)
    replayed_vectors: int = Field(default=0, ge=0)
    validated_items: int = Field(default=0, ge=0)
    validated_vectors: int = Field(default=0, ge=0)
    fence_generation: int | None = Field(default=None, ge=1)
    diagnostic_code: RebuildRefusalCode | None = None
    diagnostic_count: int = Field(default=0, ge=0)
    predecessor_vector_publication_id: str | None = None

    @property
    def vector_publication_id(self) -> str:
        """Lease-fenced physical namespace for this invocation's vector mutations."""
        if self.lease_owner is None or self.lease_generation <= 0:
            raise ValueError("generation has no claimed vector publication")
        return vector_publication_id(self.generation_id, self.lease_owner, self.lease_generation)


def vector_publication_id(generation_id: str, owner: str, lease_generation: int) -> str:
    """Name a vector staging namespace that a takeover can never share."""
    if not owner or lease_generation <= 0:
        raise ValueError("vector publication identity requires a claimed lease")
    owner_hash = hashlib.sha256(owner.encode()).hexdigest()[:16]
    return f"{generation_id}.{lease_generation}.{owner_hash}"


class RebuildRefusedError(ManiculeError):
    """A typed refusal containing only bounded aggregate-safe diagnostics."""

    def __init__(self, code: RebuildRefusalCode, estimate: RebuildEstimate) -> None:
        super().__init__(code.value)
        self.code = code
        self.estimate = estimate


class RebuildLeaseConflictError(RuntimeError):
    """An internal rebuild worker lost or failed to acquire its monotonic fence."""


class RebuildTerminalGenerationError(RebuildLeaseConflictError):
    """An internal worker attempted to claim a terminal generation."""


class RebuildPublicationConflictError(RebuildLeaseConflictError):
    """Generation evidence changed during build, resume, validation or publication."""

    def __init__(self, code: RebuildRefusalCode) -> None:
        super().__init__(code.value)
        self.code = code


class RebuildPublicationValidationError(RuntimeError):
    """The staged replacement failed a bounded build or publication invariant."""

    def __init__(self, code: RebuildRefusalCode = RebuildRefusalCode.INVALID_REPLACEMENT) -> None:
        super().__init__(code.value)
        self.code = code


class RebuildOperationError(ManiculeError):
    """A private-safe expected failure of durable rebuild work.

    Concrete subclasses deliberately carry no arbitrary exception text. Storage drivers,
    parsers and vector backends routinely include SQL parameters, source identifiers or local
    paths in their messages; the original exception remains available through ``__cause__`` for
    operator diagnostics, while every unattended surface receives this bounded vocabulary.
    """


class RebuildStorageError(RebuildOperationError):
    """Durable rebuild storage could not complete an expected operation."""

    def __init__(self) -> None:
        super().__init__("offline rebuild storage failed")


class RebuildDerivationError(RebuildOperationError):
    """A retained input could not be converted into a replacement."""

    def __init__(self) -> None:
        super().__init__("offline rebuild derivation failed")


class RebuildLeaseError(RebuildOperationError):
    """The worker no longer owns the generation's durable fence."""

    def __init__(self) -> None:
        super().__init__("offline rebuild lease was lost")


class RebuildValidationError(RebuildOperationError):
    """The complete staged replacement did not pass publication validation."""

    def __init__(self) -> None:
        super().__init__("offline rebuild validation failed")


class RebuildTerminalError(RebuildOperationError):
    """Execution was requested for a failed or canceled generation."""

    def __init__(self) -> None:
        super().__init__("offline rebuild generation is terminal")


__all__ = [
    "DerivedReplacement",
    "MissingSnapshotInput",
    "RebuildCheckpoint",
    "RebuildDerivationError",
    "RebuildEstimate",
    "RebuildLeaseConflictError",
    "RebuildLeaseError",
    "RebuildOperationError",
    "RebuildPublicationConflictError",
    "RebuildPublicationValidationError",
    "RebuildRefusalCode",
    "RebuildRefusedError",
    "RebuildState",
    "RebuildStorageError",
    "RebuildTarget",
    "RebuildTerminalError",
    "RebuildTerminalGenerationError",
    "RebuildValidationError",
    "SnapshotRebuildInput",
]
