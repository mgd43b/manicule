"""Aggregate-safe plans and outcomes for source and derived lifecycle operations."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from manicule.core.errors import ManiculeError


class LifecycleOperation(StrEnum):
    """Operations whose retention boundaries must never be conflated."""

    RESET_DERIVED = "reset_derived"
    CLEANUP_DERIVED_GENERATIONS = "cleanup_derived_generations"
    RELEASE_SOURCE_HISTORY = "release_source_history"
    DELETE_SNAPSHOT = "delete_snapshot"


class LifecyclePlan(BaseModel):
    """Privacy-safe dry-run shared by every operator surface."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    operation: LifecycleOperation
    eligible_items: int = Field(default=0, ge=0)
    eligible_bytes: int = Field(default=0, ge=0)
    protected_items: int = Field(default=0, ge=0)
    protected_bytes: int = Field(default=0, ge=0)
    snapshot_items: int = Field(default=0, ge=0)
    unrecoverable_items: int = Field(default=0, ge=0)
    unrecoverable_bytes: int = Field(default=0, ge=0)
    confirmation: str | None = None


class LifecycleOutcome(BaseModel):
    """Settled result; retrying a completed cleanup returns zero removals."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    operation: LifecycleOperation
    removed_items: int = Field(default=0, ge=0)
    released_bytes: int = Field(default=0, ge=0)
    snapshot_items: int = Field(default=0, ge=0)
    source_contacted: bool = False
    documents_retired: int = Field(default=0, ge=0)
    chunks_removed: int = Field(default=0, ge=0)
    memberships_removed: int = Field(default=0, ge=0)
    vector_rows_removed: int = Field(default=0, ge=0)
    publications_removed: int = Field(default=0, ge=0)
    generations_terminalized: int = Field(default=0, ge=0)
    vector_store_removed: bool = False
    fingerprints_cleared: bool = False
    runtime_cache_invalidated: bool = False


class LifecycleRefusalError(ManiculeError):
    """A lifecycle boundary refused before it changed authoritative state."""


__all__ = [
    "LifecycleOperation",
    "LifecycleOutcome",
    "LifecyclePlan",
    "LifecycleRefusalError",
]
