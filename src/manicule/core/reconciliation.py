"""Typed capabilities for durable, full-source reconciliation inventories."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReconciliationRunState(StrEnum):
    """Forward-only lifecycle of a full-source inventory."""

    ENUMERATING = "enumerating"
    COMPLETED = "completed"
    PROPOSED = "proposed"
    APPLIED = "applied"
    DRY_RUN = "dry_run"
    CANCELED = "canceled"


class CompletedInventory(BaseModel):
    """Storage-minted authority to diff one completed full enumeration.

    The redundant identity fields are intentional.  Storage revalidates all of them in the
    transaction that proposes or applies deletions, so a naked run id can never be confused
    with proof of completion or with another workspace, connector, or configured scope.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    run_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    connector_id: str = Field(min_length=1)
    connector: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    completed_at: datetime
    seen_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _completed_at_is_aware(self) -> CompletedInventory:
        if self.completed_at.tzinfo is None:
            msg = "inventory completion time must be timezone-aware"
            raise ValueError(msg)
        return self


class ReconciliationAssessment(BaseModel):
    """Bounded result: counts, not a second in-memory copy of corpus identities."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    connector: str
    scope: str
    seen_count: int = Field(ge=0)
    live_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    applied_count: int = Field(ge=0, default=0)
    refused: str = ""
    dry_run: bool = False


__all__ = [
    "CompletedInventory",
    "ReconciliationAssessment",
    "ReconciliationRunState",
]
