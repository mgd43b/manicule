"""Typed capabilities for durable, full-source reconciliation inventories."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

LAST_RECONCILE_KEY = "last_clean_reconcile_at"
"""Set only when a pass completed cleanly. A pass that raised must not advance it."""

LAST_DISCARDED_KEY = "last_discarded_reconcile_at"
"""When an operator discarded a refused proposal.

Discarding says "I looked, and the source is what is wrong". It is not a clean pass and does not
advance :data:`LAST_RECONCILE_KEY`, but it does restart the interval: otherwise the next
scheduled pass would enumerate the source again and re-propose the same deletion within the hour.
"""

PROPOSED_DELETION_KEY = "proposed_deletion"
"""Where a refused proposal waits for a human, rather than being recomputed from scratch."""


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
    "LAST_DISCARDED_KEY",
    "LAST_RECONCILE_KEY",
    "PROPOSED_DELETION_KEY",
    "CompletedInventory",
    "ReconciliationAssessment",
    "ReconciliationRunState",
]
