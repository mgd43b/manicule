"""Retrieval vocabulary: queries, filters, candidates and assembled context.

A retrieval pipeline is a declared list of stages, each taking candidates and returning
candidates. That uniformity is what lets the evaluation harness compare whole pipelines by
configuration instead of by editing code — which in turn is what makes "no retrieval
feature without a measured improvement" enforceable rather than a discipline to remember.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from manicule.core.content import BlockKind, Chunk, Metadata


class RetrievalProfile(StrEnum):
    """Named cost/quality settings, selectable per query."""

    FAST = "fast"
    BALANCED = "balanced"
    PRECISE = "precise"


class Filter(BaseModel):
    """A restriction on which chunks a search may return.

    Every field is a conjunct; within a field, membership is a disjunction. An unset field
    imposes no restriction.

    .. note::

       Provisional. The exact split between predicates pushed into the vector store and
       those applied as a metadata pre-filter wants contact with real data volumes, so the
       shape here is the conservative superset the relational model already supports.
       :attr:`extra` exists so a store can accept a predicate this type cannot yet express
       without anyone widening it prematurely.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str | None = Field(
        default=None,
        description="Tenant scope. Applied on every query when team mode is on; a query "
        "that forgets it is a cross-tenant leak, so stores must treat it as mandatory "
        "rather than optional in that mode.",
    )
    source: str | None = None
    document_ids: frozenset[str] = frozenset()
    collection_ids: frozenset[str] = frozenset()
    tag_ids: frozenset[str] = frozenset()
    media_types: frozenset[str] = frozenset()
    kinds: frozenset[BlockKind] = frozenset()
    updated_after: datetime | None = None
    updated_before: datetime | None = None
    extra: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _timestamps_are_aware(self) -> Self:
        for name in ("updated_after", "updated_before"):
            value: datetime | None = getattr(self, name)
            if value is not None and value.tzinfo is None:
                msg = f"{name} must be timezone-aware; a naive timestamp has no defined meaning"
                raise ValueError(msg)
        return self

    @property
    def is_empty(self) -> bool:
        """True when this filter restricts nothing."""
        return self == Filter()


class Query(BaseModel):
    """One retrieval request, as every stage in a pipeline sees it.

    Deliberately carries no shared scratch space. Stages that both need the query embedded
    each embed it; the embedding cache is keyed by model identity and text, so the second
    call is free. Threading intermediate state through the query instead would make stages
    order-dependent, and an order-dependent pipeline cannot be compared to another one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, description="Candidates the pipeline should return.")
    filter: Filter = Field(default_factory=Filter)
    profile: RetrievalProfile = RetrievalProfile.BALANCED
    metadata: Metadata = Field(default_factory=dict)


class Candidate(BaseModel):
    """A chunk under consideration, with the scores that put it there.

    The whole chunk travels rather than an id, because every stage after the first needs
    the text — a reranker scores it, context assembly measures it — and refetching per
    stage turns one pipeline into N round trips.

    :attr:`scores` keeps each stage's contribution. Fusion needs the history, and so does
    the evaluation harness: "reranking helped" is only checkable if the pre-rerank score
    survived.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk: Chunk
    score: float = Field(description="Current effective score. Comparable only within a run.")
    scores: dict[str, float] = Field(
        default_factory=dict,
        description="Score by stage name, in the order the stages ran.",
    )

    @property
    def chunk_id(self) -> str:
        return self.chunk.id

    def scored_by(self, stage: str, score: float) -> Candidate:
        """Return a copy carrying ``stage``'s score, which becomes the effective score."""
        return self.model_copy(update={"score": score, "scores": {**self.scores, stage: score}})


class Context(BaseModel):
    """The passages assembled for generation, fitted to a context window.

    Emitted by context assembly rather than by a retrieval stage: a stage returns
    candidates, and this is a different type. Keeping them separate is why a stage list is
    freely reorderable and this step is not.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: Query
    passages: tuple[Candidate, ...] = ()
    token_count: int = Field(default=0, ge=0)
    truncated: bool = Field(
        default=False,
        description="Whether candidates were dropped to fit the window. Surfaced to the "
        "caller, because an answer built from a truncated context is a weaker claim.",
    )
    metadata: Metadata = Field(default_factory=dict)


__all__ = [
    "Candidate",
    "Context",
    "Filter",
    "Query",
    "RetrievalProfile",
]
