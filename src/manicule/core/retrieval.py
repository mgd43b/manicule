"""Retrieval vocabulary: queries, filters, candidates and assembled context.

A retrieval pipeline is a declared list of stages, each taking candidates and returning
candidates. That uniformity is what lets the evaluation harness compare whole pipelines by
configuration instead of by editing code — which in turn is what makes "no retrieval
feature without a measured improvement" enforceable rather than a discipline to remember.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from manicule.core.content import BlockKind, Chunk, Metadata


class RetrievalProfile(StrEnum):
    """Named cost/quality settings, selectable per query."""

    FAST = "fast"
    BALANCED = "balanced"
    PRECISE = "precise"


class Filter(BaseModel):
    """A restriction on which chunks a search may return.

    Every field is a conjunct; within a field, membership is a disjunction. A field left at
    its default restricts nothing — except :attr:`workspace_ids`, which has no default.

    The shape is settled: ``docs/retrieval.md`` §3 closes the question ``docs/contracts.md``
    §6 had carried open since #1. There is deliberately no untyped escape hatch. This is the
    one type that carries a security boundary, so a reviewer has to be able to see the whole
    restriction, and a field whose meaning is whatever the store decides cannot be reviewed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_ids: frozenset[str] = Field(
        min_length=1,
        description="Tenant scope. Required and non-empty, because a boundary you can forget "
        "to pass is not a boundary and ``frozenset()`` reads as 'no restriction' while "
        "meaning 'match nothing'. Set-valued so that admin cross-workspace search — N scoped "
        "handles fanned out and merged — is more members rather than a way to make the field "
        "optional again.",
    )
    document_ids: frozenset[str] = frozenset()
    sources: frozenset[str] = frozenset()
    collection_ids: frozenset[str] = frozenset()
    tag_ids: frozenset[str] = frozenset()
    media_types: frozenset[str] = frozenset()
    kinds: frozenset[BlockKind] = frozenset()
    langs: frozenset[str] = frozenset()
    updated_after: datetime | None = None
    updated_before: datetime | None = None

    @model_validator(mode="after")
    def _timestamps_are_aware(self) -> Self:
        for name in ("updated_after", "updated_before"):
            value: datetime | None = getattr(self, name)
            if value is not None and value.tzinfo is None:
                msg = f"{name} must be timezone-aware; a naive timestamp has no defined meaning"
                raise ValueError(msg)
        return self

    @property
    def restricting_fields(self) -> frozenset[str]:
        """The fields this filter actually restricts on.

        Each field is compared against its own declared default rather than against an empty
        ``Filter``, because there is no such thing: :attr:`workspace_ids` is required, so it
        has no default and is always in this set.

        This is what lets a store enumerate what it was asked for and refuse the fields it
        cannot honor. A store that silently drops one returns results the filter was written
        to exclude, and the result still looks like a working search.
        """
        return frozenset(
            name
            for name, field in type(self).model_fields.items()
            if getattr(self, name) != field.get_default(call_default_factory=True)
        )


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
    filter: Filter = Field(
        description="What the search may return. Required, and has no default, because it "
        "carries the workspace scope: a query that can be built without one is a query that "
        "can be run without one."
    )
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


class PipelineIdentity(BaseModel):
    """What produced a ranking, in enough detail to refuse a dishonest comparison.

    Travels with every :class:`Confidence` and every retrieval trace. Two runs whose
    identities differ are not two measurements of the same thing: a different reranker, a
    different fusion constant or a different embedding space changes what the numbers mean,
    and averaging across them produces a figure nobody can attribute to anything.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    stages: tuple[str, ...] = Field(
        default=(), description="Stage names, in the order the runner called them."
    )
    profile: RetrievalProfile = RetrievalProfile.BALANCED
    overrides: Metadata = Field(
        default_factory=dict, description="Per-field overrides applied on top of the profile."
    )
    rrf_k: int | None = Field(
        default=None, description="The fusion constant, when a fusion stage ran."
    )
    reranker_model_id: str | None = Field(
        default=None,
        description="Which model reranked. ``None`` means none did, which is a different "
        "pipeline rather than the same one with a step skipped.",
    )
    embed_fingerprint: str | None = Field(
        default=None,
        description="Canonical identity of the vector space the query was embedded into.",
    )


class ConfidenceBand(StrEnum):
    """How well-supported an answer is by the corpus, coarsely."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class Confidence(BaseModel):
    """How well-supported an answer is by the corpus. A statement about the retrieval.

    Deliberately **not** a probability that the answer is correct: nothing here is calibrated
    against anything, and an uncalibrated number presented as a probability is the kind of
    claim that gets believed. It is computed before generation, from what was retrieved, so an
    answer can be wrong at high confidence — the evidence was there and the model misread it.

    It is also not comparable across configurations, which is why :attr:`pipeline` travels
    with it. A score produced under a reranker and one produced without are different
    measurements that happen to share a scale.

    **Absent is not zero.** A query the router answered directly carries no ``Confidence`` at
    all, because nothing was retrieved and "we did not look" is a different claim from "we
    looked and there is nothing" — which is the :attr:`ConfidenceBand.NONE` band with a
    reason.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    band: ConfidenceBand
    components: dict[str, float] = Field(
        default_factory=dict,
        description="Each admissible component's weighted contribution. A number that cannot "
        "say why it is 0.62 is a number nobody can act on.",
    )
    suppressed: dict[str, str] = Field(
        default_factory=dict,
        description="Components that did not contribute, and why. A suppressed component "
        "lowers the reachable ceiling; it is never scored zero, because zero would report the "
        "corpus as weak when the cause was a fault in the pipeline.",
    )
    ceiling: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="The highest score this run could have reached. Below 1.0 whenever a "
        "component was suppressed — which is what makes a profile's cost visible.",
    )
    reason: str = Field(
        default="",
        description="Why the band is what it is, when the number alone would mislead.",
    )
    explicit_definition: bool = Field(
        default=False,
        description="Whether the context contains a glossary entry that explicitly defines a "
        "term this query asked the meaning of. A **classification, not a quantity**: it does "
        "not enter :attr:`score` and does not move :attr:`band`. A detector's confidence that "
        "some text is a definition and a cosine between a question and a passage are different "
        "measurements on different scales, and adding one to the other would be the "
        "substitution this module refuses everywhere else. What it does is make one sentence "
        "unsayable — a result citing an exact definition can no longer also report that nothing "
        "in the corpus resembles the question, because that sentence is false in this state "
        "whatever the similarity happens to be.",
    )
    pipeline: PipelineIdentity = Field(default_factory=PipelineIdentity)


@runtime_checkable
class SupportsGeneration(Protocol):
    """A store that reports a counter changing whenever a query's answer could change.

    The invalidation signal behind the L1 query cache (``docs/retrieval.md`` §10.3). A store
    without one is not a defect — it simply cannot be cached against, and the retriever
    refuses to enable the cache rather than serving results from an index it cannot tell has
    moved.

    Structural rather than part of :class:`~manicule.core.protocols.DocStore`, because a store
    that never changes under a running process — a fixture, a read-only replica — implements
    the store perfectly well without it.
    """

    @property
    def generation(self) -> int:
        """Monotonically increasing. Any commit that changes what a query could return."""
        ...


__all__ = [
    "Candidate",
    "Confidence",
    "ConfidenceBand",
    "Context",
    "Filter",
    "PipelineIdentity",
    "Query",
    "RetrievalProfile",
    "SupportsGeneration",
]
