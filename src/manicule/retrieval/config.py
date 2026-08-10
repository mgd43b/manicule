"""Per-stage configuration, declared where registration can see it.

Separate from the stages themselves for the reason every built-in plugin separates them:
plugin discovery runs in every process that starts, and registration needs a component's
configuration model so that settings written for it are validated rather than ignored. If the
models lived beside the implementations, discovery would import ``tiktoken`` and
``sentence-transformers`` — a cross-encoder's runtime — in order to find out that a stage
exists. This module imports nothing heavier than pydantic.

Every number here is configuration rather than a constant, and that is not decoration.
Comparing two pipelines is the evaluation harness's entire method, and it requires switching
to be *declarative*: no comparison should ever need a code edit, and one that does is a defect
in this design rather than in the harness.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
"""The family pair for the embedder, and the default for a reason that is not brand tidiness.

bge-m3 was chosen substantially because it is multilingual in one space. A monolingual English
reranker placed after it takes a correctly-retrieved non-English passage and ranks it down —
undoing the property the embedder was chosen for, at the last stage before the answer, where it
is least visible. On a corpus that is entirely English a smaller monolingual reranker is very
likely the better trade, and that is what this setting is for; the *default* has to be the one
that does not silently break multilingual retrieval.
"""


class StageConfig(BaseModel):
    """Base for a stage's settings: unknown keys are refused, not ignored."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class DenseConfig(StageConfig):
    """The dense leg's over-fetch and pre-filter knobs (``docs/retrieval.md`` §3.3, §4.3).

    The over-fetch factor is *derived*, not set here. A constant multiplier is wrong in both
    directions — 2x is too little for a fifty-workspace deployment and 20x is wasteful for a
    personal one, and neither knows which it is in. What these bound is the derivation.
    """

    overfetch_min: int = Field(
        default=3,
        ge=1,
        description="Floor on ``k-prime/k``. A healthy single-workspace index still loses rows to "
        "the soft-delete grace window, in-flight documents and unswept tombstones; 3x removes "
        "the retry from the common path, and over an exhaustive search it is not measurable.",
    )
    overfetch_max: int = Field(
        default=20,
        ge=1,
        description="Cap on ``k-prime/k``. Past this the plan should have inverted to the "
        "pre-filter regime, and the cap firing is what makes that visible in the trace rather "
        "than absorbed as latency.",
    )
    absolute_row_cap: int = Field(
        default=2000,
        ge=1,
        description="Hard ceiling on rows fetched. Every over-fetched row is a chunk decode, "
        "so the work is bounded independently of the multiplier.",
    )
    prefilter_id_limit: int = Field(
        default=1000,
        ge=1,
        description="Above this many resolved document ids, push down nothing and post-filter "
        "instead. A starting value, not a measured one: a thousand quoted literals is a "
        "predicate the vector store can still plan usefully, and beyond that the predicate "
        "costs more than the over-fetch it saves. Every query records the two inputs that "
        "settle it.",
    )
    max_expansions: int = Field(
        default=2,
        ge=0,
        description="Retries when the over-fetch produced fewer than ``k`` survivors. Two, "
        "because a third would cost more than the query is worth.",
    )
    expansion_factor: int = Field(
        default=4,
        ge=2,
        description="How much each retry widens ``k-prime``. Four, because a factor that failed is "
        "usually wrong by more than a little.",
    )

    @model_validator(mode="after")
    def _floor_below_cap(self) -> Self:
        if self.overfetch_min > self.overfetch_max:
            msg = (
                f"overfetch_min ({self.overfetch_min}) is above overfetch_max "
                f"({self.overfetch_max}), so the derived factor would be clamped below its own "
                f"floor and every query would report the cap as firing. Lower the floor or "
                f"raise the cap."
            )
            raise ValueError(msg)
        return self


class LexicalConfig(StageConfig):
    """The lexical leg has nothing to configure.

    The statement is the document store's and applies the whole filter inline, before
    ``LIMIT``. What the stage adds is a re-keyed score and a merge — neither has a knob.
    """


class FusionConfig(StageConfig):
    """Reciprocal rank fusion (``docs/retrieval.md`` §5)."""

    legs: tuple[str, ...] = Field(
        default=("dense", "lexical"),
        min_length=1,
        description="Which stages' score ladders to fuse, by name. Configured rather than "
        "hardcoded: replacing the lexical leg with a learned-sparse one, or adding a third "
        "leg, is then a configuration edit and therefore a measurement rather than a rewrite.",
    )
    k: int = Field(
        default=60,
        ge=1,
        description="The RRF constant. 60 is the original paper's value and the near-universal "
        "default. It is recorded in every trace because changing it changes every number "
        "anyone has written down — and lowering it sharpens within-leg ranking at the cost of "
        "the consensus effect, which is the opposite of why RRF was chosen.",
    )

    @model_validator(mode="after")
    def _legs_are_distinct(self) -> Self:
        if len(set(self.legs)) != len(self.legs):
            msg = (
                f"fusion legs {list(self.legs)} name the same stage twice. A repeated leg is "
                f"one ladder counted twice, which weights that leg without saying so."
            )
            raise ValueError(msg)
        return self


class RerankConfig(StageConfig):
    """The cross-encoder (``docs/retrieval.md`` §6)."""

    model: str = Field(default=DEFAULT_RERANKER_MODEL, min_length=1)
    batch_size: int = Field(default=16, ge=1)
    device: str | None = Field(
        default=None,
        description="Passed to the runtime. ``None`` lets it choose, which on Apple Silicon "
        "means Metal.",
    )
    max_length: int = Field(
        default=512,
        ge=1,
        description="Sequence length for the ``(query, passage)`` pair. A passage reaching "
        "here is at most 448 embedder tokens, so this is a rail rather than a trim.",
    )


__all__ = [
    "DEFAULT_RERANKER_MODEL",
    "DenseConfig",
    "FusionConfig",
    "LexicalConfig",
    "RerankConfig",
    "StageConfig",
]
