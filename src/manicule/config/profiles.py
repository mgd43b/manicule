"""Retrieval profiles: named points on the cost/quality curve.

Three profiles, and every knob in them is one manicule actually implements. A profile that
declares switches for features nobody built reads like a feature list and behaves like a
lie; features join a profile when they exist and have been measured to help.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

from manicule.core.retrieval import RetrievalProfile


class ProfileConfig(BaseModel):
    """Retrieval settings for one profile."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidates: int = Field(ge=1, description="Candidates each retrieval stage fetches.")
    min_score: float = Field(ge=0.0, le=1.0, description="Floor below which a candidate drops.")
    final_top_k: int = Field(ge=1, description="Candidates that survive into context.")
    context_tokens: int = Field(ge=256, description="Token budget for retrieved passages.")
    history_tokens: int = Field(ge=0, description="Token budget for conversation history.")
    rerank: bool = Field(description="Whether a cross-encoder rescores the fused candidates.")


PROFILES: Mapping[RetrievalProfile, ProfileConfig] = {
    RetrievalProfile.FAST: ProfileConfig(
        candidates=10,
        min_score=0.5,
        final_top_k=3,
        context_tokens=8192,
        history_tokens=512,
        rerank=False,
    ),
    RetrievalProfile.BALANCED: ProfileConfig(
        candidates=20,
        min_score=0.3,
        final_top_k=5,
        context_tokens=16384,
        history_tokens=1024,
        rerank=True,
    ),
    RetrievalProfile.PRECISE: ProfileConfig(
        candidates=50,
        min_score=0.15,
        final_top_k=10,
        context_tokens=32768,
        history_tokens=2048,
        rerank=True,
    ),
}


def profile_config(
    profile: RetrievalProfile, overrides: Mapping[str, object] | None = None
) -> ProfileConfig:
    """The settings for ``profile``, with any per-field overrides applied.

    Overrides start from the named profile rather than from a separate set of defaults, so
    overriding one field cannot silently change the others.
    """
    base = PROFILES[profile]
    if not overrides:
        return base
    return base.model_copy(update=dict(overrides)).model_validate(
        {**base.model_dump(), **dict(overrides)}
    )


__all__ = ["PROFILES", "ProfileConfig", "profile_config"]
