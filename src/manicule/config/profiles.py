"""Retrieval profiles: named points on the cost/quality curve.

Three profiles, and every knob in them is one manicule actually implements. A profile that
declares switches for features nobody built reads like a feature list and behaves like a
lie; features join a profile when they exist and have been measured to help.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from manicule.core.retrieval import RetrievalProfile

MAX_PASSAGE_EMBED_TOKENS: Final = 448
"""The largest passage that can reach a context, in the *embedder's* tokens.

The 512-token chunk budget is split 64 for the heading breadcrumb and 448 for the text, and the
breadcrumb never appears in ``Chunk.text`` at all — so what gets assembled is at most 448
embedder tokens, not 512.
"""

VOCABULARY_RATIO: Final = 2.0
"""Assumed worst-case generator tokens per embedder token.

The two budgets are measured with different tokenizers — a SentencePiece vocabulary sized for
the embedder, and a BPE estimate for the generator — so a passage's size in one is not its size
in the other. For English the two are within a fifth of each other; 2.0 is deliberately far
past any plausible ratio, because the number it sizes is a **rail**, and a rail that binds in a
shipped configuration would silently drop a passage that was meant to fit.
"""

PASSAGE_FRAMING_TOKENS: Final = 32
"""Room per passage for the slot marker and separators the generator's prompt adds."""


class ProfileConfig(BaseModel):
    """Retrieval settings for one profile."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidates: int = Field(ge=1, description="Candidates each retrieval stage fetches.")
    min_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Floor below which a candidate drops, applied to the **dense leg's cosine "
        "similarity** and to nothing else. The other two plausible places are unusable: a "
        "fused RRF score with two legs cannot exceed 2/61, so a floor of 0.3 there discards "
        "every candidate in every profile and returns an empty result set that looks exactly "
        "like an empty corpus; and BM25 is corpus-relative and unbounded, so a constant has "
        "nothing absolute to mean. Cosine over L2-normalised vectors is the one number in the "
        "pipeline with a meaning that survives leaving the run it was computed in.",
    )
    final_top_k: int = Field(ge=1, description="Candidates that survive into context.")
    context_tokens: int = Field(ge=256, description="Token budget for retrieved passages.")
    history_tokens: int = Field(ge=0, description="Token budget for conversation history.")
    rerank: bool = Field(description="Whether a cross-encoder rescores the fused candidates.")

    @property
    def largest_possible_context(self) -> int:
        """The most this profile's passages could occupy, in generator tokens.

        A worst case rather than an expectation: every passage at the full chunk budget, and a
        vocabulary ratio chosen to be far past any plausible one. It is what makes
        ``context_tokens`` a **rail** rather than the selection mechanism — selection is
        ``final_top_k``, and the shipped budgets all sit above this number, so the fitter never
        binds in a configuration this project ships.

        Deliberately **not** a validator. Raising ``final_top_k`` without raising the budget is
        an ordinary thing to want, and typical passages are a fraction of the chunk budget, so
        refusing it would block a configuration that works in practice on the strength of a
        worst case that will not occur. What happens instead is visible rather than silent: the
        fitter skips what does not fit, ``Context.truncated`` is set, and the trace names every
        dropped passage and its size.
        """
        per_passage = MAX_PASSAGE_EMBED_TOKENS * VOCABULARY_RATIO + PASSAGE_FRAMING_TOKENS
        return int(self.final_top_k * per_passage)

    @model_validator(mode="after")
    def _head_fits_the_candidates(self) -> Self:
        if self.final_top_k > self.candidates:
            msg = (
                f"final_top_k ({self.final_top_k}) is above candidates ({self.candidates}), so "
                f"the pipeline would be asked to return more candidates than it fetched. It is "
                f"also the only configuration in which a reranker would need to concatenate an "
                f"unreranked tail onto the head it scored — one list carrying two score scales "
                f"that differ by two orders of magnitude."
            )
            raise ValueError(msg)
        return self


PROFILES: Mapping[RetrievalProfile, ProfileConfig] = {
    RetrievalProfile.FAST: ProfileConfig(
        candidates=10,
        min_score=0.5,
        final_top_k=3,
        context_tokens=4096,
        history_tokens=512,
        rerank=False,
    ),
    RetrievalProfile.BALANCED: ProfileConfig(
        candidates=20,
        min_score=0.3,
        final_top_k=5,
        context_tokens=5632,
        history_tokens=1024,
        rerank=True,
    ),
    RetrievalProfile.PRECISE: ProfileConfig(
        candidates=50,
        min_score=0.15,
        final_top_k=10,
        context_tokens=12288,
        history_tokens=2048,
        rerank=True,
    ),
}
"""The three points on the cost/quality curve.

**The token budgets are sized from what each profile can actually assemble**, and that is a
change from the numbers this file first shipped with — 8192 / 16384 / 32768, which were
inherited rather than derived. Those were unreachable by a factor of three to five: ten
passages of at most 448 embedder tokens cannot fill 32768 generator tokens under any plausible
ratio between two vocabularies, so the "budget" could never bind and told a reader nothing.
Worse, ``precise`` then failed its own startup cross-check against the model this project ships
with — 32768 + 2048 + a system prompt + a generation reserve does not fit a 32768-token window
— and ``balanced``'s 16384 against an 8k local model was an assembled context twice the size of
the window on every query.

Each budget is now roughly 1.2-1.5x the largest context its ``final_top_k`` can produce, which
keeps the rail from ever binding in a shipped configuration while leaving room for an override.
The consequence worth remembering:

* ``fast`` and ``balanced`` both fit an **8k** window, prompt and reserve included.
* ``precise`` needs **16k**, and fits the default generator's 32768 with room to spare.

The similarity floors are the one set of numbers here that remain inherited rather than
measured. They are placeholders in the right place: the embedder's cosine similarities are not
centred on zero for unrelated text, so 0.5 on ``fast`` may be discarding relevant passages and
0.15 on ``precise`` may be doing nothing at all. Calibrating them is a measurement — sweep the
floor against recall on a fixed query set — and until it runs, saying so is more useful than
implying they were tuned.
"""


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
    # Rebuilt through validation rather than copied. ``model_copy`` skips validators, and the
    # invariants on this model — the head fitting the candidate set, the budget admitting the
    # head — are exactly the ones an override is capable of breaking.
    return ProfileConfig.model_validate({**base.model_dump(), **dict(overrides)})


__all__ = [
    "MAX_PASSAGE_EMBED_TOKENS",
    "PASSAGE_FRAMING_TOKENS",
    "PROFILES",
    "VOCABULARY_RATIO",
    "ProfileConfig",
    "profile_config",
]
