"""Generation vocabulary.

There is one generation interface, and it is reached with a ``base_url``. There is no
Anthropic type and no OpenAI type: a provider-shaped type in core forces every consumer to
learn every provider, and the set of providers only grows.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class FinishReason(StrEnum):
    """Why a generation stopped."""

    STOP = "stop"
    """The model finished its answer."""

    LENGTH = "length"
    """The token budget ran out. The answer is truncated and should be labelled so."""

    CONTENT_FILTER = "content_filter"
    """The provider refused to continue."""

    ERROR = "error"
    """Generation failed part-way. ``Token.error`` carries the detail."""


class Usage(BaseModel):
    """Token accounting for one generation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Token(BaseModel):
    """One streamed piece of a generated answer.

    A stream is a sequence of these. The final element carries :attr:`finish_reason`, and
    :attr:`usage` when the provider reports it — so a consumer that only ever sees the
    stream still learns how the generation ended, without a second call or a side channel.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = ""
    finish_reason: FinishReason | None = None
    usage: Usage | None = None
    error: str | None = Field(
        default=None,
        description="Set with ``finish_reason='error'``. A stream that dies mid-answer "
        "says so in band; it does not just stop.",
    )

    @property
    def is_final(self) -> bool:
        return self.finish_reason is not None


__all__ = [
    "FinishReason",
    "Token",
    "Usage",
]
