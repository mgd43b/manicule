"""The bounds a research run declares before it starts.

Separate from :class:`~manicule.config.settings.ResearchSettings` and validated by it, so this
module imports nothing heavier than pydantic and the loop can be constructed in a test without
a whole ``Settings`` tree. The settings block is where an operator writes these; this is the
shape the loop reads.

**Every field here is a ceiling, and each one is a different failure it prevents.** A loop whose
only bound is "until the model says stop" has handed an unattended caller the machine for as
long as a model feels like planning, which is the argument ``docs/surfaces.md`` §4 already made
about ``document_reindex_stale`` — and the reason a bounded operation may be on a surface an
assistant reaches while an unbounded one may not.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ResearchLimits(BaseModel):
    """What one research run may spend."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_cycles: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Rounds of planning and searching. Each round after the first costs one "
        "model call to decide it is worth running, so this is the bound on latency as much as "
        "on retrieval.",
    )
    max_sub_questions: int = Field(
        default=4,
        ge=1,
        le=12,
        description="Searches one cycle may run. Multiplied by max_cycles this is the ceiling "
        "on retrievals for one question, and retrievals are the expensive part: each is an "
        "embedder pass and, on the reranking profiles, a cross-encoder pass.",
    )
    concurrency: int = Field(
        default=3,
        ge=1,
        le=8,
        description="Retrievals in flight at once. Small deliberately: the embedder serializes "
        "every forward pass through one worker thread, so a wider fan-out queues there while "
        "holding a database connection each, and the connection pool is what runs out.",
    )
    report_passages: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Passages the report may cite from. The reason a research report is worth "
        "more than one ask: a question with four facets cannot be answered from the five "
        "passages one retrieval's final_top_k allows.",
    )
    report_tokens: int = Field(
        default=12288,
        ge=1024,
        description="Context budget for the report, in the generator's tokenizer. Wider than "
        "any profile's, so it is **not** covered by the profile's startup window cross-check "
        "and gets its own — see manicule.research.loop.plan_problem.",
    )
    timeout_s: float = Field(
        default=300.0,
        gt=0,
        description="Wall clock for the whole run, checked between cycles. A ceiling rather "
        "than a target: the loop stops planning further cycles once it is reached and reports "
        "with what it has, because a run that returns late is better than one that returns "
        "nothing.",
    )


__all__ = ["ResearchLimits"]
