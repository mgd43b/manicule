"""What a research run produces, and what it is allowed to say about itself.

Every type here is frozen and forbids unknown fields, for the reason
:mod:`manicule.core.retrieval` gives: a payload that silently accepts a misspelled key is a
payload whose shape nobody can rely on.

**None of these carries passage text.** A :class:`ResearchStep` records what was searched and
how much came back; the passages themselves live in the ledger and reach a reader only as
verified citations on the report. That split is the same one
:class:`~manicule.generation.answers.GenerationTrace` makes, and for the same reason — a record
of a run that contains the corpus is the leak the record was supposed to help diagnose.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from manicule.core.retrieval import Candidate


class SubQuestion(BaseModel):
    """One thing to search for, and why the plan thought it was worth searching.

    ``text`` becomes a :class:`~manicule.core.retrieval.Query` verbatim. It carries no scope of
    its own: the filter is copied from the original question, because a sub-question that could
    widen its own scope would be a scope escape reachable by asking for one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1)
    reason: str = Field(
        default="",
        description="The plan's own words for why this angle matters. Recorded so a bad plan "
        "is diagnosable rather than mysterious; never used as evidence.",
    )
    cycle: int = Field(default=1, ge=1, description="Which cycle proposed it. 1 is the plan.")


class ResearchPlan(BaseModel):
    """The sub-questions a run decided to ask, in the order it decided to ask them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str = Field(min_length=1)
    sub_questions: tuple[SubQuestion, ...] = ()
    model_planned: bool = Field(
        default=True,
        description="False when planning produced nothing usable and the run fell back to "
        "searching the original question alone. Recorded rather than hidden: a run that "
        "silently degraded to one search looks exactly like a question with one facet.",
    )


class ResearchStep(BaseModel):
    """One sub-question's retrieval, as it happened.

    Carries counts and identifiers, never passage text. ``fresh`` is the count this step added
    to the ledger that no earlier step had already found, which is the number that says whether
    a cycle was worth its latency.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sub_question: str
    cycle: int = Field(ge=1)
    retrieved: int = Field(default=0, ge=0)
    fresh: int = Field(default=0, ge=0)
    confidence: float | None = Field(
        default=None,
        description="This retrieval's own confidence, verbatim. Never averaged with another "
        "step's: two retrievals are two measurements and their mean describes neither.",
    )
    confidence_band: str | None = None
    routed_away: bool = Field(
        default=False,
        description="The router answered this sub-question directly instead of retrieving, so "
        "it contributed no evidence.",
    )


class ResearchTrace(BaseModel):
    """One record per research run, beside the retrieval and generation traces.

    **It never contains passage text, document text or the question.** Naming what was
    searched is diagnostic; carrying what came back turns the trace into a copy of the corpus.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    cycles_run: int = Field(default=0, ge=0)
    cycles_allowed: int = Field(default=0, ge=0)
    steps: tuple[ResearchStep, ...] = ()
    planned: int = Field(default=0, ge=0, description="Sub-questions the plan proposed.")
    searched: int = Field(default=0, ge=0, description="Sub-questions actually retrieved for.")
    passages_found: int = Field(default=0, ge=0, description="Distinct passages in the ledger.")
    passages_reported: int = Field(
        default=0, ge=0, description="Passages that fit the report's context and were numbered."
    )
    model_calls: int = Field(
        default=0,
        ge=0,
        description="Generator calls the loop itself made — planning and gap-finding. The "
        "report's own call is counted by the generation trace, not here, so the two records "
        "never double-count one answer.",
    )
    stopped_early: str = Field(
        default="",
        description="Why the loop stopped before its cycle budget, or empty. A bound that was "
        "reached is not the same fact as a loop that ran out of things to ask.",
    )
    elapsed_ms: int = Field(default=0, ge=0)


class Evidence(BaseModel):
    """What a research run gathered, before anything is written about it.

    The passages are ordered, and **that order is the slot numbering the report will cite
    against**. Nothing downstream may reorder them between here and the prompt: slots are
    positional into ``Context.passages``, so a reordering after rendering produces citations
    that verify and point at the wrong passage.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan: ResearchPlan
    passages: tuple[Candidate, ...] = ()
    support: dict[str, int] = Field(
        default_factory=dict[str, int],
        description="How many sub-questions retrieved each chunk, by chunk id. Agreement "
        "between independently-asked searches, which is the one signal a multi-step run has "
        "that a single retrieval does not. Reported on its own and **never** folded into a "
        "confidence: that number is computed per retrieval from that retrieval's own pipeline "
        "identity, and a figure this loop invented has no business in it.",
    )
    trace: ResearchTrace = ResearchTrace()


__all__ = [
    "Evidence",
    "ResearchPlan",
    "ResearchStep",
    "ResearchTrace",
    "SubQuestion",
]
