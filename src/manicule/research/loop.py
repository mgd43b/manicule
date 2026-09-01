"""The loop: plan, retrieve, decide whether to go again, hand the evidence on.

**It does not answer.** It returns an :class:`~manicule.research.models.Evidence`, and the
caller runs the ordinary answer path over it. That split is the whole reason the citation
guarantee needs no restating here: the report is an ``ask`` with a wider context, so the
binder, the three-level verification ladder, the egress filter and the redaction projection are
the same objects doing the same work, not a second implementation that could forget one of
them.

**Every model call this module makes is a planning call**, and planning calls never see a
passage. What the loop sends a model is the question, the list of searches it has already run,
and how many passages came back — counts, not corpus. So the two prompts here are cheap, they
cannot leak a document a policy would have refused, and the thing the model is deciding is
which query to run next rather than what the evidence means.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import JsonValue

from manicule.config.profiles import ProfileConfig
from manicule.core.errors import ConfigError, ManiculeError
from manicule.core.protocols import Generator, generating
from manicule.core.retrieval import Candidate, Context, Filter, Query
from manicule.generation.answering import accepted_extras
from manicule.generation.prompt import ChatMessage
from manicule.research.config import ResearchLimits
from manicule.research.ledger import EvidenceLedger
from manicule.research.models import (
    Evidence,
    ResearchPlan,
    ResearchStep,
    ResearchTrace,
    SubQuestion,
)
from manicule.research.ports import Retrieving
from manicule.research.prompts import gap_messages, parse_queries, plan_messages


def plan_problem(limits: ResearchLimits, *, context_window: int, reserved: int) -> str:
    """Why this configuration cannot run a research report, or ``""``.

    The same shape of check as :func:`manicule.retrieval.assembly.window_problem`, and it exists
    for the same reason: ``research.report_tokens`` is a **wider** context than any profile's,
    so the startup cross-check that approved the profile says nothing about it. A budget whose
    only failure mode is the server silently truncating the prompt — discarding the system
    prompt and the citation protocol with it — is a budget that has to be refused before the
    first question rather than discovered on it.
    """
    needed = limits.report_tokens + reserved
    if context_window <= 0 or needed <= context_window:
        return ""
    return (
        f"research.report_tokens is {limits.report_tokens} and the rest of the prompt reserves "
        f"{reserved}, which needs {needed} tokens from a generator serving {context_window}. "
        f"A research report is deliberately wider than one profile's context and is not covered "
        f"by the profile's own startup check. Lower research.report_tokens, or configure a "
        f"model with a larger window."
    )


@dataclass(frozen=True, slots=True)
class _Searched:
    """One completed retrieval, before it reaches the ledger."""

    question: SubQuestion
    candidates: tuple[Candidate, ...]
    confidence: float | None
    band: str | None
    routed_away: bool


class ResearchLoop:
    """Runs several retrievals for one question and returns what they found.

    Constructed per run rather than held as a singleton: it owns a ledger, which is per-run
    state, and a shared instance would merge two questions' evidence into one report.
    """

    def __init__(
        self,
        *,
        generator: Generator,
        retriever: Retrieving,
        limits: ResearchLimits,
    ) -> None:
        self._generator = generator
        self._retriever = retriever
        self._limits = limits
        self._extras = accepted_extras(generator)
        self._calls = 0

    async def run(self, question: str, base: Query) -> Evidence:
        """Plan, search, and keep searching while the budget and the findings justify it.

        Args:
            question: The original question, as the person asked it.
            base: The query that would have been run for a plain ``ask``. Everything but its
                text is copied onto every sub-question — the filter above all, because a
                sub-question that could widen its own scope would be a scope escape reachable
                by wording a question a particular way.

        Returns:
            The plan, the de-duplicated passages in citation order, and the trace.
        """
        started = time.monotonic()
        ledger = EvidenceLedger()
        steps: list[ResearchStep] = []
        asked: list[str] = []
        deadline = started + self._limits.timeout_s

        plan = await self._plan(question)
        pending = list(plan.sub_questions)
        cycle = 0
        stopped = ""

        while pending and cycle < self._limits.max_cycles:
            cycle += 1
            if time.monotonic() >= deadline:
                stopped = "the run reached research.timeout_s before this cycle started"
                break
            found = await self._search(pending, base)
            for outcome in found:
                fresh = 0 if outcome.routed_away else ledger.add(outcome.candidates)
                asked.append(outcome.question.text)
                steps.append(
                    ResearchStep(
                        sub_question=outcome.question.text,
                        cycle=cycle,
                        retrieved=len(outcome.candidates),
                        fresh=fresh,
                        confidence=outcome.confidence,
                        confidence_band=outcome.band,
                        routed_away=outcome.routed_away,
                    )
                )
            pending = []
            if cycle >= self._limits.max_cycles:
                break
            if time.monotonic() >= deadline:
                stopped = "the run reached research.timeout_s"
                break
            more = await self._gaps(question, asked=asked, found=len(ledger), cycle=cycle + 1)
            if not more:
                stopped = "the plan proposed no further searches"
                break
            pending = list(more)

        passages = ledger.ranked()
        return Evidence(
            plan=plan,
            passages=passages,
            support=ledger.support,
            trace=ResearchTrace(
                cycles_run=cycle,
                cycles_allowed=self._limits.max_cycles,
                steps=tuple(steps),
                planned=len(plan.sub_questions),
                searched=len(steps),
                passages_found=len(passages),
                model_calls=self._calls,
                stopped_early=stopped,
                elapsed_ms=int((time.monotonic() - started) * 1000),
            ),
        )

    async def _plan(self, question: str) -> ResearchPlan:
        """One model call, or the question itself when it produces nothing usable.

        The fallback is recorded on the plan rather than papered over. A run that silently
        degraded to a single search looks, in its output, exactly like a question that only had
        one facet — and those are different facts about the run.
        """
        reply = await self._say(plan_messages(question, limit=self._limits.max_sub_questions))
        found = parse_queries(reply, limit=self._limits.max_sub_questions)
        if not found:
            return ResearchPlan(
                question=question,
                sub_questions=(
                    SubQuestion(
                        text=question,
                        reason="planning returned no usable "
                        "queries; searching the question as asked",
                    ),
                ),
                model_planned=False,
            )
        return ResearchPlan(
            question=question,
            sub_questions=tuple(
                SubQuestion(text=text, reason=reason, cycle=1) for text, reason in found
            ),
        )

    async def _gaps(
        self, question: str, *, asked: Sequence[str], found: int, cycle: int
    ) -> tuple[SubQuestion, ...]:
        """What is still worth searching, or nothing.

        A model that cannot name a gap returns an empty list and the loop stops, which is the
        common case and the desirable one. Anything it proposes that repeats a search already
        run is dropped here rather than spent: the model is asked not to repeat itself, and this
        is what happens when it does anyway.
        """
        reply = await self._say(
            gap_messages(
                question,
                searched=list(asked),
                found=found,
                limit=self._limits.max_sub_questions,
            )
        )
        already = {text.casefold() for text in asked}
        return tuple(
            SubQuestion(text=text, reason=reason, cycle=cycle)
            for text, reason in parse_queries(reply, limit=self._limits.max_sub_questions)
            if text.casefold() not in already
        )

    async def _search(self, questions: Sequence[SubQuestion], base: Query) -> list[_Searched]:
        """Retrieve for each sub-question, a bounded number at a time.

        Bounded because the gain is not linear and the cost is: the embedder serializes every
        forward pass through one worker thread by construction, so past a small width the extra
        tasks queue there while still holding a database connection each. The pool is finite,
        and exhausting it makes an unrelated query in another request wait.

        Each retrieval runs as its own task deliberately. The retrieval trace frame is a
        ``ContextVar``, so two coroutines awaited concurrently in one task would write each
        other's diagnostics — plausible numbers attached to the wrong sub-question.
        """
        gate = asyncio.Semaphore(self._limits.concurrency)

        async def one(question: SubQuestion) -> _Searched:
            async with gate:
                result = await self._retriever.retrieve(_reworded(base, question.text))
            confidence = result.confidence
            return _Searched(
                question=question,
                candidates=tuple(result.candidates or result.context.passages),
                confidence=confidence.score if confidence else None,
                band=confidence.band.value if confidence else None,
                routed_away=not result.cites_the_corpus,
            )

        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(one(question)) for question in questions]
        return [task.result() for task in tasks]

    async def _say(self, messages: list[ChatMessage]) -> str:
        """One planning call, as text.

        **Refuses a generator that cannot take a prompt.** ``messages`` is an optional keyword
        the two built-in generators declare; a third-party one need not, and one that ignores it
        would build a *citation-protocol answer prompt* out of the synthetic query and empty
        context below — instructing a model to cite slots that do not exist, for a step whose
        output is parsed as JSON. Silently getting a plan-shaped refusal back is precisely the
        failure this project keeps refusing: the one nobody can see.
        """
        if "messages" not in self._extras:
            msg = (
                f"generator {self._generator.model_id!r} does not accept a prepared prompt, so "
                f"it cannot be asked to plan a research run. Research needs a generator "
                f"declaring the optional 'messages' keyword; both built-in generators do. Use "
                f"llm.generator = 'litellm' or 'cli', or ask this question with `ask`."
            )
            raise ConfigError(msg)
        self._calls += 1
        query = Query(text=_PLANNING_QUERY, filter=_PLANNING_FILTER)
        pieces: list[str] = []
        async with generating(
            self._generator, query, Context(query=query), extra={"messages": messages}
        ) as tokens:
            async for token in tokens:
                if token.error:
                    raise ManiculeError(token.error)
                pieces.append(token.text)
        return "".join(pieces)


def _reworded(base: Query, text: str) -> Query:
    """The base query with a sub-question's text and nothing else changed.

    Everything else is copied, **the filter above all**. This is the same rule
    ``Retriever._reworded`` applies to a glossary rewrite, for the same reason: a query that can
    change its own scope turns a question into a way of reaching another workspace's documents.
    """
    return base.model_copy(update={"text": text})


_PLANNING_QUERY = "plan a research run"
"""Placeholder text for the ``Query`` a planning call is obliged to carry.

:func:`~manicule.core.protocols.generating` fixes two positional inputs and a planning call
needs neither: the prompt travels as ``messages``, which both built-in generators prefer over
rebuilding one, and :meth:`ResearchLoop._say` refuses any generator that does not. It is not
empty because ``Query.text`` has ``min_length=1``, and it says what it is so that a trace
carrying it is not a mystery.
"""

_PLANNING_FILTER_WORKSPACE = "(planning)"
"""The scope a planning call runs in, which is nothing.

Deliberately **not** the caller's workspace. Nothing is retrieved for a planning call, and
naming a real tenant on a query that never reaches a store would put that tenant's name into
whatever the generator records for a request that read none of its documents.
"""

_PLANNING_FILTER = Filter(workspace_ids=frozenset({_PLANNING_FILTER_WORKSPACE}))


def report_overrides(
    limits: ResearchLimits, base: ProfileConfig, configured: Mapping[str, JsonValue]
) -> dict[str, JsonValue]:
    """The profile overrides a research report's context is assembled under.

    Returned as **overrides** rather than as a built :class:`ProfileConfig`, so the profile is
    rebuilt by ``profile_config`` through validation. That is the rule ``config.profiles``
    already states for this exact operation: ``model_copy`` skips validators, and
    ``final_top_k <= candidates`` is precisely the invariant a widened ``final_top_k`` breaks.
    ``candidates`` is raised with it, because a profile asked for more passages than its
    pipeline was told to fetch cannot be satisfied.

    The operator's own overrides are kept and only the three fields a report widens are
    replaced. Dropping them would make a research report quietly ignore configuration that an
    ordinary answer honors.
    """
    passages = max(limits.report_passages, base.final_top_k)
    return {
        **dict(configured),
        "candidates": max(base.candidates, passages),
        "final_top_k": passages,
        "context_tokens": max(limits.report_tokens, base.context_tokens),
    }


__all__ = ["ResearchLoop", "plan_problem", "report_overrides"]
