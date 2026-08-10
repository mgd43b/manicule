"""Fitting ranked candidates into a context window, whole passages only.

**Not a retrieval stage.** It emits :class:`~manicule.core.retrieval.Context`, not a list of
candidates, and keeping the two types distinct is exactly what makes a stage list freely
reorderable while this step is not. A stage that emitted a different type would make every
stage's signature a union.

**A candidate is included entirely or not at all.** This is not a style preference. Every
anchor carries a round-trip obligation — resolving it returns the text the chunk claims, and
that is a test obligation on every parser. A page anchor's rects, a cell anchor's range and a
line anchor's span all describe the *whole* chunk. Trim the text and the anchor now points at
more than the passage says, which is the exact class of defect the anchor type was designed to
make impossible. Truncating a passage to fit, walking back to a sentence boundary and appending
an ellipsis produces a citation quoting text the source does not contain.

**A passage that does not fit is skipped, and the next one is tried.** Chunking keeps tables
and code blocks whole, so one large passage in the middle of a ranking is normal, and stopping
there would discard every good passage behind it. Order is never changed — only membership.

**The budget is a rail, not the selection mechanism**, and saying so is more useful than
implying otherwise. Selection is ``final_top_k``. A passage reaching here is at most 448
embedder tokens, so no shipped profile can fill its own budget; the fitter exists for the
configurations that do not ship — a raised chunk budget, a raised ``final_top_k`` — and it
refuses rather than handles the case where even the top-ranked passage does not fit, because
the shipped budgets make that arithmetically impossible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from manicule.core.errors import ManiculeError
from manicule.core.retrieval import Context
from manicule.retrieval.trace import AssemblyReport, record_assembly

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manicule.config.profiles import ProfileConfig
    from manicule.core.retrieval import Candidate, Query
    from manicule.retrieval.profile import Profiles
    from manicule.retrieval.tokens import ContextTokenCounter


class ContextBudgetError(ManiculeError):
    """The highest-ranked passage does not fit the context budget on its own.

    A refusal with both numbers named rather than a silently shortened context. Every shipped
    profile allows several times what its own ``final_top_k`` can hold, so reaching this means
    the budget was overridden downward or the chunk budget upward — a configuration question
    with a configuration answer, and one that a truncated context would hide.
    """


def window_problem(
    profile: ProfileConfig,
    *,
    context_window: int,
    model_id: str,
    system_prompt_tokens: int,
    generation_reserve: int,
) -> str | None:
    """Why this profile cannot run against this generator, or ``None`` if it can.

    ``context_tokens`` and ``history_tokens`` are budgets for *manicule's* content; the window
    is a property of a model configured somewhere else, and nothing compares them on its own.
    A profile that does not fit is a refusal with both numbers named, run **once at startup**,
    not a runtime truncation — because a limit that can only be discovered by exceeding it gets
    discovered in production, and the way it is discovered is a server silently truncating the
    prompt from the front, discarding the system prompt and the citation protocol, and
    presenting as a model that does not follow instructions.

    Retrieval owns the requirement and states it here; the ticket that binds a generator owns
    the enforcement point, because that is where the served window becomes known.
    """
    needed = (
        profile.context_tokens + profile.history_tokens + system_prompt_tokens + generation_reserve
    )
    if needed <= context_window:
        return None
    return (
        f"the retrieval profile needs {needed} tokens — {profile.context_tokens} of context, "
        f"{profile.history_tokens} of history, {system_prompt_tokens} for the system prompt and "
        f"{generation_reserve} reserved for the answer — but {model_id!r} serves a "
        f"{context_window}-token window. Choose a lighter profile, lower "
        f"rag.overrides.context_tokens, reduce llm.max_tokens, or configure a model with a "
        f"longer window. Left alone, every prompt would be truncated from the front, which "
        f"discards the system prompt and presents as a model that ignores instructions."
    )


class ContextAssembler:
    """Turns a ranking into the passages a generator will see."""

    def __init__(self, *, counter: ContextTokenCounter, profiles: Profiles) -> None:
        self._counter = counter
        self._profiles = profiles

    def assemble(self, query: Query, candidates: Sequence[Candidate]) -> Context:
        """Select ``final_top_k`` candidates and fit them, in rank order.

        Raises:
            ContextBudgetError: The top-ranked passage alone exceeds the budget.
        """
        profile = self._profiles.for_query(query)
        budget = profile.context_tokens
        selected = list(candidates[: profile.final_top_k])

        if selected:
            first = self._counter.count_chunk(selected[0].chunk)
            if first > budget:
                msg = (
                    f"the highest-ranked passage needs {first} tokens and the whole context "
                    f"budget is {budget} ({profile.context_tokens} from the profile, measured "
                    f"with {self._counter.identity}). Nothing can be assembled without cutting "
                    f"the passage, and a trimmed passage makes its citation quote text the "
                    f"source does not contain. Raise rag.overrides.context_tokens, or lower the "
                    f"chunk budget that produced a passage this size."
                )
                raise ContextBudgetError(msg)

        passages: list[Candidate] = []
        dropped: list[tuple[str, int]] = []
        used = 0
        for candidate in selected:
            size = self._counter.count_chunk(candidate.chunk)
            if used + size > budget:
                dropped.append((candidate.chunk.id, size))
                continue
            passages.append(candidate)
            used += size

        record_assembly(
            AssemblyReport(
                tokenizer=self._counter.identity,
                tokens_used=used,
                tokens_available=budget,
                passages=len(passages),
                dropped=tuple(dropped),
            )
        )
        return Context(
            query=query,
            passages=tuple(passages),
            token_count=used,
            truncated=bool(dropped),
            metadata={"tokenizer": self._counter.identity, "token_budget": budget},
        )


__all__ = ["ContextAssembler", "ContextBudgetError", "window_problem"]
