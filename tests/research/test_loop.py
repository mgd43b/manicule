"""What the loop does, and what it refuses to do.

The interesting cases are all failures. A loop driven by a model that returns clean JSON and a
retriever that always has an answer proves only that the happy path is wired up; what decides
whether this feature is safe is what it does when the plan is nonsense, when a sub-question
tries to change scope, and when a generator cannot be given a prompt at all.
"""

from __future__ import annotations

import pytest

from manicule.config.profiles import profile_config
from manicule.core.errors import ConfigError
from manicule.core.retrieval import Filter, Query, RetrievalProfile
from manicule.research.loop import ResearchLoop, plan_problem, report_overrides
from tests.research.fakes import (
    PromptlessGenerator,
    ScriptedGenerator,
    ScriptedRetriever,
    candidate,
    limits,
    query,
)

PLAN_TWO = '{"queries": [{"q": "retry policy", "why": "a"}, {"q": "backoff", "why": "b"}]}'
NO_MORE = '{"queries": []}'


def loop(
    generator: ScriptedGenerator | PromptlessGenerator,
    retriever: ScriptedRetriever,
    **overrides: object,
) -> ResearchLoop:
    return ResearchLoop(
        generator=generator,
        retriever=retriever,
        limits=limits(**overrides),
    )


# --- planning ------------------------------------------------------------------------------


async def test_the_plan_becomes_one_search_each() -> None:
    generator = ScriptedGenerator(replies=[PLAN_TWO, NO_MORE])
    retriever = ScriptedRetriever(default=[candidate("c1")])

    evidence = await loop(generator, retriever).run("how do retries work?", query())

    assert [step.sub_question for step in evidence.trace.steps] == ["retry policy", "backoff"]
    assert [asked.text for asked in retriever.seen] == ["retry policy", "backoff"]


async def test_a_plan_that_returns_nothing_usable_searches_the_question_as_asked() -> None:
    """And says so, because the two runs are indistinguishable in the output otherwise.

    A run that degraded to a single search looks exactly like a question that only ever had one
    facet. ``model_planned`` is the difference, and without it a persistently broken planner
    would present as a corpus of simple questions.
    """
    generator = ScriptedGenerator(replies=["I'm afraid I can't help with that.", NO_MORE])
    retriever = ScriptedRetriever(default=[candidate("c1")])

    evidence = await loop(generator, retriever).run("how do retries work?", query())

    assert evidence.plan.model_planned is False
    assert [asked.text for asked in retriever.seen] == ["how do retries work?"]


async def test_a_plan_longer_than_the_bound_is_cut_to_it() -> None:
    """The ceiling is the loop's, not the model's. A plan is a request, not an instruction."""
    generator = ScriptedGenerator(
        replies=[
            '{"queries": [{"q": "a"}, {"q": "b"}, {"q": "c"}, {"q": "d"}, {"q": "e"}]}',
            NO_MORE,
        ]
    )
    retriever = ScriptedRetriever(default=[candidate("c1")])

    evidence = await loop(generator, retriever, max_sub_questions=2).run("q", query())

    assert len(evidence.trace.steps) == 2


# --- scope ---------------------------------------------------------------------------------


async def test_every_sub_question_carries_the_original_filter_unchanged() -> None:
    """A sub-question that could widen its own scope would be a scope escape reachable by
    wording a question a particular way — so everything but the text is copied."""
    generator = ScriptedGenerator(replies=[PLAN_TWO, NO_MORE])
    retriever = ScriptedRetriever(default=[candidate("c1")])
    base = Query(
        text="how do retries work?",
        limit=5,
        profile=RetrievalProfile.PRECISE,
        filter=Filter(workspace_ids=frozenset({"tenant-a"}), sources=frozenset({"confluence"})),
    )

    await loop(generator, retriever).run("how do retries work?", base)

    assert retriever.seen, "nothing was retrieved, so nothing was checked"
    for asked in retriever.seen:
        assert asked.filter == base.filter
        assert asked.profile is base.profile
        assert asked.limit == base.limit


async def test_a_planning_call_names_no_workspace() -> None:
    """Nothing is retrieved for a planning call, so naming a real tenant on it would put that
    tenant into whatever the generator records for a request that read none of its documents."""
    generator = ScriptedGenerator(replies=[PLAN_TWO, NO_MORE])
    retriever = ScriptedRetriever(default=[candidate("c1")])
    base = Query(text="q", filter=Filter(workspace_ids=frozenset({"tenant-a"})))

    await loop(generator, retriever).run("q", base)

    assert generator.seen, "the planner was never called"
    rendered = str(generator.seen)
    assert "tenant-a" not in rendered


# --- the seam the loop refuses ---------------------------------------------------------------


async def test_a_generator_that_cannot_be_given_a_prompt_is_refused() -> None:
    """Not degraded — refused.

    ``messages`` is optional on the protocol and a third-party generator may legitimately not
    declare it. One that does not would rebuild a *citation-protocol answer prompt* out of the
    synthetic query and empty context a planning call carries, instructing a model to cite slots
    that do not exist, for a step whose reply is parsed as JSON. The plan would come back empty
    every time and the run would silently degrade to a single search forever.
    """
    with pytest.raises(ConfigError, match="prepared prompt"):
        await loop(PromptlessGenerator(), ScriptedRetriever()).run("q", query())


# --- cycles and their bounds ------------------------------------------------------------------


async def test_a_second_cycle_runs_what_the_gap_call_proposed() -> None:
    generator = ScriptedGenerator(
        replies=['{"queries": [{"q": "first"}]}', '{"queries": [{"q": "second"}]}']
    )
    retriever = ScriptedRetriever(default=[candidate("c1")])

    evidence = await loop(generator, retriever, max_cycles=2).run("q", query())

    assert [step.sub_question for step in evidence.trace.steps] == ["first", "second"]
    assert [step.cycle for step in evidence.trace.steps] == [1, 2]


async def test_the_cycle_bound_stops_a_model_that_always_wants_another_round() -> None:
    """The ceiling is declared before the run, not negotiated during it."""
    generator = ScriptedGenerator(
        replies=[f'{{"queries": [{{"q": "angle {round_}"}}]}}' for round_ in range(20)]
    )
    retriever = ScriptedRetriever(default=[candidate("c1")])

    evidence = await loop(generator, retriever, max_cycles=2).run("q", query())

    assert evidence.trace.cycles_run == 2
    assert evidence.trace.cycles_allowed == 2
    assert generator.replies, "the model was asked for more rounds than the bound allows"


async def test_a_gap_call_that_only_repeats_a_finished_search_stops_the_loop() -> None:
    """Spending a cycle re-running a search already run is the failure mode a bound alone does
    not prevent, because it never reaches the bound — it just wastes every cycle up to it."""
    generator = ScriptedGenerator(
        replies=['{"queries": [{"q": "retry policy"}]}', '{"queries": [{"q": "Retry Policy"}]}']
    )
    retriever = ScriptedRetriever(default=[candidate("c1")])

    evidence = await loop(generator, retriever, max_cycles=3).run("q", query())

    assert [step.sub_question for step in evidence.trace.steps] == ["retry policy"]
    assert evidence.trace.stopped_early


async def test_stopping_early_and_reaching_the_bound_are_different_facts() -> None:
    generator = ScriptedGenerator(replies=['{"queries": [{"q": "one"}]}', NO_MORE])
    retriever = ScriptedRetriever(default=[candidate("c1")])

    evidence = await loop(generator, retriever, max_cycles=5).run("q", query())

    assert evidence.trace.cycles_run == 1
    assert "no further searches" in evidence.trace.stopped_early


async def test_an_exhausted_time_budget_reports_with_what_it_has() -> None:
    """A run that returns late is better than one that returns nothing, so the deadline stops
    the *next* cycle rather than abandoning the evidence already gathered."""
    generator = ScriptedGenerator(replies=['{"queries": [{"q": "one"}]}'] * 5)
    retriever = ScriptedRetriever(default=[candidate("c1")], delay_s=0.05)

    evidence = await loop(generator, retriever, max_cycles=5, timeout_s=0.01).run("q", query())

    assert evidence.passages, "the deadline discarded evidence instead of reporting it"
    assert "timeout_s" in evidence.trace.stopped_early


# --- what the evidence is ----------------------------------------------------------------------


async def test_a_passage_two_searches_found_is_one_passage() -> None:
    """Nothing downstream enforces uniqueness, and the binder deduplicates by *slot* rather than
    by chunk — so a duplicate would be numbered twice, cited twice, and counted twice."""
    generator = ScriptedGenerator(replies=[PLAN_TWO, NO_MORE])
    retriever = ScriptedRetriever(
        rankings={
            "retry policy": [candidate("shared", score=0.4), candidate("only-a")],
            "backoff": [candidate("shared", score=0.9)],
        }
    )

    evidence = await loop(generator, retriever).run("q", query())

    assert [passage.chunk.id for passage in evidence.passages] == ["shared", "only-a"]


async def test_the_stronger_score_wins_when_one_passage_arrives_twice() -> None:
    """Taking the maximum is the only merge that cannot make a second search *lower* a
    confidence the first had earned."""
    generator = ScriptedGenerator(replies=[PLAN_TWO, NO_MORE])
    retriever = ScriptedRetriever(
        rankings={
            "retry policy": [candidate("shared", score=0.2)],
            "backoff": [candidate("shared", score=0.8)],
        }
    )

    evidence = await loop(generator, retriever).run("q", query())

    assert [passage.score for passage in evidence.passages] == [0.8]


async def test_a_routed_away_sub_question_contributes_no_evidence_and_says_so() -> None:
    generator = ScriptedGenerator(replies=[PLAN_TWO, NO_MORE])
    retriever = ScriptedRetriever(default=[candidate("c1")], routed_away={"backoff"})

    evidence = await loop(generator, retriever).run("q", query())

    routed = [step for step in evidence.trace.steps if step.routed_away]
    assert [step.sub_question for step in routed] == ["backoff"]
    assert routed[0].fresh == 0


async def test_fresh_counts_only_what_a_cycle_added() -> None:
    """The number that says whether a cycle earned its latency."""
    generator = ScriptedGenerator(
        replies=['{"queries": [{"q": "first"}]}', '{"queries": [{"q": "second"}]}']
    )
    retriever = ScriptedRetriever(
        rankings={"first": [candidate("c1")], "second": [candidate("c1"), candidate("c2")]}
    )

    evidence = await loop(generator, retriever, max_cycles=2).run("q", query())

    assert [step.fresh for step in evidence.trace.steps] == [1, 1]


async def test_retrievals_are_bounded_by_the_configured_concurrency() -> None:
    """The embedder serializes every forward pass through one worker thread, so a wider fan-out
    queues there while each task still holds a database connection."""
    generator = ScriptedGenerator(
        replies=['{"queries": [{"q": "a"}, {"q": "b"}, {"q": "c"}, {"q": "d"}]}', NO_MORE]
    )
    retriever = ScriptedRetriever(default=[candidate("c1")], delay_s=0.01)

    await loop(generator, retriever, max_sub_questions=4, concurrency=2).run("q", query())

    assert max(retriever.in_flight) <= 2


# --- the budget check --------------------------------------------------------------------------


def test_a_report_budget_that_cannot_fit_the_window_is_refused_before_the_first_question() -> None:
    """A profile's startup cross-check says nothing about ``report_tokens``, which is wider by
    design. A limit whose only failure mode is the server silently truncating the prompt has to
    be refused rather than discovered."""
    problem = plan_problem(limits(report_tokens=100_000), context_window=8192, reserved=1500)

    assert "research.report_tokens" in problem
    assert "8192" in problem


def test_a_report_budget_that_fits_is_not_a_problem() -> None:
    assert plan_problem(limits(report_tokens=4096), context_window=32768, reserved=1500) == ""


def test_an_unknown_window_is_not_treated_as_a_refusal() -> None:
    """A generator that does not report its window is a fact about the generator, and refusing
    every research run on that basis would disable the feature rather than bound it."""
    assert plan_problem(limits(), context_window=0, reserved=1500) == ""


def test_the_report_profile_raises_candidates_with_the_passages_it_asks_for() -> None:
    """``final_top_k <= candidates`` is the invariant a widened ``final_top_k`` breaks, and it
    is a validator — so overrides are rebuilt through validation rather than copied past it."""
    overrides = report_overrides(
        limits(report_passages=40), profile_config(RetrievalProfile.FAST, {}), {}
    )
    widened = profile_config(RetrievalProfile.FAST, overrides)

    assert widened.final_top_k == 40
    assert widened.candidates >= widened.final_top_k


def test_the_report_profile_keeps_an_operators_own_overrides() -> None:
    """Dropping them would make a research report quietly ignore configuration an ordinary
    answer honors."""
    overrides = report_overrides(
        limits(), profile_config(RetrievalProfile.BALANCED, {}), {"min_score": 0.7}
    )

    assert overrides["min_score"] == 0.7
