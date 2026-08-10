"""Context assembly and the two token counters, one of which is the wrong one."""

from __future__ import annotations

import pytest

from manicule.config.profiles import PROFILES, ProfileConfig, profile_config
from manicule.core.retrieval import Candidate, RetrievalProfile
from manicule.retrieval.assembly import ContextAssembler, ContextBudgetError, window_problem
from manicule.retrieval.tokens import ContextTokenCounter, ContextTokenDriftError
from manicule.retrieval.trace import AssemblyReport, installed
from tests.retrieval.fakes import a_query, profiles
from tests.storage_helpers import make_chunk, make_document

DOCUMENT = make_document()


def passage(position: int, words: int) -> Candidate:
    chunk = make_chunk(DOCUMENT, position, " ".join(f"word{position}x{n}" for n in range(words)))
    return Candidate(chunk=chunk, score=1.0 - position * 0.01, scores={"rrf": 0.016})


def an_assembler(**overrides: object) -> ContextAssembler:
    return ContextAssembler(counter=ContextTokenCounter(), profiles=profiles(**overrides))


def test_the_counter_names_its_encoding_and_never_a_model() -> None:
    """Naming a model that is not being used makes an estimate look authoritative.

    The generator is Ollama-hosted and runs a Llama, Qwen or Mistral vocabulary; none of them
    is tiktoken's. What this counter is honest about is exactly that.
    """
    identity = ContextTokenCounter().identity
    assert identity.startswith("tiktoken:o200k_base")
    assert "gpt" not in identity


def test_the_estimate_errs_upward() -> None:
    """Undercounting overflows the window and the server truncates; overcounting costs a passage.

    The error is pushed into the direction that is visible.
    """
    text = "authentication tokens rotate on a schedule"
    plain = ContextTokenCounter(safety_factor=1.0).count(text)
    inflated = ContextTokenCounter(safety_factor=1.5).count(text)
    assert inflated > plain


def test_a_safety_factor_below_one_is_refused() -> None:
    """It would bias the estimate into the one direction that fails silently."""
    with pytest.raises(ValueError, match=r"below 1\.0"):
        ContextTokenCounter(safety_factor=0.9)


def test_the_chunk_token_count_is_not_used_for_context_fitting() -> None:
    """The category error this module exists to prevent.

    ``Chunk.token_count`` is measured in the *embedder's* units, for a model that is not
    generating anything. It is sitting on every candidate, it is a plausible number, and it is
    wrong for this purpose by an unknown factor.
    """
    chunk = make_chunk(DOCUMENT, 0, "authentication " * 40)
    lying = chunk.model_copy(update={"token_count": 1})

    assert ContextTokenCounter().count_chunk(lying) > lying.token_count


def test_counts_are_memoised_by_a_content_derived_id() -> None:
    """Chunk ids come from content, so the cache is exact and can never go stale."""
    counter = ContextTokenCounter()
    chunk = make_chunk(DOCUMENT, 0, "authentication tokens")
    assert counter.count_chunk(chunk) == counter.count_chunk(chunk)


def test_an_under_estimate_past_tolerance_is_an_error() -> None:
    """Measuring once beats a safety factor forever — and the asymmetry is the point.

    An over-estimate wastes budget and is recorded. An under-estimate means the prompt was
    truncated from the front, which discards the system prompt and the citation protocol and
    presents as a model that does not follow instructions.
    """
    counter = ContextTokenCounter(drift_tolerance=0.1)
    with pytest.raises(ContextTokenDriftError, match="safety_factor"):
        counter.observe("qwen2.5:14b", estimated=100, actual=200)


def test_an_over_estimate_is_recorded_rather_than_raised() -> None:
    counter = ContextTokenCounter(drift_tolerance=0.1)
    drift = counter.observe("qwen2.5:14b", estimated=200, actual=100)
    assert drift == pytest.approx(1.0)
    assert counter.drift["qwen2.5:14b"] == pytest.approx(1.0)


def test_assembly_takes_whole_passages_or_none() -> None:
    """A trimmed passage makes its citation quote text the source does not contain.

    Every anchor describes the *whole* chunk — a page anchor's rects, a cell anchor's range, a
    line anchor's span — so trimming the text makes the anchor point at more than the passage
    says, which is the exact class of defect the anchor type was designed to make impossible.
    """
    assembler = an_assembler(candidates=5, final_top_k=3, context_tokens=300)
    huge = passage(0, 400)
    small = [passage(1, 5), passage(2, 5)]

    context = assembler.assemble(a_query(), [*small, huge])

    assert [candidate.chunk.text for candidate in context.passages] == [
        small[0].chunk.text,
        small[1].chunk.text,
    ]
    assert all("..." not in candidate.chunk.text for candidate in context.passages)


def test_a_passage_that_does_not_fit_is_skipped_and_the_next_is_tried() -> None:
    """Chunking keeps tables and code blocks whole, so one large passage mid-ranking is normal.

    Stopping there would discard every good passage behind it. Order is never changed — only
    membership.
    """
    assembler = an_assembler(candidates=5, final_top_k=3, context_tokens=300)
    ranked = [passage(0, 5), passage(1, 400), passage(2, 5)]

    context = assembler.assemble(a_query(), ranked)

    assert [c.chunk.position for c in context.passages] == [0, 2]
    assert context.truncated


def test_a_dropped_passage_is_named_in_the_trace() -> None:
    with installed() as frame:
        an_assembler(candidates=5, final_top_k=3, context_tokens=300).assemble(
            a_query(), [passage(0, 5), passage(1, 400)]
        )

    assert isinstance(frame.assembly, AssemblyReport)
    assert len(frame.assembly.dropped) == 1
    assert frame.assembly.tokenizer.startswith("tiktoken:")


def test_a_top_ranked_passage_that_cannot_fit_at_all_is_a_refusal() -> None:
    """A refusal with both numbers named rather than a silently shortened context.

    Every shipped profile allows several times what its own head can hold, so reaching this
    means something was overridden — a configuration question with a configuration answer, and
    one that a truncated context would hide.
    """
    assembler = an_assembler(candidates=5, final_top_k=3, context_tokens=256)
    with pytest.raises(ContextBudgetError, match="highest-ranked passage"):
        assembler.assemble(a_query(), [passage(0, 4000)])


def test_selection_is_the_head_and_the_budget_is_a_rail() -> None:
    """The fitter never binds in a shipped configuration, and saying so is more useful than
    implying otherwise."""
    context = an_assembler().assemble(a_query(), [passage(position, 20) for position in range(20)])

    assert len(context.passages) == PROFILES[RetrievalProfile.BALANCED].final_top_k
    assert not context.truncated


@pytest.mark.parametrize("profile", list(RetrievalProfile))
def test_every_shipped_profile_admits_its_own_head(profile: RetrievalProfile) -> None:
    """The property that makes the budget a rail: it cannot bind on what the head can hold.

    Checked over the shipped profiles rather than enforced as a model invariant, because
    raising ``final_top_k`` without raising the budget is an ordinary thing to want and typical
    passages are a fraction of the chunk budget. What must hold is that *these* numbers are
    consistent.
    """
    settings = PROFILES[profile]
    assert settings.context_tokens >= settings.largest_possible_context


@pytest.mark.parametrize(
    ("profile", "window"),
    [
        (RetrievalProfile.FAST, 8192),
        (RetrievalProfile.BALANCED, 8192),
        (RetrievalProfile.PRECISE, 16384),
    ],
)
def test_the_shipped_profiles_fit_the_windows_they_claim(
    profile: RetrievalProfile, window: int
) -> None:
    """``fast`` and ``balanced`` fit an 8k window; ``precise`` needs 16k.

    This is the arithmetic the startup cross-check runs, and it is why the shipped budgets are
    what they are: the numbers this file first carried put ``precise`` at 36 240 tokens against
    the 32 768-token window of the model this project ships with.
    """
    assert (
        window_problem(
            PROFILES[profile],
            context_window=window,
            model_id="test/model",
            system_prompt_tokens=400,
            generation_reserve=1024,
        )
        is None
    )


def test_a_profile_that_does_not_fit_names_both_numbers() -> None:
    """A limit that can only be discovered by exceeding it gets discovered in production."""
    problem = window_problem(
        PROFILES[RetrievalProfile.PRECISE],
        context_window=8192,
        model_id="tiny/model",
        system_prompt_tokens=400,
        generation_reserve=1024,
    )
    assert problem is not None
    assert "8192" in problem
    assert "tiny/model" in problem


def test_a_head_larger_than_the_candidate_set_is_refused() -> None:
    """You cannot return more candidates than you fetched, and it is also the only
    configuration in which a reranker would need a mixed-scale tail."""
    with pytest.raises(ValueError, match="above candidates"):
        profile_config(RetrievalProfile.BALANCED, {"final_top_k": 50})


def test_an_override_starts_from_the_named_profile() -> None:
    adjusted = profile_config(RetrievalProfile.BALANCED, {"min_score": 0.4})
    assert isinstance(adjusted, ProfileConfig)
    assert adjusted.min_score == 0.4
    assert adjusted.candidates == PROFILES[RetrievalProfile.BALANCED].candidates
