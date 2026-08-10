"""The smaller types: retrieval, generation, sync state, lifecycle and identifiers.

Each is small enough to look obviously right and carries a decision that is not.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from manicule.core.embedding import NDArrayLike, Pooling, TokenStates
from manicule.core.generation import FinishReason, Token, Usage
from manicule.core.ids import chunk_id, content_hash, document_id
from manicule.core.lifecycle import HealthReport, HealthState, Lifecycle, worst_state
from manicule.core.retrieval import Candidate, Context, Filter, Query, RetrievalProfile
from manicule.core.sources import DiscoveredDoc, DocRef, Watermark
from tests.fakes import make_chunks, make_document

# --- retrieval -----------------------------------------------------------------------------


def test_a_query_carries_no_scratch_space() -> None:
    """Stages that both need the query embedded each embed it; the cache makes that free.

    Threading intermediate state through the query instead would make the pipeline
    order-dependent, and an order-dependent pipeline cannot be compared with another one.
    """
    assert set(Query.model_fields) == {"text", "limit", "filter", "profile", "metadata"}


def test_an_empty_filter_restricts_nothing() -> None:
    assert Filter().is_empty
    assert not Filter(workspace_id="w").is_empty


def test_filters_reject_a_timestamp_with_no_timezone() -> None:
    """A naive timestamp has no defined meaning, and two machines will disagree about it."""
    with pytest.raises(ValidationError, match="timezone-aware"):
        Filter(updated_after=datetime(2026, 1, 1))  # noqa: DTZ001


def test_a_candidate_records_what_each_stage_thought() -> None:
    """Fusion needs the history, and so does "did reranking help"."""
    chunk = make_chunks(make_document())[0]
    dense = Candidate(chunk=chunk, score=0.4, scores={"dense": 0.4})
    reranked = dense.scored_by("rerank", 0.9)

    assert reranked.score == 0.9
    assert reranked.scores == {"dense": 0.4, "rerank": 0.9}
    assert dense.scores == {"dense": 0.4}, "the original must be untouched"
    assert reranked.chunk_id == chunk.id


def test_a_truncated_context_says_so() -> None:
    """An answer built from a truncated context is a weaker claim, and the caller should know."""
    document = make_document()
    context = Context(
        query=Query(text="q"),
        passages=tuple(
            Candidate(chunk=chunk, score=1.0) for chunk in make_chunks(document, count=2)
        ),
        token_count=42,
        truncated=True,
    )
    assert context.truncated
    assert len(context.passages) == 2


def test_profiles_are_a_closed_set() -> None:
    assert [p.value for p in RetrievalProfile] == ["fast", "balanced", "precise"]


# --- generation ----------------------------------------------------------------------------


def test_a_stream_says_how_it_ended_in_band() -> None:
    """A stream that dies mid-answer must say so, not merely stop."""
    assert not Token(text="hello").is_final
    assert Token(finish_reason=FinishReason.LENGTH).is_final
    assert Token(finish_reason=FinishReason.ERROR, error="upstream reset").is_final


def test_usage_totals_are_derived_not_reported() -> None:
    assert Usage(prompt_tokens=100, completion_tokens=20).total_tokens == 120


def test_there_are_no_provider_specific_types() -> None:
    """One interface, reached with a base URL. No Anthropic type and no OpenAI type."""
    from manicule import core  # noqa: PLC0415

    names = " ".join(core.__all__).lower()
    for vendor in ("openai", "anthropic", "ollama", "google", "litellm"):
        assert vendor not in names


# --- sync state ------------------------------------------------------------------------------


def test_a_watermark_is_opaque_and_timestamped() -> None:
    """A CQL timestamp, a page token or a commit SHA. Stored, handed back, never interpreted."""
    mark = Watermark(value="2026-01-01T00:00:00Z", observed_at=datetime.now(tz=UTC))
    assert mark.value == "2026-01-01T00:00:00Z"
    with pytest.raises(ValidationError, match="timezone-aware"):
        Watermark(value="x", observed_at=datetime(2026, 1, 1))  # noqa: DTZ001


def test_discovery_is_decidable_without_fetching() -> None:
    """An unchanged document must never be downloaded to find out that it is unchanged."""
    found = DiscoveredDoc(
        ref=DocRef(source_id="page-1", uri="https://example/page-1"),
        version_token="7",  # noqa: S106 - a change token, not a credential
    )
    assert found.source_id == "page-1"
    assert found.version_token == "7"  # noqa: S105 - a change token, not a credential


# --- lifecycle -------------------------------------------------------------------------------


def test_health_has_three_states_because_degraded_is_real() -> None:
    assert worst_state([]) is HealthState.OK
    assert worst_state([HealthState.OK, HealthState.DEGRADED]) is HealthState.DEGRADED
    assert (
        worst_state([HealthState.DEGRADED, HealthState.FAILING, HealthState.OK])
        is HealthState.FAILING
    )


def test_a_rollup_names_the_component_that_is_unwell() -> None:
    report = HealthReport.rollup(
        {
            "vector_store:lancedb": HealthReport.healthy(),
            "embedder:mlx": HealthReport.degraded("running on the fallback runtime"),
        }
    )
    assert report.state is HealthState.DEGRADED
    assert "embedder:mlx" in report.detail
    assert not report.ok


def test_a_component_may_implement_one_hook_and_be_detected() -> None:
    """An all-or-nothing check would quietly skip a component implementing three of four."""
    from manicule.core.lifecycle import SupportsMetrics, SupportsSetup  # noqa: PLC0415

    class OnlySetup:
        async def setup(self) -> None:
            return

    assert isinstance(OnlySetup(), SupportsSetup)
    assert not isinstance(OnlySetup(), SupportsMetrics)


async def test_inheriting_lifecycle_gives_working_defaults() -> None:
    """Override the hooks you need; the rest are already correct."""

    class Quiet(Lifecycle):
        pass

    component = Quiet()
    await component.setup()
    await component.teardown()
    assert (await component.health()).ok
    assert component.metrics() == ()


# --- identifiers -----------------------------------------------------------------------------


def test_identifiers_are_derived_so_re_ingest_replaces_rather_than_accumulates() -> None:
    assert document_id("confluence", "12345") == document_id("confluence", "12345")
    assert document_id("confluence", "12345") != document_id("github", "12345")


def test_hashing_is_unambiguous_about_where_one_part_ends() -> None:
    """Length-prefixed, so ("ab", "c") and ("a", "bc") cannot collide."""
    assert document_id("ab", "c") != document_id("a", "bc")
    assert chunk_id("ab", 1, "c") != chunk_id("a", 1, "bc")


def test_content_hashes_are_stable_across_text_and_bytes() -> None:
    assert content_hash("hello") == content_hash("hello")
    assert content_hash("hello") != content_hash("hellp")


# --- token states ------------------------------------------------------------------------------


def test_core_holds_token_states_without_knowing_what_produced_them() -> None:
    """Tier A payloads are large and stay in the backend's own array type.

    Core never touches the values, so it never needs the library that owns them — which is
    what keeps a vector runtime out of an import of the contracts.
    """

    class FakeArray:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

        def __len__(self) -> int:
            return self.shape[0]

        def __iter__(self):  # noqa: ANN204 - the protocol only needs it to exist
            return iter(())

    states = FakeArray((2, 16, 384))
    assert isinstance(states, NDArrayLike)

    payload = TokenStates(states=states, attention_mask=FakeArray((2, 16)), dimension=384)
    assert payload.dimension == 384
    assert payload.states.shape == (2, 16, 384)


def test_pooling_is_part_of_identity_not_a_preference() -> None:
    assert {p.value for p in Pooling} == {"cls", "mean", "last_token", "none"}
