"""The provider adapter and the token budget: the traps, not the plumbing."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Sequence
from typing import Any, override

import litellm
import pytest
from litellm.types.utils import (  # pyright: ignore[reportMissingTypeStubs] - the library ships none
    Delta,
    ModelResponseStream,
    StreamingChoices,
)

from manicule.config.profiles import PROFILES
from manicule.config.settings import LlmSettings
from manicule.core.errors import (
    ConfigError,
    ContentFilteredError,
    ContextWindowError,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderRequestError,
)
from manicule.core.generation import FinishReason, Usage
from manicule.core.protocols import aclose
from manicule.core.retrieval import RetrievalProfile
from manicule.generation.budget import (
    GENERATION_ENCODING,
    TokenEstimator,
    drift_problem,
    usable_prompt_tokens,
)
from manicule.generation.provider import (
    LOCAL_COST_MAP_ENV,
    LitellmGenerator,
    compose_model,
    error_table,
    map_error,
)
from manicule.retrieval.assembly import window_problem
from tests.generation.fakes import chunk as make_chunk


def chunk(
    text: str = "", finish: str | None = None, usage: Usage | None = None
) -> ModelResponseStream:
    """A real provider chunk.

    ``ModelResponseStream`` rather than a hand-rolled stand-in, because the adapter reads a
    ``Delta`` through attribute access and a mock with the same field *names* would keep
    passing after the library changed the shape underneath it.
    """
    response = ModelResponseStream(
        choices=[
            StreamingChoices(index=0, delta=Delta(content=text or None), finish_reason=finish)
        ],
    )
    if usage is not None:
        response.usage = litellm.Usage(  # the library sets this on the final chunk
            prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens
        )
    return response


class FakeStream:
    """An async iterator of provider chunks that records whether it was closed."""

    def __init__(self, chunks: Sequence[Any], fail_at: int | None = None) -> None:
        self._chunks = list(chunks)
        self._fail_at = fail_at
        self._index = 0
        self.closed = False

    def __aiter__(self) -> FakeStream:
        return self

    async def __anext__(self) -> Any:
        if self._fail_at is not None and self._index == self._fail_at:
            raise litellm.exceptions.APIConnectionError(
                message="the connection dropped", llm_provider="ollama_chat", model="qwen2.5:14b"
            )
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        self._index += 1
        return self._chunks[self._index - 1]

    async def aclose(self) -> None:
        self.closed = True


def generator(**overrides: Any) -> tuple[LitellmGenerator, list[dict[str, Any]]]:
    """A generator whose provider call is captured rather than dialled."""
    calls: list[dict[str, Any]] = []
    streams: list[Any] = list(
        overrides.pop("streams", [FakeStream([chunk("hello"), chunk(finish="stop")])])
    )

    # A call that misbehaves in a way no scripted stream can — one that hangs, or refuses
    # before a stream exists at all.
    supplied: Any = overrides.pop("completion", None)

    async def completion(**kwargs: Any) -> Any:
        calls.append(kwargs)
        if supplied is not None:
            return await supplied(**kwargs)
        outcome = streams.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    settings = LlmSettings(**overrides.pop("settings", {}))
    return LitellmGenerator(settings, completion=completion, **overrides), calls


async def drain(stream: AsyncIterator[Any]) -> list[Any]:
    return [token async for token in stream]


# --- the ollama_chat trap ------------------------------------------------------------------


def test_ollama_is_reached_through_the_chat_endpoint_not_the_generate_one() -> None:
    """``ollama/`` double-templates: the library flattens ``messages`` with its own template
    and Ollama applies the model's real one on top. It still answers, plausibly, and follows
    the citation protocol worse — so the only symptom is more dropped markers."""
    assert compose_model("ollama", "qwen2.5:14b") == "ollama_chat/qwen2.5:14b"
    assert compose_model("OpenAI", "gpt-4o-mini") == "openai/gpt-4o-mini"
    assert compose_model("anthropic", "claude-sonnet-4") == "anthropic/claude-sonnet-4"


def test_the_composed_ollama_string_is_one_the_library_actually_routes() -> None:
    """Checked against the library rather than asserted, because the prefix is the whole
    point and a rename upstream must fail here rather than in production."""
    model, provider, _, _ = litellm.get_llm_provider(model=compose_model("ollama", "qwen2.5:14b"))

    assert (model, provider) == ("qwen2.5:14b", "ollama_chat")


async def test_ollama_gets_an_explicit_window_and_a_keep_alive() -> None:
    """``num_ctx`` is derived from the profile budget, so the served window stops varying
    with the host's available memory."""
    profile = PROFILES[RetrievalProfile.FAST]
    gen, calls = generator(profile=profile, system_prompt_tokens=400)

    await drain(gen.stream([{"role": "user", "content": "hi"}]))

    assert calls[0]["num_ctx"] == profile.context_tokens + profile.history_tokens + 400 + 1024
    assert calls[0]["keep_alive"] == "10m"
    assert calls[0]["stream_options"] == {"include_usage": True}


async def test_a_hosted_provider_is_not_told_about_ollama_options() -> None:
    gen, calls = generator(settings={"provider": "openai", "model": "gpt-4o-mini"}, api_key="k")

    await drain(gen.stream([{"role": "user", "content": "hi"}]))

    assert "num_ctx" not in calls[0]
    assert "keep_alive" not in calls[0]


async def test_manicules_own_parameters_cannot_be_overridden_by_provider_extras() -> None:
    """Each of them is a guarantee this module makes; a config file must not switch one off."""
    gen, calls = generator(extra_params={"stream": False, "temperature": 1.9, "seed": 7})

    await drain(gen.stream([{"role": "user", "content": "hi"}]))

    assert calls[0]["stream"] is True
    assert calls[0]["temperature"] == 0.2
    assert calls[0]["seed"] == 7, "an extra the adapter does not set still gets through"


# --- errors ---------------------------------------------------------------------------------


def test_every_provider_exception_maps_to_its_own_error_and_not_to_the_base_one() -> None:
    """``ContextWindowExceededError`` and ``ContentPolicyViolationError`` both subclass
    ``BadRequestError``, so a generic arm above them deletes two actionable diagnoses."""
    cases = {
        litellm.exceptions.ContextWindowExceededError(
            message="too long", model="m", llm_provider="openai"
        ): ContextWindowError,
        litellm.exceptions.ContentPolicyViolationError(
            message="refused", model="m", llm_provider="openai"
        ): ContentFilteredError,
        litellm.exceptions.AuthenticationError(
            message="bad key", model="m", llm_provider="openai"
        ): ProviderAuthError,
        litellm.exceptions.RateLimitError(
            message="slow down", model="m", llm_provider="openai"
        ): ProviderRateLimitError,
        litellm.exceptions.APIConnectionError(
            message="unreachable", model="m", llm_provider="openai"
        ): ProviderConnectionError,
        litellm.exceptions.BadRequestError(
            message="nope", model="m", llm_provider="openai"
        ): ProviderRequestError,
    }
    for raised, expected in cases.items():
        assert type(map_error(raised, "m")) is expected


def test_importing_the_provider_library_does_not_fetch_a_model_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other lazy fetch on the query path, and the only one that fails silently.

    Importing ``litellm`` performs an HTTPS GET to raw.githubusercontent.com for its pricing
    and context-window table, on the path that answers a question, and falls back to the copy
    in its own wheel when that fails. So an air-gapped host paid five seconds a process for
    the local table it was going to use anyway, and a networked host used whichever revision
    that repository happened to be serving. Both now read the table pinned in the lockfile.

    Driven through ``error_table``, which is one of the four public functions that import the
    library, rather than through the importer directly: what has to hold is that *reaching*
    the library sets this, and a test that called the importer would keep passing if a caller
    grew its own ``import litellm``.
    """
    monkeypatch.delenv(LOCAL_COST_MAP_ENV, raising=False)

    error_table()

    assert os.environ[LOCAL_COST_MAP_ENV] == "True"


def test_an_operator_who_wants_the_live_model_table_keeps_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A default is not a policy. Somebody running hosted models against new releases has a
    reason to want the current table, and setting the variable is how they say so."""
    monkeypatch.setenv(LOCAL_COST_MAP_ENV, "False")

    error_table()

    assert os.environ[LOCAL_COST_MAP_ENV] == "False"


def test_the_mapping_table_is_ordered_most_specific_first() -> None:
    """The ordering is the assertion. A correct-looking reshuffle is exactly how the two
    subclasses of ``BadRequestError`` start reporting as a generic bad request."""
    order = [mapped for _, mapped in error_table()]
    specific = order.index(ContextWindowError)
    generic = order.index(ProviderRequestError)

    assert specific < generic
    assert order.index(ContentFilteredError) < generic


def test_an_error_keeps_the_provider_s_own_message_text() -> None:
    """Discarding it is how "OpenAI error: 429" becomes an operator's whole afternoon."""
    mapped = map_error(
        litellm.exceptions.RateLimitError(
            message="quota exhausted, retry after 30s", model="m", llm_provider="openai"
        ),
        "openai/m",
    )

    assert "quota exhausted, retry after 30s" in str(mapped)


def test_an_authentication_error_names_the_variable_that_was_read() -> None:
    mapped = map_error(
        litellm.exceptions.AuthenticationError(message="bad key", model="m", llm_provider="openai"),
        "openai/m",
        "OPENAI_API_KEY",
    )

    assert "OPENAI_API_KEY" in str(mapped)


# --- retries and timeouts --------------------------------------------------------------------


async def test_a_connection_failure_before_the_first_token_is_retried() -> None:
    """Nothing has been delivered, so a retry is invisible and correct."""
    failure = litellm.exceptions.APIConnectionError(
        message="refused", llm_provider="ollama_chat", model="m"
    )
    gen, calls = generator(
        streams=[failure, FakeStream([chunk("hello"), chunk(finish="stop")])],
        settings={"max_retries": 2},
    )

    tokens = await drain(gen.stream([{"role": "user", "content": "hi"}]))

    assert len(calls) == 2
    assert tokens[0].text == "hello"


async def test_a_failure_after_the_first_token_is_terminal_and_reported_in_band() -> None:
    """Restarting makes the reader watch the answer rewind; continuing splices two
    independently-sampled answers into text no single generation produced."""
    gen, calls = generator(
        streams=[FakeStream([chunk("half an "), chunk("answer")], fail_at=2)],
        settings={"max_retries": 5},
    )

    tokens = await drain(gen.stream([{"role": "user", "content": "hi"}]))

    assert len(calls) == 1, "no retry may happen once text has been delivered"
    assert [token.text for token in tokens[:2]] == ["half an ", "answer"]
    assert tokens[-1].finish_reason is FinishReason.ERROR
    assert tokens[-1].error


async def test_an_authentication_failure_is_never_retried() -> None:
    """A bad API key does not improve by being presented three times."""
    failure = litellm.exceptions.AuthenticationError(
        message="bad key", llm_provider="openai", model="m"
    )
    gen, calls = generator(
        streams=[failure, failure, failure],
        settings={"provider": "openai", "model": "m", "max_retries": 2},
        api_key="k",
    )

    with pytest.raises(ProviderAuthError):
        await drain(gen.stream([{"role": "user", "content": "hi"}]))

    assert len(calls) == 1


async def test_a_provider_that_opens_a_stream_and_stops_sending_is_an_error_not_a_hang() -> None:
    """The gap between two tokens is the interval a single connect budget does not cover."""

    class Stalling(FakeStream):
        @override
        async def __anext__(self) -> Any:
            if self._index == 0:
                self._index += 1
                return chunk("start")
            await asyncio.Event().wait()
            raise AssertionError  # pragma: no cover

    gen, _ = generator(
        streams=[Stalling([])], settings={"stream_idle_timeout_s": 0.05, "max_retries": 0}
    )

    tokens = await drain(gen.stream([{"role": "user", "content": "hi"}]))

    assert tokens[-1].finish_reason is FinishReason.ERROR
    assert "gap between tokens" in (tokens[-1].error or "")


async def test_the_provider_connection_is_closed_on_every_exit_path() -> None:
    stream = FakeStream([chunk("a"), chunk("b"), chunk(finish="stop")])
    gen, _ = generator(streams=[stream])

    tokens = gen.stream([{"role": "user", "content": "hi"}])
    assert (await anext(tokens)).text == "a"
    await aclose(tokens)

    assert stream.closed is True, "an abandoned stream holds a model that is still working"


# --- usage and drift --------------------------------------------------------------------------


async def test_a_reported_usage_reaches_the_final_token() -> None:
    gen, _ = generator(
        streams=[
            FakeStream(
                [
                    chunk("hi"),
                    chunk(finish="stop", usage=Usage(prompt_tokens=120, completion_tokens=4)),
                ]
            )
        ]
    )

    tokens = await drain(gen.stream([{"role": "user", "content": "hi"}]))

    assert tokens[-1].usage == Usage(prompt_tokens=120, completion_tokens=4)


def test_a_usage_figure_that_could_be_an_estimator_is_treated_as_unavailable() -> None:
    """A calibration loop fed an estimate on both sides agrees with itself forever and
    reports excellent health."""
    assert usable_prompt_tokens(Usage(prompt_tokens=0, completion_tokens=9), 100) is None
    assert usable_prompt_tokens(Usage(prompt_tokens=100, completion_tokens=9), 100) is None
    assert usable_prompt_tokens(Usage(prompt_tokens=118, completion_tokens=9), 100) == 118
    assert usable_prompt_tokens(None, 100) is None


def test_drift_beyond_tolerance_is_reported_with_both_numbers_and_never_auto_tuned() -> None:
    problem = drift_problem(estimate=100, measured=200, tolerance=0.15, model="ollama_chat/q")

    assert "100" in problem
    assert "200" in problem
    assert "ollama_chat/q" in problem
    assert GENERATION_ENCODING in problem
    assert drift_problem(estimate=105, measured=100, tolerance=0.15, model="m") == ""
    assert drift_problem(estimate=105, measured=None, tolerance=0.15, model="m") == ""


# --- the window cross-check ---------------------------------------------------------------------


def test_a_profile_that_does_not_fit_is_refused_with_both_numbers_named() -> None:
    """The predicate belongs to retrieval; this ticket owns the enforcement point.

    Deliberately not a second copy of the rule. One rule stated twice is two rules that will
    disagree, and the one that disagrees silently is the one that lets a prompt overflow.
    """
    problem = window_problem(
        PROFILES[RetrievalProfile.PRECISE],
        context_window=8192,
        model_id="ollama_chat/qwen2.5:14b",
        system_prompt_tokens=400,
        generation_reserve=1024,
    )

    assert problem is not None
    assert "8192" in problem
    assert "longer window" in problem
    assert "context_tokens" in problem


@pytest.mark.parametrize(
    "profile", [RetrievalProfile.FAST, RetrievalProfile.BALANCED, RetrievalProfile.PRECISE]
)
def test_every_shipped_profile_fits_a_sixteen_thousand_token_window(
    profile: RetrievalProfile,
) -> None:
    """The budgets are derived from what each ``final_top_k`` can actually hold, so a profile
    that cannot fit any plausible model is now a configuration question rather than an
    arithmetic impossibility."""
    assert (
        window_problem(
            PROFILES[profile],
            context_window=16384,
            model_id="m",
            system_prompt_tokens=400,
            generation_reserve=1024,
        )
        is None
    )


async def test_a_generator_whose_profile_does_not_fit_refuses_at_startup() -> None:
    """The enforcement point is here because this is where the served window becomes known."""
    gen, _ = generator(
        settings={"provider": "openai", "model": "gpt-4o-mini", "context_window": 4096},
        api_key="k",
        profile=PROFILES[RetrievalProfile.PRECISE],
        system_prompt_tokens=400,
    )

    with pytest.raises(ConfigError, match="4096"):
        await gen.setup()


async def test_a_model_the_library_does_not_recognise_is_refused_with_the_prefixes_listed() -> None:
    gen, _ = generator(settings={"provider": "not-a-provider", "model": "m"})

    with pytest.raises(ConfigError, match="does not recognise"):
        await gen.setup()


async def test_a_hosted_provider_without_a_credential_is_refused_before_any_call() -> None:
    gen, calls = generator(settings={"provider": "openai", "model": "gpt-4o-mini"})

    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        await gen.setup()

    assert calls == [], "a live probe costs money and latency on every start"


def test_the_estimator_uses_an_encoding_name_rather_than_a_model_it_is_not_running() -> None:
    """Naming a model that is not being used makes the estimate look authoritative."""
    estimator = TokenEstimator(safety_factor=1.0)

    assert estimator.encoding_name == "o200k_base"
    assert estimator.count("hello world") == estimator.raw_count("hello world") + 1


def test_the_safety_factor_biases_towards_overcounting() -> None:
    """Undercounting overflows the window and gets the context truncated by the server, which
    is the silent failure; overcounting costs a passage."""
    text = "a longer sentence with several words in it, repeated. " * 20

    assert TokenEstimator(safety_factor=1.5).count(text) > TokenEstimator(safety_factor=1.0).count(
        text
    )


def test_chunk_counts_are_cached_by_content_derived_id() -> None:
    estimator = TokenEstimator()
    first = estimator.count_chunk(make_chunk())
    second = estimator.count_chunk(make_chunk())

    assert first == second > 0
