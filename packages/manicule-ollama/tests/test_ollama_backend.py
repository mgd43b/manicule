"""The backend against a synthetic server, including every way a server can be wrong.

This is where the tier B measurement story is actually held. :class:`~manicule_ollama.backend.
OllamaEmbedder` makes six claims about a server it does not control, and each one is checked
here by turning exactly that thing wrong and asserting the refusal — a vocabulary that is off
by one token, a ``truncate`` flag the server ignores, a context served below the one declared,
vectors that are not unit length, a ``prompt_eval_count`` that never arrives, and a model
re-pulled underneath a running process.

The conformance suites manicule publishes run here too. A backend in another distribution
passes the same ones an in-tree backend does; what it does *not* pass is
:class:`~manicule.core.protocols.TokenStateEmbedder`, and that is asserted rather than left as
an absence.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from manicule_ollama.backend import OllamaEmbedder
from manicule_ollama.client import OllamaClient, OllamaUnavailableError
from manicule_ollama.config import OllamaEmbedderConfig
from manicule_ollama.served import resolve
from ollama_fake import DIGEST, MODEL, SPECIAL_TOKENS, FakeOllama, config_payload

from manicule.core.errors import ConfigError, ContextOverflowError
from manicule.core.protocols import Embedder, TokenStateEmbedder
from manicule.embedding.runtimes.tokenization import FastTokenizer
from manicule.testing import (
    assert_embedder_contract,
    assert_protocol_signatures,
    assert_refuses_oversized_chunks,
    write_tokenizer,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def vocabulary(tmp_path: Path) -> Path:
    directory = tmp_path / "vocabulary"
    directory.mkdir()
    write_tokenizer(directory / "tokenizer.json")
    return directory


@pytest.fixture
def counter(vocabulary: Path) -> Callable[[str], int]:
    """The server's counter, which is the model's own vocabulary rather than a stand-in."""
    tokenizer = FastTokenizer(vocabulary / "tokenizer.json")

    def count(text: str) -> int:
        return len(tokenizer.content_ids(text))

    return count


def make(server: FakeOllama, vocabulary: Path, **overrides: object) -> OllamaEmbedder:
    """Resolve and construct, without setting up — so a setup failure is the assertion."""
    config = OllamaEmbedderConfig.model_validate(
        config_payload(tokenizer=str(vocabulary), **overrides)
    )
    transport = server.transport()
    client = OllamaClient(config.base_url, transport=transport, async_transport=transport)
    served = resolve(client, server.model, config)
    return OllamaEmbedder(
        served,
        client,
        keep_alive=config.keep_alive,
        batch_size=2,
        cache_entries=128,
    )


async def ready(server: FakeOllama, vocabulary: Path, **overrides: object) -> OllamaEmbedder:
    embedder = make(server, vocabulary, **overrides)
    await embedder.setup()
    return embedder


# --- the shipped conformance suites ----------------------------------------------------------


async def test_the_ollama_backend_meets_the_shipped_conformance_suites(
    vocabulary: Path, counter: Callable[[str], int]
) -> None:
    """manicule publishes these; a backend in another distribution passes the same ones."""
    embedder = await ready(FakeOllama(count=counter), vocabulary)

    assert isinstance(embedder, Embedder)
    assert_protocol_signatures(embedder, Embedder)
    await assert_embedder_contract(embedder)
    await assert_refuses_oversized_chunks(embedder.embed_chunks, embedder)
    await embedder.teardown()


async def test_this_is_a_tier_b_backend_and_says_so(
    vocabulary: Path, counter: Callable[[str], int]
) -> None:
    """It exposes no token states, and that absence is the admission tier B makes.

    `TokenStateEmbedder` is "preferred wherever available" because manicule pools what it is
    given rather than trusting a provider's own reduction. Here the server pools and hands back
    a finished vector, so the reduction cannot be verified by inspection — only measured. A
    backend claiming the richer protocol while returning pooled output under the token-state
    name is the exact bug `docs/embeddings.md` §3.2 records a shipped library committing, so
    the claim is asserted false rather than merely not made.
    """
    embedder = await ready(FakeOllama(count=counter), vocabulary)

    assert not isinstance(embedder, TokenStateEmbedder)
    assert not hasattr(embedder, "encode")
    await embedder.teardown()


# --- what setup checks against the server ----------------------------------------------------


async def test_setup_refuses_a_tokenizer_that_disagrees_by_one_token(
    vocabulary: Path, counter: Callable[[str], int]
) -> None:
    """The check that turns a configured vocabulary from an inference into a measurement.

    One token is the whole margin. `tokenizer_id` is an identity field of `ChunkFingerprint`,
    and a budget measured with the wrong vocabulary undercounts — which is the direction that
    truncates.
    """
    embedder = make(FakeOllama(count=counter, count_offset=1), vocabulary)

    with pytest.raises(ConfigError, match="not the same vocabulary"):
        await embedder.setup()
    await embedder.teardown()


async def test_setup_refuses_a_server_that_ignores_truncate_false(
    vocabulary: Path, counter: Callable[[str], int]
) -> None:
    """The single assumption the whole backend rests on, checked rather than assumed.

    With `truncate: false` in force an over-long input is a 400. Without it the server shortens
    the input and answers with a well-formed vector describing its opening, and nothing
    anywhere raises — which is precisely the silent failure `require_within_context` exists to
    catch and the one this backend has no second guard for.
    """
    server = FakeOllama(count=counter, honors_truncate=False)
    embedder = make(server, vocabulary)

    with pytest.raises(ConfigError, match="is not being honored"):
        await embedder.setup()
    await embedder.teardown()


async def test_setup_refuses_a_server_reading_less_than_the_fingerprint_claims(
    vocabulary: Path, counter: Callable[[str], int]
) -> None:
    """The dangerous direction, caught before a corpus is built rather than inside one.

    The chunker has already been bound to this limit at construction. A server serving less
    than it means every full-length chunk is truncated, so the refusal has to happen here.
    """
    server = FakeOllama(count=counter, context_length=64, served_ceiling=20)
    embedder = make(server, vocabulary)

    with pytest.raises(ConfigError, match="reading less than this backend derived"):
        await embedder.setup()
    await embedder.teardown()


async def test_setup_refuses_a_server_whose_vectors_are_not_normalized(
    vocabulary: Path, counter: Callable[[str], int]
) -> None:
    """`normalized=True` is recorded, and every cosine score manicule computes assumes it.

    Normalizing an unnormalized model here would hide the disagreement rather than resolve it —
    the vectors would be unit length and still come from a model whose training did not assume
    that, which is a different space presented as the same one.
    """
    embedder = make(FakeOllama(count=counter, normalize=False), vocabulary)

    with pytest.raises(ConfigError, match="not 1"):
        await embedder.setup()
    await embedder.teardown()


async def test_setup_refuses_a_server_that_reports_no_token_count(
    vocabulary: Path, counter: Callable[[str], int]
) -> None:
    """Without `prompt_eval_count` the configured vocabulary cannot be checked at all.

    And this backend will not count tokens with an unverified vocabulary: the count is what
    places chunk boundaries and what refuses text the model would truncate.
    """
    embedder = make(FakeOllama(count=counter, report_counts=False), vocabulary)

    with pytest.raises(ConfigError, match="no prompt_eval_count"):
        await embedder.setup()
    await embedder.teardown()


# --- what every request carries ---------------------------------------------------------------


async def test_every_embedding_request_sends_num_ctx_and_refuses_truncation(
    vocabulary: Path, counter: Callable[[str], int]
) -> None:
    """Both flags, on every request, because both are load-bearing and neither is a default.

    Ollama's own `truncate` default is **true**, and its default context is not the model's:
    measured against a server holding `qwen3-embedding:0.6b`, which declares 32768, an
    /api/embed carrying no options was served at 4096.
    """
    server = FakeOllama(count=counter, context_length=64)
    embedder = await ready(server, vocabulary)
    server.requests.clear()

    await embedder.embed(["alpha beta", "gamma delta"])

    embeds = [body for path, body in server.requests if path == "/api/embed"]
    assert embeds
    assert all(body["truncate"] is False for body in embeds)
    assert all(body["options"] == {"num_ctx": 64} for body in embeds)
    await embedder.teardown()


async def test_oversized_text_is_refused_here_so_the_message_can_name_it(
    vocabulary: Path, counter: Callable[[str], int]
) -> None:
    """Belt and braces, and the braces are what names the offending text.

    The server would refuse this batch too — that is what `truncate: false` is for — but only
    this side still has the batch, so only this side can say which text to shorten.
    """
    embedder = await ready(FakeOllama(count=counter), vocabulary)
    limit = embedder.fingerprint.max_sequence_length

    with pytest.raises(ContextOverflowError, match="text 1"):
        await embedder.embed(["alpha", " ".join(["alpha"] * (limit + 5))])
    await embedder.teardown()


# --- what comes back ---------------------------------------------------------------------------


async def test_vectors_are_the_declared_width_and_exactly_unit_length(
    vocabulary: Path, counter: Callable[[str], int]
) -> None:
    """The two properties a tier B backend can check about output it did not produce."""
    embedder = await ready(FakeOllama(count=counter, embedding_length=12), vocabulary)

    vectors = await embedder.embed(["alpha beta", "gamma"])

    assert embedder.fingerprint.dimension == 12
    for vector in vectors:
        assert len(vector) == 12
        assert abs(sum(value * value for value in vector) - 1.0) < 1e-9
    await embedder.teardown()


async def test_a_short_answer_is_refused_rather_than_realigned(
    vocabulary: Path, counter: Callable[[str], int]
) -> None:
    """Vectors are matched to texts positionally, so a missing one cannot be recovered from.

    Filling the gap, or zipping what arrived against the first N texts, attaches every later
    vector to the wrong text — an index that accepts writes and answers confidently.
    """
    embedder = await ready(FakeOllama(count=counter, short_answer=True), vocabulary)

    with pytest.raises(OllamaUnavailableError, match="matched positionally"):
        await embedder.embed(["alpha", "beta", "gamma"])
    await embedder.teardown()


async def test_embedding_is_deterministic_batch_invariant_and_cached(
    vocabulary: Path, counter: Callable[[str], int]
) -> None:
    """A text's vector does not depend on what shared its batch, and the cache is exact.

    Batch dependence is what an unmasked mean pool produces, and it is invisible without this
    test because every individual vector looks fine. It cannot be ruled out by inspection on a
    backend that pools elsewhere, which is why it is measured.
    """
    server = FakeOllama(count=counter)
    embedder = await ready(server, vocabulary)

    alone = (await embedder.embed(["alpha beta"]))[0]
    crowded = (await embedder.embed(["gamma", "alpha beta", "delta epsilon zeta"]))[1]
    server.requests.clear()
    again = (await embedder.embed(["alpha beta"]))[0]

    assert list(alone) == list(crowded) == list(again)
    assert not [path for path, _ in server.requests if path == "/api/embed"], (
        "the third call should have been served from the cache, which is keyed on the "
        "canonical fingerprint rather than on the model's name"
    )
    await embedder.teardown()


# --- the lifecycle -------------------------------------------------------------------------------


async def test_health_fails_when_the_model_is_re_pulled_underneath_a_running_process(
    vocabulary: Path, counter: Callable[[str], int]
) -> None:
    """The failure an in-process backend does not have, and the reason the digest is in identity.

    `ollama pull` replaces a blob under an unchanged name. Nothing in manicule's own state
    changes, so without this check every vector written afterwards joins a table built from a
    different space and the index quietly stops agreeing with itself.
    """
    server = FakeOllama(count=counter)
    embedder = await ready(server, vocabulary)
    assert (await embedder.health()).ok

    server.digest = "f" * 64

    report = await embedder.health()
    assert not report.ok
    assert DIGEST in report.detail
    assert "re-pulled" in report.detail
    await embedder.teardown()


async def test_health_fails_before_setup_has_verified_anything(
    vocabulary: Path, counter: Callable[[str], int]
) -> None:
    embedder = make(FakeOllama(count=counter), vocabulary)

    report = await embedder.health()

    assert not report.ok
    assert "has not been verified" in report.detail
    await embedder.teardown()


async def test_teardown_is_safe_twice_and_after_a_failed_setup(
    vocabulary: Path, counter: Callable[[str], int]
) -> None:
    """Which is when it is most needed: a setup that refused the server still holds a pool."""
    embedder = make(FakeOllama(count=counter, count_offset=1), vocabulary)
    with pytest.raises(ConfigError):
        await embedder.setup()

    await embedder.teardown()
    await embedder.teardown()

    assert not (await embedder.health()).ok


async def test_metrics_publish_the_servers_own_measure_of_the_work(
    vocabulary: Path, counter: Callable[[str], int]
) -> None:
    """`ollama_prompt_tokens` is the server's count, not this backend's estimate of it.

    Which is what makes it worth publishing: divided by the texts embedded it is the mean
    tokens actually evaluated, and that is the number that would stop moving if inputs began
    being truncated.
    """
    embedder = await ready(FakeOllama(count=counter), vocabulary)
    await embedder.embed(["alpha beta gamma"])

    published = {metric.name: metric.value for metric in embedder.metrics()}

    assert published["embedding_texts_embedded"] == 1
    assert published["ollama_prompt_tokens"] >= 3 + SPECIAL_TOKENS
    assert published["ollama_num_ctx"] > 0
    await embedder.teardown()


async def test_the_factory_refuses_configuration_of_the_wrong_type() -> None:
    """A factory reached from outside the container has to supply the model it declares.

    Falling back to the shared model's defaults would silently drop `base_url`, `tokenizer` and
    `num_ctx` — every setting that decides where the vectors come from and what they mean.
    """
    from manicule_ollama import build_ollama  # noqa: PLC0415

    from manicule.config.settings import Settings  # noqa: PLC0415
    from manicule.embedding.config import EmbedderConfig  # noqa: PLC0415
    from manicule.plugins import BuildContext  # noqa: PLC0415

    settings = Settings()
    context = BuildContext(
        settings=settings,
        config=EmbedderConfig(),
        data_dir=settings.data_dir,
        cache_dir=settings.cache_dir,
        components=None,  # pyright: ignore[reportArgumentType] - unreached; the type check is first
    )

    with pytest.raises(ConfigError, match="would not be applied"):
        build_ollama(context)


def test_the_plugin_registers_under_the_name_its_entry_point_uses() -> None:
    """The registry rejects a manifest name that disagrees with its entry point rather than
    resolving it, so the two are held together here."""
    import tomllib  # noqa: PLC0415

    from manicule_ollama import OLLAMA_NAME, PLUGIN  # noqa: PLC0415

    root = Path(__file__).resolve().parents[1]
    declared = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    entry_points = declared["project"]["entry-points"]["manicule.plugins"]

    assert PLUGIN.manifest.name == OLLAMA_NAME
    assert set(entry_points) == {OLLAMA_NAME}
    assert entry_points[OLLAMA_NAME] == "manicule_ollama:PLUGIN"
    assert MODEL  # the synthetic model name is exercised above; keep the import honest


# --- the one measurement that is remembered ----------------------------------------------------


def _at_limit_probes(server: FakeOllama, embedder: OllamaEmbedder) -> int:
    """How many requests carried a probe at exactly the advertised limit."""
    limit = embedder.fingerprint.max_sequence_length
    sent = 0
    for path, body in server.requests:
        if path != "/api/embed":
            continue
        inputs = body.get("input")
        if not isinstance(inputs, list):
            continue
        texts = cast("list[object]", inputs)
        if any(len(str(text).split()) == limit for text in texts):
            sent += 1
    return sent


async def test_the_full_context_probe_runs_once_per_configuration(
    vocabulary: Path, counter: Callable[[str], int], tmp_path: Path
) -> None:
    """Because it is a full-context forward pass — 40 seconds on a 32768-token model, measured.

    What it establishes does not change while the digest, the served context and the derived
    limit stay the same, so repeating it every start would be forty seconds bought over and
    over for one answer. The *free* half of the check — that a probe past the context is
    refused, which is what proves `truncate: false` is in force — runs every time regardless,
    and that is the half nothing else can catch.
    """
    server = FakeOllama(count=counter)
    cache = tmp_path / "cache"

    first = await _setup_with_cache(server, vocabulary, cache)
    assert _at_limit_probes(server, first) == 1
    await first.teardown()

    server.requests.clear()
    second = await _setup_with_cache(server, vocabulary, cache)
    assert _at_limit_probes(server, second) == 0, (
        "the expensive probe was repeated for a configuration nothing about which had changed"
    )
    await second.teardown()


async def test_the_full_context_probe_returns_when_the_model_is_re_pulled(
    vocabulary: Path, counter: Callable[[str], int], tmp_path: Path
) -> None:
    """A new digest is new bytes, and nothing measured about the old ones carries over."""
    server = FakeOllama(count=counter)
    cache = tmp_path / "cache"

    first = await _setup_with_cache(server, vocabulary, cache)
    await first.teardown()

    server.digest = "c" * 64
    server.requests.clear()
    second = await _setup_with_cache(server, vocabulary, cache)

    assert _at_limit_probes(server, second) == 1
    await second.teardown()


async def test_the_full_context_probe_returns_when_num_ctx_changes(
    vocabulary: Path, counter: Callable[[str], int], tmp_path: Path
) -> None:
    """The limit is derived from `num_ctx`, so a different one is a different claim."""
    server = FakeOllama(count=counter, context_length=64)
    cache = tmp_path / "cache"

    first = await _setup_with_cache(server, vocabulary, cache, num_ctx=64)
    await first.teardown()

    server.requests.clear()
    second = await _setup_with_cache(server, vocabulary, cache, num_ctx=32)

    assert _at_limit_probes(server, second) == 1
    await second.teardown()


async def _setup_with_cache(
    server: FakeOllama, vocabulary: Path, cache: Path, **overrides: object
) -> OllamaEmbedder:
    from manicule_ollama.served import record  # noqa: PLC0415

    config = OllamaEmbedderConfig.model_validate(
        config_payload(tokenizer=str(vocabulary), **overrides)
    )
    transport = server.transport()
    client = OllamaClient(config.base_url, transport=transport, async_transport=transport)
    served = resolve(client, server.model, config)
    record(served, client, cache)
    embedder = OllamaEmbedder(
        served, client, cache_dir=cache, keep_alive=config.keep_alive, batch_size=2
    )
    await embedder.setup()
    return embedder


async def test_health_reports_rather_than_raises_whatever_the_server_does(
    vocabulary: Path, counter: Callable[[str], int]
) -> None:
    """`SupportsHealth` says a health check reports instead of raising, and it means it.

    The caller is a diagnostic asking every component at once, so one that escapes takes down
    the surface that was about to say which component is unwell. A proxy answering `/api/tags`
    with a 400 whose body happens to carry the phrase this client maps to a context overflow is
    the narrow case, and the fix is to catch everything this package raises rather than the two
    things it usually raises here.
    """
    server = FakeOllama(count=counter)
    embedder = await ready(server, vocabulary)
    server.tags_error = "the input length exceeds the context length"

    report = await embedder.health()

    assert not report.ok
    assert "exceeds the context length" in report.detail
    await embedder.teardown()


async def test_a_moved_digest_stops_embedding_rather_than_only_being_reported(
    vocabulary: Path, counter: Callable[[str], int]
) -> None:
    """Ingest and retrieval call `embed` directly and never read a health report.

    So a check that noticed an `ollama pull` and let embedding continue would watch vectors
    from the new model being appended to the old model's index between health sweeps — under a
    fingerprint that says the model did not change, which is exactly what makes it unrecoverable
    by inspection. The first observation is final.
    """
    server = FakeOllama(count=counter)
    embedder = await ready(server, vocabulary)
    assert await embedder.embed(["alpha beta"])

    server.digest = "e" * 64
    assert not (await embedder.health()).ok

    with pytest.raises(ConfigError, match="Embedding stops here"):
        await embedder.embed(["gamma delta"])
    await embedder.teardown()


async def test_setup_refuses_a_non_finite_probe_rather_than_comparing_it(
    vocabulary: Path, counter: Callable[[str], int]
) -> None:
    """`abs(nan - 1.0) > tolerance` is **false**, so a NaN passes a tolerance check.

    Without a finiteness test first, this backend would record `normalized=True` about a vector
    that has no length at all — the comparison does not fail, it simply does not fire. The
    later guard in `_finish` would catch it on a real embedding, which is after the fingerprint
    has been trusted.
    """
    with pytest.raises(ConfigError, match="non-finite component"):
        await ready(FakeOllama(count=counter, non_finite=True), vocabulary)


async def test_a_component_that_is_not_a_number_becomes_this_packages_error(
    vocabulary: Path, counter: Callable[[str], int]
) -> None:
    """Everything this client raises is one of its own errors, so the backend's mapping covers it.

    A `null` inside the array would otherwise arrive as a bare `TypeError` from inside a
    comprehension — a traceback from a library the operator did not choose, in the middle of an
    ingest, saying nothing about which server sent what.
    """
    # Raised by the dimension probe, which is the first vector this backend ever reads — so
    # the refusal lands at construction, before a fingerprint exists to be wrong about.
    with pytest.raises(OllamaUnavailableError, match="A vector is a list of numbers"):
        await ready(FakeOllama(count=counter, null_component=True), vocabulary)


async def test_a_boolean_component_is_refused_rather_than_coerced(
    vocabulary: Path, counter: Callable[[str], int]
) -> None:
    """`float(True)` is `1.0`, so a JSON `true` would become a perfectly plausible component.

    Which is the whole problem: nothing downstream could tell it from a real one. The type is
    checked before the conversion rather than the conversion being allowed to fail, because for
    a boolean it does not fail.
    """
    with pytest.raises(OllamaUnavailableError, match="bool"):
        await ready(FakeOllama(count=counter, boolean_component=True), vocabulary)


async def test_normalization_is_checked_on_every_setup_not_only_an_uncached_one(
    vocabulary: Path, counter: Callable[[str], int], tmp_path: Path
) -> None:
    """The hole the ceiling cache opened, and the reason that cache needed a second look.

    `_verify_the_limit_is_reachable` is where the norm was checked, and it returns early once
    the ceiling is recorded — so on every start after the first, nothing inspected a vector
    before ingest did. A server that stopped normalizing between runs would then be met by
    `_finish`, which normalizes rather than refuses, and `normalized=True` would go on being
    recorded about vectors that were not.
    """
    server = FakeOllama(count=counter)
    cache = tmp_path / "cache"

    first = await _setup_with_cache(server, vocabulary, cache)
    await first.teardown()

    server.normalize = False
    with pytest.raises(ConfigError, match="not 1"):
        await _setup_with_cache(server, vocabulary, cache)
