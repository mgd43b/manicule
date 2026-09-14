"""A real Ollama, and the measurements that license every claim this backend makes.

**This file is not a parity suite, and the difference from ``manicule-mlx``'s is the point.**
That one exists to earn something: ``EmbedFingerprint`` leaves ``backend`` out of identity, and
a measurement showing MLX and onnxruntime agree to cosine 1.000000 is what licenses the
exclusion. Nothing here earns a shared identity with anything. These models are not the model
the in-tree backends run, the vectors are not comparable to theirs, and
``weights_identity`` says so — it carries the backend's name, so no fingerprint this package
produces can equal one produced anywhere else. Portability between backends is an allowlisted
measurement, and this backend is not on the list.

What *is* measured here is everything the backend asserts about a server it does not control.
Four of them cannot be shown against a synthetic server at all, because they are facts about
Ollama rather than about this code:

* that a Hugging Face tokenizer reproduces the served GGUF's token counts **exactly**, which is
  the whole basis for ``tokenizer`` being configuration rather than a guess;
* that ``truncate: false`` really is refused rather than accepted-and-shortened;
* that the declared context is **not** the served one — ``qwen3-embedding:0.6b`` declares 32768
  and is served at 4096 unless ``num_ctx`` is sent;
* that the silent truncation this backend exists to prevent is real, demonstrated by letting it
  happen.

Skipped without ``OLLAMA_TEST_URL``, because a developer with no Ollama should not have a red
checkout for a reason unrelated to their change. Failed instead when ``REQUIRE_OLLAMA`` is set:
a conformance suite that skips reports green having checked nothing.
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
from manicule_ollama.backend import TOKENIZER_PROBES, OllamaEmbedder
from manicule_ollama.client import OllamaClient, OllamaContextOverflowError, OllamaUnavailableError
from manicule_ollama.config import OllamaEmbedderConfig
from manicule_ollama.served import CONTEXT_RESERVE, record, resolve

from manicule.core.embedding import Pooling, Vector
from manicule.core.errors import ConfigError, ContextOverflowError
from manicule.core.protocols import Embedder, TokenStateEmbedder
from manicule.testing import (
    assert_embedder_contract,
    assert_protocol_signatures,
    assert_refuses_oversized_chunks,
)

pytestmark = pytest.mark.anyio

OLLAMA_URL_ENV: Final = "OLLAMA_TEST_URL"
"""Where a real Ollama is. Unset means "skip this file"."""

REQUIRE_OLLAMA_ENV: Final = "REQUIRE_OLLAMA_MODELS"
"""Which served models must be present, rather than whether any must be.

The same shape as ``REQUIRE_EMBEDDING_MODELS`` and for the same reason: the models this backend
was written against are not equally cheap to check. ``nomic-embed-text`` is 274 MB and a
2048-token context, which a CPU runner embeds in seconds; ``qwen3-embedding:0.6b`` is 639 MB
and a 32768-token context, whose full-length probe took 40 seconds on an M-series laptop with
the model already resident. A boolean switch would make CI either skip both or pay for both.
So CI names the one it seeded, and the other is exercised opportunistically wherever it is
already served.

Named outside manicule's ``MANICULE_`` namespace deliberately: ``manicule_environment`` deletes
every variable with that prefix before each test, so a switch named that way would be scrubbed
before it was read and the job would go green having skipped everything. That exact failure is
recorded in ``docs/embeddings.md`` §7, found by reading a green CI log.
"""

REQUIRED_MODELS: Final[frozenset[str]] = frozenset(
    name.strip() for name in os.environ.get(REQUIRE_OLLAMA_ENV, "").replace(",", " ").split()
)
"""Read at import, before any fixture has had a chance to touch the environment."""


def is_required(model: str) -> bool:
    """Whether a missing ``model`` is a failure rather than a skip.

    Matched on the name as configured *and* on the ``:latest`` Ollama appends to a bare one, so
    a job naming ``nomic-embed-text`` arms the case whatever the server calls it.
    """
    return bool(
        REQUIRED_MODELS
        & {model, model.split(":", maxsplit=1)[0], f"{model.split(':', maxsplit=1)[0]}:latest"}
    )


@dataclass(frozen=True, slots=True)
class Served:
    """One model this backend is known to work against, and the vocabulary that matches it."""

    model: str
    tokenizer: str
    revision: str
    pooling: Pooling
    dimension: int
    context_length: int
    special_tokens: int


MODELS: Final[tuple[Served, ...]] = (
    Served(
        model="qwen3-embedding:0.6b",
        tokenizer="Qwen/Qwen3-Embedding-0.6B",
        revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        # Declared `qwen3.pooling_type = 3`. Nothing in the model's name says so, and getting
        # it wrong is a fingerprint claiming a reduction the vectors did not come from.
        pooling=Pooling.LAST_TOKEN,
        dimension=1024,
        context_length=32768,
        # `add_bos_token = False`, `add_eos_token = True`. Confirmed against the server: an
        # empty input answered prompt_eval_count of 1.
        special_tokens=1,
    ),
    Served(
        model="nomic-embed-text",
        tokenizer="nomic-ai/nomic-embed-text-v1.5",
        revision="e9b6763023c676ca8431644204f50c2b100d9aab",
        pooling=Pooling.MEAN,
        dimension=768,
        # Declared 2048, while the model's *own Modelfile* carries `PARAMETER num_ctx 8192`.
        # The architecture wins and the server serves 2048 — which is why `num_ctx` is capped
        # by the declaration rather than trusted from configuration.
        context_length=2048,
        special_tokens=2,
    ),
)

TEXTS: Final[tuple[str, ...]] = (
    "The retention window is ninety days, after which archived pages are purged.",
    "El gato se sienta en la alfombra y mira por la ventana durante horas.",
    "def embed(texts): return [pool(model(text)) for text in texts]",
    " ".join(["paragraph"] * 200),
)
"""Short, non-English, code, and long — the same shape ``manicule-mlx``'s suite uses."""


MEASUREMENT_CACHE: Final = Path(tempfile.gettempdir()) / "manicule-ollama-measured-cache"
"""One declaration cache for the whole file, outside the per-test scratch directory.

``manicule_environment`` gives every test its own ``XDG_CACHE_HOME``, which is right for a
suite that must not read a developer's real state — and wrong for the one measurement here
that is deliberately made once. Without a shared directory the full-context probe would run
for every case in this file: forty seconds each on ``qwen3-embedding:0.6b``, measured, to
re-establish something that had not changed.
"""


def server_url() -> str | None:
    return os.environ.get(OLLAMA_URL_ENV, "").strip() or None


def require_ollama() -> str:
    """The server's URL — or skip, or fail when a job has named a model it must serve.

    Named after this backend rather than generically, and that is not style.
    ``tests/test_ci_switches`` decides which modules a ``REQUIRE_*`` switch governs by searching
    every test file for the *names* of the gate functions that skip or fail — so a gate here
    sharing a name with another support module's gate reports this file as governed by that
    module's switch, which is exactly the kind of wrong mapping that check exists to catch.
    """
    url = server_url()
    if url:
        return url
    detail = f"{OLLAMA_URL_ENV} is not set, so no Ollama server is available"
    if REQUIRED_MODELS:
        pytest.fail(
            f"{detail}, and {REQUIRE_OLLAMA_ENV} names {sorted(REQUIRED_MODELS)}. Every claim "
            f"this backend makes is about a server; with none of them checked the suite "
            f"certifies nothing."
        )
    pytest.skip(detail)


def require_served(client: OllamaClient, served: Served) -> None:
    """Skip — or fail, when a job named it — unless this server actually holds ``served``.

    Every case in this file needs it, including the two that drive the client directly rather
    than through an embedder. They were the two that did not have it, and on a server holding
    only the model CI seeds they failed with a 404 from `/api/embed` instead of skipping: a
    test-harness defect that reads exactly like a backend defect, which is the reason the
    decision lives in one function rather than at each call site.
    """
    try:
        client.describe(served.model)
    except ConfigError as exc:
        if is_required(served.model):
            pytest.fail(
                f"{client.base_url} cannot serve {served.model!r}, which "
                f"{REQUIRE_OLLAMA_ENV} names: {exc}"
            )
        pytest.skip(f"{client.base_url} does not serve {served.model!r}: {exc}")


async def embedder_for(
    served: Served, *, cache_entries: int = 10_000, **overrides: object
) -> OllamaEmbedder:
    """A set-up embedder, or a skip naming the model this machine's server does not hold."""
    url = require_ollama()
    config = OllamaEmbedderConfig.model_validate(
        {
            "base_url": url,
            "tokenizer": served.tokenizer,
            "tokenizer_revision": served.revision,
            **overrides,
        }
    )
    client = OllamaClient(
        url, timeout_s=config.timeout_s, connect_timeout_s=config.connect_timeout_s
    )
    try:
        require_served(client, served)
        resolved = resolve(client, served.model, config)
    except OllamaUnavailableError as exc:
        await client.aclose()
        pytest.skip(f"{url} is not answering: {exc}")
    except BaseException:
        await client.aclose()
        raise
    record(resolved, client, MEASUREMENT_CACHE)
    embedder = OllamaEmbedder(
        resolved,
        client,
        cache_dir=MEASUREMENT_CACHE,
        keep_alive=config.keep_alive,
        batch_size=4,
        cache_entries=cache_entries,
    )
    try:
        await embedder.setup()
    except BaseException:
        # The caller never receives this embedder, so its own `finally` cannot reach it — and
        # tokenizer or context validation raising here is the *expected* path in several cases
        # below, not an exotic one. Without this the connection pool outlives every such test.
        await embedder.teardown()
        raise
    return embedder


def server_tokens(embedder: OllamaEmbedder) -> int:
    """The server's own running count of the tokens it has read, from the published metrics."""
    return int(
        next(metric.value for metric in embedder.metrics() if metric.name == "ollama_prompt_tokens")
    )


def cosine(left: Vector, right: Vector) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def norm(vector: Vector) -> float:
    return math.sqrt(sum(value * value for value in vector))


# --- what only a real server can be asked ----------------------------------------------------


@pytest.mark.parametrize("served", MODELS, ids=lambda item: item.model)
async def test_the_configured_tokenizer_reproduces_the_servers_own_token_counts(
    served: Served,
) -> None:
    """The measurement the whole ``tokenizer`` setting rests on.

    Ollama serves GGUF and exposes no tokenizer, so this backend names a Hugging Face
    repository and counts with it. That is an inference until it is checked, and
    ``EmbedFingerprint`` excludes ``backend`` from identity on the understanding that nobody
    presents an inference as a measurement. This is the check: one probe per request, because
    ``prompt_eval_count`` is a request total and a batch would pass while two inputs were wrong
    in opposite directions.

    Measured on both models over the probe set — a non-Latin script, source code, and CJK mixed
    with an astral-plane emoji — with exact agreement on every one.
    """
    embedder = await embedder_for(served, cache_entries=0)
    try:
        assert embedder.card.special_token_count == served.special_tokens
        for probe in (*TOKENIZER_PROBES, *TEXTS):
            expected = embedder.count_tokens(probe) + served.special_tokens
            # Read through `ollama_prompt_tokens`, which publishes the server's own count, so
            # this exercises the metric an operator reads as well as the agreement itself.
            before = server_tokens(embedder)
            await embedder.embed([probe])
            observed = server_tokens(embedder) - before
            assert observed == expected, (
                f"{served.tokenizer} makes {expected} tokens of {probe[:40]!r} where "
                f"{served.model} read {observed}. If this starts failing, the "
                f"tokenizer repository and the GGUF have diverged — pin a different revision "
                f"rather than widening anything, because the number this moves is the chunk "
                f"budget and undercounting is the direction that truncates"
            )
    finally:
        await embedder.teardown()


@pytest.mark.parametrize("served", MODELS, ids=lambda item: item.model)
async def test_the_declared_context_is_not_the_served_one(served: Served) -> None:
    """Ollama's default ``num_ctx`` is its own, not the model's, and the gap is silent.

    Measured on ``qwen3-embedding:0.6b``: a GGUF declaring 32768, served at **4096** when the
    request carried no options, answering a longer input with a well-formed vector built from
    its first 4095 tokens. This asserts the shape of that rather than a specific number, so it
    keeps meaning something when Ollama changes its default — what must stay true is that a
    request carrying ``num_ctx`` is served at ``num_ctx`` and a request without one is not
    guaranteed to be.
    """
    url = require_ollama()
    client = OllamaClient(url, timeout_s=300.0)
    try:
        require_served(client, served)
        long_text = " ".join(["paragraph"] * (served.context_length * 2))
        asked = await client.embed(
            served.model,
            [long_text],
            num_ctx=served.context_length,
            keep_alive="5m",
            truncate=True,
        )
        assert asked.prompt_eval_count >= served.context_length - CONTEXT_RESERVE, (
            f"{served.model} served only {asked.prompt_eval_count} tokens when asked for "
            f"num_ctx={served.context_length}. This backend derives max_sequence_length from "
            f"the number it sends, so a server that does not honor it truncates every long "
            f"chunk with no error"
        )
    finally:
        await client.aclose()


@pytest.mark.parametrize("served", MODELS, ids=lambda item: item.model)
async def test_truncation_is_real_and_is_what_truncate_false_prevents(served: Served) -> None:
    """Let the silent failure happen, and show it is silent.

    Asserting only that ``truncate: false`` refuses would leave the reader to take the danger
    on trust. So this sends the same over-long text **with Ollama's own default** and shows
    what comes back: a correctly shaped, correctly normalized vector, indistinguishable from a
    good one, that is in fact the vector of a prefix. A chunk embedded that way claims all of
    its text while its vector describes the opening — a citation quoting words the index never
    saw.

    The same text with ``truncate: false`` is a refusal, which is the only difference between
    those two outcomes and the reason every request this backend sends carries the flag.
    """
    url = require_ollama()
    client = OllamaClient(url, timeout_s=300.0)
    try:
        require_served(client, served)
        num_ctx = min(512, served.context_length)
        over = " ".join(["paragraph"] * (num_ctx * 3))

        truncated = await client.embed(
            served.model, [over], num_ctx=num_ctx, keep_alive="5m", truncate=True
        )
        assert truncated.vectors, "the server answered an over-long input, as it is documented to"
        assert truncated.prompt_eval_count <= num_ctx
        assert abs(norm(truncated.vectors[0]) - 1.0) < 1e-3, (
            "the truncated vector is a perfectly ordinary unit vector, which is exactly why "
            "nothing downstream can tell it apart from one built from the whole text"
        )

        with pytest.raises(OllamaContextOverflowError):
            await client.embed(
                served.model, [over], num_ctx=num_ctx, keep_alive="5m", truncate=False
            )
    finally:
        await client.aclose()


@pytest.mark.parametrize("served", MODELS, ids=lambda item: item.model)
async def test_the_declaration_matches_what_the_server_actually_does(served: Served) -> None:
    """Width, reduction and limit, read from the server and checked against what it returns."""
    embedder = await embedder_for(served)
    try:
        fingerprint = embedder.fingerprint
        assert fingerprint.dimension == served.dimension
        assert fingerprint.pooling is served.pooling
        assert fingerprint.normalized
        assert fingerprint.backend == "ollama"
        assert fingerprint.max_sequence_length == (
            served.context_length - CONTEXT_RESERVE - served.special_tokens
        )
        assert fingerprint.weights_identity.startswith("artifact:ollama:")
        assert fingerprint.weights_identity.endswith(":prefix=none")
        assert fingerprint.revision is not None
        assert fingerprint.revision.startswith("sha256:")

        vectors = await embedder.embed(list(TEXTS))
        assert all(len(vector) == served.dimension for vector in vectors)
        assert all(abs(norm(vector) - 1.0) < 1e-6 for vector in vectors)
    finally:
        await embedder.teardown()


@pytest.mark.parametrize("served", MODELS, ids=lambda item: item.model)
async def test_the_ollama_backend_meets_the_shipped_conformance_suites(served: Served) -> None:
    """manicule publishes these; a backend in another distribution passes the same ones.

    Including :func:`~manicule.testing.assert_refuses_oversized_chunks`, which is the one that
    matters most here — it is aimed squarely at re-embed, and re-embed against a served model
    is one ``num_ctx`` change away from a limit that fell under an unchanged fingerprint.
    """
    embedder = await embedder_for(served)
    try:
        assert isinstance(embedder, Embedder)
        assert not isinstance(embedder, TokenStateEmbedder)
        assert_protocol_signatures(embedder, Embedder)
        await assert_embedder_contract(embedder, list(TEXTS))
        await assert_refuses_oversized_chunks(embedder.embed_chunks, embedder)
    finally:
        await embedder.teardown()


@pytest.mark.parametrize("served", MODELS, ids=lambda item: item.model)
async def test_embedding_is_deterministic_and_batch_invariant(served: Served) -> None:
    """The property a tier B backend cannot establish by reading its own code.

    manicule pools tier A output itself, with the mask, so batch invariance is a property of
    code in this repository. Here the reduction happens on the other side of a socket: an
    unmasked mean pool over a padded batch would make a text's vector depend on what shared it,
    every individual vector would still look fine, and only this test would notice.
    """
    embedder = await embedder_for(served, cache_entries=0)
    try:
        alone = (await embedder.embed([TEXTS[0]]))[0]
        crowded = (await embedder.embed(list(TEXTS)))[0]
        again = (await embedder.embed([TEXTS[0]]))[0]

        assert cosine(alone, again) > 1 - 1e-6
        assert cosine(alone, crowded) > 1 - 1e-4, (
            f"{served.model}'s vector for a text changed with the batch it was in (cosine "
            f"{cosine(alone, crowded)}). That is what an unmasked reduction over padding does"
        )
    finally:
        await embedder.teardown()


@pytest.mark.parametrize("served", MODELS, ids=lambda item: item.model)
async def test_the_vectors_carry_meaning_rather_than_merely_arithmetic(served: Served) -> None:
    """A floor, not a quality benchmark — and the floor is where tier B needs one.

    Everything else here would pass against a server returning a normalized hash of the input:
    right width, unit length, deterministic, batch invariant. This is the cheapest check that
    the thing on the other end is a retrieval model at all, and it is the one that would catch
    a configuration pointed at a generative model whose hidden states Ollama pooled on request.

    Deliberately not a threshold on an absolute score. Two embedding models disagree about what
    0.7 means; both agree that a paraphrase beats an unrelated sentence.
    """
    embedder = await embedder_for(served)
    try:
        question = "how long are archived pages kept before deletion?"
        related = "The retention window is ninety days, after which archived pages are purged."
        unrelated = "def embed(texts): return [pool(model(text)) for text in texts]"

        vectors = await embedder.embed([question, related, unrelated])
        near = cosine(vectors[0], vectors[1])
        far = cosine(vectors[0], vectors[2])

        assert near > far, (
            f"{served.model} scored an unrelated code snippet ({far:.3f}) at least as close to "
            f"a question as its own answer ({near:.3f}). Every other check in this file would "
            f"pass against a server returning a normalized hash; this is the one that would not"
        )
    finally:
        await embedder.teardown()


@pytest.mark.parametrize("served", MODELS, ids=lambda item: item.model)
async def test_an_oversized_chunk_is_refused_before_the_server_sees_it(served: Served) -> None:
    """The local guard fires first, so the message can name the text rather than the batch."""
    embedder = await embedder_for(served, num_ctx=min(512, served.context_length))
    try:
        limit = embedder.fingerprint.max_sequence_length
        with pytest.raises(ContextOverflowError, match="Shorten it"):
            await embedder.embed([" ".join(["paragraph"] * (limit + 50))])
    finally:
        await embedder.teardown()


async def test_two_models_on_one_server_are_two_vector_spaces() -> None:
    """And no claim is made that either is interchangeable with anything.

    The contrast with ``manicule-mlx``'s parity suite is deliberate. There, two backends running
    one model are shown to agree, which is what licenses ``backend`` staying out of identity.
    Here there is no such measurement and none is asserted: ``weights_identity`` carries the
    backend's name, so these fingerprints cannot equal one from any other runtime even by
    accident, and an index built with one of these models refuses vectors from the other.
    """
    require_ollama()
    first = await embedder_for(MODELS[0])
    try:
        second = await embedder_for(MODELS[1])
    except BaseException:
        await first.teardown()
        raise
    try:
        assert not first.fingerprint.matches(second.fingerprint)
        assert first.fingerprint.weights_identity != second.fingerprint.weights_identity
        assert first.fingerprint.dimension != second.fingerprint.dimension
    finally:
        await first.teardown()
        await second.teardown()


def test_the_skip_switch_is_outside_maniculess_own_namespace() -> None:
    """The trap ``docs/embeddings.md`` §7 records, asserted rather than remembered.

    ``manicule_environment`` clears every ``MANICULE_``-prefixed variable before each test, so a
    switch named ``MANICULE_REQUIRE_...`` is deleted before it is read: CI sets it, every case
    skips, and the job reports success — the failure the switch exists to prevent, occurring
    inside it. Found once by reading a green log, and held here so it cannot recur by rename.
    """
    assert not REQUIRE_OLLAMA_ENV.startswith("MANICULE_")
    assert not OLLAMA_URL_ENV.startswith("MANICULE_")


@pytest.mark.parametrize("served", MODELS, ids=lambda item: item.model)
async def test_the_whole_system_wires_to_this_backend_through_ordinary_discovery(
    served: Served, tmp_path: Path
) -> None:
    """The claim with the most downstream consequence, checked all the way through.

    Four things have to be true together, and three of them are structural rather than
    protocol-level — they are read off the embedder by name, so a backend that satisfies
    ``Embedder`` and nothing else would pass every other test in this repository and still
    leave an installation unable to ingest:

    * the container resolves ``embedder.ollama`` through the public entry-point group, with no
      knowledge of this package anywhere in ``src/manicule``;
    * the chunker binds its token counter to *this* embedder rather than falling back to the
      provisional one — whose chunks ingest refuses, which is how a missing ``count_tokens``
      presents: not as an error but as a corpus that will not index;
    * :func:`manicule.ingest.workers.worker_config` finds a real ``tokenizer.json`` at
      ``embedder.card.path``, which is what isolated parse workers chunk with and the reason a
      backend whose runtime has no tokenizer still has to carry one;
    * metadata-only rebuild planning derives byte-identical identity from the recorded
      declaration, with no server contacted.
    """
    url = require_ollama()
    from manicule.config.settings import (  # noqa: PLC0415
        EmbeddingSettings,
        PluginSettings,
        Settings,
    )
    from manicule.container import keys  # noqa: PLC0415
    from manicule.container.container import Container  # noqa: PLC0415
    from manicule.core.fingerprints import ChunkFingerprint  # noqa: PLC0415
    from manicule.ingest.workers import worker_config  # noqa: PLC0415
    from manicule.plugins import discover  # noqa: PLC0415

    settings = Settings(
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        embedding=EmbeddingSettings(provider="ollama", model=served.model),
        plugins=PluginSettings(
            config={
                "embedder.ollama": {
                    "base_url": url,
                    "tokenizer": served.tokenizer,
                    "tokenizer_revision": served.revision,
                    # Small, so the one expensive probe in this file's cold path stays small
                    # too. What is under test is the wiring, not the ceiling.
                    "num_ctx": 2048,
                }
            }
        ),
    )
    found = discover()
    container = Container(settings, found.registry, discovery=found)
    try:
        try:
            embedder = container.get(keys.EMBEDDER)
        except ConfigError as exc:
            if is_required(served.model):
                raise
            pytest.skip(f"{url} does not serve {served.model!r}: {exc}")
        assert isinstance(embedder, OllamaEmbedder)

        chunker = container.get(keys.CHUNKER)
        chunk_fingerprint = getattr(chunker, "fingerprint", None)
        assert isinstance(chunk_fingerprint, ChunkFingerprint)
        assert not chunk_fingerprint.provisional, (
            "the chunker fell back to the stand-in vocabulary, which marks every chunk "
            "provisional and makes ingest refuse them — a backend that cannot count tokens "
            "does not fail, it produces a corpus that will not index"
        )
        assert chunk_fingerprint.tokenizer_id == embedder.fingerprint.tokenizer_id

        config = worker_config(settings, chunker=chunker, embedder=embedder)
        assert config.stage_tokenizer_file is not None
        assert config.stage_tokenizer_file.is_file()
        assert config.stage_tokenizer_id == embedder.fingerprint.tokenizer_id

        planned = container.metadata(keys.EMBEDDER)
        assert isinstance(planned, type(embedder.fingerprint))
        assert planned.canonical() == embedder.fingerprint.canonical()
    finally:
        await container.aclose()
