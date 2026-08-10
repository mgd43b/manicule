"""The shared embedding path, exercised through real subclasses with no weights behind them.

Everything checked here is what a backend cannot be trusted to do for itself: the rank check,
the reduction, the refusal to truncate, the context check on stored chunks, the cache. Running
it against a stub rather than a model means it runs everywhere, every time — including on a
machine that has never downloaded anything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from manicule.core.anchors import Unlocated
from manicule.core.content import Chunk
from manicule.core.errors import ContextOverflowError, TokenStateError
from manicule.core.fingerprints import ChunkFingerprint
from manicule.core.lifecycle import HealthState
from manicule.core.protocols import Embedder, TokenStateEmbedder
from manicule.embedding.cards import ModelCard, read_card
from manicule.testing import (
    assert_embedder_contract,
    assert_protocol_signatures,
    assert_refuses_oversized_chunks,
)
from tests.embedding_fakes import (
    PrePooledEmbedder,
    StubEmbedder,
    UnmaskedMeanEmbedder,
    WrongWidthEmbedder,
)
from tests.embedding_support import write_model

TEXTS = ("alpha beta", "gamma delta epsilon", "alpha beta")


@pytest.fixture
def card(tmp_path: Path) -> ModelCard:
    return read_card(str(write_model(tmp_path / "model")))


@pytest.fixture
def embedder(card: ModelCard) -> StubEmbedder:
    return StubEmbedder(card)


def chunk(text: str, token_count: int, position: int = 0) -> Chunk:
    return Chunk(
        id=f"chunk-{position}",
        document_id="doc",
        text=text,
        embed_text=text,
        anchor=Unlocated(reason="synthetic"),
        position=position,
        token_count=token_count,
    )


async def test_the_embedder_satisfies_both_tiers_of_the_protocol(embedder: StubEmbedder) -> None:
    """Tier A is the point: manicule pools, so the token states have to be reachable."""
    assert isinstance(embedder, Embedder)
    assert isinstance(embedder, TokenStateEmbedder)
    assert_protocol_signatures(embedder, Embedder)
    assert_protocol_signatures(embedder, TokenStateEmbedder)

    await assert_embedder_contract(embedder, list(TEXTS))


async def test_encode_returns_three_dimensional_token_states(embedder: StubEmbedder) -> None:
    """The assertion ticket #3 asks for, at the seam a caller actually uses.

    A check that only asserted "a vector came back" would pass on a backend handing over its
    pooled output under the token-state name — which is the exact bug this exists to catch.
    """
    encoded = await embedder.encode(list(TEXTS))

    assert len(encoded.states.shape) == 3
    batch, sequence, dimension = encoded.states.shape
    assert batch == len(TEXTS)
    assert sequence > 1, "a sequence axis of one would make the rank check vacuous"
    assert dimension == embedder.fingerprint.dimension
    assert encoded.attention_mask.shape == (batch, sequence)


async def test_a_backend_returning_a_pooled_vector_is_caught(card: ModelCard) -> None:
    """The fake that proves the rank check fires.

    Disable ``require_token_states`` and this test goes green while every vector in the batch
    becomes identical — which is what makes the check load-bearing rather than defensive.
    """
    broken = PrePooledEmbedder(card)

    with pytest.raises(TokenStateError, match="already-pooled"):
        await broken.embed(list(TEXTS))


async def test_a_backend_returning_the_wrong_width_is_caught(card: ModelCard) -> None:
    """The dimension in the fingerprint is what the vector table was created with."""
    broken = WrongWidthEmbedder(card)

    with pytest.raises(TokenStateError, match="dimensions"):
        await broken.embed(list(TEXTS))


async def test_the_same_text_embeds_to_the_same_vector_whatever_shares_its_batch(
    card: ModelCard,
) -> None:
    """Batch dependence is invisible one vector at a time and fatal across a corpus.

    The cache is disabled here on purpose: with it on, the second call is a hit and the test
    would be checking the dictionary rather than the pooling.
    """
    embedder = StubEmbedder(card, cache_entries=0)

    alone = await embedder.embed(["alpha"])
    crowded = await embedder.embed(["alpha", "alpha beta gamma delta epsilon zeta"])

    assert np.allclose(alone[0], crowded[0])


async def test_an_unmasked_mean_makes_a_vector_depend_on_its_batch(card: ModelCard) -> None:
    """The fake that proves the previous test is load-bearing.

    Same model, same text, same code path but for the mask — and the vector moves depending on
    what else was in the batch. Nothing raises, and both vectors are unit length.
    """
    broken = UnmaskedMeanEmbedder(card, cache_entries=0)

    alone = await broken.embed(["alpha"])
    crowded = await broken.embed(["alpha", "alpha beta gamma delta epsilon zeta"])

    assert not np.allclose(alone[0], crowded[0])
    assert np.isclose(np.linalg.norm(alone[0]), 1.0)


async def test_the_same_text_embeds_the_same_way_across_embedders(card: ModelCard) -> None:
    """Determinism, checked across instances rather than across calls.

    Across calls it would pass on a cache hit alone.
    """
    first = await StubEmbedder(card, cache_entries=0).embed(["alpha beta"])
    second = await StubEmbedder(card, cache_entries=0).embed(["alpha beta"])

    assert first == second


async def test_text_the_model_would_truncate_is_refused(card: ModelCard) -> None:
    """Truncation is silent, so the tokenizer never truncates and the caller hears about it.

    The stored vector would otherwise describe an opening fragment while the caller believed it
    described the whole text.
    """
    embedder = StubEmbedder(card)
    limit = embedder.fingerprint.max_sequence_length
    too_long = " ".join(["alpha"] * (limit + 5))

    with pytest.raises(ContextOverflowError, match=f"{limit}-token limit"):
        await embedder.embed([too_long])

    assert await embedder.embed([" ".join(["alpha"] * limit)])


async def test_embedding_stored_chunks_refuses_ones_the_model_cannot_read(
    embedder: StubEmbedder,
) -> None:
    """The conformance suite for the path re-embed uses.

    Re-embed reads stored ``embed_text`` without re-chunking, so the chunker's budget refusal
    never runs; and a sequence limit that fell leaves the fingerprint identical, so no
    comparison fires. This is the only guard on that route.
    """
    await assert_refuses_oversized_chunks(embedder.embed_chunks, embedder)


async def test_a_chunk_counted_with_another_vocabulary_is_refused(embedder: StubEmbedder) -> None:
    """A token count taken under a different vocabulary is not a measurement of anything."""
    counted_elsewhere = ChunkFingerprint(
        chunker="structural",
        version="0.1.0",
        max_tokens=8,
        tokenizer_id="some/other-tokenizer",
        overlap_tokens=0,
    )

    with pytest.raises(ContextOverflowError, match="tokenizes with"):
        await embedder.embed_chunks([chunk("alpha", token_count=1)], counted_elsewhere)


async def test_chunks_embed_in_the_order_they_were_given(embedder: StubEmbedder) -> None:
    """Vectors are stored parallel to chunks, so a reordering here mislabels the whole batch."""
    chunks = [chunk(text, token_count=2, position=index) for index, text in enumerate(TEXTS)]

    vectors = await embedder.embed_chunks(chunks)
    directly = await embedder.embed([item.embed_text for item in chunks])

    assert vectors == directly


async def test_repeated_text_costs_one_forward_pass(card: ModelCard) -> None:
    """Where a real corpus's hit rate comes from: boilerplate, and one attachment on forty pages."""
    embedder = StubEmbedder(card, batch_size=64)

    await embedder.embed(["alpha", "alpha", "alpha"])

    assert embedder.forward_calls == 1


async def test_a_cached_text_is_not_embedded_again(card: ModelCard) -> None:
    embedder = StubEmbedder(card)
    first = await embedder.embed(["alpha", "beta"])
    calls = embedder.forward_calls

    second = await embedder.embed(["beta", "alpha"])

    assert embedder.forward_calls == calls
    assert second == [first[1], first[0]]


async def test_batching_does_not_change_the_vectors(card: ModelCard) -> None:
    """A batch size is a throughput setting, so it must not reach the output.

    Padding is per batch, so a mask-free reduction would make this fail — which is the point.
    """
    texts = ["alpha", "beta gamma", "delta epsilon zeta", "alpha beta gamma delta"]

    one_at_a_time = await StubEmbedder(card, batch_size=1, cache_entries=0).embed(texts)
    all_at_once = await StubEmbedder(card, batch_size=64, cache_entries=0).embed(texts)

    assert np.allclose(one_at_a_time, all_at_once)


async def test_count_tokens_measures_content_not_special_tokens(embedder: StubEmbedder) -> None:
    """It is compared against ``max_sequence_length``, which is usable content tokens.

    Two numbers that get compared have to measure the same thing.
    """
    assert embedder.count_tokens("alpha beta gamma") == 3
    assert embedder.count_tokens("") == 0


async def test_an_empty_batch_costs_nothing(embedder: StubEmbedder) -> None:
    """Ingest hands over whatever a document produced, which is sometimes nothing."""
    assert await embedder.embed([]) == []
    assert (await embedder.encode([])).states.shape[0] == 0
    assert embedder.forward_calls == 0


async def test_lifecycle_reports_and_releases(embedder: StubEmbedder) -> None:
    """Teardown has to be safe after a failed setup, and safe twice."""
    await embedder.setup()
    assert (await embedder.health()).state is HealthState.OK

    await embedder.teardown()
    await embedder.teardown()


async def test_metrics_expose_the_cache(embedder: StubEmbedder) -> None:
    """A cache whose hit rate cannot be observed cannot be sized."""
    await embedder.embed(["alpha"])
    await embedder.embed(["alpha"])

    published = {metric.name: metric.value for metric in embedder.metrics()}

    assert published["embedding_cache_hits"] == 1
    assert published["embedding_cache_misses"] == 1
    assert published["embedding_texts_embedded"] == 1
