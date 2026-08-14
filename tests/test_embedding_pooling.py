"""Pooling: the reduction, the rank check in front of it, and the normalization after it.

No model is loaded here. Everything the reduction can get wrong is arithmetic on token states,
and arithmetic is testable exhaustively and instantly — including the cases a real model would
never hand you, which are the ones that matter.
"""

from __future__ import annotations

import numpy as np
import pytest

from manicule.core.embedding import Pooling, TokenStates
from manicule.core.errors import TokenStateError
from manicule.embedding.pooling import (
    TOKEN_STATE_RANK,
    l2_normalize,
    pool,
    pool_token_states,
    require_token_states,
)


def states(batch: int = 2, length: int = 4, dimension: int = 3) -> np.ndarray:
    """Token states whose every position is distinguishable from every other."""
    return np.arange(batch * length * dimension, dtype=np.float32).reshape(batch, length, dimension)


def test_token_states_must_be_three_dimensional() -> None:
    """The assertion this whole module exists behind.

    ``mlx-embeddings`` binds ``last_hidden_state`` to genuine token states on one architecture
    and to the already-pooled vector on another. Pooling the pooled vector does not fail: it
    reduces over the batch axis and hands every text in the batch the same well-shaped,
    normalized, wrong vector. A test that only checked "a vector came back" would pass on
    exactly that.
    """
    pooled = states()[:, 0, :]
    assert pooled.ndim == TOKEN_STATE_RANK - 1

    with pytest.raises(TokenStateError, match="already-pooled"):
        require_token_states(pooled, backend="mlx", model_id="BAAI/bge-m3")

    require_token_states(states(), backend="mlx", model_id="BAAI/bge-m3")


def test_the_rank_error_names_the_backend_and_the_model() -> None:
    """The rebinding is per backend *and* per architecture, so both belong in the message."""
    with pytest.raises(TokenStateError) as raised:
        require_token_states(np.zeros((2, 3)), backend="onnx", model_id="some/model")

    assert "onnx" in str(raised.value)
    assert "some/model" in str(raised.value)


def test_cls_pooling_takes_the_first_position() -> None:
    """Not the first *row*: a batch axis mistake here is the pre-pooled bug in another form."""
    pooled = pool(states(), np.ones((2, 4), dtype=np.int64), Pooling.CLS)

    assert pooled.shape == (2, 3)
    assert np.array_equal(pooled[0], [0.0, 1.0, 2.0])
    assert np.array_equal(pooled[1], [12.0, 13.0, 14.0])


def test_mean_pooling_ignores_padding() -> None:
    """A mean that averages in the padding makes the index depend on batch order.

    The same text then embeds differently depending on the longest text that happened to share
    its batch — invisible one vector at a time, and fatal across a corpus.
    """
    values = states()
    mask = np.array([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=np.int64)

    pooled = pool(values, mask, Pooling.MEAN)

    assert np.allclose(pooled[0], values[0, :2].mean(axis=0))
    assert np.allclose(pooled[1], values[1].mean(axis=0))
    assert not np.allclose(pooled[0], values[0].mean(axis=0))


def test_mean_pooling_of_one_text_does_not_depend_on_its_batch() -> None:
    """The property the mask buys, stated directly."""
    alone = pool(states(1, 2, 3), np.ones((1, 2), dtype=np.int64), Pooling.MEAN)

    padded = np.concatenate([states(1, 2, 3), np.full((1, 2, 3), 999.0, dtype=np.float32)], axis=1)
    in_batch = pool(padded, np.array([[1, 1, 0, 0]], dtype=np.int64), Pooling.MEAN)

    assert np.allclose(alone, in_batch)


def test_last_token_pooling_reads_the_mask_not_the_array_width() -> None:
    """With right padding, the last position of the array is padding for every short row."""
    values = states()
    mask = np.array([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=np.int64)

    pooled = pool(values, mask, Pooling.LAST_TOKEN)

    assert np.array_equal(pooled[0], values[0, 1])
    assert np.array_equal(pooled[1], values[1, 3])


def test_pooling_none_cannot_produce_one_vector_per_text() -> None:
    """Refused rather than quietly treated as mean, which is a different model's reduction."""
    with pytest.raises(ValueError, match="cannot"):
        pool(states(), np.ones((2, 4), dtype=np.int64), Pooling.NONE)


def test_a_mask_that_does_not_match_the_states_is_refused() -> None:
    """A misaligned mask pools padding into the vector, silently."""
    with pytest.raises(TokenStateError, match="does not describe"):
        pool(states(), np.ones((2, 9), dtype=np.int64), Pooling.MEAN)


def test_normalization_is_applied_rather_than_read_from_the_model() -> None:
    """A repository can omit its ``Normalize`` step and still publish cosine scores.

    Trusting the declaration reproduces neither the published numbers nor anyone else's index,
    so normalization is unconditional here.
    """
    normalized = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))

    lengths = (normalized * normalized).sum(axis=-1)
    assert np.allclose(lengths, [1.0, 0.0]), "a zero vector must not divide by zero"


def test_cls_and_mean_disagree_enough_to_matter() -> None:
    """The reason pooling is part of fingerprint identity rather than a preference.

    Measured on ``BAAI/bge-m3`` itself, from real token states: cosine 0.79 at 15 tokens and
    0.66 at 402. This check is the arithmetic version — the same two reductions of the same
    states are not interchangeable, so a model indexed under one and queried under the other is
    searching a space it does not live in.
    """
    values = states(1, 8, 4)
    mask = np.ones((1, 8), dtype=np.int64)

    cls = l2_normalize(pool(values, mask, Pooling.CLS))
    mean = l2_normalize(pool(values, mask, Pooling.MEAN))

    assert float(cls[0] @ mean[0]) < 0.99


def test_pooled_output_is_normalized_and_plain() -> None:
    """Vectors cross a storage boundary next, so they leave as sequences, not native arrays."""
    vectors = pool_token_states(
        TokenStates(states=states(), attention_mask=np.ones((2, 4), dtype=np.int64), dimension=3),
        Pooling.CLS,
        backend="stub",
        model_id="test/model",
    )

    assert len(vectors) == 2
    assert all(isinstance(vector, list) for vector in vectors)
    assert all(isinstance(value, float) for vector in vectors for value in vector)
    pooled = np.asarray(vectors, dtype=np.float32)
    assert np.allclose((pooled * pooled).sum(axis=-1), 1.0)


def test_token_states_narrower_than_declared_are_refused() -> None:
    """The declared dimension is what the vector table was created with.

    A disagreement is an index built to the wrong width, and every write after the first one
    succeeds.
    """
    with pytest.raises(TokenStateError, match="1024 dimensions"):
        pool_token_states(
            TokenStates(
                states=states(2, 4, 3),
                attention_mask=np.ones((2, 4), dtype=np.int64),
                dimension=1024,
            ),
            Pooling.CLS,
            backend="stub",
            model_id="test/model",
        )
