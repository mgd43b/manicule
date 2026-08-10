"""Pooling and normalisation, done here rather than taken from a backend.

Every backend offers a finished vector under a friendly name, and the friendly name is not
reliably the model's own pooling. Measured on ``BAAI/bge-m3``, a CLS-pooled model:

===============================  =====================================================
``mlx-embeddings`` field         what it actually is
===============================  =====================================================
``text_embeds``                  mean pooling — cosine 1.000 to our mean pool, and
                                 **0.66-0.80** to the CLS pooling the model declares
``pooler_output``                ``tanh(dense(CLS))`` — cosine **-0.04 to +0.02** to
                                 raw CLS, which is to say unrelated to it
``last_hidden_state``            genuine token states on ``xlm_roberta``; the *pooled*
                                 vector on ``modernbert``, under the same name
===============================  =====================================================

Three names, three different answers, no error from any of them. So the pooling path takes
token states, checks their rank, and reduces them here with the reduction the model's own
configuration declares.

The rank check is the load-bearing part. Pooling a ``(batch, dimension)`` array does not
fail — it reduces over the batch axis, and every text in the batch comes out holding the
same plausible, normalised, entirely wrong vector.
"""

from __future__ import annotations

import numpy as np

from manicule.core.embedding import NDArrayLike, Pooling, TokenStates, Vector
from manicule.core.errors import TokenStateError

TOKEN_STATE_RANK = 3
"""``(batch, sequence, dimension)``. Anything else is not token states."""

_EPSILON = 1e-12
"""Guards the division in :func:`l2_normalise`. A zero vector normalises to itself."""


def require_token_states(states: NDArrayLike, *, backend: str, model_id: str) -> None:
    """Raise unless ``states`` is a rank-3 batch of per-token hidden states.

    Called before anything is pooled, on the array as the backend handed it over. It is the
    check that keeps a library free to rename or rebind its outputs without silently changing
    what manicule stores.

    Args:
        states: Whatever the backend returned for its token states.
        backend: Named in the error, because the fix is backend-specific.
        model_id: Named in the error, because the rebinding is per architecture.

    Raises:
        TokenStateError: The array is not rank 3.
    """
    shape = tuple(states.shape)
    if len(shape) == TOKEN_STATE_RANK:
        return
    msg = (
        f"the {backend} backend returned an array of shape {shape} for {model_id} where "
        f"per-token hidden states of shape (batch, sequence, dimension) were expected. "
        f"This is what an already-pooled vector looks like arriving under a token-state "
        f"name, and pooling it would reduce over the batch axis instead of the sequence "
        f"axis — one plausible, normalised, wrong vector per text, with nothing downstream "
        f"able to tell. Read the encoder output rather than the backend's convenience field."
    )
    raise TokenStateError(msg)


def as_float32(array: NDArrayLike) -> np.ndarray:
    """Bring a backend's native array into numpy at a single, stated precision.

    Backends hand back arrays of their own: MLX ``bfloat16`` or ``float16`` on Metal,
    ``float32`` from onnxruntime. Pooling and normalising at whatever precision arrived would
    make the vectors depend on the runtime's storage choice, which is the one thing the
    platform is not allowed to change. One conversion, here, and everything after it is
    float32 arithmetic on both backends.
    """
    return np.asarray(array, dtype=np.float32)


def pool(states: np.ndarray, attention_mask: np.ndarray, pooling: Pooling) -> np.ndarray:
    """Reduce ``(batch, sequence, dimension)`` token states to ``(batch, dimension)``.

    Args:
        states: Token states, rank 3. Check them with :func:`require_token_states` first.
        attention_mask: ``(batch, sequence)``, 1 for real tokens and 0 for padding.
        pooling: The reduction the model declares. Never a default.

    Raises:
        TokenStateError: ``states`` is not rank 3, or the mask does not match it.
        ValueError: ``pooling`` names no reduction, which cannot produce one vector per text.
    """
    if states.ndim != TOKEN_STATE_RANK:
        msg = f"pool() needs rank-3 token states, got shape {states.shape}"
        raise TokenStateError(msg)
    if attention_mask.shape != states.shape[:2]:
        msg = (
            f"attention mask of shape {attention_mask.shape} does not describe token states "
            f"of shape {states.shape}. A mask that does not line up would pool padding into "
            f"the vector, making the result depend on what else shared the batch"
        )
        raise TokenStateError(msg)

    match pooling:
        case Pooling.CLS:
            return states[:, 0, :]
        case Pooling.MEAN:
            return _mean(states, attention_mask)
        case Pooling.LAST_TOKEN:
            return _last_token(states, attention_mask)
        case Pooling.NONE:
            msg = (
                "pooling is 'none', which leaves token states unreduced and so cannot "
                "produce one vector per text. A model without a declared reduction has to "
                "be given one in configuration"
            )
            raise ValueError(msg)


def _mean(states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Mean over real tokens only.

    Weighted by the attention mask, always. An unweighted mean over a padded batch averages
    in the padding, so the same text embeds differently depending on the longest text that
    happened to share its batch — an index whose contents depend on batch order, and no
    individual vector looks wrong.
    """
    weights = attention_mask[:, :, None].astype(states.dtype)
    summed = (states * weights).sum(axis=1)
    counts = np.maximum(weights.sum(axis=1), _EPSILON)
    return summed / counts


def _last_token(states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """The final real token of each row.

    Read from the mask rather than from ``states.shape[1]``: with right padding the last
    position of the array is padding for every row but the longest.
    """
    lengths = attention_mask.sum(axis=1).astype(np.int64)
    indices = np.maximum(lengths - 1, 0)
    return states[np.arange(states.shape[0]), indices, :]


def l2_normalise(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length.

    Applied unconditionally rather than read from the model's declared pipeline. A repository
    that omits its ``Normalize`` step still publishes cosine scores that assume it, so trusting
    the declaration reproduces neither the published numbers nor anyone else's index.
    """
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, _EPSILON)


def pool_token_states(
    token_states: TokenStates, pooling: Pooling, *, backend: str, model_id: str
) -> list[Vector]:
    """The whole reduction: check the rank, pool, normalise, hand back plain sequences.

    Vectors leave as lists because they cross a storage boundary next, and a native array
    that has to be converted somewhere is best converted once, here.

    Raises:
        TokenStateError: The states are not rank 3, or their width is not the dimension the
            batch declares.
    """
    require_token_states(token_states.states, backend=backend, model_id=model_id)
    states = as_float32(token_states.states)
    mask = as_float32(token_states.attention_mask).astype(np.int64)

    width = states.shape[2]
    if width != token_states.dimension:
        msg = (
            f"{model_id} produced token states {width} wide under {backend}, but the batch "
            f"declares {token_states.dimension} dimensions. The declared number is what the "
            f"vector table was created with, so a disagreement here is an index built to the "
            f"wrong width"
        )
        raise TokenStateError(msg)

    pooled = l2_normalise(pool(states, mask, pooling))
    return [row.tolist() for row in pooled]


__all__ = [
    "TOKEN_STATE_RANK",
    "as_float32",
    "l2_normalise",
    "pool",
    "pool_token_states",
    "require_token_states",
]
