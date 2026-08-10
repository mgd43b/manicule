"""Embedders and caches that break the rules, so the guards can be shown to catch them.

Kept apart from :mod:`tests.fakes` for the reason that module states about itself: everything
there is implementable without a model or a database, and everything here needs numpy and a
tokenizer. Importing one must not cost the other.

Each class corresponds to one silent failure the embedding path exists to prevent, and each is
a real subclass of :class:`~manicule.embedding.base.PooledEmbedder` rather than a stub — a fake
that does not go through the code under test proves nothing about it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import override

import numpy as np

from manicule.core.embedding import EmbedFingerprint, Vector
from manicule.embedding.base import PooledEmbedder
from manicule.embedding.cache import EmbeddingCache
from manicule.embedding.cards import ModelCard
from manicule.embedding.pooling import l2_normalise


class StubEmbedder(PooledEmbedder):
    """Deterministic token states, no weights, real everything else.

    The states are derived from each token id, so the same text always produces the same
    states, two texts produce different ones, and padding positions produce *large* states —
    which is what makes an unmasked mean pool visibly wrong instead of merely different.
    """

    def __init__(
        self, card: ModelCard, *, cache_entries: int = 10_000, batch_size: int = 32
    ) -> None:
        super().__init__(
            card,
            backend="stub",
            weights_ref="",
            batch_size=batch_size,
            cache_entries=cache_entries,
        )
        self.forward_calls = 0

    @override
    def _load(self) -> None:
        return

    @override
    def _unload(self) -> None:
        return

    @override
    def _loaded(self) -> bool:
        return True

    @override
    def _forward(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        self.forward_calls += 1
        return token_states(input_ids, self.fingerprint.dimension)


class PrePooledEmbedder(StubEmbedder):
    """Returns the *pooled* vector where token states belong.

    Exactly what ``mlx-embeddings`` does for ModernBERT checkpoints: same attribute, same
    plausible shape family, one rank down. Without the rank check, pooling reduces over the
    batch axis, every text in the batch comes out with the same vector, and nothing raises.
    """

    @override
    def _forward(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        states = token_states(input_ids, self.fingerprint.dimension)
        return states[:, 0, :]


class WrongWidthEmbedder(StubEmbedder):
    """Returns token states one dimension narrower than the fingerprint advertises.

    The vector table is created from the fingerprint's dimension, so a disagreement here is an
    index built to the wrong width — and a store that only checks length would accept the first
    write and every one after it.
    """

    @override
    def _forward(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        return token_states(input_ids, self.fingerprint.dimension - 1)


class UnmaskedMeanEmbedder(StubEmbedder):
    """Pools with a mean that ignores the attention mask.

    The classic batch-order bug. Every vector is well shaped and unit length; the only symptom
    is that a text embeds differently depending on what shared its batch, which no individual
    vector reveals and no store can detect.
    """

    @override
    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        ids, _ = self._tokenize(texts)
        states = self._forward(ids, np.ones_like(ids))
        pooled = l2_normalise(states.mean(axis=1))
        return [row.tolist() for row in pooled]


class NameKeyedCache(EmbeddingCache):
    """Keys on the model name instead of the fingerprint.

    ``PLAN.md`` §16 said "key by model identity", and this is what that phrase permits if it is
    read loosely. The same weights with a different pooling, revision or tokenizer produce a
    different vector for the same text, and none of that is in the name — so a re-embed under a
    new fingerprint is served vectors from the old space, and reports success.
    """

    @override
    @staticmethod
    def key(fingerprint: EmbedFingerprint, text: str) -> tuple[str, str]:
        return (fingerprint.model_id, text)


PAD_ID = 2
"""``<pad>`` in :data:`tests.embedding_support.VOCABULARY`."""


def token_states(input_ids: np.ndarray, dimension: int, pad_id: int = PAD_ID) -> np.ndarray:
    """Deterministic per-token states of shape ``(batch, sequence, dimension)``.

    Padding positions are given a large constant rather than a hash, so a pool that forgets the
    mask comes out *visibly* wrong instead of subtly wrong. Subtly wrong is how a broken test
    passes on tolerance.
    """
    batch, length = input_ids.shape
    states = np.zeros((batch, length, dimension), dtype=np.float32)
    for row in range(batch):
        for position in range(length):
            token = int(input_ids[row, position])
            if token == pad_id:
                states[row, position] = 100.0
                continue
            digest = hashlib.blake2b(f"{token}:{position}".encode(), digest_size=dimension).digest()
            states[row, position] = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255
    return states


__all__ = [
    "NameKeyedCache",
    "PrePooledEmbedder",
    "StubEmbedder",
    "UnmaskedMeanEmbedder",
    "WrongWidthEmbedder",
    "token_states",
]
