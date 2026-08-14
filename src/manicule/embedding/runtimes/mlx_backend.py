"""The MLX backend: Metal-native execution, in-process, on Apple Silicon.

Primary on the hardware manicule is built for, and the reason the project is
GPL-3.0-or-later: ``mlx-embeddings`` is GPL-3.0 (``docs/embeddings.md`` §1.1).

It is also version 0.1.0, and it earns the caution. ``models/xlm_roberta.py`` computes
``text_embeds = normalize_embeddings(mean_pooling(sequence_output, attention_mask))``
**unconditionally**, for every XLM-RoBERTa checkpoint — including ``bge-m3``, which declares
CLS pooling. Reaching for the obviously named field returns a correctly shaped, correctly
normalized vector from the wrong reduction, at cosine 0.66-0.80 to the right one, with nothing
raised. So this class touches ``last_hidden_state`` and nothing else, checks its rank, and
hands the states to :mod:`manicule.embedding.pooling`.

``last_hidden_state`` is itself only trustworthy by architecture — the same library rebinds it
to the *pooled* vector for ``modernbert`` — which is why the rank check is not optional and
not defensive. It is the assertion that turns a per-architecture rebinding into a loud failure
instead of a corpus.
"""

from __future__ import annotations

from typing import Any, override

import numpy as np

from manicule.core.errors import ConfigError
from manicule.core.lifecycle import HealthReport
from manicule.embedding.artifacts import mlx_repo, mlx_weights
from manicule.embedding.base import PooledEmbedder
from manicule.embedding.cards import ModelCard
from manicule.embedding.pooling import as_float32

BACKEND = "mlx"


class MlxEmbedder(PooledEmbedder):
    """Embeds with ``mlx-embeddings``, pooling in manicule's numpy.

    The weights are usually not the model's own: ``BAAI/bge-m3`` publishes a PyTorch pickle and
    no safetensors, so MLX runs a community conversion, recorded in
    :attr:`~manicule.core.embedding.EmbedFingerprint.weights_ref` and refused outright if it is
    quantized (:mod:`manicule.embedding.artifacts`).
    """

    def __init__(
        self,
        card: ModelCard,
        *,
        weights: str = "",
        batch_size: int = 32,
        cache_entries: int = 10_000,
    ) -> None:
        super().__init__(
            card,
            backend=BACKEND,
            weights_ref=mlx_repo(card.model_id, override=weights),
            batch_size=batch_size,
            cache_entries=cache_entries,
        )
        self._weights_override = weights
        self._model: Any | None = None  # mlx-embeddings ships no type information

    @override
    def _load(self) -> None:
        """Download, vet and load the weights, on the thread that will run them.

        Import and download happen here rather than at construction so that an installed
        embedder nobody selected costs one cheap import, and so that the chunker can read this
        model's sequence limit without several gigabytes arriving first.
        """
        from mlx_embeddings.utils import load_model  # noqa: PLC0415 - see docstring

        weights = mlx_weights(self.card.model_id, override=self._weights_override)
        if weights.describe() != self.fingerprint.weights_ref:
            # Only reachable if resolution stopped being a pure function of the same inputs.
            # Worth catching: the fingerprint is already fixed, so a divergence here would
            # record one artifact and execute another.
            msg = (
                f"the MLX weights resolved to {weights.describe()} at setup but "
                f"{self.fingerprint.weights_ref} was recorded in the fingerprint at "
                f"construction. Vectors would be attributed to weights that did not make them."
            )
            raise ConfigError(msg)
        self._model = load_model(weights.path, path_to_repo=str(weights.path))

    @override
    def _unload(self) -> None:
        self._model = None

    @override
    def _loaded(self) -> bool:
        return self._model is not None

    @override
    async def health(self) -> HealthReport:
        report = await super().health()
        if report.ok:
            return report
        return HealthReport.failing(
            report.detail,
            remedy="Run setup(); on non-Apple hardware use the onnx backend, which produces "
            "vectors within tolerance of these and needs no re-embed.",
        )

    @override
    def _forward(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Encoder output only, evaluated here rather than left lazy.

        ``text_embeds`` and ``pooler_output`` come back in the same object and are both
        ignored: the first is mean-pooled whatever the model declares, and the second is
        ``tanh(dense(CLS))``, which measures cosine -0.04 to +0.02 against raw CLS — unrelated
        to it rather than a worse version of it.
        """
        import mlx.core as mx  # noqa: PLC0415 - loaded with the backend, not with manicule

        model = self._model
        if model is None:
            msg = (
                f"{self.fingerprint.model_id} has not been loaded on the mlx backend. "
                f"setup() loads the weights and the container calls it; construct this "
                f"embedder through the container, or await setup() yourself."
            )
            raise ConfigError(msg)

        output = model(mx.array(input_ids), attention_mask=mx.array(attention_mask))
        # `.astype` before numpy, because MLX weights are commonly float16 or bfloat16 and
        # numpy has no bfloat16 to convert into. It also fixes the precision everything after
        # this point works at, on both backends.
        return as_float32(output.last_hidden_state.astype(mx.float32))


__all__ = ["BACKEND", "MlxEmbedder"]
