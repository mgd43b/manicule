"""The onnxruntime backend: the portable runtime, and the parity reference.

Two jobs. It is the only backend off Apple Silicon, and it is the measurement that lets
``backend`` stay out of :class:`~manicule.core.embedding.EmbedFingerprint` identity — same
text, same model, same pooling, both runtimes, vectors within a stated tolerance. Without
that measurement, moving a corpus between machines would have to mean re-embedding it.

**Outputs are selected by rank, never by name.** ``BAAI/bge-m3``'s export calls its 3-D output
``token_embeddings`` and its pooled output ``sentence_embedding``; ``mlx-embeddings`` calls the
corresponding pair ``last_hidden_state`` and ``text_embeds``. Nothing standardises either set,
and reading ``outputs[0]`` is a guess about export order. Rank is the property that is actually
stable: exactly one output is ``(batch, sequence, dimension)``, and if that is not true this
export cannot be used as a tier A backend and says so.

Worth recording, because it cuts against the obvious lesson: for ``bge-m3`` the ONNX
convenience output ``sentence_embedding`` **is** the model's declared pooling — cosine 1.000 to
our CLS pool — while MLX's ``text_embeds`` is not. The same shortcut is right on one runtime
and wrong on the other. That is precisely why neither is taken.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Final, override

import numpy as np

from manicule.core.errors import ConfigError, TokenStateError
from manicule.core.lifecycle import HealthReport
from manicule.embedding.artifacts import onnx_repo, onnx_weights
from manicule.embedding.base import PooledEmbedder
from manicule.embedding.cards import ModelCard
from manicule.embedding.pooling import TOKEN_STATE_RANK

if TYPE_CHECKING:
    import onnxruntime as ort

BACKEND = "onnx"

PROVIDERS: Final[tuple[str, ...]] = ("CPUExecutionProvider",)
"""CPU, deliberately, including on Apple Silicon.

CoreML would be faster here and is not used, because this backend's second job is to be the
fixed point the MLX backend is measured against. A reference that varies by accelerator
measures nothing, and the fast path on Apple hardware already exists — it is called MLX.
"""


class OnnxEmbedder(PooledEmbedder):
    """Embeds with onnxruntime, pooling in manicule's numpy."""

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
            weights_ref=onnx_repo(card.model_id, override=weights),
            batch_size=batch_size,
            cache_entries=cache_entries,
        )
        self._weights_override = weights
        self._session: ort.InferenceSession | None = None
        self._inputs: frozenset[str] = frozenset()

    @override
    def _load(self) -> None:
        """Download the export and open a session, on the thread that will run it."""
        import onnxruntime as ort  # noqa: PLC0415 - loaded with the backend, not with manicule

        weights, graph = onnx_weights(self.card.model_id, override=self._weights_override)
        if weights.describe() != self.fingerprint.weights_ref:
            msg = (
                f"the ONNX weights resolved to {weights.describe()} at setup but "
                f"{self.fingerprint.weights_ref} was recorded in the fingerprint at "
                f"construction. Vectors would be attributed to weights that did not make them."
            )
            raise ConfigError(msg)
        self._session = ort.InferenceSession(str(graph), providers=list(PROVIDERS))
        self._inputs = frozenset(item.name for item in self._session.get_inputs())

    @override
    def _unload(self) -> None:
        self._session = None
        self._inputs = frozenset()

    @override
    def _loaded(self) -> bool:
        return self._session is not None

    @override
    async def health(self) -> HealthReport:
        report = await super().health()
        if report.ok:
            return report
        return HealthReport.failing(
            report.detail,
            remedy=f"Run setup(); check that {self.fingerprint.model_id} publishes an ONNX "
            f"export, or name one with this embedder's `weights` setting.",
        )

    @override
    def _forward(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        session = self._session
        if session is None:
            msg = (
                f"{self.fingerprint.model_id} has not been loaded on the onnx backend. "
                f"setup() opens the session and the container calls it; construct this "
                f"embedder through the container, or await setup() yourself."
            )
            raise ConfigError(msg)

        feed: dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in self._inputs:
            # BERT-family exports declare it; XLM-RoBERTa ones do not. Sending an input the
            # graph does not declare is an error, and omitting one it does declare is another,
            # so it is driven by the graph rather than by the architecture name.
            feed["token_type_ids"] = np.zeros_like(input_ids)
        missing = self._inputs - feed.keys()
        if missing:
            msg = (
                f"{self.fingerprint.model_id}'s ONNX export requires inputs manicule does not "
                f"supply: {', '.join(sorted(missing))}. Exports needing extra inputs encode "
                f"assumptions this path cannot see; use the mlx backend, or an export taking "
                f"input_ids and attention_mask."
            )
            raise ConfigError(msg)

        outputs = session.run(None, feed)
        return self._token_states(outputs)

    def _token_states(self, outputs: Sequence[object]) -> np.ndarray:
        """The one rank-3 output, chosen by shape rather than by name or position."""
        candidates = [
            (index, array)
            for index, array in enumerate(outputs)
            if np.asarray(array).ndim == TOKEN_STATE_RANK
        ]
        if len(candidates) == 1:
            return np.asarray(candidates[0][1])

        names = [item.name for item in self._session.get_outputs()] if self._session else []
        shapes = ", ".join(
            f"{name}{np.asarray(array).shape}" for name, array in zip(names, outputs, strict=False)
        )
        if not candidates:
            msg = (
                f"{self.fingerprint.model_id}'s ONNX export returns no per-token hidden "
                f"states — outputs are {shapes}. Only a pooled vector is available, and its "
                f"reduction is whatever the exporter chose rather than what the model "
                f"declares, so it cannot be used: pooling is what decides whether these "
                f"vectors are comparable with anyone else's."
            )
            raise TokenStateError(msg)
        msg = (
            f"{self.fingerprint.model_id}'s ONNX export returns {len(candidates)} rank-3 "
            f"outputs ({shapes}), so which one is the encoder output cannot be decided by "
            f"shape. Name the export you mean with this embedder's `weights` setting."
        )
        raise TokenStateError(msg)


__all__ = ["BACKEND", "PROVIDERS", "OnnxEmbedder"]
