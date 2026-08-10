"""``sentence-transformers`` behind :class:`~manicule.retrieval.rerank.PairScorer`.

Loaded lazily and never at all under the ``fast`` profile, which is the profile's whole point:
it is the one where a second model is not on the query path, and everything else that differs
between the profiles is a rounding error beside that.

The default model is the family pair for the embedder. That is not brand tidiness — the
embedder was chosen substantially because it is multilingual in one space, and a monolingual
reranker placed after it would take a correctly-retrieved non-English passage and rank it down,
undoing that property at the last stage before the answer, where it is least visible. The cost
is honest and stated: it is an XLM-RoBERTa-large-sized model, meaningfully larger than the
small English rerankers it would be compared against, and on an all-English corpus one of those
is very likely the better trade. That is what the configuration knob is for.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from manicule.core.errors import ManiculeError

if TYPE_CHECKING:
    from collections.abc import Sequence


class RerankerUnavailableError(ManiculeError):
    """The reranker was configured but its runtime is not installed or would not load.

    Fatal to the query rather than a fallback to no reranking, for the same reason the stage
    has no ``try``/``except``: a profile that says it reranks and silently did not has told an
    evaluation harness which pipeline ran, and been wrong.
    """


class CrossEncoderScorer:
    """A ``sentence-transformers`` cross-encoder, scored off the event loop.

    ``predict`` is a synchronous forward pass that holds the GIL for as long as it runs, so it
    goes through a worker thread. Everything that touches the framework is inside this class.
    """

    def __init__(
        self,
        model: str,
        *,
        batch_size: int = 16,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        self.model_id = model
        self._batch_size = batch_size
        self._device = device
        self._max_length = max_length
        self._encoder: Any = None

    async def setup(self) -> None:
        """Load the weights.

        In ``setup`` rather than on first use, because the container only constructs a
        reranker when the profile asks for one — so "load it when it exists" and "load it when
        it is needed" are the same moment, and the first query does not stall on a download.

        Raises:
            RerankerUnavailableError: The extra is not installed, or the model would not load.
        """
        if self._encoder is not None:
            return
        try:
            # Deferred: this is where torch and the transformers stack are loaded.
            from sentence_transformers import CrossEncoder  # noqa: PLC0415
        except ImportError as error:
            msg = (
                f"reranking needs sentence-transformers, which is not installed. Install "
                f"manicule's 'rerank' extra, choose a profile that does not rerank, or set "
                f"rag.reranker to null. Requested model: {self.model_id!r}."
            )
            raise RerankerUnavailableError(msg) from error

        def load() -> Any:  # noqa: ANN401 - the library ships no usable type for this
            return CrossEncoder(self.model_id, device=self._device, max_length=self._max_length)

        try:
            self._encoder = await asyncio.to_thread(load)
        except Exception as error:
            msg = (
                f"reranker {self.model_id!r} would not load: {error}. Check the model name and "
                f"that its weights are reachable; retrieval will not fall back to an unreranked "
                f"ranking, because a profile that reports reranking must have reranked."
            )
            raise RerankerUnavailableError(msg) from error

    async def teardown(self) -> None:
        """Release the model. Safe to call when none was ever loaded."""
        self._encoder = None

    async def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """One relevance logit per pair, in order.

        Raises:
            RerankerUnavailableError: :meth:`setup` has not run.
        """
        if not pairs:
            return []
        encoder = self._encoder
        if encoder is None:
            msg = (
                f"reranker {self.model_id!r} was asked to score before its weights were "
                f"loaded. The container calls setup() in dependency order; a scorer built "
                f"outside it has to do the same."
            )
            raise RerankerUnavailableError(msg)

        def predict() -> list[float]:
            raw: Any = encoder.predict(list(pairs), batch_size=self._batch_size)
            return [float(value) for value in raw]

        return await asyncio.to_thread(predict)


__all__ = ["CrossEncoderScorer", "RerankerUnavailableError"]
