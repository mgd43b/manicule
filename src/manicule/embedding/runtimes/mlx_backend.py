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

from collections.abc import Sequence
from typing import Any, override

import numpy as np

from manicule.core.errors import ConfigError
from manicule.core.lifecycle import HealthReport, Metric
from manicule.embedding.artifacts import mlx_weights, resolve_artifact
from manicule.embedding.base import PooledEmbedder
from manicule.embedding.cards import ModelCard
from manicule.embedding.pooling import as_float32

BACKEND = "mlx"

MEGABYTE = 1024 * 1024

DEFAULT_CACHE_LIMIT_BYTES = 2048 * MEGABYTE
"""How much finished-with Metal memory MLX may keep for reuse. See :meth:`MlxEmbedder._load`.

Two gigabytes because that is comfortably above the working set of one forward pass at the
batch size an operator worried about memory actually reaches for. Measured on ``bge-m3`` fp16,
peak *live* MLX memory is 2.46 GiB at batch 1 and 4.29 GiB at batch 32; a batch-1 run holds
total physical footprint at 4.2 GiB with this bound and climbs past 25 GiB without it.
"""


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
        weights_revision: str = "",
        batch_size: int = 32,
        cache_entries: int = 10_000,
        cache_limit_bytes: int = DEFAULT_CACHE_LIMIT_BYTES,
    ) -> None:
        artifact = resolve_artifact(
            BACKEND, card.source_ref, card.revision, override=weights, revision=weights_revision
        )
        super().__init__(
            card,
            backend=BACKEND,
            weights_ref=artifact.ref,
            weights_identity=artifact.identity,
            batch_size=batch_size,
            cache_entries=cache_entries,
        )
        self._artifact = artifact
        self._cache_limit_bytes = cache_limit_bytes
        self._model: Any | None = None  # mlx-embeddings ships no type information
        # Whether this embedder has reached MLX's allocator at all. Read by `_unload`, which
        # runs on a teardown that follows a *failed* setup too — including on a machine with no
        # MLX installed, where importing it to clear a cache that cannot exist would turn a
        # clean shutdown into an ImportError.
        self._allocator_configured = False

    @override
    def _load(self) -> None:
        """Download, vet and load the weights, on the thread that will run them.

        Import and download happen here rather than at construction so that an installed
        embedder nobody selected costs one cheap import, and so that the chunker can read this
        model's sequence limit without several gigabytes arriving first.
        """
        import mlx.core as mx  # noqa: PLC0415 - see docstring
        from mlx_embeddings.utils import load_model  # noqa: PLC0415 - see docstring

        # **Bound the allocator before the first forward pass.** MLX does not return a buffer
        # to the system when a forward pass finishes with it; it keeps it in a free-buffer
        # cache, keyed by size, against a later request for the same size. That is the right
        # default for a training loop, where every step has identical shapes and the cache is
        # reused rather than grown. It is the wrong one here: manicule pads each batch to its
        # own longest member, so a batch of one gives nearly every pass a *distinct* sequence
        # length and therefore distinct buffer sizes, and almost nothing is ever reused.
        #
        # The cache is bounded only by MLX's own limit, which defaults to very nearly the whole
        # machine — measured 60.8 GiB of a 64 GiB Mac. So the free list grows until macOS starts
        # terminating processes. It is invisible to `ps`, because Metal buffers are not ordinary
        # resident anonymous pages: measured over 46 batch-one passes, physical footprint rose
        # 2.45 -> 25.0 GiB while RSS *fell*, and 96.2% of that growth was this cache. Live MLX
        # memory over the same run grew by 2 MB.
        #
        # This is process-global rather than per-embedder, which is why it is set here: on the
        # worker thread, at load, where MLX is known to be in use at all.
        mx.set_cache_limit(self._cache_limit_bytes)
        self._allocator_configured = True

        weights = mlx_weights(self._artifact)
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
        """Drop the weights, and hand the cache back rather than leaving it held.

        Dropping the model frees its buffers into the free-buffer cache, not to the system, so
        an embedder that has been torn down still holds up to the cache limit until something
        else asks MLX for memory. A process that tears an embedder down is usually shutting
        down or switching backends; in both cases the memory should leave now.
        """
        self._model = None
        if not self._allocator_configured:
            return
        import mlx.core as mx  # noqa: PLC0415 - loaded with the backend, not with manicule

        mx.clear_cache()

    @override
    def _loaded(self) -> bool:
        return self._model is not None

    @override
    def metrics(self) -> Sequence[Metric]:
        """The shared embedder metrics, plus what MLX is holding that ``ps`` will not show.

        These exist because the obvious operator question — "how much memory is embedding
        using?" — has a misleading answer on this backend. Resident memory omits Metal
        allocations entirely: during the run that motivated the cache bound, RSS sat at 1.5 GiB
        while the process held 25 GiB. An operator reading RSS alone concludes there is nothing
        to see.

        ``mlx_cache_bytes`` is the one to watch. It is memory MLX has finished with and kept;
        if it sits at the limit that is the bound working as intended, and if it exceeds the
        limit by a wide margin the bound is not being applied.
        """
        shared = tuple(super().metrics())
        if not self._allocator_configured:
            # Nothing has touched the allocator, so there are no numbers to publish — and on a
            # machine without MLX there is no module to import for them either. Reporting zero
            # would be the misleading-low-value failure this method exists to avoid.
            return shared

        import mlx.core as mx  # noqa: PLC0415 - loaded with the backend, not with manicule

        labels = {"backend": self.backend, "model": self.fingerprint.model_id}
        return (
            *shared,
            Metric(
                name="mlx_active_bytes",
                value=float(mx.get_active_memory()),
                unit="bytes",
                labels=labels,
            ),
            Metric(
                name="mlx_cache_bytes",
                value=float(mx.get_cache_memory()),
                unit="bytes",
                labels=labels,
            ),
            Metric(
                name="mlx_peak_bytes",
                value=float(mx.get_peak_memory()),
                unit="bytes",
                labels=labels,
            ),
            Metric(
                name="mlx_cache_limit_bytes",
                value=float(self._cache_limit_bytes),
                unit="bytes",
                labels=labels,
            ),
        )

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
