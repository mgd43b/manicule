"""How this suite finds a real model, and the synthetic ones it builds instead.

Both halves now live in :mod:`manicule.testing` and are re-exported here.

They moved when the MLX backend moved to its own distribution. ``manicule-mlx`` needs the same
fixtures this suite does — the same synthetic repositories, the same "are the weights here"
probes — and a second implementation of either would answer differently the first time a pin
changed. Publishing them beside :func:`~manicule.testing.assert_embedder_contract` is the same
decision for the same reason: an out-of-tree backend has to be testable on manicule's terms.

This module stays because every existing ``from tests.embedding_support import ...`` resolves
through it, and because a suite-local name is the right place to notice if manicule's own tests
ever need something the published surface should not carry.

**Nothing here imports MLX any more.** ``requires_mlx`` and ``requires_metal_allocator`` went
to ``packages/manicule-mlx/tests/test_parity.py``, which is the only suite that has ever needed
them — so manicule's test tree has no dependency, direct or optional, on the separately
licensed package.
"""

from __future__ import annotations

from manicule.testing import (
    FULL_MODEL,
    PARITY_MODEL,
    REQUIRE_MODELS_ENV,
    REQUIRED_MODELS,
    VOCABULARY,
    is_required,
    mlx_weights_available,
    model_available,
    onnx_weights_available,
    require_model,
    write_model,
    write_tokenizer,
)

__all__ = [
    "FULL_MODEL",
    "PARITY_MODEL",
    "REQUIRED_MODELS",
    "REQUIRE_MODELS_ENV",
    "VOCABULARY",
    "is_required",
    "mlx_weights_available",
    "model_available",
    "onnx_weights_available",
    "require_model",
    "write_model",
    "write_tokenizer",
]
