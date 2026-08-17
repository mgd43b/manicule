"""Configuration for the MLX backend, importable without importing MLX.

Registration needs this eagerly — a setting written for a component with no declared model is
rejected rather than ignored — so it lives apart from the backend and imports nothing heavier
than pydantic.
"""

from __future__ import annotations

from pydantic import Field

from manicule.embedding.config import EmbedderConfig


class MlxEmbedderConfig(EmbedderConfig):
    """:class:`~manicule.embedding.config.EmbedderConfig`, plus the one setting that is MLX's
    alone.

    Separate from the shared model because ``extra="forbid"`` is doing real work: a cache limit
    written under ``[plugins.config."embedder.onnx"]`` names a mechanism onnxruntime does not
    have, and silently accepting it would leave an operator believing they had bounded
    something.

    It subclasses a model from an MIT package, which is the direction that composes: this file
    is GPL-3.0-or-later, and nothing about that reaches back into what it inherits from.
    """

    cache_limit_mb: int = Field(
        default=2048,
        ge=0,
        description="Ceiling on MLX's Metal **free-buffer cache** — buffers a forward pass has "
        "finished with that MLX keeps for reuse instead of returning to the system. Its own "
        "default is near the whole machine (measured: 60.8 GiB of a 64 GiB Mac), so an "
        "unbounded run retains every distinct buffer size it has ever seen and climbs until "
        "macOS intervenes. ``0`` returns every buffer immediately, which is bounded but "
        "forfeits reuse. This is retained memory, not a working-set limit: a forward pass "
        "larger than this still runs.",
    )


__all__ = ["MlxEmbedderConfig"]
