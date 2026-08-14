"""Per-embedder configuration, importable without importing a model runtime.

Registration needs this eagerly — a setting written for a component with no declared model is
rejected rather than ignored — so it lives apart from the backends and imports nothing heavier
than pydantic.

Three fields, and every one of them exists to be *unnecessary* for a model that describes
itself properly. They are how a repository that declares nothing is used without manicule
guessing on its behalf.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from manicule.core.embedding import Pooling


class EmbedderConfig(BaseModel):
    """Settings for one embedder backend.

    The model, revision, batch size and cache size are not here: they are properties of the
    embedding as a whole and live under ``[embedding]``, so switching backend does not mean
    restating them and cannot mean accidentally changing one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    weights: str = Field(
        default="",
        description="The artifact this backend should execute, when it is not the model's own "
        "repository — a repository id or a local path. Empty means the built-in resolution: "
        "the model's own files where the backend can read them, and the recorded unquantized "
        "conversion where it cannot. A quantized artifact is refused whichever way it arrives.",
    )
    pooling: Pooling | None = Field(
        default=None,
        description="The reduction, for a model that declares none. Consulted only in that "
        "case: naming one that contradicts the model's own 1_Pooling/config.json is refused, "
        "because that setting would succeed and produce vectors from a reduction the weights "
        "were never trained for.",
    )
    max_sequence_length: int | None = Field(
        default=None,
        gt=0,
        description="Usable **content** tokens — the model's limit less the special tokens it "
        "wraps every input in. For a model that declares no max_seq_length. Set it too high "
        "and the model silently drops the tail of every long chunk, so there is no default.",
    )


class MlxEmbedderConfig(EmbedderConfig):
    """:class:`EmbedderConfig`, plus the one setting that is MLX's alone.

    Separate from the shared model because ``extra="forbid"`` is doing real work: a cache
    limit written under ``[embedders.onnx]`` names a mechanism onnxruntime does not have, and
    silently accepting it would leave an operator believing they had bounded something.
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


__all__ = ["EmbedderConfig", "MlxEmbedderConfig"]
