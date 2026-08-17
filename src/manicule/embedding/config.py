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
    weights_revision: str = Field(
        default="",
        description="Immutable 40-character commit for an explicit remote weights repository. "
        "Required with remote `weights`; rejected for local paths. Built-in routes are pinned.",
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


# A backend with a setting of its own subclasses `EmbedderConfig` in its **own** distribution
# and registers that subclass — `manicule-mlx` does exactly this for its Metal cache bound.
# There is deliberately no per-backend model here: one that named a mechanism only some
# backends have would have to be accepted by all of them, and `extra="forbid"` is doing real
# work. A cache limit written under `[plugins.config."embedder.onnx"]` names something
# onnxruntime does not have, and silently accepting it would leave an operator believing they
# had bounded something.

__all__ = ["EmbedderConfig"]
