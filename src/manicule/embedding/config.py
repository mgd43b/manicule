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
        description="The artefact this backend should execute, when it is not the model's own "
        "repository — a repository id or a local path. Empty means the built-in resolution: "
        "the model's own files where the backend can read them, and the recorded unquantised "
        "conversion where it cannot. A quantised artefact is refused whichever way it arrives.",
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


__all__ = ["EmbedderConfig"]
