"""Configuration for the Ollama backend, importable without importing the HTTP client.

Registration needs this eagerly — a setting written for a component with no declared model is
rejected rather than ignored — so it lives apart from the backend and imports nothing heavier
than pydantic.

Two of the inherited fields are refused rather than inherited quietly. ``extra="forbid"`` is
doing real work on :class:`~manicule.embedding.config.EmbedderConfig`, and a field that is
accepted and then ignored is the same defect with the sign flipped: the operator believes a
setting is in force and it is not.
"""

from __future__ import annotations

import re
from typing import Final, Self

from pydantic import Field, model_validator

from manicule.embedding.config import EmbedderConfig

DEFAULT_BASE_URL: Final = "http://localhost:11434"

_COMMIT: Final = re.compile(r"[0-9a-f]{40}")


class OllamaEmbedderConfig(EmbedderConfig):
    """:class:`~manicule.embedding.config.EmbedderConfig`, plus what a *served* model needs.

    Every field here exists because the runtime is a server rather than a file on disk, and
    the two things a file gives you for free — a tokenizer, and a limit you can read — have to
    be supplied or measured instead.
    """

    base_url: str = Field(
        default=DEFAULT_BASE_URL,
        min_length=1,
        description="Where the Ollama server is. The host is **not** part of the embedding "
        "fingerprint: moving the same model to another server does not change the vectors, "
        "and putting a deployment address inside an index's identity would mean a corpus "
        "could not be read by a second replica of the thing that wrote it.",
    )
    tokenizer: str = Field(
        default="",
        description="A Hugging Face repository id, or a local directory, holding the "
        "``tokenizer.json`` that matches the served model. **Required, and deliberately not "
        "derived from the model name.** Ollama serves GGUF and exposes no tokenizer, while "
        "manicule counts tokens to decide chunk boundaries and to refuse input the model "
        "would truncate — so a vocabulary has to come from somewhere. Guessing one from "
        "``qwen3-embedding:0.6b`` would be an inference presented as a measurement; naming "
        "one is a claim, and this backend checks it against the server's own "
        "``prompt_eval_count`` at setup and refuses a disagreement.",
    )
    tokenizer_revision: str = Field(
        default="",
        description="Immutable 40-character commit for a remote ``tokenizer`` repository. "
        "Required with one, and rejected for a local path: a branch or tag can change the "
        "vocabulary without changing the name, and the vocabulary is what chunk boundaries "
        "were measured with.",
    )
    num_ctx: int | None = Field(
        default=None,
        gt=0,
        description="The context Ollama is asked to serve, in **total** tokens. ``None`` "
        "means the model's own declared context length, read from ``/api/show``. This is not "
        "cosmetic: measured against a server holding ``qwen3-embedding:0.6b``, which declares "
        "32768, Ollama served 4096 when asked for nothing and truncated everything past it "
        "without an error. Lower it to bound the KV cache the server allocates; it can never "
        "raise the limit past what the architecture declares, and the effective number is "
        "measured at setup rather than assumed.",
    )
    timeout_s: float = Field(
        default=120.0,
        gt=0,
        description="Total wall clock for one embedding request. Generous because a cold "
        "model is loaded into GPU memory first, which is a real multi-second cost the first "
        "time and not a symptom of anything.",
    )
    connect_timeout_s: float = Field(
        default=10.0,
        gt=0,
        description="Establishing the connection only. Separate from ``timeout_s`` because "
        "the two failures need different answers: an unreachable server should say so in "
        "seconds, and a slow forward pass should be waited for.",
    )
    keep_alive: str = Field(
        default="5m",
        min_length=1,
        description="How long the server keeps the model resident between requests, in "
        "Ollama's own duration syntax. The default is Ollama's; an ingest run that pauses "
        "longer than this pays a model load on its next batch.",
    )

    @model_validator(mode="after")
    def _refuse_settings_this_backend_cannot_honor(self) -> Self:
        """Reject inherited fields that name a mechanism a served model does not have.

        ``weights`` and ``weights_revision`` describe an artifact *this process* loads. Here
        the weights are on the other side of an HTTP connection, chosen by whoever ran
        ``ollama pull``, and nothing in this package could act on either value. Accepting them
        silently would leave an operator believing they had pinned something.
        """
        if self.weights or self.weights_revision:
            msg = (
                "the ollama embedder cannot honor `weights` or `weights_revision`: the model "
                "is held by the server, not loaded from an artifact this process resolves. "
                "Pin it with `ollama pull <model>@<digest>` on the server instead — the "
                "digest the server reports is recorded in this backend's weights_identity, "
                "so a re-pull that changes it invalidates the vectors it made."
            )
            raise ValueError(msg)
        if self.tokenizer_revision and not self.tokenizer:
            msg = "`tokenizer_revision` names a revision of nothing: set `tokenizer` as well"
            raise ValueError(msg)
        if self.tokenizer_revision and not _COMMIT.fullmatch(self.tokenizer_revision):
            msg = (
                f"`tokenizer_revision` must be an exact 40-character commit, got "
                f"{self.tokenizer_revision!r}. A branch or tag can change the vocabulary "
                f"without changing the name, and every chunk boundary in the corpus was "
                f"measured with it."
            )
            raise ValueError(msg)
        return self


__all__ = ["DEFAULT_BASE_URL", "OllamaEmbedderConfig"]
