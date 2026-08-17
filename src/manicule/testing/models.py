"""Synthetic model repositories, for testing an embedder without downloading one.

Most of what the embedding stack decides — the reduction, the width, the usable length, which
artifact runs, what the cache is keyed on — is decided from a handful of small JSON files. A
model directory can therefore be *built*, which lets those decisions be exercised exhaustively,
offline, in milliseconds, including the cases no published model would ever ship: two pooling
flags at once, a dimension that disagrees with itself, a missing sequence length.

**Published rather than kept in manicule's own test tree**, because a backend is allowed to
live in another distribution — ``manicule-mlx`` is the first — and its suite needs the same
fixtures manicule's does. The alternative was a second implementation of "what does a model
repository look like", which is the sort of duplicate that agrees on the day it is written and
not afterwards. It sits beside :func:`~manicule.testing.assert_embedder_contract` for the same
reason: an out-of-tree backend has to be testable on manicule's terms.
"""

# `tokenizers` and `huggingface-hub` ship no type information, so every call into them is an
# Unknown that spreads. Relaxed for this file only, and only the rules that report exactly that
# — this is the same suppression these functions carried in the test tree they came from, and
# the same one `manicule/embedding/runtimes/` carries for the same two libraries.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Final

VOCABULARY: Final[tuple[str, ...]] = (
    "<s>",
    "</s>",
    "<pad>",
    "<unk>",
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
)
"""A word-level vocabulary. Small on purpose: a synthetic model is checked on its declaration,
not on what it would say about real text."""


def write_model(
    directory: Path,
    *,
    pooling_flags: dict[str, bool] | None = None,
    hidden_size: int | None = 8,
    word_embedding_dimension: int | None = 8,
    max_seq_length: int | None = 32,
    max_position_embeddings: int = 64,
    model_type: str = "xlm-roberta",
    pad_token_id: int = 2,
    quantization: dict[str, object] | None = None,
    tokenizer: bool = True,
    pooling_file: bool = True,
) -> Path:
    """Write a model repository with exactly the declarations a test wants to talk about.

    Every argument that can be ``None`` corresponds to a field a real repository is free to
    omit — which is the interesting case, because omission is where guessing would start.
    """
    directory.mkdir(parents=True, exist_ok=True)

    config: dict[str, object] = {
        "model_type": model_type,
        "architectures": ["XLMRobertaModel"],
        "max_position_embeddings": max_position_embeddings,
        "pad_token_id": pad_token_id,
    }
    if hidden_size is not None:
        config["hidden_size"] = hidden_size
    if quantization is not None:
        config["quantization"] = quantization
    _write_json(directory / "config.json", config)

    if pooling_file:
        pooling: dict[str, object] = dict(
            pooling_flags if pooling_flags is not None else {"pooling_mode_cls_token": True}
        )
        if word_embedding_dimension is not None:
            pooling["word_embedding_dimension"] = word_embedding_dimension
        _write_json(directory / "1_Pooling" / "config.json", pooling)

    if max_seq_length is not None:
        _write_json(directory / "sentence_bert_config.json", {"max_seq_length": max_seq_length})

    _write_json(directory / "tokenizer_config.json", {"pad_token": "<pad>"})
    if tokenizer:
        write_tokenizer(directory / "tokenizer.json")
    return directory


def write_tokenizer(path: Path) -> None:
    """A word-level tokenizer that wraps input in ``<s> … </s>``, like XLM-RoBERTa does."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors  # noqa: PLC0415

    vocab = {token: index for index, token in enumerate(VOCABULARY)}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))  # noqa: S106 - a vocabulary entry, not a credential
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> $B </s>",
        special_tokens=[("<s>", vocab["<s>"]), ("</s>", vocab["</s>"])],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


# --- finding real weights ------------------------------------------------------------------
#
# A conformance suite that skips certifies nothing, so these answer "are the weights here"
# without touching the network, and `REQUIRE_MODELS_ENV` turns a skip into a failure on a
# runner that was supposed to have pre-seeded them.

REQUIRE_MODELS_ENV: Final = "REQUIRE_EMBEDDING_MODELS"
"""Which models must be present rather than merely welcome: a comma-separated list, or ``all``.

**Deliberately outside manicule's ``MANICULE_`` namespace**, and the reason is a bug this
caught. The test environment fixture deletes every ``MANICULE_``-prefixed variable before each
test, so that a developer's own configuration cannot leak into a suite. The first version of
this switch was called ``MANICULE_REQUIRE_EMBEDDING_MODELS`` and was therefore deleted before
it was ever read: on a runner with no weights every case skipped and the job reported green —
the exact failure the switch exists to prevent, inside the mechanism meant to prevent it. This
is not application configuration; it configures a suite, and it is named accordingly.
"""

REQUIRED_MODELS: Final[frozenset[str]] = frozenset(
    name.strip() for name in os.environ.get(REQUIRE_MODELS_ENV, "").split(",") if name.strip()
)
"""Read once, at import time, before any fixture has had a chance to touch the environment."""

PARITY_MODEL: Final = "BAAI/bge-small-en-v1.5"
"""The model a backend parity suite runs on by default. Small — about 130 MB per runtime."""

FULL_MODEL: Final = "BAAI/bge-m3"
"""manicule's configured model. Exercised when its weights are already on the machine."""


def is_required(model_id: str) -> bool:
    """Whether this model must be present for a suite to be considered to have run."""
    return "all" in REQUIRED_MODELS or model_id in REQUIRED_MODELS


def model_available(model_id: str) -> bool:
    """Whether a model's declaration is on local disk, without touching the network."""
    from manicule.embedding.artifacts import builtin_model_revision  # noqa: PLC0415

    return _cached(model_id, ("config.json", "tokenizer.json"), builtin_model_revision(model_id))


def mlx_weights_available(model_id: str) -> bool:
    """Whether the MLX weights for ``model_id`` are cached.

    Here rather than in ``manicule-mlx`` although it names that backend, because what it reads
    is manicule's own artifact resolution: ``mlx_repo`` and the pinned revision stay in this
    distribution so that ``_qualified_identity`` keeps hashing both arms of the backend pair.
    """
    from manicule.embedding.artifacts import (  # noqa: PLC0415 - an embeddings extra
        builtin_revision,
        mlx_repo,
    )

    return _cached(mlx_repo(model_id), ("*.safetensors",), builtin_revision(model_id, "mlx"))


def onnx_weights_available(model_id: str) -> bool:
    """Whether the ONNX export for ``model_id`` is cached."""
    from manicule.embedding.artifacts import builtin_revision  # noqa: PLC0415

    return _cached(model_id, ("onnx/model.onnx",), builtin_revision(model_id, "onnx"))


def require_model(model_id: str, *, mlx: bool = False, onnx: bool = False) -> None:
    """Skip unless the named weights are downloaded — or fail, if this model is required."""
    import pytest  # noqa: PLC0415 - only a suite calls this

    missing: list[str] = []
    if not model_available(model_id):
        missing.append(f"{model_id} (declaration)")
    if mlx and not mlx_weights_available(model_id):
        missing.append(f"{model_id} (MLX weights)")
    if onnx and not onnx_weights_available(model_id):
        missing.append(f"{model_id} (ONNX export)")
    if not missing:
        return

    detail = ", ".join(missing)
    if is_required(model_id):
        pytest.fail(
            f"{detail} not in the local model cache, and {REQUIRE_MODELS_ENV} names "
            f"{model_id}. Pre-seed with tools/prefetch_embedding_models.py; a skipped "
            f"conformance suite reports green while checking nothing."
        )
    pytest.skip(f"{detail} not downloaded. Run tools/prefetch_embedding_models.py to enable.")


def _cached(repo: str, patterns: tuple[str, ...], revision: str | None = None) -> bool:
    from huggingface_hub import snapshot_download  # noqa: PLC0415 - an embeddings extra

    try:
        snapshot_download(
            repo, revision=revision, allow_patterns=list(patterns), local_files_only=True
        )
    except Exception:  # noqa: BLE001 - hub raises several unrelated types for "not cached"
        return False
    return True


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
