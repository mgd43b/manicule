"""Synthetic model repositories, and how the suite finds a real one.

Most of what the embedding stack decides — the reduction, the width, the usable length, which
artefact runs, what the cache is keyed on — is decided from a handful of small JSON files. A
model directory can therefore be *built*, which lets those decisions be tested exhaustively,
offline, in milliseconds, including the cases no published model would ever ship: two pooling
flags at once, a dimension that disagrees with itself, a missing sequence length.

The real models are still needed, for the one thing a synthetic repository cannot check: that
two runtimes agree. Those tests find their weights in the local Hugging Face cache and skip
when it is empty — except under ``MANICULE_REQUIRE_EMBEDDING_MODELS``, where a missing model
fails instead. CI sets it, because a conformance suite that skips certifies nothing.
"""

# This module is the suite's seam over libraries that ship no type information — MLX,
# huggingface-hub and tokenizers — mirroring what src/manicule/embedding/runtimes does for the
# implementation. Only the rules that report an Unknown coming out of one of them are relaxed,
# and only here; every test module that uses these helpers is checked strictly.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Final

import pytest

REQUIRE_MODELS_ENV: Final = "MANICULE_REQUIRE_EMBEDDING_MODELS"
"""Set in CI. Turns "this model is not downloaded" from a skip into a failure."""

PARITY_MODEL: Final = "BAAI/bge-small-en-v1.5"
"""The model the backend parity suite runs on by default.

Small — about 130 MB per runtime — because parity is a property of the *runtimes*, and a
runtime that agrees with the other on a BERT encoder and disagrees on an XLM-RoBERTa one has a
bug worth finding either way. ``BAAI/bge-m3`` is the shipped default and is checked here too
whenever it happens to be present, which on a developer's machine it usually is; requiring
4.6 GB of downloads on every CI run to learn the same thing would not buy anything.
"""

FULL_MODEL: Final = "BAAI/bge-m3"
"""manicule's configured model. Exercised when its weights are already on the machine."""


def model_available(model_id: str) -> bool:
    """Whether a model's declaration is on local disk, without touching the network."""
    return _cached(model_id, ("config.json", "tokenizer.json"))


def mlx_weights_available(model_id: str) -> bool:
    """Whether the MLX weights for ``model_id`` are cached."""
    from manicule.embedding.artifacts import mlx_repo  # noqa: PLC0415 - an embeddings extra

    return _cached(mlx_repo(model_id), ("*.safetensors",))


def onnx_weights_available(model_id: str) -> bool:
    """Whether the ONNX export for ``model_id`` is cached."""
    return _cached(model_id, ("onnx/model.onnx",))


def require_model(model_id: str, *, mlx: bool = False, onnx: bool = False) -> None:
    """Skip — or under CI, fail — unless the named weights are already downloaded."""
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
    if os.environ.get(REQUIRE_MODELS_ENV):
        pytest.fail(
            f"{detail} not in the local model cache, and {REQUIRE_MODELS_ENV} is set. "
            f"Pre-seed with tools/prefetch_embedding_models.py; a skipped conformance suite "
            f"reports green while checking nothing."
        )
    pytest.skip(f"{detail} not downloaded. Run tools/prefetch_embedding_models.py to enable.")


def requires_mlx() -> None:
    """Skip — or under CI, fail — unless the MLX runtime is importable and working."""
    try:
        import mlx.core as mx  # noqa: PLC0415 - Apple-only, and optional off Apple Silicon

        mx.eval(mx.array([1.0]) + 1)
    except Exception as error:  # noqa: BLE001 - any failure here means "no working MLX"
        if os.environ.get(REQUIRE_MODELS_ENV):
            pytest.fail(
                f"MLX is not usable on this machine ({error}), and {REQUIRE_MODELS_ENV} is set"
            )
        pytest.skip(f"MLX is not usable on this machine ({error})")


def _cached(repo: str, patterns: tuple[str, ...]) -> bool:
    from huggingface_hub import snapshot_download  # noqa: PLC0415 - an embeddings extra

    try:
        snapshot_download(repo, allow_patterns=list(patterns), local_files_only=True)
    except Exception:  # noqa: BLE001 - hub raises several unrelated types for "not cached"
        return False
    return True


# --- synthetic repositories ---------------------------------------------------------------

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
not on its embeddings, and a real vocabulary would only slow the suite down."""


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


__all__ = [
    "FULL_MODEL",
    "PARITY_MODEL",
    "REQUIRE_MODELS_ENV",
    "VOCABULARY",
    "mlx_weights_available",
    "model_available",
    "onnx_weights_available",
    "require_model",
    "requires_mlx",
    "write_model",
    "write_tokenizer",
]
