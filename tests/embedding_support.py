"""Synthetic model repositories, and how the suite finds a real one.

Most of what the embedding stack decides — the reduction, the width, the usable length, which
artifact runs, what the cache is keyed on — is decided from a handful of small JSON files. A
model directory can therefore be *built*, which lets those decisions be tested exhaustively,
offline, in milliseconds, including the cases no published model would ever ship: two pooling
flags at once, a dimension that disagrees with itself, a missing sequence length.

The real models are still needed, for the one thing a synthetic repository cannot check: that
two runtimes agree. Those tests find their weights in the local Hugging Face cache and skip when
it is empty — except for the models named in ``REQUIRE_EMBEDDING_MODELS``, where a missing model
fails instead. CI names what it pre-seeded, because a conformance suite that skips certifies
nothing.
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

REQUIRE_MODELS_ENV: Final = "REQUIRE_EMBEDDING_MODELS"
"""Which models must be present rather than merely welcome.

A comma-separated list of model ids, or ``all``. CI sets it to exactly what it pre-seeded, so a
green job means those suites *ran*; a model outside the list still skips when absent, which is
what lets ``bge-m3`` be exercised on a developer's machine without putting 4.6 GB of downloads
in every CI run.

**Deliberately outside manicule's ``MANICULE_`` namespace**, and the reason is a bug this
caught. ``manicule_environment`` deletes every ``MANICULE_``-prefixed variable before each test,
so that a developer's own configuration cannot leak into the suite. The first version of this
switch was called ``MANICULE_REQUIRE_EMBEDDING_MODELS`` and was therefore deleted before it was
ever read: on a runner with no weights every case skipped and the job reported green — the exact
failure the switch exists to prevent, inside the mechanism meant to prevent it. This is not
application configuration; it configures the suite, and it is named accordingly.
"""

REQUIRED_MODELS: Final[frozenset[str]] = frozenset(
    name.strip() for name in os.environ.get(REQUIRE_MODELS_ENV, "").split(",") if name.strip()
)
"""Read once, at import time, before any fixture has had a chance to touch the environment.

Belt and braces against the failure above: the rename is what fixes it, and reading it here is
what stops a future fixture from silently undoing the fix.
"""

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


def is_required(model_id: str) -> bool:
    """Whether this model must be present for the suite to be considered to have run."""
    return "all" in REQUIRED_MODELS or model_id in REQUIRED_MODELS


def require_model(model_id: str, *, mlx: bool = False, onnx: bool = False) -> None:
    """Skip unless the named weights are downloaded — or fail, if this model is required."""
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


def requires_mlx(model_id: str) -> None:
    """Skip unless MLX works here — or fail, if a model requiring it was named.

    Takes the model so that "MLX is missing" fails exactly where "the weights are missing"
    would. On a runner that is supposed to be measuring backend parity, an absent runtime and
    absent weights are the same outcome: the comparison did not happen.
    """
    from manicule.embedding.runtimes import mlx_usable  # noqa: PLC0415 - an embeddings extra

    if mlx_usable():
        return
    if is_required(model_id):
        pytest.fail(
            f"MLX is not usable on this machine, and {REQUIRE_MODELS_ENV} names {model_id}. "
            f"Backend parity cannot be measured with one backend"
        )
    pytest.skip("MLX is not usable on this machine")


def requires_metal() -> None:
    """Skip unless MLX is executing on **Metal**, not on its CPU device.

    :func:`requires_mlx` is a different question and does not answer this one. It asks whether
    ``mx.eval`` works, and it works on MLX's CPU device too. So do the memory counters: with
    ``mx.set_default_device(mx.cpu)``, a 16 MiB allocation reports ``get_active_memory() ==
    16777216``, indistinguishable from the Metal figure. A test that asserted "MLX reports
    non-zero memory" would therefore pass on a machine with no GPU **having exercised the wrong
    allocator entirely** — green while checking nothing, which is the failure mode this module
    exists to prevent.

    It matters because the thing being guarded is specifically a Metal phenomenon: the
    unbounded free-buffer cache is the Metal allocator's, the memory it retains is unified
    memory, and the reason ``ps`` cannot see it is that Metal buffers are not ordinary resident
    pages. None of that is true of the CPU device.

    A skip rather than a failure, because this is a property of the hardware rather than of a
    pre-seed somebody forgot — there is no action that would fix it on a machine without a GPU.
    The reason names the device so a CI log says which allocator actually ran.
    """
    from manicule.embedding.runtimes import mlx_core  # noqa: PLC0415 - an embeddings extra

    mx = mlx_core()
    if mx.metal.is_available():
        return
    pytest.skip(
        f"MLX is running on {mx.default_device()}, not Metal, so the Metal allocator this "
        f"asserts about was never exercised"
    )


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
    "REQUIRED_MODELS",
    "REQUIRE_MODELS_ENV",
    "VOCABULARY",
    "is_required",
    "mlx_weights_available",
    "model_available",
    "onnx_weights_available",
    "require_model",
    "requires_metal",
    "requires_mlx",
    "write_model",
    "write_tokenizer",
]
