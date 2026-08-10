#!/usr/bin/env python3
"""Download the model weights the embedding suites measure against.

The embedding tests never download anything themselves: a test that fetches gigabytes on first
run is a test nobody runs twice, and one that fetches them *in CI* is a test that fails when
the network does. They look in the local model cache and skip what is missing — so this script
is what makes them run, and CI calls it before setting
``MANICULE_REQUIRE_EMBEDDING_MODELS=1``, which turns a skip into a failure.

Two models, and the difference between them is deliberate:

``BAAI/bge-small-en-v1.5``
    About 130 MB per runtime. Backend parity is a property of the *runtimes*, so it is checked
    on every run against a model small enough to make that affordable.

``BAAI/bge-m3``
    manicule's configured model, and about 4.6 GB across both runtimes. Fetched only with
    ``--full``, and exercised by the suite whenever it happens to be present — which on a
    developer's machine it usually is.

Usage::

    uv run tools/prefetch_embedding_models.py           # the parity model
    uv run tools/prefetch_embedding_models.py --full    # and BAAI/bge-m3
    uv run tools/prefetch_embedding_models.py --mlx     # include the MLX weights
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from manicule.embedding.artifacts import mlx_repo  # noqa: E402
from manicule.embedding.cards import CARD_FILES  # noqa: E402

PARITY_MODEL = "BAAI/bge-small-en-v1.5"
FULL_MODEL = "BAAI/bge-m3"


def fetch(repo: str, patterns: list[str]) -> Path:
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    print(f"  {repo}  {patterns}")
    return Path(snapshot_download(repo, allow_patterns=patterns))


def prefetch(model_id: str, *, mlx: bool) -> None:
    print(f"{model_id}:")
    # The declaration, always: pooling, dimension and sequence length are read from the
    # canonical repository even when the weights come from a conversion.
    fetch(model_id, [*CARD_FILES])
    # `onnx/*` rather than `onnx/model.onnx`: bge-m3's graph is a few hundred kilobytes of
    # structure pointing at its weights in a sibling `model.onnx_data`, and opening the graph
    # without the sidecar loads a model with no parameters in it.
    fetch(model_id, ["onnx/*"])
    if mlx:
        fetch(mlx_repo(model_id), ["*.safetensors", "*.json"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help=f"also fetch {FULL_MODEL}")
    parser.add_argument("--mlx", action="store_true", help="also fetch the MLX conversions")
    arguments = parser.parse_args()

    models = [PARITY_MODEL, *([FULL_MODEL] if arguments.full else [])]
    for model_id in models:
        prefetch(model_id, mlx=arguments.mlx)
    print(f"fetched: {', '.join(models)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
