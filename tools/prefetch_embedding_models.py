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

    uv run tools/prefetch_embedding_models.py --backend mlx    # what *this install* runs

``--backend`` is the operator's form and the others are the suite's. The flags above are
additive by design — CI wants parity weights for both runtimes — which makes them the wrong
answer for somebody who only wants to take their first ``index``'s download now: on Apple
silicon ``--full --mlx`` fetches the parity model, bge-m3's 2.3 GB ONNX export and the 1.15 GB
MLX conversion, about 3.6 GB, to seed a backend that will load 1.17 GB of it. ``--backend``
fetches the configured model's card files and exactly one runtime's weights, and nothing else.

``--backend ollama`` is a fourth, different shape, because a served model is a different shape
of dependency. There are no weights to fetch here and no canonical model repository to read a
declaration from: the model runs on the Ollama server, and the GGUF it serves carries its own
metadata, read at run time by ``manicule_ollama.served.resolve``. What this backend still needs
from the hub is the tokenizer an operator names in ``[plugins.config."embedder.ollama"]``,
because Ollama exposes none of its own — narrowed, the same way
``manicule_ollama.served.TOKENIZER_FILES`` narrows it, to ``tokenizer.json`` alone.
``--tokenizer``/``--tokenizer-revision`` supply the repository and its commit directly, which is
what an image build uses where there is no config file yet to read; without them the script
reads the same setting the running backend would. Neither route naming one is a refusal, an
unpinned repository id is a refusal, and a ``tokenizer`` that is already a local directory needs
no fetch at all — the script says so and exits ``0``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from manicule.embedding.artifacts import (  # noqa: E402
    builtin_model_revision,
    builtin_revision,
    mlx_repo,
)
from manicule.embedding.cards import CARD_FILES  # noqa: E402

PARITY_MODEL = "BAAI/bge-small-en-v1.5"
FULL_MODEL = "BAAI/bge-m3"

NO_OLLAMA_TOKENIZER_MSG = (
    "the ollama embedder needs a tokenizer and none was given: pass `--tokenizer REPO` (with "
    "`--tokenizer-revision SHA` for a repository id) or configure "
    '`[plugins.config."embedder.ollama"]` with `tokenizer` (and `tokenizer_revision`) the way '
    "manicule-ollama itself reads it. Ollama serves GGUF and exposes no tokenizer, so there is "
    "no default to fall back to — for `qwen3-embedding:0.6b` that is "
    "`Qwen/Qwen3-Embedding-0.6B`; for `nomic-embed-text` it is `nomic-ai/nomic-embed-text-v1.5`."
)


def fetch(repo: str, patterns: list[str], revision: str | None) -> Path:
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    print(f"  {repo}@{revision or 'HEAD'}  {patterns}")
    return Path(snapshot_download(repo, revision=revision, allow_patterns=patterns))


def prefetch(model_id: str, *, mlx: bool) -> None:
    print(f"{model_id}:")
    # The declaration, always: pooling, dimension and sequence length are read from the
    # canonical repository even when the weights come from a conversion.
    fetch(model_id, [*CARD_FILES], builtin_model_revision(model_id))
    # `onnx/*` rather than `onnx/model.onnx`: bge-m3's graph is a few hundred kilobytes of
    # structure pointing at its weights in a sibling `model.onnx_data`, and opening the graph
    # without the sidecar loads a model with no parameters in it.
    fetch(model_id, ["onnx/*"], builtin_revision(model_id, "onnx"))
    if mlx:
        fetch(
            mlx_repo(model_id),
            ["*.safetensors", "*.json"],
            builtin_revision(model_id, "mlx"),
        )


def for_backend(model_id: str, backend: str) -> None:
    """Fetch exactly what ``backend`` will load for ``model_id``, and nothing beside it.

    The declaration is fetched either way — pooling, dimension and sequence length are read
    from the canonical repository whichever runtime executes the weights — and then one
    runtime's artifact. This is what `manicule doctor` names when it reports that a first
    index has a download in front of it.
    """
    print(f"{model_id} for the {backend} backend:")
    fetch(model_id, [*CARD_FILES], builtin_model_revision(model_id))
    if backend == "mlx":
        fetch(
            mlx_repo(model_id),
            ["*.safetensors", "*.json"],
            builtin_revision(model_id, "mlx"),
        )
    else:
        fetch(model_id, ["onnx/*", "*.json"], builtin_revision(model_id, "onnx"))


def configured_ollama_tokenizer() -> tuple[str, str]:
    """``tokenizer`` and ``tokenizer_revision`` from ``[plugins.config."embedder.ollama"]``.

    Reached only when ``--tokenizer`` was not given, so this is what an operator who already
    wrote the setting for ``manicule-ollama`` itself gets instead of restating it on a command
    line. Read raw rather than validated against ``OllamaEmbedderConfig`` here — that validation
    runs once, in :func:`for_ollama`, against whichever route actually supplied a value, so a bad
    setting is reported once rather than twice and the two reports cannot disagree.
    """
    from manicule.config.settings import Settings  # noqa: PLC0415 - kept out of import time, and
    # out of tests/test_ci_model_cache.py's cache-key coverage: this describes a served backend's
    # own configuration, not a file the seeded *weights* cache depends on.

    raw = Settings().component_config("embedder", "ollama")
    tokenizer = raw.get("tokenizer")
    revision = raw.get("tokenizer_revision")
    return (
        tokenizer if isinstance(tokenizer, str) else "",
        revision if isinstance(revision, str) else "",
    )


def for_ollama(tokenizer: str, tokenizer_revision: str) -> int:
    """Fetch the tokenizer ``manicule-ollama`` loads, and nothing else.

    There is no canonical model repository to read a declaration from here — a served model's
    metadata comes from the Ollama server, measured at run time, not from a Hugging Face
    repository this script could fetch a card from. Fetching ``CARD_FILES`` from the model name
    the way :func:`for_backend` does would be reading a different checkpoint's declaration and
    attributing it to this one, which is exactly what
    :data:`manicule_ollama.served.TOKENIZER_FILES`'s own comment explains. What this backend
    needs is exactly the tokenizer it was told to trust, and that is all this function fetches.

    Returns the process exit code: ``0`` once the tokenizer is fetched or already a local
    directory, ``1`` when nothing names a usable vocabulary.
    """
    from manicule_ollama.config import OllamaEmbedderConfig  # noqa: PLC0415 - an optional install
    from manicule_ollama.served import TOKENIZER_FILES, tokenizer_identity  # noqa: PLC0415
    from pydantic import ValidationError  # noqa: PLC0415 - kept out of import time

    # Deferred for the same reason `Settings` is in `configured_ollama_tokenizer`: kept out of
    # import time, and out of tests/test_ci_model_cache.py's cache-key coverage, since neither
    # describes a fetch input the seeded *weights* cache depends on.
    from manicule.core.errors import ConfigError  # noqa: PLC0415

    if not tokenizer:
        print(f"error: {NO_OLLAMA_TOKENIZER_MSG}", file=sys.stderr)
        return 1

    try:
        config = OllamaEmbedderConfig(tokenizer=tokenizer, tokenizer_revision=tokenizer_revision)
        # `tokenizer_identity` is the one place that already refuses an unpinned repository id
        # and a `tokenizer_revision` claimed against a local directory, with the reasoning
        # `served.py` gives for both. Calling it here rather than re-deriving the same refusal
        # is what keeps this script's rules and the backend's rules from drifting apart the day
        # one of them changes.
        identity = tokenizer_identity(config)
    except (ValidationError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if Path(tokenizer).expanduser().is_dir():
        print(f"tokenizer {tokenizer!r} is a local directory ({identity}); nothing to fetch.")
        return 0

    print(f"ollama tokenizer {tokenizer!r}:")
    fetch(tokenizer, list(TOKENIZER_FILES), tokenizer_revision)
    print(f"fetched: {tokenizer}@{tokenizer_revision} (ollama tokenizer)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help=f"also fetch {FULL_MODEL}")
    parser.add_argument("--mlx", action="store_true", help="also fetch the MLX conversions")
    parser.add_argument(
        "--backend",
        choices=("mlx", "onnx", "ollama"),
        default=None,
        help="Fetch what one backend loads for the configured model, and nothing else. The "
        "operator's form: it takes the wait a first `manicule index` would otherwise meet, "
        "without the parity weights the test suite wants. `mlx`/`onnx` fetch --model's card "
        "and one runtime's weights; `ollama` has no card or weights to fetch here and instead "
        "fetches the configured tokenizer — see `--tokenizer`.",
    )
    parser.add_argument(
        "--model",
        default=FULL_MODEL,
        help=f"Which model --backend mlx/onnx fetches. Defaults to {FULL_MODEL}, manicule's "
        "own. Unused by --backend ollama, which names no model repository: the served model is "
        "identified on the Ollama server, not fetched here.",
    )
    parser.add_argument(
        "--tokenizer",
        default="",
        help="Repository id or local directory holding the tokenizer.json a served ollama "
        "model needs. Only meaningful with --backend ollama. This is what an image build "
        "supplies, since it has no config file yet to read `tokenizer` from; omit it to read "
        '`[plugins.config."embedder.ollama"].tokenizer` instead, the way the running backend '
        "does.",
    )
    parser.add_argument(
        "--tokenizer-revision",
        default="",
        help="Exact 40-character commit for a --tokenizer repository id. Required with one, "
        "refused for a local directory, and read from the same configuration entry's "
        "`tokenizer_revision` when --tokenizer is not given either.",
    )
    arguments = parser.parse_args(argv)

    if arguments.backend == "ollama":
        tokenizer, tokenizer_revision = arguments.tokenizer, arguments.tokenizer_revision
        if not tokenizer:
            tokenizer, tokenizer_revision = configured_ollama_tokenizer()
        return for_ollama(tokenizer, tokenizer_revision)

    if arguments.backend is not None:
        for_backend(arguments.model, arguments.backend)
        print(f"fetched: {arguments.model} ({arguments.backend})")
        return 0

    models = [PARITY_MODEL, *([FULL_MODEL] if arguments.full else [])]
    for model_id in models:
        prefetch(model_id, mlx=arguments.mlx)
    print(f"fetched: {', '.join(models)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
