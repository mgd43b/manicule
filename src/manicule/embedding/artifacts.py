"""Which bytes a backend actually executes, and the ones it must refuse.

A backend rarely runs the canonical repository's own weight files. ``BAAI/bge-m3`` publishes
``pytorch_model.bin`` and no safetensors, so MLX cannot load it at all; the weights that run
on Apple hardware come from a community conversion. Which conversion is not a detail:

=========================================  ===================================
``mlx-community/bge-m3-mlx-…`` variant     cosine to the same model in fp16
=========================================  ===================================
``fp16``                                   1.0000
``8bit``                                   0.9996 - 0.9998
``4bit``                                   0.9249 - 0.9694
=========================================  ===================================

Measured, on CLS-pooled normalised vectors. A quantised conversion is a **different vector
space wearing the same name** — and since ``backend`` and ``weights_ref`` are excluded from
:class:`~manicule.core.embedding.EmbedFingerprint` identity, nothing downstream would notice
one being mixed into an index built by another. So quantisation is refused here, at load, by
the one component in a position to see it.

That is the Apple-hardware principle drawn precisely: use Metal, use fp16 storage, use
whatever runs fastest — and stop at anything that moves the vectors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast

from pydantic import BaseModel, ConfigDict, Field

from manicule.core.errors import ConfigError
from manicule.embedding.cards import read_json_if_present

MLX_WEIGHTS: Final[dict[str, str]] = {
    "BAAI/bge-m3": "mlx-community/bge-m3-mlx-fp16",
}
"""Canonical model to the unquantised MLX conversion that runs it.

Only for models whose own repository MLX cannot load. A model publishing safetensors is
loaded from its own repository, which is the case that needs no table and no trust.
"""

ONNX_SUBDIR: Final = "onnx"
"""Where a Sentence-Transformers repository conventionally puts its export."""

_MLX_WEIGHT_PATTERNS: Final[tuple[str, ...]] = ("*.safetensors", "*.json")
_ONNX_PATTERNS: Final[tuple[str, ...]] = (f"{ONNX_SUBDIR}/*", "*.json")

_QUANTISATION_KEYS: Final[tuple[str, ...]] = ("quantization", "quantization_config")


class WeightsRef(BaseModel):
    """The artefact a backend loaded, recorded so a vector can be traced to its bytes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repo: str = Field(min_length=1, description="Repository id, or a local path.")
    path: Path = Field(description="Local directory the weights were read from.")

    def describe(self) -> str:
        """What goes into :attr:`~manicule.core.embedding.EmbedFingerprint.weights_ref`."""
        return self.repo


def mlx_repo(model_id: str, *, override: str = "") -> str:
    """Which repository MLX will load for ``model_id``, without touching the network.

    Separate from :func:`mlx_weights` because the fingerprint has to exist at construction —
    the chunker takes the embedder as a construction dependency and reads its sequence limit
    — while several gigabytes of weights must not be fetched until
    :meth:`~manicule.core.lifecycle.SupportsSetup.setup`.
    """
    return override or MLX_WEIGHTS.get(model_id, model_id)


def onnx_repo(model_id: str, *, override: str = "") -> str:
    """Which repository the ONNX export comes from. Pure, for the same reason."""
    return override or model_id


def mlx_weights(model_id: str, *, override: str = "") -> WeightsRef:
    """Resolve, download and vet the MLX weights for ``model_id``.

    Args:
        model_id: The canonical model.
        override: An explicit artefact from configuration, which wins over the table.

    Raises:
        ConfigError: No artefact is known, the artefact ships no safetensors, or it is
            quantised.
    """
    repo = mlx_repo(model_id, override=override)
    path = _materialise(repo, _MLX_WEIGHT_PATTERNS)

    if not any(path.glob("*.safetensors")):
        known = MLX_WEIGHTS.get(model_id)
        remedy = (
            f"Set `weights` under this embedder's configuration to a conversion that does — "
            f"{known!r} for this model."
            if known
            else "Set `weights` under this embedder's configuration to an unquantised MLX "
            "conversion of it, or run the onnx backend, which uses the repository's own export."
        )
        msg = f"{repo} ships no safetensors, which is the only weight format MLX reads. {remedy}"
        raise ConfigError(msg)

    _refuse_quantised(repo, path)
    return WeightsRef(repo=repo, path=path)


def onnx_weights(model_id: str, *, override: str = "") -> tuple[WeightsRef, Path]:
    """Resolve and download the ONNX export for ``model_id``, and the graph file within it.

    Returns:
        The artefact, and the ``.onnx`` file to open. External-data sidecars sit beside it and
        are found by onnxruntime relative to that path, which is why the whole subdirectory is
        fetched rather than the one file: ``bge-m3``'s graph is 725 KB of structure pointing at
        2.3 GB of weights in ``model.onnx_data``, and opening the first without the second
        loads a model with no parameters in it.

    Raises:
        ConfigError: The repository publishes no ONNX export, or more than one and none named
            conventionally.
    """
    repo = onnx_repo(model_id, override=override)
    path = _materialise(repo, _ONNX_PATTERNS)
    return WeightsRef(repo=repo, path=path), _graph_file(repo, path)


def _graph_file(repo: str, path: Path) -> Path:
    conventional = path / ONNX_SUBDIR / "model.onnx"
    if conventional.is_file():
        return conventional

    candidates = sorted(path.glob(f"{ONNX_SUBDIR}/*.onnx")) or sorted(path.glob("*.onnx"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        msg = (
            f"{repo} publishes no ONNX export. The onnx backend is the portable runtime and "
            f"the one the MLX backend is checked against, so a model without an export can "
            f"only be run on Apple hardware; set `weights` under this embedder's "
            f"configuration to a repository that publishes one."
        )
        raise ConfigError(msg)
    names = ", ".join(candidate.name for candidate in candidates)
    msg = (
        f"{repo} publishes {len(candidates)} ONNX graphs ({names}) and none called "
        f"model.onnx. They are usually different quantisations of one model, which do not "
        f"produce the same vectors; name the one you mean with `weights`."
    )
    raise ConfigError(msg)


def _refuse_quantised(repo: str, path: Path) -> None:
    """Refuse weights whose precision was reduced.

    Read from the artefact's own ``config.json``, which is where ``mlx_lm``'s converter records
    it and where ``mlx-embeddings`` reads it back to rebuild the quantised layers.
    """
    config = read_json_if_present(path / "config.json")
    for key in _QUANTISATION_KEYS:
        declared = config.get(key)
        if not isinstance(declared, dict):
            continue
        bits = cast("dict[str, object]", declared).get("bits", "an unstated number of")
        msg = (
            f"{repo} is quantised to {bits} bits. Quantisation changes the vectors — 4-bit "
            f"bge-m3 sits at cosine 0.92-0.97 to the same model in fp16, which is a different "
            f"space, not a rounding error. The runtime is excluded from embedding fingerprint "
            f"identity precisely because it is not supposed to change the output, so nothing "
            f"downstream would catch this mixed into an index. Point `weights` at an "
            f"unquantised conversion."
        )
        raise ConfigError(msg)


def _materialise(repo: str, patterns: tuple[str, ...]) -> Path:
    """The artefact on local disk, downloaded if it is a repository id."""
    local = Path(repo).expanduser()
    if local.is_dir():
        return local

    # Deferred: huggingface-hub is only needed when the artefact is not already on disk.
    from manicule.embedding.runtimes.hub import snapshot  # noqa: PLC0415

    return snapshot(repo, patterns)


__all__ = [
    "MLX_WEIGHTS",
    "ONNX_SUBDIR",
    "WeightsRef",
    "mlx_repo",
    "mlx_weights",
    "onnx_repo",
    "onnx_weights",
]
