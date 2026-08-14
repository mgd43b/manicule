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

Measured, on CLS-pooled normalized vectors. A quantized conversion is a **different vector
space wearing the same name** — and since ``backend`` and ``weights_ref`` are excluded from
:class:`~manicule.core.embedding.EmbedFingerprint` identity, nothing downstream would notice
one being mixed into an index built by another. So quantization is refused here, at load, by
the one component in a position to see it.

That is the Apple-hardware principle drawn precisely: use Metal, use fp16 storage, use
whatever runs fastest — and stop at anything that moves the vectors.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from pydantic import BaseModel, ConfigDict, Field

from manicule.core.errors import ConfigError
from manicule.embedding.cards import read_json_if_present

MLX_WEIGHTS: Final[dict[str, str]] = {
    "BAAI/bge-m3": "mlx-community/bge-m3-mlx-fp16",
}
"""Canonical model to the unquantized MLX conversion that runs it.

Only for models whose own repository MLX cannot load. A model publishing safetensors is
loaded from its own repository, which is the case that needs no table and no trust.
"""

ONNX_SUBDIR: Final = "onnx"
"""Where a Sentence-Transformers repository conventionally puts its export."""

_MLX_WEIGHT_PATTERNS: Final[tuple[str, ...]] = ("*.safetensors", "*.json")
_ONNX_PATTERNS: Final[tuple[str, ...]] = (f"{ONNX_SUBDIR}/*", "*.json")

_QUANTIZATION_KEYS: Final[tuple[str, ...]] = ("quantization", "quantization_config")

_QUANTIZED_GRAPH_MARKERS: Final[tuple[str, ...]] = (
    "quantized",
    "quantized",
    "int8",
    "uint8",
    "qint8",
    "quint8",
)
"""Substrings that mark an ONNX graph as reduced-precision.

A name is weak evidence and it is the only evidence available: an ONNX graph carries no
declaration of its own precision the way an MLX conversion's ``config.json`` does. Weak
evidence is still worth acting on here, because the alternative is admitting a graph measured
at cosine 0.92-0.97 to the model it claims to be, into an index whose fingerprint says the
runtime cannot have changed the output.

``fp16`` is deliberately absent: half precision is what the MLX side already runs, and parity
holds across it.
"""


class WeightsRef(BaseModel):
    """The artifact a backend loaded, recorded so a vector can be traced to its bytes."""

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


APPROXIMATE_WEIGHT_BYTES: Final[Mapping[tuple[str, str], int]] = {
    ("mlx", "mlx-community/bge-m3-mlx-fp16"): 1_150_000_000,
    ("onnx", "BAAI/bge-m3"): 2_290_000_000,
}
"""Roughly what a first fetch moves, per ``(provider, repo)``, for the models manicule ships
a default for.

**Keyed by both, because the same repository id means two different downloads.** The ONNX
export lives in the model's own repository beside weights nothing here loads, and
:data:`_ONNX_PATTERNS` takes a slice of it; MLX loads a separate conversion. A table keyed by
repository alone would report the ONNX size for an MLX install of any model whose own
repository happens to carry safetensors.

**Approximate, recorded, and never guessed for anything absent.** These are the sizes the
Hugging Face tree API reports for the files the two pattern sets match, rounded down to the
figure a person would repeat. A model not in this table reports no size rather than an
estimate: "about 1.1 GB" that turns out to be 4 is worse than "size not known here", because
the first is what somebody plans an afternoon around.
"""


@dataclass(frozen=True, slots=True)
class WeightsPlan:
    """Which artifact a backend will load, and whether this machine already has it.

    Answered without the network, so that a wait measured in gigabytes can be announced by
    ``init`` and ``doctor`` before the first ingest meets it as an unexplained pause.
    """

    provider: str
    repo: str
    patterns: tuple[str, ...]
    present: bool
    approximate_bytes: int | None = None

    @property
    def size(self) -> str:
        """The download, as a person would say it, or a stated absence of one."""
        if self.approximate_bytes is None:
            return "an unrecorded amount"
        return f"about {self.approximate_bytes / 1_000_000_000:.1f} GB"


def planned_weights(provider: str, model_id: str, *, override: str = "") -> WeightsPlan:
    """What ``provider`` will load for ``model_id``, and whether it is already here.

    Args:
        provider: The configured embedder implementation, ``"mlx"`` or ``"onnx"``.
        model_id: The canonical model.
        override: An explicit artifact from configuration, which wins over the table.

    Returns:
        The plan. A provider this module knows no artifact route for — a third-party embedder
        registered by a plugin — reports the model id with no patterns and ``present=True``,
        because "manicule does not know how this one loads its weights" must not be rendered
        as "this machine is missing something".
    """
    routes: Mapping[str, tuple[str, tuple[str, ...]]] = {
        "mlx": (mlx_repo(model_id, override=override), _MLX_WEIGHT_PATTERNS),
        "onnx": (onnx_repo(model_id, override=override), _ONNX_PATTERNS),
    }
    chosen = routes.get(provider.strip().lower())
    if chosen is None:
        return WeightsPlan(provider=provider, repo=model_id, patterns=(), present=True)
    repo, patterns = chosen

    from manicule.embedding.runtimes.hub import is_cached  # noqa: PLC0415 - an embeddings extra

    return WeightsPlan(
        provider=provider,
        repo=repo,
        patterns=patterns,
        present=is_cached(repo, patterns),
        approximate_bytes=APPROXIMATE_WEIGHT_BYTES.get((provider.strip().lower(), repo)),
    )


def mlx_weights(model_id: str, *, override: str = "") -> WeightsRef:
    """Resolve, download and vet the MLX weights for ``model_id``.

    Args:
        model_id: The canonical model.
        override: An explicit artifact from configuration, which wins over the table.

    Raises:
        ConfigError: No artifact is known, the artifact ships no safetensors, or it is
            quantized.
    """
    repo = mlx_repo(model_id, override=override)
    path = _materialize(repo, _MLX_WEIGHT_PATTERNS)

    if not any(path.glob("*.safetensors")):
        known = MLX_WEIGHTS.get(model_id)
        remedy = (
            f"Set `weights` under this embedder's configuration to a conversion that does — "
            f"{known!r} for this model."
            if known
            else "Set `weights` under this embedder's configuration to an unquantized MLX "
            "conversion of it, or run the onnx backend, which uses the repository's own export."
        )
        msg = f"{repo} ships no safetensors, which is the only weight format MLX reads. {remedy}"
        raise ConfigError(msg)

    _refuse_quantized(repo, path)
    return WeightsRef(repo=repo, path=path)


def onnx_weights(model_id: str, *, override: str = "") -> tuple[WeightsRef, Path]:
    """Resolve and download the ONNX export for ``model_id``, and the graph file within it.

    Returns:
        The artifact, and the ``.onnx`` file to open. External-data sidecars sit beside it and
        are found by onnxruntime relative to that path, which is why the whole subdirectory is
        fetched rather than the one file: ``bge-m3``'s graph is 725 KB of structure pointing at
        2.3 GB of weights in ``model.onnx_data``, and opening the first without the second
        loads a model with no parameters in it.

    Raises:
        ConfigError: The repository publishes no ONNX export, or more than one and none named
            conventionally.
    """
    repo = onnx_repo(model_id, override=override)
    path = _materialize(repo, _ONNX_PATTERNS)
    return WeightsRef(repo=repo, path=path), _graph_file(repo, path)


def _graph_file(repo: str, path: Path) -> Path:
    conventional = path / ONNX_SUBDIR / "model.onnx"
    if conventional.is_file():
        return conventional

    candidates = sorted(path.glob(f"{ONNX_SUBDIR}/*.onnx")) or sorted(path.glob("*.onnx"))
    if len(candidates) == 1:
        return _refuse_quantized_graph(repo, candidates[0])
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
        f"model.onnx. They are usually different quantizations of one model, which do not "
        f"produce the same vectors; name the one you mean with `weights`."
    )
    raise ConfigError(msg)


def _refuse_quantized_graph(repo: str, graph: Path) -> Path:
    """Refuse an ONNX graph whose name says its precision was reduced.

    The asymmetry with the MLX path is worth stating: there, quantization is declared in the
    artifact's ``config.json`` and read. An ONNX graph declares nothing, so a repository
    publishing only ``model_quantized.onnx`` would otherwise be the *unambiguous* candidate and
    be loaded without a word — the one case the ambiguity refusal below cannot see.
    """
    lowered = graph.name.lower()
    marker = next((mark for mark in _QUANTIZED_GRAPH_MARKERS if mark in lowered), None)
    if marker is None:
        return graph
    msg = (
        f"{repo}'s only ONNX export is {graph.name}, whose name says it is quantized "
        f"({marker!r}). Quantization changes the vectors — measured at cosine 0.92-0.97 for a "
        f"4-bit conversion of the same model — while the runtime is excluded from embedding "
        f"fingerprint identity precisely because it is not supposed to. Point `weights` at a "
        f"repository publishing a full-precision export, or run the mlx backend."
    )
    raise ConfigError(msg)


def _refuse_quantized(repo: str, path: Path) -> None:
    """Refuse weights whose precision was reduced.

    Read from the artifact's own ``config.json``, which is where ``mlx_lm``'s converter records
    it and where ``mlx-embeddings`` reads it back to rebuild the quantized layers.
    """
    config = read_json_if_present(path / "config.json")
    for key in _QUANTIZATION_KEYS:
        declared = config.get(key)
        if not isinstance(declared, dict):
            continue
        bits = cast("dict[str, object]", declared).get("bits", "an unstated number of")
        msg = (
            f"{repo} is quantized to {bits} bits. Quantization changes the vectors — 4-bit "
            f"bge-m3 sits at cosine 0.92-0.97 to the same model in fp16, which is a different "
            f"space, not a rounding error. The runtime is excluded from embedding fingerprint "
            f"identity precisely because it is not supposed to change the output, so nothing "
            f"downstream would catch this mixed into an index. Point `weights` at an "
            f"unquantized conversion."
        )
        raise ConfigError(msg)


def _materialize(repo: str, patterns: tuple[str, ...]) -> Path:
    """The artifact on local disk, downloaded if it is a repository id."""
    local = Path(repo).expanduser()
    if local.is_dir():
        return local

    # Deferred: huggingface-hub is only needed when the artifact is not already on disk.
    from manicule.embedding.runtimes.hub import snapshot  # noqa: PLC0415

    return snapshot(repo, patterns)


__all__ = [
    "APPROXIMATE_WEIGHT_BYTES",
    "MLX_WEIGHTS",
    "ONNX_SUBDIR",
    "WeightsPlan",
    "WeightsRef",
    "mlx_repo",
    "mlx_weights",
    "onnx_repo",
    "onnx_weights",
    "planned_weights",
]
