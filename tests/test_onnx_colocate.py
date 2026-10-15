"""An ONNX graph and its external weights have to be openable from a hub cache of either layout.

``huggingface-hub`` 2.x keeps one shared blob store sharded by hash prefix, so a snapshot's
``model.onnx`` and ``model.onnx_data`` symlinks resolve into *different* directories, and
onnxruntime refuses a model whose external data is not beside it ("External data path escapes
model directory"). Version 1.x kept a repository's blobs together, so the check never fired. The
container smoke test caught it on a dependency bump; this pins it where it can be seen without
building an image.

**The model is real and tiny, and hand-encoded.** ``onnx`` is not a dependency, so the graph
``Y = X + W`` is written out as protobuf bytes, with ``W`` stored as external data in a sidecar.
Opening it with the installed onnxruntime is the judge — a fake directory tree would prove only
that the helper moves files, not that onnxruntime accepts the result.

**The first test is the control.** If onnxruntime ever stops refusing the sharded layout, the
helper is dead code and this test says so; without it the others could pass for a reason that has
nothing to do with the fix.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from manicule.embedding.runtimes.colocate import colocated

if TYPE_CHECKING:
    from pathlib import Path

ort = pytest.importorskip("onnxruntime")


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        low = value & 0x7F
        value >>= 7
        out.append(low | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _number(field: int, value: int) -> bytes:
    return _varint(field << 3) + _varint(value)


def _message(field: int, payload: bytes) -> bytes:
    return _varint(field << 3 | 2) + _varint(len(payload)) + payload


def _text(field: int, value: str) -> bytes:
    return _message(field, value.encode())


def _scalar_info(name: str) -> bytes:
    """A ``float[1]`` value: name, then tensor type, element type 1 (float), one dimension of 1."""
    dimension = _message(1, _number(1, 1))
    tensor = _number(1, 1) + _message(2, dimension)
    return _text(1, name) + _message(2, _message(1, tensor))


def _model(sidecar: str) -> bytes:
    """``Y = X + W`` with ``W`` held externally at ``sidecar``, offset 0, four bytes."""

    def entry(key: str, value: str) -> bytes:
        return _message(13, _text(1, key) + _text(2, value))

    weight = (
        _number(1, 1)
        + _number(2, 1)
        + _text(8, "W")
        + entry("location", sidecar)
        + entry("offset", "0")
        + entry("length", "4")
        + _number(14, 1)
    )
    node = _text(1, "X") + _text(1, "W") + _text(2, "Y") + _text(4, "Add")
    graph = (
        _message(1, node)
        + _text(2, "g")
        + _message(5, weight)
        + _message(11, _scalar_info("X"))
        + _message(12, _scalar_info("Y"))
    )
    return _number(1, 7) + _message(8, _text(1, "") + _number(2, 13)) + _message(7, graph)


SIDECAR = "model.onnx_data"
WEIGHT = struct.pack("<f", 1.0)


def _snapshot(root: Path, *, sharded: bool) -> Path:
    """A snapshot's ``onnx/`` directory, laid out as either library version lays it out."""
    blobs = root / "hub" / "blobs"
    here = root / "hub" / "models--org--repo" / "snapshots" / "rev" / "onnx"
    here.mkdir(parents=True)
    # The shard is the first two characters of the hash, so choosing hashes that differ there is
    # what puts the two files in different directories — and equal hashes' prefix keeps them
    # together, which is the flat case.
    placements = {
        "model.onnx": ("21", "217e" + "0" * 60, _model(SIDECAR)),
        SIDECAR: ("8b" if sharded else "21", "8b3c" + "0" * 60, WEIGHT),
        "config.json": ("cd" if sharded else "21", "cdfc" + "0" * 60, b"{}"),
    }
    for name, (shard, digest, content) in placements.items():
        directory = blobs / shard
        directory.mkdir(parents=True, exist_ok=True)
        (directory / digest).write_bytes(content)
        (here / name).symlink_to(directory / digest)
    return here / "model.onnx"


def _run(path: Path) -> float:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (out,) = session.run(["Y"], {"X": np.array([2.0], dtype=np.float32)})
    return float(out[0])


def test_onnxruntime_refuses_the_sharded_layout_which_is_why_any_of_this_exists(
    tmp_path: Path,
) -> None:
    """The control. If this stops raising, ``colocated`` has nothing left to do."""
    graph = _snapshot(tmp_path, sharded=True)

    with pytest.raises(Exception, match="escapes model directory"):
        _run(graph)


def test_the_sharded_layout_opens_once_its_weights_are_beside_the_graph(tmp_path: Path) -> None:
    graph = _snapshot(tmp_path, sharded=True)

    staged = colocated(graph, tmp_path / "staging")

    assert _run(staged) == 3.0, "X + W with W = 1.0 read from the sidecar"
    assert (staged.parent / SIDECAR).is_file()


def test_a_layout_that_already_works_is_left_alone(tmp_path: Path) -> None:
    """Flat, as ``huggingface-hub`` 1.x lays it out: nothing is staged, nothing is created."""
    graph = _snapshot(tmp_path, sharded=False)
    staging = tmp_path / "staging"

    assert colocated(graph, staging) == graph
    assert not staging.exists()
    assert _run(graph) == 3.0


def test_the_same_weights_are_staged_once_and_reused(tmp_path: Path) -> None:
    graph = _snapshot(tmp_path, sharded=True)
    staging = tmp_path / "staging"

    first = colocated(graph, staging)
    second = colocated(graph, staging)

    assert first == second
    assert [entry.name for entry in staging.iterdir()] == [first.parent.name], (
        "one directory, and no half-built one left beside it"
    )


def test_a_staged_graph_shares_the_weights_instead_of_copying_them(tmp_path: Path) -> None:
    """A hard link: the 2.3 GB file exists once, which is the reason to prefer it to a copy."""
    graph = _snapshot(tmp_path, sharded=True)

    staged = colocated(graph, tmp_path / "staging")

    assert (staged.parent / SIDECAR).stat().st_ino == (graph.parent / SIDECAR).stat().st_ino


def test_it_copies_where_a_link_cannot_be_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A weights layer and a cache volume are different filesystems, and a link cannot cross one."""

    def refuse(self: Path, target: Path) -> None:
        del self, target
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr("pathlib.Path.hardlink_to", refuse)
    graph = _snapshot(tmp_path, sharded=True)

    staged = colocated(graph, tmp_path / "staging")

    sidecar = staged.parent / SIDECAR
    assert sidecar.read_bytes() == WEIGHT
    assert sidecar.stat().st_ino != (graph.parent / SIDECAR).stat().st_ino
    assert _run(staged) == 3.0


def test_a_different_revision_of_the_weights_lands_somewhere_else(tmp_path: Path) -> None:
    """Named by what the files are, so a changed model can never be served the old copy."""
    staging = tmp_path / "staging"
    one = colocated(_snapshot(tmp_path / "a", sharded=True), staging)
    other_root = tmp_path / "b"
    graph = _snapshot(other_root, sharded=True)
    (other_root / "hub" / "blobs" / "8b" / ("8b3c" + "0" * 60)).rename(
        other_root / "hub" / "blobs" / "8b" / ("8b3d" + "0" * 60)
    )
    (graph.parent / SIDECAR).unlink()
    (graph.parent / SIDECAR).symlink_to(other_root / "hub" / "blobs" / "8b" / ("8b3d" + "0" * 60))

    two = colocated(graph, staging)

    assert one.parent != two.parent


def test_an_unwritable_staging_directory_is_an_error_not_a_silent_fallback(
    tmp_path: Path,
) -> None:
    """Falling back to the symlinked path would reproduce the very failure being avoided."""
    graph = _snapshot(tmp_path, sharded=True)
    blocker = tmp_path / "staging"
    blocker.write_text("a file where a directory has to go")

    with pytest.raises(OSError, match=r"."):
        colocated(graph, blocker)


def _embedder_staging_into(staging: Path):  # noqa: ANN202 - the embedder, built without a model
    """An ``OnnxEmbedder`` that has not resolved any weights, for the one method under test.

    Its constructor reads the model hub for a card and a revision, which is what these tests are
    not about; ``_openable`` needs only the directory it was told to stage into.
    """
    from manicule.embedding.runtimes.onnx_backend import OnnxEmbedder  # noqa: PLC0415

    embedder = OnnxEmbedder.__new__(OnnxEmbedder)
    embedder._staging_dir = staging  # pyright: ignore[reportPrivateUsage]
    return embedder


def test_the_embedder_opens_a_sharded_snapshot_through_its_staging_directory(
    tmp_path: Path,
) -> None:
    graph = _snapshot(tmp_path, sharded=True)
    embedder = _embedder_staging_into(tmp_path / "staging")

    opened = embedder._openable(graph)  # pyright: ignore[reportPrivateUsage]

    assert tmp_path / "staging" in opened.parents
    assert _run(opened) == 3.0


def test_an_unwritable_cache_is_reported_as_configuration_that_names_the_directory(
    tmp_path: Path,
) -> None:
    from manicule.core.errors import ConfigError  # noqa: PLC0415

    graph = _snapshot(tmp_path, sharded=True)
    blocker = tmp_path / "staging"
    blocker.write_text("a file where a directory has to go")
    embedder = _embedder_staging_into(blocker)

    with pytest.raises(ConfigError, match=r"could not be placed together under .*staging"):
        embedder._openable(graph)  # pyright: ignore[reportPrivateUsage]


def test_a_staged_directory_missing_its_weights_is_rebuilt_not_reused(tmp_path: Path) -> None:
    graph = _snapshot(tmp_path, sharded=True)
    staging = tmp_path / "staging"
    first = colocated(graph, staging)
    (first.parent / SIDECAR).unlink()

    second = colocated(graph, staging)

    assert second == first
    assert _run(second) == 3.0


def test_load_opens_the_staged_graph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``_load`` is the wiring: reverting it to ``str(graph)`` fails here, not elsewhere."""
    from types import SimpleNamespace  # noqa: PLC0415

    from manicule.embedding.runtimes import onnx_backend  # noqa: PLC0415

    graph = _snapshot(tmp_path, sharded=True)
    weights = SimpleNamespace(describe=lambda: "weights")

    def resolved(artifact: object) -> tuple[SimpleNamespace, Path]:
        del artifact
        return weights, graph

    monkeypatch.setattr(onnx_backend, "onnx_weights", resolved)
    embedder = _embedder_staging_into(tmp_path / "staging")
    embedder.fingerprint = SimpleNamespace(weights_ref="weights")  # pyright: ignore[reportAttributeAccessIssue]
    embedder._artifact = object()  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]

    embedder._load()  # pyright: ignore[reportPrivateUsage]

    session: Any = embedder._session  # pyright: ignore[reportPrivateUsage]
    assert session is not None
    (out,) = session.run(["Y"], {"X": np.array([2.0], dtype=np.float32)})
    assert float(out[0]) == 3.0
