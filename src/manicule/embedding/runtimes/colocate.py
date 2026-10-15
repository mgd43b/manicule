"""Putting an ONNX graph and its external weights in one directory, so onnxruntime will open it.

**The failure this exists for.** ``bge-m3``'s export is a 725 KB graph (``model.onnx``) whose
parameters are 2.3 GB in a sidecar, ``model.onnx_data``, that the graph names by a path relative
to itself. onnxruntime resolves symlinks on both files and then requires the sidecar to live in
the same directory as the graph it belongs to — "External data path escapes model directory" —
and refuses to open the model otherwise.

``huggingface-hub`` 1.x kept every blob of a repository together in one ``blobs/`` directory, so
a snapshot's symlinks resolved side by side and the check passed without anybody knowing it was
there. ``huggingface-hub`` 2.x keeps one shared store sharded by hash prefix
(``hub/blobs/21/…``, ``hub/blobs/8b/…``), so the same two symlinks resolve into different
directories and the model cannot be opened at all. Nothing in manicule had changed; the cache it
reads from had.

**The fix is to make the directory the check expects**, rather than to move the problem by
pinning a version: the cache layout is the hub library's to change, and a runtime that works only
under one layout breaks for whoever upgrades it next.

**Hard links, then a copy.** A hard link makes the graph and its weights siblings without a second
copy of 2.3 GB, and works wherever the staging directory and the cache share a filesystem. Where
they do not — an image whose weights are baked into a read-only layer and whose cache is a volume
— the files are copied once, into a directory that is kept, so the cost is paid on the first start
and never again.

**A layout that already works is left alone.** When every file beside the graph resolves into the
graph's own directory, nothing is staged and the original path is returned, so a flat cache costs
one directory listing.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import uuid
from pathlib import Path


def colocated(graph: Path, staging: Path, *, identity: str = "") -> Path:
    """``graph``, or a path to the same graph with every file that belongs to it beside it.

    Args:
        graph: The ``.onnx`` file, as the hub cache lays it out — usually a symlink.
        staging: A writable directory to build the co-located copy under.
        identity: What the artifact is, as the caller verified it
            (:attr:`~manicule.embedding.artifacts.WeightsRef.identity`). **Part of the name of the
            staged directory**, because a resolved filename is a content hash only in a hub cache:
            a local export may call its graph and its sidecar the same thing as a different
            export does, and keying on names alone would hand the second one the first one's
            weights and attribute the vectors to the wrong artifact.

    Returns:
        ``graph`` itself when every file under its directory already resolves inside that
        directory; otherwise the graph inside a directory under ``staging`` that holds all of
        them, **at the same relative paths** — an external-data location may name a
        subdirectory (``data/weights.bin``), and flattening it would leave the graph pointing
        at nothing.

    Raises:
        OSError: ``staging`` cannot be written to.
    """
    files = _files(graph.parent)
    home = graph.resolve().parent
    if all(path.resolve().is_relative_to(home) for path in files):
        return graph

    key = _key(graph.parent, files, identity)
    target = staging / key
    staged = target / graph.name
    if _complete(target, graph.parent, files):
        return staged
    # Present and missing a file: something removed part of it, or an earlier version staged
    # fewer. Reusing it would hand onnxruntime a graph with no weights beside it.
    if target.exists() and not _discard(target, graph.parent, files):
        return staged

    staging.mkdir(parents=True, exist_ok=True)
    # Built beside the target and renamed into place, so a process that stops halfway leaves a
    # dot-directory nobody reads rather than a half-copied one that a later start trusts.
    work = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=staging))
    try:
        for path in files:
            destination = work / path.relative_to(graph.parent)
            destination.parent.mkdir(parents=True, exist_ok=True)
            _link_or_copy(path.resolve(), destination)
        try:
            work.rename(target)
        except OSError:
            # Another process finished the same directory first, which is as good as ours.
            if not _complete(target, graph.parent, files):
                raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return staged


def _discard(target: Path, root: Path, files: list[Path]) -> bool:
    """Move an incomplete staged directory out of the way; ``False`` if it has just been completed.

    **Renamed, never deleted in place.** Two cold starts can both see the same damaged directory.
    One rebuilds and publishes it; deleting by name from the other would then remove the directory
    the first is about to open. The completeness check is repeated immediately before the move,
    and the move is to a unique name that is deleted afterwards, so a directory another process
    published in the meantime is left alone. The window that remains is the few instructions
    between that check and the rename, on a directory that was already damaged.
    """
    if _complete(target, root, files):
        return False
    aside = target.with_name(f".{target.name}.stale.{uuid.uuid4().hex}")
    try:
        target.rename(aside)
    except OSError:
        # Gone already, or replaced by a process that finished first. Either way it is not ours.
        return True
    shutil.rmtree(aside, ignore_errors=True)
    return True


def _files(root: Path) -> list[Path]:
    """Every regular file under ``root``, nested ones included, in a stable order."""
    return sorted(path for path in root.rglob("*") if path.is_file())


def _key(root: Path, files: list[Path], identity: str) -> str:
    """A name for one set of files: the caller's identity, and each file's path, size and target.

    The size and resolved name are the cheap discriminators available without reading 2.3 GB;
    in a hub cache the name is the content hash already, and ``identity`` carries the rest.
    """
    parts = [identity] + [
        f"{path.relative_to(root)}={path.resolve().name}:{path.resolve().stat().st_size}"
        for path in files
    ]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:32]


def _complete(target: Path, root: Path, files: list[Path]) -> bool:
    """Whether ``target`` holds every file that was under the graph's directory."""
    return all((target / path.relative_to(root)).is_file() for path in files)


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        destination.hardlink_to(source)
    except OSError:
        # A different filesystem, or one that does not support hard links.
        shutil.copyfile(source, destination)


__all__ = ["colocated"]
