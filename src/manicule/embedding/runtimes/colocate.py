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
from pathlib import Path


def colocated(graph: Path, staging: Path) -> Path:
    """``graph``, or a path to the same graph with its sibling files beside it.

    Args:
        graph: The ``.onnx`` file, as the hub cache lays it out — usually a symlink.
        staging: A writable directory to build the co-located copy under.

    Returns:
        ``graph`` itself when the files beside it already resolve into one directory; otherwise
        the graph inside a directory under ``staging`` that holds every regular file that was
        beside it. The directory is named by what the files *are* (their resolved blob names,
        which are content hashes), so the same weights always land in the same place and a
        different revision lands somewhere else.

    Raises:
        OSError: ``staging`` cannot be written to.
    """
    siblings = sorted(path for path in graph.parent.iterdir() if path.is_file())
    home = graph.resolve().parent
    if all(path.resolve().parent == home for path in siblings):
        return graph

    identity = "\n".join(f"{path.name}={path.resolve().name}" for path in siblings)
    key = hashlib.sha256(identity.encode()).hexdigest()[:32]
    target = staging / key
    staged = target / graph.name
    if staged.is_file():
        return staged

    staging.mkdir(parents=True, exist_ok=True)
    # Built beside the target and renamed into place, so a process that stops halfway leaves a
    # dot-directory nobody reads rather than a half-copied one that a later start trusts.
    work = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=staging))
    try:
        for path in siblings:
            _link_or_copy(path.resolve(), work / path.name)
        try:
            work.rename(target)
        except OSError:
            # Another process finished the same directory first, which is as good as ours.
            if not staged.is_file():
                raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return staged


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        destination.hardlink_to(source)
    except OSError:
        # A different filesystem, or one that does not support hard links.
        shutil.copyfile(source, destination)


__all__ = ["colocated"]
