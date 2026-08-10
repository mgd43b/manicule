"""Fetching model files, as one typed function.

``huggingface_hub`` is the only route to a model repository here, and it is confined to this
module so that the rest of the embedding stack takes a :class:`~pathlib.Path` and has no
opinion about where it came from — which is also what lets a local directory be passed
anywhere a repository id is accepted.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def snapshot(repo: str, patterns: Sequence[str], revision: str | None = None) -> Path:
    """Download ``patterns`` from ``repo`` and return the local directory holding them.

    Args:
        repo: A repository id. A local directory is the caller's business, not this module's.
        patterns: Glob patterns to fetch. Always narrowed: a model repository holds several
            formats of the same weights, and fetching all of them costs gigabytes to use one.
        revision: A commit, branch or tag.

    Returns:
        The snapshot directory. Its name is the resolved commit when the download went through
        the shared cache, which is how a revision is pinned without a second network call and
        without needing the network at all on the next run.
    """
    from huggingface_hub import snapshot_download  # noqa: PLC0415 - kept out of import time

    return Path(snapshot_download(repo, revision=revision, allow_patterns=list(patterns)))


__all__ = ["snapshot"]
