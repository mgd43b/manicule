"""Fetching model files, as one typed function.

``huggingface_hub`` is the only route to a model repository here, and it is confined to this
module so that the rest of the embedding stack takes a :class:`~pathlib.Path` and has no
opinion about where it came from — which is also what lets a local directory be passed
anywhere a repository id is accepted.

**Being the only route is also what makes the failure sayable.** An embedder is built and set
up on the path that answers a query, so a model this machine has never seen is a download
happening *inside a search* — and when there is no route to the hub, the error arrives from
inside a library the operator did not choose, naming a repository and a cache path and nothing
they can act on. There is no equivalent of :mod:`manicule.vocabularies` here and there should
not be: a 2.3 GB ONNX export is not an artifact anyone carries in a manifest, and manicule
already ships a pre-seed for it. What was missing was the sentence saying so at the moment it
matters, and that is what :func:`snapshot` adds.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from manicule.core.errors import ManiculeError

OFFLINE_ENV = "HF_HUB_OFFLINE"
"""The hub's own switch for refusing to reach the network.

Named in the failure rather than set by manicule. An install that has pre-seeded its models
should set it — it turns a first search on a machine with a slow route into an immediate,
readable refusal instead of a wait that looks like a hang — but choosing that is a
deployment's business, and the container already does.
"""


class ModelUnavailableError(ManiculeError):
    """A model's files are not on this machine and could not be fetched.

    Raised where the fetch happens rather than left to the hub's own exception, because the
    hub's is written for somebody debugging a download and this one is read by somebody whose
    *search* did not answer. It carries the repository, what was being fetched and the
    pre-seed that supplies it.
    """


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

    Raises:
        ModelUnavailableError: The files are neither cached nor reachable. The hub's own
            message is kept — it distinguishes a gated repository from a typo from an outage,
            and discarding it would trade one unhelpful error for another — and the remedy is
            added to it.
    """
    from huggingface_hub import snapshot_download  # noqa: PLC0415 - kept out of import time

    try:
        return Path(snapshot_download(repo, revision=revision, allow_patterns=list(patterns)))
    except Exception as exc:
        # Deliberately broad, for the reason the vocabulary pre-seed gives: the ways this
        # fails span the hub's own hierarchy, `requests`, and the filesystem, with no common
        # base. Enumerating them means the one that was left out escapes as a library
        # traceback in the middle of a query, which is the thing being fixed.
        raise ModelUnavailableError(_unavailable(repo, patterns, exc)) from exc


def _unavailable(repo: str, patterns: Sequence[str], exc: Exception) -> str:
    """What a query says when the model it needs was never put on this machine."""
    return (
        f"the model files for {repo!r} ({', '.join(patterns)}) are not on this machine and "
        f"could not be fetched: {exc}. manicule does not carry model weights; they are "
        f"pre-seeded, which on most installs happens during the first `manicule index`. "
        f"Seed them deliberately with `uv run tools/prefetch_embedding_models.py`, point "
        f"`embedding.model` (or a backend's `weights`) at a local directory holding them, or "
        f"see docs/deployment.md §5.1 for a host with no network. {OFFLINE_ENV} being set "
        f"means the hub was never allowed to look."
    )


__all__ = ["OFFLINE_ENV", "ModelUnavailableError", "snapshot"]
