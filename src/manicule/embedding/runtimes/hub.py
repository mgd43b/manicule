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

import contextlib
import os
from collections.abc import Generator, Sequence
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
        with _progress_worth_showing(repo, patterns, revision):
            return Path(snapshot_download(repo, revision=revision, allow_patterns=list(patterns)))
    except Exception as exc:
        # Deliberately broad, for the reason the vocabulary pre-seed gives: the ways this
        # fails span the hub's own hierarchy, `requests`, and the filesystem, with no common
        # base. Enumerating them means the one that was left out escapes as a library
        # traceback in the middle of a query, which is the thing being fixed.
        raise ModelUnavailableError(_unavailable(repo, patterns, exc)) from exc


PROGRESS_ENV = "HF_HUB_DISABLE_PROGRESS_BARS"
"""The hub's own switch for its progress bars.

Read, never written. **The variable is consulted at import time**, into a module constant, so
setting it after ``huggingface_hub`` has been imported changes nothing at all — which is the
first thing this function was written to do, and it silently did not work. It is read here so
that an installation which set it deliberately — a CI job capturing logs, a container — is
left alone rather than having its choice toggled underneath it.
"""


@contextlib.contextmanager
def _progress_worth_showing(
    repo: str, patterns: Sequence[str], revision: str | None
) -> Generator[None]:
    """Let the hub draw its progress bars only when there is progress to draw.

    ``snapshot_download`` prints ``Fetching N files`` whether or not it fetches anything, so
    every command that builds an embedder opened with two progress bars — on a warm machine,
    two bars that complete instantly and report a download that did not happen. Before every
    ``search``, for ever.

    **Silencing them outright would have been wrong**, and the reason is the case they were
    covering: a first ``index`` really does move over a gigabyte, and a bar is how somebody
    knows the process is alive rather than wedged. So the question is not "bars or no bars" but
    "is anything being downloaded", which :func:`is_cached` answers from disk. Cached, and the
    bars say nothing worth two lines; not cached, and they are the only thing on screen for a
    minute or more.

    Done through ``disable_progress_bars``/``enable_progress_bars`` rather than the
    environment: :data:`PROGRESS_ENV` is read once at import, so assigning it here would be a
    line that looks like it works and does nothing. An operator who set it is left alone.
    """
    # From `utils.tqdm` rather than the `utils` package: the re-export is not in the
    # package's `__all__`, so importing it from there is a private import a type checker is
    # right to object to.
    from huggingface_hub.utils.tqdm import (  # noqa: PLC0415 - kept out of import time
        are_progress_bars_disabled,
        disable_progress_bars,
        enable_progress_bars,
    )

    if (
        PROGRESS_ENV in os.environ
        or are_progress_bars_disabled()
        or not is_cached(repo, patterns, revision)
    ):
        yield
        return
    disable_progress_bars()
    try:
        yield
    finally:
        enable_progress_bars()


def is_cached(repo: str, patterns: Sequence[str], revision: str | None = None) -> bool:
    """Whether :func:`snapshot` would answer from disk alone. **Never touches the network.**

    The question ``doctor`` and ``init`` ask so that a gigabyte-scale download can be
    announced before it starts rather than met as a silent pause inside a first ingest. It has
    to be answerable offline, because the command asking it is a diagnostic: one that reached
    the hub to find out whether it would need to reach the hub would be the download it exists
    to warn about.

    Args:
        repo: A repository id, or a local directory — which is present by definition.
        patterns: The same globs :func:`snapshot` would pass. A repository whose ONNX export
            is cached and whose safetensors are not is present for one backend and absent for
            the other, so the answer is per pattern set rather than per repository.
        revision: A commit, branch or tag.

    Returns:
        Whether every matching file is already in the local cache.
    """
    local = Path(repo).expanduser()
    if local.is_dir():
        return True

    from huggingface_hub import snapshot_download  # noqa: PLC0415 - kept out of import time

    try:
        snapshot_download(
            repo, revision=revision, allow_patterns=list(patterns), local_files_only=True
        )
    except Exception:  # noqa: BLE001 - the reason `snapshot` gives, in a place that must not raise
        # Anything at all means "not usable from disk", which is the whole question. A
        # narrower except would let the one unlisted error escape a *diagnostic* as a
        # traceback, which is worse than the answer being conservative.
        return False
    return True


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


__all__ = ["OFFLINE_ENV", "PROGRESS_ENV", "ModelUnavailableError", "is_cached", "snapshot"]
