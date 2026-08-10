"""Test configuration.

The environment isolation and the ``settings`` fixture are shipped in
:mod:`manicule.testing.fixtures` rather than written here, so that a plugin author gets the
same setup with one import and manicule's own suite proves that import works.

The storage fixtures are local: they build a real migrated database in a temporary directory,
which is a manicule-internal concern rather than something a plugin author needs.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from manicule.testing.fixtures import manicule_environment, settings
from tests.corpus import build_all
from tests.storage_helpers import data_dir, engine, store

__all__ = [
    "corpus",
    "data_dir",
    "engine",
    "grammar_cache",
    "manicule_environment",
    "model_cache",
    "settings",
    "store",
]


@pytest.fixture(scope="session", autouse=True)
def model_cache() -> None:
    """Pin the Hugging Face cache to this machine's real one, for the whole session.

    The same hazard as ``grammar_cache`` below, arriving through a different library.
    ``manicule_environment`` redirects ``XDG_CACHE_HOME`` at every test, and recent
    ``huggingface_hub`` resolves its cache through that variable — so a model sitting on disk
    becomes invisible the moment a test imports the hub lazily, and every embedding suite skips
    on a machine where the weights are right there. Worse in CI, where the pre-seed step would
    download several hundred megabytes into a directory nothing later reads, and the suite would
    report green having checked nothing.

    Session-scoped and autouse so it runs while the environment is still the real one. The
    resolved path is written back to ``HF_HUB_CACHE``, which takes precedence over both
    ``HF_HOME`` and the XDG variable, so a later redirection cannot move it.
    """
    try:
        import huggingface_hub.constants as hub  # noqa: PLC0415 - an embeddings extra
    except ImportError:
        return
    os.environ["HF_HUB_CACHE"] = str(hub.HF_HUB_CACHE)


@pytest.fixture(scope="session", autouse=True)
def grammar_cache() -> None:
    """Pin the tree-sitter grammar cache to this machine's real one, for the whole session.

    ``manicule_environment`` redirects ``XDG_CACHE_HOME`` at every test, which is right for
    everything manicule writes and wrong for this one thing. Grammars are not per-test state:
    they are a machine resource, pre-seeded once by ``manicule doctor --fix`` or by CI, and a
    per-test cache directory is empty by construction — so every assertion about the code
    parser would skip or refuse, on a machine where the grammars are sitting right there.

    It was invisible for the worst possible reason: the pack resolves its cache through
    ``platformdirs``, which honours ``XDG_CACHE_HOME`` on Linux and uses
    ``~/Library/Caches`` on macOS. The suite therefore passed on a developer's machine and
    failed on CI, which is exactly the "one corpus, two behaviours" split that
    ``manicule.parsers.grammars`` exists to prevent — arriving through the test harness rather
    than through the code.

    Session-scoped and autouse so it runs before any function-scoped fixture has redirected
    anything: at this point the environment is the real one, so the path captured here is the
    real cache, and every later ``configure_pack`` call restores that same path explicitly.
    """
    from manicule.parsers import grammars  # noqa: PLC0415 - a parsing extra, not core

    grammars.configure_pack(grammars.DECLARED_LANGUAGES)


@pytest.fixture(scope="session")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Every generator's fixtures, built once per session.

    Built rather than committed: it keeps the repository small, makes each fixture's
    structure reviewable as code, and lets the hostile cases exist without being stored.
    """
    return build_all(tmp_path_factory.mktemp("corpus"))
