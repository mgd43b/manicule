"""Test configuration.

The environment isolation and the ``settings`` fixture are shipped in
:mod:`manicule.testing.fixtures` rather than written here, so that a plugin author gets the
same setup with one import and manicule's own suite proves that import works.

The storage fixtures are local: they build a real migrated database in a temporary directory,
which is a manicule-internal concern rather than something a plugin author needs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from manicule.testing.fixtures import manicule_environment, settings
from tests.corpus import build_all
from tests.storage_helpers import data_dir, engine, store

__all__ = [
    "corpus",
    "data_dir",
    "engine",
    "manicule_environment",
    "settings",
    "store",
]


@pytest.fixture(scope="session")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Every generator's fixtures, built once per session.

    Built rather than committed: it keeps the repository small, makes each fixture's
    structure reviewable as code, and lets the hostile cases exist without being stored.
    """
    return build_all(tmp_path_factory.mktemp("corpus"))
