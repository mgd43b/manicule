"""Fixtures shared by every parser suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from manicule.chunking import StructuralChunker
from tests.corpus import build_all
from tests.parsers.support import make_chunker


@pytest.fixture(scope="session")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Every generator's fixtures, built once per session.

    Built rather than committed: it keeps the repository small, makes each fixture's
    structure reviewable as code, and lets the hostile cases exist without being stored.
    """
    return build_all(tmp_path_factory.mktemp("corpus"))


@pytest.fixture
def chunker() -> StructuralChunker:
    """The chunker every parser suite chunks with: 512 tokens, 64 overlap."""
    return make_chunker()
