"""Fixtures shared by every parser suite.

The ``corpus`` fixture lives in the root ``conftest.py`` so that suites outside this package
— the async-generator lifecycle checks, in particular — can use the same built corpus rather
than a second copy of it.
"""

from __future__ import annotations

import pytest

from manicule.chunking import StructuralChunker
from tests.parsers.support import make_chunker


@pytest.fixture
def chunker() -> StructuralChunker:
    """The chunker every parser suite chunks with: 512 tokens, 64 overlap."""
    return make_chunker()
