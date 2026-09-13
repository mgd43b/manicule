"""Everything this plugin's tests need: manicule's environment, and a **real** store.

A real :class:`~manicule.storage.docstore.SqliteDocStore` rather than a fake, because three of
the four things this plugin does to a store are things a fake would get right by construction and
the product could get wrong: ``relate`` refuses an edge whose ends are not live chunks in this
workspace, it is idempotent, and ``search_lexical`` tokenizes the text being searched for.
The inbound pass depends on that last one in particular — a hand-written stand-in returning
"every chunk containing this substring" would make the test pass against a behavior FTS5 does
not have.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest_asyncio

from manicule.storage.docstore import SqliteDocStore
from manicule.storage.engine import create_engine
from manicule.storage.migrator import upgrade
from manicule.testing.fixtures import manicule_environment, settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine

__all__ = ["engine", "manicule_environment", "settings", "store"]


@pytest_asyncio.fixture
async def engine(tmp_path: Path) -> AsyncIterator[AsyncEngine]:
    """A migrated database in a fresh data directory."""
    built = create_engine(tmp_path / "data")
    await upgrade(built)
    try:
        yield built
    finally:
        await built.dispose()


@pytest_asyncio.fixture
async def store(engine: AsyncEngine) -> SqliteDocStore:
    made = SqliteDocStore(engine)
    await made.ensure_workspace()
    return made
