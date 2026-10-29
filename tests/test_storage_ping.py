"""``ping`` is what a readiness probe asks, so it must not queue behind the writer."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from manicule.storage.engine import writer_slot

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from manicule.storage.docstore import SqliteDocStore

BOUND = 2.0


async def test_ping_answers_while_the_writer_slot_is_held(
    store: SqliteDocStore,
    engine: AsyncEngine,
) -> None:
    async with writer_slot(engine, "long-ingest"):
        await asyncio.wait_for(store.ping(), BOUND)
