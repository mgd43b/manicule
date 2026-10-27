"""What the writer ledger says about waiting for the one writer slot, and holding it."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from manicule.storage.engine import (
    WriterLedger,
    create_engine,
    prepare_data_dir,
    writer_ledger,
    writer_slot,
)


@pytest.fixture
async def engine(tmp_path: Path) -> AsyncGenerator[AsyncEngine]:
    prepare_data_dir(tmp_path / "data")
    built = create_engine(tmp_path / "data")
    yield built
    await built.dispose()


async def test_a_second_writer_waits_for_as_long_as_the_first_holds(engine: AsyncEngine) -> None:
    first_in = asyncio.Event()

    async def first() -> None:
        async with writer_slot(engine, "publish"):
            first_in.set()
            await asyncio.sleep(0.05)

    async def second() -> None:
        await first_in.wait()
        async with writer_slot(engine, "lineage"):
            pass

    await asyncio.gather(first(), second())
    operations = writer_ledger(engine).operations

    assert operations["publish"].held >= 0.05
    assert operations["publish"].waited < 0.04
    assert operations["lineage"].waited >= 0.04, "queued behind the first holder"
    assert operations["lineage"].held < 0.04


async def test_the_longest_hold_survives_a_mean_that_hides_it(engine: AsyncEngine) -> None:
    for pause in (0.0, 0.0, 0.06, 0.0):
        async with writer_slot(engine, "publish"):
            await asyncio.sleep(pause)

    visit = writer_ledger(engine).operations["publish"]

    assert visit.calls == 4
    assert visit.longest_held >= 0.06
    assert visit.held / visit.calls < visit.longest_held / 2


async def test_a_writer_canceled_while_queued_records_its_wait_and_no_hold(
    engine: AsyncEngine,
) -> None:
    release = asyncio.Event()

    async def holder() -> None:
        async with writer_slot(engine, "holder"):
            await release.wait()

    held = asyncio.create_task(holder())
    await asyncio.sleep(0.01)
    queued = asyncio.create_task(_enter(engine, "queued"))
    await asyncio.sleep(0.03)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    release.set()
    await held

    visit = writer_ledger(engine).operations["queued"]
    assert visit.calls == 1
    assert visit.held == 0.0
    assert visit.waited >= 0.02


async def _enter(engine: AsyncEngine, name: str) -> None:
    async with writer_slot(engine, name):
        pass


def test_a_window_is_a_difference_of_totals_and_its_longest_hold_is_read_once() -> None:
    ledger = WriterLedger()
    ledger.record("publish", waited=0.1, held=0.2)
    before = ledger.totals()
    ledger.record("publish", waited=0.3, held=0.9)

    assert ledger.totals()["publish"][0] - before["publish"][0] == 1
    assert ledger.take_window_longest_held() == {"publish": 0.9}
    assert ledger.take_window_longest_held() == {"publish": 0.0}


async def test_each_engine_has_its_own_ledger(engine: AsyncEngine, tmp_path: Path) -> None:
    prepare_data_dir(tmp_path / "other")
    other = create_engine(tmp_path / "other")
    try:
        async with writer_slot(engine, "publish"):
            pass
        assert "publish" in writer_ledger(engine).operations
        assert writer_ledger(other).operations == {}
    finally:
        await other.dispose()
