"""The authoring latency benchmark's shape, at a size that runs in seconds.

Pins that the comparison measures what it says: against an unrelated source parked in its
fetch, the legacy barrier makes the first write wait out the park and the leased one does not,
and the table names every barrier and condition. No wall-clock magnitude is asserted beyond
that ordering, which holds by construction rather than by speed.
"""

from __future__ import annotations

from tests.benchmarks.authoring_latency import (
    BARRIERS,
    Result,
    measure,
    percentile,
    render,
)

PARK_S = 1.0


def _result(results: list[Result], barrier: str, condition: str) -> Result:
    (found,) = [r for r in results if r.barrier == barrier and r.condition == condition]
    return found


async def test_a_parked_source_stalls_the_legacy_barrier_and_not_the_leased_one() -> None:
    results = await measure(writes=2, cold_runs=1, park_s=PARK_S, conditions=("parked",))

    legacy = _result(results, "legacy", "parked")
    leased = _result(results, "leased", "parked")
    assert legacy.cold[0].barrier_wait_ms >= PARK_S * 1000 * 0.5, "the write waited out the park"
    assert leased.cold[0].barrier_wait_ms < PARK_S * 1000 * 0.5, "the write did not wait for it"
    # No comparison of totals: the two runs have independent storage time, and the waits above
    # are what the barrier controls.


async def test_the_table_names_every_barrier_and_condition_it_measured() -> None:
    results = await measure(writes=2, cold_runs=1, park_s=0.1, conditions=("idle",))

    table = render(results)
    for barrier in BARRIERS:
        assert barrier in table
    assert "idle" in table
    assert "cold p50" in table
    assert "warm p95" in table


def test_percentile_is_nearest_rank_and_zero_for_nothing() -> None:
    assert percentile([], 0.95) == 0.0
    assert percentile([5.0], 0.95) == 5.0
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == 3.0
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 1.0) == 5.0


async def test_a_wait_in_another_task_is_not_charged_to_the_measured_write() -> None:
    """Queued maintenance acquires the barrier in its own context; only the write's waits count."""
    import asyncio  # noqa: PLC0415
    from contextlib import asynccontextmanager  # noqa: PLC0415

    from tests.benchmarks.authoring_latency import MEASURING, Timed  # noqa: PLC0415

    @asynccontextmanager
    async def slow():  # noqa: ANN202
        await asyncio.sleep(0.05)
        yield

    waited: list[float] = []

    async def other_task() -> None:
        async with Timed(slow()):
            pass

    token = MEASURING.set(waited)
    try:
        # Created inside the measured context, so it inherits the list and IS counted ...
        await asyncio.create_task(other_task())
        inherited = len(waited)
    finally:
        MEASURING.reset(token)
    # ... and one created outside it records nothing.
    await asyncio.create_task(other_task())

    assert inherited == 1
    assert len(waited) == 1
