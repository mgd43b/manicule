"""The plumbing between stages, on its own.

Exercised through the pipeline everywhere else, which is the right level for "does a sync reach
its configured width" and the wrong one for "does a hand-off count a wait". A counter that is
load-bearing evidence — :attr:`~manicule.ingest.stages.Conveyor.blocked_puts` is the only direct
proof this repository has that backpressure reached a producer — has to be checked against a
situation whose answer is known by construction rather than inferred from a run.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from manicule.ingest.stages import Conveyor, CountedLock, Gauge, QueueReport, StageReport


async def test_a_conveyor_bounds_what_is_waiting_and_not_what_is_held() -> None:
    """Capacity is a statement about the queue, and the room comes back as an item leaves it.

    The alternative — releasing when the consumer has finished with the item — would make the
    capacity bound work in progress as well as backlog, and the two are bounded by different
    things: the queue by its depth, the work by how many consumers there are.
    """
    conveyor: Conveyor[int] = Conveyor(name="test", capacity=2, consumers=1)

    await conveyor.put(1)
    await conveyor.put(2)
    assert conveyor.depth == 2

    assert await conveyor.take() == 1
    assert conveyor.depth == 1
    await conveyor.put(3)
    assert conveyor.depth == 2


async def test_a_full_conveyor_makes_a_producer_wait_and_counts_every_one_that_did() -> None:
    """``blocked_puts`` is exact, and the exactness is what makes it evidence.

    **Copilot raised this**, on the reasoning that reading ``locked()`` and then awaiting
    ``acquire()`` lets two producers both observe room and only one get it. On an event loop
    they cannot: neither the read nor a successful acquire suspends, so the pair runs to
    completion before another producer is scheduled, and a producer that goes on to wait is
    always the one that read ``locked()`` as true. Asserted rather than argued, because "these
    two statements are atomic" is precisely the claim that stops being true when somebody adds
    an ``await`` between them.

    Capacity one and five producers: the first gets the room and the other four wait, so four is
    the answer by construction rather than by measurement.
    """
    conveyor: Conveyor[int] = Conveyor(name="test", capacity=1, consumers=1)
    producers = [asyncio.create_task(conveyor.put(number)) for number in range(5)]
    await asyncio.sleep(0)

    assert conveyor.blocked_puts == 4, "a producer waited and was not counted as having waited"

    for _ in range(5):
        await conveyor.take()
    await asyncio.gather(*producers)


async def test_a_conveyor_nobody_had_to_wait_for_counts_no_waits() -> None:
    """The other half: zero is a real answer, not an unmeasured one.

    Without this the counter could be a constant and the test above would still pass, which is
    the way a piece of evidence quietly stops being one.
    """
    conveyor: Conveyor[int] = Conveyor(name="test", capacity=5, consumers=1)

    await asyncio.gather(*(conveyor.put(number) for number in range(5)))

    assert conveyor.blocked_puts == 0
    assert conveyor.peak_depth == 5


async def test_every_consumer_of_a_finished_conveyor_reaches_the_end() -> None:
    """One sentinel per consumer, or a consumer waits for an item that is never coming.

    The failure this prevents is a run that finishes its work and then never returns, which is
    much worse to diagnose than one that fails.
    """
    conveyor: Conveyor[int] = Conveyor(name="test", capacity=2, consumers=3)
    await conveyor.put(1)
    conveyor.finish()

    async with asyncio.timeout(5):
        assert await conveyor.take() == 1
        assert [await conveyor.take() for _ in range(3)] == [None, None, None]


async def test_a_conveyor_with_several_producers_ends_only_when_the_last_one_has() -> None:
    """A stage is not done when one of its workers is, and closing early abandons the rest."""
    conveyor: Conveyor[int] = Conveyor(name="test", capacity=4, consumers=1, producers=3)

    conveyor.finish()
    conveyor.finish()
    await conveyor.put(1)
    conveyor.finish()

    async with asyncio.timeout(5):
        assert await conveyor.take() == 1
        assert await conveyor.take() is None


async def test_finishing_a_full_conveyor_does_not_wait_for_room() -> None:
    """The deadlock the semaphore bound exists to avoid.

    A producer closes the stage in front of it from a ``finally``, which is where it lands while
    the run is being torn down and its consumers are being canceled. If the end-of-stream were
    subject to the item bound, that ``finally`` would wait for a drain that is never coming — a
    teardown that never completes and a command that never returns.
    """
    conveyor: Conveyor[int] = Conveyor(name="test", capacity=1, consumers=2)
    await conveyor.put(1)

    # Synchronous, so it *cannot* wait — which is a stronger statement than any deadline this
    # test could put around it, and the reason the signature is `def` rather than `async def`.
    assert not inspect.iscoroutinefunction(conveyor.finish)
    conveyor.finish()

    assert conveyor.depth == 1, "the sentinels are not items and are not counted as any"
    async with asyncio.timeout(5):
        assert await conveyor.take() == 1
        assert [await conveyor.take() for _ in range(2)] == [None, None]


def test_a_conveyor_refuses_a_capacity_with_no_room_in_it() -> None:
    """Zero capacity is a hand-off nothing can ever cross, and it fails loudly rather than hangs."""
    with pytest.raises(ValueError, match="at least one item"):
        Conveyor(name="test", capacity=0, consumers=1)


async def test_a_conveyor_reports_its_capacity_and_high_water_mark() -> None:
    """What reaches the connector row, and it is counts rather than anything a document said."""
    conveyor: Conveyor[str] = Conveyor(name="fetch", capacity=3, consumers=1)
    await conveyor.put("something a document said")
    await conveyor.take()

    report = conveyor.report()

    assert report == QueueReport(name="fetch", capacity=3, peak_depth=1, blocked_puts=0)
    assert "something a document said" not in str(report.as_metadata())


def test_a_gauge_remembers_the_most_that_were_ever_inside_at_once() -> None:
    """A bound is a claim about a maximum, so sampling the current value asserts nothing."""
    gauge = Gauge("stage")

    with gauge.holding():
        with gauge.holding():
            assert gauge.active == 2
        assert gauge.active == 1
    assert gauge.active == 0
    assert gauge.peak == 2


def test_a_gauge_leaves_when_the_work_inside_it_raises() -> None:
    """An occupant that failed has still left, or every later reading is one too high."""
    gauge = Gauge("stage")

    def fail_inside() -> None:
        with gauge.holding():
            msg = "the work failed"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="the work failed"):
        fail_inside()

    assert gauge.active == 0
    assert gauge.peak == 1


def test_rebasing_a_gauge_forgets_the_peak_without_evicting_who_is_inside() -> None:
    """A run's report is about that run; an operation already inside the stage is still inside.

    Zeroing ``active`` instead would send the gauge negative when that operation left, and a
    negative occupancy in a diagnostic is worse than a stale one because it cannot be believed
    at all.
    """
    gauge = Gauge("stage")
    gauge.enter()
    gauge.enter()
    gauge.leave()

    gauge.rebase()

    assert gauge.active == 1
    assert gauge.peak == 1


async def test_a_counted_lock_admits_one_holder_and_says_so() -> None:
    """What "the embedder is serialized" is measured with.

    A peak above one is the whole of "two batches reached the model at once", so this is the
    measurement that assertion is made from rather than a proxy for it.
    """
    lock = CountedLock("embed")
    inside: list[int] = []

    async def hold() -> None:
        async with lock:
            inside.append(lock.gauge.active)
            await asyncio.sleep(0)

    await asyncio.gather(*(hold() for _ in range(4)))

    assert inside == [1, 1, 1, 1]
    assert lock.gauge.peak == 1
    assert not lock.locked()


def test_a_stage_report_carries_counts_and_nothing_a_document_said() -> None:
    """It is stored on the connector row and printed by ``doctor``, so its shape is a contract."""
    report = StageReport(
        accepted=12,
        fetch_queue=QueueReport(name="fetch", capacity=8, peak_depth=8, blocked_puts=3),
        parse_queue=QueueReport(name="parse", capacity=4, peak_depth=4, blocked_puts=1),
        peak_fetches=8,
        peak_parses=3,
        peak_embeds=1,
        peak_bodies=11,
        peak_discovery_records=7,
    )

    recorded = report.as_metadata()

    assert recorded["accepted"] == 12
    assert recorded["fetch_queue"] == {"capacity": 8, "peak_depth": 8, "blocked_puts": 3}
    assert recorded["peak_discovery_records"] == 7
    assert all(isinstance(value, (int, dict)) for value in recorded.values()), (
        "every field is a count or a record of counts, so no document content can reach here"
    )
