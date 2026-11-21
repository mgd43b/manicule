"""The derived barrier on its own: who may run beside whom, and what happens when a waiter leaves.

Every test here parks tasks on events and asserts about *arrivals and ordering*, never about
elapsed time, so none of them can pass by being slow and none can fail on a loaded runner.
:func:`_settle` yields to the loop enough times for every task that can make progress to do so,
which is what turns "the other task has not entered" into a fact rather than a hope.
"""

from __future__ import annotations

import asyncio

import pytest

from manicule.app.barrier import BarrierError, DerivedBarrier

AUTHORING = "authored"


def _barrier(*, authoring: str | None = AUTHORING) -> DerivedBarrier:
    return DerivedBarrier(authoring_source=lambda: authoring)


async def _settle() -> None:
    """Let every runnable task run until it parks."""
    for _ in range(10):
        await asyncio.sleep(0)


class Held:
    """One task holding the barrier until released, and what it has done so far."""

    def __init__(self, barrier: DerivedBarrier, kind: str, source: str = "") -> None:
        self.entered = asyncio.Event()
        self._release = asyncio.Event()
        self.task = asyncio.create_task(self._body(barrier, kind, source), name=f"{kind}:{source}")

    async def _body(self, barrier: DerivedBarrier, kind: str, source: str) -> None:
        manager = {
            "source": lambda: barrier.source(source),
            "publication": lambda: barrier.publication(source),
            "exclusive": barrier.exclusive,
        }[kind]()
        async with manager:
            self.entered.set()
            await self._release.wait()

    def release(self) -> None:
        self._release.set()

    async def cancel(self) -> None:
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)


def _hold(barrier: DerivedBarrier, kind: str, source: str = "") -> Held:
    return Held(barrier, kind, source)


def _entered(held: Held) -> bool:
    return held.entered.is_set()


async def _release(*holds: Held) -> None:
    for held in holds:
        held.release()
    await asyncio.gather(*(held.task for held in holds), return_exceptions=True)


# --- the two slots --------------------------------------------------------------------------


async def test_the_authoring_source_runs_beside_another_source() -> None:
    barrier = _barrier()
    other = _hold(barrier, "source", "mail")
    await _settle()
    authoring = _hold(barrier, "source", AUTHORING)
    await _settle()

    assert _entered(other)
    assert _entered(authoring), "the authoring slot is reserved so a write never queues behind it"
    await _release(other, authoring)


async def test_a_second_background_source_queues_behind_the_first() -> None:
    """Both slots full is what reintroduced the stall, so a second non-authoring run waits."""
    barrier = _barrier()
    first = _hold(barrier, "source", "mail")
    await _settle()
    second = _hold(barrier, "source", "wiki")
    await _settle()

    assert _entered(first)
    assert not _entered(second)

    await _release(first)
    await _settle()
    assert _entered(second), "the queued run goes in as the slot frees"
    await _release(second)


async def test_a_write_is_not_behind_two_queued_background_sources() -> None:
    barrier = _barrier()
    running = _hold(barrier, "source", "mail")
    await _settle()
    queued = _hold(barrier, "source", "wiki")
    await _settle()
    authoring = _hold(barrier, "source", AUTHORING)
    await _settle()

    assert not _entered(queued)
    assert _entered(authoring)
    await _release(running, queued, authoring)


async def test_two_runs_of_the_authoring_source_do_not_overlap() -> None:
    barrier = _barrier()
    first = _hold(barrier, "source", AUTHORING)
    await _settle()
    second = _hold(barrier, "source", AUTHORING)
    await _settle()

    assert _entered(first)
    assert not _entered(second), "one source never races itself"
    await _release(first, second)


async def test_with_no_authoring_every_source_shares_one_slot() -> None:
    """Exactly the old behavior: one source run at a time."""
    barrier = _barrier(authoring=None)
    first = _hold(barrier, "source", AUTHORING)
    await _settle()
    second = _hold(barrier, "source", "mail")
    await _settle()

    assert _entered(first)
    assert not _entered(second)
    await _release(first, second)


# --- maintenance ----------------------------------------------------------------------------


async def test_maintenance_waits_for_every_running_source() -> None:
    barrier = _barrier()
    mail = _hold(barrier, "source", "mail")
    authoring = _hold(barrier, "source", AUTHORING)
    await _settle()
    reset = _hold(barrier, "exclusive")
    await _settle()

    assert not _entered(reset)
    await _release(mail)
    await _settle()
    assert not _entered(reset), "one source is still running"
    await _release(authoring)
    await _settle()
    assert _entered(reset)
    await _release(reset)


async def test_a_waiting_maintenance_task_closes_admission() -> None:
    """A stream of syncs must not be able to keep a reset waiting forever."""
    barrier = _barrier()
    running = _hold(barrier, "source", "mail")
    await _settle()
    reset = _hold(barrier, "exclusive")
    await _settle()
    assert barrier.maintenance_pending
    late = _hold(barrier, "source", AUTHORING)
    await _settle()

    assert not _entered(late), "admission is closed while maintenance waits"

    await _release(running)
    await _settle()
    assert _entered(reset)
    assert not _entered(late), "and stays closed while maintenance runs"
    await _release(reset)
    await _settle()
    assert _entered(late)
    assert not barrier.maintenance_pending
    await _release(late)


async def test_maintenance_runs_one_at_a_time_in_arrival_order() -> None:
    barrier = _barrier()
    first = _hold(barrier, "exclusive")
    await _settle()
    second = _hold(barrier, "exclusive")
    await _settle()
    third = _hold(barrier, "exclusive")
    await _settle()

    assert [_entered(first), _entered(second), _entered(third)] == [True, False, False]
    await _release(first)
    await _settle()
    assert [_entered(second), _entered(third)] == [True, False]
    await _release(second, third)


async def test_a_canceled_maintenance_waiter_reopens_admission() -> None:
    """The failure this guards is admission staying closed for good after a canceled reset."""
    barrier = _barrier()
    running = _hold(barrier, "source", "mail")
    await _settle()
    reset = _hold(barrier, "exclusive")
    await _settle()
    late = _hold(barrier, "source", AUTHORING)
    await _settle()
    assert not _entered(late)

    await reset.cancel()
    await _settle()

    assert not barrier.maintenance_pending
    assert _entered(late), "the source that queued behind the canceled reset goes in"
    await _release(running, late)


async def test_a_canceled_maintenance_waiter_behind_another_keeps_the_other_waiting() -> None:
    barrier = _barrier()
    first = _hold(barrier, "exclusive")
    await _settle()
    second = _hold(barrier, "exclusive")
    await _settle()
    late = _hold(barrier, "source", "mail")
    await _settle()

    await second.cancel()
    await _settle()

    assert barrier.maintenance_pending, "the first holder is still running"
    assert not _entered(late)
    await _release(first)
    await _settle()
    assert _entered(late)
    await _release(late)


async def test_a_canceled_source_waiter_does_not_take_the_slot() -> None:
    barrier = _barrier()
    running = _hold(barrier, "source", "mail")
    await _settle()
    waiting = _hold(barrier, "source", "wiki")
    await _settle()
    await waiting.cancel()
    behind = _hold(barrier, "source", "wiki")
    await _settle()

    await _release(running)
    await _settle()
    assert _entered(behind), "the slot the canceled waiter never took is free"
    await _release(behind)


async def test_a_holder_that_fails_releases_its_slot() -> None:
    barrier = _barrier()

    async def fail() -> None:
        async with barrier.source("mail"):
            msg = "the run raised"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="the run raised"):
        await fail()
    successor = _hold(barrier, "source", "wiki")
    await _settle()

    assert _entered(successor)
    await _release(successor)


async def test_a_canceled_holder_releases_its_slot() -> None:
    barrier = _barrier()
    running = _hold(barrier, "source", "mail")
    await _settle()
    waiting = _hold(barrier, "source", "wiki")
    await _settle()

    await running.cancel()
    await _settle()

    assert _entered(waiting)
    await _release(waiting)


# --- reentrancy and the refusals ------------------------------------------------------------


async def test_a_source_lease_covers_publication_of_its_own_documents() -> None:
    barrier = _barrier()
    async with barrier.source("mail"):
        async with barrier.publication("mail"):
            assert barrier.holds()
        async with barrier.source("mail"):
            pass


async def test_maintenance_covers_every_source_and_itself() -> None:
    barrier = _barrier()
    async with (
        barrier.exclusive(),
        barrier.exclusive(),
        barrier.source("mail"),
        barrier.publication("wiki"),
    ):
        assert barrier.holds()


async def test_a_standalone_publication_is_exclusive() -> None:
    barrier = _barrier()
    running = _hold(barrier, "source", "mail")
    await _settle()
    publication = _hold(barrier, "publication", AUTHORING)
    await _settle()

    assert not _entered(publication), "no lease covers it, so it waits for every run"
    await _release(running)
    await _settle()
    assert _entered(publication)
    await _release(publication)


async def test_a_lease_holder_cannot_ask_for_maintenance() -> None:
    """It would wait for its own lease to drain; refusing is the only answer that returns."""
    barrier = _barrier()
    async with barrier.source("mail"):
        with pytest.raises(BarrierError, match="outside a sync"):
            async with barrier.exclusive():
                pytest.fail("a lease holder entered the exclusive side")


async def test_a_lease_does_not_cover_another_sources_documents() -> None:
    barrier = _barrier()
    async with barrier.source("mail"):
        with pytest.raises(BarrierError, match="does not cover"):
            async with barrier.publication("wiki"):
                pytest.fail("published a document of another source")
        with pytest.raises(BarrierError, match="do not nest"):
            async with barrier.source("wiki"):
                pytest.fail("nested a lease for another source")


async def test_a_task_that_outlives_its_lease_cannot_reuse_it() -> None:
    """A copied context still carries the hold; the hold must be spent when the lease is."""
    barrier = _barrier()
    outcome: list[str] = []
    started = asyncio.Event()
    proceed = asyncio.Event()

    async def straggler() -> None:
        started.set()
        await proceed.wait()
        # Not covered any more, so this is a standalone publication: exclusive, and free to
        # run because nothing else is.
        assert not barrier.holds()
        async with barrier.publication("mail"):
            outcome.append("acquired afresh")

    async with barrier.source("mail"):
        task = asyncio.create_task(straggler())
        await started.wait()
    proceed.set()
    await task

    assert outcome == ["acquired afresh"]


async def test_children_of_a_lease_holder_share_the_lease() -> None:
    barrier = _barrier()
    seen: list[bool] = []

    async def worker() -> None:
        async with barrier.publication("mail"):
            seen.append(barrier.holds())

    async with barrier.source("mail"):
        await asyncio.gather(worker(), worker())

    assert seen == [True, True]


async def test_a_different_barrier_is_not_covered_by_this_one() -> None:
    mine, theirs = _barrier(), _barrier()
    async with mine.exclusive():
        assert not theirs.holds()
        async with theirs.source("mail"):
            assert theirs.holds()


async def test_the_authoring_source_is_read_when_a_lease_is_taken() -> None:
    """A settings object replaced under a long-lived runtime changes which slot is reserved."""
    authoring: list[str | None] = [None]
    barrier = DerivedBarrier(authoring_source=lambda: authoring[0])

    assert barrier.slot_of("authored") == barrier.slot_of("mail")
    authoring[0] = "authored"
    assert barrier.slot_of("authored") != barrier.slot_of("mail")


async def test_a_publication_still_running_after_its_lease_exits_keeps_maintenance_out() -> None:
    """A worker that outlives the run's block must not let maintenance start under it."""
    barrier = _barrier()
    inside, leave = asyncio.Event(), asyncio.Event()

    async def straggler() -> None:
        async with barrier.publication("mail"):
            inside.set()
            await leave.wait()

    async with barrier.source("mail"):
        task = asyncio.create_task(straggler())
        await inside.wait()
    maintenance = _hold(barrier, "exclusive")
    await _settle()

    assert not _entered(maintenance), "the straggler is still publishing"
    leave.set()
    await task
    await _settle()
    assert _entered(maintenance)
    await _release(maintenance)
    follower = _hold(barrier, "source", "mail")
    await _settle()
    assert _entered(follower), "the slot is free again, and was freed exactly once"
    await _release(follower)
