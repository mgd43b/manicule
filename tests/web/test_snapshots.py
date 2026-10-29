"""The dashboard walks the index, so it is computed once and shared.

A 394k-document instance took 74 to 184 seconds per ``GET /ui``, and every refresh started
another walk behind the one still running.
"""

from __future__ import annotations

import asyncio

from manicule.app.results import Envelope, ErrorInfo
from manicule.web.rendering import Panel
from manicule.web.snapshots import SnapshotCache
from tests.web.support import backend_with_a_document, client_for


class Computation:
    """A stand-in for the walk: counts its starts, and can be held open."""

    def __init__(self) -> None:
        self.starts = 0
        self.release = asyncio.Event()
        self.ok = True

    async def __call__(self) -> dict[str, Panel]:
        self.starts += 1
        await self.release.wait()
        if self.ok:
            envelope = Envelope(op="x", ok=True, workspace="w", data={})
        else:
            envelope = Envelope(
                op="x", ok=False, workspace="w", error=ErrorInfo(type="E", message="down")
            )
        return {"x": Panel(envelope=envelope)}


async def test_concurrent_requests_share_one_computation() -> None:
    cache = SnapshotCache()
    walk = Computation()

    waiting = [asyncio.create_task(cache.get("w", walk)) for _ in range(5)]
    await asyncio.sleep(0)
    walk.release.set()
    results = await asyncio.gather(*waiting)

    assert walk.starts == 1
    assert len({id(result) for result in results}) == 1
    assert results[0] is not None


async def test_a_cancelled_waiter_does_not_cancel_the_walk() -> None:
    cache = SnapshotCache()
    walk = Computation()

    impatient = asyncio.create_task(cache.get("w", walk))
    patient = asyncio.create_task(cache.get("w", walk))
    await asyncio.sleep(0)
    impatient.cancel()
    walk.release.set()

    patient_result = await patient
    assert patient_result is not None
    assert patient_result.complete
    assert walk.starts == 1


async def test_an_expired_snapshot_is_served_at_once_and_refreshed_behind() -> None:
    cache = SnapshotCache(ttl_s=0)
    walk = Computation()
    walk.release.set()
    first = await cache.get("w", walk)

    walk.release.clear()
    again = await asyncio.wait_for(cache.get("w", walk), timeout=1)
    await asyncio.sleep(0)

    assert again is first, "the stale result is returned without waiting for the refresh"
    assert walk.starts == 2
    await cache.get("w", walk)
    assert walk.starts == 2, "a refresh already in flight is not started twice"
    walk.release.set()


async def test_a_failed_first_page_is_not_kept() -> None:
    cache = SnapshotCache()
    walk = Computation()
    walk.ok = False
    walk.release.set()

    await cache.get("w", walk)
    await cache.get("w", walk)

    assert walk.starts == 2


async def test_a_failed_refresh_replaces_the_healthy_snapshot_and_is_retried() -> None:
    cache = SnapshotCache(ttl_s=0)
    walk = Computation()
    walk.release.set()
    healthy = await cache.get("w", walk)
    assert healthy is not None

    walk.ok = False
    await cache.get("w", walk)  # serves stale, refreshes behind
    await asyncio.sleep(0.01)
    shown = await cache.get("w", walk)

    assert shown is not None
    assert not shown.complete, "an outage is shown rather than hidden behind the old page"
    await asyncio.sleep(0.01)
    assert walk.starts >= 3, "and it is retried on the next request"


async def test_a_cold_request_gives_up_waiting_but_the_walk_continues() -> None:
    cache = SnapshotCache()
    walk = Computation()

    assert await cache.get("w", walk, wait_s=0.01) is None

    walk.release.set()
    later = await cache.get("w", walk, wait_s=1)
    assert later is not None
    assert walk.starts == 1


def test_the_dashboard_says_when_its_counts_were_taken() -> None:
    with client_for(backend_with_a_document()[0]) as client:
        first = client.get("/ui")
        second = client.get("/ui")

    assert "as of " in first.text
    assert first.text.split("as of ")[1][:19] == second.text.split("as of ")[1][:19], (
        "a second load inside the TTL is the same snapshot"
    )
