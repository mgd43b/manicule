"""A page's panels, computed once and shared, for the pages whose panels cost O(index).

The dashboard's counts and diagnosis walk the corpus. On a few hundred thousand documents that
took between seventy seconds and three minutes per request, so every browser refresh queued
another walk behind the one still running on a server that holds a single writer.

This module changes *when* the walk happens, never *what* it computes: the panels are the same
envelopes :func:`~manicule.web.rendering.panels` builds, produced by the same operations.

- **Coalesced.** Concurrent requests for a workspace await one computation.
- **Served stale, refreshed behind.** Once a result exists it is returned immediately, with the
  moment it was taken, and an expired one starts a single background refresh.
- **A failure is never called fresh.** A page with a failed panel is stored already expired when
  it replaces an earlier snapshot (so an outage is shown, not hidden behind a stale healthy
  page), and is not stored at all when there is nothing to replace. Either way the next request
  recomputes, so an outage is not displayed for a TTL after it has ended.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from manicule.web.rendering import Panel

DEFAULT_TTL_S = 60.0
"""How long a computed page is called fresh. The dashboard is a glance, not a monitor."""


@dataclass(frozen=True, slots=True)
class Snapshot:
    """One computed set of panels and when it was taken."""

    panels: Mapping[str, Panel]
    taken_at: datetime
    monotonic: float

    @property
    def complete(self) -> bool:
        return all(panel.ok for panel in self.panels.values())


@dataclass
class SnapshotCache:
    """Per-key snapshots with single-flight computation. One per application."""

    ttl_s: float = DEFAULT_TTL_S
    _snapshots: dict[str, Snapshot] = field(default_factory=dict[str, Snapshot])
    _inflight: dict[str, asyncio.Task[Snapshot]] = field(
        default_factory=dict[str, asyncio.Task[Snapshot]]
    )

    def _start(
        self, key: str, compute: Callable[[], Awaitable[Mapping[str, Panel]]]
    ) -> asyncio.Task[Snapshot]:
        running = self._inflight.get(key)
        if running is not None:
            return running

        async def run() -> Snapshot:
            try:
                computed = await compute()
                snapshot = Snapshot(
                    panels=computed, taken_at=datetime.now(UTC), monotonic=time.monotonic()
                )
                if snapshot.complete:
                    self._snapshots[key] = snapshot
                elif key in self._snapshots:
                    self._snapshots[key] = replace(snapshot, monotonic=-math.inf)
                return snapshot
            finally:
                self._inflight.pop(key, None)

        task = asyncio.get_running_loop().create_task(run(), name=f"snapshot:{key}")
        self._inflight[key] = task
        # A background refresh nobody awaits must not log "exception never retrieved"; the next
        # request recomputes, and a request that awaits it still sees the exception.
        task.add_done_callback(lambda done: done.cancelled() or done.exception())
        return task

    async def get(
        self,
        key: str,
        compute: Callable[[], Awaitable[Mapping[str, Panel]]],
        *,
        wait_s: float | None = None,
    ) -> Snapshot | None:
        """The snapshot for ``key``: cached if there is one, else the shared computation.

        ``wait_s`` bounds how long a cold request waits for the first computation, which keeps
        running either way. ``None`` waits for it; on a timeout the answer is ``None`` and the
        caller renders a page that says the counts are still being computed.
        """
        held = self._snapshots.get(key)
        if held is not None:
            if time.monotonic() - held.monotonic >= self.ttl_s:
                self._start(key, compute)
            return held
        # Shielded: one impatient client closing its connection must not cancel the walk the
        # others are waiting on.
        try:
            async with asyncio.timeout(wait_s):
                return await asyncio.shield(self._start(key, compute))
        except TimeoutError:
            return None
