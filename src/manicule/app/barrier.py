"""The process-local barrier between derived-index publications and everything that fences them.

**What it is for.** Reset, rebuild, re-embed, reconciliation and the tombstone sweep each move
the pointer or the identity that a publication is written against, so none of them may run while
a document is half-published, and no publication may begin while one of them is in flight. That
is the *exclusive* side of this barrier and it is what the single reentrant lock used to be.

**What it was costing.** The lock was also taken by every connector sync for its whole walk, so
one source waiting on its server held every other publication in the process — including one
document an assistant had just written and was waiting to search for. A publication needs
exclusion from *maintenance* and from *another run of its own source*; it does not need
exclusion from a different source's fetch. This barrier says exactly that:

* **Exclusive** — one holder, which waits for every source lease to drain. While one waits,
  admission is closed, so a stream of syncs cannot starve a reset.
* **Source lease** — one run's whole lifecycle, concurrent with other leases. There are two
  slots. One belongs to the configured authoring source and one to any other source, so an
  interactive write never queues behind a background sync of something else, and a second
  background sync cannot take the authoring slot's neighbor and recreate the stall. With
  authoring unconfigured every source shares the one slot, which is the old behavior exactly.
* **Publication** — a single document's write. Covered by a lease for its own source or by the
  exclusive hold; otherwise exclusive, which keeps a bare ``ingest_raw`` as strict as before.

**What it does not do.** It is the process-local half only: the data directory's writer lock
(``manicule.lock``) still excludes other processes, and the vector store's generation pin
still fences the filesystem. It does not bound embedding, parse-pool or SQLite-writer queues,
which a write and a background run still share, and a maintenance task that is waiting closes
admission, so a write behind it queues behind that task's wait for running syncs.

**Reentrancy is by hold, not by task.** Each acquisition pushes a :class:`_Hold` onto a context
variable that child tasks inherit, and clears its ``active`` flag before releasing. A context
copied into a task that outlives its lease therefore finds a spent hold and has to acquire
again rather than publishing under a lease that was handed to someone else.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from manicule.core.errors import ManiculeError

__all__ = ["BarrierError", "DerivedBarrier"]


class BarrierError(ManiculeError):
    """A task asked the barrier for something it cannot give without waiting on itself.

    Raised rather than waited out. A task that holds a source lease and asks for the exclusive
    side would wait for its own lease to drain, and the same is true of a lease for one source
    publishing another's documents; both are programming errors, and a deadlock is the worst
    way to learn of one.
    """


class _Kind(StrEnum):
    EXCLUSIVE = "exclusive"
    SOURCE = "source"


AUTHORING_SLOT: Final = "authoring"
OTHER_SLOT: Final = "other"


@dataclass(slots=True, eq=False)
class _Hold:
    barrier: DerivedBarrier
    kind: _Kind
    source: str | None = None
    active: bool = True


_HELD: ContextVar[tuple[_Hold, ...]] = ContextVar("manicule_derived_barrier_holds", default=())


@dataclass(slots=True)
class _Slots:
    """Which slots are running, and who is waiting for the state to change."""

    running: dict[str, bool] = field(
        default_factory=lambda: {AUTHORING_SLOT: False, OTHER_SLOT: False}
    )
    waiters: list[asyncio.Future[None]] = field(default_factory=lambda: [])


class DerivedBarrier:
    """Exclusive maintenance, two source-lease slots, and covered single-document publication."""

    def __init__(self, *, authoring_source: Callable[[], str | None]) -> None:
        """Create a barrier.

        Args:
            authoring_source: The configured authoring source, read on each acquisition rather
                than captured, so a settings object replaced under a long-lived runtime is
                honored. ``None`` makes every source share one slot.
        """
        self._authoring_source = authoring_source
        self._exclusive = asyncio.Lock()
        self._exclusive_demand = 0
        self._state = _Slots()

    # --- introspection ----------------------------------------------------------------------

    @property
    def maintenance_pending(self) -> bool:
        """Whether an exclusive holder is running or queued, so admission is closed."""
        return self._exclusive_demand > 0

    def holds(self) -> bool:
        """Whether the calling context holds this barrier in any mode.

        What lets a component that a lease holder reaches (a cached pipeline, say) answer
        without queueing behind the maintenance its caller is itself keeping out.
        """
        return any(hold.barrier is self and hold.active for hold in _HELD.get())

    def slot_of(self, source: str) -> str:
        """The slot a source's lease occupies."""
        authoring = self._authoring_source()
        return AUTHORING_SLOT if authoring is not None and source == authoring else OTHER_SLOT

    # --- the three ways in ------------------------------------------------------------------

    @asynccontextmanager
    async def exclusive(self) -> AsyncGenerator[None]:
        """Run maintenance alone: closes admission, waits for every lease, then holds all."""
        held = self._holding()
        if any(hold.kind is _Kind.EXCLUSIVE for hold in held):
            yield
            return
        if held:
            msg = (
                "a task holding a source lease asked for exclusive maintenance, which waits "
                "for every lease to drain and would be waiting for its own; maintenance runs "
                "outside a sync, not inside one"
            )
            raise BarrierError(msg)
        hold = _Hold(self, _Kind.EXCLUSIVE)
        # Counted before the first await, so admission closes the moment this call starts
        # waiting for its turn and not only once it has the lock.
        self._exclusive_demand += 1
        try:
            async with self._exclusive:
                await self._until(lambda: not any(self._state.running.values()))
                token = _HELD.set((*_HELD.get(), hold))
                try:
                    yield
                finally:
                    hold.active = False
                    _HELD.reset(token)
        finally:
            self._exclusive_demand -= 1
            self._wake()

    @asynccontextmanager
    async def source(self, name: str) -> AsyncGenerator[None]:
        """Run one source's whole lifecycle, alongside leases for other slots."""
        held = self._holding()
        if self._covers(held, name):
            yield
            return
        if held:
            msg = (
                f"a task already holding the derived barrier for {self._describe(held)} asked "
                f"for a lease on source {name!r}; leases are per run and do not nest across "
                "sources"
            )
            raise BarrierError(msg)
        slot = self.slot_of(name)
        hold = _Hold(self, _Kind.SOURCE, name)
        await self._until(
            lambda: self._exclusive_demand == 0 and not self._state.running[slot],
            then=lambda: self._state.running.__setitem__(slot, True),
        )
        token = _HELD.set((*_HELD.get(), hold))
        try:
            yield
        finally:
            hold.active = False
            _HELD.reset(token)
            self._state.running[slot] = False
            self._wake()

    @asynccontextmanager
    async def publication(self, name: str) -> AsyncGenerator[None]:
        """Publish one document of source ``name``: covered, or exclusive when standing alone."""
        held = self._holding()
        if self._covers(held, name):
            yield
            return
        if held:
            msg = (
                f"a task holding the derived barrier for {self._describe(held)} published a "
                f"document of source {name!r}, which that hold does not cover"
            )
            raise BarrierError(msg)
        async with self.exclusive():
            yield

    # --- internals --------------------------------------------------------------------------

    def _holding(self) -> tuple[_Hold, ...]:
        return tuple(hold for hold in _HELD.get() if hold.barrier is self and hold.active)

    @staticmethod
    def _covers(held: tuple[_Hold, ...], source: str) -> bool:
        return any(
            hold.kind is _Kind.EXCLUSIVE or (hold.kind is _Kind.SOURCE and hold.source == source)
            for hold in held
        )

    @staticmethod
    def _describe(held: tuple[_Hold, ...]) -> str:
        return ", ".join(
            "exclusive maintenance" if hold.kind is _Kind.EXCLUSIVE else f"source {hold.source!r}"
            for hold in held
        )

    async def _until(
        self, ready: Callable[[], bool], *, then: Callable[[], object] | None = None
    ) -> None:
        """Wait until ``ready``, and run ``then`` in the same step that observed it.

        The check and the action share one synchronous stretch, which is what makes taking a
        slot atomic without a lock. Every state change calls :meth:`_wake`, so a waiter cannot
        sleep through the change it was waiting for, and a canceled waiter only withdraws its
        own future.
        """
        loop = asyncio.get_running_loop()
        while not ready():
            future = loop.create_future()
            self._state.waiters.append(future)
            try:
                await future
            finally:
                if future in self._state.waiters:
                    self._state.waiters.remove(future)
        if then is not None:
            then()

    def _wake(self) -> None:
        for future in tuple(self._state.waiters):
            if not future.done():
                future.set_result(None)
