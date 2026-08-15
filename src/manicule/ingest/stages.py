"""The plumbing between ingest stages: bounded hand-offs, and counters for what they did.

Kept out of :mod:`manicule.ingest.pipeline` because the two answer different questions. The
pipeline decides what happens to a document; this decides how many documents may be in a stage
at once and what happens when the next stage is not ready for them. Reading either one while
the other is on screen is how a concurrency bug survives review.

**Everything here is deliberately dull.** A staged pipeline expressed in clever combinators is
unreadable during the incident it causes, so a hand-off is a queue with a name, an end-of-stream
is a sentinel counted out one per consumer, and a concurrency bound is a gauge that goes up and
comes down. Nothing is inferred from a task set, and nothing depends on the order in which
tasks happen to be scheduled.

**Backpressure is why each local hand-off is bounded.** Discovery itself writes the durable
acquisition journal before these stages begin, so a slow embedder cannot age its pagination
cursor. The journal reader can still outrun fetch, and fetch can outrun parse; bounding both is
what prevents that decoupling from becoming corpus-sized memory. :attr:`Conveyor.blocked_puts`
makes those local bounds observable (``docs/ingest.md`` §8.3).
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Generator

    from manicule.core.content import Metadata


class _EndOfStream:
    """One per consumer, and the only thing that ends a consumer's loop.

    A sentinel rather than a flag, because a flag has to be checked and a consumer blocked on
    ``get`` is not checking anything. Counting one out per consumer is what makes "every
    consumer sees the end" true without a broadcast primitive.
    """

    __slots__ = ()


_END: Final = _EndOfStream()


@dataclass(frozen=True, slots=True)
class QueueReport:
    """What one hand-off did, in counts.

    Counts rather than contents, so a diagnostic can be printed, logged and stored on the
    connector row without any of it being about what the documents said.
    """

    name: str
    capacity: int
    peak_depth: int
    blocked_puts: int
    """How many times a producer found this hand-off full and had to wait.

    **Direct evidence that a local producer reached its bound.** A configured bound that never
    filled proves only its capacity; a blocked put proves the journal reader or fetch worker
    really waited. Zero is a real answer — it means the consumer kept up.
    """

    def as_metadata(self) -> Metadata:
        return {
            "capacity": self.capacity,
            "peak_depth": self.peak_depth,
            "blocked_puts": self.blocked_puts,
        }


class Conveyor[T]:
    """A bounded hand-off from one stage to the next, with an end every consumer sees.

    Three things it is not, because each is a shape somebody reaches for first:

    **Not an unbounded queue.** The capacity is the whole point — see the module docstring.

    **Not a queue plus a "done" flag.** A consumer blocked in ``get`` never reads a flag, so a
    flag needs either a poll loop or a second wake-up mechanism. One sentinel per consumer needs
    neither, and it cannot leave a consumer blocked forever after the producers have gone.

    **Not closed by whoever finishes first.** A stage with several producers is not done when
    one of them is, so :meth:`finish` counts them down and the last one out sends the
    end-of-stream. Closing on the first producer would abandon whatever the others still had.

    **The bound is a semaphore rather than the queue's own ``maxsize``, and the reason is the
    end-of-stream.** :meth:`finish` runs from a producer's ``finally``, which is where a stage
    lands when the run is being torn down — and at that moment the consumers are being canceled
    too. A ``maxsize`` queue would make sending the sentinels block on a full queue nobody will
    ever drain again, inside a ``finally``, which is a teardown that never completes and a
    command that never returns. Bounding the *items* with a semaphore leaves the sentinels free
    to be delivered into an unbounded queue, so :meth:`finish` cannot wait for anything.
    """

    def __init__(self, *, name: str, capacity: int, consumers: int, producers: int = 1) -> None:
        if capacity < 1:  # pragma: no cover - the caller derives this from validated settings
            msg = f"a conveyor needs room for at least one item; {name} was given {capacity}"
            raise ValueError(msg)
        self.name = name
        self.capacity = capacity
        self.peak_depth = 0
        self.blocked_puts = 0
        self._items: asyncio.Queue[T | _EndOfStream] = asyncio.Queue()
        self._room = asyncio.Semaphore(capacity)
        self._consumers = consumers
        self._producers = producers
        # Items waiting, kept apart from `self._items.qsize()` because that also counts the
        # end-of-stream sentinels once they have gone out, and a depth that jumped by the
        # consumer count at teardown would be a diagnostic reporting a queue that never existed.
        self._depth = 0

    @property
    def depth(self) -> int:
        """How many items are waiting right now. Read-only: :meth:`put` and :meth:`take` own it."""
        return self._depth

    async def put(self, item: T) -> None:
        """Hand an item on, waiting while the next stage is full.

        The wait is the backpressure, and it is counted rather than merely permitted.
        """
        if self._room.locked():
            self.blocked_puts += 1
        await self._room.acquire()
        self._depth += 1
        self.peak_depth = max(self.peak_depth, self._depth)
        self._items.put_nowait(item)

    async def wait_for_room(self) -> None:
        """Wait until one item could be admitted, without reserving the slot.

        Used by a single journal reader before it allocates the next page record. With one
        producer there is nobody who can steal the observed room before its following
        :meth:`put`; consumers can only create more room.
        """
        if self._room.locked():
            self.blocked_puts += 1
        await self._room.acquire()
        self._room.release()

    async def take(self) -> T | None:
        """The next item, or ``None`` when this consumer has reached the end of the stream.

        The room an item occupied is given back as it leaves the queue rather than when the
        consumer has finished with it, which is what makes the capacity a statement about how
        much is *waiting* — the quantity a queue depth is supposed to be about. What a consumer
        is holding is bounded by how many consumers there are.
        """
        item = await self._items.get()
        if isinstance(item, _EndOfStream):
            return None
        self._depth -= 1
        self._room.release()
        return item

    def finish(self) -> None:
        """One producer is done. When the last one is, every consumer is told.

        Synchronous and unable to wait, which is what makes it safe in the ``finally`` that every
        producer ends in — see the class docstring. The decrement and the comparison have no
        ``await`` between them, so two producers finishing in the same tick cannot both read zero
        and send two rounds of sentinels.
        """
        self._producers -= 1
        if self._producers > 0:
            return
        for _ in range(self._consumers):
            self._items.put_nowait(_END)

    def report(self) -> QueueReport:
        return QueueReport(
            name=self.name,
            capacity=self.capacity,
            peak_depth=self.peak_depth,
            blocked_puts=self.blocked_puts,
        )


class Gauge:
    """How many are in a stage now, and the most there have ever been at once.

    The peak is what a concurrency claim is made of. "At most ``fetch_concurrency`` fetches"
    and "the embedder is never called twice at once" are both statements about a maximum, and a
    test that samples the current value is testing whether it happened to look while the second
    one was there.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.active = 0
        self.peak = 0

    def enter(self) -> None:
        """One more occupant.

        Synchronous, and it has to be: nothing between reading ``active`` and writing it back
        may await, or two tasks would interleave a read-modify-write and undercount the peak.
        """
        self.active += 1
        self.peak = max(self.peak, self.active)

    def leave(self) -> None:
        """One fewer occupant."""
        self.active -= 1

    @contextmanager
    def holding(self) -> Generator[None]:
        """Count one occupant for the length of the block, where the block is one function.

        The pair is spelled out with :meth:`enter` and :meth:`leave` where an occupancy spans
        two stages and cannot be a block — one hand-off is exactly that case.
        """
        self.enter()
        try:
            yield
        finally:
            self.leave()

    def rebase(self) -> None:
        """Forget the peak without disturbing the count of who is inside.

        Called at the start of a run so its report is about that run. The active count is
        deliberately left alone: a second operation sharing this pipeline is *inside* the stage,
        and zeroing it would make the gauge go negative when that operation leaves.

        **Two runs overlapping on one pipeline would rebase each other's peaks**, and each would
        report the other's high-water mark rather than its own. Nothing does that today — a sync
        is awaited from one place, and the data directory admits one writer — and a gauge per run
        would be the fix if anything ever did. The alternative for the two stages that share a
        pool and an accelerator is worse: those peaks are about a process-wide resource, so a
        per-run reading of them would be a smaller number and a false one.
        """
        self.peak = self.active


class CountedLock:
    """An :class:`asyncio.Lock` that says how many holders it has had at once.

    Wrapped rather than subclassed so that what is passed to
    :func:`~manicule.ingest.embedding.embed_or_reuse` is exactly an async context manager and
    nothing more — the scope of that lock is argued at its call site, and widening the object
    would invite widening the scope.

    A peak above one is the whole of "the embedder was called concurrently", so this is the
    measurement that assertion is made from rather than a proxy for it.
    """

    def __init__(self, name: str) -> None:
        self.gauge = Gauge(name)
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> None:
        await self._lock.acquire()
        self.gauge.active += 1
        self.gauge.peak = max(self.gauge.peak, self.gauge.active)

    async def __aexit__(self, *_: object) -> None:
        self.gauge.active -= 1
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()


@dataclass(frozen=True, slots=True)
class StageReport:
    """What one run's stages did, in counts.

    On the run report rather than on the pipeline because it is a fact about a run, and because
    ``docs/ingest.md`` §14 wants queue depth and stage occupancy where ``doctor`` can read them.

    **Two of these are process-wide and are marked as such.** Parse workers and the embedder are
    one pool and one accelerator for the whole process, so a sweep running beside a sync is
    counted in :attr:`peak_parses` and :attr:`peak_embeds` too. Attributing them to a run would
    be a smaller number and a false one.
    """

    accepted: int = 0
    """Top-level source identities discovery committed. What ``--limit`` bounds."""

    fetch_queue: QueueReport = field(
        default_factory=lambda: QueueReport(name="fetch", capacity=0, peak_depth=0, blocked_puts=0)
    )
    parse_queue: QueueReport = field(
        default_factory=lambda: QueueReport(name="parse", capacity=0, peak_depth=0, blocked_puts=0)
    )
    peak_fetches: int = 0
    peak_parses: int = 0
    """Process-wide: parse attempts running at once, in the worker pool."""

    peak_embeds: int = 0
    """Process-wide: holders of the embedding lock at once. Above one is a defect."""

    peak_bodies: int = 0
    """The most fetched bodies held in memory at once, queued and in flight together.

    The memory bound the queue capacities exist to produce, measured rather than derived: it is
    what "do not accumulate the complete fetched corpus" means in a number.
    """

    peak_discovery_records: int = 0
    """Journal records held by the read page and fetch hand-off together."""

    def as_metadata(self) -> Metadata:
        return {
            "accepted": self.accepted,
            "fetch_queue": self.fetch_queue.as_metadata(),
            "parse_queue": self.parse_queue.as_metadata(),
            "peak_fetches": self.peak_fetches,
            "peak_parses": self.peak_parses,
            "peak_embeds": self.peak_embeds,
            "peak_bodies": self.peak_bodies,
            "peak_discovery_records": self.peak_discovery_records,
        }


__all__ = ["Conveyor", "CountedLock", "Gauge", "QueueReport", "StageReport"]
