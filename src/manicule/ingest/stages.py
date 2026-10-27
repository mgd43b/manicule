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

**Backpressure is why each local hand-off is bounded.** With acquisition journaling enabled,
discovery writes the durable journal before these stages begin, so a slow embedder cannot age
its pagination cursor. The journal reader can still outrun fetch, and fetch can outrun parse;
bounding both prevents that decoupling from becoming corpus-sized memory. The compatibility
fallback instead feeds bounded hand-offs directly from discovery. :attr:`Conveyor.blocked_puts`
makes both modes' local bounds observable (``docs/ingest.md`` §8.3).
"""

from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

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


class Stage(StrEnum):
    """The stages a pipeline keeps time for: a fixed few, which keeps the report content-free.

    A member rather than a string at each call site, so a misspelled stage is an error where it
    is written instead of a stray bucket in a report. Containment is by time: ``embed`` holds
    ``stored_vectors`` and the embed lock's wait and hold, ``commit`` holds ``fence``,
    ``stage_vectors``, ``upsert``, ``glossary`` and ``publish``, and ``document`` holds all of it.

    ``publish`` is the publication of a document with chunks and ``publish_chunkless`` of one
    without (a refused or unsupported member, an empty document, a failure), which never reach
    ``commit``. ``member`` holds the whole handling of one found inside another, and
    ``store_read`` the reads of the document store between stages. ``loop_lag`` is not a visit
    to anything: each call is one sample of how much later than asked the event loop woke a
    timer, so its seconds are time every worker spent waiting for the loop.
    """

    JOURNAL_START = "journal_start"
    BLOB_READ = "blob_read"
    DOCUMENT = "document"
    PARSE = "parse"
    PREPARE = "prepare"
    EMBED = "embed"
    STORED_VECTORS = "stored_vectors"
    EMBED_LOCK_WAIT = "embed_lock_wait"
    EMBED_LOCK_HELD = "embed_lock_held"
    GLOSSARY = "glossary"
    COMMIT = "commit"
    FENCE = "fence"
    STAGE_VECTORS = "stage_vectors"
    UPSERT = "upsert"
    PUBLISH = "publish"
    AFTER_STORE = "after_store"
    LINEAGE = "lineage"
    JOURNAL_SETTLE = "journal_settle"
    MEMBER = "member"
    STORE_READ = "store_read"
    PUBLISH_CHUNKLESS = "publish_chunkless"
    LOOP_LAG = "loop_lag"


@dataclass(frozen=True, slots=True)
class StageTime:
    """Wall time one stage took in total, and how many times it ran.

    Summed across workers, so a stage that four workers occupy at once can add up to more
    seconds than the clock on the wall. That is the point: seconds divided by calls is how long
    one visit takes, and seconds divided by the window is how many workers were inside it.
    """

    seconds: float = 0.0
    calls: int = 0
    longest: float = 0.0
    """The longest single visit seen. A mean hides a stall; this is how one is found."""


_scope: ContextVar[StageTimings | None] = ContextVar("stage_timings_scope", default=None)
"""The run whose stage totals a visit also counts toward, inherited by the tasks it starts."""


class StageTimings:
    """Cumulative wall time per named stage, for a pipeline that is slower than it should be.

    **Counters, not a trace.** Nothing here keeps an event, a document id or a label drawn from
    content: a stage name is one of a fixed few, so the memory is constant however long a run
    is. Read it as two snapshots and a subtraction (:meth:`since`), which is how a report says
    what one run did and a log line says what the last minute did.

    **A visit counts while it is happening, not only when it ends.** A snapshot adds the elapsed
    time of every visit still open to the time of those that have closed, and counts only the
    closed ones as calls. Otherwise one worker embedding for two minutes would be invisible in
    the first two windows and arrive whole in the third, and the seconds in a window divided by
    its length would stop meaning how many workers were inside.

    **Scoped to a run as well as to the pipeline.** Inside :meth:`scoped`, and in every task
    started there, a visit also counts toward the scope's totals, so two runs sharing a pipeline
    each report their own and the pipeline's own totals still hold both.

    Synchronous and unlocked, like :class:`Gauge`: nothing between reading a total and writing
    it back awaits, so two tasks cannot interleave a read-modify-write.
    """

    def __init__(self) -> None:
        self._seconds: dict[Stage, float] = {}
        self._calls: dict[Stage, int] = {}
        self._open: dict[object, tuple[Stage, float]] = {}
        self._longest: dict[Stage, float] = {}
        self._window_longest: dict[Stage, float] = {}

    def add(self, stage: Stage, seconds: float) -> None:
        """Count one finished visit to ``stage`` that took ``seconds``, here and in the scope."""
        self._record(stage, seconds)
        scope = _scope.get()
        if scope is not None and scope is not self:
            scope._record(stage, seconds)  # noqa: SLF001 - the same class, one level of nesting

    def _record(self, stage: Stage, seconds: float) -> None:
        self._seconds[stage] = self._seconds.get(stage, 0.0) + seconds
        self._calls[stage] = self._calls.get(stage, 0) + 1
        if seconds > self._longest.get(stage, 0.0):
            self._longest[stage] = seconds
        if seconds > self._window_longest.get(stage, 0.0):
            self._window_longest[stage] = seconds

    def take_window_longest(self) -> dict[Stage, float]:
        """The longest visit to each stage since this was last called, and start a new window.

        One reader: the progress line. The cumulative maximum in a snapshot cannot be
        subtracted, so a window's worst case has to be kept apart and cleared when read. A visit
        still open counts for the time it has taken so far, so a stall in progress shows as the
        maximum it already is rather than as zero until it ends.
        """
        window, self._window_longest = self._window_longest, {}
        now = time.perf_counter()
        for stage, started in self._open.values():
            window[stage] = max(window.get(stage, 0.0), now - started)
        return window

    def begin(self, stage: Stage) -> _Visit:
        """Open a visit. Close it with :meth:`end`, or use :meth:`timing`."""
        now = time.perf_counter()
        scope = _scope.get()
        visit = _Visit(stage, now, None if scope is self else scope)
        self._open[visit] = (stage, now)
        if visit.scope is not None:
            visit.scope.begin_scoped(visit)
        return visit

    def begin_scoped(self, visit: _Visit) -> None:
        self._open[visit] = (visit.stage, visit.started)

    def end(self, visit: _Visit) -> None:
        """Close a visit opened by :meth:`begin`, once."""
        if self._open.pop(visit, None) is None:
            return
        elapsed = time.perf_counter() - visit.started
        self._record(visit.stage, elapsed)
        if visit.scope is not None:
            visit.scope.end_scoped(visit, elapsed)

    def end_scoped(self, visit: _Visit, elapsed: float) -> None:
        if self._open.pop(visit, None) is not None:
            self._record(visit.stage, elapsed)

    @contextmanager
    def timing(self, stage: Stage) -> Generator[None]:
        """Time the block, whether it returns, raises or is canceled."""
        visit = self.begin(stage)
        try:
            yield
        finally:
            self.end(visit)

    @contextmanager
    def scoped(self) -> Generator[None]:
        """Make visits begun in this context, and in tasks started from it, also count here."""
        token = _scope.set(self)
        try:
            yield
        finally:
            _scope.reset(token)

    def snapshot(self) -> Mapping[Stage, StageTime]:
        """Totals now, with visits still open counted for the time they have taken so far."""
        now = time.perf_counter()
        seconds = dict(self._seconds)
        for stage, started in self._open.values():
            seconds[stage] = seconds.get(stage, 0.0) + (now - started)
        return {
            stage: StageTime(
                seconds=total,
                calls=self._calls.get(stage, 0),
                longest=max(self._longest.get(stage, 0.0), self._open_longest(stage, now)),
            )
            for stage, total in seconds.items()
        }

    def _open_longest(self, stage: Stage, now: float) -> float:
        return max((now - at for kind, at in self._open.values() if kind is stage), default=0.0)

    @staticmethod
    def since(
        now: Mapping[Stage, StageTime], base: Mapping[Stage, StageTime]
    ) -> dict[Stage, StageTime]:
        """What happened between two snapshots, leaving out stages that did not run."""
        delta: dict[Stage, StageTime] = {}
        for stage, after in now.items():
            before = base.get(stage, StageTime())
            if after.calls > before.calls or after.seconds > before.seconds:
                delta[stage] = StageTime(
                    seconds=after.seconds - before.seconds,
                    calls=after.calls - before.calls,
                    longest=after.longest,
                )
        return delta


@dataclass(frozen=True, eq=False, slots=True)
class _Visit:
    """One open visit to a stage. Compared by identity, so each is its own key."""

    stage: Stage
    started: float
    scope: StageTimings | None


class CountedLock:
    """An :class:`asyncio.Lock` that says how many holders it has had at once.

    Wrapped rather than subclassed so that what is passed to
    :func:`~manicule.ingest.embedding.embed_or_reuse` is exactly an async context manager and
    nothing more — the scope of that lock is argued at its call site, and widening the object
    would invite widening the scope.

    A peak above one is the whole of "the embedder was called concurrently", so this is the
    measurement that assertion is made from rather than a proxy for it.

    Given ``timings`` it also records how long each holder queued behind another and how long it
    held the lock, under the two stages named, which is the difference between an accelerator
    that is the limit and one that is waiting for work.
    """

    def __init__(
        self,
        name: str,
        timings: StageTimings | None = None,
        *,
        wait: Stage = Stage.EMBED_LOCK_WAIT,
        held: Stage = Stage.EMBED_LOCK_HELD,
    ) -> None:
        self.gauge = Gauge(name)
        self._lock = asyncio.Lock()
        self._timings = timings
        self._wait_stage = wait
        self._held_stage = held
        self._held: _Visit | None = None

    async def __aenter__(self) -> None:
        waiting = self._timings.begin(self._wait_stage) if self._timings is not None else None
        try:
            await self._lock.acquire()
        finally:
            if waiting is not None and self._timings is not None:
                self._timings.end(waiting)
        if self._timings is not None:
            self._held = self._timings.begin(self._held_stage)
        self.gauge.active += 1
        self.gauge.peak = max(self.gauge.peak, self.gauge.active)

    async def __aexit__(self, *_: object) -> None:
        self.gauge.active -= 1
        if self._timings is not None and self._held is not None:
            self._timings.end(self._held)
        self._held = None
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
    """Top-level source identities accepted; durably committed in journal mode."""

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

    timings: Mapping[Stage, StageTime] = field(default_factory=dict[Stage, StageTime])
    """Where this run's time went, per stage, summed across workers. Empty when nothing ran.

    **Process-wide, like the peaks above:** the totals belong to the pipeline, so a re-parse sweep
    running beside a sync is counted in this run's too. Subtracting what was already there when
    the run began is as far as attribution goes.
    """

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
            "timings": {
                stage: {
                    "seconds": round(spent.seconds, 3),
                    "calls": spent.calls,
                    "longest": round(spent.longest, 3),
                }
                for stage, spent in sorted(self.timings.items())
            },
        }


__all__ = [
    "Conveyor",
    "CountedLock",
    "Gauge",
    "QueueReport",
    "Stage",
    "StageReport",
    "StageTime",
    "StageTimings",
]
