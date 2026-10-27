"""What the one writer slot has cost each kind of write, kept apart from the engine.

Its own module because the ingest pipeline reads it and the pipeline is imported with the
package, which may not load an implementation: this has no SQLAlchemy in it, and the engine it is
keyed by is only ever a key.
"""

from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass
from typing import Any

_LEDGER_LOCK = threading.Lock()


@dataclass(slots=True)
class WriterVisit:
    """What one named write operation has cost the writer queue, in total."""

    calls: int = 0
    waited: float = 0.0
    """Seconds spent queued behind other writers, summed over calls."""
    held: float = 0.0
    """Seconds spent holding the slot, summed over calls."""
    longest_wait: float = 0.0
    longest_held: float = 0.0


WriterTotals = dict[str, tuple[int, float, float]]
"""``operation -> (calls, seconds waited, seconds held)``."""


def writer_totals_since(now: WriterTotals, base: WriterTotals) -> WriterTotals:
    """What each operation did between two readings, leaving out those that did not run."""
    moved: WriterTotals = {}
    for name, (calls, waited, held) in now.items():
        before_calls, before_waited, before_held = base.get(name, (0, 0.0, 0.0))
        if calls > before_calls or waited > before_waited or held > before_held:
            moved[name] = (calls - before_calls, waited - before_waited, held - before_held)
    return moved


class WriterLedger:
    """How long each kind of write waited for, and then held, an engine's one writer slot.

    **Why it exists.** Every managed write runs under :func:`writer_admission`, so a pipeline's
    writes are one serial section, and "this stage took 50 ms" cannot say whether that was the
    write or the queue in front of it. The slot is held across the whole transaction, COMMIT
    included, where SQLite's autocheckpoint runs, so a slow volume stalls every waiter at once
    and shows up only as a long maximum: totals alone average it away. Hence the longest wait
    and the longest hold are kept beside the sums.

    **A write is recorded when it ends, so who holds the slot now is kept too.** A writer stalled
    inside COMMIT has not ended, and a ledger that only counted finished writes would show
    nothing for exactly that: :meth:`current` names the holder and how long it has held, and how
    many are queued behind it.

    Keyed by the operation's function name, a code identifier and never a value, so the ledger
    is as small as the set of write methods and says nothing about a document. Synchronous and
    unlocked, like the stage counters: nothing between reading a total and writing it back
    awaits.
    """

    def __init__(self) -> None:
        self.operations: dict[str, WriterVisit] = {}
        self._holder: tuple[str, float] | None = None
        self._waiting: dict[object, tuple[str, float]] = {}
        self._windows: weakref.WeakSet[WriterWindow] = weakref.WeakSet()

    def queue(self, operation: str, at: float) -> object:
        """A write starts waiting. The token is given back to :meth:`acquired` or :meth:`record`."""
        token = object()
        self._waiting[token] = (operation, at)
        return token

    def acquired(self, token: object, operation: str, at: float) -> None:
        self._waiting.pop(token, None)
        self._holder = (operation, at)

    def record(
        self, operation: str, *, waited: float, held: float, token: object | None = None
    ) -> None:
        """Count one finished write. A ``token`` still waiting means it left without the slot."""
        if token is not None:
            self._waiting.pop(token, None)
        else:
            self._holder = None
        visit = self.operations.setdefault(operation, WriterVisit())
        visit.calls += 1
        visit.waited += waited
        visit.held += held
        visit.longest_wait = max(visit.longest_wait, waited)
        visit.longest_held = max(visit.longest_held, held)
        for window in self._windows:
            window.note(operation, waited, held)

    def totals(self, now: float | None = None) -> WriterTotals:
        """Running totals, for a window by subtraction with :func:`writer_totals_since`.

        Given ``now``, a write still waiting or holding adds the time it has taken so far, and
        counts as a call only when it ends. Otherwise a hold that spans a window would be
        invisible in it and arrive whole in the next, and the ratio of held time to the window
        would stop meaning how busy the slot was.
        """
        totals: dict[str, list[float]] = {
            name: [visit.calls, visit.waited, visit.held] for name, visit in self.operations.items()
        }
        if now is not None:
            for operation, at in self._waiting.values():
                totals.setdefault(operation, [0, 0.0, 0.0])[1] += now - at
            if self._holder is not None:
                operation, at = self._holder
                totals.setdefault(operation, [0, 0.0, 0.0])[2] += now - at
        return {name: (int(calls), waited, held) for name, (calls, waited, held) in totals.items()}

    def open_longest(self, now: float) -> dict[str, tuple[float, float]]:
        """``operation -> (longest wait, longest hold)`` among writes that have not ended."""
        longest: dict[str, tuple[float, float]] = {}
        for operation, at in self._waiting.values():
            wait, held = longest.get(operation, (0.0, 0.0))
            longest[operation] = (max(wait, now - at), held)
        if self._holder is not None:
            operation, at = self._holder
            wait, held = longest.get(operation, (0.0, 0.0))
            longest[operation] = (wait, max(held, now - at))
        return longest

    def current(self, now: float) -> tuple[str, float, int] | None:
        """``(operation holding the slot, seconds it has held it, writers queued)``, or ``None``."""
        if self._holder is None and not self._waiting:
            return None
        name, at = self._holder if self._holder is not None else ("", now)
        return name, now - at, len(self._waiting)

    def window(self) -> WriterWindow:
        """A new window onto the longest wait and hold, for one reader to take from.

        One per reader, held weakly here, so two runs sharing an engine each see their own and
        neither clears the other's.
        """
        window = WriterWindow()
        self._windows.add(window)
        return window


class WriterWindow:
    """The longest wait and hold of each operation since the last :meth:`take`."""

    def __init__(self) -> None:
        self._longest: dict[str, tuple[float, float]] = {}

    def note(self, operation: str, waited: float, held: float) -> None:
        longest_wait, longest_held = self._longest.get(operation, (0.0, 0.0))
        self._longest[operation] = (max(longest_wait, waited), max(longest_held, held))

    def take(
        self, open_now: dict[str, tuple[float, float]] | None = None
    ) -> dict[str, tuple[float, float]]:
        """``operation -> (longest wait, longest hold)`` since last asked, then start over.

        ``open_now`` is the same for writes that have not ended, so a hold that is still going
        shows as the maximum it already is instead of as zero until it finishes.
        """
        taken, self._longest = self._longest, {}
        for operation, (wait, held) in (open_now or {}).items():
            longest_wait, longest_held = taken.get(operation, (0.0, 0.0))
            taken[operation] = (max(longest_wait, wait), max(longest_held, held))
        return taken


_LEDGERS: weakref.WeakKeyDictionary[Any, WriterLedger] = weakref.WeakKeyDictionary()


def writer_ledger(engine: object) -> WriterLedger:
    """The ledger for an engine, made on first use and gone with the engine."""
    with _LEDGER_LOCK:
        ledger = _LEDGERS.get(engine)
        if ledger is None:
            ledger = WriterLedger()
            _LEDGERS[engine] = ledger
        return ledger


__all__ = [
    "WriterLedger",
    "WriterTotals",
    "WriterVisit",
    "WriterWindow",
    "writer_ledger",
    "writer_totals_since",
]
