"""Watching a local directory. A watch event is not a sync, and conflating them costs.

| | Connector sync | Watch event |
|---|---|---|
| Trigger | schedule or command | filesystem notification |
| Scope | everything since a watermark | one path |
| Watermark | advanced on success | none |
| Reconcile | on a cadence | **never** |
| Deletion | detected by reconcile | reported directly by the event |

**Watch never drives reconciliation.** A watcher sees a subtree, and a subtree is exactly the
partial enumeration that reconciliation refuses to diff on. Deletions arrive here as explicit
delete events, which are trustworthy in a way that "did not appear in this walk" is not.

**Debounce, because editors do not write files the way the naive model assumes.** One logical
save commonly produces several events: many editors write to a temporary file and rename over
the target; some truncate and rewrite in place, briefly presenting a zero-byte file. Ingesting
on the first event indexes a partial or empty document — and it does so intermittently, which
is the hardest kind of wrong to reproduce.

So: coalesce per path over a window, ignore editor scratch names, and **re-``stat`` after the
window**, because the temporary file from a rename-based save is created and removed inside
it. A rename is treated as a delete and a create by path, and content-hash dedup then collapses
it back into one unchanged document, which is what it is.

``watchfiles`` is imported inside the function that needs it, so this module — and therefore
everything that imports the pipeline — costs nothing when nobody is watching.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable, Sequence

IGNORED_PATTERNS: tuple[str, ...] = (
    "*.swp",
    "*.swx",
    "*~",
    "*.tmp",
    ".#*",
    "#*#",
    "4913",
    ".DS_Store",
    "*.part",
    "*.crdownload",
)
"""Editor and downloader scratch names.

``4913`` is Vim's write-permission probe: it creates a file with that name, checks it, and
removes it. Without this list a single ``:w`` in Vim produces an ingest attempt for a file
that no longer exists by the time anybody looks.
"""


class Change(StrEnum):
    """What happened to a path. Deliberately the same three ``watchfiles`` reports."""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass(frozen=True, slots=True)
class WatchEvent:
    """One coalesced change to one path."""

    path: Path
    change: Change


def is_ignored(path: Path, patterns: Sequence[str] = IGNORED_PATTERNS) -> bool:
    """Whether a path is editor scratch rather than a document."""
    name = path.name
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def coalesce(events: Iterable[tuple[Change, Path]]) -> list[WatchEvent]:
    """Reduce a burst to one event per path, keeping the *last* change to it.

    Last rather than first, and that is the whole of it: a rename-based save appears as
    ``added`` then ``deleted`` for the temporary file and ``modified`` for the target, and only
    the final state of each path is a fact about the world. Order is preserved by first
    appearance so that a directory's documents are still ingested in a stable order.
    """
    latest: dict[Path, Change] = {}
    for change, path in events:
        latest[path] = change
    return [WatchEvent(path=path, change=change) for path, change in latest.items()]


def settle(events: Sequence[WatchEvent]) -> list[WatchEvent]:
    """Drop ignored paths, and re-check the filesystem before believing a creation.

    The re-``stat`` is what makes debouncing correct rather than merely calmer. A temporary
    file written and renamed away inside the window still produced ``added`` events; without
    checking again, the pipeline is handed a path that is gone, and reports a fetch failure for
    a document that never existed.

    A deletion is *not* re-checked. The event is the evidence, and a path that has since been
    recreated will arrive as its own later event.
    """
    settled: list[WatchEvent] = []
    for event in events:
        if is_ignored(event.path):
            continue
        if event.change is Change.DELETED:
            settled.append(event)
            continue
        if not event.path.is_file():
            continue
        settled.append(event)
    return settled


async def watch_directory(
    root: Path, *, debounce_s: float = 0.5, stop: object | None = None
) -> AsyncIterator[list[WatchEvent]]:
    """Yield settled batches of changes under ``root``.

    Batches rather than single events, because the interesting case is a burst: a
    ``git checkout`` across a large repository produces thousands of events in a second, and
    handing them over one at a time makes every consumer reimplement the coalescing that
    belongs here.

    Args:
        root: Directory to watch, recursively.
        debounce_s: Coalescing window.
        stop: An ``asyncio.Event``-shaped object ``watchfiles`` will watch for cancellation.

    Raises:
        RuntimeError: ``watchfiles`` is not installed. Named as a missing extra rather than
            surfacing an ``ImportError`` from a module the caller never mentioned.
    """
    try:
        # watchfiles ships `py.typed`, but `awatch` is annotated in terms of an anyio event
        # type that is itself partially unknown, so the call below reads as "partially
        # unknown" through no fault of this code. The suppression is scoped to the two lines
        # it applies to.
        from watchfiles import Change as FileChange  # noqa: PLC0415 - an optional extra
        from watchfiles import awatch  # noqa: PLC0415 # pyright: ignore[reportUnknownVariableType]
    except ImportError as exc:  # pragma: no cover - exercised where the extra is absent
        msg = (
            "directory watching needs the 'ingest' extra: install manicule[ingest]. Sync is "
            "unaffected — only `watch` requires it."
        )
        raise RuntimeError(msg) from exc

    mapping = {
        FileChange.added: Change.ADDED,
        FileChange.modified: Change.MODIFIED,
        FileChange.deleted: Change.DELETED,
    }
    async for batch in awatch(root, debounce=int(debounce_s * 1000), stop_event=stop):
        events = settle(coalesce((mapping[change], Path(path)) for change, path in batch))
        if events:
            yield events


__all__ = [
    "IGNORED_PATTERNS",
    "Change",
    "WatchEvent",
    "coalesce",
    "is_ignored",
    "settle",
    "watch_directory",
]
