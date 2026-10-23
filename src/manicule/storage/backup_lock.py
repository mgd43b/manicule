"""The lock that keeps a hot backup from overlapping an operation that removes data.

A backup copies a data directory that a server may be writing, and ``docs/storage.md`` §9.1
makes it correct by ordering alone **only if nothing removes data while it copies**. Two
passes remove data: the vector sweep (which also purges soft-deleted documents) and blob
garbage collection. This module is what keeps those two, and any other pass that deletes
something a snapshot is about to read, out of a backup's way — and a backup out of theirs.

**Cross-process, because the two halves live in different processes.** ``backup`` is a
read-only operation (:data:`~manicule.app.dispatch.READ_ONLY_OPS`), so the command line runs
it in its own process while the server's scheduler runs the sweep in the server's. An
``asyncio.Lock`` would exclude nothing. The mechanism is an advisory ``flock`` on
``<data_dir>/backup.lock``, in the style of the instance lock
(:class:`~manicule.ingest.recovery.InstanceLock`): the kernel releases it when the holder goes,
however it went, so there is no stale-lock state to reason about.

**Two modes of one lock.**

- A **backup** takes it *exclusive* and waits for it, up to a bound
  (:func:`backup_lock`). The wait is a poll on the event loop rather than a blocking ``flock``
  in a worker thread: a thread that is already inside ``flock`` cannot be cancelled, so a
  backup abandoned while it waited would later acquire the lock with nobody left to release
  it, and every sweep after that would report itself blocked until the process exited.
- A **removing pass** takes it *shared* and never waits (:class:`MaintenanceHold`). It holds
  it for the whole pass, not for a probe at the start of it: a check that took the lock and
  dropped it leaves a window in which a backup starts halfway through a purge, which is the
  overlap the lock exists to prevent. When a backup holds the lock the pass declines with a
  plain reason, which the sweep already reports as ``blocked_by`` and the scheduler counts as
  ``blocked`` rather than as a failure. Shared holds do not exclude one another, so removing
  passes still overlap each other exactly as they always could.

**Every acquisition opens its own file descriptor.** ``flock`` locks belong to an open file
description, not to a process, which is what makes two holders in one process conflict
correctly — a server that is asked for a backup while its own sweep is running waits for the
sweep like any other process would — and what would make a shared descriptor wrong: one
holder's unlock would release another's lock.

**What it does not do.** Ingest and queries are not excluded; they never were meant to be.
It does not stop a copy of a directory a sync is writing from being a copy of something moving
(``docs/ingest.md`` §8.6 says so), and it is advisory: a process that does not ask is not
stopped. POSIX only, like the instance lock.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import os
import time
from typing import TYPE_CHECKING, Final, Self

from manicule.core.errors import ManiculeError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path
    from types import TracebackType

LOCK_FILENAME: Final = "backup.lock"
"""The lock's file name inside the data directory. A file of its own, never ``manicule.lock``.

The instance lock is held exclusively for a writer's whole life, so sharing its file would make
every backup refuse for as long as a server ran — the opposite of the point.
"""

DEFAULT_WAIT_S: Final = 60.0
"""How long a backup waits for a running pass to finish before it gives up.

A pass is bounded (``ingest.sweep_batch`` documents per sweep, one blob collection), so the
usual wait is seconds. The bound exists so that a pass wedged on something else reports itself
to the operator instead of hanging the command.
"""

POLL_INTERVAL_S: Final = 0.1
"""How often a waiting backup retries the lock."""


class MaintenanceBusyError(ManiculeError):
    """The lock could not be taken for exclusive use before the wait ran out.

    A :class:`~manicule.core.errors.ManiculeError` because it is a refusal manicule decided
    on rather than a defect; :func:`manicule.storage.backup.create_backup` re-raises it as the
    :class:`~manicule.storage.backup.BackupError` its callers already handle.
    """


def lock_path(data_dir: Path) -> Path:
    """Where the lock for ``data_dir`` lives."""
    return data_dir / LOCK_FILENAME


def _open(path: Path) -> int:
    """A fresh descriptor on the lock file, created ``0600`` and not inherited by children."""
    return os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)


class MaintenanceHold:
    """A shared, non-blocking hold on the backup lock, for a pass that removes data.

    Also a :class:`~manicule.ingest.sweeps.SweepGate`: :meth:`sweep_permitted` is empty exactly
    while the hold is held, so a sweep handed one asks the lock itself rather than a boolean
    somebody computed earlier.

    Use :meth:`acquire` and check what it returns; it never raises for a lock that is merely
    taken. Prefer :meth:`manicule.app.runtime.Runtime.maintenance_window`, which releases on
    every exit, to calling this directly.
    """

    def __init__(self, data_dir: Path) -> None:
        self._path = lock_path(data_dir)
        self._handle: int | None = None
        self._blocked_by = "the maintenance lock has not been taken"

    @property
    def held(self) -> bool:
        """Whether this hold currently owns the lock."""
        return self._handle is not None

    @property
    def blocked_by(self) -> str:
        """Why the last :meth:`acquire` failed. Empty while the lock is held."""
        return "" if self.held else self._blocked_by

    def acquire(self) -> str:
        """Take the lock shared without waiting.

        Returns:
            Empty when the lock is held. Otherwise a plain sentence saying what stopped it —
            a backup in progress, or a lock file this process cannot open or lock at all. The
            second is reported rather than raised for the same reason the first is: a pass
            that cannot prove no backup is running must not remove data, and declining is its
            designed way of saying so.
        """
        if self._handle is not None:
            return ""
        try:
            handle = _open(self._path)
        except OSError as exc:
            self._blocked_by = (
                f"the backup lock {self._path} cannot be opened ({exc.strerror or exc}), so "
                f"nothing can show that no backup is running"
            )
            return self._blocked_by
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(handle)
            self._blocked_by = (
                "a backup is running and holds the backup lock; this pass removes data a "
                "backup is copying, so it declines and will run at its next interval"
            )
            return self._blocked_by
        except OSError as exc:
            os.close(handle)
            self._blocked_by = (
                f"the backup lock {self._path} cannot be locked ({exc.strerror or exc}), so "
                f"nothing can show that no backup is running"
            )
            return self._blocked_by
        self._handle = handle
        return ""

    def release(self) -> None:
        """Give the lock up. Safe to call when it was never taken, and more than once."""
        handle, self._handle = self._handle, None
        if handle is not None:
            with contextlib.suppress(OSError):
                os.close(handle)

    def sweep_permitted(self) -> str:
        """Empty while the lock is held, otherwise what is in the way.

        The :class:`~manicule.ingest.sweeps.SweepGate` answer. A hold that was never acquired,
        or was released, is not permission to remove anything.
        """
        return self.blocked_by

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb
        self.release()


@contextlib.asynccontextmanager
async def backup_lock(
    data_dir: Path,
    *,
    wait_s: float = DEFAULT_WAIT_S,
    poll_s: float = POLL_INTERVAL_S,
) -> AsyncGenerator[None]:
    """Hold the backup lock exclusively, waiting up to ``wait_s`` for running passes to finish.

    Polls without blocking the event loop, and releases on every exit including cancellation.

    Args:
        data_dir: The directory whose lock to take.
        wait_s: How long to wait. ``0`` tries once.
        poll_s: How long to sleep between attempts.

    Raises:
        MaintenanceBusyError: A removing pass, or another backup, still held the lock when the
            wait ran out.
        OSError: The lock file could not be created or locked at all — a read-only directory,
            or a filesystem without ``flock``.
    """
    path = lock_path(data_dir)
    handle = _open(path)
    try:
        deadline = time.monotonic() + wait_s
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    msg = (
                        f"another backup or a maintenance pass (the scheduled vector sweep or blob "
                        f"collection) held {path} for the {wait_s:g}s this backup waited. A "
                        f"backup copies data those passes remove, so it will not run beside "
                        f"one; try again shortly"
                    )
                    raise MaintenanceBusyError(msg) from None
                await asyncio.sleep(poll_s)
                continue
            break
        yield
    finally:
        with contextlib.suppress(OSError):
            os.close(handle)


__all__ = [
    "DEFAULT_WAIT_S",
    "LOCK_FILENAME",
    "POLL_INTERVAL_S",
    "MaintenanceBusyError",
    "MaintenanceHold",
    "backup_lock",
    "lock_path",
]
