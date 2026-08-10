"""What happens when the parent dies, and what stops two parents existing.

A killed worker is attributed to its document and the run continues. A killed *parent* leaves
documents in an in-flight status with nobody to finish them, and the repair needs no new
schema: ``status`` and ``updated_at`` already exist.

The sweep selects the in-flight states **by name**::

    WHERE status IN ('fetching', 'parsing', 'embedding')   -- never NOT IN (...)

An allowlist fails closed: a status added later is simply not swept until somebody adds it to
:data:`~manicule.core.content.IN_FLIGHT`. A denylist fails open, and the failure is a terminal
document requeued forever — ``container`` has zero chunks by design and
``no_extractable_text`` has zero chunks because there was nothing to find, and both look like
"stopped before embedding" to a careless ``WHERE`` clause.

Requeueing is cheap and safe because a document that is not ``indexed`` is not served: an
interrupted document is invisible rather than wrong.
"""

from __future__ import annotations

import contextlib
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Self

from manicule.core.content import IN_FLIGHT
from manicule.core.errors import InstanceLockedError

if TYPE_CHECKING:
    from manicule.ingest.ports import IngestStore

LOCK_FILENAME = "manicule.lock"
"""Beside the database, so the lock travels with the thing it protects."""

REQUEUE_DETAIL = "interrupted; requeued"


async def requeue_interrupted(store: IngestStore, *, stale_after_s: float = 3600.0) -> int:
    """Return documents stuck in flight to ``pending``, and say how many.

    Args:
        store: Where the documents are.
        stale_after_s: How long a document may sit in flight before it counts as abandoned.
            Comfortably above any per-document limit, because the failure this must not have
            is requeueing a document another process is working on right now — which is what
            the instance lock also exists to prevent, belt and braces.

    Returns:
        The number requeued. Zero is the healthy answer and is worth reporting: a non-zero
        count at every startup means something is killing the process.
    """
    cutoff = datetime.now(UTC) - timedelta(seconds=stale_after_s)
    return await store.requeue_stale(IN_FLIGHT, cutoff, detail=REQUEUE_DETAIL)


class InstanceLock:
    """An exclusive lock on a data directory, held for the process lifetime.

    The recovery sweep above, the vector sweep and the blob GC all assume a single writer.
    WAL permits several, so the assumption is enforced rather than hoped for. A second
    instance fails to start, naming the holder's PID, instead of starting and requeueing the
    first instance's in-flight documents out from under it.

    ``flock`` rather than a PID file whose contents are checked. A PID file survives a crash
    and has to be reasoned about; an ``flock`` is released by the kernel when the process
    holding it goes, whatever way it went. The PID is written *into* the file so the message
    can name a holder, but it is never what decides the answer — which is the difference
    between a lock and a note.
    """

    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / LOCK_FILENAME
        self._handle: int | None = None

    @property
    def path(self) -> Path:
        return self._path

    def acquire(self) -> Self:
        """Take the lock, or raise naming who has it.

        Raises:
            InstanceLockedError: Another process holds this data directory.
        """
        import fcntl  # noqa: PLC0415 - POSIX only, and manicule targets POSIX

        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        handle = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            holder = _read_pid(handle)
            os.close(handle)
            msg = (
                f"another manicule process is using {self._path.parent}"
                f"{f' (pid {holder})' if holder else ''}. One instance per data directory: the "
                f"recovery sweep, the vector sweep and the blob garbage collector all assume a "
                f"single writer, and a second instance would requeue the first one's in-flight "
                f"documents while it was still working on them. Stop the other process, or "
                f"point this one at a different data directory."
            )
            raise InstanceLockedError(msg) from exc
        os.ftruncate(handle, 0)
        os.write(handle, f"{os.getpid()}\n".encode())
        self._handle = handle
        return self

    def release(self) -> None:
        """Give the lock up. Safe to call when it was never taken."""
        if self._handle is None:
            return
        with contextlib.suppress(OSError):
            os.close(self._handle)
        self._handle = None

    def __enter__(self) -> Self:
        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb
        self.release()


def _read_pid(handle: int) -> str:
    with contextlib.suppress(OSError, ValueError):
        os.lseek(handle, 0, os.SEEK_SET)
        return os.read(handle, 32).decode("utf-8", "replace").strip()
    return ""  # pragma: no cover - unreadable lock contents


__all__ = ["LOCK_FILENAME", "REQUEUE_DETAIL", "InstanceLock", "requeue_interrupted"]
