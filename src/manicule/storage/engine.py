"""Engine construction, connection configuration and the data-directory layout.

The pragmas here are not tuning. Two of them are correctness, and one of those is the single
most common way a SQLite schema full of ``REFERENCES`` clauses turns out to enforce nothing.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import stat
import threading
import time
import weakref
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Concatenate, Final, override

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.pool import AsyncAdaptedQueuePool
from sqlalchemy.util import await_only, greenlet_spawn

from manicule.core.errors import InsecureTargetError, StorageBusyError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Coroutine
    from types import CoroutineType

    from sqlalchemy.engine.interfaces import DBAPIConnection, Dialect
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.pool import ConnectionPoolEntry

MINIMUM_SQLITE = (3, 35)
"""Below this, ``VACUUM INTO`` and the FTS5 options this schema uses are not all available."""

DATABASE_FILENAME = "manicule.db"
VECTORS_DIRNAME = "vectors"
BLOBS_DIRNAME = "blobs"
LOCK_FILENAME = "manicule.lock"

PRAGMAS: tuple[tuple[str, str], ...] = (
    ("journal_mode", "WAL"),
    ("foreign_keys", "ON"),
    ("busy_timeout", "5000"),
    ("synchronous", "NORMAL"),
    ("wal_autocheckpoint", "1000"),
)
"""Applied to **every** connection, in a ``connect`` listener.

``foreign_keys`` is per-connection and defaults to OFF. Setting it once at startup leaves
every connection the pool opens later silently skipping referential integrity — the schema
still declares its foreign keys, and nothing enforces them.

``busy_timeout`` matters because ``aiosqlite`` runs each connection on its own thread, so
"async SQLAlchemy" does not serialize writers. Without it, concurrent work fails immediately
with ``SQLITE_BUSY`` rather than waiting.
"""


SQLITE_BUSY_RETRY_DELAYS: Final = (0.0, 0.01, 0.05)
"""How a writer that lost the slot waits before asking again, and how many times.

Short and bounded. What this absorbs is another writer holding SQLite for a moment, which on
one machine is what contention looks like; a conflict that outlives three attempts is a
condition an operator has to know about rather than one to keep waiting on.
"""

_WRITER_ADMISSION: weakref.WeakKeyDictionary[Any, asyncio.Lock] = weakref.WeakKeyDictionary()
_WRITER_ADMISSION_LOCK = threading.Lock()


def writer_admission(engine: AsyncEngine) -> asyncio.Lock:
    """The one queue every managed writer to this database waits in.

    SQLite has a single writer, and "async SQLAlchemy" does not serialize anything — each
    connection runs on its own thread, so concurrent writers race for the slot instead of
    queueing for it. Losing that race is not always waitable either: a transaction that has
    already read and then writes is asking for an upgrade, and SQLite refuses those
    immediately rather than risk a deadlock, whatever ``busy_timeout`` says.

    So writers queue here first, before opening a transaction. **One lock per engine, shared by
    every module that writes**, because a second queue is not a queue: the acquisition journal
    waiting politely while blob bookkeeping writes whenever it likes is the arrangement that
    produced a run-ending `database is locked` on a five-row DELETE.
    """
    with _WRITER_ADMISSION_LOCK:
        admission = _WRITER_ADMISSION.get(engine)
        if admission is None:
            admission = asyncio.Lock()
            _WRITER_ADMISSION[engine] = admission
        return admission


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
        if calls > before_calls:
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
        self._queued = 0
        self._windows: weakref.WeakSet[WriterWindow] = weakref.WeakSet()

    def queue(self) -> None:
        self._queued += 1

    def acquired(self, operation: str, at: float) -> None:
        self._queued -= 1
        self._holder = (operation, at)

    def record(self, operation: str, *, waited: float, held: float, was_queued: bool) -> None:
        """Count one finished write. ``was_queued`` is whether it left while still waiting."""
        if was_queued:
            self._queued -= 1
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

    def totals(self) -> WriterTotals:
        """Running totals, for a window by subtraction with :func:`writer_totals_since`."""
        return {
            name: (visit.calls, visit.waited, visit.held) for name, visit in self.operations.items()
        }

    def current(self, now: float) -> tuple[str, float, int] | None:
        """``(operation holding the slot, seconds it has held it, writers queued)``, or ``None``."""
        if self._holder is None and not self._queued:
            return None
        name, at = self._holder if self._holder is not None else ("", now)
        return name, now - at, self._queued

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

    def take(self) -> dict[str, tuple[float, float]]:
        """``operation -> (longest wait, longest hold)`` since last asked, then start over."""
        taken, self._longest = self._longest, {}
        return taken


_LEDGERS: weakref.WeakKeyDictionary[Any, WriterLedger] = weakref.WeakKeyDictionary()


def writer_ledger(engine: AsyncEngine) -> WriterLedger:
    """The ledger for an engine, made on first use and gone with the engine."""
    with _WRITER_ADMISSION_LOCK:
        ledger = _LEDGERS.get(engine)
        if ledger is None:
            ledger = WriterLedger()
            _LEDGERS[engine] = ledger
        return ledger


@asynccontextmanager
async def writer_slot(engine: AsyncEngine, operation: str) -> AsyncGenerator[None]:
    """Hold the engine's one writer slot, and record how long getting it and keeping it took.

    :func:`writer_admission` with a name on it. Recorded when the block ends, however it ends,
    including a cancellation while queued, where only the wait is real and the hold is zero.
    """
    ledger = writer_ledger(engine)
    requested = time.perf_counter()
    acquired: float | None = None
    ledger.queue()
    try:
        async with writer_admission(engine):
            acquired = time.perf_counter()
            ledger.acquired(operation, acquired)
            yield
    finally:
        ended = time.perf_counter()
        if acquired is None:
            ledger.record(operation, waited=ended - requested, held=0.0, was_queued=True)
        else:
            ledger.record(
                operation, waited=acquired - requested, held=ended - acquired, was_queued=False
            )


def sqlite_busy(error: BaseException) -> bool:
    """Recognize SQLITE_BUSY through SQLAlchemy without retaining its SQL-shaped wrapper.

    Here rather than beside one caller because two now ask the same question — the acquisition
    journal, which retries, and re-embed's writer transactions, which refuse — and a second
    copy of this walk would be a second answer to "is this contention or corruption". The walk
    is needed at all because SQLAlchemy wraps the driver error and a retry policy keyed on
    message text is a retry policy that stops working when a driver rewords itself.
    """
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        code = getattr(current, "sqlite_errorcode", None)
        if isinstance(code, int) and (code & 0xFF) in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
            return True
        for related in (getattr(current, "orig", None), current.__cause__, current.__context__):
            if isinstance(related, BaseException):
                pending.append(related)
    return False


def admitted_writer[**P, R](
    operation: Callable[Concatenate[Any, P], Coroutine[Any, Any, R]],
) -> Callable[Concatenate[Any, P], CoroutineType[Any, Any, R]]:
    """Queue one whole-transaction write with every other managed writer, and retry a lost slot.

    For a method of a store that has an ``engine`` and opens exactly one transaction, which
    commits or rolls back before the method returns — so running it again after SQLITE_BUSY
    repeats nothing that was kept. **Never decorate a method that another decorated method
    awaits, or that is called while ``writer_admission`` is held**: the lock is not reentrant,
    and the second acquisition waits for the first for ever.

    Exhausting the retry raises :class:`~manicule.core.errors.StorageBusyError`, which the
    pipeline treats as an orderly, retryable stop that keeps the committed prefix. Without
    this a document write that lost the slot ended the run with the driver's own error, its SQL
    and bound values included.
    """

    @wraps(operation)
    async def admitted(store: Any, *args: P.args, **kwargs: P.kwargs) -> R:  # noqa: ANN401
        async with writer_slot(store.engine, operation.__name__):
            for delay in SQLITE_BUSY_RETRY_DELAYS:
                if delay:
                    await asyncio.sleep(delay)
                try:
                    return await operation(store, *args, **kwargs)
                except Exception as error:
                    if not sqlite_busy(error):
                        raise
            # Outside the handler: no SQL, bound value or path survives into a durable report.
            raise StorageBusyError("the committed documents are kept") from None

    admitted.queued_writer = True  # type: ignore[attr-defined]
    return admitted


class StorageLayoutError(Exception):
    """The data directory is not usable as one."""


def require_supported_sqlite() -> None:
    """Raise unless the linked SQLite can do what this schema needs.

    Python links against whatever the platform provides, and a build without FTS5 fails at
    the first query rather than at install. Checking here turns that into one clear message.

    Raises:
        StorageLayoutError: The version is too old, or FTS5 is not compiled in.
    """
    if sqlite3.sqlite_version_info < MINIMUM_SQLITE:
        wanted = ".".join(str(part) for part in MINIMUM_SQLITE)
        msg = (
            f"SQLite {sqlite3.sqlite_version} is too old; manicule needs {wanted} or newer. "
            f"Python links against the platform's library, so this is usually fixed by "
            f"installing a newer Python or a newer system SQLite."
        )
        raise StorageLayoutError(msg)

    probe = sqlite3.connect(":memory:")
    try:
        probe.execute("CREATE VIRTUAL TABLE fts5_probe USING fts5(x)")
    except sqlite3.OperationalError as error:
        msg = f"this SQLite was built without FTS5, which manicule uses for lexical search: {error}"
        raise StorageLayoutError(msg) from error
    finally:
        probe.close()


def database_path(data_dir: Path) -> Path:
    """Where the SQLite database lives inside a data directory."""
    return data_dir / DATABASE_FILENAME


def prepare_data_dir(data_dir: Path) -> Path:
    """Create the data directory and its subdirectories with restrictive permissions.

    ``0700`` for directories, and the mode is set explicitly rather than left to the
    operator's ``umask``. With original bytes retained, this directory holds the corpus
    itself — every source document, byte-identical to what the connector fetched — so a
    default that depends on the invoking shell is not a default.

    Args:
        data_dir: The root. Created if absent.

    Returns:
        The same path, for chaining.
    """
    for path in (
        data_dir,
        data_dir / VECTORS_DIRNAME,
        data_dir / BLOBS_DIRNAME,
    ):
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return data_dir


EXPOSED_MODE_BITS: Final = 0o077
"""Every group and other permission bit.

One constant for directories and files together, because the rule is the same for both:
``0700`` and ``0600`` are what :func:`prepare_data_dir` and the blob store write, and both
mean *nobody outside the owning account*. Testing for exactly ``0700`` would report ``0600``
on a directory — an unusable mode, but not an exposure — as the same problem as ``0755``.
"""


def exposure(path: Path) -> int:
    """The group and other permission bits set on ``path``. ``0`` means only its owner reaches it.

    **This is asked of the data directory, and only of the directory.** POSIX gates every
    read on the modes of every ancestor, so a directory nobody else can enter is a directory
    nobody else can read *through*, whatever the files inside it say. Walking the tree would
    cost one ``stat`` per retained document — a diagnostic proportional to the corpus — to
    report paths that are already unreachable.

    The data directory holds retained source bytes, so it is a verbatim copy of everything
    indexed (``docs/storage.md`` §7.1). :func:`prepare_data_dir` creates it ``0700``; a looser
    mode means an installer, a ``umask`` or a container run as root got there first, and the
    consequence is that the corpus is readable by whoever else has an account on the machine.

    Args:
        path: The directory to inspect. Must exist and be readable.

    Returns:
        The bits, so a caller can print the mode it objected to. Always ``0`` where POSIX
        modes do not apply, because ``st_mode`` is synthesized there and would report a
        healthy directory as world-readable.

    Raises:
        OSError: ``path`` cannot be stat'ed. Left to the caller: "the data directory cannot
            be examined" is a different diagnosis from "its modes are wrong".
    """
    if os.name != "posix":
        return 0
    return stat.S_IMODE(path.stat().st_mode) & EXPOSED_MODE_BITS


def secure_output_dir(target: Path, *, operation: str, allow_insecure: bool = False) -> None:
    """Create ``target`` ``0700``, then check that is what it actually is.

    **Asking for a mode is not having one.** ``Path.mkdir(mode=0o700, exist_ok=True)`` applies
    ``mode`` only when it creates the directory: a pre-existing group- or world-readable one is
    used exactly as found. That is not the rare path, it is the ordinary one — an operator who
    writes into the same place twice creates it once — so the mode is requested and then
    *verified*. Verifying also covers the case creation alone cannot: a default POSIX ACL on
    the parent can hand back a directory wider than the one that was asked for. Both halves of
    that were learned the expensive way in
    [#60](https://github.com/mgd43b/manicule/issues/60).

    One function for `backup` and `export` both, because it is one rule about one kind of
    directory: whatever manicule writes there is a complete copy of the corpus, retained source
    bytes and all (``docs/storage.md`` §7.1). ``doctor`` fails rather than warns on that
    exposure of the data directory; a second and third copy of the same bytes get the same
    answer, in the same words, because two versions of this check would eventually be two
    different checks.

    Only the target directory is examined, for the reason :func:`exposure` gives: POSIX gates
    every read on every ancestor, so a directory nobody else may enter is one nobody else may
    read *through*, whatever the permissions on the path leading to it.

    Args:
        target: Where the copy will be written. Created if absent, and so is any parent that
            has to be invented to reach it — those get ``0700`` as well, rather than the
            umask's answer.
        operation: What is being written, as the operator typed it — ``"backup"``,
            ``"export"``. It opens the message, so the refusal names the command that stopped.
        allow_insecure: Write into an exposed target anyway. The operator asked for it in so
            many words; manicule still refuses to be the one that decided.

    Raises:
        InsecureTargetError: The target is group- or world-readable and ``allow_insecure`` was
            not given, or its mode could not be read at all.
    """
    existed = target.exists()
    for ancestor in reversed(target.parents):
        # ``mkdir(parents=True)`` creates the ones above at the umask, so a target named
        # `/srv/exports/monday` would leave `/srv/exports` at 0755 while insisting its leaf be
        # 0700. Root-first, one at a time, so each one manicule invents gets manicule's mode.
        #
        # An ancestor already there raises ``FileExistsError`` and is left exactly as found,
        # modes included. That is the intended behavior rather than a tolerated one: an
        # existing directory is the operator's, and tightening it would be a mode change to a
        # path nobody asked about. Guarding with ``if not ancestor.exists()`` first would read
        # as the thing enforcing that, and would enforce nothing.
        with suppress(FileExistsError):
            ancestor.mkdir(mode=0o700)
    target.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        exposed = exposure(target)
    except OSError as error:
        msg = f"{operation} target {target} cannot be examined: {error}"
        raise InsecureTargetError(msg) from error
    if not exposed or allow_insecure:
        return
    if not existed:
        # Created a moment ago and refused a moment later: leave nothing behind, so the next
        # run meets the directory it would have met anyway.
        with suppress(OSError):
            target.rmdir()
    msg = (
        f"{operation} target {target} carries group or other permissions ({exposed:03o}), so "
        f"what manicule writes into it would be readable by accounts other than the one "
        f"running manicule. A {operation} holds the retained source bytes of every indexed "
        f"document, which makes this an exposure of the corpus rather than a tidiness problem. "
        f"Run `chmod 0700 {target}`, choose a target only this account can read, or pass "
        f"--allow-insecure-target to write it there knowingly."
    )
    raise InsecureTargetError(msg)


def create_engine(data_dir: Path, *, echo: bool = False) -> AsyncEngine:
    """Build the async engine for a data directory, with the pragmas attached.

    Args:
        data_dir: Root of the storage layout. Created if absent.
        echo: Log emitted SQL. For debugging only.

    Returns:
        An engine whose every connection has been configured by :data:`PRAGMAS`, and whose
        pool opens each one as a step no cancellation can interrupt (:class:`_WholeOpeningPool`).
    """
    require_supported_sqlite()
    prepare_data_dir(data_dir)
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{database_path(data_dir)}",
        echo=echo,
        future=True,
        poolclass=_WholeOpeningPool,
    )
    attach_pragmas(engine)
    event.listen(engine.sync_engine, "do_connect", _open_driver_connection)
    return engine


async def _to_the_end(work: asyncio.Future[Any]) -> asyncio.CancelledError | None:
    """Wait for ``work`` to finish, through however many cancellations arrive meanwhile.

    Returns the last of them, for the caller to raise once it has dealt with what the work
    produced, or ``None`` if nobody asked. ``asyncio.wait`` rather than ``shield``: it neither
    cancels the work nor raises its failure, so the outcome stays on the future to be read.
    """
    cancellation: asyncio.CancelledError | None = None
    while not work.done():
        try:
            await asyncio.wait({work})
        except asyncio.CancelledError as error:
            cancellation = error
    return cancellation


async def _uninterrupted[T](step: Callable[[], T], discard: Callable[[T], object]) -> T:
    """Run ``step`` to its end whatever happens to the caller, and discard it if they left.

    ``step`` is SQLAlchemy code that awaits through its greenlet bridge, so it runs in a
    greenlet and a task of its own, where the caller's cancellation cannot land. A caller
    canceled meanwhile waits for it, hands what it produced to ``discard`` — also run to its end
    — and only then raises the cancellation. That wait is time the step was taking anyway; not
    waiting would leave what it produces with nobody to close it.

    A failure of the step, or of the discard, is not reported to a canceled caller. The
    cancellation is what that caller asked for, and it is what it gets.
    """
    made = asyncio.ensure_future(greenlet_spawn(step))
    cancellation = await _to_the_end(made)
    if cancellation is None:
        return made.result()
    if not made.cancelled() and made.exception() is None:
        discarding = asyncio.ensure_future(greenlet_spawn(discard, made.result()))
        await _to_the_end(discarding)
        if not discarding.cancelled():
            discarding.exception()
    raise cancellation


def _close_entry(entry: ConnectionPoolEntry) -> None:
    entry.close()


def _close_connection(connection: DBAPIConnection) -> None:
    connection.close()


class _WholeOpeningPool(AsyncAdaptedQueuePool):
    """The pool SQLAlchemy chooses for a SQLite file, opening each new connection as one step.

    **A cancellation that lands while a connection is being opened strands it.** Opening one is
    several awaits: ``aiosqlite`` starting a thread that creates the ``sqlite3`` handle, then
    every ``connect`` listener — SQLAlchemy's own, which register SQL functions and on the
    first connection inspect the database, and :func:`_apply_pragmas`. Canceled inside
    ``aiosqlite``, the connection stops its thread without closing the handle that thread has
    just opened. Canceled inside a listener, the half-built pool entry is dropped with the
    connection in it. Neither is held by anything, so neither the session nor the engine's
    disposal can close it, and it surfaces later as a ``ResourceWarning`` from the garbage
    collector — a leaked file handle per canceled caller, and callers are canceled routinely:
    a reader who goes away, a request past its deadline, an answer that ends before its
    citation checks.

    So a new entry is built by :func:`_uninterrupted`: a caller canceled meanwhile waits for
    the entry to finish opening, closes it, and then raises. The pool's own accounting is
    untouched, because to the pool this is a connection attempt that raised.
    """

    @override
    def _create_connection(self) -> ConnectionPoolEntry:
        return await_only(_uninterrupted(super()._create_connection, _close_entry))


def _open_driver_connection(
    dialect: Dialect,
    _record: ConnectionPoolEntry,
    cargs: list[Any],
    cparams: dict[str, Any],
) -> DBAPIConnection:
    """Open the driver connection exactly as SQLAlchemy would, as one uninterrupted step.

    :class:`_WholeOpeningPool` covers a new entry. An entry already in the pool whose
    connection was invalidated — which is what a statement canceled mid-flight leaves — opens
    its replacement on its next checkout, outside that step. SQLAlchemy closes the replacement
    itself if a listener is interrupted, because the entry already holds it by then; but it
    cannot close a handle ``aiosqlite`` never handed over. This ``do_connect`` listener closes
    that gap for every connection this engine opens, whichever path asked for it.
    """
    return await_only(_uninterrupted(lambda: dialect.connect(*cargs, **cparams), _close_connection))


def _apply_pragmas(dbapi_connection: DBAPIConnection, _record: ConnectionPoolEntry) -> None:
    """Configure one freshly-opened connection.

    A module-level function rather than a closure so that it is registered once per engine
    and is visible to a reader looking for what configures a connection.
    """
    cursor = dbapi_connection.cursor()
    try:
        for name, value in PRAGMAS:
            # Both halves come from the PRAGMAS constant; no caller supplies either.
            cursor.execute(f"PRAGMA {name} = {value}")
    finally:
        cursor.close()


def attach_pragmas(engine: AsyncEngine) -> None:
    """Apply :data:`PRAGMAS` to every connection this engine opens."""
    event.listen(engine.sync_engine, "connect", _apply_pragmas)


def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Sessions that do not expire attributes on commit.

    ``expire_on_commit=False`` because the store converts ORM rows into frozen domain models
    and returns them; re-fetching every attribute after the commit that just wrote it is a
    round trip for data already in hand.
    """
    return async_sessionmaker(engine, expire_on_commit=False)


__all__ = [
    "BLOBS_DIRNAME",
    "DATABASE_FILENAME",
    "LOCK_FILENAME",
    "MINIMUM_SQLITE",
    "PRAGMAS",
    "VECTORS_DIRNAME",
    "StorageLayoutError",
    "WriterLedger",
    "WriterTotals",
    "WriterVisit",
    "WriterWindow",
    "admitted_writer",
    "attach_pragmas",
    "create_engine",
    "database_path",
    "prepare_data_dir",
    "require_supported_sqlite",
    "secure_output_dir",
    "session_factory",
    "writer_ledger",
    "writer_slot",
    "writer_totals_since",
]
