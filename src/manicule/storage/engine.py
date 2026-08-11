"""Engine construction, connection configuration and the data-directory layout.

The pragmas here are not tuning. Two of them are correctness, and one of those is the single
most common way a SQLite schema full of ``REFERENCES`` clauses turns out to enforce nothing.
"""

from __future__ import annotations

import os
import sqlite3
import stat
from pathlib import Path
from typing import TYPE_CHECKING, Final

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

if TYPE_CHECKING:
    from sqlalchemy.engine.interfaces import DBAPIConnection
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
"async SQLAlchemy" does not serialise writers. Without it, concurrent work fails immediately
with ``SQLITE_BUSY`` rather than waiting.
"""


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
        modes do not apply, because ``st_mode`` is synthesised there and would report a
        healthy directory as world-readable.

    Raises:
        OSError: ``path`` cannot be stat'ed. Left to the caller: "the data directory cannot
            be examined" is a different diagnosis from "its modes are wrong".
    """
    if os.name != "posix":
        return 0
    return stat.S_IMODE(path.stat().st_mode) & EXPOSED_MODE_BITS


def create_engine(data_dir: Path, *, echo: bool = False) -> AsyncEngine:
    """Build the async engine for a data directory, with the pragmas attached.

    Args:
        data_dir: Root of the storage layout. Created if absent.
        echo: Log emitted SQL. For debugging only.

    Returns:
        An engine whose every connection has been configured by :data:`PRAGMAS`.
    """
    require_supported_sqlite()
    prepare_data_dir(data_dir)
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{database_path(data_dir)}",
        echo=echo,
        future=True,
    )
    attach_pragmas(engine)
    return engine


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
    "attach_pragmas",
    "create_engine",
    "database_path",
    "prepare_data_dir",
    "require_supported_sqlite",
    "session_factory",
]
