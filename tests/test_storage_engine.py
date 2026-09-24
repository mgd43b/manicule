"""Opening a connection is one step, and a canceled caller cannot split it.

Callers are canceled routinely — a reader who goes away, a request past its deadline, an answer
that ends before its citation checks — and a cancellation that landed while the pool was opening
a connection stranded it: ``aiosqlite`` stopped its thread without closing the handle that
thread had just opened, or SQLAlchemy dropped an entry it had not finished setting up with the
connection inside it. Nothing held either, so nothing could close it.

These hold each step of opening a connection on the thread doing it, cancel an ordinary session
read while it is held, and then ask whether anything the read opened is still open once the
engine is disposed.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from typing import TYPE_CHECKING, Any, override

import pytest
from sqlalchemy import text
from sqlalchemy.pool import QueuePool

from manicule.storage import models
from manicule.storage.engine import session_factory

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncEngine


class _Hold:
    """Stops the thread opening a connection at one named step, once, until released."""

    def __init__(self, step: str, *, armed: bool = True) -> None:
        self.step = step
        self.armed = armed
        self.reached = threading.Event()
        self.release = threading.Event()

    def at(self, step: str) -> None:
        if self.armed and step.startswith(self.step) and not self.reached.is_set():
            self.reached.set()
            assert self.release.wait(timeout=5), "the test never released the held step"


def _instrument(
    monkeypatch: pytest.MonkeyPatch,
    *holds: _Hold,
    statements: list[tuple[int, str]] | None = None,
) -> None:
    """Route every ``sqlite3`` connection the engine opens past ``holds``.

    The driver's connect, each SQL function SQLAlchemy registers and each statement a
    connection runs are the steps a hold can name. ``statements`` records the last of those,
    keyed by the connection that ran them.
    """
    connect = sqlite3.connect

    def at(step: str) -> None:
        for hold in holds:
            hold.at(step)

    class Connection(sqlite3.Connection):
        @override
        def create_function(self, *args: Any, **kwargs: Any) -> None:
            at(f"function:{args[0]}")
            super().create_function(*args, **kwargs)

    def opened(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        at("connect")
        connection = connect(*args, factory=Connection, **kwargs)
        key = id(connection)

        def trace(statement: str) -> None:
            if statements is not None:
                statements.append((key, statement))
            at(statement)

        connection.set_trace_callback(trace)
        return connection

    monkeypatch.setattr(sqlite3, "connect", opened)


async def _read(engine: AsyncEngine) -> None:
    """An ordinary ORM read through a session: the shape nearly every store method has."""
    async with session_factory(engine)() as session:
        await session.get(models.Workspace, "default")


async def _canceled_while_held(engine: AsyncEngine, hold: _Hold, statement: str | None) -> None:
    """Start a read, cancel it once it reaches ``hold``, then let the held step go on."""

    async def read() -> None:
        if statement is None:
            await _read(engine)
            return
        async with session_factory(engine)() as session:
            await session.execute(text(statement))

    task = asyncio.create_task(read())
    assert await asyncio.to_thread(hold.reached.wait, 5), f"the read never reached {hold.step!r}"
    task.cancel()
    # Long enough for the cancellation to unwind as far as it can while the step is held, which
    # before the fix was all the way out: the read returned, and the step finished for nobody.
    await asyncio.wait({task}, timeout=0.1)
    hold.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize(
    ("step", "first"),
    [
        pytest.param("connect", False, id="the-driver-opening-its-handle"),
        pytest.param("PRAGMA journal_mode", False, id="the-pragmas"),
        pytest.param("function:regexp", False, id="sqlalchemy-registering-sql-functions"),
        pytest.param("PRAGMA read_uncommitted", True, id="sqlalchemy-inspecting-the-first"),
    ],
)
async def test_a_read_canceled_while_its_connection_opens_leaves_nothing_open(
    engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
    unclosed_connections: Callable[[], list[str]],
    step: str,
    first: bool,
) -> None:
    """Every await in opening a new connection is a place a cancellation used to land.

    The driver's connect left a ``sqlite3`` handle ``aiosqlite`` never handed over. The pragmas,
    the SQL functions SQLAlchemy registers, and the inspection it makes of an engine's first
    connection are all ``connect`` listeners, and SQLAlchemy dropped an entry whose listener was
    interrupted with its connection still in it. Only the pragmas were known about when this was
    first fixed — in one read, the blob store's — and the other steps leak identically.
    """
    if not first:
        await _read(engine)  # the engine's first connection, and its inspection, are done
    await engine.dispose()  # nothing idle, so the read has to open a connection of its own
    hold = _Hold(step)
    _instrument(monkeypatch, hold)

    await _canceled_while_held(engine, hold, statement=None)

    await engine.dispose()
    assert unclosed_connections() == [], f"canceled during {step!r}, the read left a connection"


async def test_a_connection_reopened_after_a_canceled_statement_is_not_stranded_either(
    engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
    unclosed_connections: Callable[[], list[str]],
) -> None:
    """The other way a connection is opened: an entry already in the pool, reopening.

    A statement canceled mid-flight makes SQLAlchemy close its connection and return the entry
    to the pool empty, and the entry opens a replacement on its next checkout — through the
    driver's connect, but not through the pool's creation of a new entry. So protecting new
    entries alone left this one: canceled there, ``aiosqlite`` stranded the replacement exactly
    as it had the original.
    """
    await engine.dispose()
    statement = _Hold("SELECT 'held'")
    reopening = _Hold("connect", armed=False)
    _instrument(monkeypatch, statement, reopening)
    await _canceled_while_held(engine, statement, statement="SELECT 'held'")
    pool = engine.pool
    assert isinstance(pool, QueuePool)
    assert pool.checkedin() == 1, "the canceled statement's entry is back in the pool, empty"

    reopening.armed = True
    await _canceled_while_held(engine, reopening, statement=None)

    await engine.dispose()
    assert unclosed_connections() == [], "canceled while reopening, the read left a connection"


async def test_every_connection_the_pool_opens_is_configured_once(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pragmas stay a ``connect`` listener, which now runs inside the step that opens a
    connection. Run twice or skipped there, nothing fails: a second ``foreign_keys = ON`` is
    harmless and a missing one is silent, which is exactly how a schema of ``REFERENCES`` comes
    to enforce nothing."""
    await engine.dispose()
    statements: list[tuple[int, str]] = []
    _instrument(monkeypatch, statements=statements)

    async with engine.connect() as one, engine.connect() as two, engine.connect() as three:
        for connection in (one, two, three):
            assert (await connection.execute(text("PRAGMA foreign_keys"))).scalar() == 1

    configured = [key for key, sql in statements if sql == "PRAGMA foreign_keys = ON"]
    assert len(set(configured)) == 3, "three connections were held at once, so three opened"
    assert sorted(configured) == sorted(set(configured)), "one connection was configured twice"
