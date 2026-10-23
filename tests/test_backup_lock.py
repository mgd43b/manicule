"""The backup lock: a hot backup and a pass that removes data must never overlap.

``docs/storage.md`` §9.1. The lock is an ``flock`` on a file in the data directory because the
two halves live in different processes — ``backup`` is a command-line operation and the sweep
runs in the server — so the cross-process tests here are the ones that matter, and they hold the
lock from a real child process.
"""

from __future__ import annotations

import asyncio
import stat
import subprocess
import sys
import textwrap
import time
from typing import TYPE_CHECKING

import pytest

from manicule.storage.backup import BackupError, create_backup
from manicule.storage.backup_lock import (
    LOCK_FILENAME,
    MaintenanceBusyError,
    MaintenanceHold,
    backup_lock,
    lock_path,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine

_HOLDER = textwrap.dedent(
    """
    import fcntl, os, sys
    mode = fcntl.LOCK_EX if sys.argv[2] == "exclusive" else fcntl.LOCK_SH
    handle = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(handle, mode | fcntl.LOCK_NB)
    print("held", flush=True)
    sys.stdin.read()
    """
)


class _ChildHolder:
    """A separate process holding the lock until told to stop — a server, in the real case."""

    def __init__(self, data_dir: Path, mode: str) -> None:
        self._process = subprocess.Popen(  # noqa: S603 - fixed interpreter, fixed script
            [sys.executable, "-I", "-c", _HOLDER, str(lock_path(data_dir)), mode],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        assert self._process.stdout is not None
        assert self._process.stdout.readline().strip() == "held"

    def release(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
        self._process.wait(timeout=10)
        if self._process.stdout is not None:
            self._process.stdout.close()


@pytest.fixture
def child(tmp_path: Path) -> Iterator[type[_ChildHolder]]:
    """A factory for child holders, all stopped at teardown whatever the test did."""
    made: list[_ChildHolder] = []

    class Tracked(_ChildHolder):
        def __init__(self, data_dir: Path, mode: str) -> None:
            super().__init__(data_dir, mode)
            made.append(self)

    yield Tracked
    for holder in made:
        holder.release()


# --- the lock between holders in one process ---------------------------------------------------


async def test_a_backup_excludes_a_pass_that_removes_data(tmp_path: Path) -> None:
    async with backup_lock(tmp_path):
        hold = MaintenanceHold(tmp_path)
        reason = hold.acquire()

        assert reason
        assert "backup" in reason
        assert not hold.held
        assert hold.sweep_permitted() == reason, "a gate that was refused must say no"

    after = MaintenanceHold(tmp_path)
    assert after.acquire() == "", "the lock is free once the backup is done"
    after.release()


async def test_a_running_pass_excludes_a_backup_until_the_wait_runs_out(tmp_path: Path) -> None:
    with MaintenanceHold(tmp_path) as hold:
        assert hold.held
        started = time.monotonic()

        with pytest.raises(MaintenanceBusyError, match="maintenance pass"):
            async with backup_lock(tmp_path, wait_s=0.3, poll_s=0.05):
                pytest.fail("a backup got in beside a running pass")

        assert time.monotonic() - started >= 0.3, "it must wait the bound out, not refuse at once"
        assert time.monotonic() - started < 5.0


async def test_a_backup_that_waits_gets_in_when_the_pass_finishes(tmp_path: Path) -> None:
    hold = MaintenanceHold(tmp_path)
    assert hold.acquire() == ""
    asyncio.get_running_loop().call_later(0.2, hold.release)

    async with backup_lock(tmp_path, wait_s=10.0, poll_s=0.05):
        assert MaintenanceHold(tmp_path).acquire() != "", "and now it excludes the next pass"


async def test_waiting_for_the_lock_does_not_block_the_event_loop(tmp_path: Path) -> None:
    ticks = 0

    async def tick() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    ticker = asyncio.create_task(tick())
    try:
        with MaintenanceHold(tmp_path), pytest.raises(MaintenanceBusyError):
            async with backup_lock(tmp_path, wait_s=0.3, poll_s=0.05):
                pass
    finally:
        ticker.cancel()
    assert ticks >= 10, "other work stalled while a backup waited"


async def test_a_backup_canceled_while_waiting_leaves_nothing_held(tmp_path: Path) -> None:
    """The reason the wait polls rather than blocking in a thread.

    A thread already inside ``flock`` cannot be canceled: it would acquire the lock after its
    waiter was gone and nothing would ever release it, and every later pass would decline.
    """
    holder = MaintenanceHold(tmp_path)
    assert holder.acquire() == ""

    async def wait() -> None:
        async with backup_lock(tmp_path, wait_s=30.0, poll_s=0.02):
            pytest.fail("it should never have got in")

    waiting = asyncio.create_task(wait())
    await asyncio.sleep(0.1)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    holder.release()
    await asyncio.sleep(0.2)

    probe = MaintenanceHold(tmp_path)
    assert probe.acquire() == "", "a canceled backup leaked the lock"
    probe.release()
    async with backup_lock(tmp_path, wait_s=0.0):
        pass


async def test_passes_that_remove_data_do_not_exclude_each_other(tmp_path: Path) -> None:
    first, second = MaintenanceHold(tmp_path), MaintenanceHold(tmp_path)

    assert first.acquire() == ""
    assert second.acquire() == ""
    first.release()
    assert second.held, "one holder's release must not take another's lock"
    first.release()  # a second release is harmless
    assert second.held
    second.release()


def test_a_hold_that_was_never_taken_is_not_permission(tmp_path: Path) -> None:
    hold = MaintenanceHold(tmp_path)

    assert hold.sweep_permitted() != ""
    assert hold.acquire() == ""
    assert hold.sweep_permitted() == ""
    hold.release()
    assert hold.sweep_permitted() != ""


def test_the_lock_file_is_created_private(tmp_path: Path) -> None:
    with MaintenanceHold(tmp_path):
        pass

    mode = stat.S_IMODE((tmp_path / LOCK_FILENAME).stat().st_mode)
    assert mode == 0o600, oct(mode)


def test_a_lock_file_that_cannot_be_opened_declines_rather_than_raising(tmp_path: Path) -> None:
    """A pass that cannot prove no backup is running must not remove data — and must say why."""
    hold = MaintenanceHold(tmp_path / "no" / "such" / "directory")

    reason = hold.acquire()

    assert reason
    assert "cannot be opened" in reason
    assert not hold.held


# --- the lock between processes ----------------------------------------------------------------


async def test_a_pass_in_another_process_makes_a_backup_wait_and_then_refuse(
    tmp_path: Path, child: type[_ChildHolder]
) -> None:
    holder = child(tmp_path, "shared")

    with pytest.raises(MaintenanceBusyError):
        async with backup_lock(tmp_path, wait_s=0.3, poll_s=0.05):
            pytest.fail("a backup got in beside a pass in another process")

    holder.release()
    async with backup_lock(tmp_path, wait_s=2.0, poll_s=0.05):
        pass


async def test_a_backup_in_another_process_makes_a_pass_decline(
    tmp_path: Path, child: type[_ChildHolder]
) -> None:
    holder = child(tmp_path, "exclusive")

    hold = MaintenanceHold(tmp_path)
    assert "backup is running" in hold.acquire()
    assert not hold.held

    holder.release()
    assert hold.acquire() == ""
    hold.release()


# --- create_backup takes it ---------------------------------------------------------------------


async def test_create_backup_refuses_beside_a_running_pass(
    engine: AsyncEngine, data_dir: Path, tmp_path: Path
) -> None:
    target = tmp_path / "snapshot"

    with MaintenanceHold(data_dir):
        with pytest.raises(BackupError, match="maintenance pass"):
            await create_backup(engine, data_dir, target, lock_wait_s=0.2)
        assert not target.exists(), "nothing was written before the lock was held"


async def test_create_backup_holds_the_lock_for_the_whole_copy(
    engine: AsyncEngine, data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Observed from inside the copy, which is the only place it matters."""
    from manicule.storage import backup as backup_module  # noqa: PLC0415

    seen: list[str] = []
    original = backup_module._write_snapshot  # pyright: ignore[reportPrivateUsage]

    def observing(*args: object, **kwargs: object) -> dict[str, object]:
        probe = MaintenanceHold(data_dir)
        seen.append(probe.acquire())
        probe.release()
        return original(*args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(backup_module, "_write_snapshot", observing)

    manifest = await create_backup(engine, data_dir, tmp_path / "snapshot")

    assert len(seen) == 1
    assert "backup is running" in seen[0]
    assert LOCK_FILENAME not in {entry["path"] for entry in manifest["files"]}
    after = MaintenanceHold(data_dir)
    assert after.acquire() == "", "and it was given back afterwards"
    after.release()


async def test_create_backup_leaves_no_lock_file_in_a_directory_that_is_not_a_data_directory(
    engine: AsyncEngine, tmp_path: Path
) -> None:
    empty = tmp_path / "not-a-data-dir"
    empty.mkdir()

    with pytest.raises(BackupError, match="not a manicule data directory"):
        await create_backup(engine, empty, tmp_path / "snapshot")

    assert not (empty / LOCK_FILENAME).exists()


async def test_a_canceled_backup_keeps_the_lock_until_the_copy_has_stopped(
    engine: AsyncEngine, data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Canceling the await does not stop the thread, so it must not release the lock.

    A sweep that took the lock the moment the caller gave up would delete vectors or blobs the
    still-running copy is about to read.
    """
    import threading  # noqa: PLC0415

    from manicule.storage import backup as backup_module  # noqa: PLC0415

    started = threading.Event()
    finish = threading.Event()

    def blocked(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        started.set()
        finish.wait(timeout=30)
        return {}

    monkeypatch.setattr(backup_module, "_write_snapshot", blocked)
    task = asyncio.create_task(create_backup(engine, data_dir, tmp_path / "snapshot"))
    await asyncio.to_thread(started.wait, 30)

    task.cancel()
    task.cancel()  # and again: a second request must not abandon the copy either
    await asyncio.sleep(0.2)

    assert not task.done(), "the cancellation waited for the copy"
    probe = MaintenanceHold(data_dir)
    assert "backup is running" in probe.acquire(), "a pass still declines while the copy runs"
    probe.release()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    after = MaintenanceHold(data_dir)
    assert after.acquire() == "", "and the lock was given back once it stopped"
    after.release()
