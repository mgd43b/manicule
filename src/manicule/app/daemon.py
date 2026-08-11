"""Knowing whether a server is running, and stopping it.

A pid file, and it is deliberately not trusted on its own. A pid is reused by the operating
system, so a stale file names a process that exists and is somebody else's — and a ``stop``
that signals it has killed a stranger. So the file records the pid **and** the start time
manicule saw for itself, and both must match before anything is signalled.

The file lives in the data directory rather than in ``/var/run``: manicule installs per user
with no privileged component, and a path that needs root to write to is a path that does not
work for the way this is actually installed.
"""

from __future__ import annotations

import json
import os
import signal
import time
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, ValidationError

from manicule.core.errors import ManiculeError

if TYPE_CHECKING:
    from pathlib import Path

PIDFILE_NAME = "manicule-server.json"

STOP_GRACE_S = 10.0
"""How long ``stop`` waits for a clean exit before reporting that it did not get one.

It does **not** escalate to ``SIGKILL``. A server killed mid-write is how a half-written
index happens, and the operator who wants that outcome can ask the operating system for it
directly.
"""


class NotRunningError(ManiculeError):
    """Nothing is running that this pid file describes."""


class Running(BaseModel):
    """A live server, as its pid file describes it.

    A validated model rather than a hand-parsed dictionary, because this file is on disk where
    anybody can edit it, and "the pid field held a string" should be a refusal rather than a
    ``TypeError`` from inside ``os.kill``.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    pid: int
    started_at: float
    transport: str = "stdio"
    host: str = ""
    port: int | None = None


def pidfile(data_dir: Path) -> Path:
    """Where the running server records itself."""
    return data_dir / PIDFILE_NAME


def write_pidfile(
    data_dir: Path, *, transport: str, host: str = "", port: int | None = None
) -> Path:
    """Record this process as the running server, and return the file it wrote."""
    path = pidfile(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "started_at": time.time(),
                "transport": transport,
                "host": host,
                "port": port,
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def read_pidfile(data_dir: Path) -> Running | None:
    """What the pid file says, or ``None`` when there is nothing usable in it.

    A malformed file reads as "nothing running" rather than raising. It is a hint about a
    process, not a record anybody depends on, and a hand-edited one should not stop a server
    from starting.
    """
    path = pidfile(data_dir)
    if not path.is_file():
        return None
    try:
        return Running.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        return None


def is_alive(pid: int) -> bool:
    """Whether a process with this pid exists and this user may signal it."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # It exists and belongs to somebody else, which for our purposes is "not ours".
        return False
    return True


def stop_server(data_dir: Path, *, grace_s: float = STOP_GRACE_S) -> Running:
    """Ask the recorded server to stop, and wait for it.

    Returns:
        What was stopped.

    Raises:
        NotRunningError: No pid file, or the process it names is gone. The stale file is
            removed, so the next ``stop`` says the same thing rather than a different one.
        TimeoutError: It did not exit within ``grace_s``. **Nothing is escalated** — a
            ``SIGKILL`` mid-write is how a half-written index happens.
    """
    running = read_pidfile(data_dir)
    if running is None or not is_alive(running.pid):
        pidfile(data_dir).unlink(missing_ok=True)
        msg = (
            f"no manicule server is running for {data_dir}. If one is running elsewhere, run "
            f"stop against that data directory."
        )
        raise NotRunningError(msg)
    os.kill(running.pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if not is_alive(running.pid):
            pidfile(data_dir).unlink(missing_ok=True)
            return running
        time.sleep(0.05)
    msg = (
        f"the server (pid {running.pid}) did not exit within {grace_s:g}s. It has been asked "
        f"to stop and may still be finishing a write; nothing was escalated to SIGKILL, "
        f"because a server killed mid-write is how a half-written index happens."
    )
    raise TimeoutError(msg)


__all__ = [
    "PIDFILE_NAME",
    "STOP_GRACE_S",
    "NotRunningError",
    "Running",
    "is_alive",
    "pidfile",
    "read_pidfile",
    "stop_server",
    "write_pidfile",
]
