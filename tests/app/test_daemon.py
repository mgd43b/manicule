"""Stopping a server that is running, and never stopping one that is somebody else's.

A pid file is a hint, not a record. The operating system reuses pids, so a stale file names a
process that exists and belongs to a stranger — and a ``stop`` that trusts it kills them.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import pytest

from manicule.app.daemon import (
    NotRunningError,
    is_alive,
    pidfile,
    read_pidfile,
    stop_server,
    write_pidfile,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_a_running_server_records_itself(tmp_path: Path) -> None:
    written = write_pidfile(tmp_path, transport="http", host="127.0.0.1", port=8765)
    assert written == pidfile(tmp_path)
    running = read_pidfile(tmp_path)
    assert running is not None
    assert running.pid == os.getpid()
    assert running.transport == "http"
    assert running.port == 8765


def test_the_pid_file_is_not_world_readable(tmp_path: Path) -> None:
    """It sits in the data directory, which holds the corpus. 0600 is what the rest uses."""
    written = write_pidfile(tmp_path, transport="stdio")
    assert written.stat().st_mode & 0o077 == 0


def test_no_pid_file_reads_as_nothing_running(tmp_path: Path) -> None:
    assert read_pidfile(tmp_path) is None


@pytest.mark.parametrize("body", ["not json at all", '{"pid": "seven"}', "[]", "{}"])
def test_a_malformed_pid_file_reads_as_nothing_running(tmp_path: Path, body: str) -> None:
    """A hand-edited file must not stop a server from starting, or crash the one stopping it.

    ``{"pid": "seven"}`` is the case worth naming: without validation it reaches ``os.kill``
    and fails there, from a stack that names nothing about pid files.
    """
    pidfile(tmp_path).write_text(body, encoding="utf-8")
    assert read_pidfile(tmp_path) is None


def test_stopping_when_nothing_is_running_says_so_and_clears_the_stale_file(
    tmp_path: Path,
) -> None:
    """Two identical ``stop`` invocations give the same answer, not two different ones."""
    pidfile(tmp_path).write_text(
        json.dumps({"pid": 999_999_999, "started_at": 0.0, "transport": "http"}),
        encoding="utf-8",
    )
    with pytest.raises(NotRunningError) as caught:
        stop_server(tmp_path)
    assert str(tmp_path) in str(caught.value)
    assert not pidfile(tmp_path).exists()

    with pytest.raises(NotRunningError):
        stop_server(tmp_path)


def test_a_pid_that_does_not_exist_is_not_alive() -> None:
    assert not is_alive(999_999_999)
    assert is_alive(os.getpid())
