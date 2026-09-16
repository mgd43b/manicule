"""Exercise durable request logging inside the shipped image across two containers.

Run ``first`` and then ``restart`` against the same fresh, dedicated data volume.  The marker
written by the first phase proves the second process saw the exact prior active log before it
started; no phase deletes or truncates anything in the volume.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO, TypedDict, cast

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

HOST = "127.0.0.1"
PORT = 8765
HTTP_URL = f"http://{HOST}:{PORT}"
MCP_URL = f"{HTTP_URL}/mcp/"
READY_TIMEOUT_SECONDS = 30.0
STOP_TIMEOUT_SECONDS = 15.0
APPEND_TIMEOUT_SECONDS = 5.0
MAX_BYTES = 4096
BACKUP_COUNT = 2
MARKER_NAME = "request-log-smoke.json"
LOG_RELATIVE = Path("logs/requests.jsonl")


class Marker(TypedDict):
    sha256: str
    size: int
    uid: int


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("first", "restart"), required=True)
    return parser.parse_args()


def _data_dir() -> Path:
    return Path(os.environ.get("MANICULE_DATA_DIR", "/data"))


def _server_environment(data_dir: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "MANICULE_DATA_DIR": str(data_dir),
            # Keep the image's enabled/default-path behavior; only shrink retention so
            # real traffic can exercise rotation without producing megabytes of logs.
            "MANICULE_LOGGING__MAX_BYTES": str(MAX_BYTES),
            "MANICULE_LOGGING__BACKUP_COUNT": str(BACKUP_COUNT),
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def _process_output(stream: TextIO) -> str:
    stream.flush()
    position = stream.tell()
    stream.seek(0)
    output = stream.read()
    stream.seek(position)
    return output


def _wait_until_ready(process: subprocess.Popen[str], output: TextIO) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            detail = _process_output(output).strip()
            raise RuntimeError(
                f"manicule exited before readiness with status {process.returncode}: {detail}"
            )
        try:
            with urllib.request.urlopen(  # noqa: S310 - fixed loopback HTTP URL
                f"{HTTP_URL}/healthz", timeout=1.0
            ) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.1)
    raise TimeoutError(f"manicule did not answer /healthz within {READY_TIMEOUT_SECONDS:.0f}s")


@contextmanager
def _serving(data_dir: Path) -> Generator[subprocess.Popen[str], None, None]:
    """Run the owned smoke-test process and always reap it."""
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as output:
        process = subprocess.Popen(  # noqa: S603 - fixed executable and arguments
            ["manicule", "serve", "--transport", "http", "--port", str(PORT)],  # noqa: S607
            env=_server_environment(data_dir),
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
        )
        body_failed = False
        try:
            _wait_until_ready(process, output)
            yield process
        except BaseException:
            body_failed = True
            raise
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=STOP_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=STOP_TIMEOUT_SECONDS)
            if process.returncode != 0:
                detail = _process_output(output).strip()
                if body_failed:
                    print(
                        f"manicule shutdown status {process.returncode}: {detail}",
                        file=sys.stderr,
                    )
                else:
                    raise RuntimeError(
                        f"manicule did not stop cleanly (status {process.returncode}): {detail}"
                    )


def _http_get(path: str = "/healthz") -> None:
    with urllib.request.urlopen(  # noqa: S310 - fixed loopback HTTP URL
        f"{HTTP_URL}{path}", timeout=5.0
    ) as response:
        if response.status != 200:
            raise AssertionError(f"GET {path} returned HTTP {response.status}")
        response.read()


async def _call_collection_list() -> None:
    transport = StreamableHttpTransport(MCP_URL)
    async with Client(transport) as client:
        result = await client.call_tool("collection_list", {})
    envelope = result.structured_content
    if not isinstance(envelope, dict) or envelope.get("ok") is not True:
        raise AssertionError(f"collection_list failed: {envelope!r}")


def _json_lines(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed: object = json.loads(line)
        if not isinstance(parsed, dict):
            raise TypeError(f"request record in {path} is not an object")
        records.append(cast("dict[str, Any]", parsed))
    if not records:
        raise AssertionError(f"request log {path} is empty")
    return records


def _require_private_owned(path: Path) -> None:
    actual_mode = stat.S_IMODE(path.stat().st_mode)
    if actual_mode != 0o600:
        raise AssertionError(f"{path} has mode {actual_mode:o}, expected 600")
    if path.stat().st_uid != os.getuid():
        raise AssertionError(f"{path} is not owned by uid {os.getuid()}")


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _wait_for_append(path: Path, *, initial: bytes, process: subprocess.Popen[str]) -> bytes:
    """Wait until response completion has reached the file handler's flush."""
    deadline = time.monotonic() + APPEND_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"manicule exited before writing the restart request (status {process.returncode})"
            )
        current = path.read_bytes()
        if len(current) > len(initial):
            if not current.startswith(initial):
                raise AssertionError("restart replaced rather than appended to the request log")
            return current
        time.sleep(0.05)
    raise TimeoutError("restart request did not reach the request log")


def _write_marker(path: Path, *, log_content: bytes) -> None:
    marker = {
        "sha256": _digest(log_content),
        "size": len(log_content),
        "uid": os.getuid(),
    }
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(marker, stream, separators=(",", ":"))
        stream.write("\n")


def _read_initial_log(marker_path: Path, log_path: Path) -> bytes:
    value: object = json.loads(marker_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"smoke marker {marker_path} is not an object")
    marker = cast("Marker", value)
    initial = log_path.read_bytes()
    if marker.get("uid") != os.getuid():
        raise AssertionError("the restart container does not use the first container's uid")
    if marker.get("size") != len(initial) or marker.get("sha256") != _digest(initial):
        raise AssertionError("the request log changed between the first and restart phases")
    return initial


def _first(data_dir: Path) -> None:
    marker_path = data_dir / MARKER_NAME
    log_path = data_dir / LOG_RELATIVE
    if (
        marker_path.exists()
        or log_path.exists()
        or list(log_path.parent.glob(f"{log_path.name}.*"))
    ):
        raise AssertionError("the first phase requires a fresh dedicated data volume")

    with _serving(data_dir):
        asyncio.run(_call_collection_list())

    records = _json_lines(log_path)
    if not any(record.get("surface") == "http" for record in records):
        raise AssertionError("the first phase produced no HTTP request record")
    if not any(
        record.get("surface") == "mcp" and record.get("operation") == "collection_list"
        for record in records
    ):
        raise AssertionError("the first phase produced no collection_list MCP record")
    _require_private_owned(log_path)
    if os.getuid() == 0:
        raise AssertionError("the shipped image smoke ran as root")
    initial = log_path.read_bytes()
    if len(initial) >= MAX_BYTES // 2:
        raise AssertionError("the first phase unexpectedly approached the rotation threshold")
    _write_marker(marker_path, log_content=initial)
    print(
        f"container request log first phase ok: records={len(records)} bytes={len(initial)} "
        f"uid={os.getuid()}"
    )


def _restart(data_dir: Path) -> None:
    marker_path = data_dir / MARKER_NAME
    log_path = data_dir / LOG_RELATIVE
    initial = _read_initial_log(marker_path, log_path)
    _require_private_owned(log_path)

    with _serving(data_dir) as process:
        _wait_for_append(log_path, initial=initial, process=process)
        for _ in range(200):
            _http_get()
            backups = list(log_path.parent.glob(f"{log_path.name}.*"))
            if {candidate.name for candidate in backups} >= {
                f"{log_path.name}.1",
                f"{log_path.name}.2",
            }:
                break
        else:
            raise AssertionError("request traffic did not produce two rotations")

    files = [log_path, *sorted(log_path.parent.glob(f"{log_path.name}.*"))]
    if len(files) > BACKUP_COUNT + 1:
        raise AssertionError(f"rotation retained {len(files)} files, expected at most 3")
    expected_names = {log_path.name, f"{log_path.name}.1", f"{log_path.name}.2"}
    if {path.name for path in files} != expected_names:
        raise AssertionError(f"unexpected rotated files: {[path.name for path in files]}")
    for path in files:
        _require_private_owned(path)
        _json_lines(path)
    oldest = log_path.with_name(f"{log_path.name}.2").read_bytes()
    if not oldest.startswith(initial):
        raise AssertionError("the rotated chain did not retain the pre-restart log prefix")
    print(
        f"container request log restart phase ok: files={len(files)} "
        f"active_records={len(_json_lines(log_path))} uid={os.getuid()}"
    )


def main() -> None:
    arguments = _arguments()
    data_dir = _data_dir()
    if arguments.phase == "first":
        _first(data_dir)
    else:
        _restart(data_dir)


if __name__ == "__main__":
    main()
