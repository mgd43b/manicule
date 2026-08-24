"""Bounded, read-only access to one commit in a local Git repository."""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from manicule.connectors.errors import BodyUnavailableError, ConnectorError
from manicule.connectors.site_routes import SiteRouteError, normalize_repository_path
from manicule.core.errors import ConfigError

__all__ = [
    "GitBlobTooLargeError",
    "GitObjectMissingError",
    "GitSourceError",
    "GitTreeEntry",
    "PinnedGitReader",
]

_OID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_ORDINARY_MODES: Final = frozenset({"100644", "100755"})
_HEADER_LIMIT = 256
_STDERR_LIMIT = 8 * 1024
_DEFAULT_TREE_BYTES = 32 * 1024 * 1024
_DEFAULT_TREE_ENTRIES = 100_000
_DEFAULT_TIMEOUT_S = 30.0
_DEFAULT_CLOSE_TIMEOUT_S = 2.0


class GitSourceError(ConnectorError):
    """The pinned repository could not provide a complete, trustworthy object view."""


class GitObjectMissingError(BodyUnavailableError):
    """An object from the pinned inventory is no longer available."""


class GitBlobTooLargeError(GitSourceError):
    """A blob exceeded the configured ceiling before its body was requested."""


class _OutputLimitError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class GitTreeEntry:
    """One exact entry from the pinned commit tree."""

    path: str
    mode: str
    object_type: str
    object_id: str
    size: int | None

    @property
    def ordinary_blob(self) -> bool:
        return self.object_type == "blob" and self.mode in _ORDINARY_MODES

    @property
    def symlink(self) -> bool:
        return self.mode == "120000"

    @property
    def submodule(self) -> bool:
        return self.mode == "160000" or self.object_type == "commit"


class PinnedGitReader:
    """Resolve once, then read only tree and blob objects named by that commit."""

    def __init__(
        self,
        repository: Path | str,
        *,
        revision: str = "HEAD",
        max_blob_bytes: int = 256 * 1024 * 1024,
        max_tree_bytes: int = _DEFAULT_TREE_BYTES,
        max_tree_entries: int = _DEFAULT_TREE_ENTRIES,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        close_timeout_s: float = _DEFAULT_CLOSE_TIMEOUT_S,
        executable: str | None = None,
    ) -> None:
        if not revision or "\0" in revision or "\n" in revision:
            raise ConfigError("Git revision must be one non-empty command argument")
        if max_blob_bytes < 1 or max_tree_bytes < 1 or max_tree_entries < 1:
            raise ConfigError("Git reader byte and entry limits must be positive")
        self._repository_input = Path(repository).expanduser()
        self._repository: Path | None = None
        self._revision = revision
        self._max_blob_bytes = max_blob_bytes
        self._max_tree_bytes = max_tree_bytes
        self._max_tree_entries = max_tree_entries
        self._timeout_s = max(0.1, timeout_s)
        self._close_timeout_s = max(0.1, close_timeout_s)
        self._executable = executable
        self._commit: str | None = None
        self._entries: tuple[GitTreeEntry, ...] = ()
        self._entries_by_path: dict[str, GitTreeEntry] = {}
        self._blob_sizes: dict[str, int] = {}
        self._batch: asyncio.subprocess.Process | None = None
        self._batch_lock = asyncio.Lock()
        self._batch_stderr: bytearray = bytearray()
        self._batch_stderr_task: asyncio.Task[None] | None = None

    @property
    def repository(self) -> Path:
        if self._repository is None:
            raise RuntimeError("Git reader has not been set up")
        return self._repository

    @property
    def commit(self) -> str:
        if self._commit is None:
            raise RuntimeError("Git reader has not been set up")
        return self._commit

    @property
    def entries(self) -> tuple[GitTreeEntry, ...]:
        return self._entries

    async def setup(self, *, content_root: str = ".") -> tuple[GitTreeEntry, ...]:
        """Pin a commit and enumerate one complete tree prefix in raw Git order."""
        try:
            repository = self._repository_input.resolve(strict=True)
        except OSError as exc:
            raise ConfigError("configured Git repository is unavailable") from exc
        if not repository.is_dir():
            raise ConfigError("configured Git repository must be a directory")
        executable = self._executable or shutil.which("git")
        if executable is None:
            raise ConfigError("git is not installed or is not on PATH")
        self._repository = repository
        self._executable = executable
        requested = f"{self._revision}^{{commit}}"
        resolved = (
            await self._run_git(
                "rev-parse",
                "--verify",
                "--end-of-options",
                requested,
                stdout_limit=128,
            )
        ).strip()
        try:
            commit = resolved.decode("ascii", errors="strict")
        except UnicodeDecodeError as exc:  # pragma: no cover - Git emits ASCII object ids
            raise GitSourceError("Git returned an invalid commit object id") from exc
        self._validate_oid(commit)
        self._commit = commit
        root = normalize_repository_path(content_root, allow_root=True)
        arguments = ["ls-tree", "-r", "-z", "-l", "--full-tree", commit]
        if root != ".":
            arguments.extend(("--", root))
        raw = await self._run_git(*arguments, stdout_limit=self._max_tree_bytes)
        entries = self._parse_tree(raw)
        self._install_entries(entries)
        return entries

    async def lookup(self, path: str) -> GitTreeEntry | None:
        """Look up an exact pinned path, including a manifest outside ``content_root``."""
        normalized = normalize_repository_path(path)
        cached = self._entries_by_path.get(normalized)
        if cached is not None:
            return cached
        raw = await self._run_git(
            "ls-tree",
            "-z",
            "-l",
            "--full-tree",
            self.commit,
            "--",
            normalized,
            stdout_limit=min(self._max_tree_bytes, 2 * (len(normalized.encode()) + 256)),
        )
        if not raw:
            return None
        entries = self._parse_tree(raw)
        exact = next((entry for entry in entries if entry.path == normalized), None)
        if exact is not None:
            self._remember_entry(exact)
        return exact

    def ordinary_entries(self) -> tuple[GitTreeEntry, ...]:
        return tuple(entry for entry in self._entries if entry.ordinary_blob)

    async def read_entry(
        self, entry: GitTreeEntry, *, max_bytes: int | None = None
    ) -> bytes:
        known = self._entries_by_path.get(entry.path)
        if known != entry or not entry.ordinary_blob:
            raise GitSourceError("requested path is not an ordinary blob in the pinned inventory")
        return await self.read_blob(entry.object_id, max_bytes=max_bytes)

    async def read_blob(self, object_id: str, *, max_bytes: int | None = None) -> bytes:
        """Read one inventory-owned blob, refusing its known size before requesting its body."""
        self._validate_oid(object_id)
        size = self._blob_sizes.get(object_id)
        if size is None:
            raise GitObjectMissingError("requested object is not in the pinned blob inventory")
        limit = self._max_blob_bytes if max_bytes is None else min(max_bytes, self._max_blob_bytes)
        if size > limit:
            raise GitBlobTooLargeError(f"Git blob exceeds the configured {limit}-byte limit")
        async with self._batch_lock:
            process = await self._batch_process()
            try:
                return await self._read_from_batch(process, object_id, size)
            except asyncio.CancelledError:
                await asyncio.shield(self._discard_batch())
                raise
            except (TimeoutError, asyncio.IncompleteReadError) as exc:
                await self._discard_batch()
                raise GitObjectMissingError("Git could not finish the pinned blob read") from exc
            except Exception:
                await self._discard_batch()
                raise

    async def _read_from_batch(
        self, process: asyncio.subprocess.Process, object_id: str, size: int
    ) -> bytes:
        stdin = cast("asyncio.StreamWriter", process.stdin)
        stdout = cast("asyncio.StreamReader", process.stdout)
        stdin.write(f"{object_id}\n".encode("ascii"))
        await stdin.drain()
        header = await asyncio.wait_for(stdout.readline(), timeout=self._timeout_s)
        if len(header) > _HEADER_LIMIT:
            raise GitSourceError("Git returned an oversized batch header")
        if not header:
            raise GitObjectMissingError("Git ended before returning the pinned object")
        if header.rstrip().endswith(b" missing"):
            raise GitObjectMissingError("the pinned Git blob is no longer available")
        returned_id, object_type, raw_size = self._parse_batch_header(header)
        if returned_id != object_id or object_type != "blob" or raw_size != size:
            raise GitSourceError("Git returned an object different from the pinned inventory")
        body = await asyncio.wait_for(stdout.readexactly(size + 1), timeout=self._timeout_s)
        if body[-1:] != b"\n":
            raise GitSourceError("Git returned a malformed batch body")
        return body[:-1]

    async def aclose(self) -> None:
        async with self._batch_lock:
            await self._discard_batch()

    async def __aenter__(self) -> PinnedGitReader:
        return self

    async def __aexit__(self, *exc: object) -> None:
        del exc
        await self.aclose()

    def _install_entries(self, entries: tuple[GitTreeEntry, ...]) -> None:
        self._entries = entries
        self._entries_by_path.clear()
        self._blob_sizes.clear()
        for entry in entries:
            self._remember_entry(entry)

    def _remember_entry(self, entry: GitTreeEntry) -> None:
        self._entries_by_path[entry.path] = entry
        if entry.ordinary_blob and entry.size is not None:
            self._blob_sizes[entry.object_id] = entry.size

    def _parse_tree(self, raw: bytes) -> tuple[GitTreeEntry, ...]:
        entries: list[tuple[bytes, GitTreeEntry]] = []
        for encoded in raw.split(b"\0"):
            if not encoded:
                continue
            if len(entries) >= self._max_tree_entries:
                raise GitSourceError(
                    f"Git tree exceeds the configured {self._max_tree_entries}-entry limit"
                )
            try:
                header, raw_path = encoded.split(b"\t", 1)
                mode, object_type, raw_id, raw_size = header.split(b" ", 3)
                path = raw_path.decode("utf-8", errors="strict")
                entry = GitTreeEntry(
                    path=normalize_repository_path(path),
                    mode=mode.decode("ascii"),
                    object_type=object_type.decode("ascii"),
                    object_id=raw_id.decode("ascii"),
                    size=(
                        None if raw_size.strip() == b"-" else int(raw_size.strip())
                    ),
                )
                self._validate_oid(entry.object_id)
            except (ValueError, UnicodeDecodeError, SiteRouteError) as exc:
                raise GitSourceError("Git returned a malformed or unsupported tree entry") from exc
            entries.append((raw_path, entry))
        entries.sort(key=lambda item: item[0])
        return tuple(entry for _, entry in entries)

    @staticmethod
    def _parse_batch_header(header: bytes) -> tuple[str, str, int]:
        try:
            raw_id, raw_type, raw_size = header.rstrip(b"\n").split(b" ", 2)
            return raw_id.decode("ascii"), raw_type.decode("ascii"), int(raw_size)
        except (ValueError, UnicodeDecodeError) as exc:
            raise GitSourceError("Git returned a malformed batch header") from exc

    @staticmethod
    def _validate_oid(object_id: str) -> None:
        if not _OID.fullmatch(object_id):
            raise GitSourceError("Git returned or was asked for an invalid object id")

    async def _batch_process(self) -> asyncio.subprocess.Process:
        if self._batch is not None and self._batch.returncode is None:
            return self._batch
        process = await asyncio.create_subprocess_exec(
            self._git(),
            "-C",
            str(self.repository),
            "cat-file",
            "--batch",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._environment(),
            start_new_session=os.name == "posix",
        )
        self._batch = process
        self._batch_stderr.clear()
        stderr = cast("asyncio.StreamReader", process.stderr)
        self._batch_stderr_task = asyncio.create_task(
            self._drain_batch_stderr(stderr), name="git-batch-stderr"
        )
        return process

    async def _drain_batch_stderr(self, stream: asyncio.StreamReader) -> None:
        while chunk := await stream.read(4096):
            room = max(0, _STDERR_LIMIT - len(self._batch_stderr))
            self._batch_stderr.extend(chunk[:room])

    async def _discard_batch(self) -> None:
        process = self._batch
        stderr_task = self._batch_stderr_task
        self._batch = None
        self._batch_stderr_task = None
        if process is not None:
            await _stop(process, timeout_s=self._close_timeout_s)
        if stderr_task is not None:
            stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stderr_task

    async def _run_git(
        self, *arguments: str, stdout_limit: int
    ) -> bytes:
        process = await asyncio.create_subprocess_exec(
            self._git(),
            "-C",
            str(self.repository),
            *arguments,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._environment(),
            start_new_session=os.name == "posix",
        )
        stdout = cast("asyncio.StreamReader", process.stdout)
        stderr = cast("asyncio.StreamReader", process.stderr)
        stdout_task = asyncio.create_task(_read_bounded(stdout, stdout_limit))
        stderr_task = asyncio.create_task(_read_bounded(stderr, _STDERR_LIMIT))
        try:
            stdout, stderr, returncode = await asyncio.wait_for(
                asyncio.gather(stdout_task, stderr_task, process.wait()),
                timeout=self._timeout_s,
            )
        except asyncio.CancelledError:
            await asyncio.shield(_stop(process, timeout_s=self._close_timeout_s))
            raise
        except (TimeoutError, _OutputLimitError) as exc:
            await _stop(process, timeout_s=self._close_timeout_s)
            raise GitSourceError("Git exceeded a configured time or output limit") from exc
        finally:
            for task in (stdout_task, stderr_task):
                if not task.done():
                    task.cancel()
        if returncode != 0:
            del stderr
            raise GitSourceError("Git could not read the configured repository object")
        return stdout

    def _git(self) -> str:
        if self._executable is None:
            raise RuntimeError("Git reader has not been set up")
        return self._executable

    @staticmethod
    def _environment() -> dict[str, str]:
        environment = {
            key: value for key, value in os.environ.items() if not key.startswith("GIT_")
        }
        environment.update(
            {
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_OPTIONAL_LOCKS": "0",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "LC_ALL": "C",
            }
        )
        return environment


async def _read_bounded(stream: asyncio.StreamReader, limit: int) -> bytes:
    found = bytearray()
    while chunk := await stream.read(min(64 * 1024, limit - len(found) + 1)):
        found.extend(chunk)
        if len(found) > limit:
            raise _OutputLimitError
    return bytes(found)


async def _stop(process: asyncio.subprocess.Process, *, timeout_s: float) -> None:
    if process.returncode is not None:
        await process.wait()
        return
    with contextlib.suppress(ProcessLookupError):
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout_s)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        await process.wait()
