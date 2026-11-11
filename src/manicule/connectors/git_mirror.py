"""A managed bare mirror of one remote repository: how a Git source reaches a network.

:class:`~manicule.connectors.git_reader.PinnedGitReader` reads a repository that is already on
disk and never fetches, deliberately: a reader that could also reach a network would have two
jobs and two sets of failure modes. This module is the other half. It keeps one **bare** copy of a
remote under the cache directory, fetches the revision a source asked for into a ref of its own
before each sync, and hands the reader a path and a ref name. Everything after that — pinning one
commit, listing its tree, reading blobs by object id — is the reader's, unchanged.

**The mirror is a cache, and is allowed to be wrong or missing.** It holds nothing the remote
does not, so deleting its directory costs one fetch and nothing else. That is why a corrupt one is
reported with the path to remove rather than repaired in place.

**What is fetched is the tip, not the history.** ``--depth 1`` brings the commit the revision names
and the blobs in its tree, which is everything a source that indexes the tree at one commit can
use. A blobless clone would fetch less, and was rejected: reading a missing blob from a promisor
remote is one round trip per document, so a first sync of a repository of a few thousand pages
would be a few thousand requests, and what a server will do for a lazy fetch varies.

**The credential never touches the URL, the command line or the repository's configuration.** It
is passed as ``http.<origin>/.extraHeader`` through ``GIT_CONFIG_*`` in the child's environment,
which exists only for the life of that process. Scoped to the origin, so a redirect to another host
carries nothing, and the URL it is scoped to must be ``https``, because a token sent in the clear
is a token published. A remote with credentials in its address is refused at configuration time.

**Only ``https`` is spoken.** ``GIT_ALLOW_PROTOCOL`` is the allowlist, so a redirect or a
submodule url naming ``file``, ``ssh`` or ``ext`` is refused by Git itself. SSH is absent on
purpose: it needs a key file and a trusted ``known_hosts``, which is two more things to provision
into a container and two more ways for a fetch to hang. Hooks are pointed at nowhere, and nothing
can prompt.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import os
import shutil
import signal
from typing import TYPE_CHECKING, Final, cast
from urllib.parse import urlsplit

from manicule.connectors.git_reader import (
    GitSourceError,
    PinnedGitReader,
    hardened_git_environment,
)
from manicule.connectors.git_remote import require_fetchable_revision
from manicule.core.errors import ConfigError

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "GitFetchError",
    "GitMirror",
    "tip_ref",
]


def tip_ref(revision: str) -> str:
    """The ref a fetch of ``revision`` lands on, so the reader can name the commit it got.

    **One ref per revision, not one per mirror.** Two sources may name one remote and share its
    mirror, and if both fetched into one ref the later fetch would silently replace the commit the
    earlier source was about to resolve: a source indexing ``main`` would index ``develop``'s
    tree under its own name. The name is a digest, so nothing an operator typed becomes a ref.
    """
    return f"refs/manicule/{hashlib.blake2b(revision.encode(), digest_size=8).hexdigest()}"


_DEFAULT_TIMEOUT_S: Final = 300.0
_STDERR_LIMIT: Final = 8 * 1024
_MESSAGE_LIMIT: Final = 1_024
_STOP_TIMEOUT_S: Final = 2.0
_LOCKS: Final[dict[Path, asyncio.Lock]] = {}


class GitFetchError(GitSourceError):
    """A remote could not be reached, refused the request, or did not have the revision."""


class GitMirror:
    """One remote's bare mirror, refreshed on demand.

    Args:
        url: The remote, already normalized by :func:`normalize_remote_url`.
        directory: Where the bare repository lives. Created on first use.
        token: The credential, or ``None`` for a remote that needs none. Sent as the password of
            HTTP Basic authentication, which is what GitHub, GitLab and Gitea all accept for a
            personal access token.
        username: The user name that goes with the token. ``x-access-token`` is accepted by
            GitHub for any token; the others ignore it.
        timeout_s: Longest one fetch may take.
        allowed_protocols: Passed to ``GIT_ALLOW_PROTOCOL``. Anything but ``https`` exists so a
            test can fetch from a repository on disk; a configured source never changes it.
        executable: The ``git`` binary. Looked up on ``PATH`` when ``None``.
    """

    def __init__(
        self,
        url: str,
        directory: Path,
        *,
        token: str | None = None,
        username: str = "x-access-token",
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        allowed_protocols: str = "https",
        executable: str | None = None,
    ) -> None:
        self._url = url
        self._directory = directory
        self._token = token
        self._username = username
        self._timeout_s = max(1.0, timeout_s)
        self._allowed_protocols = allowed_protocols
        self._executable = executable
        # One lock per directory rather than per instance: two sources that name one remote share
        # one mirror (see :func:`mirror_directory`), and two fetches into one repository fight
        # over its locks no matter which object started them.
        self._lock = _LOCKS.setdefault(directory, asyncio.Lock())

    @property
    def directory(self) -> Path:
        return self._directory

    async def refresh(self, revision: str) -> str:
        """Fetch ``revision`` into its own ref (:func:`tip_ref`) and return the commit it names.

        Serialized, because two fetches into one repository fight over its locks and a source
        and a scheduled reconciliation can ask at once. **The ref is resolved to a commit id while
        the lock is still held**, and the caller reads that id, never the ref: another source on
        the same remote and revision may fetch again the moment the lock is released, and a ref is
        a name for whatever the last fetch brought.

        Raises:
            GitFetchError: The remote could not be reached, refused the credential, or has no such
                revision. The message names the address and the revision, never the credential.
            ConfigError: ``git`` is not installed.
        """
        try:
            require_fetchable_revision(revision)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        executable = self._executable or shutil.which("git")
        if executable is None:
            raise ConfigError("git is not installed or is not on PATH")
        self._executable = executable
        async with self._lock:
            await self._initialize(executable)
            await self._fetch(executable, revision)
            return await PinnedGitReader(
                self._directory, revision=tip_ref(revision), executable=executable
            ).resolve()

    async def _initialize(self, executable: str) -> None:
        head = self._directory / "HEAD"
        if head.exists():
            return
        self._directory.mkdir(parents=True, exist_ok=True)
        await self._run(
            [executable, "init", "--bare", "--quiet", str(self._directory)],
            what="creating the mirror",
        )

    async def _fetch(self, executable: str, revision: str) -> None:
        await self._run(
            [
                executable,
                "-C",
                str(self._directory),
                "fetch",
                "--quiet",
                "--depth=1",
                "--no-tags",
                "--force",
                "--no-write-fetch-head",
                "--no-recurse-submodules",
                self._url,
                f"+{revision}:{tip_ref(revision)}",
            ],
            what=f"fetching revision {revision!r} from {self._url}",
        )

    def environment(self) -> dict[str, str]:
        """What every Git process this mirror starts runs under, credential included.

        Public so a test can assert on exactly what Git will read, which is the only way to check
        the parts of a fetch that need a network. Contains the token: never log it.
        """
        environment = hardened_git_environment()
        entries: list[tuple[str, str]] = [
            ("core.hooksPath", os.devnull),
            ("credential.helper", ""),
            ("protocol.version", "2"),
            ("fetch.fsckObjects", "true"),
            ("fetch.recurseSubmodules", "false"),
            ("gc.autoDetach", "false"),
            ("maintenance.autoDetach", "false"),
            ("http.followRedirects", "initial"),
        ]
        if self._token is not None:
            origin = urlsplit(self._url)
            encoded = base64.b64encode(f"{self._username}:{self._token}".encode()).decode("ascii")
            entries.append(
                (
                    f"http.https://{origin.netloc}/.extraHeader",
                    f"Authorization: Basic {encoded}",
                )
            )
        environment["GIT_ALLOW_PROTOCOL"] = self._allowed_protocols
        environment["GIT_CONFIG_COUNT"] = str(len(entries))
        for index, (key, value) in enumerate(entries):
            environment[f"GIT_CONFIG_KEY_{index}"] = key
            environment[f"GIT_CONFIG_VALUE_{index}"] = value
        return environment

    async def _run(self, arguments: list[str], *, what: str) -> None:
        process = await asyncio.create_subprocess_exec(
            *arguments,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            env=self.environment(),
            start_new_session=os.name == "posix",
        )
        stderr = cast("asyncio.StreamReader", process.stderr)
        try:
            tail, returncode = await asyncio.wait_for(
                _collect(process, stderr), timeout=self._timeout_s
            )
        except asyncio.CancelledError:
            await asyncio.shield(_terminate(process, stderr))
            raise
        except TimeoutError as exc:
            await _terminate(process, stderr)
            msg = f"git timed out after {self._timeout_s:.0f}s while {what}"
            raise GitFetchError(msg) from exc
        if returncode != 0:
            msg = f"git failed (exit status {returncode}) while {what}{self._explain(tail)}"
            raise GitFetchError(msg)

    def _explain(self, stderr: bytes) -> str:
        """Git's own words, with the credential removed, plus the usual cause.

        Bounded and printable, because it comes from a remote and ends up in a log and a status
        line. The token is removed in both forms it can take, as written and as it was encoded
        into the header, in case a server or a proxy echoes the request.
        """
        text = stderr.decode("utf-8", errors="replace")
        if self._token:
            encoded = base64.b64encode(f"{self._username}:{self._token}".encode()).decode("ascii")
            text = text.replace(encoded, "***").replace(self._token, "***")
        text = "".join(ch if ch.isprintable() or ch == "\n" else " " for ch in text).strip()
        if not text:
            return ""
        lowered = text.lower()
        hint = ""
        if any(
            marker in lowered
            for marker in ("authentication failed", "could not read username", "401", "403")
        ):
            hint = " Check that the token in token_env can read this repository."
        elif "not found" in lowered or "couldn't find remote ref" in lowered:
            hint = " Check the repository address and the revision."
        else:
            hint = (
                f" If the mirror is damaged, delete {self._directory}: it is a cache, and the next "
                f"sync fetches it again."
            )
        return f": {text[:_MESSAGE_LIMIT]}.{hint}"


async def _collect(
    process: asyncio.subprocess.Process, stderr: asyncio.StreamReader
) -> tuple[bytes, int]:
    """Read the first :data:`_STDERR_LIMIT` bytes of stderr, discard the rest, and wait."""
    kept = bytearray()
    while chunk := await stderr.read(4096):
        room = _STDERR_LIMIT - len(kept)
        if room > 0:
            kept.extend(chunk[:room])
    return bytes(kept), await process.wait()


def _signal(process: asyncio.subprocess.Process, number: int) -> None:
    """Signal the whole process group, because a fetch is a tree of processes.

    ``git fetch`` starts ``git-remote-https`` and ``index-pack``, and ``start_new_session`` made
    the parent the leader of a group of its own. Signaling only the parent leaves the helpers
    running and holding the pipe, which is a hung sync that looks stopped.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError):
        if os.name == "posix":
            os.killpg(process.pid, number)
        else:  # pragma: no cover - the supported platforms are POSIX
            process.send_signal(number)


async def _terminate(process: asyncio.subprocess.Process, stderr: asyncio.StreamReader) -> None:
    """Stop the process tree, reap it, and let its pipe close so nothing is left open."""
    _signal(process, signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), timeout=_STOP_TIMEOUT_S)
    except TimeoutError:
        _signal(process, signal.SIGKILL)
        await process.wait()
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(stderr.read(), timeout=_STOP_TIMEOUT_S)
