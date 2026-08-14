"""The socket the command line writes through, and why it is not the HTTP API.

A served manicule holds the data directory's writer lock for its whole life, so every command
that writes has to reach *it* rather than open the directory itself. Something has to carry
those commands, and the obvious candidate — the HTTP API that is already there — is the wrong
one twice over.

**It would put a sync behind a route.** ``tests/api/test_routes.py`` asserts by name that the
destructive operations have none, and #113 refused one for corpus-wide reparse on the grounds
that an unattended caller could hold the accelerator for an hour. A server that syncs on its
own configuration does not cross that line; a route that lets a caller start one does. Nothing
here is reachable over a network, so nothing here is that route.

**And it would drag the bind policy and the auth model into a decision that has nothing to do
with them.** Loopback unless a non-loopback host is configured *and* the flag is passed *and*
auth is on — that policy exists to answer "who may reach this from the network". A Unix domain
socket is not on the network at all, so the question does not arise and the policy is left
exactly as it is rather than gaining an exception.

What protects this instead is the filesystem: a ``0600`` socket inside a ``0700`` directory,
both owned by the user running the server. :func:`connect` re-checks both before it writes a
byte, because the whole point of a permission is that the party relying on it looks.

**Where the socket lives, and why it is not the data directory.** Beside the pid file would be
the tidy answer and it does not fit: ``sockaddr_un.sun_path`` is 104 bytes on macOS — 102
usable — and an ordinary data directory under a macOS per-user temporary path is *already* over
that before a filename is added. So the socket goes where the platform puts sockets for running
processes (:func:`runtime_dir`), under a name derived from the data directory it serves
(:func:`socket_path`), which keeps the path short by construction and still gives one server
per data directory. The pid file stays where #126 put it; it answers a different question.

**The protocol is one request per connection, and lines of JSON in both directions.**

.. code-block:: text

    client → server   one line: an Invoke, a Handover, a Forget or a Held
    server → client   zero or more Progress lines, then exactly one Result line, then EOF

A connection *is* a request, so there is no request id, no multiplexing and no session state to
get wrong; the operation's lifetime is the connection's. Lines of JSON rather than a framing
library because the payload on the way back is already the :class:`~manicule.app.results.Envelope`
every other surface returns — which is what makes a proxied command faithful by construction
rather than by a mapping somebody maintains — and because an operator can read the wire with
``nc`` when they need to.

**A client that goes away does not stop the operation.** The server is the writer; a sync that
abandoned itself halfway because a terminal closed would be worse than one that finishes, and
the operator who wants it stopped has ``manicule stop``. :mod:`manicule.cli.proxy` says so on
the way out rather than leaving it to be discovered.

**Nothing here ever renders a request.** :class:`Handover` carries a live session, so its
cookies are :class:`~pydantic.SecretStr` and the one place they are unwrapped is
:meth:`Handover.to_line`. A failure reports the *kind* of request that failed and never its
contents — see :func:`_refusal`.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import socket as socketlib
import stat
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, SecretStr, ValidationError

from manicule.core.errors import ManiculeError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

__all__ = [
    "MAX_FRAME_BYTES",
    "SOCKET_MODE",
    "AlreadyServingError",
    "ControlServer",
    "Forget",
    "Handover",
    "Held",
    "Invoke",
    "Progress",
    "ProtocolError",
    "Request",
    "Response",
    "Result",
    "SocketUnusableError",
    "connect",
    "is_serving",
    "read_request",
    "runtime_dir",
    "socket_path",
    "unusable",
]

SOCKET_MODE = 0o600
"""What the socket must be, and what :func:`connect` refuses anything else for."""

DIRECTORY_MODE = 0o700
"""What the directory holding it must be. Checked as well as set: :func:`runtime_dir` may find
a directory an earlier run made, or one somebody else made first."""

MAX_FRAME_BYTES = 1_048_576
"""How long one line may be before the connection is refused.

A :class:`Handover` is the large frame — an instance behind single sign-on issues cookies of
its own besides Confluence's, and several kilobytes is ordinary. A megabyte is far above that
and far below anything that would matter, and it exists so that a client writing an unbounded
stream is a refusal rather than this process's memory.
"""

SOCKET_SUFFIX = ".sock"

_DIGEST_CHARS = 16
"""How much of the data directory's digest names its socket.

Long enough that two data directories on one machine will not collide, short enough that the
whole path stays well inside ``sun_path``. It is a name, not a secret: anybody who can read the
directory can already list what is in it.
"""


class ProtocolError(ManiculeError):
    """The socket carried something that is not this protocol.

    A :class:`~manicule.core.errors.ManiculeError` so that it reaches a caller as an envelope
    like every other outcome, rather than as a traceback out of a stream reader.
    """


class AlreadyServingError(ProtocolError):
    """Something is listening where this server was about to.

    **A temporary condition**, and its own class so that a supervisor can be told so. The thing
    on the other end is a process; it will stop, or somebody will stop it, and the next attempt
    then succeeds with nothing changed. See :data:`~manicule.cli.serving.EX_TEMPFAIL`.
    """


class SocketUnusableError(ProtocolError):
    """The runtime directory or the socket is not this user's, or is not private to them.

    **A permanent condition**, and the distinction from :class:`AlreadyServingError` is the whole
    reason both exist. A directory somebody else owns, a symbolic link where a directory belongs,
    or a mode granting group or other does not resolve by waiting: it needs somebody to change
    something. Retrying it every thirty seconds for ever is a supervisor doing the wrong thing
    confidently, and the exit status is how it is told which of the two it is looking at — see
    :data:`~manicule.cli.serving.EX_CONFIG`.

    Raised by :func:`_require_private` alone, which is the one place any of those three is
    decided, so the classification cannot come apart from the check.
    """


# --- where the socket lives -------------------------------------------------------------------


def runtime_dir() -> Path:
    """The directory this machine keeps sockets for running processes in, created ``0700``.

    Three candidates, in the order the platforms actually answer:

    ``$XDG_RUNTIME_DIR``
        systemd's per-user directory. Already ``0700``, already per-user, and cleaned when the
        session ends — which is the right lifetime for a socket belonging to a running process.
    ``$TMPDIR``
        macOS's per-user temporary directory, which has the same two properties.
    ``/tmp``
        The fallback, and the one that is world-writable. The ``manicule-<uid>`` component is
        what keeps two users on one machine apart, and the ownership check below is what makes
        that a guarantee rather than a naming convention.

    Returns:
        An existing directory, owned by this user, with no group or other permissions on it.

    Raises:
        ProtocolError: The directory exists and is not ours, or is not a directory, or carries
            permissions for anybody but us. Refused rather than corrected: a directory somebody
            else made where this one goes is a fact worth stopping on, and quietly
            ``chmod``-ing it would be taking it over.
    """
    base = os.environ.get("XDG_RUNTIME_DIR") or os.environ.get("TMPDIR") or tempfile.gettempdir()
    path = Path(base).expanduser() / f"manicule-{os.getuid()}"
    path.mkdir(mode=DIRECTORY_MODE, parents=True, exist_ok=True)
    _require_private(path, what="the runtime directory")
    return path


def socket_path(data_dir: Path) -> Path:
    """Where the server for ``data_dir`` listens.

    Named for a digest of the data directory rather than for the directory itself, which is what
    keeps the path inside ``sun_path`` however deep the data directory is. Two data directories
    therefore get two sockets, and one gets the same socket from every process that asks —
    which is the whole requirement.
    """
    digest = hashlib.sha256(str(_resolved(data_dir)).encode()).hexdigest()[:_DIGEST_CHARS]
    return runtime_dir() / f"{digest}{SOCKET_SUFFIX}"


def _resolved(data_dir: Path) -> Path:
    """The data directory as one canonical string, so two spellings name one socket.

    ``resolve()`` on a directory that does not exist yet still normalizes the path, which is the
    case that matters: ``manicule serve`` on a fresh installation creates the data directory
    *after* this has already decided what the socket is called.
    """
    return data_dir.expanduser().resolve()


def _require_private(path: Path, *, what: str) -> None:
    """Refuse a path this user does not own, or that anybody else can reach.

    Raises:
        SocketUnusableError: It is not ours, or its mode grants anything to group or other.
            Permanent, every one of them: each needs somebody to change something rather than
            somebody to wait.
    """
    try:
        info = path.lstat()
    except OSError as exc:
        msg = f"{what} at {path} could not be read: {exc}"
        raise SocketUnusableError(msg) from exc
    if stat.S_ISLNK(info.st_mode):
        msg = (
            f"{what} at {path} is a symbolic link. It is refused rather than followed, because "
            f"what it points at is decided by whoever made the link."
        )
        raise SocketUnusableError(msg)
    if info.st_uid != os.getuid():
        msg = (
            f"{what} at {path} belongs to uid {info.st_uid} and this process is uid "
            f"{os.getuid()}. Refused rather than used: the permissions on it are somebody "
            f"else's to change."
        )
        raise SocketUnusableError(msg)
    granted = stat.S_IMODE(info.st_mode)
    if granted & 0o077:
        msg = (
            f"{what} at {path} is mode {granted:04o}, which grants access to somebody other "
            f"than its owner. The control socket carries write commands and a live session, so "
            f"it is refused rather than narrowed under you."
        )
        raise SocketUnusableError(msg)


# --- what crosses it --------------------------------------------------------------------------


class Invoke(BaseModel):
    """Run this operation here, because you are the writer and I am not.

    ``arguments`` is JSON rather than a per-operation model because the binder on the other side
    is already the one definition of how an operation's arguments become a service call
    (:mod:`manicule.app.commands`). A model here would be a second one, and the failure of two
    is that they agree until they do not.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["invoke"] = "invoke"
    op: str = Field(min_length=1)
    arguments: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])
    workspace: str = ""
    """Which tenant to run in. Empty means the server's configured one."""

    def to_line(self) -> bytes:
        return _line(self.model_dump(mode="json"))


class Handover(BaseModel):
    """A session captured on the command line, given to the server to hold.

    The capture happens where the person is — ``--browser`` opens a window, and a window opened
    by a background server is one nobody is sitting at — and the credential has to end up where
    the syncs run. This frame is that hand-off, and it is the only thing on this socket that
    carries a secret.

    **The cookies are :class:`~pydantic.SecretStr`.** A repr, a traceback frame, a logged frame
    or a validation error therefore carries the wrapper rather than the value, and the one place
    the values are unwrapped is :meth:`to_line` — which is where a reader looks for it, on the
    same principle as :meth:`~manicule.connectors.credentials.BrowserSession.to_json`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["handover"] = "handover"
    base_url: str = Field(min_length=1)
    account: str
    captured_at: str
    """ISO-8601, as :attr:`~manicule.connectors.credentials.BrowserSession.captured_at` renders
    it. The server keeps the capture time the *client* proved, because that is what
    ``session_max_age_hours`` measures from."""

    cookies: dict[str, SecretStr] = Field(default_factory=dict[str, SecretStr])

    def to_line(self) -> bytes:
        """The frame as one line, with the secrets deliberately unwrapped.

        Serialized by hand rather than through pydantic's JSON mode, which renders every secret
        as asterisks — correct for a log and useless for a hand-off.
        """
        return _line(
            {
                "kind": self.kind,
                "base_url": self.base_url,
                "account": self.account,
                "captured_at": self.captured_at,
                "cookies": {
                    name: secret.get_secret_value() for name, secret in self.cookies.items()
                },
            }
        )


class Forget(BaseModel):
    """Drop the session for this instance, if there is one.

    A frame of its own rather than a :class:`Handover` with no cookies in it. "An empty
    credential means delete the credential" is the kind of implicit rule that reads fine in the
    code that wrote it and is a data-loss bug in the code that reads it — and this one deletes a
    credential somebody would have to open a browser to replace.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["forget"] = "forget"
    base_url: str = Field(min_length=1)

    def to_line(self) -> bytes:
        return _line(self.model_dump(mode="json"))


class Held(BaseModel):
    """Which instances are you holding a session for?

    The one question about a credential that a process which is *not* the server needs an answer
    to, and the one it cannot answer for itself: sessions live in the server's memory, so a
    ``manicule doctor`` typed at a terminal is looking at an empty vault whatever the server has.
    Reporting that emptiness as "there is no session" would be a diagnostic that is always
    alarming and never informative.

    **The reply carries no session value, no length and no digest of one** — only the instance
    and the account, which are exactly the two fields
    :meth:`~manicule.app.served.ControlHandler._accept` already answers a hand-off with. A
    credential does not become safe to echo by being asked for a second time.

    A frame of its own rather than an :class:`Invoke`, because the accept list an ``Invoke``
    names (:data:`~manicule.app.commands.BINDERS`) is the operations that can be *described as
    data and run*, and this runs nothing. It is a question about the process, in the same family
    as the two frames that put a credential there and take it away again.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["held"] = "held"

    def to_line(self) -> bytes:
        return _line(self.model_dump(mode="json"))


class Progress(BaseModel):
    """Something happened that is worth saying before the operation is over.

    One sentence, already written for a person. The server decides what is worth saying because
    the server is the only party that can see it; the client's whole job is to put it on stderr
    as it arrives, so that a long sync is visibly working rather than silently hung.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["progress"] = "progress"
    message: str

    def to_line(self) -> bytes:
        return _line(self.model_dump(mode="json"))


class Result(BaseModel):
    """The operation's outcome, as the envelope every other surface would have returned.

    Carried as already-serialized JSON rather than as a parsed :class:`Envelope` so that this
    module needs no opinion about payload models. It is the same bytes ``--json`` prints.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["result"] = "result"
    envelope: dict[str, JsonValue]

    def to_line(self) -> bytes:
        return _line(self.model_dump(mode="json"))


type Request = Invoke | Handover | Forget | Held
type Response = Progress | Result

_REQUESTS: Mapping[str, type[Invoke] | type[Handover] | type[Forget] | type[Held]] = {
    "invoke": Invoke,
    "handover": Handover,
    "forget": Forget,
    "held": Held,
}


def _line(payload: Mapping[str, object]) -> bytes:
    """One frame, as the single line that carries it.

    ``ensure_ascii`` is left on so that a frame is bytes a terminal can print whatever the
    locale is, and separators are tightened because a frame is machine-read far more often than
    it is looked at.
    """
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode()


def read_request(line: bytes) -> Request:
    """Parse one request frame, refusing anything that is not one.

    Raises:
        ProtocolError: The line is not JSON, is not an object, names no known kind, or does not
            validate. The message names the *kind* and never the contents, because one of the
            two kinds is a live session and an error carrying it would be the leak this whole
            module is arranged to prevent.
    """
    try:
        decoded = json.loads(line)
    except ValueError as exc:
        msg = f"the control socket carried something that is not JSON: {exc}"
        raise ProtocolError(msg) from exc
    if not isinstance(decoded, dict):
        msg = "the control socket carried JSON that is not an object, so it names no request"
        raise ProtocolError(msg)
    frame = cast("dict[str, object]", decoded)
    kind = frame.get("kind")
    model = _REQUESTS.get(kind) if isinstance(kind, str) else None
    if model is None or not isinstance(kind, str):
        known = ", ".join(sorted(_REQUESTS))
        msg = f"the control socket named request kind {kind!r}. Known kinds: {known}"
        raise ProtocolError(msg)
    try:
        return model.model_validate(frame)
    except ValidationError as exc:
        raise ProtocolError(_refusal(kind, exc)) from exc


def _refusal(kind: str, exc: ValidationError) -> str:
    """Why a frame was refused, without repeating the frame back.

    pydantic's own message quotes the offending input, which is right everywhere else and wrong
    for exactly one of these two kinds. So the field names are reported and the values are not,
    for both — because a rule that applies to one kind and not the other is a rule somebody
    applies to the wrong one eventually.
    """
    fields = sorted(
        {".".join(str(part) for part in error["loc"]) or "(root)" for error in exc.errors()}
    )
    return (
        f"the control socket carried a {kind!r} frame that is not one: "
        f"{len(exc.errors())} problem(s) in {', '.join(fields)}. The frame's contents are not "
        f"repeated here, because a handover carries a live session."
    )


def read_response(line: bytes) -> Response:
    """Parse one response frame.

    Raises:
        ProtocolError: It is not a response this client understands. A server that is a
            different version from its client is the realistic cause, so the message says which
            kinds this one knows rather than only that the frame was wrong.
    """
    try:
        decoded = json.loads(line)
    except ValueError as exc:
        msg = f"the server sent something that is not JSON: {exc}"
        raise ProtocolError(msg) from exc
    if not isinstance(decoded, dict):
        msg = "the server sent JSON that is not an object"
        raise ProtocolError(msg)
    frame = cast("dict[str, object]", decoded)
    kind = frame.get("kind")
    if kind == "progress":
        return Progress.model_validate(frame)
    if kind == "result":
        return Result.model_validate(frame)
    msg = f"the server sent response kind {kind!r}. Known kinds: progress, result"
    raise ProtocolError(msg)


# --- the server side --------------------------------------------------------------------------


class Handler(Protocol):
    """What the control server does with a request.

    Kept as a protocol so that this module knows nothing about operations, services or sessions.
    It owns a socket and a wire format; what a request *means* is
    :mod:`manicule.app.commands`'s business.
    """

    async def handle(self, request: Request, report: Callable[[str], None]) -> dict[str, JsonValue]:
        """Run one request and return the envelope it produced.

        Args:
            request: What arrived.
            report: Call it with one sentence to send a :class:`Progress` frame now. It never
                raises and never blocks: a handler that had to reason about whether its progress
                reached anybody would be a handler with a second failure mode.

        Returns:
            The envelope, already JSON. Failures are envelopes too — this returns rather than
            raises for anything a caller could act on, exactly as
            :func:`~manicule.app.dispatch.run_op` does.
        """
        ...


class ControlServer:
    """The listening half. One instance per served data directory.

    Started after the writer lock is held, which is what makes "the socket exists" and "somebody
    owns this data directory" the same fact. :meth:`start` is deliberately not tolerant of an
    existing live socket: two servers is the state :class:`~manicule.ingest.recovery.InstanceLock`
    exists to prevent, and this is the second place it is refused rather than the first.
    """

    def __init__(self, path: Path, handler: Handler) -> None:
        self._path = path
        self._handler = handler
        self._server: asyncio.Server | None = None
        self._connections: set[asyncio.Task[None]] = set()

    @property
    def path(self) -> Path:
        return self._path

    async def start(self) -> None:
        """Bind and listen, with the socket ``0600`` from the moment it exists.

        The mode is set by ``umask`` around the bind rather than by ``chmod`` after it. A
        ``chmod`` leaves a window — however short — in which the socket is on the filesystem at
        whatever the ambient umask allowed, and the thing on the other end of that window is a
        write command with a live session behind it. The ``chmod`` is done as well, because a
        platform that ignores umask for sockets should end in the wrong mode rather than in a
        wrong mode nobody checked.

        Raises:
            ProtocolError: Another server is listening on this path already, or the runtime
                directory is not ours.
        """
        _require_private(self._path.parent, what="the runtime directory")
        self._clear_stale()
        previous = os.umask(0o177)
        try:
            self._server = await asyncio.start_unix_server(
                self._serve_one, path=str(self._path), limit=MAX_FRAME_BYTES
            )
        finally:
            os.umask(previous)
        self._path.chmod(SOCKET_MODE)
        _require_private(self._path, what="the control socket")

    def _clear_stale(self) -> None:
        """Remove a socket file no server is behind, and refuse one that has a server.

        A Unix socket is a file that outlives the process that bound it, so "the file is there"
        answers nothing on its own. Connecting is what answers it: a refused connection means
        the file is a leftover, and an accepted one means somebody is serving.

        Raises:
            AlreadyServingError: Something answered. Starting anyway would bind over a live
                server's path and leave it listening on a socket nothing can reach. Temporary:
                that process will stop, or somebody will stop it.
        """
        if not self._path.exists():
            return
        if _answers(self._path):
            msg = (
                f"a manicule server is already listening on {self._path}. One server per data "
                f"directory: it holds the writer lock, the schedule and any captured session, "
                f"and a second would take none of them over. Stop it with `manicule stop`."
            )
            raise AlreadyServingError(msg)
        self._path.unlink(missing_ok=True)

    async def aclose(self) -> None:
        """Stop listening, let the connections in flight finish, and remove the socket.

        In-flight connections are awaited rather than canceled. Each one is a write command the
        operator typed, and the server is the only writer there is — tearing one down at the
        moment the process is asked to stop is how a document ends up half-written.
        """
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self._connections:
            await asyncio.gather(*tuple(self._connections), return_exceptions=True)
        self._path.unlink(missing_ok=True)

    async def _serve_one(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """One connection: read the request, stream progress, write exactly one result.

        Registered in :attr:`_connections` so that :meth:`aclose` can wait for it. ``asyncio``
        calls this in a task it does not hand back, so the set is how this object gets a handle
        on its own work.
        """
        task = asyncio.current_task()
        if task is not None:
            self._connections.add(task)
        try:
            await self._exchange(reader, writer)
        finally:
            if task is not None:
                self._connections.discard(task)
            writer.close()
            with contextlib.suppress(OSError, asyncio.CancelledError):
                await writer.wait_closed()

    async def _exchange(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Read one request, run it, and answer.

        A request that will not parse is answered with a failure envelope rather than by
        dropping the connection, because a client that is told nothing retries — and this is the
        path a version mismatch arrives on.
        """
        try:
            line = await self._read_line(reader)
        except ProtocolError as exc:
            await self._write(writer, Result(envelope=_protocol_failure(exc)))
            return
        if not line:
            return
        try:
            request = read_request(line)
        except ProtocolError as exc:
            await self._write(writer, Result(envelope=_protocol_failure(exc)))
            return

        pending: list[str] = []
        # Per connection rather than on the server. Two proxied commands can be in flight at
        # once — a sync and a reindex, say — and a shared event would let one command's flush
        # clear the signal the other's report had just raised, so the second would sit unsent
        # until something else happened to wake it.
        reported = asyncio.Event()

        def report(message: str) -> None:
            pending.append(message)
            reported.set()

        envelope = await self._run(request, report, writer, pending, reported)
        await self._write(writer, Result(envelope=envelope))

    async def _run(
        self,
        request: Request,
        report: Callable[[str], None],
        writer: asyncio.StreamWriter,
        pending: list[str],
        reported: asyncio.Event,
    ) -> dict[str, JsonValue]:
        """Run the handler while draining whatever it reports, and never lose the result.

        The handler's ``report`` appends to a list and sets an event, so that reporting progress
        cannot block a pipeline stage or raise into it. This waits on that event — it is the only
        place progress becomes bytes, and the only place a dead client is discovered.

        **Woken rather than polled.** An earlier version looped on a 50 ms timeout, which is
        72,000 wakeups across an hour-long sync to deliver perhaps a few hundred lines, and it
        put up to 50 ms of latency on each one. Waiting on either the report or the work finishing
        costs nothing while a sync is quiet and delivers immediately when it is not.
        """
        work = asyncio.ensure_future(self._handler.handle(request, report))
        try:
            while True:
                waking = asyncio.ensure_future(reported.wait())
                done, _ = await asyncio.wait((work, waking), return_when=asyncio.FIRST_COMPLETED)
                reported.clear()
                waking.cancel()
                await self._flush(writer, pending)
                if work in done:
                    # Flushed once more, because the handler may have reported between the last
                    # flush and returning — and that report is the last thing it said.
                    await self._flush(writer, pending)
                    return await work
        finally:
            if not work.done():  # pragma: no cover - only on cancellation of the connection
                work.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await work

    async def _flush(self, writer: asyncio.StreamWriter, pending: list[str]) -> None:
        """Send what has been reported since the last tick.

        Drained by slicing rather than by popping one at a time, so a handler reporting while
        this runs is appended after the slice rather than interleaved into it.
        """
        if not pending:
            return
        ready, pending[:] = pending[:], []
        for message in ready:
            await self._write(writer, Progress(message=message))

    async def _write(self, writer: asyncio.StreamWriter, frame: Response) -> None:
        """Write one frame, treating a client that has gone away as nothing to do.

        A closed pipe here is the ordinary case rather than an error: the operator pressed
        ``Ctrl-C``, and the operation carries on because the server is the writer. It is
        swallowed only for the *write* — the handler's own failures are envelopes and reach the
        caller through :meth:`_exchange`.
        """
        with contextlib.suppress(ConnectionResetError, BrokenPipeError, OSError):
            writer.write(frame.to_line())
            await writer.drain()

    @staticmethod
    async def _read_line(reader: asyncio.StreamReader) -> bytes:
        """One frame off the wire, bounded.

        Raises:
            ProtocolError: The line ran past :data:`MAX_FRAME_BYTES`.
        """
        try:
            return await reader.readuntil(b"\n")
        except asyncio.IncompleteReadError:
            # The client connected and closed without sending anything, which is exactly what
            # `is_serving` does. Not an error, and answering it would be answering nobody.
            return b""
        except (asyncio.LimitOverrunError, ValueError) as exc:
            msg = (
                f"the control socket carried a frame longer than {MAX_FRAME_BYTES} bytes and it "
                f"was refused unread."
            )
            raise ProtocolError(msg) from exc


def _protocol_failure(exc: ProtocolError) -> dict[str, JsonValue]:
    """A refusal from this layer, in the shape every other failure arrives in.

    Built here rather than through :func:`~manicule.app.results.failed` because a frame that
    would not parse has no operation and no workspace to name, and inventing either would put a
    word into the contract that no caller asked for.
    """
    return {
        "version": "",
        "op": "control",
        "ok": False,
        "workspace": "unknown",
        "data": None,
        "error": {"type": type(exc).__name__, "message": str(exc), "hint": ""},
    }


# --- the client side --------------------------------------------------------------------------


def is_serving(path: Path) -> bool:
    """Whether a manicule server is listening on this socket, right now.

    This is the liveness signal the command line proxies on, and it is deliberately not the pid
    file. A pid is reused by the operating system, so a pid file names a process that exists and
    may be somebody else's — :mod:`manicule.app.daemon` says as much about its own. A connection
    either completes or does not, and nothing about it can be stale.

    ``False`` for a socket with the wrong owner or mode as well as for one nothing answers, so
    that a command falls back to "no server is running" — which is a message naming the fix —
    rather than to a permission error out of a socket call.
    """
    try:
        _require_usable(path)
    except ProtocolError:
        return False
    return _answers(path)


def unusable(path: Path) -> str:
    """Why this socket cannot be used, or ``""`` when nothing is wrong with it.

    The complement of :func:`is_serving`, which answers a boolean because its caller has to
    branch. This answers the *sentence*, and it exists because "no manicule server is running"
    is the wrong thing to tell somebody whose server is running perfectly well behind a socket
    whose permissions changed underneath it — they would go and start a second one, and be told
    the data directory is taken by a process they now have two reasons to be confused about.

    A path that is simply not there returns ``""``, and so does a socket that is fine but that
    nothing is behind. Neither is a socket being unusable: the first is a server not existing
    and the second is one that stopped without tidying up, and the caller already has a sentence
    for both.
    """
    if not path.exists():
        return ""
    try:
        _require_usable(path)
    except ProtocolError as exc:
        return str(exc)
    return ""


def _answers(path: Path) -> bool:
    """Whether anything accepts a connection on this path.

    A blocking connect on a Unix socket completes or fails against the local kernel, so there is
    no timeout to choose and nothing to wait on.
    """
    probe = socketlib.socket(socketlib.AF_UNIX, socketlib.SOCK_STREAM)
    try:
        probe.connect(str(path))
    except OSError:
        return False
    else:
        return True
    finally:
        probe.close()


def _require_usable(path: Path) -> None:
    """The one answer to "may this process write to this socket", for both callers.

    Absence is checked before permissions rather than after, because "no server is running" is
    by far the commonest way to arrive here and it deserves its own sentence. The permission
    check on a path that is not there says only that it could not be read — true, unhelpful, and
    it reads like a fault in the installation rather than a server nobody started.

    Raises:
        ProtocolError: Nothing is there, or what is there is not private to this user.
    """
    _require_private(path.parent, what="the runtime directory")
    if not path.exists():
        raise ProtocolError(_nothing_listening(path))
    _require_private(path, what="the control socket")


def _nothing_listening(path: Path) -> str:
    """What an operator reads when there is no server, which is the commonest refusal here.

    It names the command that fixes it. A message that says only what was wrong leaves somebody
    to guess that a server is a thing manicule has, and this is the first place most people will
    meet the idea.
    """
    return (
        f"no manicule server is listening on {path}. A served manicule holds this data "
        f"directory's writer lock, its schedule and any captured session, so write commands go "
        f"to it rather than opening the directory themselves. Start one with `manicule serve`."
    )


async def connect(
    path: Path, request: Request, *, on_progress: Callable[[str], None]
) -> dict[str, JsonValue]:
    """Send one request to the server and return the envelope it answers with.

    Args:
        path: The socket, from :func:`socket_path`.
        request: What to run.
        on_progress: Called with each :class:`Progress` message as it arrives, so a long
            operation is visible while it runs rather than after it.

    Returns:
        The envelope, exactly as the server produced it. It is not re-wrapped, re-worded or
        re-classified on the way through: a proxied command prints what an in-process one would
        have printed because it is the same object, not because two code paths agree.

    Raises:
        ProtocolError: The socket is not ours, nothing is listening, the connection ended
            before a result, or a frame did not parse. Every one of these is something a caller
            can act on, so each becomes an envelope one level up rather than a traceback.
    """
    _require_usable(path)
    try:
        reader, writer = await asyncio.open_unix_connection(str(path), limit=MAX_FRAME_BYTES)
    except OSError as exc:
        raise ProtocolError(_nothing_listening(path)) from exc
    try:
        writer.write(request.to_line())
        await writer.drain()
        return await _await_result(reader, on_progress)
    finally:
        writer.close()
        with contextlib.suppress(OSError):
            await writer.wait_closed()


async def _await_result(
    reader: asyncio.StreamReader, on_progress: Callable[[str], None]
) -> dict[str, JsonValue]:
    """Consume progress until the one result arrives.

    Raises:
        ProtocolError: The stream ended with no result. That is a server that died mid-operation
            — the operation may well have half-happened — so it is reported as itself rather
            than as an empty success.
    """
    while True:
        try:
            line = await reader.readuntil(b"\n")
        except asyncio.IncompleteReadError as exc:
            msg = (
                "the manicule server closed the connection without answering. The operation may "
                "have started; `manicule connector list` says what the last run recorded."
            )
            raise ProtocolError(msg) from exc
        except (asyncio.LimitOverrunError, ValueError) as exc:
            msg = f"the manicule server sent a frame longer than {MAX_FRAME_BYTES} bytes"
            raise ProtocolError(msg) from exc
        answer = read_response(line)
        if isinstance(answer, Progress):
            on_progress(answer.message)
            continue
        return answer.envelope
