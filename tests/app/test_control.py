"""The control socket: what it refuses, what crosses it, and what never does.

**Every test here uses the real runtime directory**, because the two properties most worth
checking are properties of the real one: that the path it produces is short enough for
``sun_path`` on this platform, and that its permissions are what they claim. A fixture that
substituted a directory of its own would check a directory the product never uses.

Isolation comes from the *name* instead. :func:`socket_for` derives a socket from a data
directory nothing else in the suite names, so two tests never share a path, and the socket is
removed afterwards whether the test passed or not.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket as socketlib
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, override

import pytest
from pydantic import SecretStr

from manicule.app import control

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from pydantic import JsonValue


@pytest.fixture
def socket_for() -> Iterator[Callable[[], Path]]:
    """A socket path no other test names, cleaned up afterwards."""
    made: list[Path] = []

    def build() -> Path:
        path = control.socket_path(Path(f"/manicule-suite/{uuid.uuid4()}"))
        made.append(path)
        return path

    yield build
    for path in made:
        path.unlink(missing_ok=True)


class Echo:
    """A handler that answers with what it was asked, and reports whatever it was told to.

    Deliberately not a mock. What these tests check is the *wire* — that a request arrives
    whole, that progress precedes the result, that exactly one result is sent — and a handler
    that records its calls rather than answering them would leave the second half unexercised.
    """

    def __init__(
        self, *, progress: tuple[str, ...] = (), fail: BaseException | None = None
    ) -> None:
        self.progress = progress
        self.fail = fail
        self.seen: list[control.Request] = []
        self.released = asyncio.Event()
        self.hold = False

    async def handle(
        self, request: control.Request, report: Callable[[str], None]
    ) -> dict[str, JsonValue]:
        self.seen.append(request)
        for message in self.progress:
            report(message)
        if self.hold:
            await self.released.wait()
        if self.fail is not None:
            raise self.fail
        return {
            "version": "0",
            "op": getattr(request, "op", "handover"),
            "ok": True,
            "workspace": "default",
            "data": {"echoed": request.kind},
            "error": None,
        }


async def serving(path: Path, handler: control.Handler) -> control.ControlServer:
    server = control.ControlServer(path, handler)
    await server.start()
    return server


# --- where it lives, and what it is ------------------------------------------------------------


def test_the_socket_path_fits_inside_what_bind_accepts(socket_for: Callable[[], Path]) -> None:
    """The reason the socket is not in the data directory, asserted rather than asserted about.

    ``sun_path`` is 104 bytes on macOS and 108 on Linux, and a pytest temporary directory on
    macOS is already 103 characters before a filename is added — so "put it beside the pid
    file" is not a style preference that was overruled, it is an arrangement that does not
    bind. This asserts the arrangement that does, by binding it.
    """
    path = socket_for()
    listener = socketlib.socket(socketlib.AF_UNIX, socketlib.SOCK_STREAM)
    try:
        listener.bind(str(path))
    finally:
        listener.close()
    assert path.exists(), "the socket the product would use did not bind on this platform"


def test_two_data_directories_get_two_sockets_and_one_gets_the_same_socket_twice() -> None:
    """The name is a function of the data directory, which is the whole requirement.

    One server per data directory needs both halves: two directories must not collide onto one
    socket, and one directory must resolve to the same socket from every process that asks —
    otherwise a client and its server would name different files and neither would say why.
    """
    first = control.socket_path(Path("/data/alpha"))
    second = control.socket_path(Path("/data/beta"))
    assert first != second
    assert control.socket_path(Path("/data/alpha")) == first
    assert control.socket_path(Path("/data/alpha/../alpha")) == first, (
        "two spellings of one directory named two sockets"
    )


def test_the_runtime_directory_is_private_to_this_user() -> None:
    """``0700`` and ours, which is what the socket's own mode is resting on.

    A ``0600`` socket inside a directory anybody may write to is a ``0600`` socket somebody else
    can replace, so the directory is half of the protection rather than a detail of where files
    were put.
    """
    directory = control.runtime_dir()
    granted = directory.stat().st_mode & 0o777
    assert granted & 0o077 == 0, f"the runtime directory is {granted:04o}"
    assert directory.stat().st_uid == os.getuid()


async def test_the_socket_is_created_private_rather_than_narrowed_afterwards(
    socket_for: Callable[[], Path],
) -> None:
    """``0600`` from the moment it exists on the filesystem.

    Checked through the umask the server sets rather than only through the final mode, because
    the final mode is the same either way: a ``chmod`` after the bind leaves a window in which
    the socket is whatever the ambient umask allowed, and this asserts there is no such window
    by making the ambient umask permissive and finding the socket private anyway.
    """
    path = socket_for()
    permissive = os.umask(0o000)
    try:
        server = await serving(path, Echo())
    finally:
        os.umask(permissive)
    try:
        assert path.stat().st_mode & 0o777 == control.SOCKET_MODE
    finally:
        await server.aclose()


async def test_a_socket_with_wider_permissions_is_refused(
    socket_for: Callable[[], Path],
) -> None:
    """A client looks before it writes, because a permission nobody checks protects nothing.

    The socket carries write commands and a live session. One that anybody on the machine can
    connect to is refused with a message rather than used, and ``is_serving`` reports it as "no
    server" so the command line falls back to a message naming the fix.
    """
    path = socket_for()
    server = await serving(path, Echo())
    try:
        path.chmod(0o666)
        assert control.is_serving(path) is False
        with pytest.raises(control.ProtocolError, match="grants access to somebody other"):
            await control.connect(path, control.Invoke(op="doctor"), on_progress=lambda _: None)
    finally:
        await server.aclose()


# --- liveness -----------------------------------------------------------------------------------


def test_nothing_listening_is_not_serving(socket_for: Callable[[], Path]) -> None:
    """No socket at all is the ordinary case, and it is a boolean rather than an exception."""
    assert control.is_serving(socket_for()) is False


async def test_a_socket_file_left_behind_by_a_dead_server_is_not_serving(
    socket_for: Callable[[], Path],
) -> None:
    """A Unix socket outlives the process that bound it, so the file answers nothing.

    This is why the command line proxies on a connection rather than on a pid file: the file is
    present and complete here, and nothing is behind it.
    """
    path = socket_for()
    # Bound with a raw socket and closed without unlinking, which is what a process that was
    # killed leaves behind. `ControlServer.aclose` tidies up and asyncio's own server unlinks on
    # close, so neither of them can produce this state — and this state is the one the liveness
    # question exists for.
    abandoned = socketlib.socket(socketlib.AF_UNIX, socketlib.SOCK_STREAM)
    abandoned.bind(str(path))
    abandoned.listen(1)
    abandoned.close()
    assert path.exists(), "the file is still there, which is the point"
    assert control.is_serving(path) is False


async def test_a_stale_socket_file_does_not_stop_the_next_server(
    socket_for: Callable[[], Path],
) -> None:
    """The leftover is removed, because otherwise one unclean exit needs manual repair.

    A server that refused to start until somebody deleted a file would turn every crash into a
    support question, and the file carries no information worth preserving.
    """
    path = socket_for()
    path.touch()
    server = await serving(path, Echo())
    try:
        assert control.is_serving(path) is True
    finally:
        await server.aclose()


async def test_a_second_server_on_one_socket_refuses(socket_for: Callable[[], Path]) -> None:
    """Two servers is the state the instance lock exists to prevent, refused here as well.

    Belt and braces on purpose: the lock is what makes it true, and this is what makes it *say
    so* if the lock is ever taken somewhere the socket is not.
    """
    path = socket_for()
    first = await serving(path, Echo())
    try:
        with pytest.raises(control.ProtocolError, match="already listening"):
            await serving(path, Echo())
        assert control.is_serving(path) is True, "the first server is still the one serving"
    finally:
        await first.aclose()


async def test_closing_the_server_removes_the_socket(socket_for: Callable[[], Path]) -> None:
    """Nothing is left for the next start to reason about."""
    path = socket_for()
    server = await serving(path, Echo())
    await server.aclose()
    assert not path.exists()


# --- what crosses it ------------------------------------------------------------------------------


async def test_an_invocation_crosses_whole_and_comes_back_as_an_envelope(
    socket_for: Callable[[], Path],
) -> None:
    """The operation, its arguments and its workspace arrive as they were sent."""
    path = socket_for()
    handler = Echo()
    server = await serving(path, handler)
    try:
        envelope = await control.connect(
            path,
            control.Invoke(
                op="connector_sync", arguments={"name": "handbook", "limit": 5}, workspace="docs"
            ),
            on_progress=lambda _: None,
        )
    finally:
        await server.aclose()
    (seen,) = handler.seen
    assert isinstance(seen, control.Invoke)
    assert seen.op == "connector_sync"
    assert seen.arguments == {"name": "handbook", "limit": 5}
    assert seen.workspace == "docs"
    assert envelope["ok"] is True
    assert envelope["data"] == {"echoed": "invoke"}


async def test_progress_arrives_before_the_result_rather_than_with_it(
    socket_for: Callable[[], Path],
) -> None:
    """A long operation is visibly working, which is the whole reason progress exists.

    The handler is parked after reporting, so the assertion is that progress reached the client
    **while the operation was still running**. Batching it alongside the result would satisfy an
    ordering check and leave a sync silent for an hour, which is the failure this is about.
    """
    path = socket_for()
    handler = Echo(progress=("fetching 1 of 40", "fetching 2 of 40"))
    handler.hold = True
    server = await serving(path, handler)
    seen: list[str] = []
    both_arrived = asyncio.Event()

    def note(message: str) -> None:
        seen.append(message)
        if len(seen) == 2:
            both_arrived.set()

    try:
        call = asyncio.create_task(
            control.connect(path, control.Invoke(op="connector_sync"), on_progress=note)
        )
        await asyncio.wait_for(both_arrived.wait(), timeout=5)
        assert not call.done(), "the result had already arrived, so nothing was streamed"
        handler.released.set()
        envelope = await asyncio.wait_for(call, timeout=5)
    finally:
        handler.released.set()
        await server.aclose()
    assert seen == ["fetching 1 of 40", "fetching 2 of 40"]
    assert envelope["ok"] is True


async def test_a_second_command_is_answered_while_the_first_is_still_running(
    socket_for: Callable[[], Path],
) -> None:
    """A server answers more than one command at a time, and each gets its own progress.

    One connection is parked mid-operation and a second runs to completion underneath it. Both
    halves matter: a server that serialized connections would make every proxied command wait
    behind whatever sync happened to be running, and a server that crossed their streams would
    show one command's progress to the other.

    **It does not isolate the progress signal's scoping**, and the name is careful not to claim
    it does. That signal is per connection rather than per server, which is the right shape —
    but each connection also has its own ``pending`` list, so a shared event would cost spurious
    wakeups rather than lost lines, and there is no interleaving in which this test could tell
    the two apart. Verified by making it shared and watching this stay green.
    """
    path = socket_for()
    first = Echo(progress=("alpha one", "alpha two"))
    first.hold = True
    handler = _Routing({"alpha": first, "beta": Echo(progress=("beta one",))})
    server = await serving(path, handler)

    alpha: list[str] = []
    beta: list[str] = []
    both = asyncio.Event()

    def note_alpha(message: str) -> None:
        alpha.append(message)
        if len(alpha) == 2:
            both.set()

    try:
        held = asyncio.create_task(
            control.connect(path, control.Invoke(op="alpha"), on_progress=note_alpha)
        )
        await asyncio.wait_for(both.wait(), timeout=5)
        # The first command is parked with its progress delivered. The second runs to completion
        # underneath it, and must get its own.
        answered = await asyncio.wait_for(
            control.connect(path, control.Invoke(op="beta"), on_progress=beta.append), timeout=5
        )
        assert answered["ok"] is True
        assert beta == ["beta one"], beta
        assert not held.done(), "the parked command finished early, so nothing overlapped"
        first.released.set()
        await asyncio.wait_for(held, timeout=5)
    finally:
        first.released.set()
        await server.aclose()

    assert alpha == ["alpha one", "alpha two"], alpha


class _Routing:
    """Hands each request to the handler registered under its operation."""

    def __init__(self, handlers: dict[str, Echo]) -> None:
        self.handlers = handlers

    async def handle(
        self, request: control.Request, report: Callable[[str], None]
    ) -> dict[str, JsonValue]:
        return await self.handlers[getattr(request, "op", "")].handle(request, report)


async def test_a_failure_envelope_crosses_unchanged(socket_for: Callable[[], Path]) -> None:
    """A refusal is a result, not an exception, so it travels the same way a success does."""
    path = socket_for()

    class Refusing(Echo):
        @override
        async def handle(
            self, request: control.Request, report: Callable[[str], None]
        ) -> dict[str, JsonValue]:
            del request, report
            return {
                "version": "0",
                "op": "connector_sync",
                "ok": False,
                "workspace": "default",
                "data": None,
                "error": {
                    "type": "PolicyError",
                    "message": "enabled = false",
                    "hint": "Turn it on.",
                },
            }

    server = await serving(path, Refusing())
    try:
        envelope = await control.connect(
            path, control.Invoke(op="connector_sync"), on_progress=lambda _: None
        )
    finally:
        await server.aclose()
    assert envelope["ok"] is False
    assert envelope["error"] == {
        "type": "PolicyError",
        "message": "enabled = false",
        "hint": "Turn it on.",
    }


async def test_a_frame_that_is_not_this_protocol_is_answered_rather_than_dropped(
    socket_for: Callable[[], Path],
) -> None:
    """A client told nothing retries; a client told why does not.

    This is the path a version mismatch arrives on, so the connection is answered with a
    failure envelope instead of being closed under the caller.
    """
    path = socket_for()
    server = await serving(path, Echo())
    try:
        reader, writer = await asyncio.open_unix_connection(str(path))
        writer.write(b'{"kind": "sync-everything-now"}\n')
        await writer.drain()
        answer = json.loads(await asyncio.wait_for(reader.readuntil(b"\n"), timeout=5))
        writer.close()
    finally:
        await server.aclose()
    assert answer["envelope"]["ok"] is False
    assert "sync-everything-now" in answer["envelope"]["error"]["message"]


def test_a_frame_naming_no_known_kind_is_refused() -> None:
    """Parsed directly, because the refusal has to be a refusal rather than a KeyError."""
    with pytest.raises(control.ProtocolError, match="Known kinds"):
        control.read_request(b'{"kind": "run-arbitrary-code"}')


def test_a_line_that_is_not_json_is_refused() -> None:
    with pytest.raises(control.ProtocolError, match="not JSON"):
        control.read_request(b"connector sync handbook\n")


# --- the session, and everywhere it must not appear ------------------------------------------


def test_a_handover_carries_its_cookies_and_shows_nobody() -> None:
    """The one frame with a secret in it, and the two things that have to be true of it.

    It has to carry the value — a hand-off that lost it would be useless — and it has to show
    the value nowhere but the wire. ``SecretStr`` is what makes the second one structural: a
    repr, a log line and a traceback frame all render the wrapper.
    """
    frame = control.Handover(
        base_url="https://wiki.example.test",
        account="sync-account",
        captured_at="2026-08-14T10:00:00+00:00",
        cookies={"JSESSIONID": SecretStr("s3cr3t-value-abc")},
    )
    assert "s3cr3t-value-abc" not in repr(frame)
    assert "s3cr3t-value-abc" not in str(frame)
    assert "s3cr3t-value-abc" not in json.dumps(frame.model_dump(mode="json"))
    assert json.loads(frame.to_line())["cookies"] == {"JSESSIONID": "s3cr3t-value-abc"}, (
        "the wire is the one place the value is unwrapped, and it has to be unwrapped there"
    )


def test_a_handover_that_does_not_validate_is_refused_without_quoting_itself() -> None:
    """pydantic quotes the offending input, which is right everywhere but here.

    The frame carrying a live session is the frame whose validation error must not repeat it. A
    session that reached a terminal because a field was the wrong type would be a leak with no
    attacker in it at all.
    """
    line = json.dumps(
        {
            "kind": "handover",
            "base_url": "https://wiki.example.test",
            "account": "sync-account",
            "captured_at": "2026-08-14T10:00:00+00:00",
            "cookies": {"JSESSIONID": ["s3cr3t-value-abc"]},
        }
    ).encode()
    with pytest.raises(control.ProtocolError) as raised:
        control.read_request(line)
    assert "s3cr3t-value-abc" not in str(raised.value)
    assert "cookies.JSESSIONID" in str(raised.value)


async def test_a_handover_crosses_the_socket_with_its_value_intact(
    socket_for: Callable[[], Path],
) -> None:
    """The hand-off works, which is the half that the redaction tests cannot show."""
    path = socket_for()
    handler = Echo()
    server = await serving(path, handler)
    try:
        await control.connect(
            path,
            control.Handover(
                base_url="https://wiki.example.test",
                account="sync-account",
                captured_at="2026-08-14T10:00:00+00:00",
                cookies={"JSESSIONID": SecretStr("s3cr3t-value-abc")},
            ),
            on_progress=lambda _: None,
        )
    finally:
        await server.aclose()
    (seen,) = handler.seen
    assert isinstance(seen, control.Handover)
    assert seen.cookies["JSESSIONID"].get_secret_value() == "s3cr3t-value-abc"


# --- the client's own failures ----------------------------------------------------------------


async def test_connecting_to_nothing_is_a_refusal_a_caller_can_act_on(
    socket_for: Callable[[], Path],
) -> None:
    """Not an OSError out of a socket call: the command line turns this into an envelope."""
    with pytest.raises(control.ProtocolError, match="no manicule server is listening"):
        await control.connect(socket_for(), control.Invoke(op="doctor"), on_progress=lambda _: None)


async def test_an_unexpected_handler_failure_returns_one_private_safe_envelope(
    socket_for: Callable[[], Path],
) -> None:
    """A live accepted request never becomes EOF, and private exception text never crosses."""
    path = socket_for()
    private = "SELECT secret FROM /private/source?credential=never-print"
    handler = Echo(fail=RuntimeError(private))
    server = await serving(path, handler)
    try:
        envelope = await control.connect(
            path, control.Invoke(op="connector_sync"), on_progress=lambda _: None
        )
        assert envelope["ok"] is False
        assert envelope["error"] == {
            "type": "ControlOperationError",
            "message": "the served operation failed before producing its normal result",
            "hint": "Inspect aggregate lifecycle status, then retry the same operation.",
        }
        assert private not in json.dumps(envelope)
        handler.fail = None
        following = await control.connect(
            path, control.Invoke(op="connector_list"), on_progress=lambda _: None
        )
        assert following["ok"] is True
    finally:
        await server.aclose()
