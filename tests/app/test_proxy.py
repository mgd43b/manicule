"""A proxied write command and a local one are the same command, and this is what proves it.

**Equivalence is asserted against the in-process path, never against a fixture.** A test that
compared a proxied run to a string somebody wrote down would pass whenever the two agreed with
the string and say nothing about whether they agree with each other — and the failure it has to
catch is precisely a drift between the two. So every case here runs the *same* command both ways
against the *same* kind of service and compares the results to each other.

The comparison is made three ways because "the same" has three meanings to somebody using this:
the envelope (what a program reads), the terminal output (what a person reads), and the exit
status (what a shell branches on).
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

from manicule.app import commands, control
from manicule.app.commands import Command
from manicule.app.dispatch import run_op
from manicule.app.served import ControlHandler
from manicule.app.service import ApplicationService
from manicule.cli import main as cli
from manicule.cli import proxy
from manicule.connectors.sessions import SessionVault
from tests.app.fakes import FakeBackend, make_chunk, make_document

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from manicule.app.results import Envelope


@pytest.fixture
def socket_for() -> Iterator[Callable[[], Path]]:
    """A socket path no other test names, removed afterwards."""
    made: list[Path] = []

    def build() -> Path:
        path = control.socket_path(Path(f"/manicule-suite/{uuid.uuid4()}"))
        made.append(path)
        return path

    yield build
    for path in made:
        path.unlink(missing_ok=True)


def a_service() -> ApplicationService:
    """A service over a fake backend holding one document.

    A function rather than a fixture, because every equivalence case needs **two** of them —
    one for each side of the comparison — and two calls to a fixture is one object.
    """
    backend = FakeBackend()
    document = make_document(backend.workspace)
    backend.store.add(document, make_chunk(document))
    return ApplicationService(backend)


async def locally(service: ApplicationService, command: Command) -> Envelope:
    """The command run here, through the same binder a server would use."""
    return await run_op(
        command.op,
        service.workspace,
        lambda: commands.run(service, command, commands.silent),
    )


async def through_a_server(path: Path, service: ApplicationService, command: Command) -> Envelope:
    """The command run in a server on the other end of a real socket."""
    server = control.ControlServer(path, ControlHandler(service, SessionVault()))
    await server.start()
    try:
        return await proxy.forward(path, command, workspace=service.workspace)
    finally:
        await server.aclose()


def run(argv: Sequence[str]) -> Any:
    return CliRunner().invoke(cli.app, list(argv))


def bind_local(monkeypatch: pytest.MonkeyPatch, service: ApplicationService) -> None:
    """Make the command line run write commands here, as it does with no server."""

    async def dispatch(command: Command) -> Envelope:
        return await locally(service, command)

    monkeypatch.setattr(cli, "_dispatch", dispatch)


def bind_served(
    monkeypatch: pytest.MonkeyPatch, path: Path, service: ApplicationService
) -> control.ControlServer:
    """Make the command line find a server on ``path`` and use the real proxy to reach it.

    Only ``listening`` is substituted — the thing that would otherwise read this machine's
    configuration to find a data directory. ``_dispatch``, ``forward``, the socket, the frames
    and the handler are all the real ones, which is the point: what is being compared has to be
    the shipping path or the comparison is between a local run and a mock.
    """
    monkeypatch.setattr(proxy, "listening", lambda overrides: path)
    return control.ControlServer(path, ControlHandler(service, SessionVault()))


# --- the same command, both ways ----------------------------------------------------------------


DOCUMENT_ID = make_document(FakeBackend().workspace).id
"""The id the fake backend's one document gets, which is derived rather than random."""

CASES: list[tuple[str, Command, list[str]]] = [
    (
        "a document is deleted",
        Command("document_delete", {"document_id": DOCUMENT_ID, "hard": False}),
        ["document", "delete", DOCUMENT_ID],
    ),
    (
        "a document that is not there is reindexed",
        Command("document_reindex", {"document_id": "missing"}),
        ["document", "reindex", "missing"],
    ),
    (
        "a connector that is not configured is synced",
        Command("connector_sync", {"name": "nowhere", "limit": None}),
        ["connector", "sync", "nowhere"],
    ),
    (
        "the index is reset",
        Command("reset_index"),
        ["reset-index", "--yes"],
    ),
]
"""One success and three refusals, because equivalence has to hold for both.

The refusals are the half that a careless proxy gets wrong: an error crossing a socket is the
thing most likely to arrive re-wrapped, re-worded or re-typed, and a caller that branches on
``error.type`` would then branch differently depending on whether a server happened to be
running.

The success is a **delete** rather than the more obvious create, and the reason is worth
recording so nobody swaps it back. Every create in this vocabulary stamps a ``created_at`` from
the clock, so running one twice produces two payloads that differ by microseconds — and the two
ways of making that comparison pass are both bad. Blanking the field before comparing means the
test no longer checks the one part of the payload most likely to be mangled in transit, and
freezing the clock means asserting that two runs of a stubbed clock agree. A delete reports an
id, a boolean and a mode, all three derived from the request, so equality is equality.
"""


@pytest.mark.parametrize(("why", "command", "argv"), CASES, ids=[case[0] for case in CASES])
async def test_a_proxied_command_produces_the_same_envelope_as_a_local_one(
    why: str, command: Command, argv: list[str], socket_for: Callable[[], Path]
) -> None:
    """Compared field by field against the local run, not against anything written here."""
    del why, argv
    local = await locally(a_service(), command)
    proxied = await through_a_server(socket_for(), a_service(), command)

    assert proxied.ok == local.ok
    assert proxied.op == local.op
    assert proxied.workspace == local.workspace
    assert proxied.data == local.data
    assert proxied.error == local.error
    assert proxied.as_json() == local.as_json(), (
        "the envelope a program reads differs depending on whether a server is running"
    )


@pytest.mark.parametrize(("why", "command", "argv"), CASES, ids=[case[0] for case in CASES])
def test_a_proxied_command_prints_the_same_thing_and_exits_the_same_way(
    why: str,
    command: Command,
    argv: list[str],
    socket_for: Callable[[], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other two meanings of "the same": what a person reads and what a shell branches on.

    Driven through the real command line both times — real parsing, real ``submit``, real
    rendering — so what is compared is the whole surface rather than the envelope alone.
    """
    del why, command
    bind_local(monkeypatch, a_service())
    local = run(argv)

    path = socket_for()
    served = a_service()
    server = bind_served(monkeypatch, path, served)

    import asyncio  # noqa: PLC0415 - the server has to outlive one synchronous CLI call

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(server.start())
        proxied = run(argv)
    finally:
        loop.run_until_complete(server.aclose())
        loop.close()

    assert proxied.exit_code == local.exit_code, (
        f"a shell branches differently depending on whether a server is running: "
        f"{proxied.exit_code} proxied against {local.exit_code} local"
    )
    assert proxied.stdout == local.stdout, "a person reads something different"


async def test_a_proxied_failure_exits_non_zero_and_carries_its_message(
    socket_for: Callable[[], Path],
) -> None:
    """The message, the type and the hint, all the way from the server's own dispatch.

    Stated separately from the equivalence cases above because it is the property an operator
    depends on: a proxied refusal has to say what a local one would say, or the socket has
    turned a fixable problem into "something went wrong".
    """
    command = Command("connector_sync", {"name": "nowhere", "limit": None})
    local = await locally(a_service(), command)
    proxied = await through_a_server(socket_for(), a_service(), command)

    assert local.ok is False
    assert proxied.error is not None
    assert local.error is not None
    assert proxied.error.message == local.error.message
    assert "nowhere" in proxied.error.message, "the refusal names what was asked for"
    assert proxied.error.type == local.error.type
    assert proxied.error.hint == local.error.hint


async def test_progress_from_a_long_operation_reaches_the_caller_before_the_result(
    socket_for: Callable[[], Path],
) -> None:
    """A sync that says nothing for an hour is indistinguishable from one that has hung.

    The fake ingestion reports once per run, which is enough to prove the wiring end to end: the
    pipeline's ``watching`` callback, the binder that passes it, the handler's reporter, the
    frame, and the client's callback.
    """
    path = socket_for()
    service = a_service()
    service.settings.connectors["handbook"] = _configured_source()
    server = control.ControlServer(path, ControlHandler(service, SessionVault()))
    await server.start()
    seen: list[str] = []
    try:
        envelope = await control.connect(
            path,
            control.Invoke(op="connector_sync", arguments={"name": "handbook", "limit": None}),
            on_progress=seen.append,
        )
    finally:
        await server.aclose()

    assert envelope["ok"] is True
    assert seen, "a long operation reported nothing at all"
    assert all("handbook" in line for line in seen), seen


def _configured_source() -> Any:
    from manicule.config.settings import ConnectorSettings  # noqa: PLC0415

    return ConnectorSettings.model_validate({"type": "filesystem", "options": {"root": "."}})


# --- with no server ------------------------------------------------------------------------------


def test_a_write_command_with_no_server_refuses_and_names_the_fix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not a fallback to a local run, and not a daemon started on somebody's behalf."""
    monkeypatch.setattr(proxy, "listening", lambda overrides: None)
    result = run(["connector", "sync", "handbook"])

    assert result.exit_code == 1
    assert "manicule serve" in result.output, result.output
    assert "connector_sync" in result.output


async def test_a_socket_that_cannot_be_used_is_not_reported_as_no_server(
    socket_for: Callable[[], Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two different situations with two different fixes, and one message for both is a trap.

    **Found by running it rather than by reading it.** A server was running, its socket was
    chmod-ed wide, and the command line said "no manicule server is running ... start one with
    `manicule serve`" — which would have sent the operator to an instance-lock refusal naming a
    process they had just been told did not exist. The socket being unusable is about the
    socket; the process is fine.
    """
    path = socket_for()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("MANICULE_DATA_DIR", str(data_dir))
    monkeypatch.setattr(control, "socket_path", lambda _: path)

    server = control.ControlServer(path, ControlHandler(a_service(), SessionVault()))
    await server.start()
    try:
        path.chmod(0o666)
        envelope = proxy.refuse(
            Command("connector_sync", {"name": "handbook"}),
            workspace="default",
            overrides={},
        )
    finally:
        path.chmod(0o600)
        await server.aclose()

    assert envelope.error is not None
    message = envelope.error.message
    assert "no manicule server is running" not in message, message
    assert "cannot be used" in message, message
    assert "0666" in message, message
    assert "A server may well be running" in message, message


def test_a_write_command_with_no_server_starts_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asserted over process creation, because "it must not silently start a daemon" is a claim
    about what the machine has running afterwards rather than about a message.

    A command line that spawned a background writer the operator did not ask for would hold the
    lock, outlive the terminal, and be found later by somebody wondering what has the data
    directory. Every way this process could start another one is stubbed to fail loudly.
    """
    import os  # noqa: PLC0415
    import subprocess  # noqa: PLC0415 - only this assertion needs the module

    monkeypatch.setattr(proxy, "listening", lambda overrides: None)

    started: list[object] = []

    def refuse(*args: object, **kwargs: object) -> None:
        started.append(args)
        msg = "the command line started a process"
        raise AssertionError(msg)

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(os, "fork", refuse)
    monkeypatch.setattr(os, "posix_spawn", refuse)

    result = run(["connector", "sync", "handbook"])

    assert started == [], "a write command with no server started a process"
    assert result.exit_code == 1


def test_a_read_command_needs_no_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """The other half of the arrangement, and the one that must not regress.

    ``search``, ``ask`` and ``doctor`` take no writer lock and have no reason to want a server.
    If they started needing one, a machine whose server had stopped would stop answering
    questions as well as stop syncing — which is a far worse failure than the one this design
    accepts.
    """
    service = a_service()

    async def execute(op: str, call: Any) -> Envelope:
        return await run_op(op, service.workspace, lambda: call(service))

    monkeypatch.setattr(cli, "_execute", execute)

    def no_server(overrides: object) -> None:
        msg = "a read command asked whether a server was running"
        raise AssertionError(msg)

    monkeypatch.setattr(proxy, "listening", no_server)

    result = run(["--json", "document", "list"])

    assert result.exit_code == 0, result.output


def test_collection_orphans_needs_a_server_only_when_it_would_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one operation whose writing is a property of the invocation rather than the name.

    Listing documents that belong to no collection is a read. Trashing every one of them is not,
    and it used to run without the writer lock because the classification is by name — which,
    once a server owns the directory, would have meant deleting from a directory another process
    was writing to.
    """
    asked: list[str] = []
    monkeypatch.setattr(proxy, "listening", lambda overrides: (asked.append("asked"), None)[1])
    service = a_service()

    async def execute(op: str, call: Any) -> Envelope:
        return await run_op(op, service.workspace, lambda: call(service))

    monkeypatch.setattr(cli, "_execute", execute)
    bind_local(monkeypatch, service)

    listing = run(["--json", "collection", "orphans"])
    assert listing.exit_code == 0, listing.output
    assert asked == [], "listing orphans asked for a server it does not need"

    monkeypatch.undo()
    monkeypatch.setattr(proxy, "listening", lambda overrides: None)
    deleting = run(["collection", "orphans", "--confirm"])

    assert deleting.exit_code == 1
    assert "manicule serve" in deleting.output, deleting.output


# --- two writers cannot exist ---------------------------------------------------------------


def test_a_write_refuses_while_a_server_holds_the_lock_and_cannot_be_reached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The awkward middle state, and what an operator has to be told in it.

    A server is holding the data directory and its control socket is gone — the runtime
    directory was cleaned, or the socket's permissions changed under it. The command line cannot
    reach it, so it refuses and says to start one; starting one then refuses too, naming the
    process that has the directory. Neither refusal is wrong and neither is the whole story, so
    this pins both — the operator's path out is to find and stop the process the second message
    names.

    What must **not** happen is the write proceeding. Two writers on one directory is what
    #126's lock exists to prevent, and a command line that fell back to a local run when it
    could not find a server would be exactly that.
    """
    from manicule.core.errors import InstanceLockedError  # noqa: PLC0415
    from manicule.ingest.recovery import InstanceLock  # noqa: PLC0415

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("MANICULE_DATA_DIR", str(data_dir))
    # No socket, and the directory held: a server that is running and unreachable looks exactly
    # like this from outside.
    monkeypatch.setattr(proxy, "listening", lambda overrides: None)

    with InstanceLock(data_dir):
        refused = run(["connector", "sync", "handbook"])

        assert refused.exit_code == 1
        assert "manicule serve" in refused.output, refused.output

        with pytest.raises(InstanceLockedError) as second:
            InstanceLock(data_dir).acquire()

    assert "one instance per data directory" in str(second.value).lower()


async def test_a_second_control_server_on_one_data_directory_refuses(
    socket_for: Callable[[], Path],
) -> None:
    """The socket is the second place two servers are refused, after the lock.

    Belt and braces deliberately: the lock is what makes one-writer true, and this is what makes
    it *say so* if the lock is ever taken somewhere the socket is not. A second server binding
    over a live one's path would leave the first listening on a socket nothing can reach — a
    server that is running, holding the lock, and invisible.
    """
    path = socket_for()
    first = control.ControlServer(path, ControlHandler(a_service(), SessionVault()))
    await first.start()
    try:
        second = control.ControlServer(path, ControlHandler(a_service(), SessionVault()))
        with pytest.raises(control.ProtocolError, match="already listening"):
            await second.start()
        assert control.is_serving(path) is True, "the first server is still the one serving"
    finally:
        await first.aclose()
