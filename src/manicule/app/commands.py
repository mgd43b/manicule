"""An operation described as data, so the same description can run here or be sent to a server.

A served manicule holds the data directory's writer lock for its whole life, so a command that
writes has to reach that process rather than open the directory itself. The command line
therefore needs a way to say *what it wants done* that survives a socket — and a closure over an
:class:`~manicule.app.service.ApplicationService` does not.

**So a write command is a :class:`Command`: an operation's name and its arguments as JSON.**
Both paths then go through the same table. :func:`run` is the only place an operation's
arguments become a service call, whether the caller is the command line running in-process or a
server running one that arrived on the control socket. That is what makes a proxied command
faithful *structurally* rather than by two code paths agreeing: there is one code path, and the
socket only decides which process it runs in.

**Read commands are deliberately not here.** They keep the closure they always had, through
:func:`~manicule.cli.main.emit`, and the reason is not tidiness: ``ask`` takes an ``on_event``
callback so that a person watching a terminal sees tokens as they arrive, and a callback is not
JSON. A read command never takes the writer lock and never needs a server, so it has nothing to
gain from crossing a socket and something real to lose. The split is visible at every call site
— ``emit`` reads in this process, ``submit`` may write in another — and
``tests/app/test_commands.py`` holds the two sets to being exactly the operations the command
line has.

**Arguments are read strictly.** Every reader below refuses a value of the wrong shape by name
rather than coercing it, because the caller on the other side of a socket is a program and the
failure of a lenient reader is an operation that ran with an argument nobody sent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

from manicule.app import control
from manicule.app.dispatch import writes as writes_by_name
from manicule.ingest.reindex import DEFAULT_SWEEP_BATCH

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from pydantic import JsonValue

    from manicule.app.results import Payload
    from manicule.app.service import ApplicationService

__all__ = [
    "BINDERS",
    "Arguments",
    "Command",
    "Reporter",
    "run",
]

type Reporter = Callable[[str], None]
"""How a running operation says what it is doing, one sentence at a time.

It never blocks and never raises, so an operation does not have to reason about whether anybody
is listening. In-process there is nobody and it does nothing; over the socket it becomes a
:class:`~manicule.app.control.Progress` frame.
"""


def silent(message: str) -> None:
    """The reporter for a command running in this process, where progress is already on screen."""
    del message


@dataclass(frozen=True, slots=True)
class Arguments:
    """One operation's arguments, read by type and refused when they are not that type.

    The op's name travels with them so that a refusal says which operation was asked for. A
    message naming only the field sends somebody to the wrong command when two of them take a
    ``name``.
    """

    op: str
    values: Mapping[str, JsonValue] = field(default_factory=dict[str, "JsonValue"])

    def text(self, name: str) -> str:
        value = self.values.get(name)
        if not isinstance(value, str):
            self._refuse(name, "a string", value)
        return value

    def optional_text(self, name: str) -> str | None:
        value = self.values.get(name)
        if value is None:
            return None
        if not isinstance(value, str):
            self._refuse(name, "a string or nothing", value)
        return value

    def flag(self, name: str) -> bool:
        value = self.values.get(name, False)
        if not isinstance(value, bool):
            self._refuse(name, "true or false", value)
        return value

    def count(self, name: str, *, default: int) -> int:
        value = self.values.get(name)
        if value is None:
            return default
        # `bool` before `int`, because in Python a bool *is* an int and `--batch true` would
        # otherwise be accepted as a batch of one.
        if isinstance(value, bool) or not isinstance(value, int):
            self._refuse(name, "a whole number", value)
        return value

    def optional_count(self, name: str) -> int | None:
        value = self.values.get(name)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            self._refuse(name, "a whole number or nothing", value)
        return value

    def timestamp(self, name: str) -> datetime:
        value = self.text(name)
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{self.op} requires {name} to be an ISO-8601 timestamp") from exc
        if parsed.tzinfo is None:
            raise ValueError(f"{self.op} requires {name} to include a UTC offset")
        return parsed

    def path(self, name: str) -> Path:
        return Path(self.text(name))

    def optional_path(self, name: str) -> Path | None:
        value = self.optional_text(name)
        return None if value is None else Path(value)

    def texts(self, name: str) -> tuple[str, ...]:
        value = self.values.get(name, [])
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            self._refuse(name, "a list of strings", value)
        return tuple(str(item) for item in value)

    def _refuse(self, name: str, expected: str, value: JsonValue) -> NoReturn:
        """Refuse an argument, naming the type it was and never quoting the value.

        The type rather than the value on purpose. Nothing on this socket is a secret except a
        handover, which is a different frame entirely — but an argument reader is exactly the
        sort of thing that grows a new caller later, and a reader that has never quoted a value
        cannot start leaking one by being reused.

        **A ``ValueError`` rather than the ``TypeError`` the shape suggests**, and the choice is
        load-bearing rather than stylistic. :func:`~manicule.app.dispatch.run_op` turns
        ``ManiculeError``, ``ValueError`` and ``OSError`` into failure envelopes and lets
        everything else propagate as a defect. A client sending an argument of the wrong type —
        an older command line against a newer server, most likely — is an outcome the caller can
        act on and should read as one, not a traceback out of a socket handler.
        """
        msg = (
            f"{self.op} was given {name}={type(value).__name__}, and it takes {expected}. The "
            f"value is not repeated here."
        )
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Command:
    """One operation and its arguments, in a form that survives a socket.

    Constructed at the command line, run by :func:`run` — in this process when there is no
    server, and in the server's when there is.
    """

    op: str
    arguments: Mapping[str, JsonValue] = field(default_factory=dict[str, "JsonValue"])

    def writes(self) -> bool:
        """Whether this *invocation* needs the data directory's writer lock.

        Almost always this is :func:`~manicule.app.dispatch.writes` of the operation's name, and
        that stays the default so ``tests/app/test_process_exclusion.py`` keeps enumerating the
        vocabulary it always has: an operation nobody classified is a writer, which is the safe
        direction to be wrong in.

        ``collection_orphans`` is one operation whose answer is not a property of its name.
        Called plainly it lists live documents belonging to no collection, which is a read;
        called with ``delete`` it moves every one of them to the trash, which is not.
        Classifying it by name meant picking one, and the pick was "read" — so
        ``manicule collection orphans --confirm`` trashed documents *without* the writer lock,
        and once a server exists it would do that to a data directory another process owns.
        Naming it here is narrower than reclassifying the operation, which would make the
        listing take an exclusive lock it has no use for. Lifecycle operations have the same
        invocation boundary: their aggregate dry runs are reads, while applying the plan is a
        write. Each command routes its normal dry run through the read path; this check also
        keeps a serialized dry-run command honest when called directly.
        """
        if self.op == "collection_orphans":
            return Arguments(self.op, self.arguments).flag("delete")
        if self.op.startswith("lifecycle_") and Arguments(self.op, self.arguments).flag("dry_run"):
            return False
        return writes_by_name(self.op)

    def invoke(self, workspace: str) -> control.Invoke:
        """This command as the frame that carries it to a server."""
        return control.Invoke(op=self.op, arguments=dict(self.arguments), workspace=workspace)


async def run(service: ApplicationService, command: Command, report: Reporter) -> Payload:
    """Run one command against a service, and return what it produced.

    The single definition of what an operation *is*, shared by the in-process path and the
    server's socket handler.

    Raises:
        ValueError: The command names an operation with no binder, or carries an argument of
            the wrong shape. Both reach a caller as a failure envelope through
            :func:`~manicule.app.dispatch.run_op`, which is right for either: an unknown
            operation is what a client one version ahead of its server sends.
    """
    binder = BINDERS.get(command.op)
    if binder is None:
        known = ", ".join(sorted(BINDERS))
        msg = (
            f"no operation named {command.op!r} can be run this way. This is the list of "
            f"operations that write, and it is what a server accepts on its control socket: "
            f"{known}."
        )
        raise ValueError(msg)
    return await binder(service, Arguments(command.op, command.arguments), report)


type Binder = Callable[[ApplicationService, Arguments, Reporter], Awaitable["Payload"]]

BINDERS: Mapping[str, Binder] = {
    "auth_create_key": lambda service, args, report: service.api_key_create(
        args.text("name"),
        role=args.text("role"),
        expires_days=args.optional_count("expires_days"),
    ),
    # `role` is `text` rather than `optional_text` deliberately: the command line declares a
    # default of "member", so it always sends a string, and a reader that accepted nothing
    # would let a caller mint a key with no role at all.
    "auth_revoke_key": lambda service, args, report: service.api_key_revoke(
        args.text("name_or_id")
    ),
    "collection_add": lambda service, args, report: service.collection_add(
        args.text("collection_id"), args.texts("document_ids")
    ),
    "collection_create": lambda service, args, report: service.collection_create(
        args.text("name"), description=args.optional_text("description")
    ),
    "collection_delete": lambda service, args, report: service.collection_delete(
        args.text("collection_id")
    ),
    "collection_orphans": lambda service, args, report: service.collection_orphans(
        delete=args.flag("delete")
    ),
    "collection_remove": lambda service, args, report: service.collection_remove(
        args.text("collection_id"), args.texts("document_ids")
    ),
    "collection_rename": lambda service, args, report: service.collection_rename(
        args.text("collection_id"), args.text("name")
    ),
    "collection_update": lambda service, args, report: service.collection_update(
        args.text("collection_id"), description=args.text("description")
    ),
    "config_set": lambda service, args, report: service.config_set(
        args.text("key"), args.text("value")
    ),
    "connector_sidecar": lambda service, args, report: service.connector_sidecar(
        args.optional_path("root"), source=args.text("source"), force=args.flag("force")
    ),
    "connector_sync": lambda service, args, report: service.connector_sync(
        args.text("name"),
        limit=args.optional_count("limit"),
        watching=report,
        acquire_only=args.flag("acquire_only"),
    ),
    "document_delete": lambda service, args, report: service.document_delete(
        args.text("document_id"), hard=args.flag("hard")
    ),
    "document_redetect_glossary": lambda service, args, report: service.document_redetect_glossary(
        batch=args.count("batch", default=DEFAULT_SWEEP_BATCH), dry_run=args.flag("dry_run")
    ),
    "document_reindex": lambda service, args, report: service.document_reindex(
        args.text("document_id")
    ),
    "document_reindex_stale": lambda service, args, report: service.document_reindex_stale(
        batch=args.count("batch", default=DEFAULT_SWEEP_BATCH), dry_run=args.flag("dry_run")
    ),
    "reembed_plan": lambda service, args, report: service.reembed_plan(),
    "reembed_start": lambda service, args, report: service.reembed_start(args.text("run_id")),
    "reembed_resume": lambda service, args, report: service.reembed_resume(args.text("run_id")),
    "reembed_abandon": lambda service, args, report: service.reembed_abandon(args.text("run_id")),
    "reembed_cleanup": lambda service, args, report: service.reembed_cleanup(args.text("run_id")),
    "import": lambda service, args, report: service.import_corpus(
        args.path("path"), force=args.flag("force")
    ),
    "index_path": lambda service, args, report: service.index_path(
        args.path("path"),
        source=args.text("source"),
        limit=args.optional_count("limit"),
        force=args.flag("force"),
        watching=report,
    ),
    "init": lambda service, args, report: service.initialize(force=args.flag("force")),
    "lifecycle_reset_derived": lambda service, args, report: service.lifecycle_reset_derived(
        dry_run=args.flag("dry_run")
    ),
    "lifecycle_cleanup_generations": lambda service, args, report: (
        service.lifecycle_cleanup_generations(dry_run=args.flag("dry_run"))
    ),
    "lifecycle_release_history": lambda service, args, report: service.lifecycle_release_history(
        args.timestamp("cutoff"), dry_run=args.flag("dry_run")
    ),
    "lifecycle_delete_snapshot": lambda service, args, report: service.lifecycle_delete_snapshot(
        args.text("run_id"),
        confirmation=args.optional_text("confirmation"),
        dry_run=args.flag("dry_run"),
    ),
    "plugin_add": lambda service, args, report: service.plugin_add(args.text("name")),
    "plugin_remove": lambda service, args, report: service.plugin_remove(args.text("name")),
    "reset_index": lambda service, args, report: service.reset_index(),
    "restore": lambda service, args, report: service.restore(
        args.path("source"), force=args.flag("force")
    ),
    "upgrade": lambda service, args, report: service.upgrade(
        version=args.optional_text("version"), skip_backup=args.flag("skip_backup")
    ),
    "workspace_switch": lambda service, args, report: service.workspace_switch(
        args.text("name"), create=args.flag("create")
    ),
}
"""Every operation that can be described as data, and how each becomes a service call.

**This is also the server's accept list.** A control socket that took a method name and called
it would be arbitrary invocation behind a file permission; this table is what makes the socket's
vocabulary a fixed set somebody wrote down. An operation absent from here cannot be asked for
over it, whatever the caller sends.

Alphabetical by operation, one entry each, and each entry is the call that used to sit at the
command-line call site — moved rather than duplicated, so there is still exactly one place that
says what ``connector_sync`` does with its arguments.

``report`` is accepted by every binder and used by the two operations that can run for a long
time. A uniform signature rather than two kinds of binder: an operation that grows a progress
report later should not also have to change shape to get one.
"""
