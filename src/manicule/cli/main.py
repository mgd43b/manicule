"""The command line: nineteen commands, each a few lines over the application service.

Typer derives every option and argument from the type hints, so a command's signature and the
interface a person types cannot drift apart. What remains here is parsing, one service call,
and rendering — no command decides anything, and none of them reaches past
:class:`~manicule.app.service.ApplicationService`.

Three things hold everywhere:

**``--json`` on everything that emits data, in either position.** ``manicule --json doctor``
and ``manicule doctor --json`` are the same invocation. The same envelope the MCP tools return,
printed to stdout with nothing else in it. Human output and failures go to stderr, so ``| jq``
on a failed run reads an empty stream rather than a prose error.

**Failures exit non-zero and say what to do.** A shell script can branch on that; a person
gets the hint.

**Nothing binds a socket unless it is asked to, and never widely by accident.** ``start``
serves MCP over stdio by default, and every other transport goes through
:func:`~manicule.app.bind.resolve_bind`.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import typer
from typer.core import TyperGroup, TyperOption

from manicule.app import commands
from manicule.app import results as r
from manicule.app.commands import Command
from manicule.app.dispatch import error_info, run_op, writes
from manicule.app.results import Envelope, failed
from manicule.app.runtime import Runtime
from manicule.app.service import DEFAULT_SOURCE, DEFAULT_SWEEP_BATCH, ApplicationService
from manicule.cli import proxy, render
from manicule.core.errors import ConfigError, ManiculeError
from manicule.core.version import CORE_VERSION
from manicule.generation.answers import EventKind

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pydantic import JsonValue

    from manicule.app.results import Payload
    from manicule.generation.answers import AnswerEvent

RESET_NEEDS_CONFIRMATION = (
    "this removes the workspace's derived search state while retaining durable source snapshots. "
    "Search remains unavailable until an offline rebuild completes. Pass --yes to confirm."
)
"""Why ``reset-index`` refuses without ``--yes``.

A constant rather than a literal at the raise site, so a test can assert that the refusal
names the flag without reading it back out of a rendered terminal box — which wraps, colors
and elides differently on every machine.
"""

BACKUP_NEEDS_A_TARGET = "pass --output to say where the backup goes"

BACKUP_IS_NOT_A_RESTORE = (
    "pass either --output to take a backup or --restore to put one back, not both"
)

INSECURE_TARGET_IS_A_BACKUP_OPTION = (
    "--allow-insecure-target applies to --output, which is where a backup is written. A "
    "restore reads from the directory you name and writes into the data directory, whose "
    "permissions manicule sets itself."
)
"""Why ``--restore --allow-insecure-target`` is refused rather than ignored.

Accepting a security flag that has no effect on the operation being run is the exact shape of
the defect this option exists to fix: an option that reads as a decision and reaches nothing.
"""

REINDEX_IS_ONE_OR_ALL = (
    "name a document id or pass --stale, not both. --stale sweeps the whole corpus for "
    "documents an installed parser has moved past, and an id says which single document to "
    "re-parse; running the sweep against one id would be a much larger operation than the "
    "argument asks for."
)
"""Why ``document reindex <id> --stale`` is refused rather than resolved.

The same shape as :data:`BACKUP_IS_NOT_A_RESTORE` and refused for the same reason: two
contradictory instructions in one invocation is a typo, not a plan. This one is worth refusing
loudly because the two readings differ by the size of the corpus — a person who meant one
document and got a sweep has committed the machine to re-parsing and re-embedding everything a
parser bump touched, and would find out from the elapsed time.
"""

REINDEX_NEEDS_A_TARGET = (
    "say what to re-parse: a document id, --stale for every document whose parse fingerprint "
    "is no longer one an installed parser produces, or --stale-glossary for every document "
    "whose glossary came from a detector this build has changed"
)

REINDEX_IS_ONE_RUNG = (
    "pass --stale or --stale-glossary, not both. They are different repairs at different "
    "prices: --stale re-parses from retained bytes and re-embeds whatever moves, which is "
    "corpus-sized GPU work, and --stale-glossary reads the chunks already stored and runs "
    "regular expressions over them. Running them together would charge the second's job at "
    "the first's price with no way to see which one did what."
)
"""Why ``--stale --stale-glossary`` is refused rather than run in sequence.

The same shape as :data:`REINDEX_IS_ONE_OR_ALL`, and refused for the sharper of its two
reasons. There the two readings differ by the size of the corpus; here they differ by whether
the machine spends an afternoon embedding. A person who meant the cheap repair and got both
finds out from the elapsed time, which is the one way nobody should have to find out.

They are not otherwise exclusive, and the ordering when both are wanted is stated rather than
implied: run ``--stale`` first. A re-parse re-runs detection on every document it rebuilds, so
doing it second would redo work the glossary sweep had just finished.
"""

GLOSSARY_IS_NOT_A_DOCUMENT_SWEEP = (
    "--stale-glossary sweeps the whole corpus for documents whose glossary came from a "
    "superseded detector; an id names one document to re-parse from its retained bytes. "
    "Re-parsing one document already re-runs detection on it, so there is nothing the pair "
    "could mean that the id alone does not already do."
)

DRY_RUN_IS_A_SWEEP_OPTION = (
    "--dry-run applies to --stale and --stale-glossary, which select documents before "
    "repairing them. Re-parsing one named document has no selection to plan, so a dry run of "
    "it would report nothing and look like a run that found nothing to do."
)
"""Why ``document reindex <id> --dry-run`` is refused rather than ignored.

Accepting a flag that has no effect on the operation being run is the defect
:data:`INSECURE_TARGET_IS_A_BACKUP_OPTION` exists to name, one command along. Here it is worse
than inert: the flag reads as "show me what this would do", and silently doing it instead is
the one outcome the person typing it was trying to avoid.
"""

INSECURE_STATE_IS_AN_IMPORT_OPTION = (
    "--allow-insecure-state applies to --browser-state, which reads a file whose permissions "
    "decide whether importing it is safe. --browser opens a browser and takes cookies from it "
    "in memory; there is no file for the flag to consent to."
)
"""Why ``--browser --allow-insecure-state`` is refused rather than ignored.

The same rule :data:`INSECURE_TARGET_IS_A_BACKUP_OPTION` states, and it bites hardest on a
*security* flag: an option that reads as "I have considered this risk and accept it" and reaches
nothing is the precise defect the option exists to prevent. Found by re-reading the diff against
the convention rather than by anything failing.
"""

BROWSER_TIMEOUT_IS_A_BROWSER_OPTION = (
    "--timeout is how long to wait for a person to finish signing in, which only the --browser "
    "path waits for. Importing a state file and forgetting a session are both immediate, and "
    "pasting a Cookie header waits on you rather than on a clock."
)
"""Why ``--timeout`` without ``--browser`` is refused rather than ignored.

Milder than the one above — nothing unsafe follows from it — and refused on the same principle,
because a person who passed a timeout and watched the command return instantly has been told
something untrue about what it did.
"""

BROWSER_TIMEOUT_MUST_BE_POSITIVE = (
    "--timeout is how many seconds to wait for sign-in, so it has to be more than zero. "
    "browser_timeout_seconds is constrained the same way."
)
"""Why ``--timeout 0`` is refused rather than treated as "use the default".

Zero is falsy, and the first version of this folded it into the configured default — so somebody
who asked to wait no time would have waited five minutes instead. Refusing is the only reading
that invents nothing: neither "you meant the default" nor "you meant to give up immediately" is
something the command should decide on their behalf.
"""

UNKNOWN_WORKSPACE = "unknown"
"""What an envelope reports when configuration could not be loaded at all.

The workspace is on every envelope including failures, and a configuration that will not load
is precisely the case where manicule does not know which one it was going to serve. Saying so
beats guessing ``default``, which would be a claim.
"""


@dataclass
class State:
    """What the root callback parsed, for the command that runs next."""

    json_output: bool = False
    workspace: str | None = None
    overrides: dict[str, Any] = field(default_factory=dict[str, Any])
    text_already_streamed: bool = False
    """Whether ``ask`` has already written the answer to the terminal as it arrived.

    Set by the one command that streams, read by the one renderer that would otherwise repeat
    itself. It is not a rendering *decision* — the payload is identical either way — it is a
    record of what has already reached the screen.
    """


STATE = State()

JSON_HELP = "Emit the result envelope as JSON on stdout, and nothing else."
"""One sentence for ``--json``, wherever it is declared.

Both declarations are the same option, so two spellings of the help would be two answers to
``--help`` depending on where the reader asked.
"""

WORKSPACE_HELP = "Run in this workspace instead of the configured one."
"""One sentence for ``--workspace``, for the same reason :data:`JSON_HELP` is one sentence."""

WORKSPACE_NAMED_TWICE = (
    "--workspace was given twice with different values, {before!r} before the command and "
    "{after!r} after it. Name one workspace: manicule cannot run an operation in both, and "
    "choosing between them silently would run it in a tenant you also named and did not get."
)
"""Why two *different* workspaces in one invocation is refused rather than resolved.

A constant rather than a literal at the raise site, so a test can assert the refusal names both
values without reading them back out of a rendered terminal box, which wraps, colors and
elides differently on every machine.

**This is the one place the shared options behave differently, and the difference is the
point.** ``--json`` is a flag: naming it twice says the same thing twice, and the two positions
cannot disagree. ``--workspace`` carries a value, across a *tenancy boundary*. Last-wins would
be defensible if the positions meant "general" then "specific" — but by construction this is
the same option in two places, so there is no specificity to appeal to, and picking one would
run the operation in a workspace the operator also named while the envelope reported the winner
as though it were the whole request. A wrong-tenant run that looks exactly like a correct one
is the worst failure available in a system where scope is an auditability property and
cross-workspace access is a 5xx.

The same shape as :data:`BACKUP_IS_NOT_A_RESTORE`, and refused for the same reason: two
contradictory instructions in one invocation is a typo, not a plan.

Identical values are accepted. They are unambiguous, and refusing them would break the case
that makes ``--json`` twice acceptable — a script that already passes the option, and somebody
adding it at the other end.
"""


def _accept_json(ctx: object, param: object, value: bool) -> bool:
    """Record ``--json`` typed *after* the command name.

    Click invokes the group's callback before it parses the subcommand's arguments, so
    :func:`main_callback` has already written ``STATE.json_output`` by the time this runs. That
    is why this **or**\\ s rather than assigns: ``manicule --json doctor --json`` names the same
    intention twice, and an assignment would let the second, defaulted-to-``False`` position
    quietly cancel the first.
    """
    del ctx, param
    STATE.json_output = STATE.json_output or value
    return value


def _accept_workspace(ctx: object, param: object, value: str | None) -> str | None:
    """Record ``--workspace`` typed *after* the command name, or refuse a contradiction.

    Same ordering as :func:`_accept_json` — the root callback has already run — but a different
    rule, for the reason :data:`WORKSPACE_NAMED_TWICE` gives: a flag cannot disagree with
    itself and a value can.

    ``None`` means the option was not given here, which is distinguishable from every real
    workspace name, so absent and given never have to be told apart by guesswork.

    Raises:
        typer.BadParameter: Both positions named a workspace and they differ. A usage error, so
            it exits 2 alongside everything else Typer rejects before the service is reached,
            rather than inventing a fourth exit status for this one refusal.
    """
    del ctx, param
    if value is None:
        return value
    if STATE.workspace is not None and STATE.workspace != value:
        raise typer.BadParameter(WORKSPACE_NAMED_TWICE.format(before=STATE.workspace, after=value))
    STATE.workspace = value
    # Merged rather than replaced. `overrides` is splatted into `Runtime.open`, so it is a bag
    # that can hold more than one key — and the next value-carrying shared option, written by
    # analogy with this line, would silently drop the workspace on its way past.
    STATE.overrides = {**STATE.overrides, "workspace": value}
    return value


SHARED_OPTIONS: dict[str, Callable[[], TyperOption]] = {
    "--json": lambda: TyperOption(
        param_decls=["--json"],
        is_flag=True,
        default=False,
        expose_value=False,
        callback=_accept_json,
        help=JSON_HELP,
    ),
    "--workspace": lambda: TyperOption(
        param_decls=["--workspace", "-w"],
        default=None,
        expose_value=False,
        callback=_accept_workspace,
        help=WORKSPACE_HELP,
    ),
}
"""The root callback's options that every command accepts too, keyed by their long name.

A fresh option per command rather than one shared object: a Click parameter belongs to the
command holding it, and handing one instance to thirty-one commands is the kind of sharing that
works right up until something keeps state on it.

Public, because it is half of an accounting against the root callback's real signature. Every
option ``manicule`` declares is either named here or carries a written reason for not being,
and ``tests/app/test_cli.py`` walks the built tree to check — so a fourth root option added
tomorrow fails until somebody classifies it, rather than quietly becoming the next thing that
cannot be typed where people type it.
"""


class CommandsShareTheRootOptions(TyperGroup):
    """A command group whose leaf commands also accept the options declared on ``manicule``.

    ``--json`` was declared once, on the root callback. That made ``manicule --json doctor``
    work and ``manicule doctor --json`` an *unknown option* — exit 2, no output, from an
    interface whose own module docstring said ``--json`` was on everything that emits data. It
    is the position everybody types first, it was reported as a missing feature, and it had
    been papered over twice: once by correcting the example in ``docs/surfaces.md`` that
    happened to use it, and once by writing the restriction up in the README as though somebody
    had chosen it.

    Attaching the option here rather than to thirty-odd command signatures is what keeps the
    two positions from drifting: there is one option object per command, built from one
    declaration, so a command added tomorrow accepts ``--json`` tomorrow without anybody
    remembering. ``tests/app/test_cli.py`` derives the set of data-emitting commands from the
    built command tree and asserts both positions for every one of them.

    ``expose_value=False`` is what makes it possible at all. Typer generates each command's
    callback from its Python signature, so an option Click passed through as a keyword argument
    would be a ``TypeError`` on a parameter the function never declared. The value reaches
    :data:`STATE` through the option's callback instead, which is where the root's copy already
    puts it.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401 - Click's own signature
        super().__init__(*args, **kwargs)
        for command in self.commands.values():
            # Groups are skipped: `manicule document` emits nothing itself, and its own leaves
            # are handled when that group is constructed — by this same class, because every
            # sub-application below declares it too.
            if isinstance(command, TyperGroup):
                continue
            for build in SHARED_OPTIONS.values():
                option = build()
                # Any overlap at all, not just the long name. A command declaring `-w` for
                # something of its own would otherwise end up with two parameters answering to
                # one string, which Click resolves by silently preferring one of them. Skipping
                # leaves that command without the shared option, which the accounting in
                # `tests/app/test_cli.py` fails on rather than tolerates.
                taken = {name for existing in command.params for name in existing.opts}
                if taken & set(option.opts):
                    continue
                command.params.append(option)


app = typer.Typer(
    name="manicule",
    help="Self-hosted document search and answers, with citations that resolve.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
    cls=CommandsShareTheRootOptions,
)
document_app = typer.Typer(
    help="Inspect and manage indexed documents.",
    no_args_is_help=True,
    cls=CommandsShareTheRootOptions,
)
connector_app = typer.Typer(
    help="Configured sources.", no_args_is_help=True, cls=CommandsShareTheRootOptions
)
workspace_app = typer.Typer(
    help="Workspaces, and which one is active.",
    no_args_is_help=True,
    cls=CommandsShareTheRootOptions,
)
plugin_app = typer.Typer(
    help="Installed plugins.", no_args_is_help=True, cls=CommandsShareTheRootOptions
)
config_app = typer.Typer(
    help="Read and write configuration.", no_args_is_help=True, cls=CommandsShareTheRootOptions
)
auth_app = typer.Typer(
    help="API keys for this workspace.", no_args_is_help=True, cls=CommandsShareTheRootOptions
)
collection_app = typer.Typer(
    help="Named sets of documents.", no_args_is_help=True, cls=CommandsShareTheRootOptions
)
reembed_app = typer.Typer(
    help="Offline, resumable whole-index embedding migration.",
    no_args_is_help=True,
    cls=CommandsShareTheRootOptions,
)
app.add_typer(document_app, name="document")
app.add_typer(collection_app, name="collection")
app.add_typer(connector_app, name="connector")
app.add_typer(workspace_app, name="workspace")
app.add_typer(plugin_app, name="plugin")
app.add_typer(config_app, name="config")
app.add_typer(auth_app, name="auth")
app.add_typer(reembed_app, name="reembed")


JsonOption = Annotated[bool, typer.Option("--json", help=JSON_HELP)]
WorkspaceOption = Annotated[str | None, typer.Option("--workspace", "-w", help=WORKSPACE_HELP)]


def _show_version(value: bool) -> None:
    """Print the version and stop, before any command runs.

    Written to the stream rather than printed through Rich: a version is read by scripts far
    more often than by people, and it should be one line with nothing around it.
    """
    if value:
        sys.stdout.write(f"{CORE_VERSION}\n")
        raise typer.Exit


@app.callback()
def main_callback(
    json_output: JsonOption = False,
    workspace: WorkspaceOption = None,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Print the installed version and exit.",
            callback=_show_version,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Options every command shares."""
    del version  # handled by its eager callback, before any command runs
    STATE.json_output = json_output
    STATE.workspace = workspace
    STATE.overrides = {"workspace": workspace} if workspace else {}
    STATE.text_already_streamed = False


# --- running one operation --------------------------------------------------------------------


async def _execute(op: str, call: Callable[[ApplicationService], Awaitable[Payload]]) -> Envelope:
    """Open a runtime, run one operation, close it. Never leaks an exception a caller owns.

    **This is the read path.** Every operation reaching it is named in
    :data:`~manicule.app.dispatch.READ_ONLY_OPS`, so the runtime is opened without the writer
    lock and no server has to exist for it to work — which is the property that keeps ``search``,
    ``ask`` and ``doctor`` answering while a sync is running somewhere else. Operations that
    write arrive through :func:`submit` instead.

    The ``async with`` is inside the ``try`` because a runtime can fail to open for reasons that
    have nothing to do with the lock. Configuration that will not load is exactly as much an
    outcome a caller can act on, and it gets the same envelope rather than a traceback.
    """
    try:
        runtime = Runtime.open(writer=writes(op), **STATE.overrides)
        async with runtime:
            service = ApplicationService(runtime)
            return await run_op(op, service.workspace, lambda: call(service))
    except (ManiculeError, ValueError, OSError) as exc:
        # Configuration that will not load is the one failure that happens before there is a
        # workspace to report, and it is also the commonest. It gets the same envelope as
        # everything else rather than a traceback.
        return failed(op, STATE.workspace or UNKNOWN_WORKSPACE, error_info(exc))


def emit(op: str, call: Callable[[ApplicationService], Awaitable[Payload]]) -> None:
    """Run a read operation in this process, print it, and set the exit status."""
    print_envelope(asyncio.run(_execute(op, call)))


def submit(command: Command) -> None:
    """Run a command that may write — here or in the server — print it, and set the exit status.

    The counterpart to :func:`emit`, and the difference between the two is which process the
    work happens in. A reader can tell at every call site which is which, which is the point of
    there being two functions rather than one that branches.
    """
    print_envelope(asyncio.run(_dispatch(command)))


async def _dispatch(command: Command) -> Envelope:
    """Decide where a command runs, and run it there.

    Three outcomes, and the middle one is the change this whole arrangement is for:

    * It does not write — run it here, with no lock and no server. ``collection orphans``
      without ``--confirm`` is the case that makes this a property of the *invocation* rather
      than of the operation's name.
    * It writes and a server is answering — send it, and print what comes back.
    * It writes and nothing is answering — refuse, naming ``manicule serve``. Not a local
      fallback: the credential a sync needs lives in the server's memory and nowhere else, so a
      local run would be one that cannot authenticate, reported as a sync that failed.

    ``init`` reaches the first branch rather than being special-cased here. It writes the
    configuration file and the cache and touches no data directory, so
    :data:`~manicule.app.dispatch.READ_ONLY_OPS` names it and ``writes()`` answers ``False`` —
    which is what keeps ``manicule init`` from requiring a server before there is anything for
    one to serve.
    """
    if not command.writes():
        return await _locally(command)
    served = proxy.listening(STATE.overrides)
    if served is None:
        return proxy.refuse(
            command, workspace=STATE.workspace or UNKNOWN_WORKSPACE, overrides=STATE.overrides
        )
    return await proxy.forward(served, command, workspace=STATE.workspace or UNKNOWN_WORKSPACE)


async def _locally(command: Command) -> Envelope:
    """Run a command in this process, taking the writer lock only if it needs one."""
    try:
        runtime = Runtime.open(writer=command.writes(), **STATE.overrides)
        async with runtime:
            service = ApplicationService(runtime)
            return await run_op(
                command.op,
                service.workspace,
                lambda: commands.run(service, command, commands.silent),
            )
    except (ManiculeError, ValueError, OSError) as exc:
        return failed(command.op, STATE.workspace or UNKNOWN_WORKSPACE, error_info(exc))


def print_envelope(envelope: Envelope) -> None:
    """Write one result the way the caller asked for, and set the exit status.

    Public because three commands produce an envelope without going through :func:`emit` —
    ``completion`` needs no runtime, and ``start`` and ``stop`` produce theirs elsewhere — and
    a second copy of this branching in each of them is three places to forget that a failure
    goes to stderr.
    """
    if STATE.json_output:
        # stdout carries the envelope and nothing else, so a pipe is parseable whether the
        # operation succeeded or not.
        sys.stdout.write(json.dumps(envelope.as_json(), indent=2, sort_keys=True) + "\n")
        if not envelope.ok:
            raise typer.Exit(1)
        return
    if envelope.data is not None:
        payload = PAYLOADS[envelope.op].model_validate(envelope.data)
        console = render.console()
        if isinstance(payload, r.AnswerResultPayload):
            console.print()
            render.render_answer(console, payload, text_already_shown=STATE.text_already_streamed)
            if envelope.ok:
                return
            raise typer.Exit(1)
        # `stop` and `start` share a payload — an address is an address — and only the
        # operation says which way the server was going. Without this, stopping one printed
        # the banner that announces one, down to the URL of a browser surface that had just
        # gone away.
        if isinstance(payload, r.ServerAddress) and envelope.op == "stop":
            render.render_address(console, payload, stopped=True)
            if envelope.ok:
                return
            raise typer.Exit(1)
        render.render(console, payload)
        if envelope.ok:
            return
        raise typer.Exit(1)
    if envelope.error is not None:
        render.render_error(render.console(stderr=True), envelope.op, envelope.error)
    raise typer.Exit(1)


PAYLOADS: dict[str, type[Payload]] = {
    "ask": r.AnswerResultPayload,
    "search": r.SearchResult,
    "index_path": r.IngestReport,
    "index_changes": r.IngestReport,
    "index_status": r.IndexStatus,
    "stats": r.Stats,
    "document_list": r.DocumentList,
    "document_get": r.DocumentDetail,
    "document_delete": r.DocumentDeleted,
    "document_reindex": r.DocumentReindexed,
    "document_reindex_stale": r.StaleReparseReport,
    "document_redetect_glossary": r.StaleGlossaryReport,
    "reembed_plan": r.ReembedPlanReport,
    "reembed_start": r.ReembedRunReport,
    "reembed_resume": r.ReembedRunReport,
    "reembed_status": r.ReembedRunReport,
    "reembed_abandon": r.ReembedRunReport,
    "reembed_cleanup": r.ReembedCleanupReport,
    "lifecycle_reset_derived": r.LifecycleReport,
    "lifecycle_cleanup_generations": r.LifecycleReport,
    "lifecycle_release_history": r.LifecycleReport,
    "lifecycle_delete_snapshot": r.LifecycleReport,
    "doctor": r.Diagnosis,
    "connector_list": r.ConnectorList,
    "snapshot_status": r.SnapshotStatusReport,
    "snapshot_verify": r.SnapshotStatusReport,
    "connector_login": r.ConnectorSignedIn,
    "connector_sidecar": r.SidecarReport,
    "connector_sync": r.IngestReport,
    "config_get": r.ConfigValue,
    "config_set": r.ConfigChange,
    "workspace_list": r.WorkspaceList,
    "workspace_switch": r.WorkspaceSwitched,
    "plugin_list": r.PluginList,
    "plugin_add": r.PluginChanged,
    "plugin_remove": r.PluginChanged,
    "backup": r.BackupReport,
    "restore": r.RestoreReport,
    "export": r.ExportReport,
    "import": r.IngestReport,
    "reset_index": r.ResetReport,
    "init": r.InitReport,
    "start": r.ServerAddress,
    "stop": r.ServerAddress,
    "upgrade": r.UpgradeReport,
    "completion": r.CompletionScript,
    "auth_create_key": r.ApiKeyIssued,
    "auth_list_keys": r.ApiKeyList,
    "auth_revoke_key": r.ApiKeyRevoked,
    "collection_create": r.CollectionSummary,
    "collection_list": r.CollectionList,
    "collection_rename": r.CollectionSummary,
    "collection_update": r.CollectionSummary,
    "collection_delete": r.CollectionDeleted,
    "collection_add": r.CollectionMembership,
    "collection_remove": r.CollectionMembership,
    "collection_documents": r.DocumentList,
    "collection_counts": r.CollectionCounts,
    "collection_orphans": r.CollectionOrphans,
}
"""Which payload each operation produces.

The envelope carries JSON, so rendering has to know what to parse it back into. A table rather
than a field on the envelope, because the wire format is what an external consumer reads and
a Python class name means nothing to one.

Public, because it is half of a contract with :data:`manicule.cli.render.RENDERERS` — every
type named here must have a renderer there, and every renderer there must be reachable from
some operation here. Neither table can state that on its own, so
``tests/app/test_cli.py`` states it about the pair.
"""


SESSION_PROMPT = (
    "Paste the Cookie header from a browser already signed in to Confluence "
    "(it will not be echoed): "
)


def read_secret(prompt: str) -> str:
    """A secret from the terminal without echoing it, or from a pipe when there is no terminal.

    Public so that the no-echo path can be asserted directly. It is the one piece of this
    command that is not a service call, and a test that drove it through the terminal instead
    would be asserting on what a pseudo-terminal echoes.

    Never an argument. A credential on the command line is a credential in the shell's history
    and in every process listing on the machine, and this one is a live session against a
    corporate system.

    Raises:
        ConfigError: Nothing was given.
    """
    import getpass  # noqa: PLC0415 - only this function needs it

    given = getpass.getpass(prompt) if sys.stdin.isatty() else sys.stdin.readline()
    text = given.strip()
    if not text:
        msg = "nothing was pasted, so there is no session to store"
        raise ConfigError(msg)
    return text


def _from_stdin(given: str | None) -> str:
    """The argument, or stdin when there is none and stdin is not a terminal.

    ``echo "what is the retry policy" | manicule ask`` and
    ``manicule search < query.txt`` both work, and neither needs a flag: an argument that was
    not given and a pipe that is attached is unambiguous.
    """
    if given is not None:
        return given
    if sys.stdin.isatty():
        msg = "no query given, and stdin is a terminal. Pass one as an argument."
        raise ConfigError(msg)
    text = sys.stdin.read().strip()
    if not text:
        msg = "stdin was empty"
        raise ConfigError(msg)
    return text


# --- ask --------------------------------------------------------------------------------------


@app.command()
def ask(
    question: Annotated[
        str | None, typer.Argument(help="The question. Reads stdin if absent.")
    ] = None,
    *,
    profile: Annotated[str | None, typer.Option(help="fast, balanced or precise.")] = None,
    limit: Annotated[int | None, typer.Option(help="Passages to retrieve.")] = None,
    source: Annotated[list[str] | None, typer.Option(help="Restrict to these sources.")] = None,
    collection: Annotated[
        list[str] | None,
        typer.Option(help="Restrict to these collections, by name. Repeat to union them."),
    ] = None,
    conversation: Annotated[str | None, typer.Option(help="Continue this conversation.")] = None,
    repl: Annotated[bool, typer.Option("--repl", help="Ask repeatedly, interactively.")] = False,
) -> None:
    """Answer a question from the corpus, with citations.

    With no question and a terminal attached, this starts the interactive prompt. With no
    question and a pipe attached, it reads the question from stdin.
    """
    if repl or (question is None and sys.stdin.isatty()):
        from manicule.cli.repl import run_repl  # noqa: PLC0415 - only the REPL needs it

        raise typer.Exit(
            run_repl(
                profile=profile,
                limit=limit,
                sources=tuple(source or ()),
                overrides=STATE.overrides,
            )
        )
    # Streamed only when there is a person watching. Under `--json` the envelope is the
    # whole of stdout, and into a pipe the tokens would interleave with whatever the consumer
    # is doing with them.
    stream = not STATE.json_output and sys.stdout.isatty()
    out = render.console()

    def on_event(event: AnswerEvent) -> None:
        if event.kind is EventKind.DELTA and event.text:
            out.file.write(event.text)
            out.file.flush()

    # The renderer is told the text has already been shown, so a streamed answer is not
    # printed a second time underneath itself.
    STATE.text_already_streamed = stream

    # `_from_stdin` is called **inside** the thunk, so a missing question becomes the same
    # failure envelope as everything else rather than a traceback out of the command body.
    emit(
        "ask",
        lambda service: service.ask(
            _from_stdin(question),
            profile=profile,
            limit=limit,
            sources=tuple(source or ()),
            collections=tuple(collection or ()),
            conversation_id=conversation,
            on_event=on_event if stream else None,
        ),
    )


# --- search -----------------------------------------------------------------------------------


@app.command()
def search(
    query: Annotated[
        str | None, typer.Argument(help="What to search for. Reads stdin if absent.")
    ] = None,
    *,
    top: Annotated[int, typer.Option("--top", "-n", help="How many passages to return.")] = 10,
    profile: Annotated[str | None, typer.Option(help="fast, balanced or precise.")] = None,
    source: Annotated[list[str] | None, typer.Option(help="Restrict to these sources.")] = None,
    media_type: Annotated[
        list[str] | None, typer.Option("--type", help="Restrict to these media types.")
    ] = None,
    collection: Annotated[
        list[str] | None,
        typer.Option(help="Restrict to these collections, by name. Repeat to union them."),
    ] = None,
) -> None:
    """Rank passages for a query, without asking a model anything."""
    emit(
        "search",
        lambda service: service.search(
            _from_stdin(query),
            limit=top,
            profile=profile,
            sources=tuple(source or ()),
            media_types=tuple(media_type or ()),
            collections=tuple(collection or ()),
        ),
    )


# --- index ------------------------------------------------------------------------------------


@app.command()
def index(
    path: Annotated[
        Path | None, typer.Argument(help="File or directory. Omit to report status.")
    ] = None,
    *,
    source: Annotated[
        str, typer.Option(help="Source name to record documents under.")
    ] = DEFAULT_SOURCE,
    limit: Annotated[int | None, typer.Option(help="Stop after this many documents.")] = None,
    reindex: Annotated[
        bool, typer.Option("--reindex", help="Re-parse unchanged documents.")
    ] = False,
    watch: Annotated[bool, typer.Option("--watch", help="Keep indexing as files change.")] = False,
    stats: Annotated[
        bool, typer.Option("--stats", help="Report corpus statistics instead.")
    ] = False,
) -> None:
    """Index a path, or — with no path — report what is already in the index."""
    if path is None:
        if watch:
            message = "--watch needs a path to watch"
            raise typer.BadParameter(message)
        if stats:
            emit("stats", lambda service: service.stats())
        else:
            emit("index_status", lambda service: service.index_status())
        return
    if watch:
        from manicule.cli.watch import watch_path  # noqa: PLC0415 - the only user of watchfiles

        raise typer.Exit(
            watch_path(path, source=source, reindex=reindex, overrides=STATE.overrides)
        )
    submit(
        Command(
            "index_path",
            {"path": str(path), "source": source, "limit": limit, "force": reindex},
        )
    )


# --- document ---------------------------------------------------------------------------------


@document_app.command("list")
def document_list(
    limit: Annotated[int, typer.Option(help="Page size.")] = 50,
    offset: Annotated[int, typer.Option(help="How many to skip.")] = 0,
    source: Annotated[str | None, typer.Option(help="Restrict to one source.")] = None,
    media_type: Annotated[str | None, typer.Option("--type", help="Restrict to one type.")] = None,
) -> None:
    """List indexed documents, newest first."""
    emit(
        "document_list",
        lambda service: service.document_list(
            limit=limit, offset=offset, source=source, media_type=media_type
        ),
    )


@document_app.command("get")
def document_get(
    document_id: Annotated[str, typer.Argument(help="The document id.")],
    chunks: Annotated[bool, typer.Option("--chunks", help="Include the stored chunks.")] = False,
) -> None:
    """Read one document, and optionally every chunk it was split into."""
    emit("document_get", lambda service: service.document_get(document_id, chunks=chunks))


@document_app.command("delete")
def document_delete(
    document_id: Annotated[str, typer.Argument(help="The document id.")],
    hard: Annotated[
        bool, typer.Option("--hard", help="Delete outright, not into the trash.")
    ] = False,
) -> None:
    """Remove a document from the index."""
    submit(Command("document_delete", {"document_id": document_id, "hard": hard}))


@document_app.command("reindex")
def document_reindex(
    document_id: Annotated[
        str | None, typer.Argument(help="The document id. Omit it and pass --stale instead.")
    ] = None,
    *,
    stale: Annotated[
        bool,
        typer.Option(
            "--stale",
            help="Re-parse every document an installed parser has moved past, instead of one.",
        ),
    ] = False,
    stale_glossary: Annotated[
        bool,
        typer.Option(
            "--stale-glossary",
            help="Re-detect definitions for every document a changed detector has moved past. "
            "Reads stored chunks: no parser, no connector, no embedder.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="With a sweep: report the plan and write nothing."),
    ] = False,
    batch: Annotated[
        int, typer.Option("--batch", min=1, help="With a sweep: documents per page.")
    ] = DEFAULT_SWEEP_BATCH,
) -> None:
    """Rebuild from what is already stored: one document, or every stale one.

    Touches no network on any path. `--stale` rebuilds every document whose recorded parse
    fingerprint is not one an installed parser would produce now, which is what a parser
    upgrade leaves behind; chunks that do not move keep their ids and their vectors, and a
    document with no retained bytes is named rather than failing the sweep.

    `--stale-glossary` is the rung below it and the cheaper repair. Detection is a stage of its
    own — its grammar, its evidence weights and its normalization change without moving any
    parse, chunk or embedding fingerprint — so a corrected detector reaches an existing corpus
    through this and through nothing else. It reads the chunks already stored, runs no parser,
    fetches nothing and produces no vector, so it costs no GPU time at all.

    Stopping either is safe at a document boundary; running it again resumes.
    """
    sweeping = stale or stale_glossary
    if stale and stale_glossary:
        raise typer.BadParameter(REINDEX_IS_ONE_RUNG)
    if stale and document_id is not None:
        raise typer.BadParameter(REINDEX_IS_ONE_OR_ALL)
    if stale_glossary and document_id is not None:
        raise typer.BadParameter(GLOSSARY_IS_NOT_A_DOCUMENT_SWEEP)
    if not sweeping:
        if document_id is None:
            raise typer.BadParameter(REINDEX_NEEDS_A_TARGET)
        if dry_run:
            raise typer.BadParameter(DRY_RUN_IS_A_SWEEP_OPTION)
        submit(Command("document_reindex", {"document_id": document_id}))
        return
    if stale_glossary:
        submit(Command("document_redetect_glossary", {"batch": batch, "dry_run": dry_run}))
        return
    submit(Command("document_reindex_stale", {"batch": batch, "dry_run": dry_run}))


# --- whole-index re-embedding ---------------------------------------------------------------


@reembed_app.command("plan")
def reembed_plan() -> None:
    """Price the configured migration; makes zero connector, parser, or embedding calls."""
    submit(Command("reembed_plan"))


@reembed_app.command("start")
def reembed_start(
    run_id: Annotated[
        str,
        typer.Argument(
            help="Operator-chosen recovery id; rerun the same command after a lost reply."
        ),
    ],
) -> None:
    """Create an immediately resumable migration under a recovery id you already know."""
    submit(Command("reembed_start", {"run_id": run_id}))


@reembed_app.command("execute")
@reembed_app.command("resume")
def reembed_resume(
    run_id: Annotated[str, typer.Argument(help="Run id chosen for `reembed start`.")],
) -> None:
    """Execute or resume a run after a crash or transient refusal."""
    submit(Command("reembed_resume", {"run_id": run_id}))


@reembed_app.command("inspect")
@reembed_app.command("status")
def reembed_status(
    run_id: Annotated[str, typer.Argument(help="Run id chosen for `reembed start`.")],
) -> None:
    """Show aggregate durable progress without source identifiers or paths."""
    emit("reembed_status", lambda service: service.reembed_status(run_id))


@reembed_app.command("abandon")
def reembed_abandon(
    run_id: Annotated[str, typer.Argument(help="Unfinished run to make cleanup-eligible.")],
) -> None:
    """Stop an unfinished run without touching the live generation."""
    submit(Command("reembed_abandon", {"run_id": run_id}))


@reembed_app.command("cleanup")
def reembed_cleanup(
    run_id: Annotated[str, typer.Argument(help="Failed or superseded run to remove.")],
) -> None:
    """Delete terminal non-live shadow storage."""
    submit(Command("reembed_cleanup", {"run_id": run_id}))


# --- collection -------------------------------------------------------------------------------


@collection_app.command("create")
def collection_create(
    name: Annotated[str, typer.Argument(help="The collection's name.")],
    description: Annotated[str | None, typer.Option(help="What this collection is for.")] = None,
) -> None:
    """Create a collection. A name already in use is refused rather than merged."""
    submit(Command("collection_create", {"name": name, "description": description}))


@collection_app.command("list")
def collection_list() -> None:
    """List every collection in this workspace."""
    emit("collection_list", lambda service: service.collection_list())


@collection_app.command("rename")
def collection_rename(
    collection_id: Annotated[str, typer.Argument(help="The collection id.")],
    name: Annotated[str, typer.Argument(help="The new name.")],
) -> None:
    """Rename a collection. Nothing is re-indexed and no membership moves."""
    submit(Command("collection_rename", {"collection_id": collection_id, "name": name}))


@collection_app.command("update")
def collection_update(
    collection_id: Annotated[str, typer.Argument(help="The collection id.")],
    description: Annotated[
        str, typer.Argument(help="What this collection is for. Pass '' to clear it.")
    ],
) -> None:
    """Set a collection's description, leaving its membership alone.

    An argument rather than an option, because the write is a set and not a merge: as an
    optional flag, `collection update <id>` with nothing else erased the description it exists
    to change.
    """
    submit(
        Command("collection_update", {"collection_id": collection_id, "description": description})
    )


@collection_app.command("delete")
def collection_delete(
    collection_id: Annotated[str, typer.Argument(help="The collection id.")],
) -> None:
    """Delete a collection. The documents in it are untouched."""
    submit(Command("collection_delete", {"collection_id": collection_id}))


@collection_app.command("add")
def collection_add(
    collection_id: Annotated[str, typer.Argument(help="The collection id.")],
    document_id: Annotated[list[str], typer.Argument(help="Documents to add.")],
) -> None:
    """Add documents to a collection. Nothing is re-embedded."""
    # Cast rather than annotated: `list` is invariant, so a `list[str]` is not a
    # `list[JsonValue]` however it is spelled, and every list of strings is a list of JSON
    # values. `Arguments.texts` checks the shape again on the way out, which is where a caller
    # that sent something else is refused.
    documents = cast("list[JsonValue]", list(document_id))
    submit(Command("collection_add", {"collection_id": collection_id, "document_ids": documents}))


@collection_app.command("remove")
def collection_remove(
    collection_id: Annotated[str, typer.Argument(help="The collection id.")],
    document_id: Annotated[list[str], typer.Argument(help="Documents to remove.")],
) -> None:
    """Remove documents from a collection. The documents themselves survive."""
    documents = cast("list[JsonValue]", list(document_id))
    submit(
        Command("collection_remove", {"collection_id": collection_id, "document_ids": documents})
    )


@collection_app.command("documents")
def collection_documents(
    collection_id: Annotated[str, typer.Argument(help="The collection id.")],
    limit: Annotated[int, typer.Option(help="Page size.")] = 50,
    offset: Annotated[int, typer.Option(help="How many to skip.")] = 0,
) -> None:
    """List a collection's documents, manual members and rule-selected alike."""
    emit(
        "collection_documents",
        lambda service: service.collection_documents(collection_id, limit=limit, offset=offset),
    )


@collection_app.command("counts")
def collection_counts(
    collection_id: Annotated[str, typer.Argument(help="The collection id.")],
) -> None:
    """Count a collection's documents and chunks, now rather than from a stored total."""
    emit("collection_counts", lambda service: service.collection_counts(collection_id))


@collection_app.command("orphans")
def collection_orphans(
    confirm: Annotated[
        bool,
        typer.Option("--confirm", help="Move the orphans to the trash instead of listing them."),
    ] = False,
) -> None:
    """List live documents in no collection, and optionally move them to the trash.

    Reporting is the default. In a corpus where collections are optional "in no collection"
    describes most of it, so deletion has to be asked for by name -- and even then it is the
    same soft delete `document delete` performs, so everything it removes can be restored.

    Command line only. It destroys data, and this project keeps that class of operation off
    the surfaces an unattended caller reaches.
    """
    submit(Command("collection_orphans", {"delete": confirm}))


# --- connector --------------------------------------------------------------------------------


@connector_app.command("list")
def connector_list() -> None:
    """List configured sources and what their last sync recorded."""
    emit("connector_list", lambda service: service.connector_list())


@connector_app.command("sync")
def connector_sync(
    name: Annotated[str, typer.Argument(help="The configured source's name.")],
    limit: Annotated[int | None, typer.Option(help="Stop after this many documents.")] = None,
    acquire_only: Annotated[
        bool,
        typer.Option(
            "--acquire-only",
            help="Promote retained source bytes and stop before local derivation.",
        ),
    ] = False,
) -> None:
    """Run one configured connector."""
    submit(
        Command(
            "connector_sync",
            {"name": name, "limit": limit, "acquire_only": acquire_only},
        )
    )


@connector_app.command("snapshot")
def connector_snapshot(
    name: Annotated[str, typer.Argument(help="The configured source's name.")],
) -> None:
    """Show aggregate status for the source's active durable snapshot."""
    emit("snapshot_status", lambda service: service.snapshot_status(name))


@connector_app.command("verify")
def connector_verify(
    snapshot_id: Annotated[str, typer.Argument(help="Opaque snapshot id from snapshot status.")],
) -> None:
    """Verify a durable snapshot manifest without reading source content."""
    emit("snapshot_verify", lambda service: service.snapshot_verify(snapshot_id))


@connector_app.command("sidecar")
def connector_sidecar(
    root: Annotated[
        Path | None,
        typer.Argument(
            help="Directory of enriched HTML pages. With --source, a subdirectory of that "
            "source's root; omit it to convert the whole root."
        ),
    ] = None,
    source: Annotated[
        str,
        typer.Option(
            "--source",
            help="A configured filesystem source. Takes its root and its enriched profiles "
            "from the connector `connector sync` would run.",
        ),
    ] = "",
    force: Annotated[
        bool, typer.Option("--force", help="Replace manifests that already exist.")
    ] = False,
) -> None:
    """Record what enriched HTML pages say about themselves, for filesystem ingestion.

    Writes `<page>.source.json` beside each page. The pages themselves are never modified.

    Pass `--source` for a corpus whose exporter spells its markers differently. Without it the
    run uses the built-in default profile, which reports `no_profile` for every page of a
    corpus the configured sync adapts perfectly well.

    Command line only. It is the one operation that writes into the corpus *directory* rather
    than into the index, so it stays on the surface where a person is present.
    """
    submit(
        Command(
            "connector_sidecar",
            {"root": None if root is None else str(root), "source": source, "force": force},
        )
    )


@connector_app.command("login")
def connector_login(
    name: Annotated[str, typer.Argument(help="The configured source's name.")],
    *,
    browser: Annotated[
        bool,
        typer.Option("--browser", help="Open a browser and sign in there instead of pasting."),
    ] = False,
    browser_state: Annotated[
        Path | None,
        typer.Option(
            "--browser-state",
            help="Import cookies from a Playwright storage_state JSON file.",
        ),
    ] = None,
    timeout: Annotated[
        float | None,
        typer.Option(
            "--timeout",
            help="Seconds to wait for browser sign-in. Defaults to browser_timeout_seconds.",
        ),
    ] = None,
    allow_insecure_state: Annotated[
        bool,
        typer.Option(
            "--allow-insecure-state",
            help="Import a --browser-state file that other users on this machine can read.",
        ),
    ] = False,
    forget: Annotated[
        bool, typer.Option("--forget", help="Remove the stored session instead of taking one.")
    ] = False,
) -> None:
    """Capture the browser session a Confluence source behind single sign-on signs in with.

    `--browser` opens a browser at the source's URL and waits while you sign in — including any
    second factor or conditional-access step. Nothing is typed for you and no page content is
    read; manicule watches only the cookie jar, and stores nothing until the instance confirms
    who you are. It needs `manicule[browser-auth]`.

    Without it, sign in to Confluence in your own browser and paste the Cookie header from its
    developer tools when this asks. That path needs nothing installed and keeps the stronger
    property: with no browser to drive, manicule cannot see the page you type into.

    manicule never asks for your password, cannot use one, and has nowhere to put one.

    **A server has to be running.** The session is handed to it and held in its memory, and
    nothing is written to disk on any path — so a capture with nowhere to go is refused before
    you are asked to sign in, rather than after.

    Command line only. It opens a window on this machine, which does not belong on a surface an
    unattended caller can reach.
    """
    # An option that reaches nothing is refused rather than accepted, on the rule
    # `INSECURE_TARGET_IS_A_BACKUP_OPTION` states. Here rather than in the service because these
    # are facts about the *invocation* — which flags were typed together — and Typer is what
    # turns that into the error a person reads. Mutual exclusion between the three ways in stays
    # in the service, because that one is a fact about the operation and every surface needs it.
    if allow_insecure_state and browser_state is None:
        raise typer.BadParameter(INSECURE_STATE_IS_AN_IMPORT_OPTION)
    if timeout is not None and not browser:
        raise typer.BadParameter(BROWSER_TIMEOUT_IS_A_BROWSER_OPTION)
    if timeout is not None and timeout <= 0:
        raise typer.BadParameter(BROWSER_TIMEOUT_MUST_BE_POSITIVE)

    # Checked before anything is captured, and before anybody is asked to sign in. A person who
    # opens a browser, completes a second factor and *then* hears there is nowhere to put the
    # result has been made to do the work twice.
    served = proxy.listening(STATE.overrides)
    if served is None:
        print_envelope(
            proxy.refuse(
                Command("connector_login", {"name": name}),
                workspace=STATE.workspace or UNKNOWN_WORKSPACE,
                overrides=STATE.overrides,
            )
        )
        return

    # Prompting is skipped for every path that is not the paste, so `--browser` does not stop to
    # ask for the thing it is there to avoid — and `--forget` does not ask for a secret it is
    # about to delete.
    manual = not (browser or browser_state is not None or forget)
    cookies = read_secret(SESSION_PROMPT) if manual else ""
    # Run here rather than sent to the server, and the store is what crosses instead. The
    # browser has to open where the person is; the credential has to end up where the syncs
    # run. `HandoverStore` is that seam, so the capture flow itself is unchanged.
    emit(
        "connector_login",
        lambda service: service.connector_login(
            name,
            cookies=cookies,
            forget=forget,
            browser=browser,
            browser_state=browser_state,
            timeout_seconds=timeout,
            allow_insecure_state=allow_insecure_state,
            store=proxy.HandoverStore(served),
        ),
    )


# --- workspace --------------------------------------------------------------------------------


@workspace_app.command("list")
def workspace_list() -> None:
    """List workspaces, and say which one is active."""
    emit("workspace_list", lambda service: service.workspace_list())


@workspace_app.command("switch")
def workspace_switch(
    name: Annotated[str, typer.Argument(help="The workspace to make active.")],
    create: Annotated[
        bool, typer.Option("--create", help="Accept a name that does not exist.")
    ] = False,
) -> None:
    """Record a different active workspace. It takes effect at the next start."""
    submit(Command("workspace_switch", {"name": name, "create": create}))


# --- auth -------------------------------------------------------------------------------------


@auth_app.command("create-key")
def auth_create_key(
    name: Annotated[str, typer.Argument(help="A label for the key.")],
    role: Annotated[str, typer.Option(help="admin, member or viewer.")] = "member",
    expires_days: Annotated[int | None, typer.Option(help="Days until it expires.")] = None,
) -> None:
    """Mint an API key for this workspace. The secret is shown once and never stored."""
    submit(Command("auth_create_key", {"name": name, "role": role, "expires_days": expires_days}))


@auth_app.command("list-keys")
def auth_list_keys() -> None:
    """List this workspace's API keys. Never their secrets — only digests are kept."""
    emit("auth_list_keys", lambda service: service.api_key_list())


@auth_app.command("revoke-key")
def auth_revoke_key(
    name_or_id: Annotated[str, typer.Argument(help="The key's name or id.")],
) -> None:
    """Revoke an API key. Immediate, and irreversible."""
    submit(Command("auth_revoke_key", {"name_or_id": name_or_id}))


# --- plugin -----------------------------------------------------------------------------------


@plugin_app.command("list")
def plugin_list(
    registry: Annotated[
        bool, typer.Option("--registry", help="Also list the community registry.")
    ] = False,
) -> None:
    """List installed plugins and the components each registers."""
    emit("plugin_list", lambda service: service.plugin_list(registry=registry))


@plugin_app.command("add")
def plugin_add(name: Annotated[str, typer.Argument(help="The plugin's name.")]) -> None:
    """Enable an installed plugin. This never installs one — see the output if it is missing."""
    submit(Command("plugin_add", {"name": name}))


@plugin_app.command("remove")
def plugin_remove(name: Annotated[str, typer.Argument(help="The plugin's name.")]) -> None:
    """Disable a plugin. The distribution stays installed and is not touched."""
    submit(Command("plugin_remove", {"name": name}))


# --- config -----------------------------------------------------------------------------------


@config_app.command("get")
def config_get(
    key: Annotated[str, typer.Argument(help="A dotted key. Omit for everything.")] = "",
) -> None:
    """Read configuration, with every credential masked."""
    emit("config_get", lambda service: service.config_get(key))


@config_app.command("show")
def config_show() -> None:
    """Print the whole configuration, with every credential masked."""
    emit("config_get", lambda service: service.config_get(""))


@config_app.command("set")
def config_set(
    key: Annotated[str, typer.Argument(help="A dotted key, e.g. rag.profile.")],
    value: Annotated[str, typer.Argument(help="JSON when it parses; a string when it does not.")],
) -> None:
    """Write one setting, validating the whole configuration before saving it."""
    submit(Command("config_set", {"key": key, "value": value}))


# --- operations -------------------------------------------------------------------------------


@app.command()
def backup(
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Where to write it.")
    ] = None,
    restore: Annotated[
        Path | None, typer.Option("--restore", help="Restore from here instead.")
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing target.")] = False,
    allow_insecure_target: Annotated[
        bool,
        typer.Option(
            "--allow-insecure-target",
            help="Write into a group- or world-readable directory. It holds the whole corpus.",
        ),
    ] = False,
) -> None:
    """Take a consistent copy of the data directory, or put one back."""
    if output is not None and restore is not None:
        raise typer.BadParameter(BACKUP_IS_NOT_A_RESTORE)
    if restore is not None:
        if allow_insecure_target:
            raise typer.BadParameter(INSECURE_TARGET_IS_A_BACKUP_OPTION)
        submit(Command("restore", {"source": str(restore), "force": force}))
        return
    if output is None:
        raise typer.BadParameter(BACKUP_NEEDS_A_TARGET)
    emit(
        "backup",
        lambda service: service.backup(output, allow_insecure_target=allow_insecure_target),
    )


@app.command("export")
def export_corpus(
    output: Annotated[Path, typer.Option("--output", "-o", help="Directory to write.")],
    allow_insecure_target: Annotated[
        bool,
        typer.Option(
            "--allow-insecure-target",
            help="Write into a group- or world-readable directory. It holds the whole corpus.",
        ),
    ] = False,
) -> None:
    """Write a portable archive: retained source bytes and metadata, never chunks or vectors."""
    emit(
        "export",
        lambda service: service.export_corpus(output, allow_insecure_target=allow_insecure_target),
    )


@app.command("import")
def import_corpus(
    path: Annotated[Path, typer.Argument(help="An archive written by `manicule export`.")],
    force: Annotated[bool, typer.Option("--force", help="Re-parse unchanged documents.")] = False,
) -> None:
    """Ingest an exported archive, re-deriving chunks and vectors on this machine."""
    submit(Command("import", {"path": str(path), "force": force}))


@app.command("reset-index")
def reset_index(
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            help="Confirm the derived reset. Source snapshots and history are retained.",
        ),
    ] = False,
) -> None:
    """Reset derived search state while retaining durable source snapshots."""
    if not yes:
        raise typer.BadParameter(RESET_NEEDS_CONFIRMATION)
    submit(Command("reset_index"))


@app.command("reset-derived")
def reset_derived(
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report aggregate impact without changing state.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Confirm the derived-only reset.")] = False,
) -> None:
    """Reset chunks, FTS, glossary and vectors; retain snapshots and source history."""
    if not dry_run and not yes:
        raise typer.BadParameter("reset-derived requires --yes, or use --dry-run")
    if dry_run:
        emit(
            "lifecycle_reset_derived",
            lambda service: service.lifecycle_reset_derived(dry_run=True),
        )
        return
    submit(Command("lifecycle_reset_derived", {"dry_run": dry_run}))


@app.command("cleanup-derived-generations")
def cleanup_derived_generations(
    yes: Annotated[
        bool, typer.Option("--yes", help="Remove eligible terminal generations.")
    ] = False,
) -> None:
    """Plan by default; remove only obsolete terminal derived generations with --yes."""
    if not yes:
        emit(
            "lifecycle_cleanup_generations",
            lambda service: service.lifecycle_cleanup_generations(dry_run=True),
        )
        return
    submit(Command("lifecycle_cleanup_generations", {"dry_run": not yes}))


@app.command("release-source-history")
def release_source_history(
    before: Annotated[
        str, typer.Argument(help="Release historical versions older than this ISO-8601 time.")
    ],
    yes: Annotated[
        bool, typer.Option("--yes", help="Apply the retention release; otherwise dry-run.")
    ] = False,
) -> None:
    """Release eligible prior source versions without touching current snapshots."""
    if not yes:
        from datetime import datetime  # noqa: PLC0415 - only this command parses a timestamp

        try:
            cutoff = datetime.fromisoformat(before.replace("Z", "+00:00"))
        except ValueError as exc:
            raise typer.BadParameter("before must be an ISO-8601 timestamp") from exc
        emit(
            "lifecycle_release_history",
            lambda service: service.lifecycle_release_history(cutoff, dry_run=True),
        )
        return
    submit(
        Command(
            "lifecycle_release_history",
            {"cutoff": before, "dry_run": not yes},
        )
    )


@app.command("snapshot-delete")
def snapshot_delete(
    run_id: Annotated[str, typer.Argument(help="The promoted snapshot run identity.")],
    confirm: Annotated[
        str | None,
        typer.Option(
            "--confirm",
            help="Exact token from the latest dry run. Omit it to perform the dry run.",
        ),
    ] = None,
) -> None:
    """Plan deletion by default; delete authoritative snapshot ownership only with its token."""
    if confirm is None:
        emit(
            "lifecycle_delete_snapshot",
            lambda service: service.lifecycle_delete_snapshot(run_id, dry_run=True),
        )
        return
    submit(
        Command(
            "lifecycle_delete_snapshot",
            {"run_id": run_id, "confirmation": confirm, "dry_run": False},
        )
    )


@app.command()
def doctor(
    fix: Annotated[
        bool,
        typer.Option(
            "--fix",
            help="Repair what can be repaired, then report. Today that is the two artifacts "
            "no wheel ships: the declared code grammars, and the BPE vocabularies every "
            "search measures a context with. Each is seeded from an offline bundle if one is "
            "installed and from its upstream otherwise. It is the only part of this command "
            "that writes to the machine or uses the network, which is why it is a flag.",
        ),
    ] = False,
) -> None:
    """Check configuration, plugins, storage, the index, grammars, vocabularies and the bind."""
    emit("doctor", lambda service: service.doctor(fix=fix))


@app.command()
def init(
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite an existing config file.")
    ] = False,
) -> None:
    """Write a starting configuration, choosing what this machine can actually run."""
    submit(Command("init", {"force": force}))


@app.command()
def upgrade(
    version: Annotated[
        str | None, typer.Option("--version", help="Upgrade to this version.")
    ] = None,
    skip_backup: Annotated[
        bool, typer.Option("--skip-backup", help="Do not back up first.")
    ] = False,
) -> None:
    """Back up, then report exactly how to upgrade. It does not run a package manager."""
    submit(Command("upgrade", {"version": version, "skip_backup": skip_backup}))


@app.command()
def completion(
    shell: Annotated[str, typer.Option("--shell", help="bash, zsh, fish or powershell.")] = "bash",
) -> None:
    """Print a shell completion script. Redirect it into the file your shell sources."""
    from manicule.cli.shell import completion_script  # noqa: PLC0415 - only this command needs it

    try:
        script = completion_script(shell)
    except ManiculeError as exc:
        render.render_error(render.console(stderr=True), "completion", error_info(exc))
        raise typer.Exit(1) from exc
    payload = r.CompletionScript(shell=shell, script=script)
    print_envelope(
        Envelope(
            op="completion",
            ok=True,
            workspace=STATE.workspace or UNKNOWN_WORKSPACE,
            data=payload.model_dump(mode="json"),
        )
    )


# --- serving ----------------------------------------------------------------------------------


@app.command("serve")
@app.command("start")
def start(
    *,
    mcp_only: Annotated[
        bool, typer.Option("--mcp-only", help="Serve MCP alone, without the API or the UI.")
    ] = False,
    transport: Annotated[str, typer.Option(help="stdio or http.")] = "stdio",
    host: Annotated[str | None, typer.Option(help="Bind address. Loopback unless widened.")] = None,
    port: Annotated[int | None, typer.Option("--port", "-p", help="Bind port.")] = None,
    allow_public_bind: Annotated[
        bool,
        typer.Option(
            "--allow-public-bind",
            help="Say explicitly that a non-loopback bind is intended. Needs auth as well.",
        ),
    ] = False,
    no_web: Annotated[bool, typer.Option("--no-web", help="Do not serve the web UI.")] = False,
) -> None:
    """Serve manicule: the HTTP API, the browser surface and MCP together, or MCP alone.

    ``stdio`` is the default and opens no socket at all — and it is always MCP, because the
    HTTP API has no stdio form. ``--transport http`` serves the **HTTP API, the browser surface
    and MCP** on one port, with MCP at ``/mcp/``; add ``--mcp-only`` to serve MCP alone there.

    **MCP over a socket carries the read-only tools only**, whichever of those two modes you
    are in. Indexing, deleting, syncing, writing configuration and enabling a plugin are not
    registered on it — over stdio they are unreachable from a network by construction, and a
    socket has to replace that property rather than assume it. They stay on this command line,
    on stdio, and on the control socket a served manicule answers write commands on.

    Either way the address goes through one bind policy: loopback unless a non-loopback host
    is configured **and** ``--allow-public-bind`` is passed **and** authentication is on. Any
    one missing is a refusal naming which.

    ``--no-web`` leaves the browser surface unmounted, so every ``/ui`` path answers 404 and
    the process serves the JSON API and MCP. It applies to ``--transport http`` without
    ``--mcp-only``, which is the only mode that has a browser surface to suppress.
    """
    from manicule.cli.serving import serve_forever  # noqa: PLC0415 - only this command serves

    raise typer.Exit(
        serve_forever(
            transport=transport,
            host=host,
            port=port,
            allow_public=allow_public_bind,
            overrides=STATE.overrides,
            json_output=STATE.json_output,
            mcp_only=mcp_only,
            web=not no_web,
        )
    )


@app.command()
def stop() -> None:
    """Ask a running manicule server to stop, and wait for it."""
    from manicule.cli.serving import stop_running  # noqa: PLC0415 - beside its sibling

    print_envelope(stop_running(STATE.overrides, workspace=STATE.workspace or UNKNOWN_WORKSPACE))


def main() -> None:
    """The console-script entry point."""
    app()


__all__ = [
    "BACKUP_IS_NOT_A_RESTORE",
    "BACKUP_NEEDS_A_TARGET",
    "PAYLOADS",
    "RESET_NEEDS_CONFIRMATION",
    "SHARED_OPTIONS",
    "STATE",
    "WORKSPACE_NAMED_TWICE",
    "CommandsShareTheRootOptions",
    "State",
    "app",
    "emit",
    "main",
    "print_envelope",
    "submit",
]
