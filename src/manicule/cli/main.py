"""The command line: nineteen commands, each a few lines over the application service.

Typer derives every option and argument from the type hints, so a command's signature and the
interface a person types cannot drift apart. What remains here is parsing, one service call,
and rendering — no command decides anything, and none of them reaches past
:class:`~manicule.app.service.ApplicationService`.

Three things hold everywhere:

**``--json`` on everything that emits data.** The same envelope the MCP tools return, printed
to stdout with nothing else in it. Human output and failures go to stderr, so ``| jq`` on a
failed run reads an empty stream rather than a prose error.

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
from typing import TYPE_CHECKING, Annotated, Any

import typer

from manicule.app import results as r
from manicule.app.dispatch import error_info, run_op
from manicule.app.results import Envelope, failed
from manicule.app.runtime import Runtime
from manicule.app.service import DEFAULT_SOURCE, ApplicationService
from manicule.cli import render
from manicule.core.errors import ConfigError, ManiculeError
from manicule.core.version import CORE_VERSION
from manicule.generation.answers import EventKind

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from manicule.app.results import Payload
    from manicule.generation.answers import AnswerEvent

RESET_NEEDS_CONFIRMATION = (
    "this deletes every document, chunk and vector in this workspace and cannot be undone. "
    "Pass --yes to confirm."
)
"""Why ``reset-index`` refuses without ``--yes``.

A constant rather than a literal at the raise site, so a test can assert that the refusal
names the flag without reading it back out of a rendered terminal box — which wraps, colours
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

app = typer.Typer(
    name="manicule",
    help="Self-hosted document search and answers, with citations that resolve.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
document_app = typer.Typer(help="Inspect and manage indexed documents.", no_args_is_help=True)
connector_app = typer.Typer(help="Configured sources.", no_args_is_help=True)
workspace_app = typer.Typer(help="Workspaces, and which one is active.", no_args_is_help=True)
plugin_app = typer.Typer(help="Installed plugins.", no_args_is_help=True)
config_app = typer.Typer(help="Read and write configuration.", no_args_is_help=True)
auth_app = typer.Typer(help="API keys for this workspace.", no_args_is_help=True)
app.add_typer(document_app, name="document")
app.add_typer(connector_app, name="connector")
app.add_typer(workspace_app, name="workspace")
app.add_typer(plugin_app, name="plugin")
app.add_typer(config_app, name="config")
app.add_typer(auth_app, name="auth")


JsonOption = Annotated[
    bool,
    typer.Option("--json", help="Emit the result envelope as JSON on stdout, and nothing else."),
]
WorkspaceOption = Annotated[
    str | None,
    typer.Option("--workspace", "-w", help="Run in this workspace instead of the configured one."),
]


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
    """Open a runtime, run one operation, close it. Never leaks an exception a caller owns."""
    try:
        runtime = Runtime.open(**STATE.overrides)
    except (ManiculeError, ValueError, OSError) as exc:
        # Configuration that will not load is the one failure that happens before there is a
        # workspace to report, and it is also the commonest. It gets the same envelope as
        # everything else rather than a traceback.
        return failed(op, STATE.workspace or UNKNOWN_WORKSPACE, error_info(exc))
    async with runtime:
        service = ApplicationService(runtime)
        return await run_op(op, service.workspace, lambda: call(service))


def emit(op: str, call: Callable[[ApplicationService], Awaitable[Payload]]) -> None:
    """Run an operation, print it the way the caller asked for, and set the exit status."""
    print_envelope(asyncio.run(_execute(op, call)))


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
    if envelope.ok and envelope.data is not None:
        payload = _PAYLOADS[envelope.op].model_validate(envelope.data)
        console = render.console()
        if isinstance(payload, r.AnswerResultPayload):
            console.print()
            render.render_answer(console, payload, text_already_shown=STATE.text_already_streamed)
            return
        render.render(console, payload)
        return
    if envelope.error is not None:
        render.render_error(render.console(stderr=True), envelope.op, envelope.error)
    raise typer.Exit(1)


_PAYLOADS: dict[str, type[Payload]] = {
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
    "doctor": r.Diagnosis,
    "connector_list": r.ConnectorList,
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
}
"""Which payload each operation produces.

The envelope carries JSON, so rendering has to know what to parse it back into. A table rather
than a field on the envelope, because the wire format is what an external consumer reads and
a Python class name means nothing to one.
"""


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
    top: Annotated[int, typer.Option("--top", "-n", help="How many passages to return.")] = 10,
    profile: Annotated[str | None, typer.Option(help="fast, balanced or precise.")] = None,
    source: Annotated[list[str] | None, typer.Option(help="Restrict to these sources.")] = None,
    media_type: Annotated[
        list[str] | None, typer.Option("--type", help="Restrict to these media types.")
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
    emit(
        "index_path",
        lambda service: service.index_path(path, source=source, limit=limit, force=reindex),
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
    emit("document_delete", lambda service: service.document_delete(document_id, hard=hard))


@document_app.command("reindex")
def document_reindex(
    document_id: Annotated[str, typer.Argument(help="The document id.")],
) -> None:
    """Re-parse one document from the bytes ingest retained. Touches no network."""
    emit("document_reindex", lambda service: service.document_reindex(document_id))


# --- connector --------------------------------------------------------------------------------


@connector_app.command("list")
def connector_list() -> None:
    """List configured sources and what their last sync recorded."""
    emit("connector_list", lambda service: service.connector_list())


@connector_app.command("sync")
def connector_sync(
    name: Annotated[str, typer.Argument(help="The configured source's name.")],
    limit: Annotated[int | None, typer.Option(help="Stop after this many documents.")] = None,
) -> None:
    """Run one configured connector."""
    emit("connector_sync", lambda service: service.connector_sync(name, limit=limit))


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
    emit("workspace_switch", lambda service: service.workspace_switch(name, create=create))


# --- auth -------------------------------------------------------------------------------------


@auth_app.command("create-key")
def auth_create_key(
    name: Annotated[str, typer.Argument(help="A label for the key.")],
    role: Annotated[str, typer.Option(help="admin, member or viewer.")] = "member",
    expires_days: Annotated[int | None, typer.Option(help="Days until it expires.")] = None,
) -> None:
    """Mint an API key for this workspace. The secret is shown once and never stored."""
    emit(
        "auth_create_key",
        lambda service: service.api_key_create(name, role=role, expires_days=expires_days),
    )


@auth_app.command("list-keys")
def auth_list_keys() -> None:
    """List this workspace's API keys. Never their secrets — only digests are kept."""
    emit("auth_list_keys", lambda service: service.api_key_list())


@auth_app.command("revoke-key")
def auth_revoke_key(
    name_or_id: Annotated[str, typer.Argument(help="The key's name or id.")],
) -> None:
    """Revoke an API key. Immediate, and irreversible."""
    emit("auth_revoke_key", lambda service: service.api_key_revoke(name_or_id))


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
    emit("plugin_add", lambda service: service.plugin_add(name))


@plugin_app.command("remove")
def plugin_remove(name: Annotated[str, typer.Argument(help="The plugin's name.")]) -> None:
    """Disable a plugin. The distribution stays installed and is not touched."""
    emit("plugin_remove", lambda service: service.plugin_remove(name))


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
    emit("config_set", lambda service: service.config_set(key, value))


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
        emit("restore", lambda service: service.restore(restore, force=force))
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
) -> None:
    """Write a portable archive: retained source bytes and metadata, never chunks or vectors."""
    emit("export", lambda service: service.export_corpus(output))


@app.command("import")
def import_corpus(
    path: Annotated[Path, typer.Argument(help="An archive written by `manicule export`.")],
    force: Annotated[bool, typer.Option("--force", help="Re-parse unchanged documents.")] = False,
) -> None:
    """Ingest an exported archive, re-deriving chunks and vectors on this machine."""
    emit("import", lambda service: service.import_corpus(path, force=force))


@app.command("reset-index")
def reset_index(
    yes: Annotated[
        bool, typer.Option("--yes", help="Confirm. Required; this cannot be undone.")
    ] = False,
) -> None:
    """Delete every document, chunk and vector in this workspace."""
    if not yes:
        raise typer.BadParameter(RESET_NEEDS_CONFIRMATION)
    emit("reset_index", lambda service: service.reset_index())


@app.command()
def doctor(
    fix: Annotated[
        bool,
        typer.Option(
            "--fix",
            help="Repair what can be repaired, then report. Today that is one thing: seeding "
            "the declared code grammars, from an offline bundle if one is installed and from "
            "the grammar release otherwise. It is the only part of this command that writes "
            "to the machine or uses the network, which is why it is a flag.",
        ),
    ] = False,
) -> None:
    """Check configuration, plugins, storage, the index, grammars and the network bind."""
    emit("doctor", lambda service: service.doctor(fix=fix))


@app.command()
def init(
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite an existing config file.")
    ] = False,
) -> None:
    """Write a starting configuration, choosing what this machine can actually run."""
    emit("init", lambda service: service.initialise(force=force))


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
    emit("upgrade", lambda service: service.upgrade(version=version, skip_backup=skip_backup))


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


@app.command()
def start(
    *,
    mcp_only: Annotated[
        bool, typer.Option("--mcp-only", help="Serve MCP and nothing else.")
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
    """Serve manicule: the HTTP API over a socket, or MCP.

    ``stdio`` is the default and opens no socket at all — and it is always MCP, because the
    HTTP API has no stdio form. ``--transport http`` serves the **HTTP API**; add
    ``--mcp-only`` to serve the MCP protocol over that socket instead.

    Either way the address goes through one bind policy: loopback unless a non-loopback host
    is configured **and** ``--allow-public-bind`` is passed **and** authentication is on. Any
    one missing is a refusal naming which.

    ``--no-web`` leaves the browser surface unmounted, so every ``/ui`` path answers 404 and
    the process serves only the JSON API. It applies to ``--transport http`` without
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
    "RESET_NEEDS_CONFIRMATION",
    "STATE",
    "State",
    "app",
    "emit",
    "main",
    "print_envelope",
]
