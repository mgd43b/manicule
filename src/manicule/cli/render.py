"""Turning a payload into something worth reading in a terminal.

Rendering only. Every function here takes a payload the service already produced and writes it
to a console; none of them decides anything, computes anything or reaches for a store. That is
what keeps the command line an adapter — and it is what makes ``--json`` and the human output
two views of one result rather than two results.

The renderers are looked up by payload type, so adding an operation means adding a payload and
a renderer, and forgetting the renderer is a loud failure rather than a blank screen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from manicule.app import results as r

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from manicule.app.results import Payload

_STATE_STYLE: Mapping[str, str] = {
    "ok": "green",
    "degraded": "yellow",
    "failing": "red",
    "unknown": "dim",
}

_STATUS_STYLE: Mapping[str, str] = {
    "indexed": "green",
    "failed": "red",
    "pending": "yellow",
    "parsing": "yellow",
    "chunking": "yellow",
    "embedding": "yellow",
    "no_extractable_text": "yellow",
    "unsupported": "dim",
}


def console(*, stderr: bool = False) -> Console:
    """A console. ``stderr`` for anything that must not land in a pipe's payload.

    **Automatic highlighting is off, and that is a correctness setting rather than a taste
    one.** Rich's highlighter styles what it takes for numbers, paths, URLs and options by
    inserting escape sequences *inside* the token — so with colour enabled a version prints as
    ``\x1b[1;36m0.1\x1b[0m.\x1b[1;36m0\x1b[0m`` and a document id, a fingerprint or an
    anchor comes out of a pipe in pieces. This output is mostly identifiers, and an identifier
    nobody can copy is not one. Everything that *is* styled here is styled deliberately, with
    markup, around the value rather than through it.
    """
    return Console(stderr=stderr, soft_wrap=False, highlight=False)


def render_error(out: Console, op: str, error: r.ErrorInfo) -> None:
    """Print a failure. Always to stderr, so ``| jq`` on a failed run reads an empty stream."""
    out.print(f"[red]{escape(op)} failed:[/red] [bold]{escape(error.type)}[/bold]")
    out.print(escape(error.message))
    if error.hint:
        out.print(f"[dim]{escape(error.hint)}[/dim]")


# --- answers and search ---------------------------------------------------------------------


def render_answer(
    out: Console, payload: r.AnswerResultPayload, *, text_already_shown: bool = False
) -> None:
    """The answer, then its citations, then what was true of the run.

    Citations are listed in full rather than summarised. A citation is the product here — an
    answer whose sources are collapsed into "3 sources" is an answer nobody can check.

    ``text_already_shown`` is for the caller that streamed the tokens as they arrived. The
    answer is the same either way — the payload is identical whether or not anything was
    streamed — so printing it a second time would show a reader the same paragraph twice and
    leave them wondering which one to trust.
    """
    if payload.text and not text_already_shown:
        out.print(Panel(Text(payload.text), title="answer", border_style="cyan"))
    if payload.error:
        out.print(f"[red]{escape(payload.error)}[/red]")
    if payload.citations:
        table = Table("slot", "document", "location", "verified", box=None, pad_edge=False)
        for citation in payload.citations:
            table.add_row(
                str(citation.slot),
                escape(citation.title or citation.uri),
                escape(_anchor_summary(citation.anchor)),
                escape(citation.verification),
            )
        out.print(table)
    elif payload.corpus_consulted:
        out.print("[yellow]no citation survived verification[/yellow]")

    facts: list[str] = []
    if not payload.corpus_consulted:
        facts.append("the corpus was not consulted, so this answer carries no citations")
    if payload.ungrounded:
        facts.append("[red]ungrounded[/red]: passages were found and none could be verified")
    if payload.confidence is not None:
        band = f" ({payload.confidence_band})" if payload.confidence_band else ""
        facts.append(f"confidence {payload.confidence:.2f}{band}")
    if payload.dropped:
        facts.append(f"{payload.dropped} citation(s) dropped")
    if payload.context_truncated:
        facts.append("context truncated to fit the model's window")
    if payload.redacted:
        facts.append("personal data was redacted before sending")
    facts.append(f"{payload.elapsed_ms} ms")
    out.print(f"[dim]{' · '.join(facts)}[/dim]")


def render_search(out: Console, payload: r.SearchResult) -> None:
    """Ranked passages, most relevant first."""
    if not payload.hits:
        # "nothing matched" is the same sentence whether the corpus is empty or the query
        # simply missed, and those need opposite things from the reader. The payload carries
        # no corpus size, so this says which question to ask rather than guessing the answer.
        out.print("[yellow]nothing matched[/yellow]")
        out.print(
            "[dim]if this is a new installation nothing may be indexed yet — "
            "[/dim]manicule index[dim] reports what is there, and "
            "[/dim]manicule index <path>[dim] adds to it[/dim]"
        )
    for position, hit in enumerate(payload.hits, start=1):
        heading = " / ".join(hit.heading_path)
        title = hit.title or hit.uri
        out.print(
            f"[bold]{position}.[/bold] {escape(title)}"
            f"[dim] {escape(heading)}  score {hit.score:.4f}[/dim]"
        )
        out.print(Text(_clip(hit.text), style="none"))
        out.print(f"[dim]{escape(hit.uri)} · {escape(_anchor_summary(hit.anchor))}[/dim]\n")
    summary = [f"{payload.count} hit(s)", f"profile {payload.profile}"]
    if payload.confidence is not None:
        summary.append(f"confidence {payload.confidence:.2f} ({payload.confidence_band})")
    if payload.cached:
        summary.append("served from the query cache")
    summary.append(f"{payload.elapsed_ms} ms")
    out.print(f"[dim]{' · '.join(summary)}[/dim]")


# --- documents --------------------------------------------------------------------------------


def render_document_list(out: Console, payload: r.DocumentList) -> None:
    """One record per document: the id on its own line, then what it is.

    Not a table, and the reason is the id. A listing exists to produce an identifier somebody
    pastes into the next command, and a table cell elides — so at any width narrow enough, an
    id reads as complete and is not. On its own line it wraps at worst, and every character
    survives. The prose that follows is what a table would have aligned, and it costs nothing
    to read as a run-on.
    """
    for document in payload.documents:
        out.print(document.id)
        facts = [
            escape(_clip(document.title or document.uri, 60)),
            escape(document.source),
            escape(document.media_type),
            _status(document.status),
        ]
        out.print(f"  [dim]{' · '.join(facts)}[/dim]")
    if not payload.documents:
        out.print("[dim]no documents[/dim]")
    out.print(
        f"[dim]{payload.count} shown, offset {payload.offset}, page size {payload.limit}[/dim]"
    )


def render_document(out: Console, payload: r.DocumentDetail) -> None:
    document = payload.document
    table = Table(box=None, show_header=False, pad_edge=False)
    table.add_column()
    # Values here are identifiers, URIs and hashes. Folded for the same reason the listing
    # folds its id column: an elided one reads as complete.
    table.add_column(overflow="fold")
    table.add_row("id", document.id)
    table.add_row("title", escape(document.title))
    table.add_row("uri", escape(document.uri))
    table.add_row("source", f"{escape(document.source)} / {escape(document.source_id)}")
    table.add_row("media type", escape(document.media_type))
    table.add_row("status", _status(document.status))
    if document.status_detail:
        table.add_row("detail", escape(document.status_detail))
    if document.failed_stage:
        table.add_row("failed at", escape(document.failed_stage))
    table.add_row("content hash", document.content_hash)
    out.print(table)
    for chunk in payload.chunks:
        heading = " / ".join(chunk.heading_path)
        out.print(
            f"\n[bold]#{chunk.position}[/bold] [dim]{escape(chunk.kind)} · "
            f"{chunk.token_count} tokens · {escape(heading)}[/dim]"
        )
        out.print(Text(chunk.text))


def render_document_deleted(out: Console, payload: r.DocumentDeleted) -> None:
    where = "the trash" if payload.mode == "soft" else "the index, permanently"
    out.print(f"removed [bold]{payload.document_id}[/bold] into {where}")


def render_document_reindexed(out: Console, payload: r.DocumentReindexed) -> None:
    out.print(
        f"{escape(payload.status)} [bold]{payload.document_id}[/bold] ({payload.chunks} chunk(s))"
    )
    if payload.detail:
        out.print(f"[yellow]{escape(payload.detail)}[/yellow]")


# --- ingest -----------------------------------------------------------------------------------


def render_ingest(out: Console, payload: r.IngestReport) -> None:
    table = Table(box=None, show_header=False, pad_edge=False)
    table.add_row("source", escape(payload.connector))
    table.add_row("discovered", str(payload.discovered))
    table.add_row("indexed", str(payload.ingested))
    table.add_row("unchanged", str(payload.skipped))
    table.add_row("failed", str(payload.failed))
    if payload.expanded:
        table.add_row("found inside others", str(payload.expanded))
    for status, count in sorted(payload.by_status.items()):
        table.add_row(f"  {status}", str(count))
    table.add_row("elapsed", f"{payload.elapsed_ms} ms")
    out.print(table)
    if payload.error:
        out.print(f"[red]the run did not finish: {escape(payload.error)}[/red]")
        out.print("[dim]the watermark was not advanced, so running it again resumes[/dim]")


# --- state ------------------------------------------------------------------------------------


def render_index_status(out: Console, payload: r.IndexStatus) -> None:
    table = _folding_table()
    table.add_row("documents", str(payload.documents))
    table.add_row("chunks", str(payload.chunks))
    for status, count in sorted(payload.by_status.items()):
        table.add_row(f"  {status}", str(count))
    table.add_row("embedding", escape(payload.embedding or "not yet committed"))
    table.add_row("chunking", escape(payload.chunking or "not yet committed"))
    table.add_row("schema", escape(payload.schema_revision or "unmigrated"))
    table.add_row("data directory", escape(payload.data_dir))
    out.print(table)
    # On their own lines rather than in the table: a fingerprint is an identity, and a table
    # cell truncates. A truncated identity is one nobody can compare, which is the only thing
    # it is for.
    if payload.embed_fingerprint:
        out.print(f"[dim]embed fingerprint: {escape(payload.embed_fingerprint)}[/dim]")
    if payload.chunk_fingerprint:
        out.print(f"[dim]chunk fingerprint: {escape(payload.chunk_fingerprint)}[/dim]")
    # An index reporting zeros is the state a first run is in, and a screen of zeros with no
    # next action is where that run stops.
    if not payload.documents:
        out.print("\n[dim]nothing is indexed yet. [/dim]manicule index <path>[dim] fills it[/dim]")


def render_stats(out: Console, payload: r.Stats) -> None:
    out.print(f"[bold]{payload.documents}[/bold] document(s), {payload.chunks} chunk(s)")
    for title, counts in (
        ("by source", payload.by_source),
        ("by media type", payload.by_media_type),
        ("by status", payload.by_status),
    ):
        if not counts:
            continue
        table = Table(title, "", box=None, pad_edge=False)
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            table.add_row(escape(key), str(count))
        out.print(table)


def render_diagnosis(out: Console, payload: r.Diagnosis) -> None:
    for check in payload.checks:
        style = _STATE_STYLE.get(check.state, "dim")
        out.print(
            f"[{style}]{check.state:>9}[/{style}]  [bold]{escape(check.name)}[/bold]  "
            f"{escape(check.detail)}"
        )
    style = _STATE_STYLE.get(payload.state, "dim")
    out.print(f"\noverall: [{style}]{payload.state}[/{style}]")


# --- connectors, configuration, workspaces, plugins -------------------------------------------


def render_connectors(out: Console, payload: r.ConnectorList) -> None:
    table = Table("name", "type", "enabled", "installed", "documents", box=None, pad_edge=False)
    for connector in payload.connectors:
        table.add_row(
            escape(connector.name),
            escape(connector.type),
            "yes" if connector.enabled else "no",
            "yes" if connector.installed else "[red]no[/red]",
            "—" if connector.documents is None else str(connector.documents),
        )
    out.print(table)
    if not payload.connectors:
        out.print("[dim]no sources configured[/dim]")


def render_config_value(out: Console, payload: r.ConfigValue) -> None:
    out.print_json(data=payload.value)
    out.print(f"[dim]from {escape(payload.source)}[/dim]")


def render_config_change(out: Console, payload: r.ConfigChange) -> None:
    out.print(f"[bold]{escape(payload.key)}[/bold]: {payload.previous!r} → {payload.value!r}")
    out.print(f"[dim]written to {escape(payload.path)}[/dim]")


def render_workspaces(out: Console, payload: r.WorkspaceList) -> None:
    table = Table("", "workspace", "mode", "documents", box=None, pad_edge=False)
    for workspace in payload.workspaces:
        table.add_row(
            "*" if workspace.active else " ",
            escape(workspace.id),
            escape(workspace.mode),
            "—" if workspace.documents is None else str(workspace.documents),
        )
    out.print(table)
    out.print("[dim]counts are reported for the active workspace only[/dim]")


def render_workspace_switched(out: Console, payload: r.WorkspaceSwitched) -> None:
    out.print(
        f"active workspace: {escape(payload.previous)} → [bold]{escape(payload.active)}[/bold]"
    )
    out.print(f"[dim]written to {escape(payload.path)}; it takes effect at the next start[/dim]")


def render_plugins(out: Console, payload: r.PluginList) -> None:
    for plugin in payload.plugins:
        out.print(
            f"[bold]{escape(plugin.name)}[/bold] {escape(plugin.version)} "
            f"[dim](core {escape(plugin.core_version)})[/dim]"
        )
        if plugin.summary:
            out.print(f"  {escape(plugin.summary)}")
        for component in plugin.components:
            out.print(f"    [dim]{escape(component.kind)}:{escape(component.name)}[/dim]")
    for name in payload.disabled:
        out.print(f"[dim]{escape(name)} (disabled)[/dim]")
    if payload.available:
        out.print("\n[bold]community registry[/bold]")
        table = Table("name", "version", "compatible", "installed", box=None, pad_edge=False)
        for entry in payload.available:
            table.add_row(
                escape(entry.name),
                escape(entry.version),
                "yes" if entry.compatible else f"[red]no[/red] {escape(entry.incompatible_reason)}",
                "yes" if entry.installed else "no",
            )
        out.print(table)
    if payload.registry_error:
        out.print(f"[yellow]{escape(payload.registry_error)}[/yellow]")


def render_plugin_changed(out: Console, payload: r.PluginChanged) -> None:
    state = "enabled" if payload.enabled else "disabled"
    out.print(f"[bold]{escape(payload.name)}[/bold]: {state}")
    if payload.detail:
        out.print(escape(payload.detail))
    if payload.path:
        out.print(f"[dim]written to {escape(payload.path)}[/dim]")


# --- operations ---------------------------------------------------------------------------------


def render_backup(out: Console, payload: r.BackupReport) -> None:
    out.print(f"backup written to [bold]{escape(payload.path)}[/bold]")
    out.print(
        f"[dim]{payload.files} file(s), {payload.bytes} bytes, schema "
        f"{escape(payload.schema_revision or 'unknown')}[/dim]"
    )
    for table, count in sorted(payload.counts.items()):
        out.print(f"  [dim]{escape(table)}: {count}[/dim]")


def render_restore(out: Console, payload: r.RestoreReport) -> None:
    out.print(f"restored {payload.files} file(s) into [bold]{escape(payload.data_dir)}[/bold]")
    out.print("[dim]start manicule again to reopen the restored database[/dim]")


def render_export(out: Console, payload: r.ExportReport) -> None:
    out.print(f"exported {payload.documents} document(s) to [bold]{escape(payload.path)}[/bold]")
    out.print(
        "[dim]source bytes and metadata only. The importing installation re-derives chunks "
        "and vectors with its own fingerprints[/dim]"
    )


def render_reset(out: Console, payload: r.ResetReport) -> None:
    out.print(f"removed {payload.documents} document(s) and {payload.chunks} chunk(s)")
    if not payload.vectors_removed:
        out.print("[dim]no vectors to remove[/dim]")


def render_init(out: Console, payload: r.InitReport) -> None:
    out.print(f"configuration written to [bold]{escape(payload.path)}[/bold]")
    table = _folding_table()
    table.add_row("data directory", escape(payload.data_dir))
    table.add_row(
        "embedding", f"{escape(payload.embedding_provider)} · {escape(payload.embedding_model)}"
    )
    table.add_row("generation", f"{escape(payload.llm_provider)} · {escape(payload.llm_model)}")
    out.print(table)
    for note in payload.notes:
        out.print(f"[dim]{escape(note)}[/dim]")
    # The one thing a person needs after `init`, and the only command that has to come next.
    # Without it the first run ends on a report with nothing to do about it.
    out.print("\n[dim]next: [/dim]manicule index <path>[dim], then[/dim] manicule search <query>")


API_TRANSPORT = "http-api"
"""What the HTTP API records as its transport, mirroring :data:`manicule.api.serve.TRANSPORT`.

Named here rather than imported, because importing it would pull FastAPI into every ``manicule
--help``. :func:`render_address` is the one reader, and ``tests/app/test_serving.py`` asserts
the two strings agree so the mirror cannot drift.
"""


def render_address(out: Console, payload: r.ServerAddress, *, web: bool | None = None) -> None:
    """Where the server is listening, and which surface is on it.

    **Which surface is read off the transport, not passed in.** The MCP server records
    ``http`` and the HTTP API records ``http-api`` — a distinction the pid file has always
    carried and this renderer ignored, announcing every socket as "MCP server" including the
    one people open in a browser. ``manicule start --transport http`` printed ``MCP server on
    http://127.0.0.1:8765, 11 tool(s)`` while serving the REST API and the browser surface, and
    named neither URL; ``manicule stop`` said the same thing about the server it had just
    stopped.

    ``web`` is the one fact the payload cannot carry, because ``--no-web`` is not recorded
    anywhere. ``None`` means nobody said — which is the honest answer for ``stop``, reading a
    pid file written by a process whose flags it never saw — and prints no claim either way.
    """
    if payload.transport == "stdio":
        out.print(f"MCP server on stdio, {payload.tools} tool(s). [dim]No socket is open.[/dim]")
        return
    where = f"http://{payload.host}:{payload.port}"
    serves_api = payload.transport == API_TRANSPORT
    what = "HTTP API" if serves_api else "MCP server"
    if payload.loopback:
        out.print(f"{what} on {where} [dim](this machine only)[/dim]")
    else:
        out.print(f"[red]{what} on {where} — reachable from the network[/red]")
    if not serves_api:
        out.print(f"[dim]{payload.tools} tool(s)[/dim]")
        return
    # The signposts a person wants next, and none of them were printed before. A server whose
    # browser surface is not named is one nobody finds without reading the source.
    if web is True:
        out.print(f"[dim]browser surface[/dim]  {where}/ui")
    elif web is False:
        out.print("[dim]browser surface off (--no-web)[/dim]")
    out.print(f"[dim]API documentation[/dim] {where}/api/docs")


def render_upgrade(out: Console, payload: r.UpgradeReport) -> None:
    out.print(
        f"installed: [bold]{escape(payload.current)}[/bold], requested: {escape(payload.target)}"
    )
    if payload.backup:
        out.print(f"[dim]backup taken at {escape(payload.backup)}[/dim]")
    out.print(escape(payload.detail))


def render_completion(out: Console, payload: r.CompletionScript) -> None:
    # Printed raw. This output is meant to be redirected into a file the shell sources, so
    # markup or wrapping would corrupt it.
    out.file.write(payload.script)
    if not payload.script.endswith("\n"):
        out.file.write("\n")


def render_api_key_issued(out: Console, payload: r.ApiKeyIssued) -> None:
    out.print(
        f"key [bold]{escape(payload.key.name)}[/bold] created for {escape(payload.key.workspace)}"
    )
    out.print(Panel(payload.secret, title="shown once", border_style="yellow"))
    out.print("[dim]only its digest is stored; a lost key is reissued, never recovered[/dim]")


def render_api_keys(out: Console, payload: r.ApiKeyList) -> None:
    table = Table("name", "prefix", "role", "expires", "revoked", box=None, pad_edge=False)
    for key in payload.keys:
        table.add_row(
            escape(key.name),
            escape(key.prefix),
            escape(key.role),
            escape(key.expires_at or "never"),
            "yes" if key.revoked else "no",
        )
    out.print(table)


def render_api_key_revoked(out: Console, payload: r.ApiKeyRevoked) -> None:
    out.print(f"revoked [bold]{escape(payload.name)}[/bold] ({payload.id})")


RENDERERS: Mapping[type[Payload], Callable[[Console, Payload], None]] = {
    r.AnswerResultPayload: lambda out, p: render_answer(out, _as(r.AnswerResultPayload, p)),
    r.SearchResult: lambda out, p: render_search(out, _as(r.SearchResult, p)),
    r.DocumentList: lambda out, p: render_document_list(out, _as(r.DocumentList, p)),
    r.DocumentDetail: lambda out, p: render_document(out, _as(r.DocumentDetail, p)),
    r.DocumentDeleted: lambda out, p: render_document_deleted(out, _as(r.DocumentDeleted, p)),
    r.DocumentReindexed: lambda out, p: render_document_reindexed(out, _as(r.DocumentReindexed, p)),
    r.IngestReport: lambda out, p: render_ingest(out, _as(r.IngestReport, p)),
    r.IndexStatus: lambda out, p: render_index_status(out, _as(r.IndexStatus, p)),
    r.Stats: lambda out, p: render_stats(out, _as(r.Stats, p)),
    r.Diagnosis: lambda out, p: render_diagnosis(out, _as(r.Diagnosis, p)),
    r.ConnectorList: lambda out, p: render_connectors(out, _as(r.ConnectorList, p)),
    r.ConfigValue: lambda out, p: render_config_value(out, _as(r.ConfigValue, p)),
    r.ConfigChange: lambda out, p: render_config_change(out, _as(r.ConfigChange, p)),
    r.WorkspaceList: lambda out, p: render_workspaces(out, _as(r.WorkspaceList, p)),
    r.WorkspaceSwitched: lambda out, p: render_workspace_switched(out, _as(r.WorkspaceSwitched, p)),
    r.PluginList: lambda out, p: render_plugins(out, _as(r.PluginList, p)),
    r.PluginChanged: lambda out, p: render_plugin_changed(out, _as(r.PluginChanged, p)),
    r.BackupReport: lambda out, p: render_backup(out, _as(r.BackupReport, p)),
    r.RestoreReport: lambda out, p: render_restore(out, _as(r.RestoreReport, p)),
    r.ExportReport: lambda out, p: render_export(out, _as(r.ExportReport, p)),
    r.ResetReport: lambda out, p: render_reset(out, _as(r.ResetReport, p)),
    r.InitReport: lambda out, p: render_init(out, _as(r.InitReport, p)),
    r.ServerAddress: lambda out, p: render_address(out, _as(r.ServerAddress, p)),
    r.UpgradeReport: lambda out, p: render_upgrade(out, _as(r.UpgradeReport, p)),
    r.CompletionScript: lambda out, p: render_completion(out, _as(r.CompletionScript, p)),
    r.ApiKeyIssued: lambda out, p: render_api_key_issued(out, _as(r.ApiKeyIssued, p)),
    r.ApiKeyList: lambda out, p: render_api_keys(out, _as(r.ApiKeyList, p)),
    r.ApiKeyRevoked: lambda out, p: render_api_key_revoked(out, _as(r.ApiKeyRevoked, p)),
}
"""Payload type to renderer.

A table rather than a chain of ``isinstance``, so an operation that gained a payload and no
renderer fails loudly at the one place that knows, instead of printing nothing.
"""


def render(out: Console, payload: Payload) -> None:
    """Write ``payload`` to ``out``.

    Raises:
        KeyError: No renderer for that payload type. Deliberately not a fallback that dumps
            the model: a payload nobody wrote a view for is an operation whose output nobody
            has looked at, and silence is how that ships.
    """
    RENDERERS[type(payload)](out, payload)


def _as[T: Payload](kind: type[T], payload: Payload) -> T:
    if not isinstance(payload, kind):  # pragma: no cover - the table is keyed by exact type
        msg = f"expected {kind.__name__}, got {type(payload).__name__}"
        raise TypeError(msg)
    return payload


def _folding_table() -> Table:
    """A two-column table whose value column wraps rather than elides.

    Rich truncates a cell that does not fit and marks it with ``…``, which is the right default
    for prose and the wrong one for a path. ``manicule init`` and ``manicule index`` both report
    the data directory, and on an ordinary terminal both printed it as
    ``/private/tmp/…/scratchpad/c4fdec3b…`` — a location the reader cannot cd to, copy, or check
    the permissions of, presented as though it were the whole value.

    :func:`render_document` already folds for exactly this reason. This is the same decision in
    the two other places that print a filesystem path.
    """
    table = Table(box=None, show_header=False, pad_edge=False)
    table.add_column()
    table.add_column(overflow="fold")
    return table


def _status(status: str) -> str:
    style = _STATUS_STYLE.get(status, "dim")
    return f"[{style}]{escape(status)}[/{style}]"


def _heading_trail(anchor: Mapping[str, object]) -> str:
    trail: object = anchor.get("path")
    if isinstance(trail, (list, tuple)):
        parts = cast("Sequence[object]", trail)
        return " / ".join(str(part) for part in parts)
    return "heading"


_ANCHOR_SUMMARIES: Mapping[str, Callable[[Mapping[str, object]], str]] = {
    "page": lambda a: f"page {a.get('page')}",
    "line": lambda a: f"lines {a.get('start_line')}-{a.get('end_line')}",
    "heading": _heading_trail,
    "cell": lambda a: f"{a.get('sheet')}!{a.get('ref')}",
    "unlocated": lambda a: f"no location ({a.get('reason', 'unstated')})",
}
"""How each anchor kind reads in one line.

A table rather than a chain of comparisons so that an anchor kind added later is a missing
key — visible as ``unknown`` — rather than a branch somebody forgot to add.
"""


def _anchor_summary(anchor: Mapping[str, object]) -> str:
    """One line describing a location, without inventing precision it does not have."""
    kind = anchor.get("kind")
    summarise = _ANCHOR_SUMMARIES.get(str(kind)) if kind is not None else None
    if summarise is not None:
        return summarise(anchor)
    return str(kind or "unknown")


def _clip(text: str, limit: int = 400) -> str:
    stripped = " ".join(text.split())
    return stripped if len(stripped) <= limit else stripped[: limit - 1] + "…"


__all__ = [
    "RENDERERS",
    "console",
    "render",
    "render_error",
]
