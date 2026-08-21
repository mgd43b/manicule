"""Turning a payload into something worth reading in a terminal.

Rendering only. Every function here takes a payload the service already produced and writes it
to a console; none of them decides anything, computes anything or reaches for a store. That is
what keeps the command line an adapter — and it is what makes ``--json`` and the human output
two views of one result rather than two results.

The renderers are looked up by payload type, so adding an operation means adding a payload and
a renderer, and forgetting the renderer is a loud failure rather than a blank screen.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final, cast

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from manicule.app import frontdoor
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
    inserting escape sequences *inside* the token — so with color enabled a version prints as
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


EXPLICIT_DEFINITION: Final = "an explicit definition of a term you asked about is cited above"
"""What the terminal says when ``explicit_definition`` is true.

Printed **beside** the confidence figure rather than in place of it, and it names a passage
rather than a score. That is the whole of the presentation rule: the number is unchanged — a
run that reports ``0.00 (none)`` still reports ``0.00 (none)`` — and what the reader gains is
the fact that one of the passages in front of them is the corpus's own statement of what the
term means. A line that dressed this up as better evidence would be doing exactly what the
classification exists to avoid, which is letting a lookup be read as a similarity.

The reason line under it still prints, because ``none`` and ``low`` are the bands whose reason
is worth the space and this is a case that lands squarely in ``none``.
"""


def render_answer(
    out: Console, payload: r.AnswerResultPayload, *, text_already_shown: bool = False
) -> None:
    """The answer, then its citations, then what was true of the run.

    Citations are listed in full rather than summarized. A citation is the product here — an
    answer whose sources are collapsed into "3 sources" is an answer nobody can check.

    ``text_already_shown`` is for the caller that streamed the tokens as they arrived. The
    answer is the same either way — the payload is identical whether or not anything was
    streamed — so printing it a second time would show a reader the same paragraph twice and
    leave them wondering which one to trust.
    """
    _render_glossary(out, payload.expansions, payload.conflicts)
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
    if payload.explicit_definition:
        facts.append(EXPLICIT_DEFINITION)
    if payload.dropped:
        facts.append(f"{payload.dropped} citation(s) dropped")
    if payload.context_truncated:
        facts.append("context truncated to fit the model's window")
    if payload.redacted:
        facts.append("personal data was redacted before sending")
    facts.append(f"{payload.elapsed_ms} ms")
    out.print(f"[dim]{' · '.join(facts)}[/dim]")
    _render_confidence_reason(out, payload.confidence_band, payload.confidence_reason)


def _render_glossary(
    out: Console,
    expansions: Sequence[r.GlossaryExpansion],
    conflicts: Sequence[r.GlossaryConflict],
    expanded_query: str = "",
) -> None:
    """Say which words the search actually ran on, and where they came from.

    Printed **before** the results rather than in the summary line, because it changes how the
    results should be read: a reader looking at passages retrieved partly through words they
    did not type needs to know that before they read them, not after.

    Every line names its source. That is not presentation — ``bugs/bug2.md`` §3 forbids showing
    an expansion without its provenance, and this is the surface where the rule is easiest to
    break by writing one fewer field.
    """
    for expansion in expansions:
        out.print(
            f"[cyan]{escape(expansion.display)}[/cyan] expands to "
            f"[bold]{escape(expansion.expansion)}[/bold]"
            f"[dim] — defined in {escape(expansion.title or expansion.uri)}"
            f"{', ' + escape(expansion.location) if expansion.location else ''}"
            f" ({escape(expansion.reason)})[/dim]"
        )
    if expanded_query:
        out.print(f"[dim]also searched: {escape(expanded_query)}[/dim]")
    for conflict in conflicts:
        # Never resolved, here or anywhere. Both definitions are printed with their sources so
        # the reader picks; choosing one and showing it as *the* expansion is the failure this
        # whole feature is most able to cause.
        out.print(
            f"[yellow]{escape(conflict.acronym)} has "
            f"{len(conflict.candidates)} conflicting definitions in scope, so it was not "
            f"expanded:[/yellow]"
        )
        for candidate in conflict.candidates:
            out.print(
                f"  [dim]·[/dim] {escape(candidate.expansion)}"
                f"[dim] — {escape(candidate.title or candidate.uri)}"
                f"{', ' + escape(candidate.location) if candidate.location else ''}[/dim]"
            )


def render_search(out: Console, payload: r.SearchResult) -> None:
    """Ranked passages, most relevant first."""
    _render_glossary(out, payload.expansions, payload.conflicts, payload.expanded_query)
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
    if payload.explicit_definition:
        summary.append(EXPLICIT_DEFINITION)
    if payload.cached:
        summary.append("served from the query cache")
    summary.append(f"{payload.elapsed_ms} ms")
    out.print(f"[dim]{' · '.join(summary)}[/dim]")
    _render_confidence_reason(out, payload.confidence_band, payload.confidence_reason)


UNCONVINCED_BANDS: Final[frozenset[str]] = frozenset({"none", "low"})
"""Bands whose reason is worth the line it takes.

The reason for a *high* confidence is "the passages scored well", which the scores beside
each hit already said. The reason for `none` is that every passage retrieved is one the
corpus would have returned for any question at all — and without that sentence the reader is
looking at three plausible-looking excerpts and one number they have no scale for.
"""


def _render_confidence_reason(out: Console, band: str | None, reason: str | None) -> None:
    """Say *why* the score is what it is, when the score alone is misleading.

    ``SearchResult`` has carried this since retrieval learned to admit ignorance, and the
    browser surface renders it (``web/templates/search.html``) while the command line did
    not — so the same query showed a sentence in one place and a bare ``0.00 (none)`` above
    three plausible-looking excerpts in the other. The band gates it because a reason nobody
    needed, printed every time, is how a reader learns to skip the last line.

    Both commands now, which is the point: the number is the same judgment whether it is
    printed under passages or under an answer, and it was legible in one place only.
    """
    if reason and band in UNCONVINCED_BANDS:
        out.print(f"[yellow]{escape(reason)}[/yellow]")


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
    """One document's outcome, with the detail colored by whether it is a problem.

    ``superseded`` is dim rather than yellow for the same reason the sweep's list of them is:
    it is the only one of the four statuses that needs nothing done about it, and coloring it
    like an unrepairable document would send somebody looking for a fault that is not there.
    """
    out.print(
        f"{escape(payload.status)} [bold]{payload.document_id}[/bold] ({payload.chunks} chunk(s))"
    )
    if payload.detail:
        style = "dim" if payload.status == "superseded" else "yellow"
        out.print(f"[{style}]{escape(payload.detail)}[/{style}]")


def render_stale_reparse(out: Console, payload: r.StaleReparseReport) -> None:
    """The sweep's counts, and the documents somebody has to do something about.

    The same numbers ``--json`` carries, off the same payload. Nothing is computed here: a
    percentage or a "looks healthy" would be this surface's own opinion about a result the
    other surfaces report without one.
    """
    if payload.dry_run:
        out.print("[dim]dry run: nothing was parsed, embedded or written[/dim]")
    table = Table(box=None, show_header=False, pad_edge=False)
    table.add_row("selected", str(payload.selected))
    if not payload.dry_run:
        table.add_row("re-parsed", str(payload.reparsed))
        table.add_row("  unchanged", str(payload.unchanged))
        table.add_row("  changed", str(payload.changed))
        table.add_row("chunks produced anew", str(payload.chunks_new))
        # "vector rows", not "vectors". A kept chunk keeps the row every citation to it resolves
        # through; the row's contents are written again, because what is embedded carries the
        # heading breadcrumb and that can move under an id that did not.
        table.add_row("chunks kept, with their vector rows", str(payload.chunks_kept))
        # Only on a real run, like the counts above it: a plan writes nothing, so there is
        # nothing for a concurrent sync to overtake and a zero here would be a fact about
        # arithmetic rather than about the corpus.
        table.add_row("superseded by a newer sync", str(payload.superseded))
        # Under its own heading rather than beside the chunk counts, because the two answer
        # different questions and the row above used to be read as an answer to this one.
        # Whether the vector inside a kept row survived is the block below, not this line.
        cost = payload.embedding
        table.add_row("", "")
        table.add_row("embedding", "")
        table.add_row("  vectors reused", str(cost.reused))
        table.add_row("  chunks embedded", str(cost.embedded))
        table.add_row("    input changed", str(cost.input_changed))
        table.add_row("    not seen before", str(cost.first_seen))
        table.add_row("    vector missing or corrupt", str(cost.repaired))
        table.add_row("    into a new row", str(cost.vectors_new))
        table.add_row("    over an existing row", str(cost.vectors_replaced))
        table.add_row("  forward calls", str(cost.forward_calls))
        if cost.cache_hits:
            # Only when it happened. A zero would invite the reading this whole block exists to
            # prevent — that the in-memory cache is what avoided the work.
            table.add_row("  served by the warm cache", str(cost.cache_hits))
        if cost.vectors_backfilled:
            table.add_row("  identities backfilled", str(cost.vectors_backfilled))
    table.add_row("unrepairable", str(payload.unrepairable))
    table.add_row("failed", str(payload.failed))
    out.print(table)
    # Named individually, unlike the counts. These are the only two classes an operator can act
    # on, and "3 unrepairable" without the ids is a number nobody can do anything with.
    for line in payload.unrepairable_documents:
        out.print(f"[yellow]{escape(line)}[/yellow]")
    for line in payload.failures:
        out.print(f"[red]{escape(line)}[/red]")
    # Dim rather than yellow, and after both: this is the one list here that is not a call to
    # action. The document is current because something else made it current, and an operator
    # who reads these in color order should reach it last and stop.
    for line in payload.superseded_documents:
        out.print(f"[dim]{escape(line)}[/dim]")
    if payload.dry_run and payload.selected:
        out.print("\n[dim]next: [/dim]manicule document reindex --stale")
    elif payload.unrepairable:
        out.print(
            "\n[dim]a document with no retained bytes can only be repaired by fetching it "
            "again: [/dim]manicule connector sync <name>"
        )


def render_stale_glossary(out: Console, payload: r.StaleGlossaryReport) -> None:
    """The glossary sweep's counts, and the documents whose text the detector could not read.

    Entries rather than chunks, because this rung is charged in neither. What it spent is a
    pass over stored text; what it produced is a vocabulary, and the two totals either side of
    it are the honest way to report a pass that removed eleven wrong entries and added eleven
    right ones.

    **No definition reaches this function.** The payload carries none, which is deliberate — the
    subject of this command is the corpus's own vocabulary, and a terminal is the last place to
    print it.
    """
    if payload.dry_run:
        out.print("[dim]dry run: nothing was detected or written[/dim]")
    table = Table(box=None, show_header=False, pad_edge=False)
    table.add_row("selected", str(payload.selected))
    if not payload.dry_run:
        table.add_row("re-detected", str(payload.redetected))
        table.add_row("  unchanged", str(payload.unchanged))
        table.add_row("  changed", str(payload.changed))
        table.add_row("entries", f"{payload.entries_before} -> {payload.entries_after}")
    table.add_row("superseded", str(payload.superseded))
    table.add_row("unrepairable", str(payload.unrepairable))
    table.add_row("failed", str(payload.failed))
    out.print(table)
    # Named individually, unlike the counts, and in increasing order of how much somebody has to
    # do about them: a supersession is dim because nothing needs doing, an unrepairable document
    # is yellow because it needs a command, and a failure is red because it is a defect.
    for line in payload.superseded_documents:
        out.print(f"[dim]{escape(line)}[/dim]")
    for line in payload.unrepairable_documents:
        out.print(f"[yellow]{escape(line)}[/yellow]")
    for line in payload.failures:
        out.print(f"[red]{escape(line)}[/red]")
    if payload.unrepairable:
        out.print(
            "\n[dim]a document whose chunks are gone needs them rebuilt from retained bytes "
            "first: [/dim]manicule document reindex --stale"
        )
    if payload.dry_run and payload.selected:
        out.print("\n[dim]next: [/dim]manicule document reindex --stale-glossary")


# --- ingest -----------------------------------------------------------------------------------


def _add_enumeration_rows(table: Table, payload: r.IngestReport) -> None:
    """Adaptive-pagination rows, only when the source's latency changed the request shape.

    A row reading "no" on every ordinary run is a row nobody reads on the run where it
    finally says something.
    """
    if payload.enumeration_timeout_retries:
        table.add_row("source timeout retries", str(payload.enumeration_timeout_retries))
    if payload.enumeration_page_size_reduced and payload.enumeration_page_size is not None:
        table.add_row("page size reduced to", str(payload.enumeration_page_size))
    if payload.enumeration_failure_code:
        table.add_row("enumeration stopped by", escape(payload.enumeration_failure_code))


def _retry_advice(payload: r.IngestReport) -> str:
    """What to do about an incomplete run, in terms of the thing that would change it."""
    if payload.enumeration_failure_code == "source_timeout":
        # The one remedy that is a setting rather than a repeat: the source stayed healthy
        # and the bounded shrink ran out, so the dials are what change the outcome. Named,
        # because "run it again" on its own would loop.
        return (
            "the source stayed reachable but a request kept timing out after the page size "
            "was reduced as far as adaptive_min_page_size allows; the durable prefix was kept "
            "and the watermark was not advanced. Running again resumes; if it recurs, lower "
            "adaptive_min_page_size or raise adaptive_max_attempts_per_offset"
        )
    if payload.inventory_recovery == "reenumeration_required":
        return (
            "the watermark was not advanced; running again starts a fresh fenced source enumeration"
        )
    return "the watermark was not advanced, so running it again resumes"


def render_ingest(out: Console, payload: r.IngestReport) -> None:
    table = Table(box=None, show_header=False, pad_edge=False)
    table.add_row("source", escape(payload.connector))
    table.add_row("discovered", str(payload.discovered))
    table.add_row("indexed", str(payload.ingested))
    table.add_row("unchanged", str(payload.skipped))
    table.add_row("failed", str(payload.failed))
    table.add_row("outcome", payload.outcome)
    table.add_row("enumeration completed", "yes" if payload.enumeration_completed else "no")
    table.add_row("watermark advanced", "yes" if payload.watermark_advanced else "no")
    table.add_row("retry required", "yes" if payload.retry_required else "no")
    if payload.full_inventory_authority:
        table.add_row("full inventory authority", escape(payload.full_inventory_authority))
    if payload.inventory_recovery:
        table.add_row("inventory recovery", escape(payload.inventory_recovery))
    if payload.reconciled_deleted_items:
        table.add_row("disappeared items reconciled", str(payload.reconciled_deleted_items))
    _add_enumeration_rows(table, payload)
    if payload.derivation_deferred:
        table.add_row("derivation deferred", "yes (snapshot retained locally)")
    if payload.expanded:
        table.add_row("found inside others", str(payload.expanded))
    for status, count in sorted(payload.by_status.items()):
        table.add_row(f"  {status}", str(count))
    table.add_row("elapsed", f"{payload.elapsed_ms} ms")
    out.print(table)
    if payload.error:
        out.print(f"[red]the run did not finish: {escape(payload.error)}[/red]")
        out.print("[dim]the watermark was not advanced, so running it again resumes[/dim]")
        return
    if payload.retry_required:
        detail = payload.incomplete_reason.message if payload.incomplete_reason else "unknown"
        out.print(f"[red]the run did not finish: {escape(detail)}[/red]")
        out.print(f"[dim]{_retry_advice(payload)}[/dim]")
        return
    if payload.intentionally_bounded:
        out.print(
            "[dim]the requested limit stopped discovery; the watermark was not advanced[/dim]"
        )
        return
    # The longest command in the first run ends here, often after minutes, and ended on a
    # table with nothing to do about it — the same gap `init` had. Only when something was
    # actually indexed: after a run that added nothing, "now search it" is advice about
    # somebody else's corpus.
    if payload.ingested:
        out.print(
            "\n[dim]next: [/dim]manicule search <query>[dim], or[/dim] manicule ask <question>"
        )


# --- state ------------------------------------------------------------------------------------


_VECTOR_INDEX_STYLE: Mapping[str, str] = {
    "disabled": "dim",
    "exhaustive": "green",
    "pending": "yellow",
    "ready": "green",
    "stale": "yellow",
}
"""How each lifecycle reads at a glance.

``exhaustive`` is green rather than dim: exact search is the design working, not a feature
switched off. ``pending`` and ``stale`` are the two that want somebody, and neither is red —
both are slow rather than wrong, and coloring a latency finding like a fault trains people to
ignore the color that means data is missing.
"""


def _build_hint(reason: str) -> str:
    """``reason``, dimmed, with the command to type left bright inside it.

    The markup has to balance within the one string: a cell that opened no ``[dim]`` and closed
    one would raise :class:`rich.errors.MarkupError` at render time rather than merely looking
    wrong, and it would do it only on the branch where a build is due — which is the branch
    somebody sees when something needs doing.
    """
    return f"[dim]{escape(reason)}; run [/dim]manicule build-vector-index --yes"


def _add_vector_index_rows(table: Table, index: r.VectorIndexState) -> None:
    """The dense search path, as two or three lines under the counts.

    Only ever a summary. What the index *is* — its partitions, its metric, its build — is on
    `--json` for anyone comparing two installations, because a person running `status` is
    asking whether search is all right rather than how it is configured.
    """
    style = _VECTOR_INDEX_STYLE.get(index.lifecycle, "dim")
    table.add_row("vector search", f"[{style}]{escape(index.lifecycle)}[/{style}]")
    if index.exact:
        table.add_row(
            "  neighbors",
            "exact over every vector" + (f", {index.rows} of them" if index.rows else ""),
        )
        if index.due:
            table.add_row("  awaiting build", _build_hint(f"past the {index.threshold} threshold"))
        return
    table.add_row("  index", escape(index.index_name or "unnamed"))
    if index.unindexed_rows:
        # Named as a scan rather than as a gap, because that is what it costs. The rows are
        # searched either way; what the number measures is how much of each query is linear.
        table.add_row("  scanned per query", f"{index.unindexed_rows} row(s) not yet indexed")
    if index.due:
        table.add_row("  refresh", _build_hint("the uncovered tail crossed the threshold"))


def render_vector_sweep(out: Console, payload: r.VectorSweepReport) -> None:
    """What one pass removed, or why it declined.

    A pass that removed nothing and a pass that never ran are printed differently, because they
    mean opposite things: the first is a clean index and the second is work still outstanding.
    """
    if not payload.ran:
        out.print(f"[yellow]the sweep did not run:[/yellow] {escape(payload.blocked_by)}")
        return
    table = _folding_table()
    table.add_row("vectors removed", str(payload.vectors_removed))
    table.add_row("documents purged", str(payload.documents_purged))
    out.print(table)
    if not payload.vectors_removed and not payload.documents_purged:
        out.print("[dim]nothing was waiting to be swept[/dim]")


def _add_vector_checksum_rows(table: Table, coverage: r.VectorChecksumCoverageReport) -> None:
    """Numerical integrity, as one line under the counts and a second only when it is owed.

    Silent on a table nobody has looked at and on one that is fully covered by a mere count,
    because a status page that printed a line about checksums every time would train people to
    skip it — and the one time it says ``failed`` is the time it must not be skipped.
    """
    if not coverage.scanned:
        return
    if coverage.failed:
        listed = ", ".join(f"{count} {kind}" for kind, count in sorted(coverage.failures.items()))
        table.add_row("vector integrity", f"[red]{coverage.failed} row(s) failed[/red]")
        table.add_row("  refusals", escape(listed))
        return
    if coverage.unverified:
        table.add_row(
            "vector integrity",
            f"[yellow]{coverage.recorded} of {coverage.rows} row(s) checksummed[/yellow]",
        )
        table.add_row(
            "  unverified",
            f"[dim]{coverage.unverified} row(s) predate checksums; run "
            f"[/dim]manicule vector-checksum --yes",
        )
        return
    if coverage.recomputed:
        table.add_row("vector integrity", f"[green]{coverage.verified} row(s) verified[/green]")


def render_vector_checksum(out: Console, payload: r.VectorChecksumReport) -> None:
    """Coverage, what the pass did, and the one sentence saying what to do next.

    The detail line is always printed, including when nothing is wrong — this is a command
    somebody runs *because* they are worried about a directory, and "nothing here needs doing"
    is the answer they came for rather than noise.
    """
    if not payload.supported:
        out.print(f"[yellow]{escape(payload.detail)}[/yellow]")
        return
    coverage = payload.coverage
    table = _folding_table()
    table.add_row("vectors", str(coverage.rows))
    table.add_row("checksummed", f"{coverage.recorded} of {coverage.rows}")
    if coverage.recomputed:
        table.add_row("verified", str(coverage.verified))
        table.add_row("failed", str(coverage.failed))
    if payload.written or payload.remaining or payload.unhashable:
        table.add_row("written this pass", f"{payload.written} of {payload.scanned} read")
        if payload.unhashable:
            table.add_row("left untouched", f"{payload.unhashable} unhashable row(s)")
        table.add_row("remaining", str(payload.remaining))
    out.print(table)
    if payload.dry_run and payload.remaining:
        out.print("[dim]nothing was written; add [/dim]--yes[dim] to run one pass[/dim]")
    out.print(f"[dim]{escape(payload.detail)}[/dim]")


def render_vector_index(out: Console, payload: r.VectorIndexReport) -> None:
    table = _folding_table()
    before, after = payload.before, payload.after
    table.add_row("was", escape(before.lifecycle))
    table.add_row("now", escape(after.lifecycle))
    table.add_row("vectors", str(after.rows))
    if after.index_name:
        table.add_row("index", escape(after.index_name))
        table.add_row(
            "shape",
            f"{escape(after.index_type or 'unknown')}, "
            f"{after.num_partitions} partition(s), {after.num_sub_vectors} sub-vector(s), "
            f"{escape(after.distance_metric or 'unknown')} distance",
        )
        table.add_row(
            "coverage", f"{after.coverage:.1%} of {after.indexed_rows + after.unindexed_rows}"
        )
    if payload.replaced:
        table.add_row("replaced", escape(payload.replaced))
    out.print(table)
    if payload.detail:
        out.print(f"[dim]{escape(payload.detail)}[/dim]")
    if payload.dry_run:
        out.print("\n[dim]a plan. [/dim]--yes[dim] performs it[/dim]")


def render_index_status(out: Console, payload: r.IndexStatus) -> None:
    table = _folding_table()
    table.add_row("documents", str(payload.documents))
    table.add_row("chunks", str(payload.chunks))
    for status, count in sorted(payload.by_status.items()):
        table.add_row(f"  {status}", str(count))
    table.add_row("embedding", escape(payload.embedding or "not yet committed"))
    table.add_row("chunking", escape(payload.chunking or "not yet committed"))
    table.add_row("glossary", escape(payload.glossary or "not reported"))
    if payload.stale_glossary:
        # Beside the detector it disagrees with, rather than left to `doctor`. Somebody reading
        # `status` after upgrading is asking exactly this question, and a line naming the
        # detector with nothing next to it would read as "current".
        table.add_row("  awaiting re-detection", str(payload.stale_glossary))
    if payload.vector_index is not None:
        _add_vector_index_rows(table, payload.vector_index)
    if payload.vector_checksums is not None:
        _add_vector_checksum_rows(table, payload.vector_checksums)
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
    table = Table(
        "name",
        "type",
        "enabled",
        "installed",
        "documents",
        "last outcome",
        "retry",
        "inventory recovery",
        "full inventory authority",
        box=None,
        pad_edge=False,
    )
    for connector in payload.connectors:
        table.add_row(
            escape(connector.name),
            escape(connector.type),
            "yes" if connector.enabled else "no",
            "yes" if connector.installed else "[red]no[/red]",
            "—" if connector.documents is None else str(connector.documents),
            connector.last_outcome or "—",
            "[red]yes[/red]" if connector.retry_required else "no",
            (
                connector.last_lifecycle.inventory_recovery
                if connector.last_lifecycle is not None
                and connector.last_lifecycle.inventory_recovery
                else "—"
            ),
            connector.full_inventory_authority or "—",
        )
    out.print(table)
    if not payload.connectors:
        out.print("[dim]no sources configured[/dim]")


def render_sidecar(out: Console, payload: r.SidecarReport) -> None:
    """What a conversion did, and every page it did nothing for.

    Skipped pages are printed rather than counted. "Considered 40, wrote 0" is equally the shape
    of a wrong directory, a page format this does not recognize, and a corpus already converted —
    and which of the three it is decides the operator's next move entirely.
    """
    out.print(
        f"[bold]{payload.written}[/bold] of {payload.considered} page(s) under "
        f"{escape(payload.root)}"
    )
    # Which profiles ran, always, and not only when something went wrong. "No pages adapted" and
    # "no pages adapted, looking for the default profile" are the same line to a reader who has
    # to remember whether they typed `--source`, and the second is the one that says what to fix.
    origin = f"source {escape(payload.source)}" if payload.source else "no configured source"
    out.print(f"[dim]{origin}; profile(s): {escape(', '.join(payload.profiles) or 'none')}[/dim]")
    if payload.considered and not payload.written:
        # Said outright rather than left to be read off a zero. "0 of 40" is a number somebody
        # scans past; "adapted no pages at all" is the sentence that makes them read the reasons
        # underneath it, which is where the answer to "why not" actually is.
        out.print(
            "[yellow]adapted no pages at all[/yellow] — every considered file is listed below"
        )
    if not payload.skipped:
        return
    table = Table("skipped", "why", box=None, pad_edge=False)
    for skip in payload.skipped:
        table.add_row(escape(skip.path), escape(skip.reason))
    out.print(table)


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
    out.print(
        f"reset derived state for {payload.documents} document(s); "
        f"removed {payload.chunks} chunk(s)"
    )
    out.print(f"[dim]{payload.snapshots_retained} durable snapshot item(s) retained[/dim]")
    if not payload.vectors_removed:
        out.print("[dim]no vectors to remove[/dim]")


def render_lifecycle(out: Console, payload: r.LifecycleReport) -> None:
    action = "dry run" if payload.dry_run else "completed"
    out.print(f"[bold]{escape(payload.operation)}[/bold] · {action}")
    if payload.dry_run:
        out.print(
            f"eligible: {payload.eligible_items} item(s), {payload.eligible_bytes} bytes; "
            f"protected: {payload.protected_items} item(s)"
        )
        if payload.operation == "delete_snapshot":
            out.print(
                f"snapshot: {payload.snapshot_items} item(s); locally unrecoverable: "
                f"{payload.unrecoverable_items} item(s), {payload.unrecoverable_bytes} bytes"
            )
            if payload.confirmation:
                out.print(f"confirmation: [bold]{escape(payload.confirmation)}[/bold]")
    else:
        out.print(
            f"removed: {payload.removed_items} item(s); released: {payload.released_bytes} bytes"
        )
        out.print("[dim]source contacted: no[/dim]")


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
    # Not dim, and above `next:` rather than among the notes. `init` seeds the grammars and
    # the vocabularies and leaves the weights, so the command it is about to recommend is the
    # one that spends several minutes downloading a model — and a reader who was not told that
    # reads the pause as a hang and kills it. One line, before the instruction it qualifies.
    if payload.weights_pending:
        out.print(
            "\n[yellow]the embedding model is not on this machine yet[/yellow][dim]: the first "
            "[/dim]index[dim] downloads it before indexing anything. Expect minutes, once.[/dim]"
        )
    # The one thing a person needs after `init`, and the only command that has to come next.
    # Without it the first run ends on a report with nothing to do about it.
    out.print("\n[dim]next: [/dim]manicule index <path>[dim], then[/dim] manicule search <query>")


API_TRANSPORT = "http-api"
"""What the HTTP API records as its transport, mirroring :data:`manicule.api.serve.TRANSPORT`.

Named here rather than imported, because importing it would pull FastAPI into every ``manicule
--help``. :func:`render_address` is the one reader, and ``tests/app/test_serving.py`` asserts
the two strings agree so the mirror cannot drift.
"""


_SIGNPOST_WIDTH: Final = len("API documentation")
"""How wide the labels below the address line are padded, so their addresses share a column.

The longest of them, named rather than counted, because the thing being kept true is that no
label is wider than the padding — a label that overflowed would push its own address out of
line and read as the one that is broken.
"""


def _signpost(out: Console, label: str, target: str) -> None:
    """One "here is where X is" line under the address, in the shared column."""
    out.print(f"[dim]{label.ljust(_SIGNPOST_WIDTH)}[/dim]  {target}")


def render_address(
    out: Console, payload: r.ServerAddress, *, web: bool | None = None, stopped: bool = False
) -> None:
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

    ``stopped`` is the other thing the payload cannot carry: it describes an address, and an
    address reads the same whether a server has just arrived at it or just left it. Without
    this, ``manicule stop`` printed a start banner — "HTTP API on http://127.0.0.1:8765 (this
    machine only)", then the URLs of a browser surface that is no longer there.
    """
    if stopped:
        where = "stdio" if payload.transport == "stdio" else f"http://{payload.host}:{payload.port}"
        what = "the HTTP API" if payload.transport == API_TRANSPORT else "the MCP server"
        out.print(f"stopped {what} [dim]that was on {where}[/dim]")
        return
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
        # `--mcp-only`. The bare address above is not what a client is configured with, and
        # until this line the trailing-slash endpoint appeared in no output at all — so the
        # next thing an operator did was paste an address that answers nothing.
        _signpost(out, "MCP endpoint", f"{where}{frontdoor.MCP_ENDPOINT}")
        out.print(f"[dim]{payload.tools} tool(s)[/dim]")
        return
    # The signposts a person wants next, and none of them were printed before. A server whose
    # browser surface is not named is one nobody finds without reading the source.
    if web is True:
        _signpost(out, "browser surface", f"{where}{frontdoor.UI}")
    elif web is False:
        _signpost(out, "browser surface", "off (--no-web)")
    # MCP has been on this port since #143 and was named in none of this. An operator wiring up
    # a client had the port and had to know the path.
    _signpost(out, "MCP endpoint", f"{where}{frontdoor.MCP_ENDPOINT}")
    _signpost(out, "API documentation", f"{where}{frontdoor.DOCS}")


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


def render_connector_signed_in(out: Console, payload: r.ConnectorSignedIn) -> None:
    """What was captured, where it went, and when it stops working. Never the session itself."""
    if payload.forgotten:
        out.print(f"forgot the stored session for [bold]{escape(payload.name)}[/bold]")
        return
    out.print(
        f"signed in to [bold]{escape(payload.base_url)}[/bold] as "
        f"[bold]{escape(payload.account)}[/bold]"
    )
    out.print(f"  stored in {escape(payload.stored_in)}")
    out.print(f"  manicule will use it until {escape(payload.expires_at)}")


def render_messaging_host_installed(out: Console, payload: r.MessagingHostInstalled) -> None:
    """Which browsers were set up, and the one next step this cannot do for the operator."""
    out.print(f"installed the messaging host for [bold]{len(payload.installed)}[/bold] browser(s)")
    for path in payload.installed:
        out.print(f"  {escape(path)}")
    out.print(f"  only extension [bold]{escape(payload.extension_id)}[/bold] may start it")
    out.print("")
    out.print("")
    out.print("Next, in Chrome:")
    out.print("  1. open [bold]chrome://extensions[/bold]")
    out.print("  2. turn on [bold]Developer mode[/bold] (top right)")
    out.print("  3. click [bold]Load unpacked[/bold] and choose:")
    out.print(f"       {escape(payload.extension_dir)}")
    out.print("  4. open the extension's popup and enter your Confluence URL")


def render_snapshot_status(out: Console, payload: r.SnapshotStatusReport) -> None:
    """Render only aggregate manifest facts; member identities never reach the payload."""
    progress = payload.lifecycle
    out.print(
        f"[bold]{escape(payload.state)}[/bold] {escape(payload.snapshot_id)} "
        f"({progress.acquired_items}/{progress.enumerated_items} acquired)"
    )
    out.print(
        f"verified: {'yes' if payload.verified else 'no'}; "
        f"promoted: {'yes' if progress.snapshot_promoted else 'no'}; "
        f"offline continuation: {'yes' if progress.can_continue_offline else 'no'}"
    )
    if payload.full_inventory_authority:
        out.print(f"full inventory authority: {escape(payload.full_inventory_authority)}")
    if progress.inventory_recovery:
        out.print(
            f"inventory recovery: {escape(progress.inventory_recovery)}; "
            f"disappeared items reconciled: {progress.reconciled_deleted_items}"
        )
    # The line that separates "adapting to a slow source" from "hung": a walk that is
    # shrinking its requests is making progress, and only these counts say so.
    if progress.enumeration_timeout_retries or progress.enumeration_page_size_reduced:
        offset = progress.enumeration_offset
        out.print(
            f"source timeout retries: {progress.enumeration_timeout_retries}; "
            f"requested page size: {progress.enumeration_page_size or '—'}; "
            f"offset: {offset if offset is not None else '—'}"
        )
    if progress.enumeration_reached_empty_page is not None:
        out.print(
            "authoritative inventory reached its explicit end: "
            f"{'yes' if progress.enumeration_reached_empty_page else 'no'}"
        )
    if progress.enumeration_failure_code:
        out.print(f"enumeration stopped by: {escape(progress.enumeration_failure_code)}")


def render_collection(out: Console, payload: r.CollectionSummary) -> None:
    out.print(f"[bold]{escape(payload.name)}[/bold] [dim]{escape(payload.id)}[/dim]")
    if payload.description:
        out.print(escape(payload.description))
    if payload.rule is not None:
        # The rule is named, not evaluated. What it currently selects is what
        # `collection documents` and `collection counts` answer, and a second count computed
        # here would be a cheaper-looking number free to disagree with both.
        out.print(f"[dim]rule: {escape(json.dumps(payload.rule, sort_keys=True))}[/dim]")


def render_collections(out: Console, payload: r.CollectionList) -> None:
    table = Table("name", "id", "rule", "description", box=None, pad_edge=False)
    for collection in payload.collections:
        table.add_row(
            escape(collection.name),
            escape(collection.id),
            "yes" if collection.rule is not None else "—",
            escape(collection.description or "—"),
        )
    out.print(table)
    if not payload.collections:
        out.print("[dim]no collections[/dim]")


def render_collection_deleted(out: Console, payload: r.CollectionDeleted) -> None:
    out.print(f"deleted collection [bold]{escape(payload.collection_id)}[/bold]")
    out.print("[dim]the documents it held are untouched[/dim]")


def render_collection_membership(out: Console, payload: r.CollectionMembership) -> None:
    out.print(
        f"[bold]{payload.changed}[/bold] of {len(payload.document_ids)} "
        f"changed in {escape(payload.collection_id)}"
    )
    out.print("[dim]membership is metadata; nothing was re-embedded[/dim]")


def render_collection_counts(out: Console, payload: r.CollectionCounts) -> None:
    out.print(
        f"[bold]{escape(payload.name)}[/bold]: "
        f"{payload.documents} documents, {payload.chunks} chunks"
    )


def render_collection_orphans(out: Console, payload: r.CollectionOrphans) -> None:
    """Report first, and say plainly which of the two things just happened.

    One payload covers the listing and the cleanup, and the difference between them matters
    more than anything else on it: one moved documents out of the corpus and the other moved
    nothing at all.
    """
    if not payload.count:
        out.print("[dim]no documents outside every collection[/dim]")
        return
    for document_id in payload.document_ids:
        out.print(escape(document_id))
    if payload.deleted:
        out.print(f"[bold]{payload.count}[/bold] moved to the trash, and restorable from it")
    else:
        out.print(
            f"[bold]{payload.count}[/bold] in no collection. "
            f"[dim]nothing was deleted; pass --confirm to move them to the trash[/dim]"
        )


def render_reembed_plan(out: Console, payload: r.ReembedPlanReport) -> None:
    out.print(
        f"[bold]{payload.documents}[/bold] documents, [bold]{payload.chunks}[/bold] chunks; "
        f"about {payload.estimated_seconds:.1f}s"
    )
    out.print(
        f"peak memory {payload.peak_memory_bytes} bytes; temporary disk "
        f"{payload.temporary_disk_bytes} bytes; target {payload.target_identity[:12]} "
        f"({payload.target_dimension} dimensions)"
    )
    if payload.unrepairable_documents:
        out.print(f"[red]{payload.unrepairable_documents} document(s) cannot be rebuilt[/red]")


def render_reembed_run(out: Console, payload: r.ReembedRunReport) -> None:
    out.print(f"[bold]{escape(payload.state)}[/bold] {escape(payload.run_id)}")
    out.print(
        f"{payload.documents_completed}/{payload.documents} documents; "
        f"{payload.chunks_completed}/{payload.chunks} chunks"
    )
    if payload.retry_required:
        out.print("[dim]retry required: use `manicule reembed resume RUN_ID`[/dim]")


def render_reembed_cleanup(out: Console, payload: r.ReembedCleanupReport) -> None:
    action = "removed" if payload.removed else "already absent"
    out.print(f"{action}: {escape(payload.run_id)}")


def render_rebuild_plan(out: Console, payload: r.RebuildPlanReport) -> None:
    outcome = "runnable" if payload.runnable else f"refused: {payload.refusal_code or 'unknown'}"
    out.print(
        f"[bold]{payload.documents}[/bold] documents, about "
        f"[bold]{payload.estimated_chunks}[/bold] chunks; {escape(outcome)}"
    )
    out.print(
        f"about {payload.estimated_seconds:.1f}s; peak memory "
        f"{payload.estimated_peak_memory_bytes} bytes; temporary disk "
        f"{payload.estimated_temporary_bytes} bytes"
    )
    out.print(
        f"estimated embedding work: {payload.estimated_embedding_chunks} chunks; "
        f"missing retained inputs: {payload.missing_count}; "
        f"network required: {'yes' if payload.network_required else 'no'}"
    )
    out.print(
        f"current chunk identity: {escape(payload.current_chunk_fingerprint or 'unrecorded')}"
    )
    out.print(f"target chunk identity: {escape(payload.target_chunk_fingerprint)}")
    if payload.over_budget_chunks:
        out.print(
            f"stored budget diagnostic: {payload.over_budget_chunks} chunk(s) over the "
            f"recorded budget; maximum stored token count {payload.max_stored_chunk_tokens}"
        )


def render_rebuild_run(out: Console, payload: r.RebuildRunReport) -> None:
    out.print(f"[bold]{escape(payload.state)}[/bold] {escape(payload.generation_id)}")
    out.print(
        f"{payload.documents_built}/{payload.expected_items} documents, "
        f"{payload.chunks_built} chunks; "
        f"{payload.vectors_reused} vectors reused, {payload.vectors_embedded} embedded"
    )
    if payload.diagnostic_code:
        out.print(f"[red]diagnostic: {escape(payload.diagnostic_code)}[/red]")


RENDERERS: Mapping[type[Payload], Callable[[Console, Payload], None]] = {
    r.AnswerResultPayload: lambda out, p: render_answer(out, _as(r.AnswerResultPayload, p)),
    r.SearchResult: lambda out, p: render_search(out, _as(r.SearchResult, p)),
    r.DocumentList: lambda out, p: render_document_list(out, _as(r.DocumentList, p)),
    r.DocumentDetail: lambda out, p: render_document(out, _as(r.DocumentDetail, p)),
    r.DocumentDeleted: lambda out, p: render_document_deleted(out, _as(r.DocumentDeleted, p)),
    r.DocumentReindexed: lambda out, p: render_document_reindexed(out, _as(r.DocumentReindexed, p)),
    r.StaleReparseReport: lambda out, p: render_stale_reparse(out, _as(r.StaleReparseReport, p)),
    r.StaleGlossaryReport: lambda out, p: render_stale_glossary(out, _as(r.StaleGlossaryReport, p)),
    r.ReembedPlanReport: lambda out, p: render_reembed_plan(out, _as(r.ReembedPlanReport, p)),
    r.ReembedRunReport: lambda out, p: render_reembed_run(out, _as(r.ReembedRunReport, p)),
    r.ReembedCleanupReport: lambda out, p: render_reembed_cleanup(
        out, _as(r.ReembedCleanupReport, p)
    ),
    r.RebuildPlanReport: lambda out, p: render_rebuild_plan(out, _as(r.RebuildPlanReport, p)),
    r.RebuildRunReport: lambda out, p: render_rebuild_run(out, _as(r.RebuildRunReport, p)),
    r.IngestReport: lambda out, p: render_ingest(out, _as(r.IngestReport, p)),
    r.SnapshotStatusReport: lambda out, p: render_snapshot_status(
        out, _as(r.SnapshotStatusReport, p)
    ),
    r.IndexStatus: lambda out, p: render_index_status(out, _as(r.IndexStatus, p)),
    r.VectorIndexReport: lambda out, p: render_vector_index(out, _as(r.VectorIndexReport, p)),
    r.VectorSweepReport: lambda out, p: render_vector_sweep(out, _as(r.VectorSweepReport, p)),
    r.Stats: lambda out, p: render_stats(out, _as(r.Stats, p)),
    r.Diagnosis: lambda out, p: render_diagnosis(out, _as(r.Diagnosis, p)),
    r.ConnectorList: lambda out, p: render_connectors(out, _as(r.ConnectorList, p)),
    r.MessagingHostInstalled: lambda out, p: render_messaging_host_installed(
        out, _as(r.MessagingHostInstalled, p)
    ),
    r.ConnectorSignedIn: lambda out, p: render_connector_signed_in(
        out, _as(r.ConnectorSignedIn, p)
    ),
    r.SidecarReport: lambda out, p: render_sidecar(out, _as(r.SidecarReport, p)),
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
    r.LifecycleReport: lambda out, p: render_lifecycle(out, _as(r.LifecycleReport, p)),
    r.InitReport: lambda out, p: render_init(out, _as(r.InitReport, p)),
    r.ServerAddress: lambda out, p: render_address(out, _as(r.ServerAddress, p)),
    r.UpgradeReport: lambda out, p: render_upgrade(out, _as(r.UpgradeReport, p)),
    r.CompletionScript: lambda out, p: render_completion(out, _as(r.CompletionScript, p)),
    r.ApiKeyIssued: lambda out, p: render_api_key_issued(out, _as(r.ApiKeyIssued, p)),
    r.ApiKeyList: lambda out, p: render_api_keys(out, _as(r.ApiKeyList, p)),
    r.ApiKeyRevoked: lambda out, p: render_api_key_revoked(out, _as(r.ApiKeyRevoked, p)),
    r.CollectionSummary: lambda out, p: render_collection(out, _as(r.CollectionSummary, p)),
    r.CollectionList: lambda out, p: render_collections(out, _as(r.CollectionList, p)),
    r.CollectionDeleted: lambda out, p: render_collection_deleted(out, _as(r.CollectionDeleted, p)),
    r.CollectionMembership: lambda out, p: render_collection_membership(
        out, _as(r.CollectionMembership, p)
    ),
    r.CollectionCounts: lambda out, p: render_collection_counts(out, _as(r.CollectionCounts, p)),
    r.CollectionOrphans: lambda out, p: render_collection_orphans(out, _as(r.CollectionOrphans, p)),
    r.VectorChecksumReport: lambda out, p: render_vector_checksum(
        out, _as(r.VectorChecksumReport, p)
    ),
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
    summarize = _ANCHOR_SUMMARIES.get(str(kind)) if kind is not None else None
    if summarize is not None:
        return summarize(anchor)
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
