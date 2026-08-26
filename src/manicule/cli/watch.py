"""``manicule index --watch``: keep a directory indexed as it changes.

The loop is here and the decisions are not. What a changed path *means* — which document it
is, whether a deletion should remove one — lives in
:meth:`~manicule.app.service.ApplicationService.index_changes`, because a document's identity
is derived from ``(workspace, source, source_id)`` and an identity computed in a surface is an
identity that can be computed differently in the next surface.

Debouncing, editor scratch files and the re-``stat`` after the window all belong to
:mod:`manicule.ingest.watch`, which already knows that one logical save produces several
events and that ingesting on the first indexes a half-written file.
"""

from __future__ import annotations

import asyncio
import json
import sys
from functools import partial
from typing import TYPE_CHECKING, Any

from manicule.app.dispatch import error_info, run_op
from manicule.app.runtime import Runtime
from manicule.app.service import ApplicationService
from manicule.cli import render
from manicule.core.errors import ManiculeError

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from manicule.app.dispatch import Envelope


def watch_path(
    path: Path,
    *,
    source: str,
    reindex: bool = False,
    json_output: bool = False,
    overrides: Mapping[str, Any] | None = None,
) -> int:
    """Index ``path``, then keep it indexed. Returns the process's exit status.

    ``json_output`` is carried because ``index --watch`` accepts ``--json`` and this ignored
    it: every batch was rendered as Rich tables to stdout, so the one long-running indexing
    command was the one place ``--json`` did not mean what ``docs/surfaces.md`` says it means.
    A program watching a directory and parsing the stream got decorated text.
    """
    try:
        return asyncio.run(
            _watch(
                path,
                source=source,
                reindex=reindex,
                json_output=json_output,
                overrides=dict(overrides or {}),
            )
        )
    except KeyboardInterrupt:  # pragma: no cover - a person pressing ^C
        return 130


async def _watch(
    path: Path,
    *,
    source: str,
    reindex: bool,
    json_output: bool,
    overrides: dict[str, Any],
) -> int:
    from manicule.ingest.watch import Change, watch_directory  # noqa: PLC0415 - optional extra

    # Under `--json` stdout carries envelopes and nothing else, so the human console is moved
    # to stderr rather than silenced: the "watching …" line and any progress still reach a
    # person, and a consumer parsing stdout sees one envelope per line and no decoration.
    out = render.console(stderr=json_output)

    def emit_envelope(envelope: Envelope) -> None:
        """One envelope on stdout, serialized exactly as `print_envelope` serializes one."""
        sys.stdout.write(json.dumps(envelope.as_json(), indent=2, sort_keys=True) + "\n")
        sys.stdout.flush()

    try:
        # Watch mode indexes whatever changes, indefinitely. It is a writer for its whole
        # life, and the refusal belongs on the way in rather than at the first change.
        runtime = Runtime.open(**overrides)
        runtime.acquire()
    except (ManiculeError, ValueError, OSError) as exc:
        render.render_error(render.console(stderr=True), "index_path", error_info(exc))
        return 1

    async with runtime:
        service = ApplicationService(runtime)
        first = await run_op(
            "index_path",
            service.workspace,
            lambda: service.index_path(path, source=source, force=reindex),
        )
        if not first.ok:
            if first.error is not None:
                render.render_error(render.console(stderr=True), "index_path", first.error)
            return 1
        if json_output:
            emit_envelope(first)
        elif first.data is not None:
            from manicule.app.results import IngestReport  # noqa: PLC0415

            render.render_ingest(out, IngestReport.model_validate(first.data))
        out.print(f"[dim]watching {path} — press ^C to stop[/dim]")

        try:
            async for batch in watch_directory(
                path, debounce_s=runtime.settings.ingest.watch_debounce_s
            ):
                changed = [event.path for event in batch if event.change is not Change.DELETED]
                removed = [event.path for event in batch if event.change is Change.DELETED]
                if not changed and not removed:
                    continue
                # `partial`, so the batch is bound now rather than read from the loop
                # variable whenever the call happens to run.
                envelope = await run_op(
                    "index_changes",
                    service.workspace,
                    partial(service.index_changes, changed, source=source, removed=removed),
                )
                if json_output:
                    # A failed batch is an envelope too, and it goes to stdout with the rest:
                    # the stream is the record of what the watch did, so a consumer must see
                    # the failure in the same place and the same shape as the successes.
                    emit_envelope(envelope)
                elif envelope.ok and envelope.data is not None:
                    from manicule.app.results import IngestReport  # noqa: PLC0415

                    render.render_ingest(out, IngestReport.model_validate(envelope.data))
                elif envelope.error is not None:
                    # A batch that failed is reported and the watch continues. A watcher that
                    # exited on one bad file would stop indexing everything else, which is the
                    # same "one bad document must never stop the rest" rule the pipeline holds.
                    render.render_error(
                        render.console(stderr=True), "index_changes", envelope.error
                    )
        except RuntimeError as exc:
            # `watch_directory` raises this when its optional dependency is absent, named as a
            # missing extra rather than as an ImportError from a module nobody mentioned.
            render.render_error(render.console(stderr=True), "index_path", error_info(exc))
            return 1
    return 0


__all__ = ["watch_path"]
