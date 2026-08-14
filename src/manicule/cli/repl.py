"""The interactive prompt.

One runtime for the whole session, which is the point: the first question pays for the model
runtime and the rest do not. A loop that opened a runtime per question would make every
question the first one.

Each question is answered on its own. Multi-turn memory needs a conversation record, and
``ask --conversation ID`` is how a caller that has one continues it — the prompt does not
create conversations, because a session that silently accumulated one would be persisting
somebody's questions without being asked to.
"""

from __future__ import annotations

import asyncio
from functools import partial
from typing import TYPE_CHECKING

from manicule.app.dispatch import run_op
from manicule.app.runtime import Runtime
from manicule.app.service import ApplicationService
from manicule.cli import render
from manicule.core.errors import ManiculeError
from manicule.generation.answers import EventKind

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from manicule.app.results import AnswerResultPayload
    from manicule.generation.answers import AnswerEvent

BANNER = """\
manicule — ask a question, or type one of:
  :quit        leave
  :profile P   switch retrieval profile (fast, balanced, precise)
  :sources A,B restrict to these sources; empty clears the restriction
Each question is answered on its own. Pass --conversation to continue a stored one.
"""


def run_repl(
    *,
    profile: str | None = None,
    limit: int | None = None,
    sources: Sequence[str] = (),
    overrides: Mapping[str, Any] | None = None,
) -> int:
    """Run the prompt until it is left. Returns the process's exit status."""
    try:
        return asyncio.run(
            _loop(
                profile=profile,
                limit=limit,
                sources=tuple(sources),
                overrides=dict(overrides or {}),
            )
        )
    except KeyboardInterrupt:  # pragma: no cover - a person pressing ^C
        return 130


async def _loop(
    *,
    profile: str | None,
    limit: int | None,
    sources: tuple[str, ...],
    overrides: dict[str, Any],
) -> int:
    out = render.console()

    def show(event: AnswerEvent) -> None:
        if event.kind is EventKind.DELTA and event.text:
            out.file.write(event.text)
            out.file.flush()

    try:
        # A REPL can run any operation, including every writer, so it holds the directory
        # for its whole session. Taken here so the refusal is the rendered error below
        # rather than a traceback out of the loop.
        runtime = Runtime.open(**overrides)
        runtime.acquire()
    except (ManiculeError, ValueError, OSError) as exc:
        from manicule.app.dispatch import error_info  # noqa: PLC0415 - only the failure path

        render.render_error(render.console(stderr=True), "ask", error_info(exc))
        return 1

    async with runtime:
        service = ApplicationService(runtime)
        out.print(BANNER)
        out.print(f"[dim]workspace {service.workspace}[/dim]")
        while True:
            try:
                # Off the event loop: `input` blocks, and the runtime this session
                # holds has background work — a shielded persist, a pool's teardown —
                # that must keep running while somebody is typing.
                typed = await asyncio.to_thread(input, "manicule> ")
            except EOFError:
                out.print()
                return 0
            line = typed.strip()
            if not line:
                continue
            if line in {":quit", ":q", ":exit"}:
                return 0
            if line.startswith(":profile"):
                profile = line.removeprefix(":profile").strip() or None
                out.print(f"[dim]profile: {profile or 'configured default'}[/dim]")
                continue
            if line.startswith(":sources"):
                listed = line.removeprefix(":sources").strip()
                sources = tuple(part.strip() for part in listed.split(",") if part.strip())
                out.print(f"[dim]sources: {', '.join(sources) or 'all'}[/dim]")
                continue

            # `partial` rather than a lambda, because the arguments are bound **now**. A
            # closure over the loop variables would read whatever they held when the call
            # eventually ran, which is a different question than the one that was typed.
            envelope = await run_op(
                "ask",
                service.workspace,
                partial(
                    service.ask,
                    line,
                    profile=profile,
                    limit=limit,
                    sources=sources,
                    on_event=show,
                ),
            )
            out.print()
            if envelope.ok and envelope.data is not None:
                from manicule.app.results import AnswerResultPayload as Answer  # noqa: PLC0415

                payload: AnswerResultPayload = Answer.model_validate(envelope.data)
                # The tokens have already been written by `show`, so only the
                # citations and the facts about the run are printed here.
                render.render_answer(out, payload, text_already_shown=True)
            elif envelope.error is not None:
                render.render_error(render.console(stderr=True), "ask", envelope.error)


__all__ = ["BANNER", "run_repl"]
