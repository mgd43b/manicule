"""What a served manicule does that a command-line one does not.

Two things, and they are here together because they are the same claim from two directions: this
process is the writer. It answers write commands that arrive on the control socket
(:class:`ControlHandler`), and it runs the syncs configuration asked for without anybody typing
anything (:class:`Scheduler`).

**The scheduler is internal, and there is no route that starts a sync.** That boundary is the
one thing about this module worth reading twice. ``tests/api/test_routes.py`` asserts *by name*
that the destructive operations have no HTTP route, and #113 refused one for corpus-wide reparse
because an unattended caller could hold the accelerator for an hour. A server that syncs on its
own configuration does not cross that line — the schedule is a setting an operator wrote, read
by the process they started. A route that let a caller start a sync would cross it, and there is
none. The control socket is not that route either: it is unreachable over a network, and what it
accepts is a fixed list somebody wrote down (:data:`~manicule.app.commands.BINDERS`).

**A scheduled sync and a proxied one run concurrently, deliberately.** They share one
:class:`~manicule.ingest.pipeline.IngestPipeline`, which is exactly the interleaving the
compare-and-swap in #119, the keyed per-document lock, the embedding partition in #120 and the
glossary lineage write in #122 were built for. Nothing here adds a lock of its own to keep them
apart, because a lock here would be a second answer to a question those four already answer —
and the one that got used would be whichever was reached first.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from manicule.app import commands, control
from manicule.app.dispatch import run_op
from manicule.connectors.credentials import BrowserSession
from manicule.core.errors import ManiculeError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from pydantic import JsonValue

    from manicule.app.service import ApplicationService
    from manicule.connectors.sessions import SessionVault

__all__ = ["ControlHandler", "ScheduledSource", "Scheduler", "Serving"]


class ControlHandler:
    """Runs what arrives on the control socket, and nothing else.

    Three request kinds, three things it can do, and the list is closed:

    ``Invoke``
        Run one operation through :func:`~manicule.app.commands.run`, which is the same function
        the command line calls when it runs one in-process. That shared call is what makes a
        proxied command faithful rather than similar.
    ``Handover``
        Take a session the command line captured and hold it. The only way a credential enters
        this process.
    ``Forget``
        Drop one.

    **It returns envelopes and does not raise for anything a caller could act on**, because the
    thing on the other end of the socket is a person's terminal and a traceback is not a result.
    :func:`~manicule.app.dispatch.run_op` makes that decision once for every surface, and this
    uses it rather than repeating it.
    """

    def __init__(self, service: ApplicationService, vault: SessionVault) -> None:
        self._service = service
        self._vault = vault

    async def handle(
        self, request: control.Request, report: Callable[[str], None]
    ) -> dict[str, JsonValue]:
        if isinstance(request, control.Invoke):
            return await self._invoke(request, report)
        if isinstance(request, control.Handover):
            return await self._accept(request)
        return await self._forget(request)

    async def _invoke(
        self, request: control.Invoke, report: Callable[[str], None]
    ) -> dict[str, JsonValue]:
        """Run one operation on behalf of a command line that cannot take the lock.

        ``workspace`` on the frame is ignored when it is empty, which is what a caller that did
        not pass ``--workspace`` sends. A server serves the workspace it was started for.
        """
        service = self._service
        envelope = await run_op(
            request.op,
            service.workspace,
            lambda: commands.run(service, commands.Command(request.op, request.arguments), report),
        )
        return envelope.as_json()

    async def _accept(self, request: control.Handover) -> dict[str, JsonValue]:
        """Hold a session the command line proved, and say so without saying what it is.

        The reply names the instance and the account and carries no cookie, no length and no
        digest of one. There is nothing a caller needs from the value it just sent, and a reply
        that echoed any part of it would put the credential into a second place — a terminal's
        scrollback — for no purpose at all.
        """
        from datetime import UTC, datetime  # noqa: PLC0415 - only this method parses a stamp

        try:
            captured_at = datetime.fromisoformat(request.captured_at)
        except ValueError as exc:
            return _failed(
                "connector_login",
                self._service.workspace,
                f"the session handed over for {request.base_url} carried a capture time this "
                f"version cannot read ({exc}). Nothing has been stored.",
            )
        if captured_at.tzinfo is None:
            # A frame carrying a time with no offset is read as UTC rather than refused, which
            # is what `BrowserSession.from_json` did for the record this replaces. It matters
            # more here than it did there: `session_max_age_hours` compares this against an
            # aware `now()`, so a naive value is a `TypeError` out of `authorize()` on the first
            # request of the next sync — a crash, at the far end of the system, for a field
            # nobody would think to look at.
            captured_at = captured_at.replace(tzinfo=UTC)
        if not request.cookies:
            return _failed(
                "connector_login",
                self._service.workspace,
                f"the session handed over for {request.base_url} carried no cookies, and a "
                f"session with no cookies authenticates as nobody. Nothing has been stored.",
            )
        await self._vault.save(
            BrowserSession(
                base_url=request.base_url,
                account=request.account,
                captured_at=captured_at,
                cookies=dict(request.cookies),
            )
        )
        return _succeeded(
            "connector_login",
            self._service.workspace,
            {"base_url": request.base_url, "account": request.account, "held": True},
        )

    async def _forget(self, request: control.Forget) -> dict[str, JsonValue]:
        forgotten = await self._vault.forget(request.base_url)
        return _succeeded(
            "connector_login",
            self._service.workspace,
            {"base_url": request.base_url, "forgotten": forgotten},
        )


def _succeeded(op: str, workspace: str, data: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """A hand-off result, in the envelope shape every surface uses.

    Assembled here rather than through :func:`~manicule.app.results.succeeded` because that one
    takes a :class:`~manicule.app.results.Payload`, and a hand-off produces no payload the
    contract knows: it is an acknowledgment between two manicule processes rather than an
    operation's result. Giving it a payload model would put it in ``--json``'s vocabulary, where
    nothing would ever read it.
    """
    from manicule.app.results import CONTRACT_VERSION  # noqa: PLC0415 - avoids an import cycle

    return {
        "version": CONTRACT_VERSION,
        "op": op,
        "ok": True,
        "workspace": workspace,
        "data": data,
        "error": None,
    }


def _failed(op: str, workspace: str, message: str) -> dict[str, JsonValue]:
    from manicule.app.results import CONTRACT_VERSION  # noqa: PLC0415 - avoids an import cycle

    return {
        "version": CONTRACT_VERSION,
        "op": op,
        "ok": False,
        "workspace": workspace,
        "data": None,
        "error": {"type": "ConfigError", "message": message, "hint": ""},
    }


@dataclass(slots=True)
class ScheduledSource:
    """One source's schedule, and what its last automatic run did.

    Kept so that the server can be asked what its scheduler has been doing without reading a
    log. ``failures`` counts refusals the sync reported — a disabled source, an expired session —
    and nothing ends a loop short of the server stopping, which is why there is no third counter
    for that.
    """

    name: str
    interval_s: float
    runs: int = 0
    failures: int = 0


class Scheduler:
    """Syncs that happen because configuration said so.

    One task per scheduled source, each an ordinary loop: wait the interval, sync, repeat.

    **One task per source rather than one loop over all of them**, because a shared loop makes
    every source's cadence a function of every other source's slowest sync. Two sources at ten
    minutes and one at an hour is three independent statements, and three tasks is how they stay
    independent.

    **A source never overlaps itself.** The loop awaits its own sync before waiting again, so a
    source whose sync takes longer than its interval runs back to back rather than piling up.
    That is the honest reading of "every ten minutes" for a job that takes twenty: the machine
    cannot do more, and starting a second one would only make the first slower.

    **The first run is one interval after startup.** A server that swept everything at startup
    would make restarting it — which is how a session is re-taken, so something an operator does
    deliberately and often — into a full sync of every scheduled source.

    **A sync that fails does not stop the schedule.** A refusal is recorded and the loop waits
    and tries again, because the commonest failures here are transient or fixable elsewhere: an
    instance that was down, a session that expired and needs a person. A loop that exited on the
    first of those would need a restart to resume and would give no sign it had stopped.
    """

    def __init__(
        self,
        service: ApplicationService,
        sources: Mapping[str, float],
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._service = service
        self._sources = dict(sources)
        self._sleep = sleep
        self._tasks: set[asyncio.Task[None]] = set()
        self.scheduled: dict[str, ScheduledSource] = {
            name: ScheduledSource(name=name, interval_s=interval)
            for name, interval in self._sources.items()
        }

    @staticmethod
    def configure(service: ApplicationService) -> dict[str, float]:
        """Which sources this installation schedules, and how often.

        A source with no ``schedule_s`` is not scheduled, and a source configured
        ``enabled = false`` is not scheduled whatever its ``schedule_s`` says. The second is the
        one worth being explicit about: a schedule is exactly where a disabled source would come
        back to life without anybody typing anything, and ``connector sync`` refuses a disabled
        source loudly — a scheduler that ran one anyway would make the switch a lie in the one
        place nobody is watching.
        """
        return {
            name: configured.schedule_s
            for name, configured in service.settings.connectors.items()
            if configured.schedule_s is not None and configured.enabled
        }

    def start(self) -> None:
        """Begin every source's loop. Returns immediately; the loops run until :meth:`aclose`."""
        for name, interval in self._sources.items():
            task = asyncio.create_task(self._run(name, interval), name=f"schedule:{name}")
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def aclose(self) -> None:
        """Stop every loop and wait for it.

        Canceling is safe at any point in the loop: a sync interrupted mid-run leaves documents
        in an in-flight status that the recovery sweep finishes, and does not advance a
        watermark — which is the same guarantee ``Ctrl-C`` on a typed sync has always had.
        """
        for task in tuple(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)

    async def _run(self, name: str, interval_s: float) -> None:
        """One source's loop: wait, sync, record, repeat."""
        record = self.scheduled[name]
        while True:
            await self._sleep(interval_s)
            try:
                await self._service.connector_sync(name)
            except asyncio.CancelledError:
                raise
            except (ManiculeError, ValueError, OSError) as exc:
                # The outcomes a caller could act on, which here means the operator reading the
                # server's output. Recorded and carried on, because a source that was
                # unreachable at ten past may be reachable at twenty past, and a session that
                # expired needs a person rather than a stopped loop.
                record.failures += 1
                _announce(f"scheduled sync of {name!r} failed: {exc}")
            else:
                record.runs += 1


def _announce(message: str) -> None:
    """Say something on the server's own stderr.

    stderr rather than stdout because a ``stdio`` transport serves the protocol on stdout, and a
    line of prose there is a corrupt message — the same reason
    :func:`~manicule.cli.serving._serve` announces its address there.
    """
    sys.stderr.write(f"{message}\n")
    sys.stderr.flush()


@dataclass(slots=True)
class Serving:
    """The control socket and the scheduler, started and stopped together.

    A small object rather than two variables in :func:`~manicule.cli.serving._serve`, because
    the thing that must not be got wrong is that both are closed on every path out — including
    the one where the transport raised — and one ``async with`` is harder to get wrong than two
    ``finally`` blocks.
    """

    server: control.ControlServer
    scheduler: Scheduler

    async def __aenter__(self) -> Serving:
        await self.server.start()
        self.scheduler.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        del exc
        with contextlib.suppress(Exception):
            await self.scheduler.aclose()
        await self.server.aclose()
