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
from manicule.connectors.errors import SessionMissingError
from manicule.core.errors import ManiculeError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from pydantic import JsonValue

    from manicule.app.service import ApplicationService
    from manicule.connectors.sessions import SessionVault

SESSIONS_HELD = "sessions_held"
"""The ``op`` a :class:`~manicule.app.control.Held` reply carries.

Named rather than written at the one place it is used, because the *other* end reads it too:
anything keying off an envelope's ``op`` — a log, a shell pipeline, a future client — joins on
this string, and a literal is how the two ends come to disagree about one word.

Deliberately not ``connector_login``, which the hand-off and the forget replies carry. Those two
are the server side of that command; this is a question about the process that no command emits,
so borrowing the name would file every one of these under an operation nobody ran.
"""

__all__ = [
    "SESSIONS_HELD",
    "ControlHandler",
    "ScheduledSource",
    "Scheduler",
    "Serving",
    "announce",
]

_LIFECYCLE_PLAN_OPS = frozenset(
    {
        "lifecycle_reset_derived",
        "lifecycle_cleanup_generations",
        "lifecycle_release_history",
        "lifecycle_delete_snapshot",
    }
)


class ControlHandler:
    """Runs what arrives on the control socket, and nothing else.

    Four request kinds, four things it can do, and the list is closed:

    ``Invoke``
        Run one operation through :func:`~manicule.app.commands.run`, which is the same function
        the command line calls when it runs one in-process. That shared call is what makes a
        proxied command faithful rather than similar.
    ``Handover``
        Take a session the command line captured and hold it. The only way a credential enters
        this process.
    ``Forget``
        Drop one.
    ``Held``
        Say which instances it is holding a session for, and for whose account. The answer
        carries no session value — see :class:`~manicule.app.control.Held`.

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
        if isinstance(request, control.Held):
            return self._held()
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

    def _held(self) -> dict[str, JsonValue]:
        """Which instances this process holds a session for, and for whose account.

        Synchronous, and the only handler that is: it reads a dictionary. Declared ``def`` rather
        than ``async def`` returning immediately, because a coroutine that never awaits is a
        claim about needing the event loop that this does not make.

        The reply is a list of two-field objects rather than a mapping, so a base URL containing
        anything awkward is a value rather than a key — and so the shape has somewhere to grow if
        a capture time is ever wanted here, which a mapping would not.

        **The envelope's ``op`` is** :data:`SESSIONS_HELD` **rather than ``connector_login``**,
        which the other two session frames use. Those two are the server side of that command —
        capturing and forgetting are what ``connector login`` does — and this is not: it is a
        question ``doctor`` asks, run by nobody's command. An envelope naming the operation
        wrongly is a log that cannot be read six months later, and ``op`` is the field a reader
        joins on.
        """
        return _succeeded(
            SESSIONS_HELD,
            self._service.workspace,
            {
                "held": [
                    {"base_url": base_url, "account": account}
                    for base_url, account in sorted(self._vault.holding().items())
                ]
            },
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
    log. ``failures`` counts both refusals and returned incomplete outcomes; nothing ends a loop
    short of the server stopping, so either kind is retried on the next interval.
    """

    name: str
    interval_s: float
    runs: int = 0
    failures: int = 0
    awaiting_sign_in: bool = False
    """Whether the last run failed because nobody has signed in to this source's instance.

    A field rather than a fourth counter, because it is a *state* and not an event: it is true
    from the first failure until a run succeeds, which is exactly how long somebody needs to open
    a browser. A count would say "it failed nine times" where what an operator needs is "it is
    still waiting".

    Separate from ``failures`` rather than replacing it, because the run did fail and the
    counters should agree with each other. What this adds is *which kind* — a restart lost the
    session and a person has to sign in, as against an instance that was unreachable and will
    probably be reachable at twenty past. Both are refusals; only one of them needs anybody.

    Cleared by any returned ingest report, because reaching the pipeline proves session
    acquisition succeeded even when a later cursor or store failure makes that run incomplete.
    """

    last_outcome: str = ""
    retry_required: bool = False
    last_error_type: str = ""


@dataclass(slots=True)
class ScheduledReembedRecovery:
    """Aggregate-safe outcome of restart recovery for durable re-embedding runs."""

    recovered: int = 0
    failures: int = 0
    complete: bool = False
    last_error_type: str = ""


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

    **One refusal is reported as itself, and it is the one a restart causes.** A session lives in
    this process's memory, so launchd restarting the server — a crash, a logout, a reboot — ends
    every one of them and the next scheduled sync cannot authenticate. Reported like any other
    failure, that reads at three in the morning exactly like an instance being down — and the two
    need opposite things, one a person at a browser and the other nothing at all. So
    :class:`~manicule.connectors.errors.SessionMissingError` is matched on its type, announced in
    its own sentence naming the command that fixes it, and recorded as
    :attr:`ScheduledSource.awaiting_sign_in` so that the state is inspectable rather than only
    printed.
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
        self.reembedding = ScheduledReembedRecovery()

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
        recovery = asyncio.create_task(self._recover_reembedding(), name="recover:reembedding")
        self._tasks.add(recovery)
        recovery.add_done_callback(self._tasks.discard)
        for name, interval in self._sources.items():
            task = asyncio.create_task(self._run(name, interval), name=f"schedule:{name}")
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def plan_lifecycle(self, command: commands.Command) -> dict[str, JsonValue]:
        """Run one scheduler-owned lifecycle dry run through the canonical dispatcher.

        Retention schedulers may inspect aggregate impact, but they receive no destructive
        authority: the command must be one of the four lifecycle operations and must explicitly
        carry ``dry_run=true``.  The returned envelope is consequently identical to a proxied
        CLI invocation of the same plan.
        """
        if command.op not in _LIFECYCLE_PLAN_OPS or command.writes():
            raise ValueError("the scheduler may run lifecycle aggregate dry runs only")
        envelope = await run_op(
            command.op,
            self._service.workspace,
            lambda: commands.run(self._service, command, lambda _message: None),
        )
        return envelope.as_json()

    async def _recover_reembedding(self) -> None:
        """Resume ownerless durable runs once at startup; run status remains authoritative."""
        try:
            outcome = await self._service.reembed_recover_pending()
        except asyncio.CancelledError:
            raise
        except (ManiculeError, ValueError, OSError) as exc:
            self.reembedding.failures += 1
            self.reembedding.last_error_type = type(exc).__name__
            announce(
                "durable re-embedding restart recovery failed; inspect the saved run status "
                f"and retry ({type(exc).__name__})"
            )
        else:
            self.reembedding.recovered += outcome.recovered
            self.reembedding.failures += outcome.failures
            self.reembedding.last_error_type = ",".join(outcome.failure_types)
            self.reembedding.complete = True

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
                report = await self._service.connector_sync(name)
            except asyncio.CancelledError:
                raise
            except SessionMissingError as exc:
                # Ahead of the general clause, and it is a `ConfigError` so the general clause
                # would otherwise swallow it. Nothing has been asked of the instance and nothing
                # is wrong with it: nobody has signed in since this process started, which after
                # a restart is the ordinary state. It is announced as itself so that the log line
                # an operator finds says which of the two refusals it was.
                record.failures += 1
                record.awaiting_sign_in = True
                announce(
                    f"scheduled sync of {name!r} did not run: this server holds no Confluence "
                    f"session, so it cannot authenticate. Sessions live in the server's memory "
                    f"and a restart ends them, so this is expected after one rather than a "
                    f"fault, and the instance itself has not been contacted. Sign in again with "
                    f"`manicule connector login {name} --browser`; the schedule keeps trying "
                    f"every {interval_s:g}s until you do. Details: {exc}"
                )
            except (ManiculeError, ValueError, OSError) as exc:
                # The outcomes a caller could act on, which here means the operator reading the
                # server's output. Recorded and carried on, because a source that was
                # unreachable at ten past may be reachable at twenty past, and a session that
                # expired needs a person rather than a stopped loop.
                record.failures += 1
                announce(f"scheduled sync of {name!r} failed: {exc}")
            else:
                # A returned report proves this source got past session acquisition. Keeping a
                # prior missing-session flag raised would send the operator to sign in for a
                # later cursor or store failure that needs a retry instead.
                record.awaiting_sign_in = False
                record.last_outcome = report.outcome
                record.retry_required = report.retry_required
                record.last_error_type = (
                    report.incomplete_reason.type if report.incomplete_reason is not None else ""
                )
                if report.retry_required:
                    record.failures += 1
                    detail = (
                        report.incomplete_reason.message
                        if report.incomplete_reason is not None
                        else report.error
                    )
                    announce(
                        f"scheduled sync of {name!r} was incomplete and will be retried: "
                        f"{record.last_error_type or 'IncompleteIngestError'}: "
                        f"{detail}"
                    )
                else:
                    record.runs += 1


def announce(message: str) -> None:
    """Say something on the server's own stderr.

    stderr rather than stdout because a ``stdio`` transport serves the protocol on stdout, and a
    line of prose there is a corrupt message — the same reason
    :func:`~manicule.cli.serving._serve` announces its address there.

    Public because the shutdown says what it is doing through it as well as the scheduler, and
    those two are the whole of what a served manicule tells an operator without being asked. A
    second way of writing a line to stderr would be a second thing to get the stream wrong in.
    """
    sys.stderr.write(f"{message}\n")
    sys.stderr.flush()


@dataclass(slots=True)
class Serving:
    """The control socket and the scheduler, started and stopped together, in that order.

    A small object rather than two variables in :func:`~manicule.cli.serving._serve`, because
    the thing that must not be got wrong is that both are closed on every path out — including
    the one where the transport raised — and one ``async with`` is harder to get wrong than two
    ``finally`` blocks.

    **The closing order is the first half of the server's shutdown and is not arbitrary.**

    1. The scheduler, so that no new sync starts. Canceling its loops cancels whatever sync each
       was inside, and a canceled run drains its ingest stages within ``ingest.shutdown_grace_s``
       (#138) rather than abandoning a document mid-write.
    2. The control socket, whose ``aclose`` waits for the write commands already in flight. It is
       second because a proxied sync that is still running is work an operator asked for and is
       watching, and because closing it first would leave those commands with nowhere to answer.

    The second half — the MCP sessions and the HTTP server — belongs to whatever is serving the
    transport and happens after this returns, which is why it is not here. See
    :func:`~manicule.cli.serving._serve`.
    """

    server: control.ControlServer
    scheduler: Scheduler
    _closed: bool = False
    """Whether :meth:`aclose` has run. Set once, checked once, and it is not a nicety.

    The shutdown sequence closes this explicitly — it has to, because the transport is closed
    *after* it and a context manager's exit runs last — and then the ``async with`` around it
    exits and closes it again. Both calls are harmless to the objects underneath, and the second
    one's *announcements* are not: an operator reading "stopping the scheduler" twice has every
    reason to think something restarted.
    """

    async def astart(self) -> None:
        """Bind the socket, then begin the schedule.

        Socket first, so that the moment anything is syncing there is somewhere for
        ``manicule stop`` and a proxied command to reach. The reverse order leaves a window in
        which a sync is running and the process cannot be talked to.
        """
        await self.server.start()
        self.scheduler.start()

    async def aclose(self) -> None:
        """Stop the schedule, then the socket. Both, on every path, in that order.

        The scheduler's ``aclose`` is wrapped because a loop that raised while being canceled
        must not stop the socket being closed: the socket is a *file*, and one left behind with
        no server answering is what the next start has to clear.

        **Each step says it is starting**, on the server's own stderr. A stop can take as long as
        ``ingest.shutdown_grace_s`` plus whatever a proxied command is still doing, which is long
        enough that an operator watching a terminal needs to know it is working rather than hung
        — and it is exactly the interval in which somebody reaches for ``kill -9`` and gets the
        half-written index :data:`~manicule.app.daemon.STOP_GRACE_S` refuses to produce.
        """
        if self._closed:
            return
        self._closed = True
        announce("stopping the scheduler, so no further sync starts")
        with contextlib.suppress(Exception):
            await self.scheduler.aclose()
        announce("closing the control socket, waiting for any write command still running")
        await self.server.aclose()

    async def __aenter__(self) -> Serving:
        await self.astart()
        return self

    async def __aexit__(self, *exc: object) -> None:
        del exc
        await self.aclose()
