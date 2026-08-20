"""Syncs that happen because configuration said so, and the ones that must not.

**No test here sleeps for a schedule to come round.** A scheduler test written with real time
either takes as long as the interval or shortens the interval until it is racing the machine,
and both pass on an idle laptop and fail on a loaded runner. The clock is a parameter
(:class:`Clock` below), so a tick is something a test *causes* and then waits for, and every
assertion is about an arrival rather than an elapsed time.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from manicule.app.served import Scheduler
from manicule.app.service import ApplicationService
from manicule.config.settings import ConnectorSettings
from manicule.connectors.errors import ConnectorError, SessionMissingError
from manicule.ingest.reembed import ReembedRecovery
from manicule.ingest.sweeps import SweepResult
from tests.app.fakes import FakeBackend, FakeIngestion

if TYPE_CHECKING:
    from collections.abc import Mapping


class Clock:
    """A stand-in for :func:`asyncio.sleep` that a test advances by hand.

    Each waiter announces the interval it asked for and then blocks until the test releases it.
    ``arrived`` is what makes the waiting itself an assertion: a loop that never reached its
    sleep never asked for one.
    """

    def __init__(self) -> None:
        self.asked: list[float] = []
        self.arrived = asyncio.Event()
        self._release = asyncio.Event()

    async def sleep(self, seconds: float) -> None:
        self.asked.append(seconds)
        self.arrived.set()
        await self._release.wait()
        # One release per round: cleared here so the next loop iteration blocks again rather
        # than spinning, which is what makes "exactly one tick" a thing a test can ask for.
        self._release.clear()

    async def tick(self) -> None:
        """Let a waiting loop run once, and return only when it is asleep again.

        **The waiting is the assertion.** Returning once the loop is back at its next sleep
        means the sync it ran has finished, so a test can assert about it on the next line
        without polling and without a sleep of its own — and a loop that never came back fails
        on the timeout rather than by an assertion about an empty list that had not filled yet.
        """
        self.arrived.clear()
        self._release.set()
        await asyncio.wait_for(self.arrived.wait(), timeout=5)


def service_with(
    sources: Mapping[str, ConnectorSettings],
) -> tuple[ApplicationService, FakeIngestion]:
    """A service over a fake backend, with the fake that records what got synced.

    Handed back together rather than reached through the service afterwards, so no test has to
    open a private attribute to find out whether the thing it is testing happened.
    """
    backend = FakeBackend()
    service = ApplicationService(backend)
    service.settings.connectors.clear()
    service.settings.connectors.update(sources)
    return service, backend.ingestion_


def source(*, schedule_s: float | None = None, enabled: bool = True) -> ConnectorSettings:
    return ConnectorSettings.model_validate(
        {
            "type": "filesystem",
            "enabled": enabled,
            "schedule_s": schedule_s,
            "options": {"root": "."},
        }
    )


# --- what gets scheduled --------------------------------------------------------------------------


def test_a_source_with_no_schedule_is_not_scheduled() -> None:
    """The default, and the behavior every existing installation keeps.

    ``schedule_s`` is absent from every configuration written before this existed, so the
    absence has to mean "syncs when somebody asks" rather than "syncs on some default".
    """
    service, _ = service_with({"handbook": source(), "runbooks": source(schedule_s=600)})

    assert Scheduler.configure(service) == {"runbooks": 600}


def test_a_disabled_source_is_never_scheduled_whatever_its_schedule_says() -> None:
    """A schedule is exactly where a disabled source would come back to life unobserved.

    ``connector sync`` refuses a disabled source loudly, and #98 made it do so because an
    operator who turned a source off and checked was being told it was off by the same program
    that would then sync it. A scheduler that ran one anyway would reintroduce that in the one
    place nobody is watching — and this is the assertion that says so.
    """
    service, _ = service_with(
        {
            "handbook": source(schedule_s=600, enabled=False),
            "runbooks": source(schedule_s=900, enabled=True),
        }
    )

    assert Scheduler.configure(service) == {"runbooks": 900}


def test_each_source_keeps_its_own_interval() -> None:
    """One number for the whole installation would be tuned for whichever source got noticed."""
    service, _ = service_with(
        {"handbook": source(schedule_s=3600), "runbooks": source(schedule_s=600)}
    )

    assert Scheduler.configure(service) == {"handbook": 3600, "runbooks": 600}


def test_lifecycle_planning_cadence_is_carried_by_production_configuration() -> None:
    service, _ = service_with({})
    service.settings.storage.lifecycle_plan_schedule_s = 300

    configured = Scheduler.configure(service)

    assert configured == {}
    assert configured.lifecycle_interval_s == 300


# --- what it does ---------------------------------------------------------------------------------


async def test_a_scheduled_sync_runs_without_a_command_being_typed() -> None:
    """The whole point, asserted through the service the command line would have called.

    #98 removed ``schedule_s`` because nothing read it. This is the test that would have failed
    then and passes now, and it is deliberately about the *sync happening* rather than about the
    setting being present.
    """
    service, ingestion = service_with({"handbook": source(schedule_s=600)})
    clock = Clock()
    scheduler = Scheduler(service, Scheduler.configure(service), sleep=clock.sleep)

    scheduler.start()
    try:
        await asyncio.wait_for(clock.arrived.wait(), timeout=5)
        assert ingestion.synced == [], "a source was synced before its first interval elapsed"
        await clock.tick()
        assert ingestion.synced == ["handbook"]
    finally:
        await scheduler.aclose()

    # The set rather than the first element. Two loops share this clock — the source's and the
    # vector sweep's — so which one reaches its sleep first is task-start order and not a claim
    # worth asserting. What is worth asserting is that each waited for its own configured
    # interval rather than for a default, and naming both is what keeps a cadence that quietly
    # stopped being armed from passing here.
    assert set(clock.asked) == {600, service.settings.ingest.sweep_interval_s}
    assert scheduler.scheduled["handbook"].runs == 1


async def test_the_vector_sweep_runs_on_cadence_without_a_command_being_typed() -> None:
    """The loop `docs/storage.md` §8.2 has always described and nothing ran.

    The delete trigger has been writing tombstones since #33. Until this loop existed nothing
    read them, so a chunk deleted or re-chunked left its vector in LanceDB permanently — still
    consuming a top-`k` slot ahead of the join that hides it — and `ingest.sweep_interval_s` was
    a setting with nothing behind it.

    Asserted through the service the command line would have called, and about the *sweep
    happening* rather than about the setting being present, on the same reasoning as the source
    loop above.
    """
    service, ingestion = service_with({})
    ingestion.vector_sweep = SweepResult(vectors_removed=7, documents_purged=2)
    clock = Clock()
    scheduler = Scheduler(service, Scheduler.configure(service), sleep=clock.sleep)

    scheduler.start()
    try:
        await asyncio.wait_for(clock.arrived.wait(), timeout=5)
        assert scheduler.sweeps is not None
        assert ingestion.vector_sweeps == [], "a sweep ran before its first interval elapsed"
        await clock.tick()
        assert scheduler.sweeps.runs == 1
    finally:
        await scheduler.aclose()

    assert set(clock.asked) == {service.settings.ingest.sweep_interval_s}
    assert scheduler.sweeps is not None
    assert scheduler.sweeps.vectors_removed == 7
    assert scheduler.sweeps.documents_purged == 2
    assert scheduler.sweeps.failures == 0


async def test_the_sweep_loop_carries_the_settings_that_bound_it() -> None:
    """Two settings that parsed and reached nothing until this loop existed.

    A batch that never arrives is indistinguishable from one that works right up until somebody
    changes it and nothing happens, which is the failure this records rather than infers.
    """
    service, ingestion = service_with({})
    service.settings.ingest.sweep_batch = 17
    service.settings.ingest.soft_delete_grace_s = 42.0
    clock = Clock()
    scheduler = Scheduler(service, Scheduler.configure(service), sleep=clock.sleep)

    scheduler.start()
    try:
        await asyncio.wait_for(clock.arrived.wait(), timeout=5)
        await clock.tick()
    finally:
        await scheduler.aclose()

    assert ingestion.vector_sweeps == [(17, 42.0)]


async def test_a_sweep_that_declined_is_counted_apart_from_one_that_failed() -> None:
    """They need opposite things: one is the design working, the other wants somebody.

    A pass blocked by a backup or a lifecycle mutation is the gate doing its job. Folding it
    into `failures` would make a healthy installation look like it was erroring hourly.
    """
    service, ingestion = service_with({})
    ingestion.vector_sweep = SweepResult(blocked_by="a backup is running")
    clock = Clock()
    scheduler = Scheduler(service, Scheduler.configure(service), sleep=clock.sleep)

    scheduler.start()
    try:
        await asyncio.wait_for(clock.arrived.wait(), timeout=5)
        await clock.tick()
    finally:
        await scheduler.aclose()

    assert scheduler.sweeps is not None
    assert scheduler.sweeps.blocked == 1
    assert scheduler.sweeps.failures == 0
    assert scheduler.sweeps.vectors_removed == 0


async def test_a_failing_sweep_does_not_end_the_loop() -> None:
    """Every failure this can see is transient, and the work is idempotent.

    A loop that exited on the first would need a restart to resume and would give no sign it had
    stopped — which for a sweep means the tombstone table growing silently, the exact condition
    this whole loop exists to prevent.
    """
    service, ingestion = service_with({})
    calls = 0

    async def failing(*, batch: int, soft_delete_grace_s: float) -> SweepResult:
        del batch, soft_delete_grace_s
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("the writer was busy")
        return SweepResult(vectors_removed=3)

    ingestion.sweep_vectors = failing
    clock = Clock()
    scheduler = Scheduler(service, Scheduler.configure(service), sleep=clock.sleep)

    scheduler.start()
    try:
        await asyncio.wait_for(clock.arrived.wait(), timeout=5)
        await clock.tick()
        await clock.tick()
    finally:
        await scheduler.aclose()

    assert scheduler.sweeps is not None
    assert scheduler.sweeps.failures == 1
    assert scheduler.sweeps.last_error_type == "OSError"
    assert scheduler.sweeps.vectors_removed == 3, "the pass after the failure still ran"


async def test_configured_lifecycle_plans_run_as_a_real_aggregate_only_scheduler_job() -> None:
    service, _ = service_with({})
    service.settings.storage.lifecycle_plan_schedule_s = 300
    service.settings.storage.source_history_retention_days = 30
    service.settings.storage.snapshot_plan_run_id = "snapshot-run"
    clock = Clock()
    scheduler = Scheduler(
        service,
        Scheduler.configure(service),
        sleep=clock.sleep,
        now=lambda: datetime(2026, 8, 15, 12, tzinfo=UTC),
    )

    scheduler.start()
    try:
        await asyncio.wait_for(clock.arrived.wait(), timeout=5)
        assert scheduler.lifecycle is not None
        assert scheduler.lifecycle.runs == 0
        await clock.tick()
        assert scheduler.lifecycle.runs == 1
    finally:
        await scheduler.aclose()

    assert clock.asked[0] == 300
    assert scheduler.lifecycle is not None
    assert scheduler.lifecycle.failures == 0
    assert set(scheduler.lifecycle.reports) == {
        "lifecycle_reset_derived",
        "lifecycle_cleanup_generations",
        "lifecycle_release_history",
        "lifecycle_delete_snapshot",
    }
    assert {report["operation"] for report in scheduler.lifecycle.reports.values()} == {
        "reset_derived",
        "cleanup_derived_generations",
        "release_source_history",
        "delete_snapshot",
    }


async def test_nothing_is_synced_at_startup() -> None:
    """A restart is how a session is re-taken, so it is something an operator does often.

    A server that swept every scheduled source the moment it started would turn each of those
    into a full corpus sync nobody asked for — on the exact command somebody runs when they are
    already waiting to get back to work.
    """
    service, ingestion = service_with({"handbook": source(schedule_s=600)})
    clock = Clock()
    scheduler = Scheduler(service, Scheduler.configure(service), sleep=clock.sleep)

    scheduler.start()
    try:
        await asyncio.wait_for(clock.arrived.wait(), timeout=5)
    finally:
        await scheduler.aclose()

    assert ingestion.synced == []


async def test_startup_runs_durable_reembedding_recovery_as_an_inspectable_job() -> None:
    service, ingestion = service_with({})
    await ingestion.reembed_start("recoverable", "synthetic-owner")
    scheduler = Scheduler(service, {})

    scheduler.start()
    try:
        for _ in range(20):
            if scheduler.reembedding.complete:
                break
            await asyncio.sleep(0)
        assert scheduler.reembedding.complete
        assert scheduler.reembedding.recovered == 1
        assert scheduler.reembedding.failures == 0
        assert ingestion.reembed_recoveries == 1
    finally:
        await scheduler.aclose()


async def test_reembedding_recovery_reports_isolated_failures_without_run_ids() -> None:
    service, ingestion = service_with({})
    ingestion.reembed_recovery_outcome = ReembedRecovery(
        recovered=2, failures=1, failure_types=("ReembedValidationError",)
    )
    scheduler = Scheduler(service, {})

    scheduler.start()
    try:
        for _ in range(20):
            if scheduler.reembedding.complete:
                break
            await asyncio.sleep(0)
        assert scheduler.reembedding.complete
        assert scheduler.reembedding.recovered == 2
        assert scheduler.reembedding.failures == 1
        assert scheduler.reembedding.last_error_type == "ReembedValidationError"
        assert "run" not in scheduler.reembedding.last_error_type.lower()
    finally:
        await scheduler.aclose()


async def test_a_disabled_source_is_not_synced_even_with_the_scheduler_running() -> None:
    """The configuration test above says it is not scheduled; this says it is not synced.

    Two assertions rather than one, because "the list is right" and "nothing ran" fail
    separately: a scheduler that ignored its own configuration and swept every configured source
    would pass the first and not the second.
    """
    service, ingestion = service_with(
        {"off": source(schedule_s=600, enabled=False), "on": source(schedule_s=600)}
    )
    clock = Clock()
    scheduler = Scheduler(service, Scheduler.configure(service), sleep=clock.sleep)

    scheduler.start()
    try:
        await asyncio.wait_for(clock.arrived.wait(), timeout=5)
        await clock.tick()
        assert ingestion.synced == ["on"]
    finally:
        await scheduler.aclose()

    assert "off" not in ingestion.synced


async def test_a_failing_sync_does_not_stop_the_schedule() -> None:
    """The commonest failures here are transient or need a person, and neither is a reason to
    stop trying: an instance that was down at ten past may be up at twenty past.

    A loop that exited on the first refusal would need a server restart to resume and would give
    no sign it had stopped, which is the worst of both.
    """
    service, _ = service_with({"nowhere": source(schedule_s=600)})
    clock = Clock()
    # The source is scheduled and not configured for the sync to find, so every run refuses.
    service.settings.connectors.pop("nowhere")
    scheduler = Scheduler(service, {"nowhere": 600}, sleep=clock.sleep)

    scheduler.start()
    try:
        await asyncio.wait_for(clock.arrived.wait(), timeout=5)
        await clock.tick()
        assert scheduler.scheduled["nowhere"].failures == 1
        await clock.tick()
        assert scheduler.scheduled["nowhere"].failures == 2, (
            "the loop stopped after its first refusal"
        )
    finally:
        await scheduler.aclose()

    assert scheduler.scheduled["nowhere"].runs == 0


async def test_a_missing_session_is_reported_as_itself_rather_than_as_a_failed_sync(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The failure a restart causes, and the one most likely to be misread at three in the morning.

    A session lives in the server's memory, so launchd restarting the process ends it and the
    next scheduled sync cannot authenticate. Reported like every other refusal it reads exactly
    like the instance being unreachable — and the two need opposite things: one needs a person at
    a browser and the other needs nothing at all.

    So three things are asserted, and each is a different way of being told. The sentence says
    the *server* holds no session; it says the instance was not contacted, which is what
    distinguishes this from an outage; and it names the command that fixes it, with the source's
    own name in it rather than a placeholder somebody has to substitute.
    """
    service, ingestion = service_with({"handbook": source(schedule_s=600)})
    ingestion.failure = SessionMissingError("no Confluence browser session is held for the site")
    clock = Clock()
    scheduler = Scheduler(service, Scheduler.configure(service), sleep=clock.sleep)

    scheduler.start()
    try:
        await asyncio.wait_for(clock.arrived.wait(), timeout=5)
        await clock.tick()
    finally:
        await scheduler.aclose()

    said = capsys.readouterr().err
    assert "holds no Confluence session" in said, said
    assert "has not been contacted" in said, (
        f"the message does not rule out an outage at the instance, which is the other thing this "
        f"could be:\n{said}"
    )
    assert "manicule connector login handbook --browser" in said, said


async def test_the_source_waiting_to_be_signed_in_to_is_a_state_and_not_only_a_log_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Recorded on the schedule, so the server can be asked rather than have its log read.

    A counter would say "it failed nine times"; what an operator needs is "it is still waiting",
    which is a state and stays true until somebody signs in. Cleared by a run that succeeds, and
    that half is asserted too — a flag nothing clears is a flag that means "at some point".
    """
    service, ingestion = service_with({"handbook": source(schedule_s=600)})
    ingestion.failure = SessionMissingError("no Confluence browser session is held")
    clock = Clock()
    scheduler = Scheduler(service, Scheduler.configure(service), sleep=clock.sleep)

    scheduler.start()
    try:
        await asyncio.wait_for(clock.arrived.wait(), timeout=5)
        await clock.tick()
        record = scheduler.scheduled["handbook"]
        assert record.awaiting_sign_in is True
        assert record.failures == 1, "it is a failure as well as a state"

        ingestion.failure = None
        await clock.tick()
        assert record.awaiting_sign_in is False, "a run that worked left the flag raised"
        assert record.runs == 1
    finally:
        await scheduler.aclose()
    capsys.readouterr()


async def test_a_returned_incomplete_run_clears_a_prior_missing_session_state(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reaching the pipeline proves authentication succeeded, whatever stopped the run later."""
    from manicule.ingest.pipeline import RunReport  # noqa: PLC0415

    service, ingestion = service_with({"handbook": source(schedule_s=600)})
    ingestion.failure = SessionMissingError("no Confluence browser session is held")
    clock = Clock()
    scheduler = Scheduler(service, Scheduler.configure(service), sleep=clock.sleep)

    scheduler.start()
    try:
        await asyncio.wait_for(clock.arrived.wait(), timeout=5)
        await clock.tick()
        assert scheduler.scheduled["handbook"].awaiting_sign_in is True

        ingestion.failure = None
        ingestion.report = RunReport(
            connector="handbook",
            error="CursorExpiredError: the search cursor expired",
            error_type="CursorExpiredError",
            error_message="the search cursor expired",
            enumeration_completed=False,
        )
        await clock.tick()
        record = scheduler.scheduled["handbook"]
        assert record.awaiting_sign_in is False
        assert record.last_outcome == "incomplete"
        assert record.failures == 2
        assert record.runs == 0
    finally:
        await scheduler.aclose()
    capsys.readouterr()


async def test_an_unreachable_instance_is_not_reported_as_a_missing_session(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The control, without which "it says the right thing" is "it says that about everything".

    A connector error is what an instance that is down produces. It has to keep the ordinary
    reporting and must not raise :attr:`~manicule.app.served.ScheduledSource.awaiting_sign_in`,
    because sending somebody to open a browser over a network outage is worse than saying
    nothing: they do it, it works, and the sync fails again for the reason nobody looked at.
    """
    service, ingestion = service_with({"handbook": source(schedule_s=600)})
    ingestion.failure = ConnectorError("the instance answered 503")
    clock = Clock()
    scheduler = Scheduler(service, Scheduler.configure(service), sleep=clock.sleep)

    scheduler.start()
    try:
        await asyncio.wait_for(clock.arrived.wait(), timeout=5)
        await clock.tick()
    finally:
        await scheduler.aclose()

    assert scheduler.scheduled["handbook"].awaiting_sign_in is False
    assert scheduler.scheduled["handbook"].failures == 1
    said = capsys.readouterr().err
    assert "503" in said, said
    assert "connector login" not in said, (
        f"an instance that is down sent the operator to a browser:\n{said}"
    )


async def test_stopping_the_scheduler_leaves_no_task_running() -> None:
    """A server that stopped and left a loop syncing would be a second writer with no lock."""
    service, _ = service_with(
        {"handbook": source(schedule_s=600), "runbooks": source(schedule_s=60)}
    )
    clock = Clock()
    scheduler = Scheduler(service, Scheduler.configure(service), sleep=clock.sleep)

    before = {task.get_name() for task in asyncio.all_tasks()}
    scheduler.start()
    await asyncio.wait_for(clock.arrived.wait(), timeout=5)
    await scheduler.aclose()

    after = {task.get_name() for task in asyncio.all_tasks()}
    assert not {name for name in after - before if name.startswith("schedule:")}


@pytest.mark.parametrize("interval", [0, -1, -600])
def test_an_interval_of_zero_or_less_is_refused_at_configuration(interval: int) -> None:
    """A loop with no wait in it is not a schedule, and clamping would pick a number nobody
    wrote — the two candidates being "never" and "constantly"."""
    from pydantic import ValidationError  # noqa: PLC0415

    with pytest.raises(ValidationError, match="schedule_s"):
        ConnectorSettings.model_validate({"type": "filesystem", "schedule_s": interval})


# --- what a schedule must never do --------------------------------------------------------------


def test_a_scheduled_sync_can_never_launch_a_browser() -> None:
    """Acceptance criterion 13, asserted structurally because the behavioral version cannot fail.

    A scheduled sync resolves its credential through `credential_for` -> `load_session`, which
    reads the vault and nothing else. There is no provider anywhere on that path, so no browser
    can open — and a test that merely ran a scheduled sync and observed no window would pass just
    as happily against an implementation that had one, on a machine with no browser installed.

    So the scheduler's **own source** is what is read, rather than the module's: `served.py` also
    holds `ControlHandler`, which legitimately receives the session `connector login` captured,
    and a module-wide search would forbid the one place the credential is *supposed* to arrive.

    This fails the moment somebody "helpfully" makes an unattended job re-authenticate, which is
    the change the criterion exists to prevent: a server that opens a sign-in window at three in
    the morning on a machine nobody is sitting at, and blocks the schedule until it times out.
    """
    import inspect  # noqa: PLC0415 - kept beside its only use

    from manicule.app.served import Scheduler  # noqa: PLC0415

    body = inspect.getsource(Scheduler)
    for opener in (
        "PlaywrightProvider",
        "InstalledChromiumProvider",
        "connector_login(",
        "launch_persistent_context",
        "_driver_for",
        "authenticate(",
    ):
        assert opener not in body, (
            f"{opener} appears in the scheduler. An unattended job that can authenticate is one "
            f"that opens a window on a machine nobody is sitting at."
        )


async def test_a_scheduled_sync_with_no_session_refuses_and_keeps_its_place() -> None:
    """The behavior the criterion pairs with: refuse, say so, change nothing durable.

    `awaiting_sign_in` is the state a person acts on. What matters beside it is that the run
    recorded a failure rather than a success — a schedule that counted an unauthenticated run as
    done would advance nothing and report everything as fine.
    """
    service, ingestion = service_with({"handbook": source(schedule_s=600)})
    ingestion.failure = SessionMissingError("nobody has signed in to this instance")
    clock = Clock()
    scheduler = Scheduler(service, Scheduler.configure(service), sleep=clock.sleep)

    scheduler.start()
    try:
        await asyncio.wait_for(clock.arrived.wait(), timeout=5)
        await clock.tick()
    finally:
        await scheduler.aclose()

    record = scheduler.scheduled["handbook"]
    assert record.awaiting_sign_in is True
    assert record.failures == 1
    assert record.runs == 0, "an unauthenticated run must not be counted as a completed sync"
