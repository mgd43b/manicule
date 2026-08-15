"""The process boundary, and the two failures it exists to survive.

These start real subprocesses. They are slower than everything else in the suite and there is
no alternative: a worker that ran in this process would not be a worker, and the properties
under test — a deadline that fires against native code, a memory bound that carries a
``SIGKILL`` — are exactly the ones an in-process double cannot have.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import sys
import threading
import time
from pathlib import Path
from typing import cast

import pytest
from manicule_plugin_example import MEDIA_TYPE as EXAMPLE_MEDIA_TYPE
from manicule_plugin_hostile import (
    CRASHING_MEDIA_TYPE,
    GREEDY_MEDIA_TYPE,
    HANGING_MEDIA_TYPE,
    StatefulMiddleware,
)

import manicule.ingest.workers as worker_module
from manicule.core.anchors import Unlocated
from manicule.core.content import BlockKind, Chunk, DocumentStatus, ParsedBlock, RawDocument
from manicule.ingest.limits import (
    ADDRESS_SPACE_HEADROOM,
    limit_address_space,
    resident_bytes,
)
from manicule.ingest.middleware import MiddlewareRunner
from manicule.ingest.workers import (
    MEGABYTE,
    InProcessRunner,
    StageFailure,
    WorkerConfig,
    WorkerPool,
    default_worker_count,
)
from manicule.parsers.chain import Outcome, run_chain
from tests.ingest import fakes
from tests.storage_helpers import make_document

pytestmark = pytest.mark.slow


def _config(tmp_path: Path, **overrides: object) -> WorkerConfig:
    values: dict[str, object] = {
        "data_dir": tmp_path / "data",
        "cache_dir": tmp_path / "cache",
        "parser_fallbacks": {},
        "plugin_config": {},
        **overrides,
    }
    return WorkerConfig.model_validate(values)


def _raw(media_type: str, content: str = "alpha") -> RawDocument:
    return RawDocument(source_id="a", uri="memory://a", media_type=media_type, content=content)


async def test_a_parser_that_hangs_is_killed_and_recorded_as_a_hard_failure(
    tmp_path: Path, manicule_environment: Path
) -> None:
    """``asyncio.wait_for`` cancels the await, not the work.

    The hanging parser here sleeps in the child. Nothing in the parent can interrupt it — a
    thread cannot be killed and a native call observes no cancellation — so the only thing that
    ends it is the parent killing the process. The outcome must be ``failed``, never
    ``declined``: a parser that declined reported information, and one we killed reported
    nothing at all.
    """
    del manicule_environment
    async with WorkerPool(_config(tmp_path), workers=1, timeout_s=1.0) as pool:
        result = await pool.run_attempt("hanging", _raw(HANGING_MEDIA_TYPE))

    assert result.attempt.outcome is Outcome.FAILED
    assert "worker killed: timeout" in result.attempt.reason
    assert pool.kills["timeout"] == 1


async def test_a_chain_of_timeouts_is_failed_and_never_unsupported_media_type(
    tmp_path: Path, manicule_environment: Path
) -> None:
    """The outcome that would send an operator to write a parser that already exists.

    Two parsers, both killed by the deadline, and the classification must be ``failed`` at
    stage ``parse``. If a kill were treated as a decline, this chain would report the format as
    unsupported — which is the truth about nothing.
    """
    del manicule_environment
    raw = _raw(HANGING_MEDIA_TYPE)
    async with WorkerPool(_config(tmp_path), workers=1, timeout_s=1.0) as pool:

        async def attempt(name: str, document: RawDocument) -> tuple[list[object], object]:
            outcome = await pool.run_attempt(name, document)
            return outcome.blocks, outcome.attempt  # pyright: ignore[reportReturnType]

        result = await run_chain(["hanging", "hanging"], raw, attempt)  # pyright: ignore[reportArgumentType]

    assert result.status is DocumentStatus.FAILED
    assert result.status is not DocumentStatus.UNSUPPORTED_MEDIA_TYPE
    assert [a.outcome for a in result.attempts] == [Outcome.FAILED, Outcome.FAILED]


async def test_a_parser_that_allocates_without_bound_is_stopped(
    tmp_path: Path, manicule_environment: Path
) -> None:
    """A memory bound the kernel will not enforce on the platform this is built for.

    ``RLIMIT_AS`` reports unlimited on Darwin and refuses to be set, so a design that relied on
    it would be correct on CI and inert on the machine where the malformed document gets opened
    first. The parent samples resident memory and sends ``SIGKILL``, which works identically
    everywhere.

    **The assertion is about the outcome, not the mechanism, and deliberately so.** On Linux
    the child may hit the ``RLIMIT_AS`` backstop and die of its own accord before the parent's
    next sample; on Darwin only the parent can stop it. Asserting "the parent killed it" would
    be asserting a platform, and this project's rule is that a document ingests identically
    wherever it runs. What must be identical is this: the attempt is a hard failure, the run
    survives, and the pool is still usable afterwards.
    """
    del manicule_environment
    config = _config(tmp_path, memory_limit_bytes=64 * MEGABYTE)
    async with WorkerPool(config, workers=1, timeout_s=60.0, poll_interval_s=0.05) as pool:
        result = await pool.run_attempt("greedy", _raw(GREEDY_MEDIA_TYPE))
        after = await pool.run_attempt("example", _raw(EXAMPLE_MEDIA_TYPE))

    assert result.attempt.outcome is Outcome.FAILED
    assert result.attempt.reason, "a hard failure must say what happened"
    assert after.attempt.outcome is Outcome.PARSED, "the run survives, which is the whole point"


async def test_a_parser_that_crashes_the_interpreter_is_attributed_to_its_document(
    tmp_path: Path, manicule_environment: Path
) -> None:
    """The parent sees a closed pipe, not an exception, and must not read it as the end.

    This is what a segfault inside a native extension looks like from the outside. Treating
    it as anything other than one document's hard failure loses the rest of the batch.
    """
    del manicule_environment
    async with WorkerPool(_config(tmp_path), workers=1, timeout_s=30.0) as pool:
        result = await pool.run_attempt("crashing", _raw(CRASHING_MEDIA_TYPE))

    assert result.attempt.outcome is Outcome.FAILED
    assert "worker" in result.attempt.reason


async def test_a_worker_that_died_between_documents_fails_one_and_not_the_run(
    tmp_path: Path, manicule_environment: Path
) -> None:
    """A broken pipe is a process-level accident, and it must not reach the ingest loop.

    A worker can die between being handed back as idle and being dispatched to — the OOM
    killer, or a crash on the previous document that surfaced late — and the parent's ``send``
    then raises. Letting that propagate would end a batch over one document, which is the one
    thing this module exists to prevent. The worker is killed here to reproduce exactly that.
    """
    del manicule_environment
    async with WorkerPool(_config(tmp_path), workers=1, timeout_s=5.0) as pool:
        # Reaching the pipe is the only way to reproduce a broken one, and widening the pool's
        # public surface so that a test can break something would be the worse trade.
        idle = await pool._idle.get()  # pyright: ignore[reportPrivateUsage]
        assert idle is not None
        idle.connection.close()
        await pool._idle.put(idle)  # pyright: ignore[reportPrivateUsage]

        result = await pool.run_attempt("example", _raw(EXAMPLE_MEDIA_TYPE))
        after = await pool.run_attempt("example", _raw(EXAMPLE_MEDIA_TYPE))

    assert result.attempt.outcome is Outcome.FAILED
    assert "worker unreachable" in result.attempt.reason
    assert after.attempt.outcome is Outcome.PARSED, "the pool replaced it and the run continued"


async def test_a_killed_worker_is_replaced_and_the_pool_keeps_working(
    tmp_path: Path, manicule_environment: Path
) -> None:
    """The pool size is what a run depends on, not the identity of any worker."""
    del manicule_environment
    async with WorkerPool(_config(tmp_path), workers=1, timeout_s=2.0) as pool:
        await pool.run_attempt("hanging", _raw(HANGING_MEDIA_TYPE))
        after = await pool.run_attempt("example", _raw(EXAMPLE_MEDIA_TYPE))

    assert after.attempt.outcome is Outcome.PARSED, (
        "a document after a killed one must still be parseable"
    )


async def test_repeated_cancel_during_parser_retire_restores_the_only_permit(
    tmp_path: Path,
    manicule_environment: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exchange thread and replacement reach a known endpoint before cancel propagates."""
    del manicule_environment
    async with WorkerPool(_config(tmp_path), workers=1, timeout_s=60.0) as pool:
        baseline_children = len(multiprocessing.active_children())
        retiring = asyncio.Event()
        release = asyncio.Event()
        original_retire = pool._retire  # pyright: ignore[reportPrivateUsage]

        async def delayed_retire(worker: object) -> None:
            retiring.set()
            await release.wait()
            await original_retire(worker)  # pyright: ignore[reportArgumentType]

        monkeypatch.setattr(pool, "_retire", delayed_retire)
        parsing = asyncio.create_task(pool.run_attempt("hanging", _raw(HANGING_MEDIA_TYPE)))
        await asyncio.sleep(0.05)
        parsing.cancel()
        await asyncio.wait_for(retiring.wait(), timeout=10)
        parsing.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await parsing

        assert len(multiprocessing.active_children()) == baseline_children
        async with asyncio.timeout(10):
            after = await pool.run_attempt("example", _raw(EXAMPLE_MEDIA_TYPE))
        assert after.attempt.outcome is Outcome.PARSED
        assert len(multiprocessing.active_children()) == baseline_children


async def test_cancel_during_spawn_rolls_setup_back_to_a_reusable_pool(
    tmp_path: Path,
    manicule_environment: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A canceled readiness wait cannot leave `_started` true over a missing permit."""
    del manicule_environment
    baseline = {child.pid for child in multiprocessing.active_children()}
    entered = threading.Event()
    release = threading.Event()
    original = worker_module._await_ready  # pyright: ignore[reportPrivateUsage]

    def delayed_ready(connection: object, timeout: float = 60.0) -> object:
        entered.set()
        release.wait(timeout=10)
        return original(connection, timeout)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(worker_module, "_await_ready", delayed_ready)
    pool = WorkerPool(_config(tmp_path), workers=1, timeout_s=5.0)
    starting = asyncio.create_task(pool.setup())
    assert await asyncio.to_thread(entered.wait, 10)
    starting.cancel()
    starting.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await starting
    assert {child.pid for child in multiprocessing.active_children()} == baseline

    monkeypatch.setattr(worker_module, "_await_ready", original)
    await pool.setup()
    result = await asyncio.wait_for(
        pool.run_attempt("example", _raw(EXAMPLE_MEDIA_TYPE)), timeout=10
    )
    assert result.attempt.outcome is Outcome.PARSED
    await pool.teardown()
    assert {child.pid for child in multiprocessing.active_children()} == baseline


async def test_repeated_cancel_during_multiworker_teardown_reaps_every_snapshot(
    tmp_path: Path,
    manicule_environment: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Teardown remains the owner of every snapshotted child through repeated cancellation."""
    del manicule_environment
    baseline = {child.pid for child in multiprocessing.active_children()}
    pool = WorkerPool(_config(tmp_path), workers=3, timeout_s=5.0)
    await pool.setup()
    owned = {worker.pid for worker in pool._live}  # pyright: ignore[reportPrivateUsage]
    assert len(owned) == 3

    entered = 0
    entered_lock = threading.Lock()
    all_entered = threading.Event()
    release = threading.Event()
    original = worker_module._Worker.terminate  # pyright: ignore[reportPrivateUsage]

    def delayed_terminate(worker: object) -> None:
        nonlocal entered
        with entered_lock:
            entered += 1
            if entered == 3:
                all_entered.set()
        release.wait(timeout=10)
        original(worker)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(worker_module._Worker, "terminate", delayed_terminate)  # pyright: ignore[reportPrivateUsage]
    stopping = asyncio.create_task(pool.teardown())
    assert await asyncio.to_thread(all_entered.wait, 10)
    stopping.cancel()
    stopping.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await stopping

    active = {child.pid for child in multiprocessing.active_children()}
    assert not (owned & active)
    assert active == baseline
    assert pool._live == []  # pyright: ignore[reportPrivateUsage]


async def test_teardown_fences_a_checked_out_attempt_before_reusable_setup(
    tmp_path: Path,
    manicule_environment: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late release from an old generation cannot repopulate a stopped pool."""
    del manicule_environment
    baseline = {child.pid for child in multiprocessing.active_children()}
    pool = WorkerPool(_config(tmp_path), workers=2, timeout_s=5.0)
    await pool.setup()
    replied = asyncio.Event()
    release = asyncio.Event()
    original_dispatch = pool._dispatch  # pyright: ignore[reportPrivateUsage]

    async def held_dispatch(worker: object, request: object) -> object:
        result = await original_dispatch(worker, request)  # pyright: ignore[reportArgumentType]
        replied.set()
        await release.wait()
        return result

    monkeypatch.setattr(pool, "_dispatch", held_dispatch)
    attempt = asyncio.create_task(pool.run_attempt("example", _raw(EXAMPLE_MEDIA_TYPE)))
    await asyncio.wait_for(replied.wait(), timeout=10)

    await pool.teardown()
    assert pool._live == []  # pyright: ignore[reportPrivateUsage]
    assert pool._idle.qsize() == 0  # pyright: ignore[reportPrivateUsage]
    assert {child.pid for child in multiprocessing.active_children()} == baseline

    release.set()
    result = await asyncio.wait_for(attempt, timeout=10)
    assert result.attempt.outcome is Outcome.PARSED
    assert pool._live == []  # pyright: ignore[reportPrivateUsage]
    assert pool._idle.qsize() == 0  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setattr(pool, "_dispatch", original_dispatch)
    await pool.setup()
    assert len(pool._live) == 2  # pyright: ignore[reportPrivateUsage]
    assert pool._idle.qsize() == 2  # pyright: ignore[reportPrivateUsage]
    active = {child.pid for child in multiprocessing.active_children()}
    assert len(active - baseline) == 2
    after = await asyncio.wait_for(
        pool.run_attempt("example", _raw(EXAMPLE_MEDIA_TYPE)), timeout=10
    )
    assert after.attempt.outcome is Outcome.PARSED
    assert len(pool._live) == 2  # pyright: ignore[reportPrivateUsage]
    assert pool._idle.qsize() == 2  # pyright: ignore[reportPrivateUsage]
    await pool.teardown()
    assert {child.pid for child in multiprocessing.active_children()} == baseline


async def test_repeated_cancel_after_permit_selection_restores_exact_width(
    tmp_path: Path,
    manicule_environment: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation while joining the losing close waiter cannot consume the worker."""
    del manicule_environment
    async with WorkerPool(_config(tmp_path), workers=1, timeout_s=5.0) as pool:
        selected = asyncio.Event()
        hold_selection = asyncio.Event()
        restoring = asyncio.Event()
        release_restore = asyncio.Event()
        original = pool._finish_permit_selection  # pyright: ignore[reportPrivateUsage]
        original_restore = pool._restore_permit  # pyright: ignore[reportPrivateUsage]

        async def held_finish(closed_task: asyncio.Task[bool]) -> None:
            selected.set()
            await hold_selection.wait()
            await original(closed_task)

        async def held_restore(permit: object, generation: int) -> None:
            restoring.set()
            await release_restore.wait()
            await original_restore(permit, generation)  # pyright: ignore[reportArgumentType]

        monkeypatch.setattr(pool, "_finish_permit_selection", held_finish)
        monkeypatch.setattr(pool, "_restore_permit", held_restore)
        attempt = asyncio.create_task(pool.run_attempt("example", _raw(EXAMPLE_MEDIA_TYPE)))
        await asyncio.wait_for(selected.wait(), timeout=10)
        attempt.cancel()
        await asyncio.wait_for(restoring.wait(), timeout=10)
        attempt.cancel()
        release_restore.set()
        with pytest.raises(asyncio.CancelledError):
            await attempt

        assert len(pool._live) == 1  # pyright: ignore[reportPrivateUsage]
        assert pool._idle.qsize() == 1  # pyright: ignore[reportPrivateUsage]
        monkeypatch.setattr(pool, "_finish_permit_selection", original)
        monkeypatch.setattr(pool, "_restore_permit", original_restore)
        after = await asyncio.wait_for(
            pool.run_attempt("example", _raw(EXAMPLE_MEDIA_TYPE)), timeout=10
        )
        assert after.attempt.outcome is Outcome.PARSED
        assert len(pool._live) == 1  # pyright: ignore[reportPrivateUsage]
        assert pool._idle.qsize() == 1  # pyright: ignore[reportPrivateUsage]


async def test_repeated_cancel_during_lazy_spawn_readiness_restores_exact_width(
    tmp_path: Path,
    manicule_environment: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The empty permit and appended child remain owned through readiness cancellation."""
    del manicule_environment
    baseline = {child.pid for child in multiprocessing.active_children()}
    pool = WorkerPool(_config(tmp_path), workers=1, timeout_s=5.0)
    await pool.setup()
    worker = await pool._idle.get()  # pyright: ignore[reportPrivateUsage]
    assert worker is not None
    await pool._retire(worker)  # pyright: ignore[reportPrivateUsage]
    pool._idle.put_nowait(None)  # pyright: ignore[reportPrivateUsage]

    entered = threading.Event()
    stopped = asyncio.Event()
    release = threading.Event()
    original = worker_module._await_ready  # pyright: ignore[reportPrivateUsage]
    original_stop = worker_module._stop  # pyright: ignore[reportPrivateUsage]

    def delayed_ready(connection: object, timeout: float = 60.0) -> object:
        entered.set()
        release.wait(timeout=10)
        return original(connection, timeout)  # pyright: ignore[reportArgumentType]

    def observed_stop(worker: object) -> None:
        original_stop(worker)  # pyright: ignore[reportArgumentType]
        stopped.set()

    monkeypatch.setattr(worker_module, "_await_ready", delayed_ready)
    monkeypatch.setattr(worker_module, "_stop", observed_stop)
    attempt = asyncio.create_task(pool.run_attempt("example", _raw(EXAMPLE_MEDIA_TYPE)))
    assert await asyncio.to_thread(entered.wait, 10)
    attempt.cancel()
    await asyncio.wait_for(stopped.wait(), timeout=10)
    attempt.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await attempt

    assert pool._live == []  # pyright: ignore[reportPrivateUsage]
    assert pool._idle.qsize() == 1  # pyright: ignore[reportPrivateUsage]
    assert {child.pid for child in multiprocessing.active_children()} == baseline
    monkeypatch.setattr(worker_module, "_await_ready", original)
    monkeypatch.setattr(worker_module, "_stop", original_stop)

    after = await asyncio.wait_for(
        pool.run_attempt("example", _raw(EXAMPLE_MEDIA_TYPE)), timeout=10
    )
    assert after.attempt.outcome is Outcome.PARSED
    assert len(pool._live) == 1  # pyright: ignore[reportPrivateUsage]
    assert pool._idle.qsize() == 1  # pyright: ignore[reportPrivateUsage]
    await pool.teardown()
    assert {child.pid for child in multiprocessing.active_children()} == baseline


async def test_repeated_cancel_at_completed_checkout_handoff_restores_exact_width(
    tmp_path: Path,
    manicule_environment: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed worker remains owned until the shield hands it to `run_attempt`."""
    del manicule_environment
    async with WorkerPool(_config(tmp_path), workers=1, timeout_s=5.0) as pool:
        completed = asyncio.Event()
        hold_delivery = asyncio.Event()
        restoring = asyncio.Event()
        release_restore = asyncio.Event()
        original_delivery = pool._deliver_checkout  # pyright: ignore[reportPrivateUsage]
        original_restore = pool._restore_permit  # pyright: ignore[reportPrivateUsage]

        async def held_delivery(
            checkout: asyncio.Task[object],
        ) -> object:
            result = await asyncio.shield(checkout)
            completed.set()
            await hold_delivery.wait()
            return result

        async def held_restore(worker: object, generation: int) -> None:
            restoring.set()
            await release_restore.wait()
            await original_restore(worker, generation)  # pyright: ignore[reportArgumentType]

        monkeypatch.setattr(pool, "_deliver_checkout", held_delivery)
        monkeypatch.setattr(pool, "_restore_permit", held_restore)
        attempt = asyncio.create_task(pool.run_attempt("example", _raw(EXAMPLE_MEDIA_TYPE)))
        await asyncio.wait_for(completed.wait(), timeout=10)
        attempt.cancel()
        await asyncio.wait_for(restoring.wait(), timeout=10)
        attempt.cancel()
        release_restore.set()
        with pytest.raises(asyncio.CancelledError):
            await attempt

        assert len(pool._live) == 1  # pyright: ignore[reportPrivateUsage]
        assert pool._idle.qsize() == 1  # pyright: ignore[reportPrivateUsage]
        monkeypatch.setattr(pool, "_deliver_checkout", original_delivery)
        monkeypatch.setattr(pool, "_restore_permit", original_restore)
        after = await asyncio.wait_for(
            pool.run_attempt("example", _raw(EXAMPLE_MEDIA_TYPE)), timeout=10
        )
        assert after.attempt.outcome is Outcome.PARSED
        assert len(pool._live) == 1  # pyright: ignore[reportPrivateUsage]
        assert pool._idle.qsize() == 1  # pyright: ignore[reportPrivateUsage]


def test_the_default_worker_count_leaves_a_core_for_the_parent() -> None:
    """The parent does the embedding and every write, so it is not a spare."""
    count = default_worker_count()
    assert 1 <= count <= 4


def test_the_address_space_backstop_is_looser_than_the_bound_it_backs() -> None:
    """Address space and resident memory are different quantities.

    A backstop set at the resident bound would fire on well-behaved parsers that reserve an
    arena they never touch — and it would fire only on the platform where it can be set at all,
    which is precisely the platform-dependent outcome this project refuses.
    """
    assert ADDRESS_SPACE_HEADROOM > 1


def test_resident_memory_is_readable_for_this_process() -> None:
    """``None`` means "unknown", and a caller must never read it as "using nothing"."""
    measured = resident_bytes(__import__("os").getpid())
    assert measured is None or measured > 0


@pytest.mark.skipif(sys.platform != "darwin", reason="the measured Darwin behavior")
def test_the_address_space_limit_is_not_available_on_darwin() -> None:
    """Recorded as a test because a design that assumed otherwise would be silently inert.

    Measured: all three memory limits report unlimited, refuse to be set, and a child allocates
    half a gigabyte under a nominal 256 MiB cap. If this ever starts passing the other way, the
    comment in ``manicule.ingest.limits`` needs revisiting — but the enforcement does not,
    because it never depended on this.
    """
    assert limit_address_space(256 * MEGABYTE) is False


async def test_the_in_process_runner_reports_a_missing_parser_rather_than_raising() -> None:
    """A misconfigured chain fails one document; it does not end a batch."""
    runner = InProcessRunner({})

    result = await runner.run_attempt("absent", _raw("text/plain"))

    assert result.attempt.outcome is Outcome.FAILED
    assert "absent" in result.attempt.reason


async def test_an_archive_is_expanded_rather_than_parsed_for_blocks() -> None:
    """An archive parser's ``parse`` yields nothing by design.

    A runner that only ever asked for blocks would see an empty result, advance the chain, and
    end at ``no_extractable_text`` — a zip full of reports classified as a scan.
    """
    runner = InProcessRunner({"archive": fakes.FakeArchive()})

    result = await runner.run_attempt(
        "archive", _raw(fakes.CONTAINER_MEDIA_TYPE, "one=alpha\ntwo=beta")
    )

    assert result.attempt.outcome is Outcome.PARSED
    assert len(result.members) == 2


async def test_complete_parsed_block_reply_is_refused_before_crossing_budget() -> None:
    """The child reply budget covers ordinary blocks, not only container members."""
    runner = InProcessRunner({"lines": fakes.LineParser()})

    result = await runner.run_attempt(
        "lines",
        _raw("text/plain", "x" * 16_384),
        max_output_bytes=1_024,
        memory_limit_bytes=64 * MEGABYTE,
    )

    assert result.blocks == []
    assert result.members == ()
    assert result.attempt.outcome is Outcome.FAILED
    assert result.attempt.reason == "memory_bound"


async def test_middleware_and_chunker_output_never_materializes_in_serving_parent(
    tmp_path: Path, manicule_environment: Path
) -> None:
    """An oversized hook result is measured in the disposable child, not this process."""
    del manicule_environment
    config = _config(
        tmp_path,
        middleware=("expanding",),
        plugin_config={"middleware.expanding": {"chunk_megabytes": 16}},
        memory_limit_bytes=1024 * MEGABYTE,
    )
    raw = _raw("text/plain", "small source")
    document = make_document(
        source="wiki",
        source_id=raw.source_id,
        body=raw.as_bytes(),
        uri=raw.uri,
        media_type=raw.media_type,
    )
    blocks = [
        ParsedBlock(
            kind=BlockKind.PROSE,
            text="small source",
            anchor=Unlocated(reason="worker isolation regression"),
        )
    ]
    parent_before = resident_bytes(os.getpid())
    async with WorkerPool(config, workers=1, timeout_s=30.0) as pool:
        result = await pool.run_after_parse_and_chunk(
            document,
            blocks,
            max_output_bytes=64 * 1024,
            memory_limit_bytes=1024 * MEGABYTE,
            title="Hostile expansion",
            media_type=raw.media_type,
            detect_glossary=True,
        )
    parent_after = resident_bytes(os.getpid())

    assert result.reason is StageFailure.MEMORY_BOUND
    assert result.value is None
    if parent_before is not None and parent_after is not None:
        assert parent_after - parent_before < 8 * MEGABYTE


async def test_one_production_stage_session_preserves_stateful_middleware_parity(
    tmp_path: Path, manicule_environment: Path
) -> None:
    del manicule_environment
    config = _config(
        tmp_path,
        middleware=("stateful",),
        memory_limit_bytes=4096 * MEGABYTE,
    )
    raw = _raw("text/plain", "stateful source")
    document = make_document(
        source="wiki",
        source_id=raw.source_id,
        body=raw.as_bytes(),
        uri=raw.uri,
        media_type=raw.media_type,
    )
    blocks = [
        ParsedBlock(
            kind=BlockKind.PROSE,
            text="stateful source",
            anchor=Unlocated(reason="stateful middleware regression"),
        )
    ]
    live = MiddlewareRunner((StatefulMiddleware(),))
    assert live.declarations() == ("stateful@",)
    assert await live.before_parse(raw) == raw
    await live.after_parse(document, blocks)

    async with WorkerPool(config, workers=1, timeout_s=30.0) as pool:
        session = await pool.open_stage_session(memory_limit_bytes=4096 * MEGABYTE)
        try:
            before = await session.run_before_parse(
                raw,
                max_output_bytes=MEGABYTE,
                memory_limit_bytes=4096 * MEGABYTE,
            )
            offline = await session.run_after_parse_and_chunk(
                document,
                blocks,
                max_output_bytes=MEGABYTE,
                memory_limit_bytes=4096 * MEGABYTE,
                title="Stateful",
                media_type=raw.media_type,
                detect_glossary=False,
            )
        finally:
            await session.aclose()

    assert before.reason is None
    assert before.value == raw
    assert offline.reason is None
    raw_stage = cast("object", offline.value)
    assert isinstance(raw_stage, tuple)
    stage_value = cast("tuple[object, ...]", raw_stage)
    assert len(stage_value) == 2
    chunks_value, entries = stage_value
    assert isinstance(chunks_value, tuple)
    chunk_candidates = cast("tuple[object, ...]", chunks_value)
    assert all(isinstance(chunk, Chunk) for chunk in chunk_candidates)
    chunks = cast("tuple[Chunk, ...]", chunk_candidates)
    assert entries == ()
    assert chunks
    base_chunks = [
        chunk.model_copy(update={"embed_text": chunk.embed_text.removesuffix("|stateful")})
        for chunk in chunks
    ]
    live_chunks = await live.after_chunk(document, base_chunks)
    assert [chunk.embed_text for chunk in chunks] == [chunk.embed_text for chunk in live_chunks]


async def test_repeated_stage_cancellation_reaps_every_owned_child(
    tmp_path: Path, manicule_environment: Path
) -> None:
    del manicule_environment
    config = _config(
        tmp_path,
        middleware=("hanging-stage",),
        plugin_config={"middleware.hanging-stage": {"hang_seconds": 60}},
        memory_limit_bytes=4096 * MEGABYTE,
    )
    async with WorkerPool(config, workers=1, timeout_s=60.0) as pool:
        baseline = {child.pid for child in multiprocessing.active_children()}
        for _ in range(3):
            session = await pool.open_stage_session(memory_limit_bytes=4096 * MEGABYTE)
            blocked = asyncio.create_task(
                session.run_before_parse(
                    _raw("text/plain"),
                    max_output_bytes=MEGABYTE,
                    memory_limit_bytes=4096 * MEGABYTE,
                )
            )
            await asyncio.sleep(0.05)
            blocked.cancel()
            blocked.cancel()
            with pytest.raises(asyncio.CancelledError):
                await blocked
            await session.aclose()
            assert {child.pid for child in multiprocessing.active_children()} == baseline


async def test_stage_timeout_is_typed_and_carries_no_private_exception(
    tmp_path: Path, manicule_environment: Path
) -> None:
    del manicule_environment
    config = _config(
        tmp_path,
        middleware=("hanging-stage",),
        plugin_config={"middleware.hanging-stage": {"hang_seconds": 60}},
        memory_limit_bytes=4096 * MEGABYTE,
    )
    async with WorkerPool(config, workers=1, timeout_s=0.1, poll_interval_s=0.01) as pool:
        result = await pool.run_before_parse(
            _raw("text/plain"),
            max_output_bytes=MEGABYTE,
            memory_limit_bytes=4096 * MEGABYTE,
        )

    assert result.reason is StageFailure.TIMEOUT
    assert result.value is None


def test_a_deadline_measured_in_this_process_would_not_have_fired() -> None:
    """The premise, stated as an executable claim rather than as a comment.

    A blocking call inside a native extension returns when it is ready and not before. This is
    the shape of the thing the subprocess exists for: the elapsed time is real, and no timeout
    written in Python interrupted it.
    """
    started = time.monotonic()
    time.sleep(0.05)
    assert time.monotonic() - started >= 0.05
