"""The process boundary, and the two failures it exists to survive.

These start real subprocesses. They are slower than everything else in the suite and there is
no alternative: a worker that ran in this process would not be a worker, and the properties
under test — a deadline that fires against native code, a memory bound that carries a
``SIGKILL`` — are exactly the ones an in-process double cannot have.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from manicule_plugin_example import MEDIA_TYPE as EXAMPLE_MEDIA_TYPE
from manicule_plugin_hostile import (
    CRASHING_MEDIA_TYPE,
    GREEDY_MEDIA_TYPE,
    HANGING_MEDIA_TYPE,
)

from manicule.core.content import DocumentStatus, RawDocument
from manicule.ingest.limits import (
    ADDRESS_SPACE_HEADROOM,
    limit_address_space,
    resident_bytes,
)
from manicule.ingest.workers import (
    MEGABYTE,
    InProcessRunner,
    WorkerConfig,
    WorkerPool,
    default_worker_count,
)
from manicule.parsers.chain import Outcome, run_chain
from tests.ingest import fakes

pytestmark = pytest.mark.slow


def _config(tmp_path: Path, **overrides: object) -> WorkerConfig:
    return WorkerConfig(
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        parser_fallbacks={},
        plugin_config={},
        **overrides,  # pyright: ignore[reportArgumentType] - test-local keyword passthrough
    )


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


async def test_a_parser_that_allocates_without_bound_is_killed(
    tmp_path: Path, manicule_environment: Path
) -> None:
    """A memory bound the kernel will not enforce on the platform this is built for.

    ``RLIMIT_AS`` reports unlimited on Darwin and refuses to be set, so a design that relied on
    it would be correct on CI and inert on the machine where the malformed document gets opened
    first. The parent samples resident memory and sends ``SIGKILL``, which works identically
    everywhere — so the outcome is the same on both platforms even though the mechanism is not.
    """
    del manicule_environment
    config = _config(tmp_path, memory_limit_bytes=64 * MEGABYTE)
    async with WorkerPool(config, workers=1, timeout_s=60.0, poll_interval_s=0.05) as pool:
        result = await pool.run_attempt("greedy", _raw(GREEDY_MEDIA_TYPE))

    assert result.attempt.outcome is Outcome.FAILED
    assert "worker killed" in result.attempt.reason


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


@pytest.mark.skipif(sys.platform != "darwin", reason="the measured Darwin behaviour")
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


def test_a_deadline_measured_in_this_process_would_not_have_fired() -> None:
    """The premise, stated as an executable claim rather than as a comment.

    A blocking call inside a native extension returns when it is ready and not before. This is
    the shape of the thing the subprocess exists for: the elapsed time is real, and no timeout
    written in Python interrupted it.
    """
    started = time.monotonic()
    time.sleep(0.05)
    assert time.monotonic() - started >= 0.05
