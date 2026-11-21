"""The Qdrant coverage benchmark is executed by the suite, at a size that costs nothing.

A benchmark nobody runs is a script that stops working and says nothing, and this one is the
evidence for a claim about latency: that counting checksum coverage on a server stops costing a
full scan of every payload. So the suite runs it — small, against a throwaway server it is given,
skipping where there is none — and asserts the shape of what comes back rather than its
magnitude. A size this small cannot show the speedup, and a test that tried to would be a flake.

What is tested without a server is the part that has to be right for the numbers to mean
anything and for the program to be safe to point at a shared one: the percentile arithmetic, the
planted payloads, and the gate that refuses to touch a collection the run did not make.

CI's Qdrant job runs only ``tests/test_storage_qdrant.py`` and
``tests/test_storage_vector_migration.py``, so this module is skipped there by construction. The
correctness of the counts is held by the first of those; this guards the program.
"""

from __future__ import annotations

import pytest
from qdrant_client import AsyncQdrantClient

from manicule.core.embedding import VECTOR_CHECKSUM_VERSION
from manicule.storage.vector_schema import CHECKSUM_COLUMN, CHECKSUM_VERSION_COLUMN, CHUNK_COLUMN
from tests.benchmarks.qdrant_checksum_coverage import (
    ARMS,
    ArmResult,
    Responsiveness,
    Run,
    Truth,
    measure,
    owned,
    parser,
    payload_for,
    percentile,
    render,
)
from tests.qdrant_support import TEST_COLLECTION_PREFIX, remote_client, require_server


def test_a_percentile_is_a_sample_that_happened() -> None:
    """Nearest-rank, so a p99 over a few hundred samples is never a number between two."""
    samples = [float(value) for value in range(1, 101)]

    assert percentile(samples, 0.50) == 50.0
    assert percentile(samples, 0.95) == 95.0
    assert percentile(samples, 0.99) == 99.0
    assert percentile([7.0], 0.99) == 7.0
    assert percentile(list(reversed(samples)), 0.50) == 50.0, "order must not matter"
    with pytest.raises(ValueError, match="at least one sample"):
        percentile([], 0.5)


def test_the_planted_payloads_are_sized_like_chunks_and_record_what_they_say() -> None:
    """Representative weight, because a count that reads payloads costs what they weigh."""
    import random  # noqa: PLC0415 - only this test needs a generator of its own

    generator = random.Random(1)  # noqa: S311 - synthetic data
    recorded = payload_for(1, generator, kind="recorded", payload_bytes=(1024, 8192))
    checksum_only = payload_for(2, generator, kind="checksum-only", payload_bytes=(1024, 8192))
    version_only = payload_for(3, generator, kind="version-only", payload_bytes=(1024, 8192))
    neither = payload_for(4, generator, kind="neither", payload_bytes=(1024, 8192))

    assert recorded[CHECKSUM_VERSION_COLUMN] == VECTOR_CHECKSUM_VERSION
    assert CHECKSUM_COLUMN in recorded
    assert CHECKSUM_COLUMN in checksum_only
    assert CHECKSUM_VERSION_COLUMN not in checksum_only
    assert CHECKSUM_VERSION_COLUMN in version_only
    assert CHECKSUM_COLUMN not in version_only
    assert CHECKSUM_COLUMN not in neither
    assert CHECKSUM_VERSION_COLUMN not in neither
    sizes = [
        len(str(point[CHUNK_COLUMN])) for point in (recorded, checksum_only, version_only, neither)
    ]
    assert all(1024 <= size <= 8192 + 64 for size in sizes), sizes


def test_the_program_will_not_touch_a_collection_it_did_not_make() -> None:
    """The gate in front of every deletion: minted by this run, and under the test prefix."""
    minted = [f"{TEST_COLLECTION_PREFIX}_abc_chunks__deadbeef"]

    assert owned(minted[0], minted) == minted[0]
    with pytest.raises(ValueError, match="did not make"):
        owned(f"{TEST_COLLECTION_PREFIX}_someone_else", minted)
    with pytest.raises(ValueError, match="did not make"):
        owned("manicule_chunks_default", ["manicule_chunks_default"])


def test_the_command_line_has_no_way_to_name_a_collection() -> None:
    """A switch that took a collection is a switch that could be pointed at a real one."""
    options = {option for action in parser()._actions for option in action.option_strings}

    assert not {option for option in options if "collection" in option or "prefix" in option}
    assert "--points" in options


def test_the_rendering_names_every_arm_and_both_responsiveness_phases() -> None:
    """Pasted into pull requests, so it has to read without the program."""
    sample = Responsiveness(
        search_p50_ms=1.0, search_p99_ms=2.0, ping_p50_ms=0.5, ping_p99_ms=0.9, samples=10
    )
    arms = tuple(
        ArmResult(
            arm=arm,
            requests=4 if arm == "candidate-two-branch" else 3,
            half_written_p50_ms=10.0 / (index + 1),
            half_written_p95_ms=12.0 / (index + 1),
            coverage_p50_ms=30.0 / (index + 1),
            coverage_p95_ms=35.0 / (index + 1),
            repeats=3,
            idle=sample,
            busy=sample,
            background_passes=4,
            busy_window_epoch=(1.0, 2.0),
        )
        for index, arm in enumerate(ARMS)
    )
    run = Run(
        collection=f"{TEST_COLLECTION_PREFIX}_x",
        points=100,
        dimension=8,
        payload_bytes_mean=2048,
        index_build_seconds=0.25,
        during_index_build=sample,
        truth=Truth(rows=100, unrecorded=3, half_written=6),
        arms=arms,
    )

    rendered = render(run).splitlines()

    for arm in ARMS:
        rows = [
            line for line in rendered if line.startswith(arm) and line.endswith(("idle", "busy"))
        ]
        assert len(rows) == 2, arm
    assert sum(line.endswith("idle") for line in rendered) == 3
    assert sum(line.endswith("busy") for line in rendered) == 3
    assert all(len(line) <= 120 for line in rendered)
    assert any(line.startswith("baseline over shipped") for line in rendered)
    assert any(line.startswith("shipped over candidate-two-branch") for line in rendered)


def test_the_arms_are_named_for_what_they_are() -> None:
    """The unshipped alternative is labelled as one, so a table cannot be read as a result."""
    assert ARMS == ("baseline", "shipped", "candidate-two-branch")


async def test_a_run_reports_every_arm_exactly_and_removes_only_what_it_made() -> None:
    """Against a throwaway server, small: the arms agree with what was planted, and it cleans up.

    ``measure`` refuses to return an arm whose counts differ from what it planted, so reaching
    the assertions is already the exactness check. What is asserted here is the shape — three
    arms in order, three requests for the baseline and for what is shipped and four for the
    candidate that is not, monotone percentiles, a background that ran —
    and that no collection outside the test prefix was created, changed or removed, and none
    this run made is left.
    """
    url = require_server()
    observer: AsyncQdrantClient = remote_client(url)
    try:
        before = {item.name for item in (await observer.get_collections()).collections}
        phases: list[tuple[str, str, str]] = []

        run = await measure(
            url=url,
            points=240,
            dimension=8,
            payload_bytes=(1024, 2048),
            repeats=3,
            probes=10,
            interval=0.0,
            min_busy_seconds=0.5,
            on_phase=lambda arm, phase, collection: phases.append((arm, phase, collection)),
        )

        after = {item.name for item in (await observer.get_collections()).collections}
        assert [result.arm for result in run.arms] == list(ARMS)
        assert [result.requests for result in run.arms] == [3, 3, 4]
        assert run.truth == Truth(rows=240, unrecorded=3, half_written=6)
        assert run.collection.startswith(TEST_COLLECTION_PREFIX)
        assert run.index_build_seconds >= 0
        assert run.during_index_build.samples >= 1
        for result in run.arms:
            assert result.half_written_p50_ms <= result.half_written_p95_ms
            assert result.coverage_p50_ms <= result.coverage_p95_ms
            assert result.half_written_p50_ms <= result.coverage_p50_ms
            assert result.idle.samples >= 10
            assert result.busy.samples >= 10
            assert result.busy.search_p50_ms <= result.busy.search_p99_ms
            assert result.background_passes >= 1
            assert result.busy_window_epoch[0] < result.busy_window_epoch[1]
        assert {collection for _, _, collection in phases} == {run.collection}
        assert run.collection not in after, "the run left its collection behind"
        assert {name for name in after if not name.startswith(TEST_COLLECTION_PREFIX)} == {
            name for name in before if not name.startswith(TEST_COLLECTION_PREFIX)
        }, "a collection outside the test prefix changed"
    finally:
        await observer.close()
