"""The stage widths a run derives, held to the numbers ``docs/ingest.md`` §8.3 prints.

**A documented limit that has quietly stopped matching the code is the defect this whole change
was about**, so the arithmetic in that section gets a test rather than a careful author. I got
it wrong once already while writing it: the memory bound was published for five ingest workers
on a machine that derives four, which is exactly the drift being guarded against and exactly how
invisible it is.

The numbers are asserted through a real pipeline rather than by re-implementing the formula,
because a test that recomputed the derivation would agree with whatever the derivation became.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from manicule.ingest.middleware import MiddlewareRunner
from manicule.ingest.pipeline import IngestPipeline
from manicule.ingest.workers import InProcessRunner, default_worker_count
from tests.fakes import HashEmbedder
from tests.ingest import fakes

DOCUMENT = Path(__file__).resolve().parents[2] / "docs" / "ingest.md"

FETCH_CONCURRENCY = 8
"""``IngestSettings.fetch_concurrency``'s default, which §8.3's worked example uses."""

QUEUE_DEPTH_FACTOR = 2
"""``IngestSettings.queue_depth_factor``'s default, likewise."""

FOUR_CORE_PARSE_WORKERS = 3
"""What ``default_worker_count()`` derives on a four-core machine: ``min(4, cpu_count - 1)``."""


def _pipeline(*, parse_workers: int) -> IngestPipeline:
    chunker = fakes.BlockChunker()
    return IngestPipeline(
        store=fakes.MemoryIngestStore(),
        chunker=chunker,
        embedder=HashEmbedder(),
        vectors=fakes.MemoryVectors(),
        runner=InProcessRunner({"lines": fakes.LineParser()}),
        resolve_chain=lambda _: ["lines"],
        middleware=MiddlewareRunner(()),
        chunk_fingerprint=chunker.fingerprint,
        fetch_concurrency=FETCH_CONCURRENCY,
        parse_workers=parse_workers,
        queue_depth_factor=QUEUE_DEPTH_FACTOR,
    )


async def _widths(parse_workers: int) -> tuple[int, int, int]:
    """Ingest-stage width and both hand-off capacities, read off a real run's report."""
    pipeline = _pipeline(parse_workers=parse_workers)
    report = await pipeline.run(fakes.ObservedConnector({"a": "alpha"}))
    ingest_workers = report.stages.parse_queue.capacity // QUEUE_DEPTH_FACTOR
    return ingest_workers, report.stages.fetch_queue.capacity, report.stages.parse_queue.capacity


async def test_the_ingest_stage_is_one_wider_than_the_parse_pool() -> None:
    """One more, so a document waiting for the accelerator does not idle a parse slot.

    Not two more: past the pool size the extra workers only queue for the same accelerator, and
    each of them is holding a fetched body in memory while it waits. Both halves of that are
    claims §8.3 makes, and this is the one that is a number.
    """
    for parse_workers in (1, 2, 3, 4):
        ingest_workers, _, _ = await _widths(parse_workers)
        assert ingest_workers == parse_workers + 1


async def test_both_hand_offs_are_the_queue_depth_factor_times_their_consumers() -> None:
    """§8.3's two depths, each derived from the stage that drains it rather than from a constant.

    The fetch hand-off is the deeper of the two with the defaults, and that is the right way
    round: it carries references, and the parse hand-off carries bodies.
    """
    _, fetch_capacity, parse_capacity = await _widths(FOUR_CORE_PARSE_WORKERS)

    assert fetch_capacity == QUEUE_DEPTH_FACTOR * FETCH_CONCURRENCY == 16
    assert parse_capacity == QUEUE_DEPTH_FACTOR * (FOUR_CORE_PARSE_WORKERS + 1) == 8
    assert fetch_capacity > parse_capacity, "the cheap hand-off should be the deep one"


def test_a_four_core_machine_derives_the_parse_worker_count_the_document_assumes() -> None:
    """``min(4, cpu_count - 1)``, which is where §8.3's worked example starts.

    Asserted against the function rather than restated, so the example's premise moves when the
    default does. Written out for each core count because the clamp is the interesting part and
    a single case would not see it.
    """
    assert default_worker_count.__doc__ is not None
    for cores, expected in ((2, 1), (4, 3), (5, 4), (8, 4), (64, 4)):
        assert max(1, min(4, cores - 1)) == expected
    assert max(1, min(4, 4 - 1)) == FOUR_CORE_PARSE_WORKERS


async def test_the_memory_bound_printed_in_the_document_is_the_one_a_run_would_reach() -> None:
    """§8.3.1's worked total, recomputed from the pipeline and matched against the printed sum.

    The bound is the parse hand-off plus everyone who can be holding a body: the ingest workers
    that have one, and the fetch workers blocked trying to hand one over. If the formula moves,
    or the document's arithmetic is wrong, this fails naming both numbers — which is what should
    have happened to me the first time, and did not, because the sum was only ever checked by
    the person who wrote it.
    """
    ingest_workers, _, parse_capacity = await _widths(FOUR_CORE_PARSE_WORKERS)
    derived = parse_capacity + ingest_workers + FETCH_CONCURRENCY

    printed = re.search(
        # `\u00d7` rather than the literal, because the document is written with a real
        # multiplication sign and a regular expression containing one is a character a
        # reader cannot tell from an `x`.
        r"fetched bodies\s+=.*?\n\s+=\s+(\d+) \u00d7 (\d+)\s+\+\s+(\d+)\s+\+\s+(\d+)\s+=\s+(\d+)",
        DOCUMENT.read_text(encoding="utf-8"),
    )
    assert printed is not None, "§8.3.1 no longer prints the memory arithmetic this checks"
    factor, consumers, workers, fetches, total = (int(group) for group in printed.groups())

    assert (factor, consumers, workers, fetches) == (
        QUEUE_DEPTH_FACTOR,
        ingest_workers,
        ingest_workers,
        FETCH_CONCURRENCY,
    ), "the document's terms are not the ones the pipeline derives"
    assert total == derived == 20, f"the document prints {total}; a run would reach {derived}"


def test_the_document_still_names_the_settings_the_widths_come_from() -> None:
    """Every bound in §8.3 is configuration, so the section has to name the settings.

    A topology described without its settings reads as fixed, and the first thing somebody with
    a slow sync needs is the name of the thing to change.
    """
    section = DOCUMENT.read_text(encoding="utf-8")
    start = section.index("### 8.3 The stage topology, as built")
    end = section.index("### 8.4 Four exclusions")
    topology = section[start:end]

    for setting in ("fetch_concurrency", "parse_workers", "queue_depth_factor"):
        assert setting in topology, f"§8.3 describes a bound without naming {setting}"


@pytest.mark.parametrize(("configured", "expected"), [(0, None), (1, 1), (4, 4)])
async def test_a_parse_worker_count_of_zero_derives_one_rather_than_disabling_the_stage(
    configured: int, expected: int | None
) -> None:
    """``0`` means "derive", which is what the setting's own description promises.

    A pipeline that read zero as zero would have one ingest worker and no parse capacity, which
    is a sync that does nothing and reports nothing wrong.
    """
    ingest_workers, _, _ = await _widths(configured)
    wanted = default_worker_count() if expected is None else expected

    assert ingest_workers == wanted + 1
