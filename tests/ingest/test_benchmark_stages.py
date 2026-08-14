"""The staged-ingest benchmark still runs, and still measures what it claims to.

A benchmark nothing exercises is a script that stops working and says nothing, which is the
same failure as a document that stops describing the code. So this runs it small and asserts on
the *invariants* it reports — the stage widths, the serialized model, the bounded memory — and
never on a wall time. Timings are what a benchmark is for and what a test must not assert, since
a shared runner will produce whatever number it likes.
"""

from __future__ import annotations

import pytest
from tools.benchmark_ingest_stages import TOO_SMALL_TO_COMPARE_S, Measurement, corpus, measure

SMALL = {
    "documents": 8,
    "fetch_concurrency": 3,
    "parse_workers": 2,
    "latency_s": 0.002,
    "parse_s": 0.001,
    "embed_s": 0.001,
}


async def _measure(scenario: str, strategy: str) -> Measurement:
    return await measure(scenario, strategy, **SMALL)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("scenario", ["first", "unchanged", "partial"])
async def test_each_scenario_runs_both_strategies_over_the_same_corpus(scenario: str) -> None:
    """The three scenarios cost different things, and each must reach the same corpus state.

    A benchmark whose two strategies indexed different numbers of documents would be comparing
    two different amounts of work and reporting the difference as a speedup.
    """
    sequential = await _measure(scenario, "sequential")
    staged = await _measure(scenario, "staged")

    assert sequential.discovered == staged.discovered == 8
    assert sequential.indexed == staged.indexed
    assert sequential.skipped == staged.skipped
    assert sequential.embed_batches == staged.embed_batches, (
        "the two strategies asked the model for different amounts of work"
    )


async def test_the_benchmark_observes_the_stage_widths_it_reports() -> None:
    """The measured run is concurrent, or the numbers beside it describe nothing."""
    staged = await _measure("first", "staged")

    assert staged.peak_fetches > 1, "the benchmark reported a sequential run as a staged one"
    assert staged.peak_parses > 1
    assert staged.peak_fetches <= SMALL["fetch_concurrency"]
    assert staged.peak_parses <= SMALL["parse_workers"] + 1


async def test_the_benchmark_would_notice_a_concurrent_model_call() -> None:
    """Its embedder raises on overlap, so a speedup bought that way cannot be reported."""
    for strategy in ("sequential", "staged"):
        measurement = await _measure("first", strategy)
        assert measurement.peak_embeds == 1


async def test_the_benchmark_reports_a_bounded_memory_proxy_rather_than_the_corpus() -> None:
    """Retained bodies, bounded by the hand-offs, rather than a resident-bytes reading.

    The staged run holds at most the parse hand-off plus everyone carrying a body: with three
    fetch workers, three ingest workers and a hand-off of six, that is twelve — comfortably under
    the eight-document corpus only because the corpus is small, which is why the assertion is
    against the derived bound and not against the corpus size.
    """
    staged = await _measure("first", "staged")
    ingest_workers = SMALL["parse_workers"] + 1
    bound = 2 * ingest_workers + ingest_workers + SMALL["fetch_concurrency"]

    assert 0 < staged.peak_retained_documents <= bound
    assert staged.peak_traced_kib > 0, "tracemalloc measured nothing, so the column is a zero"


async def test_an_unchanged_sync_touches_neither_the_source_nor_the_model() -> None:
    """The scenario that exists to show what change detection is worth.

    Level-1 detection answers before the fetch, so a corpus that did not move costs a store read
    per document. A benchmark that reported forward passes here would be reporting that change
    detection had stopped working.
    """
    staged = await _measure("unchanged", "staged")

    assert staged.skipped == 8
    assert staged.indexed == 0
    assert staged.embed_batches == 0
    assert staged.peak_fetches == 0


async def test_a_partial_sync_re_embeds_only_what_moved() -> None:
    """Every tenth document is edited, so a corpus of eight has exactly one."""
    staged = await _measure("partial", "staged")

    assert staged.indexed == 1
    assert staged.skipped == 7
    assert staged.embed_batches == 1


def test_the_corpus_is_invented_and_shaped_the_way_the_benchmark_says() -> None:
    """Eight paragraphs per document, so a document is eight chunks and a batch is worth having."""
    pages = corpus(3)

    assert len(pages) == 3
    assert all(len(text.splitlines()) == 8 for text in pages.values())
    assert all("invented page" in text for text in pages.values())


def test_the_too_small_threshold_is_above_what_a_scheduler_costs() -> None:
    """The guard that stops a millisecond ratio being printed as a speedup.

    Set below any run that does real work and above any run that does not, so the report says
    "scheduling noise" for the unchanged scenario rather than a number somebody quotes.
    """
    assert 0.005 < TOO_SMALL_TO_COMPARE_S < 1.0
