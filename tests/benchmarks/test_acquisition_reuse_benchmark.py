"""Keep the full-size acquisition reuse evidence program executable in the normal suite."""

from __future__ import annotations

from tests.benchmarks.acquisition_reuse import measure


async def test_exact_reuse_measurement_is_bounded_and_deduplicates_shared_blobs() -> None:
    measured = await measure(records=16, workers=8)

    assert measured.acquired == measured.reused == measured.records == 16
    assert measured.backlog_bytes > 0
    assert measured.statements <= measured.records * 12 + 3
    assert measured.peak_rss_bytes > 0
