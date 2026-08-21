"""The benchmark is executed by the suite, at a size that costs nothing.

A benchmark nobody runs is a script that stops working and says nothing, and this one is the
evidence for the claim the whole feature rests on: that checking a stored vector's numbers is
affordable enough to do routinely, which re-embedding is not. So the suite runs it — small, so
it is free — and asserts the shape of the result rather than its magnitude.

Three properties are worth a test. The phases stay separated, so "the checksum is slow" can
never be a misreading of "reading a million vectors off disk is slow". The scan is bounded by
its page size rather than by the table. And validation never reaches an embedder — measured
against a stand-in that raises, so the day somebody adds a fallback this fails rather than
quietly getting slower.

The magnitudes in ``docs/storage.md`` §6.2.5 come from the default size, which is deliberately
large enough to swamp per-call overhead. This is the guard that the program still works, not a
re-taking of the published numbers.
"""

from __future__ import annotations

import pytest

from tests.benchmarks.vector_checksum import RefusingEmbedder, measure, render

ROWS = 256
DIMENSION = 64


async def test_every_phase_is_measured_separately_and_none_of_them_embeds() -> None:
    """The three numbers the design argument needs, kept apart."""
    measurements = await measure(rows=ROWS, dimension=DIMENSION)

    assert [measurement.phase for measurement in measurements] == ["hash", "verify", "scan"]
    assert all(measurement.rows == ROWS for measurement in measurements)
    assert all(measurement.rows_per_second > 0 for measurement in measurements)
    assert all(measurement.bytes_per_second > 0 for measurement in measurements)
    assert all(measurement.peak_rss_bytes > 0 for measurement in measurements)
    assert all(measurement.embedder_calls == 0 for measurement in measurements), (
        "validating a checksum reached an embedder. The entire argument for a checksum is that "
        "it costs a hash rather than a forward pass"
    )
    assert measurements[-1].checksum_coverage == 1.0


async def test_the_storage_scan_is_the_slower_of_the_two_and_that_is_the_point() -> None:
    """The subtraction the report exists to make possible.

    ``scan`` pays LanceDB's read and deserialization on top of the same verification ``verify``
    performs alone. If the two were ever reported as one number, an operator looking at a slow
    validation could not tell whether to change the checksum or the storage layout — and only
    one of those is a thing anybody can act on.
    """
    measurements = {
        measurement.phase: measurement
        for measurement in await measure(rows=ROWS, dimension=DIMENSION)
    }

    assert measurements["scan"].seconds >= measurements["verify"].seconds


async def test_the_refusing_embedder_would_actually_fail_if_something_called_it() -> None:
    """The instrument, checked. A stand-in that silently passed would prove nothing."""
    with pytest.raises(AssertionError, match="forward pass"):
        await RefusingEmbedder(DIMENSION).embed(["anything"])


async def test_the_rendering_lines_up_and_fits_a_terminal() -> None:
    """It is pasted into pull requests, so it has to survive being pasted."""
    rendered = render(await measure(rows=ROWS, dimension=DIMENSION)).splitlines()

    assert len({len(line) for line in rendered}) == 1, (
        "every row ends in the same column, which is what makes three phases comparable by eye"
    )
    assert all(len(line) <= 80 for line in rendered)
    assert "embeds" in rendered[0], "the zero that matters most has to be visible in the header"
