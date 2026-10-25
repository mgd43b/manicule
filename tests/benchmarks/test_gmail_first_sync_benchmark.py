"""A Gmail first sync costs the same per message at the end as at the start (issue #492)."""

from __future__ import annotations

from tests.benchmarks.gmail_first_sync import measure


async def test_acquisition_work_per_message_does_not_grow_with_the_journal() -> None:
    """Compares the first and last windows by SQLite steps, not time or statements.

    Before the fix the last window cost several times the first: each retained blob was admitted
    by a scan of every blob retained before it, while the statement count per message stayed
    flat. The bound is loose enough for the journal's own growth, which is logarithmic.
    """
    measured = await measure(messages=1_000, window=200, acquire_only=True)

    assert measured.error == ""
    assert len(measured.windows) >= 4
    first, last = measured.windows[0], measured.windows[-1]
    assert first.steps > 0
    assert last.steps < first.steps * 1.5, (first, last)
