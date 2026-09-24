"""Security alert detection: exact thresholds, a fake clock, and the once-per-window rule."""

from __future__ import annotations

from manicule.app.alerts import AlertMonitor
from manicule.config.settings import AlertSettings


class FakeClock:
    """A clock that only moves when a test tells it to."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _monitor(**overrides: object) -> tuple[AlertMonitor, FakeClock]:
    clock = FakeClock()
    settings = AlertSettings(**overrides)  # pyright: ignore[reportArgumentType] - test overrides
    return AlertMonitor(settings, max_tracked=100, clock=clock), clock


def test_disabled_detects_nothing() -> None:
    monitor, _clock = _monitor(enabled=False, failed_auth_threshold=1)
    assert monitor.record_failed_auth("203.0.113.5") is None


def test_brute_force_fires_at_the_threshold_and_not_one_short() -> None:
    """``>=`` is the rule: one short of the threshold is silence, exactly at it is an alert."""
    monitor, _clock = _monitor(failed_auth_threshold=3, window_s=60)
    address = "203.0.113.5"
    assert monitor.record_failed_auth(address) is None
    assert monitor.record_failed_auth(address) is None
    event = monitor.record_failed_auth(address)
    assert event is not None
    assert event.kind == "brute_force"
    assert event.subject == address
    assert event.details["count"] == 3


def test_an_alert_fires_at_most_once_per_window() -> None:
    monitor, clock = _monitor(failed_auth_threshold=1, window_s=60)
    address = "203.0.113.7"
    first = monitor.record_failed_auth(address)
    assert first is not None
    # Still within the window: the pattern continues, but the alert does not repeat.
    clock.advance(30)
    assert monitor.record_failed_auth(address) is None
    # Past the window: the same subject may fire again.
    clock.advance(31)
    second = monitor.record_failed_auth(address)
    assert second is not None


def test_key_abuse_counts_distinct_addresses_not_total_presentations() -> None:
    """A key presented ten times from the one address it always uses is not shared or leaked."""
    monitor, _clock = _monitor(key_address_threshold=3)
    key_id = "key-1"
    for _ in range(10):
        assert monitor.record_key_presentation(key_id, "203.0.113.1") is None
    assert monitor.record_key_presentation(key_id, "203.0.113.2") is None
    event = monitor.record_key_presentation(key_id, "203.0.113.3")
    assert event is not None
    assert event.kind == "key_abuse"
    assert event.subject == key_id
    assert event.details["addresses"] == 3


def test_export_volume_counts_distinct_documents_not_total_reads() -> None:
    """Re-reading the one document a caller always reads must never look like an export."""
    monitor, _clock = _monitor(export_document_threshold=2)
    actor = "user-1"
    for _ in range(10):
        assert monitor.record_document_read(actor, "doc-1") is None
    event = monitor.record_document_read(actor, "doc-2")
    assert event is not None
    assert event.kind == "export_volume"
    assert event.subject == actor
    assert event.details["documents"] == 2


def test_rate_limited_refusals_of_one_key_repeated_are_key_abuse() -> None:
    monitor, _clock = _monitor(failed_auth_threshold=2)
    key_id = "key-2"
    assert monitor.record_rate_limited(key_id) is None
    event = monitor.record_rate_limited(key_id)
    assert event is not None
    assert event.kind == "key_abuse"
    assert event.subject == key_id


def test_entries_outside_the_window_are_evicted() -> None:
    """A pattern that stopped is not a pattern: old occurrences must not count toward a new one."""
    monitor, clock = _monitor(failed_auth_threshold=3, window_s=60)
    address = "203.0.113.11"
    monitor.record_failed_auth(address)
    monitor.record_failed_auth(address)
    clock.advance(61)  # both above fall out of the window
    assert monitor.record_failed_auth(address) is None, "only one recent occurrence, not three"


def test_memory_is_bounded_by_max_tracked() -> None:
    """The least-recently-touched subject's history is forgotten past ``max_tracked``.

    Proven through detection behavior rather than a private attribute: a subject evicted to
    bound memory has to build its pattern up again from nothing on its next occurrence, rather
    than being remembered forever.
    """
    clock = FakeClock()
    settings = AlertSettings(failed_auth_threshold=2, window_s=1000)
    monitor = AlertMonitor(settings, max_tracked=2, clock=clock)
    monitor.record_failed_auth("addr-a")  # one of two toward the threshold
    monitor.record_failed_auth("addr-b")  # both "a" and "b" now tracked
    monitor.record_failed_auth("addr-c")  # a third subject evicts the least-recently-touched: "a"
    # "a"'s one recorded failure is gone, so one more is not yet the threshold of two.
    assert monitor.record_failed_auth("addr-a") is None


def test_a_document_read_again_counts_from_its_latest_read() -> None:
    """A distinct value is dated by when it was last seen, not when it was first seen.

    Two documents read at the start of the window, one of them read again near its end: once
    the window has moved past the first reads, only the re-read document is still counted —
    and reading one document a thousand times is still one document.
    """
    monitor, clock = _monitor(window_s=60, export_document_threshold=3)
    monitor.record_document_read("user:ada", "doc-a")
    monitor.record_document_read("user:ada", "doc-b")
    clock.advance(50)
    for _ in range(1000):
        assert monitor.record_document_read("user:ada", "doc-a") is None
    clock.advance(20)
    # doc-b's only read is now 70 s old and out of the window; doc-a's latest is 20 s old.
    assert monitor.record_document_read("user:ada", "doc-c") is None
    clock.advance(1)
    assert monitor.record_document_read("user:ada", "doc-d") is not None
