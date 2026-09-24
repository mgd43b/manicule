"""The in-process rate limiter: exact boundaries, a fake clock, and the LRU bound.

Every test here drives a fake clock by hand rather than sleeping, on
``manicule.app.throttle``'s own reasoning: "refill after the computed interval" is an assertion
against an exact number, not a timing-sensitive guess.
"""

from __future__ import annotations

import pytest

from manicule.app.caller import Caller
from manicule.app.throttle import KeyedLimiter, RateLimiter, TokenBucket, caller_key
from manicule.config.settings import RateLimitSettings


class FakeClock:
    """A clock that only moves when a test tells it to."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# --- TokenBucket ---------------------------------------------------------------------------


def test_exactly_burst_requests_succeed_then_the_next_is_refused() -> None:
    """The boundary the whole feature is named after: burst is a hard ceiling, not a suggestion."""
    bucket = TokenBucket(capacity=3, refill_per_second=0, now=0.0)
    assert bucket.take(0.0) is True
    assert bucket.take(0.0) is True
    assert bucket.take(0.0) is True
    assert bucket.take(0.0) is False, "a fourth request in the same instant must be refused"


def test_a_bucket_refills_after_exactly_the_computed_interval() -> None:
    """Refill is linear in elapsed time, and the boundary is exact rather than approximate."""
    bucket = TokenBucket(capacity=1, refill_per_second=2.0, now=0.0)
    assert bucket.take(0.0) is True
    assert bucket.take(0.4) is False, "half the needed interval must not be enough"
    assert bucket.retry_after(0.4) == pytest.approx(0.1), (
        "0.1s remaining at a 2 token/s refill rate"
    )
    assert bucket.take(0.5) is True, "exactly the computed interval must refill the token"


def test_a_bucket_never_holds_more_than_its_capacity() -> None:
    """Refilling across a long idle gap does not let a caller bank tokens past the ceiling."""
    bucket = TokenBucket(capacity=2, refill_per_second=100.0, now=0.0)
    bucket.take(0.0)
    assert bucket.available(1000.0, cost=2) is True
    assert bucket.available(1000.0, cost=3) is False


def test_a_clock_that_does_not_move_changes_nothing() -> None:
    """Two calls in the same instant must not refill between them."""
    bucket = TokenBucket(capacity=1, refill_per_second=1000.0, now=5.0)
    bucket.take(5.0)
    assert bucket.take(5.0) is False


# --- KeyedLimiter ----------------------------------------------------------------------------


def test_a_per_key_override_wins_over_the_limiters_default() -> None:
    """An ``ApiKey.rate_limit`` replaces the limiter's default for that key alone."""
    clock = FakeClock()
    limiter = KeyedLimiter(per_minute=600, burst=600, max_tracked=10, clock=clock)
    # The default burst would admit far more than 2; the override caps this key at 2.
    assert limiter.take("key:capped", burst=2).allowed is True
    assert limiter.take("key:capped", burst=2).allowed is True
    assert limiter.take("key:capped", burst=2).allowed is False
    # A different key, using the limiter's own default, is unaffected by the first key's cap.
    assert limiter.take("key:default").allowed is True


def test_the_lru_bound_holds() -> None:
    """Tracking never exceeds ``max_tracked``, however many distinct keys are seen."""
    clock = FakeClock()
    limiter = KeyedLimiter(per_minute=60, burst=10, max_tracked=3, clock=clock)
    for index in range(10):
        limiter.take(f"key-{index}")
    assert limiter.tracked == 3


def test_eviction_takes_the_least_recently_touched_key() -> None:
    """Touching a key keeps it warm, so eviction is LRU rather than insertion order."""
    clock = FakeClock()
    limiter = KeyedLimiter(per_minute=60, burst=10, max_tracked=2, clock=clock)
    limiter.take("a")
    limiter.take("b")
    limiter.take("a")  # "a" is now more recently touched than "b"
    limiter.take("c")  # forces an eviction: "b" goes, not "a"
    assert limiter.tracked == 2
    # "a" still has its original bucket state (partially spent, not reset by eviction+recreate).
    fresh = KeyedLimiter(per_minute=60, burst=10, max_tracked=2, clock=clock)
    fresh.take("a")
    assert fresh.tracked == 1


# --- RateLimiter -------------------------------------------------------------------------------


def test_disabled_means_unlimited() -> None:
    """``enabled = false`` reports every request admitted, without touching a bucket."""
    limiter = RateLimiter(RateLimitSettings(enabled=False, burst=1, per_minute=1))
    for _ in range(50):
        assert limiter.charge_caller("addr:anyone").allowed is True
        assert limiter.charge_failed_auth("addr:anyone").allowed is True


def test_failed_authentications_are_admitted_up_to_the_budget_and_refused_after() -> None:
    """The address's allowance of failures is exactly ``failed_auth_per_minute``.

    Each failure is charged as it happens; the one past the budget is refused with how long to
    wait, and a refill after that wait admits one more.
    """
    clock = FakeClock()
    limiter = RateLimiter(
        RateLimitSettings(enabled=True, failed_auth_per_minute=2, per_minute=600, burst=600),
        clock=clock,
    )
    address = "203.0.113.5"
    assert limiter.charge_failed_auth(address).allowed is True
    assert limiter.charge_failed_auth(address).allowed is True
    refused = limiter.charge_failed_auth(address)
    assert refused.allowed is False
    assert refused.retry_after_s > 0
    clock.advance(refused.retry_after_s)
    assert limiter.charge_failed_auth(address).allowed is True


def test_charge_caller_uses_the_override_as_both_rate_and_capacity() -> None:
    """A key's own ``rate_limit`` caps its burst too, not only its refill rate.

    Otherwise a key limited to one request per minute could still spend the installation's
    whole default burst the moment its bucket happened to be full, which would make a low
    per-key limit close to decorative — see the docstring of
    :meth:`~manicule.app.throttle.RateLimiter.charge_caller`.
    """
    clock = FakeClock()
    limiter = RateLimiter(RateLimitSettings(enabled=True, per_minute=6000, burst=6000), clock=clock)
    assert limiter.charge_caller("key:capped", rate_limit=1).allowed is True
    assert limiter.charge_caller("key:capped", rate_limit=1).allowed is False
    # An unrelated key is unaffected and still gets the installation's own generous default.
    assert limiter.charge_caller("key:uncapped").allowed is True


def test_caller_key_prefers_the_key_then_the_user_then_the_address() -> None:
    assert caller_key(Caller(key_id="k1", user_id="u1", address="a1")) == "key:k1"
    assert caller_key(Caller(user_id="u1", address="a1")) == "user:u1"
    assert caller_key(Caller(address="a1")) == "addr:a1"
    assert caller_key(Caller()) == "addr:unknown"
