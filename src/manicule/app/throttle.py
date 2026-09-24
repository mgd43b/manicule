"""In-process rate limiting: one token bucket per caller, memory bounded by how many are kept.

Pure Python, no dependency — the whole feature is a handful of floats and a dict, and a
dependency for that would be a bigger liability than the code it replaced.

**One bucket per caller, and a separate, much smaller one per address for failed
authentication.** :class:`~manicule.config.settings.RateLimitSettings` is the specification;
this is the mechanism it describes. The caller bucket meters ordinary traffic — refilled at
``per_minute`` tokens per minute, holding at most ``burst`` at once — and a network surface
charges it once per admitted request. The failed-auth bucket meters *guessing*: it is consulted
before a presented credential is checked at all, so an address that has already exhausted it is
refused before the database does a single hash comparison, and it is only ever charged when a
presented credential turns out not to work.

**Every bucket lives in a bounded map.** ``max_tracked`` entries are kept, least-recently-touched
evicted first, so an installation cannot be made to leak memory by presenting a fresh identity —
a random address, a made-up key id — on every request. Losing an entry to eviction is
indistinguishable from a caller nobody has seen in a while starting fresh, which is the correct
behavior for a rate limiter: it protects against a *sustained* pattern, not a single burst that
happened to arrive after a quiet spell.

**The clock is injectable.** Every method that would otherwise read the wall clock takes it from
a ``Clock`` this object was built with, which defaults to :func:`time.monotonic` — monotonic
rather than wall-clock, because a limiter must not be fooled by an NTP step. Tests pass a fake
that they advance by hand, so "refill after the computed interval" is an assertion rather than a
``sleep``.

**Safe for a single event loop, and for more than one thread.** Nothing here ``await``s, so a
coroutine holds the lock for the whole of a bucket's read-modify-write and another coroutine on
the same loop never observes a half-updated bucket even without the lock; the lock exists for
the case a limiter is shared with a thread outside the loop — the command line's own dispatch,
or a test — where the GIL alone would still make individual dict operations atomic but not the
combination of *check, refill, decrement* this performs.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from manicule.app.caller import Caller
    from manicule.config.settings import RateLimitSettings


class Clock(Protocol):
    def __call__(self) -> float: ...


@dataclass(frozen=True, slots=True)
class RateDecision:
    """Whether a request may proceed, and how long to wait if it may not."""

    allowed: bool
    retry_after_s: float = 0.0


class TokenBucket:
    """One caller's allowance: ``capacity`` tokens, refilled at ``refill_per_second``.

    Refill is computed lazily, from the elapsed time since the bucket was last touched, rather
    than by a background task — there is no background task, and a bucket nobody has asked about
    in an hour costs nothing until it is asked about again.
    """

    __slots__ = ("capacity", "refill_per_second", "tokens", "updated_at")

    def __init__(self, *, capacity: float, refill_per_second: float, now: float) -> None:
        self.capacity = capacity
        self.refill_per_second = refill_per_second
        self.tokens = capacity
        self.updated_at = now

    def refill(self, now: float) -> None:
        elapsed = now - self.updated_at
        if elapsed <= 0:
            # A clock that went backwards, or two calls in the same instant. Neither refills
            # negative time, and neither is treated as an error: a fake clock in a test that
            # has not been advanced yet must see the bucket exactly as it was.
            return
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_second)
        self.updated_at = now

    def take(self, now: float, *, cost: float = 1.0) -> bool:
        """Consume ``cost`` tokens if there are enough, and report whether it did."""
        self.refill(now)
        if self.tokens < cost:
            return False
        self.tokens -= cost
        return True

    def available(self, now: float, *, cost: float = 1.0) -> bool:
        """Whether ``cost`` tokens could be taken right now, without taking them."""
        self.refill(now)
        return self.tokens >= cost

    def retry_after(self, now: float, *, cost: float = 1.0) -> float:
        """Seconds until ``cost`` tokens would be available. ``0`` if they already are."""
        self.refill(now)
        deficit = cost - self.tokens
        if deficit <= 0:
            return 0.0
        if self.refill_per_second <= 0:
            return float("inf")
        return deficit / self.refill_per_second


class KeyedLimiter:
    """One :class:`TokenBucket` per key, bounded by ``max_tracked`` with LRU eviction.

    ``per_minute`` and ``burst`` are this limiter's defaults; a call may override either — the
    per-key rate an :class:`~manicule.storage.models.ApiKey` carries replaces
    ``security.rate_limit.per_minute`` for that key alone, and the override is read at call
    time rather than baked into the bucket, so a key's own limit changing takes effect on its
    very next request rather than waiting for its bucket to be evicted.
    """

    def __init__(
        self,
        *,
        per_minute: int,
        burst: int,
        max_tracked: int,
        clock: Clock = time.monotonic,
    ) -> None:
        self._per_minute = per_minute
        self._burst = burst
        self._max_tracked = max_tracked
        self._clock = clock
        self._buckets: OrderedDict[str, TokenBucket] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def tracked(self) -> int:
        """How many distinct keys currently hold a bucket. Never more than ``max_tracked``."""
        return len(self._buckets)

    def _bucket(
        self, key: str, *, per_minute: int | None, burst: int | None, now: float
    ) -> TokenBucket:
        capacity = float(burst if burst is not None else self._burst)
        refill = (per_minute if per_minute is not None else self._per_minute) / 60.0
        existing = self._buckets.get(key)
        if existing is not None:
            self._buckets.move_to_end(key)
            if existing.capacity != capacity or existing.refill_per_second != refill:
                # Refilled at the old rate up to now, then held to the new terms from now on,
                # with no more in hand than the new capacity allows.
                existing.refill(now)
                existing.capacity = capacity
                existing.refill_per_second = refill
                existing.tokens = min(existing.tokens, capacity)
            return existing
        bucket = TokenBucket(capacity=capacity, refill_per_second=refill, now=now)
        self._buckets[key] = bucket
        if len(self._buckets) > self._max_tracked:
            # Least recently touched, not least recently created — `move_to_end` above keeps
            # a key that is still being asked about at the warm end of the queue.
            self._buckets.popitem(last=False)
        return bucket

    def take(
        self,
        key: str,
        *,
        per_minute: int | None = None,
        burst: int | None = None,
        cost: float = 1.0,
    ) -> RateDecision:
        """Charge ``key`` for one request, admitting it if the bucket can afford it."""
        with self._lock:
            now = self._clock()
            bucket = self._bucket(key, per_minute=per_minute, burst=burst, now=now)
            if bucket.take(now, cost=cost):
                return RateDecision(allowed=True)
            return RateDecision(allowed=False, retry_after_s=bucket.retry_after(now, cost=cost))

    def available(
        self,
        key: str,
        *,
        per_minute: int | None = None,
        burst: int | None = None,
        cost: float = 1.0,
    ) -> RateDecision:
        """Whether ``key`` could afford one request right now, without charging it.

        Used to decide *before* doing expensive work whether it is even worth attempting — the
        failed-authentication bucket is consulted this way, so an address that has already used
        up its allowance is refused before a credential is hashed and looked up.
        """
        with self._lock:
            now = self._clock()
            bucket = self._bucket(key, per_minute=per_minute, burst=burst, now=now)
            if bucket.available(now, cost=cost):
                return RateDecision(allowed=True)
            return RateDecision(allowed=False, retry_after_s=bucket.retry_after(now, cost=cost))


def caller_key(caller: Caller) -> str:
    """The bucket a caller is charged against.

    The API key first, because two people sharing a key are meant to share its budget; then the
    signed-in person, for a caller a future surface identifies without a key; then the client
    address, for anybody presenting neither — which includes every caller today, since sessions
    are not yet wired to a caller. ``"unknown"`` stands in for an address nothing could establish,
    so that every caller still lands in exactly one bucket rather than sharing the empty string.
    """
    if caller.key_id:
        return f"key:{caller.key_id}"
    if caller.user_id:
        return f"user:{caller.user_id}"
    return f"addr:{caller.address or 'unknown'}"


class RateLimiter:
    """The two buckets :class:`~manicule.config.settings.RateLimitSettings` describes.

    Owned by :class:`~manicule.app.service.ApplicationService`, one instance per served process,
    so its buckets persist for as long as the buckets they meter matter — across requests, not
    across processes. ``enabled = false`` makes every method report unlimited, rather than
    branching at every call site: a network surface always calls through this object and never
    has to ask first whether limiting is on.
    """

    def __init__(self, settings: RateLimitSettings, *, clock: Clock = time.monotonic) -> None:
        self._enabled = settings.enabled
        self._caller = KeyedLimiter(
            per_minute=settings.per_minute,
            burst=settings.burst,
            max_tracked=settings.max_tracked,
            clock=clock,
        )
        # No separate burst setting for failed authentication, so the bucket's capacity is its
        # own rate: an address may spend a whole minute's allowance of failures at once and then
        # only refills gradually. That is the conservative reading of one number doing two jobs
        # — a smaller, separately-configured burst would let an address fail *faster* than
        # `failed_auth_per_minute` names, which is the opposite of what the setting promises.
        self._failed_auth = KeyedLimiter(
            per_minute=settings.failed_auth_per_minute,
            burst=settings.failed_auth_per_minute,
            max_tracked=settings.max_tracked,
            clock=clock,
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def charge_failed_auth(self, address: str) -> RateDecision:
        """Record one failed authentication from ``address``, and say whether it was within budget.

        Only ever called after a presented credential turned out not to work — a request that
        offered nothing, or one whose credential checked out, never touches this bucket, because
        neither is a guess. **So a working credential is never refused by it**, whoever else
        shares the address: behind a proxy nobody configured as trusted, or one office's NAT,
        every caller has the same address, and a bucket that refused correct keys once one
        client had spent it would let a single stale key sign everybody out. A key is 256 bits
        and a session cookie is signed, so refusing a correct one buys no protection from
        guessing; what this bucket bounds is how fast an address may keep guessing.
        """
        if not self._enabled:
            return RateDecision(allowed=True)
        return self._failed_auth.take(_address_key(address))

    def charge_caller(self, key: str, *, rate_limit: int | None = None) -> RateDecision:
        """Charge ``key`` — see :func:`caller_key` — for one ordinary request.

        ``rate_limit``, when given, replaces **both** the refill rate and the bucket's own
        capacity for this key — on the failed-auth bucket's own reasoning: a key capped at, say,
        5 requests per minute that could still burst up to the installation's global ``burst``
        whenever its bucket happened to be full from disuse would make a low per-key limit
        nearly decorative. Capacity equal to the rate is what makes "N requests per minute"
        mean what it says, whichever direction a key's own number sits from the installation's.
        """
        if not self._enabled:
            return RateDecision(allowed=True)
        return self._caller.take(key, per_minute=rate_limit, burst=rate_limit)


def _address_key(address: str) -> str:
    return f"addr:{address or 'unknown'}"


__all__ = [
    "Clock",
    "KeyedLimiter",
    "RateDecision",
    "RateLimiter",
    "TokenBucket",
    "caller_key",
]
