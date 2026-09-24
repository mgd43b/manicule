"""Security alerts: a pattern across requests, not a single one of them.

:class:`~manicule.config.settings.AlertSettings` is the specification — read its docstring
first, it names the four patterns and their thresholds. This module is pure detection: it holds
sliding windows of recent activity and decides when a threshold has been crossed. It writes
nothing anywhere and knows nothing about a store, a logger or an audit trail — those belong to
whoever calls it, because deciding *that* something happened and deciding *what to do about it*
are different jobs, and only the second one needs a workspace, a clock the operator trusts, and
the authority to write.

**Four patterns, four windows, one shape.** Each window is a bounded map from a subject —
an address, a key id, an actor — to the recent timestamps (or timestamped values, for the two
patterns that count *distinct* things) that matter to it. Recording an occurrence evicts
anything older than ``window_s`` and returns the count that survives, which is compared against
its threshold with ``>=``: reaching the threshold is what "N things in the window" means, not
exceeding it by one.

**Each ``(kind, subject)`` fires at most once per window.** Once :meth:`AlertMonitor` has
returned an event for an address's brute-force attempt, the same address hammering the same
bucket for the rest of that window produces no second event — a sustained attack is one alert,
not one per request, which is the difference between something a person can act on and a queue
they give up reading.

**Memory is bounded the same way the rate limiter's is.** Every window, and the record of what
has already fired, is an ``OrderedDict`` capped at ``max_tracked`` entries with the
least-recently-touched evicted first — by convention the same number
:class:`~manicule.config.settings.RateLimitSettings.max_tracked` names, since both structures are
answering the same question: how many distinct callers this process remembers at once.
"""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping

    from manicule.config.settings import AlertSettings

AlertKind = Literal["brute_force", "key_abuse", "export_volume"]
"""No fourth kind exists yet, and this alias is the reason it cannot be spelled wrong:
:class:`~manicule.storage.models.SecurityAlert` carries the identical three values as a
``CHECK`` constraint, so a kind this module could produce and that table could not store is
caught by a type checker instead of by a write that fails at the database."""


class Clock(Protocol):
    def __call__(self) -> float: ...


@dataclass(frozen=True, slots=True)
class AlertEvent:
    """One alert this monitor decided to fire. What happens next is not this module's job."""

    kind: AlertKind
    subject: str
    details: Mapping[str, object] = field(default_factory=dict[str, object])


class _CountWindow:
    """Subject -> recent timestamps, evicted past ``window_s``, bounded to ``max_tracked``
    subjects."""

    def __init__(self, *, window_s: float, max_tracked: int) -> None:
        self._window_s = window_s
        self._max_tracked = max_tracked
        self._entries: OrderedDict[str, deque[float]] = OrderedDict()

    def record(self, subject: str, now: float) -> int:
        entries = self._touch(subject)
        entries.append(now)
        self._evict(entries, now)
        return len(entries)

    def _touch(self, subject: str) -> deque[float]:
        entries = self._entries.get(subject)
        if entries is not None:
            self._entries.move_to_end(subject)
            return entries
        entries = deque[float]()
        self._entries[subject] = entries
        if len(self._entries) > self._max_tracked:
            self._entries.popitem(last=False)
        return entries

    def _evict(self, entries: deque[float], now: float) -> None:
        cutoff = now - self._window_s
        while entries and entries[0] < cutoff:
            entries.popleft()


class _DistinctWindow:
    """Subject -> recent (timestamp, value) pairs; reports distinct values within the window.

    The shape :meth:`AlertMonitor.record_key_presentation` and
    :meth:`AlertMonitor.record_document_read` both need: not "how many times", but "how many
    different addresses" or "how many different documents" — a caller re-reading the one
    document it always reads must never look like an export.
    """

    def __init__(self, *, window_s: float, max_tracked: int) -> None:
        self._window_s = window_s
        self._max_tracked = max_tracked
        self._entries: OrderedDict[str, deque[tuple[float, str]]] = OrderedDict()

    def record(self, subject: str, value: str, now: float) -> int:
        entries = self._touch(subject)
        entries.append((now, value))
        self._evict(entries, now)
        return len({item[1] for item in entries})

    def _touch(self, subject: str) -> deque[tuple[float, str]]:
        entries = self._entries.get(subject)
        if entries is not None:
            self._entries.move_to_end(subject)
            return entries
        entries = deque[tuple[float, str]]()
        self._entries[subject] = entries
        if len(self._entries) > self._max_tracked:
            self._entries.popitem(last=False)
        return entries

    def _evict(self, entries: deque[tuple[float, str]], now: float) -> None:
        cutoff = now - self._window_s
        while entries and entries[0][0] < cutoff:
            entries.popleft()


class AlertMonitor:
    """Detects the four patterns :class:`~manicule.config.settings.AlertSettings` names.

    ``enabled = false`` makes every ``record_*`` method report nothing, on
    :class:`~manicule.app.throttle.RateLimiter`'s rule: a caller always calls through, and never
    has to ask first whether detection is switched on.
    """

    def __init__(
        self, settings: AlertSettings, *, max_tracked: int, clock: Clock = time.monotonic
    ) -> None:
        self._settings = settings
        self._clock = clock
        window_s = float(settings.window_s)
        self._failed_auth = _CountWindow(window_s=window_s, max_tracked=max_tracked)
        self._key_addresses = _DistinctWindow(window_s=window_s, max_tracked=max_tracked)
        self._rate_limited = _CountWindow(window_s=window_s, max_tracked=max_tracked)
        self._document_reads = _DistinctWindow(window_s=window_s, max_tracked=max_tracked)
        self._last_fired: OrderedDict[tuple[str, str], float] = OrderedDict()
        self._max_tracked = max_tracked

    @property
    def enabled(self) -> bool:
        return self._settings.enabled

    def record_failed_auth(self, address: str) -> AlertEvent | None:
        """A brute-force pattern: many failed authentications from one address."""
        if not self.enabled or not address:
            return None
        now = self._clock()
        count = self._failed_auth.record(address, now)
        if count < self._settings.failed_auth_threshold:
            return None
        return self._fire(
            "brute_force", address, now, {"count": count, "window_s": self._settings.window_s}
        )

    def record_key_presentation(self, key_id: str, address: str) -> AlertEvent | None:
        """A key-abuse pattern: one key presented from many distinct addresses.

        A key that has leaked or is being shared looks exactly like this — the same secret
        arriving from more places than one person's devices plausibly explain.
        """
        if not self.enabled or not key_id or not address:
            return None
        now = self._clock()
        count = self._key_addresses.record(key_id, address, now)
        if count < self._settings.key_address_threshold:
            return None
        return self._fire(
            "key_abuse", key_id, now, {"addresses": count, "window_s": self._settings.window_s}
        )

    def record_rate_limited(self, key_id: str) -> AlertEvent | None:
        """The other key-abuse pattern: one key repeatedly refused for exceeding its own budget.

        Reuses :attr:`~manicule.config.settings.AlertSettings.failed_auth_threshold` rather than
        adding a fourth number to the settings model — a key hammering its own rate limit is the
        same shape of problem as an address hammering authentication, and the two are worth the
        same amount of tolerance before they are worth a person's attention.
        """
        if not self.enabled or not key_id:
            return None
        now = self._clock()
        count = self._rate_limited.record(key_id, now)
        if count < self._settings.failed_auth_threshold:
            return None
        return self._fire(
            "key_abuse", key_id, now, {"rate_limited": count, "window_s": self._settings.window_s}
        )

    def record_document_read(self, actor: str, document_id: str) -> AlertEvent | None:
        """An export-volume pattern: one caller reading an unusual number of distinct documents."""
        if not self.enabled or not actor or not document_id:
            return None
        now = self._clock()
        count = self._document_reads.record(actor, document_id, now)
        if count < self._settings.export_document_threshold:
            return None
        return self._fire(
            "export_volume", actor, now, {"documents": count, "window_s": self._settings.window_s}
        )

    def _fire(
        self, kind: AlertKind, subject: str, now: float, details: dict[str, object]
    ) -> AlertEvent | None:
        """Admit one alert for ``(kind, subject)``, or refuse it if this window already fired
        one."""
        key = (kind, subject)
        last = self._last_fired.get(key)
        if last is not None:
            self._last_fired.move_to_end(key)
            if now - last < self._settings.window_s:
                return None
        self._last_fired[key] = now
        self._last_fired.move_to_end(key)
        if len(self._last_fired) > self._max_tracked:
            self._last_fired.popitem(last=False)
        return AlertEvent(kind=kind, subject=subject, details=details)


__all__ = ["AlertEvent", "AlertKind", "AlertMonitor", "Clock"]
