"""Lifecycle: setup, teardown, health and metrics.

Every component the container builds may implement these. All four have working defaults,
so a component opts in to the ones it needs by overriding them and stays silent about the
rest — the alternative is four stub methods on every implementation, which nobody reads and
everybody copies wrong.

Dependencies arrive through the constructor, so :meth:`Lifecycle.setup` takes no context
argument. It exists for work that cannot happen during construction: opening a connection,
loading a model, creating a directory.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Protocol, override, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class HealthState(StrEnum):
    """How well a component is working.

    Three states, not two: a component that is answering but with a degraded model, a stale
    cache or a missing optional dependency is neither healthy nor down, and collapsing that
    into a boolean means either false alarms or silent degradation.
    """

    OK = "ok"
    DEGRADED = "degraded"
    FAILING = "failing"

    @property
    def severity(self) -> int:
        return _SEVERITY[self]


_SEVERITY: Mapping[HealthState, int] = {
    HealthState.OK: 0,
    HealthState.DEGRADED: 1,
    HealthState.FAILING: 2,
}


def worst_state(states: Iterable[HealthState]) -> HealthState:
    """The most severe of ``states``, or :attr:`HealthState.OK` when empty."""
    return max(states, key=lambda s: s.severity, default=HealthState.OK)


class HealthCheck(BaseModel):
    """One named observation contributing to a component's health."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    state: HealthState
    detail: str = ""


class HealthReport(BaseModel):
    """What a component says about itself when asked.

    :attr:`remedy` is the difference between a diagnostic that is read and one that is
    dismissed: it names the action that fixes the problem, in the words of whoever has to
    do it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: HealthState = HealthState.OK
    detail: str = ""
    remedy: str = Field(
        default="",
        description="What to do about it, when there is something to do.",
    )
    checks: tuple[HealthCheck, ...] = ()

    @property
    def ok(self) -> bool:
        return self.state is HealthState.OK

    @classmethod
    def healthy(cls, detail: str = "") -> HealthReport:
        return cls(state=HealthState.OK, detail=detail)

    @classmethod
    def failing(cls, detail: str, remedy: str = "") -> HealthReport:
        return cls(state=HealthState.FAILING, detail=detail, remedy=remedy)

    @classmethod
    def degraded(cls, detail: str, remedy: str = "") -> HealthReport:
        return cls(state=HealthState.DEGRADED, detail=detail, remedy=remedy)

    @classmethod
    def rollup(cls, reports: Mapping[str, HealthReport]) -> HealthReport:
        """Combine per-component reports into one, taking the worst state.

        Each component becomes a named check, so the summary says which part is unwell
        rather than only that something is.
        """
        checks = tuple(
            HealthCheck(name=name, state=report.state, detail=report.detail)
            for name, report in reports.items()
        )
        worst = worst_state(check.state for check in checks)
        unwell = [c.name for c in checks if c.state is not HealthState.OK]
        detail = f"unhealthy: {', '.join(sorted(unwell))}" if unwell else "all components healthy"
        return cls(state=worst, detail=detail, checks=checks)


class Metric(BaseModel):
    """One measurement a component publishes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    value: float
    unit: str = ""
    labels: Mapping[str, str] = Field(default_factory=dict)


@runtime_checkable
class SupportsSetup(Protocol):
    """Has work to do after construction and before first use."""

    async def setup(self) -> None:
        """Acquire resources: open a connection, load a model, create a directory.

        Called once. Dependencies arrive through the constructor and are set up before their
        dependents, so anything injected into this component is ready by the time this runs.
        """
        ...


@runtime_checkable
class SupportsTeardown(Protocol):
    """Has resources to release."""

    async def teardown(self) -> None:
        """Release resources. Called once, in reverse setup order.

        Must be safe to call after a failed :meth:`SupportsSetup.setup`, because that is
        exactly when it is most needed.
        """
        ...


@runtime_checkable
class SupportsHealth(Protocol):
    """Can say how it is."""

    async def health(self) -> HealthReport:
        """Report on this component right now. Should not raise; report instead."""
        ...


@runtime_checkable
class SupportsMetrics(Protocol):
    """Publishes measurements."""

    def metrics(self) -> Sequence[Metric]:
        """Current measurements. Cheap and synchronous — no I/O behind this."""
        ...


class Lifecycle(SupportsSetup, SupportsTeardown, SupportsHealth, SupportsMetrics, Protocol):
    """All four hooks, with working defaults.

    Every hook is optional, and each is detected separately: a component implementing only
    :meth:`~SupportsSetup.setup` gets its setup called and is not asked for metrics. That is
    why the four are separate protocols rather than one — an all-or-nothing check would
    quietly skip a component that implemented three of them.

    Inherit from this to pick up the no-op defaults and override the ones you want. No
    protocol in :mod:`manicule.core.protocols` requires it, so an implementation that needs
    none of the four writes none of them.
    """

    @override
    async def setup(self) -> None:
        return

    @override
    async def teardown(self) -> None:
        return

    @override
    async def health(self) -> HealthReport:
        return HealthReport.healthy()

    @override
    def metrics(self) -> Sequence[Metric]:
        return ()


__all__ = [
    "HealthCheck",
    "HealthReport",
    "HealthState",
    "Lifecycle",
    "Metric",
    "SupportsHealth",
    "SupportsMetrics",
    "SupportsSetup",
    "SupportsTeardown",
    "worst_state",
]
