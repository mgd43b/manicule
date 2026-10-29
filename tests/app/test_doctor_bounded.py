"""``doctor`` degrades to partial results when a check stalls, instead of hanging.

A served process can sit behind a writer or a slow disk for minutes. The diagnosis is what a
dashboard and a health poll wait on, so one check that never returns must cost a sentence in
the output rather than the whole answer.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, override

from manicule.app import service as service_module
from manicule.app.service import ApplicationService
from tests.app.fakes import FakeBackend, FakeStore

if TYPE_CHECKING:
    from collections.abc import Collection

    import pytest

    from manicule.core.content import DocumentStatus

CEILING = 30.0
"""Seconds; the per-check bound below is 0.2, so this only fails on a diagnosis that hangs."""


class StalledCount(FakeStore):
    """A store whose counts never return. **Deliberately stalled.**"""

    @override
    async def count_documents(
        self,
        *,
        source: str | None = None,
        statuses: Collection[DocumentStatus] | None = None,
        glossary_fp_other_than: str | None = None,
        glossary_fp_unrecorded: bool = False,
    ) -> int:
        del source, statuses, glossary_fp_other_than, glossary_fp_unrecorded
        await asyncio.Event().wait()
        return 0


async def test_a_stalled_check_is_reported_unknown_and_the_rest_still_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_module, "DOCTOR_CHECK_SECONDS", 0.2)
    backend = FakeBackend()
    backend.store = StalledCount(workspace_id=backend.settings.workspace)
    started = time.monotonic()
    diagnosis = await ApplicationService(backend).doctor()
    assert time.monotonic() - started < CEILING
    checks = {check.name: check for check in diagnosis.checks}
    assert checks["configuration"].state != "unknown"
    stalled = [check for check in diagnosis.checks if check.facts.get("measured") is False]
    assert stalled, "the stalled count should have left at least one unmeasured check"
    assert all(check.state == "unknown" for check in stalled)
    assert all("not measured" in check.detail for check in stalled)
    assert diagnosis.state != "ok"


async def test_checks_after_the_budget_is_spent_are_reported_unmeasured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_module, "DOCTOR_BUDGET_SECONDS", 0.0)
    diagnosis = await ApplicationService(FakeBackend()).doctor()
    ran_out = [check for check in diagnosis.checks if "ran out of time" in check.detail]
    assert {check.name for check in ran_out} >= {"storage", "index", "models"}
