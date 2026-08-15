"""Aggregate operator contracts above the durable re-embedding port."""

from __future__ import annotations

import pytest

from manicule.app.service import ApplicationService
from manicule.core.errors import UnknownEntityError
from manicule.ingest.reembed import ReembedState
from tests.app.fakes import FakeBackend


async def test_plan_and_run_reports_never_serialize_private_commitment_fields() -> None:
    backend = FakeBackend()
    service = ApplicationService(backend)

    plan = await service.reembed_plan()
    started = await service.reembed_start()
    status = await service.reembed_status(started.run_id)

    assert plan.documents == status.documents == 2
    assert plan.chunks == status.chunks == 7
    assert status.state == ReembedState.PLANNED
    assert status.retry_required
    exposed = plan.model_dump_json() + status.model_dump_json()
    for private in (
        "private-snapshot",
        "private-target-config",
        "private-weights-path",
        "private-inventory",
        "private-chunks",
    ):
        assert private not in exposed


async def test_resume_abandon_cleanup_and_missing_status_have_typed_semantics() -> None:
    backend = FakeBackend()
    service = ApplicationService(backend)

    completed = await service.reembed_start()
    completed = await service.reembed_resume(completed.run_id)
    assert completed.state == ReembedState.PUBLISHED
    assert completed.published
    assert completed.terminal
    assert not completed.retry_required

    abandoned = await service.reembed_start()
    abandoned = await service.reembed_abandon(abandoned.run_id)
    assert abandoned.state == ReembedState.FAILED
    assert abandoned.terminal
    assert (await service.reembed_cleanup(abandoned.run_id)).removed

    with pytest.raises(UnknownEntityError, match="no durable re-embedding run"):
        await service.reembed_status("missing")
