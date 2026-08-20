"""Aggregate operator contracts above the durable re-embedding port."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from manicule.api.envelopes import SERVICE_UNAVAILABLE, status_for
from manicule.app.dispatch import run_op
from manicule.app.service import ApplicationService
from manicule.core.errors import StorageBusyError, UnknownEntityError
from manicule.ingest.reembed import ReembedState
from manicule.web.rendering import panel
from tests.api.support import client_for
from tests.app.fakes import FakeBackend

if TYPE_CHECKING:
    from manicule.ingest.reembed import ReembedPlan


async def test_plan_and_run_reports_never_serialize_private_commitment_fields() -> None:
    backend = FakeBackend()
    service = ApplicationService(backend)

    plan = await service.reembed_plan()
    started = await service.reembed_start("service-plan")
    status = await service.reembed_status(started.run_id)

    assert plan.documents == status.documents == 2
    assert plan.chunks == status.chunks == 7
    assert status.state == ReembedState.PLANNED
    assert status.retry_required
    assert plan.lifecycle.dry_run
    assert plan.lifecycle.estimated_remaining_items == 7
    assert status.lifecycle.phase == "reembedding"
    assert status.lifecycle.outcome == "running"
    assert status.lifecycle.pending_items == 7
    assert status.lifecycle.can_continue_offline
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

    completed = await service.reembed_start("service-complete")
    completed = await service.reembed_resume(completed.run_id)
    assert completed.state == ReembedState.PUBLISHED
    assert completed.published
    assert completed.terminal
    assert not completed.retry_required
    assert completed.lifecycle.phase == "complete"
    assert completed.lifecycle.outcome == "complete"
    assert completed.lifecycle.pending_items == 0

    abandoned = await service.reembed_start("service-abandon")
    abandoned = await service.reembed_abandon(abandoned.run_id)
    assert abandoned.state == ReembedState.FAILED
    assert abandoned.terminal
    assert abandoned.lifecycle.phase == "failed"
    assert abandoned.lifecycle.outcome == "failed"
    assert (await service.reembed_cleanup(abandoned.run_id)).removed

    with pytest.raises(UnknownEntityError, match="no durable re-embedding run"):
        await service.reembed_status("missing")


async def test_failed_reembed_execution_is_not_a_success_envelope() -> None:
    backend = FakeBackend()
    service = ApplicationService(backend)
    started = await service.reembed_start("service-failed-envelope")
    await service.reembed_abandon(started.run_id)

    envelope = await run_op(
        "reembed_resume",
        service.workspace,
        lambda: service.reembed_status(started.run_id),
    )

    assert envelope.ok is False
    assert envelope.error is not None
    assert envelope.error.type == "ReembedLifecycleError"
    assert envelope.data is None


async def test_planning_refused_the_writer_slot_is_a_typed_result_rather_than_a_500() -> None:
    """The planning page is an inspection surface, and it used to be able to crash one.

    Pricing a re-embed opens a durable corpus snapshot, and that snapshot takes SQLite's
    writer slot with ``BEGIN IMMEDIATE``. While an offline rebuild held the writer, the
    refusal arrived as the driver's own ``database is locked`` — an exception
    :func:`~manicule.app.dispatch.run_op` deliberately does not convert, because converting
    arbitrary exceptions is how a defect gets reported as a tidy result. So opening
    ``/ui/reembed`` produced an unhandled ASGI traceback and an opaque 503-shaped problem
    dressed as a 500, over and over, for a page that had asked a read-only question.

    Contention is now refused in the vocabulary every surface already renders, so the page,
    the API and the tool all say the same bounded thing and none of them says it in SQL.
    """
    backend = FakeBackend()
    service = ApplicationService(backend)

    async def busy() -> tuple[ReembedPlan, str, int]:
        raise StorageBusyError("no plan was made and nothing durable changed")

    backend.ingestion_.reembed_plan = busy

    envelope = await panel("reembed_plan", service, service.reembed_plan)
    with client_for(backend) as client:
        page = client.get("/ui/reembed")

    assert envelope.envelope.ok is False
    error = envelope.envelope.error
    assert error is not None
    assert error.type == "StorageBusyError"
    assert error.hint, "a busy writer is a retry, and the caller has to be told that"
    assert status_for(envelope.envelope) == SERVICE_UNAVAILABLE, (
        "contention is temporary and the request was well formed, so it is not a client error"
    )
    assert page.status_code == 200, "an inspection page renders its refusal, it does not crash"
    assert "text/html" in page.headers["content-type"]
    rendered = page.text.lower()
    assert "temporarily busy" in rendered
    for driver_text in ("begin immediate", "database is locked", "traceback", ".sqlite"):
        assert driver_text not in rendered


async def test_a_busy_refusal_reads_the_same_over_http_as_it_does_on_the_page() -> None:
    """One refusal, one shape. A page that agreed with nothing would be a second contract."""
    backend = FakeBackend()
    service = ApplicationService(backend)

    async def busy() -> tuple[ReembedPlan, str, int]:
        raise StorageBusyError("no plan was made and nothing durable changed")

    backend.ingestion_.reembed_plan = busy

    with client_for(backend) as client:
        response = client.get("/api/v1/admin/reembed")
    http = cast("dict[str, Any]", response.json())
    web = (await panel("reembed_plan", service, service.reembed_plan)).envelope.as_json()

    assert response.status_code == SERVICE_UNAVAILABLE
    assert http == web
