"""The admin group: what this installation is doing, and what it has done.

Every route here takes an admin. That is not caution about the *reads* being expensive — it is
what they contain. Query logs hold the questions people asked, which is a record of what a
team was working on; the audit trail names who did what and from where; connector status
names the systems this installation reaches into. None of that is a viewer's business.

**Search quality reports; it does not measure.** ``manicule.evaluation`` is the only thing in
this project that decides whether one retrieval configuration beats another, and it refuses to
report at all for a system it cannot distinguish from guessing. So this route reads the
harness's own store and renders the harness's own report, caveats included. An endpoint that
computed a second score would produce a number nobody could reconcile with the first — and an
**example** query set rendered as a percentage is not a measurement, which is why
``is_evidence`` travels with it.

**Two things this group deliberately does not offer.** There is no route that creates or
edits a connector: sources are declared in configuration, where the whole set is reviewable in
one place, and a route that wrote one would be a second way to edit the file. And there is no
benchmark endpoint: a benchmark run on request, from a surface an unattended caller can reach,
is a way to make an installation unusable with one HTTP request.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Query, Response

from manicule.api.context import Service
from manicule.api.envelopes import respond
from manicule.api.models import SyncBody
from manicule.api.security import AdminPrincipal

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


@router.get("/stats", name="index_status", summary="Index counts and fingerprints.")
async def stats(service: Service, caller: AdminPrincipal) -> Response:
    """What is in the index and what it was built with.

    The embedding and chunking fingerprints are here in their canonical form as well as a
    readable one. The canonical form is what a re-embed compares, and a prettied identity is
    one nobody can compare.
    """
    del caller
    return await respond("index_status", service, service.index_status)


@router.get(
    "/reembed/{run_id}",
    name="reembed_status",
    summary="Aggregate progress for a durable re-embedding run.",
)
async def reembed_status(service: Service, caller: AdminPrincipal, run_id: str) -> Response:
    """Read private-safe progress; starting or executing migrations remains operator-only."""
    del caller
    return await respond("reembed_status", service, lambda: service.reembed_status(run_id))


@router.get("/reembed", name="reembed_plan", summary="Aggregate dry-run re-embedding plan.")
async def reembed_plan(service: Service, caller: AdminPrincipal) -> Response:
    del caller
    return await respond("reembed_plan", service, service.reembed_plan)


@router.post(
    "/reembed/{run_id}/start", name="reembed_start", summary="Create a durable re-embedding run."
)
async def reembed_start(service: Service, caller: AdminPrincipal, run_id: str) -> Response:
    del caller
    return await respond("reembed_start", service, lambda: service.reembed_start(run_id))


@router.post(
    "/reembed/{run_id}/resume", name="reembed_resume", summary="Resume a durable re-embedding run."
)
async def reembed_resume(service: Service, caller: AdminPrincipal, run_id: str) -> Response:
    del caller
    return await respond("reembed_resume", service, lambda: service.reembed_resume(run_id))


@router.post(
    "/reembed/{run_id}/abandon",
    name="reembed_abandon",
    summary="Abandon an unfinished durable re-embedding run.",
)
async def reembed_abandon(service: Service, caller: AdminPrincipal, run_id: str) -> Response:
    del caller
    return await respond("reembed_abandon", service, lambda: service.reembed_abandon(run_id))


@router.delete(
    "/reembed/{run_id}",
    name="reembed_cleanup",
    summary="Clean terminal non-live re-embedding storage.",
)
async def reembed_cleanup(service: Service, caller: AdminPrincipal, run_id: str) -> Response:
    del caller
    return await respond("reembed_cleanup", service, lambda: service.reembed_cleanup(run_id))


@router.get("/lifecycle/reset-derived", name="lifecycle_reset_derived")
async def lifecycle_reset_derived(service: Service, caller: AdminPrincipal) -> Response:
    """Aggregate dry run only; destructive lifecycle actions remain local operator actions."""
    del caller
    return await respond(
        "lifecycle_reset_derived",
        service,
        lambda: service.lifecycle_reset_derived(dry_run=True),
    )


@router.get(
    "/lifecycle/derived-generations",
    name="lifecycle_cleanup_generations",
    summary="Aggregate dry run for obsolete derived-generation cleanup.",
)
async def lifecycle_cleanup_generations(service: Service, caller: AdminPrincipal) -> Response:
    del caller
    return await respond(
        "lifecycle_cleanup_generations",
        service,
        lambda: service.lifecycle_cleanup_generations(dry_run=True),
    )


@router.get(
    "/lifecycle/source-history",
    name="lifecycle_release_history",
    summary="Aggregate dry run for historical source retention release.",
)
async def lifecycle_release_history(
    service: Service,
    caller: AdminPrincipal,
    before: datetime,
) -> Response:
    del caller
    return await respond(
        "lifecycle_release_history",
        service,
        lambda: service.lifecycle_release_history(before, dry_run=True),
    )


@router.get(
    "/lifecycle/snapshots/{run_id}",
    name="lifecycle_delete_snapshot",
    summary="Aggregate local-loss plan and confirmation token for snapshot deletion.",
)
async def lifecycle_delete_snapshot(
    service: Service,
    caller: AdminPrincipal,
    run_id: str,
) -> Response:
    del caller
    return await respond(
        "lifecycle_delete_snapshot",
        service,
        lambda: service.lifecycle_delete_snapshot(run_id, dry_run=True),
    )


@router.get("/query-logs", name="query_logs", summary="Retrieval telemetry, newest first.")
async def query_logs(
    service: Service,
    caller: AdminPrincipal,
    *,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Response:
    """A page of what has actually been asked, with the total so a client can page.

    Written by the service on every ``search`` and every ``ask``, so this describes every
    retrieval this installation ran rather than the ones one surface remembered to record.
    The stored chunk ids are reported as a **count**: corpus structure has no business
    traveling with a telemetry listing.
    """
    del caller
    return await respond(
        "query_logs", service, lambda: service.query_logs(limit=limit, offset=offset)
    )


@router.get(
    "/connectors/{name}/snapshot",
    name="snapshot_status",
    summary="Aggregate status of a connector's active durable snapshot.",
)
async def snapshot_status(service: Service, caller: AdminPrincipal, name: str) -> Response:
    del caller
    return await respond("snapshot_status", service, lambda: service.snapshot_status(name))


@router.get(
    "/snapshots/{snapshot_id}/verify",
    name="snapshot_verify",
    summary="Verify one workspace-owned durable snapshot manifest.",
)
async def snapshot_verify(service: Service, caller: AdminPrincipal, snapshot_id: str) -> Response:
    del caller
    return await respond("snapshot_verify", service, lambda: service.snapshot_verify(snapshot_id))


@router.get("/audit-logs", name="audit_log", summary="The audit trail, newest first.")
async def audit_logs(
    service: Service,
    caller: AdminPrincipal,
    *,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    event_type: Annotated[str | None, Query()] = None,
) -> Response:
    """Security-relevant events, with whether auditing is switched on at all.

    ``enabled`` is on the payload rather than implied by an empty list, because "nothing
    happened" and "nothing was recorded" are different answers and an operator reading an
    empty audit log needs to know which one they have.
    """
    del caller
    return await respond(
        "audit_log",
        service,
        lambda: service.audit_log(limit=limit, offset=offset, event_type=event_type),
    )


@router.get(
    "/search-quality", name="search_quality", summary="What the evaluation harness has recorded."
)
async def search_quality(service: Service, caller: AdminPrincipal) -> Response:
    """The pairwise harness's own report, verbatim, or an honest statement that there is none.

    ``available`` false means nobody has judged any pairs — which is the truth, and is not the
    same as a score of zero. ``is_evidence`` false means the query set behind the numbers is
    an example one, and the harness's own caveat travels with them.
    """
    del caller
    return await respond("search_quality", service, service.search_quality)


@router.get("/plugins", name="plugin_health", summary="Plugin health.")
async def plugin_health(service: Service, caller: AdminPrincipal) -> Response:
    """Installed plugins and the health of whatever each one has constructed.

    A component nothing has asked for yet is ``unknown``, not ``ok``. That distinction is the
    report's whole value: a plugin that has never been built has not been proved healthy.
    """
    del caller
    return await respond("plugin_health", service, service.plugin_health)


@router.get(
    "/connectors",
    name="connector_list",
    summary="Configured sources and what their last run recorded.",
)
async def connectors(service: Service, caller: AdminPrincipal) -> Response:
    """Every configured connector, whether its implementation is installed, and its watermark."""
    del caller
    return await respond("connector_list", service, service.connector_list)


@router.post(
    "/connectors/{name}/sync", name="connector_sync", summary="Run one configured connector."
)
async def sync_connector(
    service: Service, caller: AdminPrincipal, name: str, body: SyncBody
) -> Response:
    """Ingest what changed since this connector's watermark.

    Only a connector that **configuration already names** can be run. There is no route that
    creates one: a connector holds credentials and reaches a remote system, and declaring one
    over HTTP would make this surface a way to point the installation at somewhere new.
    """
    del caller
    return await respond(
        "connector_sync",
        service,
        lambda: service.connector_sync(name, limit=body.limit, acquire_only=body.acquire_only),
    )


__all__ = ["router"]
