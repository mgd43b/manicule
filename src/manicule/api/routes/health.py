"""Health, readiness and the two counts everybody asks for.

``/healthz`` and ``/readyz`` are the only routes that answer without an envelope, and the
reason is that they answer to something that is not a person: a process supervisor, a
container orchestrator, a monitoring probe. Those read a status code and nothing else, and a
liveness probe that has to parse JSON is a liveness probe that reports unhealthy when the
serialiser changes.

They also answer **different questions**, which is why there are two. ``/healthz`` says this
process is running and can serve a request — it opens nothing. ``/readyz`` says the index is
actually usable, which means asking the store, which means it can fail while the process is
perfectly alive. Collapsing them gives you a probe that restarts a healthy process because a
database file is briefly locked.
"""

from __future__ import annotations

from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse

from manicule.api.context import Service
from manicule.api.envelopes import OK, SERVER_ERROR, respond
from manicule.api.security import AnonymousPrincipal, ViewerPrincipal

router = APIRouter(tags=["health"])


@router.get("/healthz", name="healthz", summary="Liveness. Opens nothing.")
async def healthz(caller: AnonymousPrincipal) -> Response:
    """Whether this process is up. Deliberately unauthenticated and deliberately cheap.

    It touches no store, so it cannot be made to fail by a database that is busy — which is
    the whole point of a liveness probe as distinct from a readiness one.
    """
    del caller
    return JSONResponse(content={"status": "ok"}, status_code=OK)


@router.get("/readyz", name="readyz", summary="Readiness. Asks the index whether it can serve.")
async def readyz(service: Service, caller: AnonymousPrincipal) -> Response:
    """Whether this installation can actually answer a question.

    What "ready" means is the **service's** decision, not this route's: a probe that decided it
    here would be a second opinion about what a working installation is, on the one endpoint an
    orchestrator uses to restart things.

    A failure is a **503 in spirit and a 500 in status**: the process is alive and the index is
    not usable, and a probe reading the status code learns the right thing either way. It
    reports no counts, no configuration and no reason — an unauthenticated endpoint saying "412
    documents, chunker structural" would be a corpus fingerprint for anyone who can reach the
    port, and so, more quietly, would the text of a database error.
    """
    del caller
    if not await service.ready():
        return JSONResponse(content={"status": "unready"}, status_code=SERVER_ERROR)
    return JSONResponse(content={"status": "ready"}, status_code=OK)


@router.get("/api/v1/health", name="doctor", summary="Diagnostics, in the ordinary envelope.")
async def health(service: Service, caller: ViewerPrincipal) -> Response:
    """Everything ``manicule doctor`` checks, as the same envelope every other surface emits.

    Authenticated, unlike the two probes above, because it names configuration, plugins, the
    schema revision and the bind — which together describe the installation well enough to
    plan against it.
    """
    del caller
    return await respond("doctor", service, service.doctor)


@router.get("/api/v1/stats", name="stats", summary="Counts, grouped three ways.")
async def stats(service: Service, caller: ViewerPrincipal) -> Response:
    """Documents and chunks by source, media type and status."""
    del caller
    return await respond("stats", service, service.stats)


@router.get(
    "/api/v1/workspaces", name="workspace_list", summary="Workspaces this installation knows about."
)
async def workspaces(service: Service, caller: ViewerPrincipal) -> Response:
    """Every workspace, with the active one marked.

    Counts are reported for the active workspace only. This process is scoped to one tenant
    and cannot read another's rows, including to count them.
    """
    del caller
    return await respond("workspace_list", service, service.workspace_list)


__all__ = ["router"]
