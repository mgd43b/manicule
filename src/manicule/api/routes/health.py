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
from manicule.core.errors import ManiculeError

router = APIRouter(tags=["health"])


@router.get("/healthz", summary="Liveness. Opens nothing.")
async def healthz(caller: AnonymousPrincipal) -> Response:
    """Whether this process is up. Deliberately unauthenticated and deliberately cheap.

    It touches no store, so it cannot be made to fail by a database that is busy — which is
    the whole point of a liveness probe as distinct from a readiness one.
    """
    del caller
    return JSONResponse(content={"status": "ok"}, status_code=OK)


@router.get("/readyz", summary="Readiness. Asks the index whether it can serve.")
async def readyz(service: Service, caller: AnonymousPrincipal) -> Response:
    """Whether this installation can actually answer a question.

    Counts documents, which opens the store and runs a query. A failure here is a **503 in
    spirit and a 500 in status**: the process is alive and the index is not usable, and a
    probe that reads the status code learns the right thing either way.

    It reports no counts and no configuration. An unauthenticated endpoint that said "412
    documents, chunker structural" would be a corpus fingerprint for anyone who can reach the
    port.
    """
    del caller
    try:
        store = await service.backend.documents()
        await store.count_documents()
    except (ManiculeError, ValueError, OSError) as exc:
        return JSONResponse(
            content={"status": "unready", "detail": f"{type(exc).__name__}: {exc}"},
            status_code=SERVER_ERROR,
        )
    return JSONResponse(content={"status": "ready"}, status_code=OK)


@router.get("/api/v1/health", summary="Diagnostics, in the ordinary envelope.")
async def health(service: Service, caller: ViewerPrincipal) -> Response:
    """Everything ``manicule doctor`` checks, as the same envelope every other surface emits.

    Authenticated, unlike the two probes above, because it names configuration, plugins, the
    schema revision and the bind — which together describe the installation well enough to
    plan against it.
    """
    del caller
    return await respond("doctor", service, service.doctor)


@router.get("/api/v1/stats", summary="Counts, grouped three ways.")
async def stats(service: Service, caller: ViewerPrincipal) -> Response:
    """Documents and chunks by source, media type and status."""
    del caller
    return await respond("stats", service, service.stats)


@router.get("/api/v1/workspaces", summary="Workspaces this installation knows about.")
async def workspaces(service: Service, caller: ViewerPrincipal) -> Response:
    """Every workspace, with the active one marked.

    Counts are reported for the active workspace only. This process is scoped to one tenant
    and cannot read another's rows, including to count them.
    """
    del caller
    return await respond("workspace_list", service, service.workspace_list)


__all__ = ["router"]
