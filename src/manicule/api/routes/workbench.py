"""The workbench: one document, as the index actually holds it.

A single read-only endpoint, and it earns its place by answering a question nothing else can.
A ranked list that looks wrong has two possible causes — the retrieval ordered badly, or the
chunker split the document somewhere that destroyed the passage — and they are
indistinguishable from the outside. The workbench shows the units: every stored chunk, in
order, with its token count and the anchor that locates it.

It invents nothing. The blocks are the chunks retrieval scores, read through the same
workspace-scoped lookup ``GET /documents/{id}`` uses, so what is displayed here is what is
indexed rather than a re-derivation that might chunk differently today.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Response

from manicule.api.context import Service
from manicule.api.envelopes import respond
from manicule.api.security import ViewerPrincipal

router = APIRouter(prefix="/api/v1", tags=["workbench"])


@router.get("/workbench", summary="One document as it was chunked.")
async def workbench(
    service: Service,
    caller: ViewerPrincipal,
    document_id: Annotated[str, Query(min_length=1, description="The document to inspect.")],
) -> Response:
    """Every stored chunk of one document, with its anchor and token count.

    One document at a time, by required parameter rather than by an optional one that would
    otherwise default to "all of them" — this returns full passage text, and a corpus-wide
    version of it is an export by another name.
    """
    del caller
    return await respond("workbench", service, lambda: service.workbench(document_id))


__all__ = ["router"]
