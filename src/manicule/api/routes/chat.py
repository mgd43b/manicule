"""Asking a question: settled, streamed over SSE, and rated afterwards.

The three routes here are one operation seen three ways. ``POST /chat`` waits for the answer,
``POST /chat/stream`` sends the same answer as it is written, and both end with the **same
envelope** — the streamed one as its last frame. That is not a convenience: a client that
cannot stream must not be on a different code path with different guarantees, and the only way
to keep that true is for the settled payload to be built once, by the service, for both.

Feedback is here rather than under conversations because it rates an *answer*: a message
exists for every generation, including a truncated or failed one, and those are exactly the
answers most worth rating.
"""

from __future__ import annotations

from fastapi import APIRouter, Response
from fastapi.responses import StreamingResponse

from manicule.api.context import Service
from manicule.api.envelopes import respond
from manicule.api.models import AskBody, FeedbackBody
from manicule.api.security import MemberPrincipal
from manicule.api.streaming import SSE_HEADERS, SSE_MEDIA_TYPE, answer_frames, sse

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.post("", summary="Answer a question, and wait for it.")
async def chat(service: Service, caller: MemberPrincipal, body: AskBody) -> Response:
    """One answer with citations that resolve.

    A member rather than a viewer, because answering writes: the turn is persisted when a
    conversation is named, and the retrieval is recorded as telemetry either way.
    """
    del caller
    return await respond(
        "ask",
        service,
        lambda: service.ask(
            body.question,
            profile=body.profile,
            limit=body.limit,
            sources=body.sources,
            conversation_id=body.conversation_id,
        ),
    )


@router.post("/stream", summary="The same answer, as it is written.")
async def chat_stream(
    service: Service, caller: MemberPrincipal, body: AskBody
) -> StreamingResponse:
    """Server-sent events: ``delta``, ``citation``, ``drop``, and one ``final``.

    The ``final`` frame is the ordinary envelope — the identical bytes ``POST /chat`` would
    have returned. A client that reads only that frame has made the non-streaming call.

    **Abandonment is not swallowed.** If the caller disconnects, the consuming task is
    cancelled, the generator is closed, and the answer path's own cleanup releases the open
    response to the model. A stream that caught cancellation to "tidy up" would leave a
    provider connection generating tokens nobody will read.
    """
    del caller

    async def frames() -> object:
        async for name, payload in answer_frames(
            service,
            question=body.question,
            profile=body.profile,
            limit=body.limit,
            sources=body.sources,
            conversation_id=body.conversation_id,
        ):
            yield sse(name, payload)

    return StreamingResponse(
        frames(),  # pyright: ignore[reportArgumentType] - an async generator of str is a valid body
        media_type=SSE_MEDIA_TYPE,
        headers=dict(SSE_HEADERS),
    )


@router.post("/feedback", summary="Rate one answer.")
async def feedback(service: Service, caller: MemberPrincipal, body: FeedbackBody) -> Response:
    """Rate an answer by its message id.

    A rating on an id that matched nothing is **refused** rather than accepted. Silently
    accepting it is how feedback disappears into a table nobody can join.
    """
    del caller
    return await respond(
        "chat_feedback",
        service,
        lambda: service.chat_feedback(
            body.message_id,
            feedback=body.feedback,
            reason=body.reason,
            comment=body.comment,
        ),
    )


__all__ = ["router"]
