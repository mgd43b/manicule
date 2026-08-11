"""Turning one answer stream into frames, for the two transports that carry it.

``ask_stream`` yields deltas, citations, drops and one final envelope. SSE and the websocket
carry exactly the same sequence, and the framing is here so they cannot diverge — a websocket
that emitted a differently-shaped citation from the SSE endpoint would be a second contract
nobody wrote down.

Three properties this module is responsible for:

**A stream always ends with an envelope.** Including when generation raised: the failure is
serialised as the same ``ok: false`` envelope every other surface returns and sent as the
final frame. A stream that just stops is indistinguishable, at the client, from a network that
dropped — and the answer path deliberately persists partial answers precisely so they are not
lost.

**Nothing is interpolated into a frame.** Every frame is ``json.dumps`` of a model dump, so a
newline inside answer text cannot terminate an SSE event early. Hand-built ``data:`` lines are
the classic way an SSE endpoint becomes a frame-injection primitive.

**Abandonment propagates.** When the client goes away the consuming task is cancelled, the
generator is closed, and ``ask_stream``'s own ``finally`` runs — which is what releases the
open response to the model. Nothing here swallows ``CancelledError``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from manicule.app.dispatch import error_info
from manicule.app.results import failed, succeeded
from manicule.app.service import AskAside
from manicule.core.errors import ManiculeError
from manicule.generation.answers import EventKind

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from manicule.app.results import Envelope
    from manicule.app.service import ApplicationService
    from manicule.generation.answers import AnswerEvent

SSE_MEDIA_TYPE = "text/event-stream"

SSE_HEADERS = {
    # Proxies and browsers buffer by default, and a buffered token stream arrives all at once
    # at the end — which is a non-streaming endpoint with extra steps.
    "Cache-Control": "no-cache, no-store",
    "Connection": "keep-alive",
    # nginx specifically. Without it a default reverse-proxy configuration buffers the whole
    # response, and the symptom is "streaming works locally and not in production".
    "X-Accel-Buffering": "no",
}


def frame(event: AnswerEvent) -> dict[str, Any]:
    """One stream event as plain JSON data.

    The citation is dumped whole. An anonymous *shared* transcript gets labels instead, and
    that projection happens in storage — this is the authenticated live stream, whose reader
    is a member of the workspace and could have retrieved the passage themselves.
    """
    payload: dict[str, Any] = {"kind": event.kind.value}
    if event.kind is EventKind.DELTA:
        payload["text"] = event.text
    elif event.kind is EventKind.CITATION and event.citation is not None:
        payload["citation"] = event.citation.model_dump(mode="json")
    elif event.kind is EventKind.DROP and event.drop is not None:
        payload["drop"] = event.drop.model_dump(mode="json")
    return payload


def sse(name: str, data: object) -> str:
    """One server-sent event, framed.

    ``json.dumps`` rather than a format string: SSE terminates a frame at a blank line, so
    answer text containing one would end the event early and the remainder would be parsed as
    a new frame with no event name. JSON has no unescaped newline.
    """
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def answer_frames(
    service: ApplicationService,
    *,
    question: str,
    profile: str | None,
    limit: int | None,
    sources: Sequence[str],
    conversation_id: str | None,
) -> AsyncIterator[tuple[str, object]]:
    """Every frame of one answer, as ``(event name, payload)``.

    The final frame is always the envelope, whichever way the generation ended. A failure that
    a caller could act on becomes ``ok: false``; a defect propagates, because a bug dressed as
    a tidy result is a bug nobody fixes — and by then the response has already started, so the
    only honest thing is for the connection to break rather than for the stream to claim it
    finished.
    """
    aside = AskAside()
    try:
        stream = service.ask_stream(
            question,
            profile=profile,
            limit=limit,
            sources=sources,
            conversation_id=conversation_id,
            aside=aside,
        )
        async for event in stream:
            if event.kind is EventKind.FINAL:
                continue
            yield event.kind.value, frame(event)
    except (ManiculeError, ValueError, OSError) as exc:
        yield "final", failed("ask", service.workspace, error_info(exc)).as_json()
        return
    yield "final", _final(service, aside).as_json()


def _final(service: ApplicationService, aside: AskAside) -> Envelope:
    """The settled result, **built by the service** rather than reassembled here.

    ``ask_stream`` fills ``aside.payload`` from the same private builder the non-streaming
    ``ask`` returns, so the streamed final frame and a plain ``POST /chat`` on the same
    question produce identical bytes. A surface that rebuilt the payload would be a second
    answer to "what did this run produce", drifting apart exactly where a consumer compares
    them.
    """
    if aside.payload is None:  # pragma: no cover - the answer path always ends with `final`
        return failed(
            "ask",
            service.workspace,
            error_info(ManiculeError("the answer stream ended without a final event")),
        )
    return succeeded("ask", service.workspace, aside.payload)


__all__ = ["SSE_HEADERS", "SSE_MEDIA_TYPE", "answer_frames", "frame", "sse"]
