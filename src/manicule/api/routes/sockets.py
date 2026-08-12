"""The websocket chat channel.

Same operation as the SSE endpoint, same frames, one connection carrying several questions.
It exists for the case SSE genuinely does not cover — a long-lived client asking follow-ups —
and it is deliberately the *only* thing here that is not a plain HTTP route, because every
other route gets FastAPI's dependency machinery and this one does not.

That gap is the reason authentication is written out below rather than delegated. **A
websocket route with no dependency looks exactly like an HTTP route with no dependency**, and
on this surface one of those is a hole. So the handshake resolves a principal and refuses
before ``accept``, and the refusal is a close code rather than a JSON body, because a client
that never completed a handshake has nowhere to read a body from.

**The credential is never a query parameter.** A browser cannot set headers on a ``WebSocket``,
and the usual workaround puts the key in the URL — where it lands in the access log, the
browser history and any ``Referer`` the page sends. manicule reads it from the subprotocol
header instead, which is the one field a browser *can* set, and echoes the chosen subprotocol
back so the handshake completes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from manicule.api.context import policy_of, service_of
from manicule.api.models import AskBody
from manicule.api.proxy import FORWARDED_FOR
from manicule.api.security import Principal, require, websocket_token
from manicule.api.streaming import answer_frames
from manicule.app.dispatch import error_info
from manicule.app.results import failed
from manicule.config.settings import Role
from manicule.core.errors import ManiculeError

if TYPE_CHECKING:
    from manicule.app.service import ApplicationService

router = APIRouter(tags=["chat"])

POLICY_VIOLATION = 1008
"""The websocket close code for "you may not do this". RFC 6455 §7.4.1."""

INVALID_PAYLOAD = 1007
"""The close code for a message this server cannot read."""


# ``name="ask"``, the same as the two HTTP chat routes, because it is the same operation and
# the envelopes it sends say so. Without it the route's name is the handler's — ``chat_socket``
# — which is a word that appears in no other surface's vocabulary. Nothing noticed for a while
# because the check that enumerates route names was walking a shape FastAPI had stopped using.
@router.websocket("/api/v1/chat/ws", name="ask")
async def chat_socket(websocket: WebSocket) -> None:
    """Stream answers over one connection, one question per message.

    Each inbound message is an :class:`~manicule.api.models.AskBody`. Each answer is the same
    frame sequence the SSE endpoint emits — ``delta``, ``citation``, ``drop``, ``final`` — so
    a client that already renders one renders the other.

    A malformed message closes the connection rather than being skipped. A socket that quietly
    ignores what it cannot parse is one where a client waits forever for an answer to a
    question the server never understood.
    """
    service = service_of(websocket)  # pyright: ignore[reportArgumentType] - reads `app.state` only
    policy = policy_of(websocket)  # pyright: ignore[reportArgumentType] - reads `app.state` only
    token, subprotocol = websocket_token(websocket)
    client = websocket.client
    principal = Principal(
        identity=await service.authenticate(token),
        address=policy.client_address(
            peer=client.host if client is not None else None,
            forwarded_for=websocket.headers.get(FORWARDED_FOR),
        ),
    )
    try:
        require(principal, Role.MEMBER)
    except ManiculeError as exc:
        # Closed **before** `accept`, so an unauthenticated caller never reaches a state in
        # which it could send a question. A server that accepted first and closed after would
        # have a window in which the first message was already queued.
        await websocket.close(code=POLICY_VIOLATION, reason=str(exc)[:120])
        return

    await websocket.accept(subprotocol=subprotocol)
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                body = AskBody.model_validate_json(raw)
            except ValidationError as exc:
                await websocket.send_json(
                    failed("ask", service.workspace, error_info(ValueError(str(exc)))).as_json()
                )
                await websocket.close(code=INVALID_PAYLOAD, reason="unreadable message")
                return
            await _answer(websocket, service, body)
    except WebSocketDisconnect:
        # The client went away. Not an error, and not something to log as one: the generator
        # is closed by the cancellation that follows, which is what releases the open response
        # to the model.
        return


async def _answer(websocket: WebSocket, service: ApplicationService, body: AskBody) -> None:
    """Stream one answer down an already-authenticated socket."""
    async for name, payload in answer_frames(
        service,
        question=body.question,
        profile=body.profile,
        limit=body.limit,
        sources=body.sources,
        conversation_id=body.conversation_id,
    ):
        await websocket.send_json({"event": name, "data": payload})


__all__ = ["INVALID_PAYLOAD", "POLICY_VIOLATION", "router"]
