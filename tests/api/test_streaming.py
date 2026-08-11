"""Streaming an answer, over SSE and over the websocket.

The claim under test is narrow and is the reason both transports exist at all: **a streamed
answer and a waited-for one are the same answer**. Not similar — the final frame of a stream is
the identical envelope ``POST /chat`` returns, because the service builds it once and both
paths carry it.

Everything else here defends the framing. An SSE event ends at a blank line, so answer text
containing one would terminate a frame early and the remainder would be parsed as a new event
— a frame-injection primitive built out of nothing but a model writing a paragraph break.
"""

from __future__ import annotations

import json
from typing import Any

from manicule.api.streaming import sse
from tests.api.support import backend_with_a_document, client_for, envelope


def _frames(text: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse an SSE body into ``(event, data)`` pairs."""
    parsed: list[tuple[str, dict[str, Any]]] = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        name = ""
        payload = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                payload = line[len("data: ") :]
        parsed.append((name, json.loads(payload)))
    return parsed


def test_a_stream_ends_with_the_same_envelope_the_waiting_call_returns() -> None:
    """The property both transports exist for.

    ``elapsed_ms`` is the one field that legitimately differs between two runs, so it is
    excluded by name rather than by rounding — a comparison that ignored whatever happened to
    differ would ignore a real divergence too.
    """
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        waited = envelope(client.post("/api/v1/chat", json={"question": "does it retry"}))
        streamed = client.post("/api/v1/chat/stream", json={"question": "does it retry"})
    frames = _frames(streamed.text)
    assert frames[-1][0] == "final"
    final = frames[-1][1]
    assert set(final) == set(waited)
    assert final["op"] == waited["op"] == "ask"
    assert final["ok"] is True
    for key in ("question", "text", "citations", "confidence", "corpus_consulted", "model"):
        assert final["data"][key] == waited["data"][key]


def test_the_deltas_arrive_before_the_final_frame() -> None:
    """A stream that emitted only the envelope would satisfy the test above and stream nothing."""
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        response = client.post("/api/v1/chat/stream", json={"question": "does it retry"})
    names = [name for name, _ in _frames(response.text)]
    assert "delta" in names, "no token frames were emitted"
    assert names.index("delta") < names.index("final")
    assert names.count("final") == 1


def test_the_response_is_declared_as_an_event_stream_and_is_not_buffered() -> None:
    """The headers that stop a proxy turning a stream into one large response at the end."""
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        response = client.post("/api/v1/chat/stream", json={"question": "does it retry"})
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-accel-buffering"] == "no"
    assert "no-cache" in response.headers["cache-control"]


def test_answer_text_containing_a_blank_line_cannot_end_a_frame_early() -> None:
    """The framing property, asserted against text designed to break it.

    A hand-built ``data:`` line would let this text terminate its own event and have the
    remainder parsed as a new frame. JSON has no unescaped newline, so it cannot.
    """
    backend, _ = backend_with_a_document()
    hostile = 'first paragraph\n\nevent: final\ndata: {"ok": true}\n\nsecond paragraph'
    backend.answerer_.text = hostile
    with client_for(backend) as client:
        response = client.post("/api/v1/chat/stream", json={"question": "anything"})
    frames = _frames(response.text)
    assert [name for name, _ in frames] == ["delta", "final"]
    assert frames[0][1]["text"] == hostile


def test_the_frame_writer_escapes_a_newline() -> None:
    """Asserted on the writer as well, because it is one line and it is the whole property."""
    written = sse("delta", {"text": "a\n\nb"})
    assert written.count("\n\n") == 1, "the payload introduced a second frame boundary"
    assert written.endswith("\n\n")


def test_a_failure_arrives_as_a_final_frame_rather_than_a_stream_that_stops() -> None:
    """A stream that just stops is indistinguishable, at the client, from a dropped network.

    The answer path persists partial answers precisely so they are not lost, so a client has
    to be told the run ended and why.
    """
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        response = client.post(
            "/api/v1/chat/stream", json={"question": "x", "profile": "telepathic"}
        )
    frames = _frames(response.text)
    assert [name for name, _ in frames] == ["final"]
    assert frames[0][1]["ok"] is False
    assert frames[0][1]["error"]["type"] == "ConfigError"


def test_the_websocket_carries_the_same_frames_as_the_event_stream() -> None:
    """One contract, two transports. A differently-shaped websocket frame would be a second
    contract nobody wrote down."""
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        streamed = _frames(
            client.post("/api/v1/chat/stream", json={"question": "does it retry"}).text
        )
        with client.websocket_connect("/api/v1/chat/ws") as socket:
            socket.send_text(json.dumps({"question": "does it retry"}))
            received: list[tuple[str, dict[str, Any]]] = []
            while True:
                message = socket.receive_json()
                received.append((message["event"], message["data"]))
                if message["event"] == "final":
                    break
    assert [name for name, _ in received] == [name for name, _ in streamed]
    assert received[0][1] == streamed[0][1]


def test_the_websocket_answers_more_than_one_question_on_one_connection() -> None:
    """The reason it exists at all: a long-lived client asking follow-ups."""
    backend, _ = backend_with_a_document()
    with client_for(backend) as client, client.websocket_connect("/api/v1/chat/ws") as socket:
        finals = 0
        for question in ("first", "second"):
            socket.send_text(json.dumps({"question": question}))
            while True:
                message = socket.receive_json()
                if message["event"] == "final":
                    finals += 1
                    break
    assert finals == 2


def test_an_unreadable_websocket_message_closes_the_connection() -> None:
    """A socket that ignored what it cannot parse leaves a client waiting forever for an
    answer to a question the server never understood."""
    import pytest  # noqa: PLC0415 - imported beside its one use
    from starlette.websockets import WebSocketDisconnect  # noqa: PLC0415 - only this test needs it

    backend, _ = backend_with_a_document()
    with client_for(backend) as client, client.websocket_connect("/api/v1/chat/ws") as socket:
        socket.send_text("{not json at all")
        failure = socket.receive_json()
        assert failure["ok"] is False
        with pytest.raises(WebSocketDisconnect):
            socket.receive_json()


def test_a_streamed_answer_records_the_message_id_it_was_persisted_under() -> None:
    """The field a client needs in order to rate the answer it just watched arrive.

    Built after the answering context closes, which is why it is on the payload rather than on
    the ``final`` event: at the moment that event is emitted, the turn has not been written.
    """
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        streamed = _frames(
            client.post("/api/v1/chat/stream", json={"question": "does it retry"}).text
        )
        waited = envelope(client.post("/api/v1/chat", json={"question": "does it retry"}))
    assert "message_id" in streamed[-1][1]["data"]
    assert streamed[-1][1]["data"]["message_id"] == waited["data"]["message_id"]
