"""Building the real application over a fake backend.

The application under test is the **production** one: real routing, real dependency
resolution, real middleware, real exception handlers. Only the backend is a fake, and the
fakes it is made of are the ones :mod:`tests.app.fakes` already provides — including the two
that are deliberately broken, which is what lets the tenancy suite here see the surface's own
guard fire rather than the store's.

The client sets a peer address explicitly. Starlette's test client defaults to ``testclient``,
which is not an address at all, so a trusted-proxy suite driven through the default would be
testing what happens when the peer is unparseable — which is a real case and not the one that
matters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi.testclient import TestClient

from manicule.api.app import build_app
from manicule.app.service import ApplicationService
from manicule.config.settings import Settings
from tests.app.fakes import FakeBackend, make_chunk, make_document

if TYPE_CHECKING:
    from fastapi import FastAPI

    # What Starlette's test client returns. Imported under TYPE_CHECKING and from the package
    # the client is written against, so the suites are typed without manicule itself gaining a
    # second HTTP client.
    from httpx2 import Response

    from manicule.core.content import Document

WORKSPACE = "default"

LOCAL_PEER = "127.0.0.1"
"""The peer the test client presents unless a test says otherwise."""


def backend_with_a_document(**overrides: Any) -> tuple[FakeBackend, Document]:
    """A backend holding one indexed document, and the document.

    The document's id is *derived* the way the real one is, so a tenancy assertion cannot be
    made to pass by editing a literal until it matches.
    """
    settings = Settings(**overrides) if overrides else Settings()
    backend = FakeBackend(settings=settings)
    backend.store.workspace_id = settings.workspace
    backend.organisation_.workspace_id = settings.workspace
    backend.keys_.workspace = settings.workspace
    document = make_document(settings.workspace)
    backend.store.add(document, make_chunk(document))
    backend.organisation_.documents[document.id] = document
    return backend, document


def app_for(backend: FakeBackend) -> FastAPI:
    """The production application, over this backend."""
    return build_app(ApplicationService(backend))


def client_for(backend: FakeBackend, *, peer: str = LOCAL_PEER) -> TestClient:
    """A test client whose requests arrive from ``peer``.

    ``client=(host, port)`` is what fills ``request.client``, which is the socket peer every
    address decision starts from. Without it the peer is the literal string ``testclient`` and
    no allowlist can ever match it.
    """
    return TestClient(app_for(backend), client=(peer, 41234))


def envelope(response: Response) -> dict[str, Any]:
    """The parsed envelope of a response, asserting it is one.

    Every route in this surface returns an envelope, including on a failure, so a test that
    reads ``response.json()["data"]`` directly would silently pass on a body that is not one.
    """
    payload: dict[str, Any] = response.json()
    assert set(payload) == {"version", "op", "ok", "workspace", "data", "error"}, payload
    return payload


__all__ = [
    "LOCAL_PEER",
    "WORKSPACE",
    "app_for",
    "backend_with_a_document",
    "client_for",
    "envelope",
]
