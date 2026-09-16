"""How the Qdrant suite gets a Qdrant, and when a missing one is a failure.

Two engines answer to ``qdrant-client``, and the difference matters to what a test proves:

- **Local mode** (``AsyncQdrantClient(":memory:")``) is a Python reimplementation of the
  storage and search engine inside this process. It is what the conformance suites run on, so
  that "every vector store passes these" needs no service and no marker — the guarantees those
  suites check are the store's, not the engine's.
- **A real server** is the configuration this backend exists for, and it is the only place
  four things can be observed at all: payload indexes (local mode ignores them and says so),
  HNSW rather than a brute-force scan, a collection's shape changing after it was created
  (local mode ignores every update and says nothing), and the REST transport's JSON rounding of
  a stored ``float32`` — which is the one difference that could make a healthy corpus read as
  corrupt.

:data:`REQUIRE_QDRANT_ENV` is what tells a developer's machine from CI, and it is
**deliberately outside manicule's ``MANICULE_`` namespace**: ``manicule_environment`` deletes
every variable with that prefix before each test, so a switch named that way is scrubbed before
it is ever read and the job goes green having skipped everything.

**Local mode also re-normalizes a vector on write, and the real server does not.** Measured on
500 random unit vectors at 64 dimensions: 47 came back from ``:memory:`` with one component
moved by a single ulp, against 0 from ``qdrant/qdrant:v1.19.1``. It is float64 arithmetic in a
Python reimplementation rather than the server's, so which vectors move depends on the platform.

The consequence for anything checking a *stored* vector: assert numeric fidelity against a
server, and assert against local mode only what is not the vector — counts, point ids, payload
fields. A test that compares stored components against what was written will pass on one machine
and fail on another, for no reason in the code under test.
"""

from __future__ import annotations

import os
from typing import Final

import pytest
from qdrant_client import AsyncQdrantClient

QDRANT_URL_ENV: Final = "QDRANT_TEST_URL"
"""Where a real Qdrant is, for the suite that needs one. Unset means "skip those"."""

REQUIRE_QDRANT_ENV: Final = "REQUIRE_QDRANT"
"""Set to any non-empty value to turn this suite's skips into failures. CI sets it."""

QDRANT_REQUIRED: Final = bool(os.environ.get(REQUIRE_QDRANT_ENV, "").strip())
"""Read at import, before any fixture has had a chance to touch the environment."""

TEST_COLLECTION_PREFIX: Final = "manicule_test"
"""Kept away from the default so a suite pointed at a real server cannot touch a real corpus."""


def local_client() -> AsyncQdrantClient:
    """A Qdrant running inside this process, holding nothing.

    In-memory rather than on a path: a ``path=`` client takes a file lock the next one in the
    same process cannot have, which turns a suite into a flake that depends on collection order.
    """
    return AsyncQdrantClient(":memory:")


def remote_client(url: str) -> AsyncQdrantClient:
    """A client on the real server at ``url``, built the way the product builds one.

    That is, without the client's version check, which the plugin factory also turns off. The
    check is a blocking request on a background thread that *warns* when client and server are
    more than a minor version apart, and this project turns warnings into errors — so a suite
    pointed at the Qdrant an installation actually runs failed on that warning, before reaching
    any property it was written to check. The image CI pins is not the only server worth
    pointing it at.
    """
    return AsyncQdrantClient(url=url, timeout=30, check_compatibility=False)


def server_url() -> str | None:
    """The real server this machine has, if it has one."""
    return os.environ.get(QDRANT_URL_ENV, "").strip() or None


def require_server() -> str:
    """The real server's URL — or skip, or fail under the CI switch.

    A developer with no Qdrant running should skip: nothing is wrong, and failing would make a
    first checkout red for a reason that has nothing to do with the change under test. CI,
    which starts one as an explicit step, must fail instead — a skipped suite certifies
    nothing, and the properties only a server can show are exactly the ones that would
    otherwise be claimed and never checked.
    """
    url = server_url()
    if url:
        return url
    detail = f"{QDRANT_URL_ENV} is not set, so no Qdrant server is available"
    if QDRANT_REQUIRED:
        pytest.fail(
            f"{detail}, and {REQUIRE_QDRANT_ENV} is set. Start a Qdrant and point "
            f"{QDRANT_URL_ENV} at it; a skipped server suite reports green while proving "
            f"nothing about the transport, the payload indexes or the real index."
        )
    pytest.skip(f"{detail}. Start `docker run -p 6333:6333 qdrant/qdrant` and set it to enable.")


__all__ = [
    "QDRANT_REQUIRED",
    "QDRANT_URL_ENV",
    "REQUIRE_QDRANT_ENV",
    "TEST_COLLECTION_PREFIX",
    "local_client",
    "remote_client",
    "require_server",
    "server_url",
]
