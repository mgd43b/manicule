"""The two probes, which are the only routes that answer something other than a person.

They answer different questions and that is the whole reason there are two. ``/healthz`` says
the process is up and opens nothing, so it cannot be made to fail by a database that is busy.
``/readyz`` says the index is usable, which means asking the store, which means it *can* fail
while the process is perfectly alive.

Collapsing them gives you a liveness probe that restarts a healthy process because a database
file was briefly locked — so each is asserted against a backend where the other's answer would
be wrong.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from manicule.core.errors import ManiculeError
from tests.api.support import backend_with_a_document, client_for
from tests.app.fakes import FakeStore

if TYPE_CHECKING:
    from collections.abc import Collection

    from manicule.core.content import DocumentStatus

OK = 200
SERVER_ERROR = 500


class UnreachableStore(FakeStore):
    """A store that cannot be counted. **Deliberately broken.**

    What a locked database, a missing file or a half-applied migration looks like from the
    surface: the process is fine and the store is not.
    """

    @override
    async def count_documents(
        self,
        *,
        source: str | None = None,
        statuses: Collection[DocumentStatus] | None = None,
        glossary_fp_other_than: str | None = None,
        glossary_fp_unrecorded: bool = False,
    ) -> int:
        del source, statuses, glossary_fp_other_than, glossary_fp_unrecorded
        msg = "database is locked"
        raise ManiculeError(msg)


def test_liveness_answers_while_the_store_is_unreachable() -> None:
    """The property that makes it a *liveness* probe.

    If this failed when the store did, an orchestrator would restart a process whose only
    problem is a busy database — and restarting it does not unbusy the database.
    """
    backend, _ = backend_with_a_document()
    backend.store = UnreachableStore(workspace_id=backend.settings.workspace)
    with client_for(backend) as client:
        response = client.get("/healthz")
    assert response.status_code == OK
    assert response.json() == {"status": "ok"}


def test_readiness_fails_while_the_store_is_unreachable() -> None:
    """The other half. A readiness probe that never fails is not a readiness probe."""
    backend, _ = backend_with_a_document()
    backend.store = UnreachableStore(workspace_id=backend.settings.workspace)
    with client_for(backend) as client:
        response = client.get("/readyz")
    assert response.status_code == SERVER_ERROR
    assert response.json() == {"status": "unready"}


def test_readiness_answers_when_the_store_is_reachable() -> None:
    """The positive control, without which "always unready" would pass the test above."""
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        response = client.get("/readyz")
    assert response.status_code == OK
    assert response.json() == {"status": "ready"}


def test_neither_probe_discloses_anything_about_the_corpus() -> None:
    """Both are unauthenticated, so both are readable by whatever can reach the port.

    A probe that said "412 documents, chunker structural" would be a corpus fingerprint; so,
    more quietly, would the text of a database error, which is why the unready body carries no
    reason.
    """
    backend, _ = backend_with_a_document()
    backend.store = UnreachableStore(workspace_id=backend.settings.workspace)
    with client_for(backend) as client:
        bodies = [client.get("/healthz").text, client.get("/readyz").text]
    for body in bodies:
        assert "locked" not in body
        assert "document" not in body.lower()
        assert set(body) <= set('{}"abcdefghijklmnopqrstuvwxyz:_'), body


def test_the_probes_answer_without_an_envelope() -> None:
    """Deliberately the two exceptions to the one-shape rule.

    A supervisor reads a status code. A liveness check that has to parse an envelope is one
    that reports unhealthy when the serialiser changes.
    """
    backend, _ = backend_with_a_document()
    with client_for(backend) as client:
        for path in ("/healthz", "/readyz"):
            body: dict[str, object] = client.get(path).json()
            assert set(body) == {"status"}
