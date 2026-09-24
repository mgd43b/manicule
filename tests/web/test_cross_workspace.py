"""The search page's workspace picker: offered to an administrator, and to nobody else.

The page takes the floor its route takes, so a spanning search from a viewer is refused as a
page however the URL was made; the service's own refusal is behind that. What the page adds is
the rendering — which workspaces an administrator may pick, and which workspace each hit of a
merged ranking came from — asserted on the HTML, because a page is where a hit is read.
"""

from __future__ import annotations

import asyncio

from manicule.app.service import ApplicationService
from tests.api.support import client_for
from tests.app.fakes import SpanningBackend, spanning_backend

AUTHENTICATED = {"auth": {"mode": "api_key"}}
"""``security`` with authentication on, so a caller's role comes from the key it presents."""
OK = 200
FORBIDDEN = 403
PICKER = 'name="workspaces"'


def _secret(role: str) -> tuple[SpanningBackend, str]:
    backend = spanning_backend(security=AUTHENTICATED)
    issued = asyncio.run(ApplicationService(backend).api_key_create(f"{role}-key", role=role))
    return backend, issued.secret


def test_an_administrator_is_offered_every_workspace_to_search_together() -> None:
    """The local operator is an administrator, so the picker lists this data directory's."""
    with client_for(spanning_backend()) as client:
        response = client.get("/ui/search")

    assert response.status_code == OK
    assert PICKER in response.text
    assert '<option value="alpha"' in response.text
    assert '<option value="beta"' in response.text


def test_a_viewer_is_offered_no_picker_and_is_refused_a_spanning_search() -> None:
    """The picker is absent, and a hand-made URL is refused rather than served.

    The ordinary search still works for the same reader — the positive control that keeps "the
    page refuses a viewer everything" from passing this test.
    """
    backend, secret = _secret("viewer")
    headers = {"X-API-Key": secret}

    with client_for(backend) as client:
        blank = client.get("/ui/search", headers=headers)
        spanning = client.get(
            "/ui/search",
            params={"q": "runbook", "workspaces": ["alpha", "beta"]},
            headers=headers,
        )
        ordinary = client.get("/ui/search", params={"q": "runbook"}, headers=headers)

    assert blank.status_code == OK
    assert PICKER not in blank.text
    assert spanning.status_code == FORBIDDEN
    assert "Beta runbook" not in spanning.text
    assert ordinary.status_code == OK
    assert "Alpha runbook" in ordinary.text


def test_every_hit_of_a_spanning_search_names_its_workspace() -> None:
    """A merged ranking is read line by line, and each line says where it came from.

    Beta's passage is named rather than linked: a document page is the serving workspace's, and
    a link to beta's document would open a page that cannot find it.
    """
    backend = spanning_backend()
    beta_hit = backend.spanning.across[0][1]

    with client_for(backend) as client:
        response = client.get(
            "/ui/search", params={"q": "runbook", "workspaces": ["alpha", "beta"]}
        )

    assert response.status_code == OK
    assert '<span class="pill" title="Workspace">beta</span>' in response.text
    assert '<span class="pill" title="Workspace">alpha</span>' in response.text
    assert f"/ui/documents/{beta_hit.chunk.document_id}" not in response.text
    assert '<option value="beta" selected' in response.text, "the picker forgot what ran"


def test_an_ordinary_search_labels_nothing() -> None:
    """Every hit of an ordinary search is this workspace's, so a label on each is only noise."""
    with client_for(spanning_backend()) as client:
        response = client.get("/ui/search", params={"q": "runbook"})

    assert response.status_code == OK
    assert 'title="Workspace">alpha</span>' not in response.text.split("<main", 1)[-1]
