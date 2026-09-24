"""A search spanning workspaces, on the two surfaces a network caller reaches.

The service refuses anyone short of an administrator, and that is the rule
(``tests/app/test_cross_workspace.py``). What is held here is that the HTTP route and the MCP
tool also ask for the admin floor when — and only when — the service would call the search an
administrator's, so that the refusal arrives as each surface's ordinary 403 envelope before any
workspace is opened, whatever identity the service has been told it is acting for. Every
refusal has its positive control beside it: a surface that refused every ``workspaces`` would
pass the negatives and be useless.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import pytest
from mcp.types import TextContent

from manicule.app.service import ApplicationService
from manicule.mcp.server import build_server
from tests.api.live import mounted
from tests.api.support import client_for, envelope
from tests.app.fakes import spanning_backend

if TYPE_CHECKING:
    from tests.app.fakes import SpanningBackend

AUTHENTICATED = {"auth": {"mode": "api_key"}}
"""``security`` with authentication on, so a caller's role comes from the key it presents."""
OK = 200
BAD_REQUEST = 400
FORBIDDEN = 403
SPANNING = ["alpha", "beta"]


@pytest.fixture
def keyed() -> tuple[SpanningBackend, dict[str, str]]:
    """Alpha serving beside beta, authentication on, and one secret per role."""
    backend = spanning_backend(security=AUTHENTICATED)
    service = ApplicationService(backend)

    async def issue(role: str) -> str:
        return (await service.api_key_create(f"{role}-key", role=role)).secret

    return backend, {role: asyncio.run(issue(role)) for role in ("viewer", "member", "admin")}


def _search(client: Any, secret: str, **params: Any) -> Any:
    return client.get(
        "/api/v1/search", params={"q": "runbook", **params}, headers={"X-API-Key": secret}
    )


# --- HTTP --------------------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["viewer", "member"])
def test_the_route_refuses_a_spanning_search_below_admin(
    keyed: tuple[SpanningBackend, dict[str, str]], role: str
) -> None:
    """The ordinary 403 envelope, and no workspace opened on the way to it."""
    backend, secrets = keyed

    with client_for(backend) as client:
        response = _search(client, secrets[role], workspaces=SPANNING)

    assert response.status_code == FORBIDDEN
    assert envelope(response)["error"]["type"] == "ForbiddenError"
    assert backend.opened == []


def test_the_route_serves_a_spanning_search_to_an_administrator(
    keyed: tuple[SpanningBackend, dict[str, str]],
) -> None:
    """The positive control, with every hit attributed on the wire."""
    backend, secrets = keyed

    with client_for(backend) as client:
        response = _search(client, secrets["admin"], workspaces=SPANNING)

    assert response.status_code == OK
    data = envelope(response)["data"]
    assert data["workspaces"] == SPANNING
    assert [(hit["title"], hit["workspace"]) for hit in data["hits"]] == [
        ("Beta runbook", "beta"),
        ("Alpha runbook", "alpha"),
    ]


@pytest.mark.parametrize("workspaces", [None, ["alpha"]])
def test_a_viewer_s_ordinary_search_is_not_raised_to_admin(
    keyed: tuple[SpanningBackend, dict[str, str]], workspaces: list[str] | None
) -> None:
    """Naming only this workspace is the ordinary search, so the floor must not fire on it.

    The route asks the service's own question rather than testing whether the parameter is
    present; a floor that fired on ``workspaces=alpha`` would refuse a viewer a search the
    service runs for them as an ordinary one.
    """
    backend, secrets = keyed
    params = {} if workspaces is None else {"workspaces": workspaces}

    with client_for(backend) as client:
        response = _search(client, secrets["viewer"], **params)

    assert response.status_code == OK
    assert envelope(response)["data"]["workspaces"] == ["alpha"]


def test_a_blank_workspace_name_is_the_ordinary_refusal_rather_than_a_server_error() -> None:
    """Asked inside the dispatched call, the floor's own question fails as an envelope."""
    backend = spanning_backend()

    with client_for(backend) as client:
        response = client.get("/api/v1/search", params={"q": "x", "workspaces": ["alpha", " "]})

    assert response.status_code == BAD_REQUEST
    assert envelope(response)["error"]["type"] == "ConfigError"


def test_the_local_operator_spans_workspaces_with_no_credential() -> None:
    """``auth.mode = none`` resolves a loopback caller to an administrator, as it always has."""
    backend = spanning_backend()

    with client_for(backend) as client:
        response = client.get("/api/v1/search", params={"q": "x", "workspaces": SPANNING})

    assert response.status_code == OK
    assert envelope(response)["data"]["workspaces"] == SPANNING


# --- MCP ---------------------------------------------------------------------------------------


def _body(result: Any) -> dict[str, Any]:
    content = result.content[0]
    assert isinstance(content, TextContent)
    return json.loads(content.text)


@pytest.mark.parametrize(("role", "ok"), [("viewer", False), ("member", False), ("admin", True)])
async def test_the_network_tool_asks_for_admin_when_it_is_asked_to_span(
    keyed: tuple[SpanningBackend, dict[str, str]], role: str, ok: bool
) -> None:
    """The mount admits a viewer, because it carries the read surface; this tool asks for more.

    Without its own floor a viewer key would reach, over MCP, a search the HTTP route refuses
    it — the shape ``require_network_member`` was written to close for authoring.
    """
    backend, secrets = keyed

    async with mounted(backend, credential={"X-API-Key": secrets[role]}) as client:
        spanning = _body(
            await client.call_tool("search", {"query": "runbook", "workspaces": SPANNING})
        )
        ordinary = _body(await client.call_tool("search", {"query": "runbook"}))

    assert spanning["ok"] is ok
    if not ok:
        assert spanning["error"]["type"] == "ForbiddenError"
    assert ordinary["ok"] is True, "the floor fired on a search that spans nothing"


async def test_over_stdio_the_operator_spans_workspaces_with_no_floor() -> None:
    """No HTTP request, no principal: the caller is whoever started the process."""
    backend = spanning_backend()

    result = await build_server(ApplicationService(backend)).call_tool(
        "search", {"query": "runbook", "workspaces": SPANNING}
    )

    data = result.structured_content
    assert data is not None
    assert data["ok"] is True
    assert [hit["workspace"] for hit in data["data"]["hits"]] == ["beta", "alpha"]


async def test_the_tool_publishes_the_parameter_and_says_who_may_use_it() -> None:
    """An assistant learns what a tool accepts from its schema and its description only."""
    tools = {
        tool.name: tool
        for tool in await build_server(ApplicationService(spanning_backend())).list_tools()
    }

    search = tools["search"]
    assert "workspaces" in search.parameters["properties"]
    assert "administrator" in (search.description or "")
