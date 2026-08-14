"""``GET /`` answers, in every mode this project can be served in.

Three modes, and a test named for each, because a front door asserted only in the default
configuration is a front door that surprises somebody in the other two. That is the whole reason
this file is not two assertions inside ``tests/api``: ``--mcp-only`` never builds a FastAPI
application at all — :func:`manicule.mcp.serve.serve` hands the MCP server straight to the
library — so the mode most likely to be forgotten is the one an API-shaped suite cannot see.

**Full serve** redirects to ``/ui``. **``--no-web``** and **``--mcp-only``** say what the process
*is* serving, and neither redirects: sending somebody to a browser surface an operator switched
off would be worse than the 404 this replaces, because it spends a second request to arrive at
the same answer having first claimed the thing was somewhere.

The two structural checks at the end are the ones worth reading. The wording tests drive the
application, so they prove the door says the right thing when it is there; nothing they do
proves ``--mcp-only`` *installs* it, because that happens inside the call that starts a server.
So the serving path is read instead — the same technique ``tests/mcp/test_transports.py`` uses
on the read-only rule, and stated here with what it does not cover: it checks that ``serve``
names :func:`~manicule.mcp.serve.signpost`, not that the resulting socket answers.
"""

from __future__ import annotations

import ast
import inspect
import io
import re
import textwrap
from typing import TYPE_CHECKING, Any, cast

import pytest
from fastapi.testclient import TestClient
from rich.console import Console
from starlette.testclient import TestClient as StarletteTestClient

from manicule.api.app import MCP_PATH, build_app
from manicule.app import frontdoor
from manicule.app.results import ServerAddress
from manicule.app.service import ApplicationService
from manicule.cli import render
from manicule.mcp import serve as mcp_serve
from tests.api.support import LOCAL_PEER, backend_with_a_document

if TYPE_CHECKING:
    from httpx2 import Response

OK = 200
TEMPORARY_REDIRECT = 307
PERMANENT_REDIRECTS = frozenset({301, 308})
"""The statuses this door must never answer with. See :data:`frontdoor.TEMPORARY_REDIRECT`."""


def _client(*, web: bool) -> TestClient:
    """The production application, with the browser surface on or off."""
    backend, _ = backend_with_a_document()
    return TestClient(build_app(ApplicationService(backend), web=web), client=(LOCAL_PEER, 41234))


def _front_door(*, web: bool, query: str = "") -> Response:
    """``GET /`` without following the redirect, which is the thing under test."""
    return _client(web=web).get(f"/?{query}" if query else "/", follow_redirects=False)


def _mcp_only() -> StarletteTestClient:
    """The application ``--mcp-only`` serves: the read-only surface with a front door on it.

    Assembled the way :func:`manicule.mcp.serve.serve` assembles it — the same ``surface`` call,
    the same :func:`~manicule.mcp.serve.signpost`, the same path — rather than by starting a
    server on a port. What is under test is the route table, and a socket would add a port to
    every assertion without adding a fact.
    """
    backend, _ = backend_with_a_document()
    server = mcp_serve.surface(ApplicationService(backend), transport="http").server
    mcp_serve.signpost(server)
    return StarletteTestClient(
        server.http_app(path=frontdoor.MCP_ENDPOINT, stateless_http=True, json_response=True)
    )


# --- full serve -------------------------------------------------------------------------------


def test_the_front_door_sends_a_browser_to_the_browser_surface() -> None:
    """``manicule serve --transport http`` prints an address, and the address now goes somewhere.

    It was a 404 — from the process somebody had just started, at the address it had just
    printed. Finding the dashboard meant already knowing it was at ``/ui``, which is the one
    thing a front door exists to make unnecessary.
    """
    response = _front_door(web=True)
    assert response.status_code == TEMPORARY_REDIRECT, response.text
    assert response.headers["location"] == frontdoor.UI


def test_the_redirect_is_temporary_so_a_browser_does_not_keep_it() -> None:
    """A permanent redirect is cached and there is no way to reach out and clear it.

    301 and 308 would make ``/`` → ``/ui`` a decision that outlives the layout on every machine
    that ever visited. The layout has already moved once — MCP arrived on this port in #143 —
    so this is a default rather than a promise, and the status has to say which.
    """
    assert frontdoor.TEMPORARY_REDIRECT == TEMPORARY_REDIRECT
    status = _front_door(web=True).status_code
    # Both, and the first is what keeps the name honest: "not permanent" is also true of a 200,
    # so on its own this would go on passing after the redirect had been removed altogether.
    assert status == TEMPORARY_REDIRECT
    assert status not in PERMANENT_REDIRECTS


def test_the_redirect_carries_the_query_string_rather_than_dropping_it() -> None:
    """A door that silently discards what somebody typed to get through it is a broken door.

    Nothing the dashboard reads arrives this way today, which is why this is worth fixing now:
    the day something does, the failure would be a parameter that vanishes between two URLs
    rather than an error anybody sees.
    """
    response = _front_door(web=True, query="q=retry&limit=5")
    assert response.headers["location"] == f"{frontdoor.UI}?q=retry&limit=5"


def test_the_redirect_lands_on_a_page_that_exists() -> None:
    """Followed, not merely inspected — a signpost to a 404 is worse than no signpost.

    Without this the suite above passes against a ``Location`` naming any path at all, which is
    exactly the failure mode being defended against everywhere else in this file.
    """
    response = _client(web=True).get("/", follow_redirects=True)
    assert response.status_code == OK, response.text
    assert response.headers["content-type"].startswith("text/html")


# --- --no-web ---------------------------------------------------------------------------------


def test_no_web_says_what_is_served_instead_of_redirecting() -> None:
    """The browser surface was switched off on purpose, so ``/`` names what is left.

    A redirect here would send somebody to a second 404 having told them the thing was
    somewhere, which is worse than the first 404 on its own.
    """
    response = _front_door(web=False)
    assert response.status_code == OK, response.text
    assert response.headers["content-type"].startswith("text/plain")


@pytest.mark.parametrize(
    ("what", "path"),
    [("the JSON API", frontdoor.API), ("its documentation", frontdoor.DOCS), ("MCP", MCP_PATH)],
)
def test_no_web_names_every_surface_that_is_still_served(what: str, path: str) -> None:
    """Each one by address, so the next thing an operator does is not a guess."""
    body = _front_door(web=False).text
    assert path in body, f"{what} is served and the front door does not name it:\n{body}"


def test_no_web_does_not_offer_the_browser_surface_it_just_refused_to_serve() -> None:
    """The point of not redirecting, asserted as an absence rather than assumed from the status.

    ``--no-web`` may still be *named* — a person at this address wants to know how to get the
    pages back — but no address under ``/ui`` may appear, because every one of them is a 404 on
    this process.
    """
    body = _front_door(web=False).text
    assert frontdoor.UI not in body, (
        f"the front door pointed at a surface that is not there:\n{body}"
    )
    assert "--no-web" in body, "the front door did not say why there is no browser surface"


def test_no_web_prints_addresses_a_person_can_copy() -> None:
    """Absolute, off the address the request actually arrived on rather than off the bind.

    A bare path is a fragment somebody has to assemble against whatever is in the address bar,
    and through a proxy or a forwarded port the bind is not that address at all.
    """
    body = _front_door(web=False).text
    assert f"http://testserver{frontdoor.MCP_ENDPOINT}" in body, body


# --- --mcp-only -------------------------------------------------------------------------------


def test_mcp_only_says_mcp_is_the_only_thing_here() -> None:
    """The mode where this matters most, and the one no API-shaped suite can see.

    ``--mcp-only`` never builds a FastAPI application, so the front door on it is
    :func:`manicule.mcp.serve.signpost` rather than a route in ``manicule.api.app``. Its
    operator is also the one about to paste an address into a client's configuration, and the
    address the banner used to print was not the one that works.
    """
    with _mcp_only() as client:
        response = client.get("/", follow_redirects=False)
    assert response.status_code == OK, response.text
    assert response.headers["content-type"].startswith("text/plain")
    assert "--mcp-only" in response.text
    assert frontdoor.MCP_ENDPOINT in response.text
    assert frontdoor.UI not in response.text, "there is no browser surface on this process"


def _initialize(client: StarletteTestClient, target: str) -> Response:
    """The first call an MCP client makes, at ``target``, without following a redirect.

    ``follow_redirects=False`` is the whole point. The client follows by default, so a suite
    that let it would report an endpoint reachable at an address that in fact answers 307 — and
    a 307 on a ``POST`` is precisely what the clients that will not re-send a body do not
    survive. What is being asserted is that the *advertised* address answers, not that some
    address does.
    """
    return client.post(
        target,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "front-door-suite", "version": "1"},
            },
        },
        headers={
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        },
        follow_redirects=False,
    )


def test_the_mcp_only_front_door_does_not_shadow_the_endpoint_it_advertises() -> None:
    """The failure this arrangement could have had, driven rather than reasoned about.

    A custom route at ``/`` wins against an MCP mount made at ``/`` — which is what
    ``manicule.api.app`` does, and is why :func:`~manicule.mcp.serve.signpost` is called on the
    ``--mcp-only`` server and on nothing else. Here the endpoint is at ``/mcp/``, so the two do
    not collide; that is a claim about a library's routing, so it is asserted by speaking to it.
    """
    with _mcp_only() as client:
        response = _initialize(client, frontdoor.MCP_ENDPOINT)
    assert response.status_code == OK, response.text
    body: dict[str, Any] = response.json()
    assert body["result"]["protocolVersion"], body


def test_the_advertised_mcp_address_answers_directly_on_both_ways_of_serving_it() -> None:
    """``/mcp/`` is what the banner, the front door, the README and §6.1 all name. It has to work.

    It did not, on ``--mcp-only``. Left to the library's default the endpoint sat at ``/mcp``
    and ``/mcp/`` redirected to it — the opposite direction from the whole server, where the
    mount makes ``/mcp/`` the real path. Both modes answered *something* at both addresses, so
    nothing failed and no browser noticed; what paid for it was a client configured from the
    documentation, sending a ``POST`` that came back a 307.

    Asserted over both ways of serving MCP in one test, because the defect is the two of them
    disagreeing rather than either one being wrong on its own.
    """
    backend, _ = backend_with_a_document()
    whole_server = TestClient(build_app(ApplicationService(backend)), client=(LOCAL_PEER, 41234))
    with whole_server as client:
        mounted = _initialize(client, frontdoor.MCP_ENDPOINT)
    with _mcp_only() as client:
        alone = _initialize(client, frontdoor.MCP_ENDPOINT)

    assert mounted.status_code == OK, mounted.text
    assert alone.status_code == OK, (
        f"POST {frontdoor.MCP_ENDPOINT} answered {alone.status_code} on an --mcp-only server "
        f"(location: {alone.headers.get('location')!r}). That is the address the banner prints "
        f"and the documentation gives out, and a redirect on a POST is what a client that will "
        f"not re-send a body cannot follow."
    )


# --- the serving path actually installs it ------------------------------------------------------


def _calls(function: object) -> set[str]:
    """Every plain function name called in ``function``'s body, read off its source."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))  # type: ignore[arg-type]
    return {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def _keywords(function: object, called: str) -> dict[str, str]:
    """The keyword arguments a named call in ``function``'s body is given, and their source.

    The *value* as written, not merely the name, because "a path was passed" and "the advertised
    path was passed" are different claims and only the second is the one worth keeping.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))  # type: ignore[arg-type]
    return {
        keyword.arg: ast.unparse(keyword.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == called
        for keyword in node.keywords
        if keyword.arg is not None
    }


def test_the_mcp_only_serving_path_installs_the_front_door() -> None:
    """``serve`` is what ``--mcp-only`` runs, and the door is only there if ``serve`` puts it there.

    The tests above build the same server this does and would pass unchanged if the line in
    ``serve`` were deleted, because they call :func:`~manicule.mcp.serve.signpost` themselves.
    This is the gap, and it is closed by reading the serving path rather than by starting one:
    what it proves is that ``serve`` names the function, not that the resulting socket answers.
    """
    assert "signpost" in _calls(mcp_serve.serve), (
        "manicule.mcp.serve.serve no longer installs the front door, so `--mcp-only` is back to "
        "answering 404 at the address it prints"
    )


def test_the_mcp_path_is_passed_to_the_library_rather_than_left_to_its_default() -> None:
    """What is advertised has to be what is served, and a default is not under our control.

    The banner and the front door both print :data:`frontdoor.MCP_ENDPOINT`. If the path were
    left to the library, a release that moved its default would make this process print an
    address it does not answer on — and nothing here would fail.

    The value is checked as well as its presence. Passing ``frontdoor.MCP`` instead would put the
    endpoint one redirect away from the address every other part of this project gives out,
    which is the defect this whole pair of tests exists for and is not caught by "a path was
    passed".
    """
    assert _keywords(mcp_serve.serve, "run_http_async").get("path") == "frontdoor.MCP_ENDPOINT"


# --- the line that tells an operator the address in the first place -----------------------------


def _banner(*, transport: str, web: bool | None = None) -> str:
    """What ``manicule serve`` prints before it starts listening, as written to a terminal."""
    console = Console(file=io.StringIO(), width=100, no_color=True, highlight=False)
    render.render_address(
        console,
        ServerAddress(transport=transport, host="127.0.0.1", port=8765, loopback=True, tools=13),
        web=web,
    )
    return cast("io.StringIO", console.file).getvalue()


def test_the_banner_names_the_mcp_endpoint_when_mcp_is_all_that_is_served() -> None:
    """``--mcp-only`` printed a bare address and a tool count, and the address is not the one.

    The endpoint has a path, the path was in no output anywhere, and the next thing this
    mode's operator does is paste an address into a client's configuration. So they pasted the
    one that was printed, which answers a front door rather than a protocol.
    """
    assert frontdoor.MCP_ENDPOINT in _banner(transport="http"), _banner(transport="http")


@pytest.mark.parametrize("path", [frontdoor.UI, frontdoor.MCP_ENDPOINT, frontdoor.DOCS])
def test_the_banner_names_every_path_a_whole_server_answers_on(path: str) -> None:
    """MCP arrived on this port in #143 and the banner never mentioned it.

    An operator wiring up a client had the port and had to know the path — from the source, or
    from a document they had no reason to be reading at that moment.
    """
    printed = _banner(transport=render.API_TRANSPORT, web=True)
    assert path in printed, f"the startup banner does not name {path}:\n{printed}"


def test_the_banner_says_the_browser_surface_is_off_rather_than_naming_a_path() -> None:
    """``--no-web``: the other lines stay, and the one that would be a lie is replaced."""
    printed = _banner(transport=render.API_TRANSPORT, web=False)
    assert "--no-web" in printed
    assert frontdoor.UI not in printed, f"the banner named a surface that is not served:\n{printed}"
    assert frontdoor.MCP_ENDPOINT in printed, "--no-web took MCP out of the banner with it"


SIGNPOST = re.compile(r"^((?:browser surface|MCP endpoint|API documentation)\s\s+)\S")
"""A signpost line under the address, and everything up to where its value starts.

Anchored on the three labels rather than on "a line with two spaces in it", so the address line
above them — which has a space-separated parenthetical on the end — is not mistaken for one.
"""


@pytest.mark.parametrize("web", [True, False])
def test_the_banner_lines_its_signposts_up_in_one_column(web: bool) -> None:
    """Three labels of different lengths under the address, and their values share a column.

    They did not: ``browser surface`` was padded to one width and ``API documentation`` to
    another, so two addresses under the same heading started one character apart. Asserted
    rather than eyeballed because the way it breaks again is somebody adding a fourth label
    longer than the padding, which reads as *that line* being the broken one.
    """
    printed = _banner(transport=render.API_TRANSPORT, web=web)
    found = [match for line in printed.splitlines() if (match := SIGNPOST.match(line))]
    # Counted first, because "every column agrees" is trivially true of one line — which is what
    # this would degrade to if a label were renamed and stopped matching.
    assert len(found) == 3, (
        f"expected three signposts under the address, got {len(found)}:\n{printed}"
    )
    columns = {len(match.group(1)) for match in found}
    assert len(columns) == 1, f"the signposts start in {len(columns)} different columns:\n{printed}"


# --- one path, one constant ---------------------------------------------------------------------


def test_the_advertised_endpoint_is_where_the_mount_actually_is() -> None:
    """Three readers of one path: the mount, the banner, and the door. A copy could drift."""
    assert MCP_PATH == frontdoor.MCP
    assert frontdoor.MCP_ENDPOINT.rstrip("/") == frontdoor.MCP
    assert frontdoor.MCP_ENDPOINT.endswith("/")
