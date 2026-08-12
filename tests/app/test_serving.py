"""``--no-web`` actually removes the browser surface, and no other flag is swallowed.

Two suites, and the second is the point of the first.

``--no-web`` was accepted and discarded from the moment it was added. It was a reasonable
placeholder when there was no browser surface to suppress — the command body said as much, and
said it honestly. Then the browser surface was mounted unconditionally and nothing connected
the two, so a flag whose entire purpose is to *reduce* what a process exposes silently stopped
doing it: an operator who passed ``--no-web`` was served every ``/ui`` page anyway.

So the first suite asserts on **what the server answers**, not on a log line or a parameter
value: a flag that suppresses a surface is only suppressing it if the surface is gone from the
response. And the second suite is the structural check that would have caught the decay the
moment it happened — a command that names an option and then discards it is a command whose
interface is a claim nothing tests.
"""

from __future__ import annotations

import ast
import inspect
import io
import textwrap
from pathlib import Path
from typing import cast

import pytest
from fastapi.testclient import TestClient
from rich.console import Console

from manicule.api.app import build_app
from manicule.api.serve import TRANSPORT as API_TRANSPORT
from manicule.app.results import ServerAddress
from manicule.app.service import ApplicationService
from manicule.cli import main as cli_main
from manicule.cli import render
from tests.api.support import LOCAL_PEER, backend_with_a_document

NOT_FOUND = 404
OK = 200

UI_PATHS: tuple[str, ...] = (
    "/ui",
    "/ui/chat",
    "/ui/documents",
    "/ui/search?q=retry",
    "/ui/health",
    "/ui/settings",
    "/ui/auth",
    "/ui/static/manicule.css",
    "/ui/static/manicule.js",
)
"""Enough of the surface to catch a partial mount.

The two static assets are here deliberately. They are served by the same router as the pages,
so a mount that dropped the pages and kept the assets — or the reverse — would be a surface
that is half absent, which is not what the flag promises.
"""


def _client(*, web: bool) -> TestClient:
    backend, _ = backend_with_a_document()
    app = build_app(ApplicationService(backend), web=web)
    return TestClient(app, client=(LOCAL_PEER, 41234))


@pytest.mark.parametrize("path", UI_PATHS)
def test_no_web_leaves_no_ui_path_served(path: str) -> None:
    """With the browser surface off, every ``/ui`` path is simply not there.

    404 rather than a redirect or an empty page: the claim ``--no-web`` makes is that the
    surface is not mounted, and a route that answers anything at all is mounted.
    """
    response = _client(web=False).get(path)
    assert response.status_code == NOT_FOUND, (
        f"{path} answered {response.status_code} with the browser surface switched off. "
        "--no-web claims the surface is not served; a response means it is."
    )


@pytest.mark.parametrize("path", UI_PATHS)
def test_the_same_paths_are_served_when_the_web_surface_is_on(path: str) -> None:
    """The mirror of the above, so the first suite cannot pass by breaking the surface.

    Without this, deleting the browser surface entirely would turn every assertion above
    green — which is the failure mode of every test that only asserts an absence.
    """
    response = _client(web=True).get(path)
    assert response.status_code == OK, (
        f"{path} answered {response.status_code} with the browser surface on."
    )


def test_the_api_is_still_served_without_the_browser_surface() -> None:
    """``--no-web`` removes the browser surface and nothing else.

    A flag that switched off more than it named would be its own defect, and the obvious way
    to make the suite above pass is to mount nothing at all.
    """
    response = _client(web=False).get("/healthz")
    assert response.status_code == OK, "--no-web took the JSON API with it"


def _swallowed_options(function: object) -> list[str]:
    """Parameters a command deletes without using, by reading its body.

    ``del x`` on a parameter is how a Typer command says "this option is accepted and ignored".
    That is sometimes deliberate and always worth a test's attention, because the option is on
    ``--help`` either way and a reader has no way to tell the difference.
    """
    # Dedented, because a nested function's source arrives indented and `ast.parse` refuses it.
    source = textwrap.dedent(inspect.getsource(function))  # type: ignore[arg-type]
    tree = ast.parse(source)
    body = tree.body[0]
    if not isinstance(body, ast.FunctionDef):  # pragma: no cover - every command is one
        return []
    names = {argument.arg for argument in [*body.args.args, *body.args.kwonlyargs]}
    deleted: list[str] = []
    for node in ast.walk(body):
        if isinstance(node, ast.Delete):
            deleted.extend(
                target.id
                for target in node.targets
                if isinstance(target, ast.Name) and target.id in names
            )
    return deleted


ALLOWED_TO_BE_DISCARDED: frozenset[tuple[str, str]] = frozenset({("main_callback", "version")})
"""The one parameter a command may delete, and why.

``--version`` is handled by an eager Typer callback that exits before the root callback's body
runs, so the parameter genuinely has nothing to do. Every other deleted parameter is an option
a person can pass, that appears on ``--help``, and that changes nothing.

Keyed by **command and parameter** rather than by name alone. ``upgrade`` also takes a
``version``, and it is a real option that is passed through — an exemption keyed only on the
name would quietly excuse discarding that one too.
"""


def _commands() -> list[tuple[str, object]]:
    """Every command Typer has registered, including the ones under a sub-application.

    Read off the Typer app rather than listed here, so a command added later is checked
    without anyone remembering to add it — which is the failure this whole module is about.
    """
    callbacks = [info.callback for info in cli_main.app.registered_commands]
    for group in cli_main.app.registered_groups:
        typer_app = group.typer_instance
        if typer_app is None:  # pragma: no cover - every group has one
            continue
        callbacks.extend(info.callback for info in typer_app.registered_commands)
    callbacks.append(cli_main.main_callback)
    return [(callback.__name__, callback) for callback in callbacks if callback is not None]


def test_no_command_accepts_an_option_and_throws_it_away() -> None:
    """Every option a command declares reaches something.

    This is the check that was missing. ``--no-web`` was declared, documented on ``--help`` as
    "Do not serve the web UI", and deleted in the first line of the body — for four releases,
    while the web UI it named was mounted unconditionally. Nothing failed, because nothing
    asserted that an accepted option does anything.
    """
    swallowed: dict[str, list[str]] = {}
    for name, callback in _commands():
        discarded = [
            parameter
            for parameter in _swallowed_options(callback)
            if (name, parameter) not in ALLOWED_TO_BE_DISCARDED
        ]
        if discarded:
            swallowed[name] = discarded
    assert not swallowed, (
        f"these commands accept options and discard them: {swallowed}. An option on --help "
        "that reaches nothing is a promise the command does not keep — wire it through, or "
        "remove it from the signature so it stops appearing in the help."
    )


def test_the_swallowed_option_check_can_see_a_swallowed_option() -> None:
    """The detector finds a discard, so the suite above is not green by blindness.

    A structural check that cannot fail is worse than no check, because it reads as coverage.
    """

    def pretend(*, flag: bool = False) -> None:
        del flag

    assert _swallowed_options(pretend) == ["flag"]


def test_the_renderer_names_the_surface_the_transport_says_it_is() -> None:
    """``stop`` names the surface too, because it reads the same field.

    The pid file has always recorded ``http-api`` for the REST API and ``http`` for
    MCP-over-HTTP, and the renderer ignored the difference — so ``manicule stop`` announced
    "MCP server" about the API server it had just stopped. This is the regression test for
    reading it rather than being told.
    """
    console = Console(file=io.StringIO(), width=100, no_color=True, highlight=False)
    render.render_address(
        console,
        ServerAddress(transport=API_TRANSPORT, host="127.0.0.1", port=8765, loopback=True),
    )
    written = cast("io.StringIO", console.file).getvalue()
    assert "HTTP API" in written, written
    assert "MCP server" not in written, "the API server was announced as an MCP server"


def test_the_mirrored_transport_constant_agrees_with_the_api() -> None:
    """``render`` names the API's transport without importing FastAPI to learn it.

    A copied constant is a constant that can drift, so the copy is asserted equal to the
    original here rather than trusted.
    """
    assert render.API_TRANSPORT == API_TRANSPORT


def test_the_help_text_for_no_web_is_not_the_old_claim() -> None:
    """The docstring no longer says the browser surface does not exist.

    It said "the web UI is not part of this build" for the whole time the web UI was part of
    the build. Help text that is false is worse than help text that is missing.
    """
    documentation = inspect.getdoc(cli_main.start) or ""
    assert "not part of this build" not in documentation, (
        "`manicule start --help` still claims the web UI is not built"
    )
    assert Path(cli_main.__file__).exists()
