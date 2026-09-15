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
from typing import TYPE_CHECKING, cast
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from rich.console import Console
from typer.testing import CliRunner

from manicule.api.app import build_app
from manicule.api.serve import TRANSPORT as API_TRANSPORT
from manicule.app.results import ServerAddress
from manicule.app.service import ApplicationService
from manicule.cli import main as cli_main
from manicule.cli import render, serving
from manicule.config.settings import Settings
from tests.api.support import LOCAL_PEER, backend_with_a_document

if TYPE_CHECKING:
    from manicule.mcp.serve import Transport

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


def _announced(*, loopback: bool, unauthenticated: bool, authoring: str = "memories") -> str:
    """The start banner, rendered as an operator's terminal would receive it."""
    console = Console(file=io.StringIO(), width=100, no_color=True, highlight=False)
    render.render_address(
        console,
        ServerAddress(
            transport=API_TRANSPORT,
            host="127.0.0.1" if loopback else "0.0.0.0",  # noqa: S104 - the address being announced
            port=8765,
            loopback=loopback,
        ),
        unauthenticated=unauthenticated,
        authoring=authoring,
    )
    return cast("io.StringIO", console.file).getvalue()


def test_serving_unauthenticated_is_announced_and_names_the_flag() -> None:
    """A server that asks callers for nothing says so, in the words that produced it.

    **The flag is named because that is the fix.** An operator who did not mean this removes an
    argument from the command they just ran; a warning that described the state without naming
    the argument would send them to the configuration file, where this is deliberately not.

    The other two lines are the consequences that surprise people. Without a credential there is
    no viewer — every caller is an administrator — and the socket carries no write tool, so an
    operator who had configured authoring learns that here rather than from a client reporting
    an unknown tool several minutes later.
    """
    written = _announced(loopback=False, unauthenticated=True)

    assert "--no-authentication" in written, written
    assert "administrator" in written, written
    assert "writes             " in written or "writes  " in written, (
        "the writes line is not padded to the signpost column, which happens when Rich markup "
        "is put in the label: ljust counts characters Rich then strips"
    )
    assert "memories" in written, (
        "the banner does not name the corpus this bind can be written into, which is the fact "
        "the warning exists for — an operator is told the risk, not left to infer it"
    )
    assert "author" in written, written


def test_a_half_configured_install_is_not_warned_about_writes_it_cannot_take() -> None:
    """A source with no collections is not authoring, so nothing may say it can be written into.

    ``AuthoringSettings.configured`` is ``source and collections`` precisely because half of it
    is a state people reach, and in that state ``document_create`` refuses every call naming the
    settings it needs. A banner keyed off the source alone announces an exposure that does not
    exist — and ``doctor`` reports the same condition from ``configured``, so the two would
    disagree about one installation.

    That matters more here than it looks: this is the warning an operator is meant to read on the
    installs where it *is* true, and a line that cries wolf is one they learn to skip.
    """
    written = _announced(loopback=False, unauthenticated=True, authoring="")

    assert "--no-authentication" in written, "the flag itself is still announced"
    assert "author" not in written, written


def test_the_ordinary_banner_makes_no_claim_about_authentication() -> None:
    """The control, without which the assertions above pass against a banner that always warns.

    A warning printed on every start is a warning nobody reads, which would cost exactly the
    thing the test above is for.
    """
    written = _announced(loopback=True, unauthenticated=False)

    assert "--no-authentication" not in written, written
    assert "administrator" not in written, written


def test_the_warning_is_printed_on_loopback_too() -> None:
    """Loopback changes the sentence, not whether it is said.

    ``--no-authentication`` on loopback is the transport somebody is most likely to try it on
    first, and it is not inert there: it waives the authoring refusal, so that socket loses its
    write as well. A flag that printed nothing on the bind people reach for first is a flag whose
    effect they discover later, on the bind where it matters.
    """
    written = _announced(loopback=True, unauthenticated=True)

    assert "--no-authentication" in written, written
    assert "this machine" in written, written


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


def test_the_no_authentication_flag_reaches_the_bind_policy() -> None:
    """The flag is declared, typed, and arrives as ``allow_unauthenticated``.

    **The name a person types and the name the policy reads are deliberately different**, and
    that gap is exactly what this pins. ``--no-authentication`` says what an operator is
    choosing; ``allow_unauthenticated`` says what the bind is being permitted. A rename on
    either side that did not reach the other would leave a flag Typer accepts and nothing acts
    on — which is the failure ``_swallowed_options`` above exists for, and which that check
    cannot see here because the parameter *is* passed on, to a function called under a name this
    module does not resolve.
    """
    captured: dict[str, object] = {}

    def capture(**arguments: object) -> int:
        captured.update(arguments)
        return 0

    with mock.patch("manicule.cli.serving.serve_forever", capture):
        result = CliRunner().invoke(
            cli_main.app, ["serve", "--allow-public-bind", "--no-authentication"]
        )

    assert result.exit_code == 0, result.output
    assert captured["allow_unauthenticated"] is True
    assert captured["allow_public"] is True


def test_the_two_flags_are_independent_at_the_command_line() -> None:
    """Neither flag implies the other, asserted where an operator types them.

    The bind policy keeps them separate (``tests/app/test_bind.py``) and this is the other end of
    that: a command line naming one must not arrive with both set. Otherwise an operator asking
    for a private unauthenticated install would get a public one from the only argument offered.
    """
    captured: dict[str, object] = {}

    def capture(**arguments: object) -> int:
        captured.update(arguments)
        return 0

    with mock.patch("manicule.cli.serving.serve_forever", capture):
        CliRunner().invoke(cli_main.app, ["serve", "--no-authentication"])

    assert captured["allow_unauthenticated"] is True
    assert captured["allow_public"] is False


UNAUTHENTICATED = Settings(security={"auth": {"mode": "none"}})  # pyright: ignore[reportArgumentType]
AUTHENTICATED = Settings(security={"auth": {"mode": "api_key"}})  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("settings", "transport", "flag", "expected", "why"),
    [
        (UNAUTHENTICATED, "http", True, True, "the case the flag exists for"),
        (UNAUTHENTICATED, "http", False, False, "nobody asked"),
        (AUTHENTICATED, "http", True, False, "the flag buys nothing where there is a key"),
        (UNAUTHENTICATED, "stdio", True, False, "a pipe has no port to be reachable on"),
    ],
)
def test_serving_unauthenticated_needs_the_flag_the_mode_and_a_socket(
    settings: Settings, transport: str, flag: bool, expected: bool, why: str
) -> None:
    """All three, and the third is the one that is easy to leave out.

    ``doctor``'s finding says this process "is bound to" an address and that callers "can route
    to the port". On **stdio** both are false whatever ``security.transport.bind_host`` holds —
    that setting is one nothing acted on — so a state computed from the flag and the mode alone
    would have a serving process assert a bind it never made. The middle row is the other
    direction: a warning shown to an operator who has a key and therefore no exposure is how
    warnings stop being read.

    One function rather than an expression at each site, because the banner and the service's
    answer to ``doctor`` both ask this and two expressions of one rule are free to disagree.
    """
    decided = serving.serving_unauthenticated(
        settings, transport=cast("Transport", transport), allow_unauthenticated=flag
    )

    assert decided is expected, why
