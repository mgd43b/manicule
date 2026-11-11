"""``manicule connector login`` for an Outlook source, through the real command line.

Argument parsing, the choice between the Confluence path, Gmail and this one, the sink, rendering
and the exit status are the real thing; the service is the one in ``tests/app/fakes``, and
Microsoft is :class:`~tests.connectors.fake_graph.FakeGraph`. What these tests are for is the
promise the command makes about **where the token may appear**: a file only the caller can read,
or a standard output that is not a terminal, and on no other stream, log or result.
"""

from __future__ import annotations

import re
import stat
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

import manicule.cli.main as cli
from manicule.app.dispatch import run_op
from manicule.app.service import ApplicationService
from manicule.config.settings import ConnectorSettings, Settings
from tests.app.fakes import FakeBackend
from tests.connectors.fake_google_signin import visit
from tests.connectors.fake_graph import ADDRESS, CLIENT_ID, TENANT, FakeGraph

if TYPE_CHECKING:
    from pathlib import Path

    from manicule.app.results import Envelope

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(output: str) -> str:
    """A refusal as the sentence it is: without color, box drawing or its wrapping."""
    stripped = "".join(c for c in _ANSI.sub("", output) if c not in "│╭╮╰╯─")
    return " ".join(stripped.split())


@pytest.fixture
def graph(monkeypatch: pytest.MonkeyPatch) -> FakeGraph:
    """Microsoft, reachable only through the fake; the command line is bound to a service."""
    fake = FakeGraph()
    backend = FakeBackend()
    backend.settings = Settings(
        connectors={
            "work": ConnectorSettings(
                type="outlook",
                options={"account": ADDRESS, "tenant_id": TENANT, "client_id": CLIENT_ID},
            ),
            "wiki": ConnectorSettings(
                type="confluence",
                options={
                    "base_url": "https://wiki.example.com/confluence",
                    "deployment": "server",
                    "auth": "browser_session",
                },
            ),
        }
    )
    service = ApplicationService(backend)

    async def execute(op: str, call: Any) -> Envelope:
        return await run_op(op, service.workspace, lambda: call(service))

    real = ApplicationService.connector_login_outlook

    def browse(url: str) -> bool:
        visit(fake.consent(url))
        return True

    async def through_the_fake(self: ApplicationService, name: str, **kwargs: Any) -> Any:
        return await real(self, name, open_browser=browse, transport=fake.transport(), **kwargs)

    monkeypatch.setattr(cli, "_execute", execute)
    monkeypatch.setattr(ApplicationService, "connector_login_outlook", through_the_fake)

    def configured_type(name: str) -> str:
        return backend.settings.connectors[name].type

    monkeypatch.setattr(cli, "_configured_type", configured_type)
    monkeypatch.delenv("OUTLOOK_REFRESH_TOKEN", raising=False)
    return fake


def login(*argv: str) -> Any:
    # A bounded wait, so a sign-in that never arrives fails the test instead of hanging it.
    return CliRunner().invoke(cli.app, ["connector", "login", *argv, "--timeout", "20"])


def test_a_token_file_is_created_private_and_the_result_never_shows_the_token(
    graph: FakeGraph, tmp_path: Path
) -> None:
    target = tmp_path / "token"

    result = login("work", "--token-file", str(target))

    assert result.exit_code == 0, result.output
    token = target.read_text().strip()
    assert token.startswith("refresh-token-rotated-")
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert token not in result.output
    assert token not in result.stderr
    assert ADDRESS in _plain(result.output)
    assert len(graph.authorizations) == 1


def test_a_second_run_refuses_to_replace_the_file_unless_told_to(
    graph: FakeGraph, tmp_path: Path
) -> None:
    target = tmp_path / "token"
    assert login("work", "--token-file", str(target)).exit_code == 0
    first = target.read_text()

    refused = login("work", "--token-file", str(target))
    replaced = login("work", "--token-file", str(target), "--overwrite-token-file")

    assert refused.exit_code != 0
    assert "already exists" in _plain(refused.output)
    assert replaced.exit_code == 0
    assert target.read_text() != first, "a fresh token replaced the old one"


def test_without_a_file_the_token_goes_to_standard_output_and_the_result_to_standard_error(
    graph: FakeGraph,
) -> None:
    result = login("work")

    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("refresh-token-rotated-")
    assert result.stdout.count("\n") == 1, "standard output carries the token and nothing else"
    assert ADDRESS in _plain(result.stderr)


def test_json_is_refused_when_standard_output_is_where_the_token_goes(graph: FakeGraph) -> None:
    result = CliRunner().invoke(cli.app, ["--json", "connector", "login", "work"])

    assert result.exit_code != 0
    assert "--json" in _plain(result.output)
    assert graph.authorizations == []


def test_confluence_options_on_an_outlook_source_are_refused(graph: FakeGraph) -> None:
    result = login("work", "--browser")

    assert result.exit_code != 0
    assert "Outlook source" in _plain(result.output)
    assert graph.authorizations == []


def test_a_token_file_on_a_confluence_source_is_refused(graph: FakeGraph, tmp_path: Path) -> None:
    result = login("wiki", "--token-file", str(tmp_path / "t"))

    assert result.exit_code != 0
    assert "Gmail or Outlook" in _plain(result.output)


def test_overwrite_without_a_file_is_refused(graph: FakeGraph) -> None:
    result = login("work", "--overwrite-token-file")

    assert result.exit_code != 0
    assert graph.authorizations == []
