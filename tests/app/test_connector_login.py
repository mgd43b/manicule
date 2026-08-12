"""``connector login``: the operation that captures a Confluence browser session.

Command-line only, deliberately (``docs/surfaces.md`` §4). It mints a credential, and it reads
a secret from a terminal without echoing it — a surface that could not do the second would have
to take the session as a parameter, and a session cookie in a tool call is a session cookie in a
transcript.

What is checked here is the wiring rather than the capture itself: that the operation reaches
the connector's own flow, that it refuses a source with no session to capture, and that the
result it reports carries no part of the credential. The capture and the Keychain are exercised
against a synthetic instance and a real Keychain in ``tests/connectors/test_browser_sso.py``.
"""

from __future__ import annotations

import sys
from io import StringIO
from unittest.mock import patch

import pytest

import manicule.cli.main as cli
from manicule.app.service import ApplicationService
from manicule.config.settings import ConnectorSettings, Settings
from manicule.connectors import sessions
from manicule.connectors.sessions import MemoryStore
from manicule.core.errors import ConfigError, UnknownEntityError
from tests.app.fakes import FakeBackend

SITE = "https://wiki.example.com/confluence"


@pytest.fixture
def configured() -> ApplicationService:
    """A workspace with one Confluence source and one that is not Confluence."""
    backend = FakeBackend()
    backend.settings = Settings(
        connectors={
            "wiki": ConnectorSettings(
                type="confluence",
                options={"base_url": SITE, "deployment": "server", "auth": "browser_session"},
            ),
            "notes": ConnectorSettings(type="filesystem", options={"root": "/srv/notes"}),
        }
    )
    return ApplicationService(backend)


async def test_a_source_that_is_not_configured_lists_what_is(
    configured: ApplicationService,
) -> None:
    with pytest.raises(UnknownEntityError, match=r"notes, wiki"):
        await configured.connector_login("missing", cookies="JSESSIONID=x")


async def test_a_source_with_no_session_to_capture_says_so(
    configured: ApplicationService,
) -> None:
    """A directory has no identity provider in front of it, and no session to hold."""
    with pytest.raises(ConfigError, match="filesystem"):
        await configured.connector_login("notes", cookies="JSESSIONID=x")


async def test_forgetting_removes_the_stored_session(
    configured: ApplicationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stored corporate session with no way to remove it would be a wart of its own."""
    store = MemoryStore()
    monkeypatch.setattr(sessions, "default_store", lambda: store)
    from datetime import UTC, datetime  # noqa: PLC0415

    from manicule.connectors.credentials import BrowserSession  # noqa: PLC0415

    store.save(BrowserSession(base_url=SITE, account="sync.user", captured_at=datetime.now(tz=UTC)))
    outcome = await configured.connector_login("wiki", forget=True)

    assert outcome.forgotten is True
    assert store.load(SITE) is None


async def test_what_is_reported_carries_no_part_of_the_credential(
    configured: ApplicationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The payload is rendered, logged and returned as JSON by ``--json``. None of those is a
    place for a live session against a corporate system."""
    store = MemoryStore()
    monkeypatch.setattr(sessions, "default_store", lambda: store)
    outcome = await configured.connector_login("wiki", forget=True)

    assert "JSESSIONID" not in outcome.model_dump_json()


# --- the command-line adapter ------------------------------------------------------------------


def test_the_command_is_registered_under_connector() -> None:
    """A command written and never attached is in the file and not in the interface."""
    import typer.main  # noqa: PLC0415 - only this assertion needs the built tree

    tree = typer.main.get_command(cli.app)
    connector = getattr(tree, "commands", {})["connector"]

    assert "login" in getattr(connector, "commands", {})


def test_the_session_is_read_from_a_pipe_when_there_is_no_terminal() -> None:
    """So that the flow is scriptable without ever putting the session in an argument."""
    with patch.object(sys, "stdin", StringIO("JSESSIONID=ABC123\n")):
        assert cli.read_secret("paste: ") == "JSESSIONID=ABC123"


def test_pasting_nothing_is_refused_rather_than_stored() -> None:
    """An empty paste that reached capture would be a request with no cookies at all, which an
    instance answers with a sign-in page — a confusing way to be told the paste was empty."""
    with (
        patch.object(sys, "stdin", StringIO("\n")),
        pytest.raises(ConfigError, match="nothing was pasted"),
    ):
        cli.read_secret("paste: ")


def test_the_prompt_says_manicule_does_not_want_the_password() -> None:
    """The likeliest wrong thing to paste is a password, and the prompt is where that is
    prevented rather than diagnosed."""
    assert "Cookie header" in cli.SESSION_PROMPT
    assert "not be echoed" in cli.SESSION_PROMPT
