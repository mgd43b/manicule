"""``manicule connector login`` for a Gmail source, through the real command line.

Argument parsing, the choice between the Confluence path and this one, the sink, rendering and
the exit status are the real thing; the service is the one in ``tests/app/fakes``, and Google is
:class:`~tests.connectors.fake_google_signin.FakeGoogleSignIn`. What these tests are for is the
promise the command makes about **where the token may appear**: a file only the caller can read,
or a standard output that is not a terminal — and on no other stream, log or result.
"""

from __future__ import annotations

import logging
import stat
import sys
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

import manicule.cli.main as cli
from manicule.app.dispatch import run_op
from manicule.app.service import ApplicationService
from manicule.config.settings import ConnectorSettings, Settings
from tests.app.fakes import FakeBackend
from tests.connectors.fake_google_oauth import CLIENT_SECRET, REFRESH_TOKEN
from tests.connectors.fake_google_signin import FakeGoogleSignIn

if TYPE_CHECKING:
    from pathlib import Path

    from manicule.app.results import Envelope

ACCOUNT = "mail@example.org"


@pytest.fixture
def google(monkeypatch: pytest.MonkeyPatch) -> FakeGoogleSignIn:
    """Google, reachable only through the fake; the command line is bound to a service."""
    fake = FakeGoogleSignIn(address=ACCOUNT)
    backend = FakeBackend()
    backend.settings = Settings(
        connectors={
            "mail": ConnectorSettings(
                type="gmail",
                options={
                    "account": ACCOUNT,
                    "client_id": "1234-example.apps.googleusercontent.com",
                },
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

    real = ApplicationService.connector_login_gmail

    async def through_the_fakes(self: ApplicationService, name: str, **kwargs: Any) -> Any:
        return await real(self, name, open_browser=fake.browse, transport=fake.transport, **kwargs)

    monkeypatch.setattr(cli, "_execute", execute)
    monkeypatch.setattr(ApplicationService, "connector_login_gmail", through_the_fakes)

    def configured_type(name: str) -> str:
        return backend.settings.connectors[name].type

    monkeypatch.setattr(cli, "_configured_type", configured_type)
    monkeypatch.setenv("GMAIL_CLIENT_SECRET", CLIENT_SECRET)
    return fake


def login(*argv: str) -> Any:
    # A bounded wait, so a sign-in that never arrives fails the test instead of hanging it.
    return CliRunner().invoke(cli.app, ["connector", "login", *argv, "--timeout", "20"])


def test_a_token_file_is_created_private_and_the_result_never_shows_the_token(
    google: FakeGoogleSignIn, tmp_path: Path
) -> None:
    target = tmp_path / "token"

    result = login("mail", "--token-file", str(target))

    assert result.exit_code == 0, result.output
    assert google.visits == [200]
    assert target.read_text() == REFRESH_TOKEN + "\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert REFRESH_TOKEN not in result.output
    assert REFRESH_TOKEN not in result.stderr
    assert str(target) in result.output.replace("\n", "")
    assert ACCOUNT in result.output


def test_json_with_a_token_file_prints_an_envelope_without_the_token(
    google: FakeGoogleSignIn, tmp_path: Path
) -> None:
    result = CliRunner().invoke(
        cli.app, ["--json", "connector", "login", "mail", "--token-file", str(tmp_path / "t")]
    )

    assert result.exit_code == 0, result.output
    assert '"ok": true' in result.stdout
    assert REFRESH_TOKEN not in result.stdout


def test_standard_output_that_is_not_a_terminal_carries_the_token_alone(
    google: FakeGoogleSignIn,
) -> None:
    """``manicule connector login mail | secret-store``: the pipe gets the token, nothing else."""
    result = login("mail")

    assert result.exit_code == 0, result.output
    assert result.stdout == REFRESH_TOKEN + "\n"
    # The person's view of the result, and the address to open, went to the other stream.
    assert REFRESH_TOKEN not in result.stderr
    assert "minted a refresh token" in result.stderr
    assert "accounts.google.com" in result.stderr


def test_a_terminal_never_receives_the_token(
    google: FakeGoogleSignIn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With standard output on a terminal and no file, the command refuses and prints nothing."""
    real = cli._gmail_login  # pyright: ignore[reportPrivateUsage]

    def on_a_terminal(*args: Any, **kwargs: Any) -> None:
        # The runner owns `sys.stdout` for the length of the invocation, so the stream the
        # command will inspect is made to claim it is a terminal from inside.
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
        real(*args, **kwargs)

    monkeypatch.setattr(cli, "_gmail_login", on_a_terminal)

    result = login("mail")

    assert result.exit_code != 0
    assert google.authorizations == []  # refused before anybody was sent to a browser
    assert "--token-file" in result.output.replace("\n", " ")
    assert REFRESH_TOKEN not in result.output


def test_no_log_record_carries_the_token(
    google: FakeGoogleSignIn, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)

    result = login("mail", "--token-file", str(tmp_path / "token"))

    assert result.exit_code == 0, result.output
    assert caplog.records, "a log with nothing in it proves nothing"
    assert REFRESH_TOKEN not in "\n".join(r.getMessage() for r in caplog.records)


def test_json_without_a_file_is_refused_rather_than_mixed_with_the_token(
    google: FakeGoogleSignIn,
) -> None:
    result = CliRunner().invoke(cli.app, ["--json", "connector", "login", "mail"])

    assert result.exit_code != 0
    assert REFRESH_TOKEN not in result.output
    assert "--token-file" in result.output.replace("\n", " ")


def test_an_existing_file_is_refused_and_untouched(
    google: FakeGoogleSignIn, tmp_path: Path
) -> None:
    target = tmp_path / "token"
    target.write_text("keep")

    result = login("mail", "--token-file", str(target))

    assert result.exit_code != 0
    assert target.read_text() == "keep"
    assert google.authorizations == []

    replaced = login("mail", "--token-file", str(target), "--overwrite-token-file")
    assert replaced.exit_code == 0, replaced.output
    assert target.read_text() == REFRESH_TOKEN + "\n"


@pytest.mark.parametrize(
    "flag", [["--browser"], ["--manual-cookie"], ["--forget"], ["--browser-state", "x.json"]]
)
def test_the_confluence_options_are_refused_for_a_gmail_source(
    google: FakeGoogleSignIn, flag: list[str]
) -> None:
    result = login("mail", *flag)

    assert result.exit_code != 0
    assert "Gmail" in result.output
    assert google.authorizations == []


def test_the_gmail_options_are_refused_for_a_confluence_source(
    google: FakeGoogleSignIn, tmp_path: Path
) -> None:
    result = login("wiki", "--token-file", str(tmp_path / "token"))

    assert result.exit_code != 0
    assert "not a Gmail source" in result.output.replace("\n", " ")


def test_overwriting_needs_a_file_to_overwrite(google: FakeGoogleSignIn) -> None:
    result = login("mail", "--overwrite-token-file")

    assert result.exit_code != 0
    assert "--token-file" in result.output.replace("\n", " ")


def test_a_gmail_source_is_routed_by_the_real_configuration_and_needs_no_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Nothing is bound: the configuration file, the type lookup and the runtime are real.

    The bound fixtures above replace both the lookup that sends a Gmail source down this path and
    the runtime it runs in, so on their own they would pass even if a real Gmail source fell into
    the Confluence branch — which would tell its operator to start a server. With no client secret
    in the environment, the first thing the real path can say is that one is missing; the
    Confluence path's refusal is that no server is running.
    """
    config = tmp_path / "manicule.toml"
    config.write_text(
        "[connectors.mail]\n"
        'type = "gmail"\n'
        "[connectors.mail.options]\n"
        f'account = "{ACCOUNT}"\n'
        'client_id = "1234-example.apps.googleusercontent.com"\n'
    )
    monkeypatch.setenv("MANICULE_CONFIG_FILE", str(config))
    monkeypatch.setenv("MANICULE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("GMAIL_CLIENT_SECRET", raising=False)
    cli.STATE.overrides = {}
    cli.STATE.workspace = None

    result = login("mail", "--token-file", str(tmp_path / "token"))

    assert result.exit_code != 0
    text = result.output.replace("\n", " ")
    assert "GMAIL_CLIENT_SECRET" in text
    assert "no manicule server" not in text
    assert not (tmp_path / "token").exists()


@pytest.mark.parametrize(
    ("config", "reports"),
    [
        ("[connectors.mail]\ntype = 'gmail'\nsurprise = 1\n", "invalid configuration"),
        ("[connectors.other]\ntype = 'filesystem'\n", "no connector named"),
    ],
)
def test_a_configuration_problem_is_reported_as_itself_not_as_a_non_gmail_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config: str, reports: str
) -> None:
    """The type could not be told, so the Gmail path's own report is the one the operator needs."""
    path = tmp_path / "manicule.toml"
    path.write_text(config)
    monkeypatch.setenv("MANICULE_CONFIG_FILE", str(path))
    monkeypatch.setenv("MANICULE_DATA_DIR", str(tmp_path / "data"))
    cli.STATE.overrides = {}
    cli.STATE.workspace = None

    result = login("mail", "--token-file", str(tmp_path / "token"))

    assert result.exit_code != 0
    text = result.output.replace("\n", " ")
    assert reports in text
    assert "not a Gmail source" not in text
    assert not (tmp_path / "token").exists()
